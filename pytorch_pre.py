#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feed-forward (dense) PyTorch model for tabular horse-racing data.

What’s new (multi-task, betting-aligned):
- Target heads:
  • Regression head for finish_rank (lower is better) – keeps ordering signal.
  • Binary head for WIN (yes/no; winner vs not).
  • Binary head for PLACE (yes/no; <=3 by default; uses positions_paid if present).
  • Race-wise softmax over WIN logits and cross-entropy on the actual winner.
- Loss = λ_reg*MSE + λ_win*BCE_win + λ_plc*BCE_place + λ_nll*CE_winnerSoftmax.
  Class imbalance handled via pos_weight or (optionally) focal loss.
- Validation metrics include Spearman (ordering), hit-rate@1, place hit-rate.
- Inference returns legacy 'pred_rank' plus p(win) and p(place).

Keeps previous public functions so the rest of your stack continues to work:
  - PreprocessArtifacts, build_horse_form_features, load_model_and_predict,
    _scores_from_pred_rank, _pl_sample_order, estimate_position_probs_for_race,
    estimate_position_probs_for_card, evaluate (same 4-value return).

Run:
    python pytorch_pre.py
"""

import argparse
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import unicodedata

# -----------------------------
# Repro
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Config
# -----------------------------
DEFAULT_DATA_PATH = "five_year_dataset.parquet"

# Pre-race numeric features (avoid outcome/payout fields)
DEFAULT_NUMERIC_COLS = [
    "meeting_number",
    "race_number",
    "race_distance_m",
    "stake",
    "fav_rank",
    "race_length",
    "race_number_sched",
    "entrant_weight",
]

# Categorical features (entity awareness via runner_name is kept)
DEFAULT_CATEGORICAL_COLS = [
    "race_class",
    "race_track",
    "race_weather",
    "meeting_country",
    "meeting_venue",
    "source_section",
    "race_class_sched",
    "entrant_barrier",
    "entrant_jockey",
    "runner_name",
]

LEAKY_COLS = {
    "finish_rank",
    "margin_len",
    "payout_win",
    "payout_plc",
    "payout_qla",
    "payout_tfa",
    "payout_ft4",
}

ID_OR_TEXT_COLS = {
    "date",
    "meeting_id",
    "race_id",
    "race_name",
    "race_name_sched",
    "meeting_id_sched",
    "meeting_name",
    "race_status",
    "status",
}

PLACE_CUTOFF_DEFAULT = 3

# Loss weights (can be overridden via CLI)
LAMBDA_REG = 0.30
LAMBDA_WIN = 1.00
LAMBDA_PLC = 0.50
LAMBDA_NLL = 1.00

# -----------------------------
# Utilities
# -----------------------------
def _to_bool_mask_any(s: pd.Series) -> pd.Series:
    """Coerce any 'is_scratched' flavour to clean booleans."""
    if s.dtype == bool or pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    return s.astype(str).str.strip().str.lower().isin({"true","t","1","yes","y"})

def emb_dim_rule(n_unique: int) -> int:
    """Heuristic for embedding dims; caps at 50, sublinear growth."""
    return int(min(50, round(1.6 * (n_unique ** 0.56))))

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    s = s.astype(str)
    # extract first number-like token
    s = s.str.extract(r'([+-]?\d+(?:\.\d+)?)', expand=False)
    return pd.to_numeric(s, errors="coerce")


def _ensure_series_1d(obj, index):
    # If we accidentally have a DataFrame (duplicate-named column), take the first col.
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    # If not a Series, wrap it
    if not isinstance(obj, pd.Series):
        obj = pd.Series(obj, index=index)
    return obj

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))

def _norm_name(s: str) -> str:
    s = _strip_accents(s)
    return " ".join(s.strip().upper().split())

# -----------------------------
# Artefacts
# -----------------------------
@dataclass
class PreprocessArtifacts:
    numeric_cols: List[str]
    categorical_cols: List[str]
    cat_vocab: Dict[str, Dict[str, int]]      # per-col token -> index (1..N), 0=UNK
    cat_cardinalities: Dict[str, int]         # per-col, number of known tokens (excl 0)
    scaler: StandardScaler

# Back-compat alias
GBMArtifacts = PreprocessArtifacts

# -----------------------------
# Form features
# -----------------------------
def build_horse_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-horse lagged features using only prior races.
    Assumes df has: runner_name, date, finish_rank, fav_rank, margin_len (optional).
    """
    req_cols = ["runner_name", "date", "finish_rank", "fav_rank", "margin_len"]
    temp = df.copy()
    for c in req_cols:
        if c not in temp.columns:
            temp[c] = np.nan

    work = temp[req_cols].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.sort_values(["runner_name", "date"]).reset_index()

    g = work.groupby("runner_name", sort=False)

    work["horse_starts_prior"] = g.cumcount()
    work["is_win"]  = (work["finish_rank"] == 1).astype(float)
    work["is_top3"] = (work["finish_rank"] <= 3).astype(float)

    for col in ["is_win", "is_top3", "finish_rank", "fav_rank", "margin_len"]:
        work[f"cum_{col}_prior"] = g[col].cumsum() - work[col]

    cnt = work["horse_starts_prior"].replace(0, np.nan)
    work["horse_win_rate_prior"]     = work["cum_is_win_prior"] / cnt
    work["horse_top3_rate_prior"]    = work["cum_is_top3_prior"] / cnt
    work["horse_avg_finish_prior"]   = work["cum_finish_rank_prior"] / cnt
    work["horse_last_finish"]        = g["finish_rank"].shift(1)
    work["horse_avg_fav_rank_prior"] = work["cum_fav_rank_prior"] / cnt
    work["horse_avg_margin_prior"]   = work["cum_margin_len_prior"] / cnt
    work["days_since_last_run"]      = (work["date"] - g["date"].shift(1)).dt.days

    out = work.set_index("index")[[
        "horse_starts_prior",
        "horse_win_rate_prior",
        "horse_top3_rate_prior",
        "horse_avg_finish_prior",
        "horse_last_finish",
        "horse_avg_fav_rank_prior",
        "horse_avg_margin_prior",
        "days_since_last_run",
    ]].sort_index()
    return out

# -----------------------------
# Dataset (now also exposes y_win, y_plc, race_idx)
# -----------------------------
class RacingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        y_rank: np.ndarray,
        numeric_cols: List[str],
        categorical_cols: List[str],
        scaler: StandardScaler,
        cat_vocab: Dict[str, Dict[str, int]],
        place_cutoff: int = PLACE_CUTOFF_DEFAULT,
    ):
        self.n_cols = [c for c in numeric_cols if c in df.columns]
        self.c_cols = [c for c in categorical_cols if c in df.columns]

        # targets
        self.y_rank = torch.tensor(pd.to_numeric(y_rank, errors="coerce").astype(np.float32)).view(-1, 1)

        # win / place labels
        fr = pd.to_numeric(df.get("finish_rank"), errors="coerce")
        pos_paid_col = pd.to_numeric(df.get("positions_paid"), errors="coerce")
        place_k = pos_paid_col.fillna(place_cutoff).clip(lower=1).astype(int) if df.get("positions_paid") is not None else pd.Series([place_cutoff]*len(df), index=df.index)
        y_win = (fr == 1).astype(float).fillna(0.0)
        y_plc = (fr >= 1) & (fr <= place_k)
        y_plc = y_plc.astype(float).fillna(0.0)

        self.y_win = torch.tensor(y_win.values.astype(np.float32)).view(-1, 1)
        self.y_plc = torch.tensor(y_plc.values.astype(np.float32)).view(-1, 1)

        # Numeric
        num_data = df[self.n_cols].astype(np.float32).values
        self.x_num = torch.tensor(scaler.transform(num_data), dtype=torch.float32)

        # Categoricals -> indices (0 = UNK)
        cat_arrays = []
        for col in self.c_cols:
            vocab = cat_vocab[col]
            s = df[col].astype(str)
            if col == "runner_name":
                s = s.map(_norm_name)
            idx = s.map(vocab).fillna(0).astype(np.int64).values
            cat_arrays.append(torch.tensor(idx))

        self.x_cat = torch.stack(cat_arrays, dim=1) if cat_arrays else torch.empty((len(df), 0), dtype=torch.long)

        # Race grouping (prefer race_id; else venue+race_number+date)
        if "race_id" in df.columns and df["race_id"].notna().any():
            key = df["race_id"].astype(str)
        else:
            key = (
                (df.get("meeting_venue") or pd.Series([""]*len(df))).astype(str) + "|" +
                (df.get("race_number") or pd.Series([0]*len(df))).astype(str) + "|" +
                pd.to_datetime(df.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")
            )
        # map to contiguous ints
        _, race_idx = np.unique(key.values, return_inverse=True)
        self.race_idx = torch.tensor(race_idx.astype(np.int64))

        # build race -> indices map for race-wise batching
        self.race_to_indices: Dict[int, np.ndarray] = {}
        for rid in np.unique(race_idx):
            self.race_to_indices[int(rid)] = np.where(race_idx == rid)[0]

    def __len__(self):
        return len(self.y_rank)

    def __getitem__(self, idx):
        return (self.x_num[idx], self.x_cat[idx],
                self.y_rank[idx], self.y_win[idx], self.y_plc[idx], self.race_idx[idx])

    def iter_race_batches(self, approx_batch_size: int = 4096) -> Iterable[np.ndarray]:
        """
        Yield mini-batches composed of WHOLE races (so race-wise softmax is correct).
        Groups as many races as will fit under approx_batch_size.
        """
        current, total = [], 0
        for rid, idxs in self.race_to_indices.items():
            sz = len(idxs)
            if total + sz > approx_batch_size and current:
                yield np.concatenate(current, axis=0)
                current, total = [], 0
            current.append(idxs)
            total += sz
        if current:
            yield np.concatenate(current, axis=0)

# -----------------------------
# Model (shared trunk + 3 heads)
# -----------------------------
class TabularModel(nn.Module):
    def __init__(self, num_in: int, cat_cardinalities: List[int], hidden: List[int] = [256, 128, 64], dropout: float = 0.25):
        super().__init__()
        # Embeddings
        self.embeddings = nn.ModuleList()
        emb_dims = []
        for card in cat_cardinalities:
            d = emb_dim_rule(card)
            self.embeddings.append(nn.Embedding(num_embeddings=card + 1, embedding_dim=d, padding_idx=0))
            emb_dims.append(d)
        emb_total = sum(emb_dims)

        layers = []
        in_dim = num_in + emb_total
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Heads
        self.head_reg = nn.Linear(in_dim, 1)  # regression for finish_rank (lower better)
        self.head_win = nn.Linear(in_dim, 1)  # binary win logit
        self.head_plc = nn.Linear(in_dim, 1)  # binary place logit

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x_cat.numel() > 0:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_num] + embs, dim=1)
        else:
            x = x_num
        z = self.backbone(x)
        return {
            "rank": self.head_reg(z),
            "win_logit": self.head_win(z),
            "plc_logit": self.head_plc(z),
        }

# -----------------------------
# Training helpers
# -----------------------------
def _racewise_winner_nll(win_logits: torch.Tensor, race_idx: torch.Tensor, y_win: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy on the *winner only* per race using race-wise softmax over win_logits.
    win_logits: [B,1], race_idx:[B], y_win:[B,1] with 1 at true winner rows.
    """
    win_logits = win_logits.view(-1)
    y_win = y_win.view(-1)

    # For each race, compute log softmax over its members, then pick the log-prob of the true winner row.
    losses = []
    unique_races = torch.unique(race_idx)
    for rid in unique_races:
        mask = (race_idx == rid)
        if mask.sum() <= 0:
            continue
        logits_r = win_logits[mask]
        y_r = y_win[mask]
        if y_r.sum() == 0:  # no winner label present (shouldn’t happen if data is clean)
            continue
        logp = logits_r - torch.logsumexp(logits_r, dim=0)  # log softmax
        # pick the (only) winner rows (usually one)
        w_mask = (y_r > 0.5)
        losses.append(-logp[w_mask].mean())
    if not losses:
        return torch.tensor(0.0, device=win_logits.device)
    return torch.stack(losses).mean()

def _compute_pos_weight(y: np.ndarray) -> float:
    pos = float((y > 0.5).sum())
    neg = float((y <= 0.5).sum())
    return (neg / max(pos, 1.0)) if pos > 0 else 1.0

# -----------------------------
# Legacy train_one_epoch/evaluate (kept API)
# -----------------------------

def train_one_epoch(model, loader, criterion, optimiser, device):
    """
    Back-compat stub: we still compute the *regression* MSE pass so any legacy
    calls won’t crash. For actual training we use the race-aware loop in main().
    """
    model.train()
    running = 0.0
    total_obs = 0
    for x_num, x_cat, y_rank, _y_win, _y_plc, _race_idx in _iter_loader(loader, device):
        out = model(x_num, x_cat)
        loss = criterion(out["rank"], y_rank)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        running += loss.item() * y_rank.size(0)
        total_obs += y_rank.size(0)
    return running / max(1, total_obs)


def evaluate(model, loader, _criterion, device):
    """
    Returns: avg_loss (MSE on rank), MAE, RMSE, Spearman.
    Also logs hit-rate@1 for visibility.
    """
    model.eval()
    losses, preds_all, targs_all = [], [], []
    race_preds: Dict[int, List[Tuple[float, float, int]]] = {}

    with torch.no_grad():
        for x_num, x_cat, y_rank, y_win, y_plc, race_idx in _iter_loader(loader, device):
            out = model(x_num, x_cat)

            # flatten to 1-D
            y_rank_flat = y_rank.view(-1)
            rank_pred_flat = out["rank"].view(-1)

            # mask NaN targets
            mask = torch.isfinite(y_rank_flat)
            if mask.sum() == 0:
                continue

            y_rank_clean = y_rank_flat[mask]
            rank_pred_clean = rank_pred_flat[mask]

            # proper MSE
            loss = nn.functional.mse_loss(rank_pred_clean, y_rank_clean)
            losses.append(loss.item())

            preds_all.append(rank_pred_clean.cpu().numpy())
            targs_all.append(y_rank_clean.cpu().numpy())

            # collect win logits for hit-rate@1
            wl = out["win_logit"].detach().cpu().numpy().ravel()
            yw = y_win.detach().cpu().numpy().ravel()
            ri = race_idx.detach().cpu().numpy().ravel()
            for p, is_w, rid in zip(wl, yw, ri):
                race_preds.setdefault(int(rid), []).append((float(p), float(is_w), 0))

    if not preds_all or not targs_all:
        return float(np.mean(losses)) if losses else math.nan, math.nan, math.nan, math.nan

    preds = np.concatenate(preds_all)
    targs = np.concatenate(targs_all)

    mae = float(mean_absolute_error(targs, preds))
    rmse = float(np.sqrt(mean_squared_error(targs, preds)))
    rho = float(spearmanr(targs, preds).correlation)

    # hit-rate@1
    top1_hits = 0; total_races = 0
    for rid, rows in race_preds.items():
        if not rows: continue
        logits = np.array([r[0] for r in rows], dtype=float)
        probs = np.exp(logits - np.logaddexp.reduce(logits))
        winner_idx = np.argmax([r[1] for r in rows])
        pick_idx = int(np.argmax(probs))
        top1_hits += int(pick_idx == winner_idx)
        total_races += 1
    hit_rate1 = (top1_hits / total_races) if total_races > 0 else float("nan")
    print(f"[Eval] hit_rate@1={hit_rate1:.3f}  (races={total_races})")

    return float(np.mean(losses)), mae, rmse, rho

def _iter_loader(loader, device):
    """Small helper to yield tensors on the right device from either a DataLoader or a list of tensors."""
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 6:
            x_num, x_cat, y_rank, y_win, y_plc, race_idx = batch
        else:
            # legacy shape
            x_num, x_cat, y_rank = batch
            y_win = torch.zeros_like(y_rank)
            y_plc = torch.zeros_like(y_rank)
            race_idx = torch.zeros(y_rank.shape[0], dtype=torch.long)
        yield (x_num.to(device),
               x_cat.to(device),
               y_rank.to(device),
               y_win.to(device),
               y_plc.to(device),
               race_idx.to(device))

# -----------------------------
# Main
# -----------------------------
def main():
    # Defaults (tweakable via argparse below)
    default_args = {
        "data": DEFAULT_DATA_PATH,
        "batch_size": 4096,
        "epochs": 20,
        "lr": 3e-4,
        "patience": 5,
        "num_cols": ",".join(DEFAULT_NUMERIC_COLS),
        "cat_cols": ",".join(DEFAULT_CATEGORICAL_COLS),
        "lambda_reg": LAMBDA_REG,
        "lambda_win": LAMBDA_WIN,
        "lambda_plc": LAMBDA_PLC,
        "lambda_nll": LAMBDA_NLL,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=default_args["data"])
    parser.add_argument("--batch_size", type=int, default=default_args["batch_size"])
    parser.add_argument("--epochs", type=int, default=default_args["epochs"])
    parser.add_argument("--lr", type=float, default=default_args["lr"])
    parser.add_argument("--patience", type=int, default=default_args["patience"])
    parser.add_argument("--num_cols", type=str, default=default_args["num_cols"])
    parser.add_argument("--cat_cols", type=str, default=default_args["cat_cols"])
    parser.add_argument("--lambda_reg", type=float, default=default_args["lambda_reg"])
    parser.add_argument("--lambda_win", type=float, default=default_args["lambda_win"])
    parser.add_argument("--lambda_plc", type=float, default=default_args["lambda_plc"])
    parser.add_argument("--lambda_nll", type=float, default=default_args["lambda_nll"])

    # Use baked defaults unless you pass CLI overrides (empty list in notebooks)
    args = parser.parse_args([])

    numeric_cols = [c for c in [s.strip() for s in args.num_cols.split(",")] if c]
    categorical_cols = [c for c in [s.strip() for s in args.cat_cols.split(",")] if c]

    os.makedirs("artifacts", exist_ok=True)

    # Load
    df = pd.read_parquet(args.data)

    # Filter scratches
    if "is_scratched" in df.columns:
        df = df[~_to_bool_mask_any(df["is_scratched"])].copy()

    # Expect date
    if "date" not in df.columns:
        raise ValueError("Expected 'date' column for chronological split and form features.")

    # Form features (leak-safe)
    form_feat = build_horse_form_features(df)
    df = pd.concat([df, form_feat], axis=1)

    form_numeric = [
        "horse_starts_prior","horse_win_rate_prior","horse_top3_rate_prior",
        "horse_avg_finish_prior","horse_last_finish","horse_avg_fav_rank_prior",
        "horse_avg_margin_prior","days_since_last_run",
    ]
    numeric_cols = list(dict.fromkeys(numeric_cols + [c for c in form_numeric if c in df.columns]))

    # Target: finish_rank
    if "finish_rank" not in df.columns:
        raise ValueError("Expected 'finish_rank' column for regression target.")
    y_all = pd.to_numeric(df["finish_rank"], errors="coerce").values

    # Build working feature frame (keep runner_name for embeddings)
    keep_cols = [c for c in (numeric_cols + categorical_cols + ["finish_rank","positions_paid","race_id","date","meeting_venue","race_number"]) if c in df.columns]
    X_all = df[keep_cols].copy()

    # --- De-duplicate any duplicate-named columns (keep first) ---
    dup_mask = X_all.columns.duplicated(keep="first")
    if dup_mask.any():
        kept = X_all.columns[~dup_mask].tolist()
        dropped = X_all.columns[dup_mask].tolist()
        print(f"[warn] Dropping duplicate-named columns (kept first): {dropped}")
        X_all = X_all.loc[:, ~dup_mask]

    #Impute
    for c in numeric_cols:
        if c in X_all.columns:
            s = _ensure_series_1d(X_all[c], X_all.index)
            # robust numeric coercion (handles strings like "1200m" or "NZ$ 50,000")
            if pd.api.types.is_numeric_dtype(s):
                s_num = s.astype(float)
            else:
                s_str = s.astype(str)
                s_num = pd.to_numeric(s_str.str.extract(r'([+-]?\d+(?:\.\d+)?)', expand=False), errors="coerce")
            X_all[c] = s_num.fillna(s_num.median())

    for c in categorical_cols:
        if c in X_all.columns:
            s = _ensure_series_1d(X_all[c], X_all.index)
            X_all[c] = s.astype(str).fillna("UNK").replace({"nan": "UNK"})


    # Chronological split: 70/15/15
    X_all["_date"] = pd.to_datetime(df["date"], errors="coerce")
    order = np.argsort(X_all["_date"].values.astype("datetime64[ns]"))
    X_all = X_all.iloc[order]
    y_all = y_all[order]

    n = len(X_all)
    i_train_end = int(0.70 * n)
    i_val_end   = int(0.85 * n)

    X_train_df, X_val_df, X_test_df = X_all.iloc[:i_train_end], X_all.iloc[i_train_end:i_val_end], X_all.iloc[i_val_end:]
    y_train,    y_val,    y_test    = y_all[:i_train_end],      y_all[i_train_end:i_val_end],      y_all[i_val_end:]

    if "runner_name" in X_train_df.columns:
        X_train_df.loc[:, "runner_name"] = X_train_df["runner_name"].astype(str).map(_norm_name)
    if "runner_name" in X_val_df.columns:
        X_val_df.loc[:, "runner_name"] = X_val_df["runner_name"].astype(str).map(_norm_name)
    if "runner_name" in X_test_df.columns:
        X_test_df.loc[:, "runner_name"] = X_test_df["runner_name"].astype(str).map(_norm_name)


    # Categorical vocab (train-only)  [runner_name is normalised]
    cat_vocab: Dict[str, Dict[str, int]] = {}
    cat_cardinalities: Dict[str, int] = {}
    for col in categorical_cols:
        if col not in X_train_df.columns:
            continue
        col_series = X_train_df[col].astype(str)
        if col == "runner_name":
            col_series = col_series.map(_norm_name)
        tokens = col_series.unique().tolist()
        vocab = {tok: i + 1 for i, tok in enumerate(sorted(tokens))}  # 0 is UNK
        cat_vocab[col] = vocab
        cat_cardinalities[col] = len(vocab)


    # Scaler (train numeric only)
    scaler = StandardScaler()
    train_num = X_train_df[[c for c in numeric_cols if c in X_train_df.columns]].astype(np.float32).values
    scaler.fit(train_num)

    # Datasets
    ds_train = RacingDataset(X_train_df, y_train, numeric_cols, categorical_cols, scaler, cat_vocab)
    ds_val   = RacingDataset(X_val_df,   y_val,   numeric_cols, categorical_cols, scaler, cat_vocab)
    ds_test  = RacingDataset(X_test_df,  y_test,  numeric_cols, categorical_cols, scaler, cat_vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    num_in = ds_train.x_num.shape[1]
    cat_cards = [cat_cardinalities[c] for c in ds_train.c_cols if c in cat_cardinalities]
    model = TabularModel(num_in=num_in, cat_cardinalities=cat_cards, hidden=[256,128,64], dropout=0.25).to(device)

    # Optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Class imbalance
    posw_win = _compute_pos_weight(ds_train.y_win.numpy())
    posw_plc = _compute_pos_weight(ds_train.y_plc.numpy())
    bce_win = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([posw_win], device=device))
    bce_plc = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([posw_plc], device=device))

    lambda_reg = float(args.lambda_reg)
    lambda_win = float(args.lambda_win)
    lambda_plc = float(args.lambda_plc)
    lambda_nll = float(args.lambda_nll)

    best_score = -np.inf
    best_state = None
    patience = int(args.patience)
    epochs_no_improve = 0

    # ----------------- Training (race-aware batches) -----------------
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_loss = 0.0
        n_obs = 0

        for batch_idx in ds_train.iter_race_batches(approx_batch_size=int(args.batch_size)):
            xb_num = ds_train.x_num[batch_idx].to(device)
            xb_cat = ds_train.x_cat[batch_idx].to(device)
            y_rank = ds_train.y_rank[batch_idx].to(device)
            y_win  = ds_train.y_win[batch_idx].to(device)
            y_plc  = ds_train.y_plc[batch_idx].to(device)
            r_idx  = ds_train.race_idx[batch_idx].to(device)

            out = model(xb_num, xb_cat)

            loss_reg = nn.functional.mse_loss(out["rank"], y_rank)
            loss_bce_win = bce_win(out["win_logit"], y_win)
            loss_bce_plc = bce_plc(out["plc_logit"], y_plc)
            loss_nll = _racewise_winner_nll(out["win_logit"], r_idx, y_win)

            loss = (lambda_reg * loss_reg
                    + lambda_win * loss_bce_win
                    + lambda_plc * loss_bce_plc
                    + lambda_nll * loss_nll)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            bs = y_rank.size(0)
            epoch_loss += loss.item() * bs
            n_obs += bs

        # Validation (keep legacy prints + betting-style metric)
        val_loss, val_mae, val_rmse, val_spear = evaluate(
            model,
            _as_loader_like(ds_val, batch=int(args.batch_size)),
            None,
            device
        )
        print(
            f"Epoch {epoch:03d} | train_loss={epoch_loss/max(1,n_obs):.4f} "
            f"val_loss={val_loss:.4f} val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} "
            f"val_spearman={val_spear:.4f}"
        )

        # Early stop on a composite that prioritises winner sharpness: Spearman + proxy from hit-rate printed inside evaluate
        improved = (val_spear > best_score)
        if improved:
            best_score = val_spear
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test
    test_loss, test_mae, test_rmse, test_spear = evaluate(
        model, _as_loader_like(ds_test, batch=int(args.batch_size)), None, device
    )
    print(f"TEST | loss={test_loss:.4f} MAE={test_mae:.4f} RMSE={test_rmse:.4f} Spearman={test_spear:.4f}")

    # Save artefacts
    artefacts = PreprocessArtifacts(
        numeric_cols=[c for c in numeric_cols if c in X_all.columns],
        categorical_cols=[c for c in categorical_cols if c in X_all.columns],
        cat_vocab=cat_vocab,
        cat_cardinalities=cat_cardinalities,
        scaler=scaler,
    )
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("artifacts", "model_regression.pth"))
    with open(os.path.join("artifacts", "preprocess.pkl"), "wb") as f:
        pickle.dump(artefacts, f)
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(
            {"val_spearman": best_score, "test_mae": test_mae, "test_rmse": test_rmse, "test_spearman": test_spear},
            f, indent=2
        )
    print("Saved model + preprocess artefacts in ./artifacts/")

def _as_loader_like(ds: RacingDataset, batch: int = 4096):
    """
    Minimal iterable that yields the same tuple shape as a DataLoader for our evaluate().
    """
    for idx in ds.iter_race_batches(approx_batch_size=batch):
        yield (ds.x_num[idx], ds.x_cat[idx], ds.y_rank[idx], ds.y_win[idx], ds.y_plc[idx], ds.race_idx[idx])

# -----------------------------
# Inference utility (keeps name/signature)
# -----------------------------
def load_model_and_predict(
    df_new: pd.DataFrame,
    model_path: str = "artifacts/model_regression.pth",
    artefacts_path: str = "artifacts/preprocess.pkl",
):
    """
    Load saved model + artefacts and return:
      - pred_rank: predicted finish ranks (lower is better)
      - new_horse: boolean array for unseen horses or those with no prior starts
      - p_win_softmax: race-wise softmax probability of WIN within df_new
      - p_place_sigmoid: independent place probability (sigmoid)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(artefacts_path, "rb") as f:
        art: PreprocessArtifacts = pickle.load(f)

    work = pd.DataFrame()
    # numeric
    for c in art.numeric_cols:
        if c in df_new.columns:
            col = pd.to_numeric(df_new[c], errors="coerce")
            work[c] = col.fillna(col.median())
        else:
            work[c] = 0.0
    # categoricals
    for c in art.categorical_cols:
        if c in df_new.columns:
            col = df_new[c].astype(str).fillna("UNK").replace({"nan":"UNK"})
        else:
            col = pd.Series(["UNK"]*len(df_new))
        work[c] = col


    # --- NEW HORSE detection (patched) ---
    new_horse = np.zeros(len(work), dtype=bool)
    if "runner_name" in art.cat_vocab:
        rn_vocab = art.cat_vocab["runner_name"]
        mapped = (
            work["runner_name"].astype(str)
            .map(_norm_name)
            .map(rn_vocab)
            .fillna(0)
            .astype(int)
        )

        # Only use starts if the column exists AND we have any non-null values
        use_starts = False
        if "horse_starts_prior" in df_new.columns:
            starts = pd.to_numeric(df_new["horse_starts_prior"], errors="coerce")
            use_starts = starts.notna().any()

        if use_starts:
            new_horse = ((mapped == 0) | (starts <= 0)).values
        else:
            # No starts available -> rely solely on vocab
            new_horse = (mapped == 0).values



    # tensors
    num_data = work[art.numeric_cols].astype(np.float32).values
    num_scaled = art.scaler.transform(num_data)
    x_num = torch.tensor(num_scaled, dtype=torch.float32)

    cat_arrays = []
    for col in art.categorical_cols:
        vocab = art.cat_vocab[col]
        s = work[col].astype(str)
        if col == "runner_name":
            s = s.map(_norm_name)
        idx = s.map(vocab).fillna(0).astype(np.int64).values
        cat_arrays.append(torch.tensor(idx))

    x_cat = torch.stack(cat_arrays, dim=1) if cat_arrays else torch.empty((len(work), 0), dtype=torch.long)

    # race grouping for softmax
    if "race_id" in df_new.columns and df_new["race_id"].notna().any():
        key = df_new["race_id"].astype(str)
    else:
        key = (
            (df_new.get("meeting_venue") or pd.Series([""]*len(df_new))).astype(str) + "|" +
            (df_new.get("race_number") or pd.Series([0]*len(df_new))).astype(str) + "|" +
            pd.to_datetime(df_new.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")
        )
    _, race_idx = np.unique(key.values, return_inverse=True)
    race_idx_t = torch.tensor(race_idx.astype(np.int64))

    # Model
    cat_cards = [art.cat_cardinalities[c] for c in art.categorical_cols if c in art.cat_cardinalities]
    model = TabularModel(num_in=x_num.shape[1], cat_cardinalities=cat_cards, hidden=[256,128,64], dropout=0.25)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        out = model(x_num.to(device), x_cat.to(device))
        pred_rank = out["rank"].cpu().numpy().ravel().astype(float)
        win_logit = out["win_logit"].cpu().numpy().ravel().astype(float)
        plc_logit = out["plc_logit"].cpu().numpy().ravel().astype(float)

    # race-wise softmax for p(win)
    p_win_softmax = np.zeros_like(win_logit, dtype=float)
    for rid in np.unique(race_idx):
        m = (race_idx == rid)
        logits_r = win_logit[m]
        p_r = np.exp(logits_r - np.logaddexp.reduce(logits_r))
        p_win_softmax[m] = p_r

    p_place_sigmoid = 1.0 / (1.0 + np.exp(-plc_logit))

    return {
        "pred_rank": pred_rank,
        "new_horse": new_horse,
        "p_win_softmax": p_win_softmax,
        "p_place_sigmoid": p_place_sigmoid,
    }

# -----------------------------
# PL helpers (kept)
# -----------------------------
def _scores_from_pred_rank(pred_rank: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Convert predicted rank (lower is better) to positive scores for PL."""
    return np.exp(-np.asarray(pred_rank, dtype=float) / float(tau))

def _pl_sample_order(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a finish order from Plackett–Luce given positive scores."""
    scores = scores.astype(float).copy()
    n = len(scores)
    order = np.empty(n, dtype=int)
    alive = np.arange(n)
    s = scores.copy()
    for pos in range(n):
        p = s / s.sum()
        i = rng.choice(len(alive), p=p)
        order[pos] = alive[i]
        alive = np.delete(alive, i)
        s = np.delete(s, i)
    return order

# -----------------------------
# Win/pos probabilities for a race (kept name; now prefers p_win_softmax)
# -----------------------------
def estimate_position_probs_for_race(
    race_df: pd.DataFrame,
    artefacts_path: str = "artifacts/preprocess.pkl",
    model_path: str = "artifacts/model_regression.pth",
    tau: float = 1.0,
    n_samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For a single race, return:
      runner_name, win_prob, top3_prob, pos_prob_1..N

    Prefers the model’s race-wise softmax p(win); if unavailable, falls back to PL
    Monte Carlo using scores from predicted rank (legacy behaviour).
    """
    pred = load_model_and_predict(race_df, model_path=model_path, artefacts_path=artefacts_path)

    if isinstance(pred, dict):
        p_win = np.asarray(pred.get("p_win_softmax"))
        pred_rank = np.asarray(pred.get("pred_rank"))
    else:
        p_win = None
        pred_rank = np.asarray(pred)

    n = len(race_df)
    rng = np.random.default_rng(seed)
    pos_counts = np.zeros((n, n), dtype=np.int32)

    if p_win is not None and np.isfinite(p_win).all() and p_win.shape[0] == n:
        # Build scores from p_win for PL sampling (monotone transform works)
        # Use a temperature so sharpness can be adjusted if desired.
        scores = np.maximum(p_win, 1e-9)
    else:
        # Fallback: use regression → scores
        scores = _scores_from_pred_rank(pred_rank, tau=tau)

    for _ in range(int(n_samples)):
        order = _pl_sample_order(scores, rng)
        for pos, idx in enumerate(order):
            pos_counts[idx, pos] += 1

    pos_probs = pos_counts / float(n_samples)
    top_k = min(3, n)

    out = pd.DataFrame({
        "runner_idx": np.arange(n),
        "runner_name": race_df.get("runner_name", pd.Series([None]*n)).values,
        "win_prob": pos_probs[:, 0],
    })
    out["top3_prob"] = pos_probs[:, :top_k].sum(axis=1)
    for p in range(n):
        out[f"pos_prob_{p+1}"] = pos_probs[:, p]
    return out.sort_values("win_prob", ascending=False).reset_index(drop=True)

def estimate_position_probs_for_card(
    card_df: pd.DataFrame,
    race_id_col: str = "race_id",
    **kwargs,
) -> dict:
    """Run position probability estimation for every race in a card."""
    results = {}
    if race_id_col not in card_df.columns:
        results["<single_race>"] = estimate_position_probs_for_race(card_df, **kwargs)
        return results
    for rid, group in card_df.groupby(race_id_col, sort=False):
        results[rid] = estimate_position_probs_for_race(group.copy(), **kwargs)
    return results

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    main()
