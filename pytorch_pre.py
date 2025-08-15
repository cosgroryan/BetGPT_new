#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feed-forward (dense) PyTorch model for tabular horse-racing data.
Target: finish_rank regression (lower is better). This keeps ordering
so 2nd is closer to 1st than 10th.

What this script does:
1) Load parquet (defaults to five_year_dataset.parquet)
2) Build leak-safe per-horse form features from prior starts only
3) Chronological train/val/test split (70/15/15) by date
4) Fit scaler + categorical vocabularies from TRAIN only
5) Train with early stopping on validation Spearman ρ
6) Save model + preprocessing artefacts to ./artifacts/

Run:
    python pytorch_pre.py

If you get a Parquet engine error, install one:
    pip install pyarrow
"""

import argparse
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

def _to_bool_mask_any(s: pd.Series) -> pd.Series:
    """Coerce any 'is_scratched' flavour to clean booleans."""
    if s.dtype == bool or pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    # strings/object: true/1/yes/y => True
    return s.astype(str).str.strip().str.lower().isin({"true","t","1","yes","y"})


# Repro
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Config
# -----------------------------
DEFAULT_DATA_PATH = "five_year_dataset.parquet"

# Pre-race numeric features (avoid outcome/payout fields)
DEFAULT_NUMERIC_COLS = [
    "meeting_number",
    "race_number",
    "race_distance_m",
    "stake",                # prize pool; present pre-race
    "fav_rank",             # market favourite rank (pre-race if available)
    "race_length",          # scheduled length
    "race_number_sched",
    "entrant_weight",       # handicap
]

# Categorical features (now including runner_name for entity awareness)
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
    "runner_name",          # <- key addition: entity embedding
]

# Columns that cause leakage if used as features
LEAKY_COLS = {
    "finish_rank",  # target only
    "margin_len",
    "payout_win",
    "payout_plc",
    "payout_qla",
    "payout_tfa",
    "payout_ft4",
}

# IDs/text we generally exclude as features (runner_name is intentionally NOT here)
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
    # "runner_name" is excluded on purpose so it can be used as a categorical feature
}

# -----------------------------
# Utilities
# -----------------------------

def emb_dim_rule(n_unique: int) -> int:
    """Heuristic for embedding dims; caps at 50, sublinear growth."""
    return int(min(50, round(1.6 * (n_unique ** 0.56))))


@dataclass
class PreprocessArtifacts:
    numeric_cols: List[str]
    categorical_cols: List[str]
    cat_vocab: Dict[str, Dict[str, int]]      # per-col token -> index (1..N), 0=UNK
    cat_cardinalities: Dict[str, int]         # per-col, number of known tokens (excludes 0)
    scaler: StandardScaler

# --- Back-compat for older pickles that expect GBMArtifacts
GBMArtifacts = PreprocessArtifacts

class RacingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        numeric_cols: List[str],
        categorical_cols: List[str],
        scaler: StandardScaler,
        cat_vocab: Dict[str, Dict[str, int]],
    ):
        self.y = torch.tensor(y.astype(np.float32)).view(-1, 1)

        # Numeric
        num_data = df[numeric_cols].astype(np.float32).values
        num_scaled = scaler.transform(num_data)
        self.x_num = torch.tensor(num_scaled, dtype=torch.float32)

        # Categoricals -> indices
        cat_arrays = []
        for col in categorical_cols:
            vocab = cat_vocab[col]
            idx = df[col].astype(str).map(vocab).fillna(0).astype(np.int64).values  # 0 = UNK
            cat_arrays.append(torch.tensor(idx))
        self.x_cat = (
            torch.stack(cat_arrays, dim=1) if cat_arrays else torch.empty((len(df), 0), dtype=torch.long)
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


class TabularModel(nn.Module):
    def __init__(self, num_in: int, cat_cardinalities: List[int], hidden: List[int] = [256, 128, 64], dropout: float = 0.25):
        super().__init__()
        # Embeddings per categorical feature (0 reserved for UNK)
        self.embeddings = nn.ModuleList()
        emb_dims = []
        for card in cat_cardinalities:
            d = emb_dim_rule(card)
            # num_embeddings = card + 1 because 0 is UNK and card counts known tokens
            self.embeddings.append(nn.Embedding(num_embeddings=card + 1, embedding_dim=d, padding_idx=0))
            emb_dims.append(d)
        emb_total = sum(emb_dims)

        layers = []
        in_dim = num_in + emb_total
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]  # scalar output for regression
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if x_cat.numel() > 0:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_num] + embs, dim=1)
        else:
            x = x_num
        return self.mlp(x)


def build_horse_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-horse lagged features using only prior races.
    Assumes df has columns: runner_name, date, finish_rank, fav_rank, margin_len (optional).
    Returns a frame aligned to df.index with new columns.
    """
    req_cols = ["runner_name", "date", "finish_rank", "fav_rank", "margin_len"]
    missing = [c for c in req_cols if c not in df.columns]
    # add any missing optional to keep code simple
    temp = df.copy()
    for c in missing:
        temp[c] = np.nan

    work = temp[req_cols].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    # Sort by horse then date so shifts are prior runs
    work = work.sort_values(["runner_name", "date"]).reset_index()

    g = work.groupby("runner_name", sort=False)

    # Prior starts count
    work["horse_starts_prior"] = g.cumcount()

    # Binary outcomes for cumulative rates
    work["is_win"] = (work["finish_rank"] == 1).astype(float)
    work["is_top3"] = (work["finish_rank"] <= 3).astype(float)

    # Cumulative sums minus current row ⇒ prior only
    for col in ["is_win", "is_top3", "finish_rank", "fav_rank", "margin_len"]:
        s = work[col]
        work[f"cum_{col}_prior"] = g[col].cumsum() - s

    cnt = work["horse_starts_prior"].replace(0, np.nan)

    work["horse_win_rate_prior"]      = work["cum_is_win_prior"] / cnt
    work["horse_top3_rate_prior"]     = work["cum_is_top3_prior"] / cnt
    work["horse_avg_finish_prior"]    = work["cum_finish_rank_prior"] / cnt
    work["horse_avg_fav_rank_prior"]  = work["cum_fav_rank_prior"] / cnt
    work["horse_avg_margin_prior"]    = work["cum_margin_len_prior"] / cnt

    work["horse_last_finish"] = g["finish_rank"].shift(1)
    last_date = g["date"].shift(1)
    work["days_since_last_run"] = (work["date"] - last_date).dt.days

    out = work.set_index("index")[
        [
            "horse_starts_prior",
            "horse_win_rate_prior",
            "horse_top3_rate_prior",
            "horse_avg_finish_prior",
            "horse_last_finish",
            "horse_avg_fav_rank_prior",
            "horse_avg_margin_prior",
            "days_since_last_run",
        ]
    ].sort_index()

    return out


def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    running = 0.0
    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        optimiser.zero_grad()
        preds = model(x_num, x_cat)
        loss = criterion(preds, y)
        loss.backward()
        optimiser.step()

        running += loss.item() * y.size(0)
    return running / len(loader.dataset)


# --- replace your evaluate() with this ---
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds_all, targs_all = [], [], []
    with torch.no_grad():
        for xb_num, xb_cat, yb in loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            yb = yb.to(device)
            out = model(xb_num, xb_cat)
            loss = criterion(out, yb)
            losses.append(loss.item())
            preds_all.append(out.detach().cpu().numpy().ravel())
            targs_all.append(yb.detach().cpu().numpy().ravel())

    preds = np.concatenate(preds_all).astype(float) if preds_all else np.array([])
    targs = np.concatenate(targs_all).astype(float) if targs_all else np.array([])

    # mask non-finite to avoid sklearn crashes
    mask = np.isfinite(preds) & np.isfinite(targs)
    if mask.sum() == 0:
        # return averages/NaNs gracefully
        avg_loss = float(np.mean(losses)) if losses else math.nan
        return avg_loss, math.nan, math.nan, math.nan

    preds_m = preds[mask]
    targs_m = targs[mask]

    mae = float(mean_absolute_error(targs_m, preds_m))
    rmse = float(np.sqrt(mean_squared_error(targs_m, preds_m)))
    rho = float(spearmanr(targs_m, preds_m).correlation)
    return float(np.mean(losses)), mae, rmse, rho

# -----------------------------
# Main
# -----------------------------

def main():
    # Defaults baked in (no CLI typing)
    default_args = {
        "data": DEFAULT_DATA_PATH,
        "batch_size": 4096,
        "epochs": 20,
        "lr": 3e-4,
        "patience": 5,
        "num_cols": ",".join(DEFAULT_NUMERIC_COLS),
        "cat_cols": ",".join(DEFAULT_CATEGORICAL_COLS),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=default_args["data"], help="Path to parquet")
    parser.add_argument("--batch_size", type=int, default=default_args["batch_size"])
    parser.add_argument("--epochs", type=int, default=default_args["epochs"])
    parser.add_argument("--lr", type=float, default=default_args["lr"])
    parser.add_argument("--patience", type=int, default=default_args["patience"])
    parser.add_argument("--num_cols", type=str, default=default_args["num_cols"])
    parser.add_argument("--cat_cols", type=str, default=default_args["cat_cols"])

    # Use baked defaults unless you pass overrides (empty list to parse_args)
    args = parser.parse_args([])

    numeric_cols = [c for c in [s.strip() for s in args.num_cols.split(",")] if c]
    categorical_cols = [c for c in [s.strip() for s in args.cat_cols.split(",")] if c]

    os.makedirs("artifacts", exist_ok=True)

    # Load
    df = pd.read_parquet(args.data)

    # Filter scratches
    if "is_scratched" in df.columns:
        scr_mask = _to_bool_mask_any(df["is_scratched"])
        df = df[~scr_mask]


    # Ensure date for chronological split + form
    if "date" not in df.columns:
        raise ValueError("Expected 'date' column for chronological split and form features.")

    # Build per-horse form features (leak-safe)
    form_feat = build_horse_form_features(df)
    df = pd.concat([df, form_feat], axis=1)

    # Extend numeric list with form features (if present)
    form_numeric = [
        "horse_starts_prior",
        "horse_win_rate_prior",
        "horse_top3_rate_prior",
        "horse_avg_finish_prior",
        "horse_last_finish",
        "horse_avg_fav_rank_prior",
        "horse_avg_margin_prior",
        "days_since_last_run",
    ]
    numeric_cols = list(dict.fromkeys(numeric_cols + [c for c in form_numeric if c in df.columns]))

    # Target: finish_rank
    if "finish_rank" not in df.columns:
        raise ValueError("Expected 'finish_rank' column for regression target.")
    y_all = pd.to_numeric(df["finish_rank"], errors="coerce").values

    # Build working feature frame (do NOT drop runner_name)
    keep_cols = [c for c in (numeric_cols + categorical_cols) if c in df.columns]
    missing = [c for c in (numeric_cols + categorical_cols) if c not in df.columns]
    if missing:
        print(f"[warn] Missing feature columns (skipping): {missing}")
    X_all = df[keep_cols].copy()

    # Impute
    for c in numeric_cols:
        if c in X_all.columns:
            col = pd.to_numeric(X_all[c], errors="coerce")
            X_all[c] = col.fillna(col.median())
    for c in categorical_cols:
        if c in X_all.columns:
            X_all[c] = X_all[c].astype(str).fillna("UNK").replace({"nan": "UNK"})

    # Chronological split: 70/15/15
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    order = np.argsort(df["_date"].values.astype("datetime64[ns]"))
    X_all = X_all.iloc[order]
    y_all = y_all[order]

    n = len(X_all)
    i_train_end = int(0.70 * n)
    i_val_end = int(0.85 * n)

    X_train_df = X_all.iloc[:i_train_end]
    X_val_df   = X_all.iloc[i_train_end:i_val_end]
    X_test_df  = X_all.iloc[i_val_end:]

    y_train = y_all[:i_train_end]
    y_val   = y_all[i_train_end:i_val_end]
    y_test  = y_all[i_val_end:]

    # Build categorical vocabularies from TRAIN only
    cat_vocab: Dict[str, Dict[str, int]] = {}
    cat_cardinalities: Dict[str, int] = {}
    for col in categorical_cols:
        if col not in X_train_df.columns:
            continue
        tokens = X_train_df[col].astype(str).unique().tolist()
        # 0 reserved for UNK; map tokens to 1..N
        vocab = {tok: i + 1 for i, tok in enumerate(sorted(tokens))}
        cat_vocab[col] = vocab
        cat_cardinalities[col] = len(vocab)

    # Fit scaler on TRAIN numeric only
    scaler = StandardScaler()
    train_num = X_train_df[[c for c in numeric_cols if c in X_train_df.columns]].astype(np.float32).values
    scaler.fit(train_num)

    # Datasets & loaders
    ds_train = RacingDataset(
        X_train_df, y_train,
        [c for c in numeric_cols if c in X_train_df.columns],
        [c for c in categorical_cols if c in X_train_df.columns],
        scaler, cat_vocab
    )
    ds_val = RacingDataset(
        X_val_df, y_val,
        [c for c in numeric_cols if c in X_val_df.columns],
        [c for c in categorical_cols if c in X_val_df.columns],
        scaler, cat_vocab
    )
    ds_test = RacingDataset(
        X_test_df, y_test,
        [c for c in numeric_cols if c in X_test_df.columns],
        [c for c in categorical_cols if c in X_test_df.columns],
        scaler, cat_vocab
    )

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # right before calling evaluate() on test_loader
    # (only for debugging)
    # count NaNs/Infs in y
    ys = []
    for *_x, yb in test_loader:
        ys.append(yb.numpy().ravel())
    ys = np.concatenate(ys) if ys else np.array([])
    print(f"[DEBUG] test y bad count:",
        np.count_nonzero(~np.isfinite(ys)))
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    num_in = ds_train.x_num.shape[1]
    cat_cards = [cat_cardinalities[c] for c in categorical_cols if c in cat_cardinalities]
    model = TabularModel(num_in=num_in, cat_cardinalities=cat_cards, hidden=[256, 128, 64], dropout=0.25).to(device)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Train with early stopping on val Spearman
    best_score = -np.inf
    best_state = None
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss, val_mae, val_rmse, val_spear = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} val_spearman={val_spear:.4f}"
        )

        improved = val_spear > best_score
        if improved:
            best_score = val_spear
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on test
    test_loss, test_mae, test_rmse, test_spear = evaluate(model, test_loader, criterion, device)
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


# -----------------------------
# Inference utility
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
    # categorial
    for c in art.categorical_cols:
        if c in df_new.columns:
            col = df_new[c].astype(str).fillna("UNK").replace({"nan": "UNK"})
        else:
            col = pd.Series(["UNK"] * len(df_new))
        work[c] = col

    # New horse detection
    new_horse = np.array([False] * len(work))
    if "runner_name" in art.cat_vocab:
        rn_vocab = art.cat_vocab["runner_name"]
        mapped = work["runner_name"].astype(str).map(rn_vocab).fillna(0).astype(int)
        starts = df_new.get("horse_starts_prior")
        if starts is not None:
            starts = pd.to_numeric(starts, errors="coerce").fillna(0)
            new_horse = ((mapped == 0) | (starts <= 0)).values
        else:
            new_horse = (mapped == 0).values

    # Tensors
    num_data = work[art.numeric_cols].astype(np.float32).values
    num_scaled = art.scaler.transform(num_data)
    x_num = torch.tensor(num_scaled, dtype=torch.float32)

    cat_arrays = []
    for col in art.categorical_cols:
        vocab = art.cat_vocab[col]
        idx = work[col].astype(str).map(vocab).fillna(0).astype(np.int64).values
        cat_arrays.append(torch.tensor(idx))
    x_cat = torch.stack(cat_arrays, dim=1) if cat_arrays else torch.empty((len(work), 0), dtype=torch.long)

    # Model
    cat_cards = [art.cat_cardinalities[c] for c in art.categorical_cols if c in art.cat_cardinalities]
    model = TabularModel(num_in=x_num.shape[1], cat_cardinalities=cat_cards, hidden=[256, 128, 64], dropout=0.25)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        preds = model(x_num.to(device), x_cat.to(device)).cpu().numpy().ravel()

    return {"pred_rank": preds, "new_horse": new_horse}

def _scores_from_pred_rank(pred_rank: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """
    Convert predicted rank (lower is better) to positive 'skill' scores for PL.
    tau is a temperature: smaller = sharper differences.
    """
    # Shift so best rank ~ highest score; use exp(-rank/tau) to get positive scores
    return np.exp(-np.asarray(pred_rank, dtype=float) / float(tau))


def _pl_sample_order(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a finishing order (permutation) from Plackett–Luce given positive scores.
    Returns indices in finishing order (position 1 is index at [0]).
    """
    scores = scores.astype(float).copy()
    n = len(scores)
    order = np.empty(n, dtype=int)
    alive = np.arange(n)
    s = scores.copy()
    for pos in range(n):
        # sample one index proportional to current scores
        p = s / s.sum()
        i = rng.choice(len(alive), p=p)
        order[pos] = alive[i]
        # remove chosen runner
        alive = np.delete(alive, i)
        s = np.delete(s, i)
    return order


def estimate_position_probs_for_race(
    race_df: pd.DataFrame,
    artefacts_path: str = "artifacts/preprocess.pkl",
    model_path: str = "artifacts/model_regression.pth",
    tau: float = 1.0,
    n_samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For a single race (multiple rows = runners), return a DataFrame with:
      runner_name, win_prob, top3_prob, and pos_probs_[1..field_size]

    Uses your trained regression model to predict finish ranks,
    converts them to PL 'scores', and Monte Carlo samples finish orders.
    """
    # 1) Predict ranks for this race
    pred = load_model_and_predict(race_df, model_path=model_path, artefacts_path=artefacts_path)
    if isinstance(pred, dict) and "pred_rank" in pred:
        pred_rank = np.asarray(pred["pred_rank"])
    else:
        pred_rank = np.asarray(pred)

    # 2) Convert to scores for PL
    scores = _scores_from_pred_rank(pred_rank, tau=tau)

    # 3) Monte Carlo sampling of finish orders
    rng = np.random.default_rng(seed)
    n = len(scores)
    pos_counts = np.zeros((n, n), dtype=np.int32)  # [runner_idx, position-1]
    for _ in range(n_samples):
        order = _pl_sample_order(scores, rng)  # indices in finish order
        for pos, idx in enumerate(order):
            pos_counts[idx, pos] += 1

    pos_probs = pos_counts / float(n_samples)  # empirical probabilities

    # 4) Assemble output
    out = pd.DataFrame({
        "runner_idx": np.arange(n),
        "runner_name": race_df.get("runner_name", pd.Series([None]*n)).values,
        "win_prob": pos_probs[:, 0],
    })
    # Top‑3 (or clamp to field size)
    top_k = min(3, n)
    out["top3_prob"] = pos_probs[:, :top_k].sum(axis=1)

    # Add per‑position columns
    for p in range(n):
        out[f"pos_prob_{p+1}"] = pos_probs[:, p]

    # Sort by win_prob desc for convenience
    out = out.sort_values("win_prob", ascending=False).reset_index(drop=True)
    return out


def estimate_position_probs_for_card(
    card_df: pd.DataFrame,
    race_id_col: str = "race_id",
    **kwargs,
) -> dict:
    """
    Run position probability estimation for every race in a card (many races).
    Returns a dict: race_id -> DataFrame from estimate_position_probs_for_race.
    """
    results = {}
    if race_id_col not in card_df.columns:
        # treat the whole frame as one race
        results["<single_race>"] = estimate_position_probs_for_race(card_df, **kwargs)
        return results

    for rid, group in card_df.groupby(race_id_col, sort=False):
        results[rid] = estimate_position_probs_for_race(group.copy(), **kwargs)
    return results


if __name__ == "__main__":
    main()
