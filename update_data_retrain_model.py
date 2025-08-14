#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

# --------------------------
# Repro
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------
# Config (feature lists)
# --------------------------
NUMERIC_COLS_BASE = [
    "meeting_number", "race_number", "race_distance_m", "stake",
    "fav_rank", "race_length", "race_number_sched", "entrant_weight",
]
FORM_NUMERIC = [
    "horse_starts_prior", "horse_win_rate_prior", "horse_top3_rate_prior",
    "horse_avg_finish_prior", "horse_last_finish", "horse_avg_fav_rank_prior",
    "horse_avg_margin_prior", "days_since_last_run",
]
CATEGORICAL_COLS = [
    "race_class", "race_track", "race_weather",
    "meeting_country", "meeting_venue", "source_section",
    "race_class_sched", "entrant_barrier", "entrant_jockey",
    "runner_name",
]

# --------------------------
# Utilities
# --------------------------
def _norm_name(s: str) -> str:
    return " ".join(str(s).strip().upper().split()) if s is not None else "UNK"

def clean_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop scratches, incomplete races, and invalid labels (prevents NaNs in eval)."""
    x = df.copy()

    if "is_scratched" in x.columns:
        x = x[~x["is_scratched"].fillna(False)]

    is_complete = None
    for col in ("status", "race_status"):
        if col in x.columns:
            s = x[col].astype(str).str.lower()
            m = s.eq("complete")
            is_complete = m if is_complete is None else (is_complete | m)
    if is_complete is not None:
        x = x[is_complete]

    x["finish_rank"] = pd.to_numeric(x["finish_rank"], errors="coerce")
    x = x[x["finish_rank"].notna() & (x["finish_rank"] > 0)]
    return x

def build_horse_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leak-safe per-horse lagged stats from prior runs."""
    req = ["runner_name", "date", "finish_rank", "fav_rank", "margin_len"]
    w = df.copy()
    for c in req:
        if c not in w.columns:
            w[c] = np.nan

    w["runner_name"] = w["runner_name"].map(_norm_name)
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.sort_values(["runner_name", "date"]).reset_index()

    g = w.groupby("runner_name", sort=False)
    w["horse_starts_prior"] = g.cumcount()

    w["is_win"] = (w["finish_rank"] == 1).astype(float)
    w["is_top3"] = (w["finish_rank"] <= 3).astype(float)

    for col in ["is_win", "is_top3", "finish_rank", "fav_rank", "margin_len"]:
        s = w[col]
        w[f"cum_{col}_prior"] = g[col].cumsum() - s

    cnt = w["horse_starts_prior"].replace(0, np.nan)
    w["horse_win_rate_prior"] = w["cum_is_win_prior"] / cnt
    w["horse_top3_rate_prior"] = w["cum_is_top3_prior"] / cnt
    w["horse_avg_finish_prior"] = w["cum_finish_rank_prior"] / cnt
    w["horse_avg_fav_rank_prior"] = w["cum_fav_rank_prior"] / cnt
    w["horse_avg_margin_prior"] = w["cum_margin_len_prior"] / cnt

    w["horse_last_finish"] = g["finish_rank"].shift(1)
    last_date = g["date"].shift(1)
    w["days_since_last_run"] = (w["date"] - last_date).dt.days

    out = w.set_index("index")[
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

# --------------------------
# Artifacts
# --------------------------
@dataclass
class PreprocessArtifacts:
    numeric_cols: List[str]
    categorical_cols: List[str]
    num_means: Dict[str, float]
    num_stds: Dict[str, float]
    cat_maps: Dict[str, Dict[str, int]]          # category -> index (includes "UNK")
    cat_cardinalities: Dict[str, int]            # for embeddings
    target_name: str = "finish_rank"

# --------------------------
# Dataset
# --------------------------
class TabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame, art: PreprocessArtifacts):
        y = pd.to_numeric(df[art.target_name], errors="coerce").astype(float).values

        # numeric
        Xn = []
        for c in art.numeric_cols:
            col = pd.to_numeric(df[c], errors="coerce").astype(float)
            col = (col - art.num_means[c]) / (art.num_stds[c] if art.num_stds[c] > 0 else 1.0)
            Xn.append(col.values)
        Xn = np.stack(Xn, axis=1).astype(np.float32) if Xn else np.zeros((len(df), 0), np.float32)

        # categorical -> indices (unknown => "UNK")
        Xc = []
        for c in art.categorical_cols:
            m = art.cat_maps[c]
            vals = df[c].astype(str).fillna("UNK").replace({"nan": "UNK"})
            if c == "runner_name":
                vals = vals.map(_norm_name)
            idx = vals.map(lambda s: m.get(s, m.get("UNK", 0))).astype(np.int64).values
            Xc.append(idx)
        Xc = np.stack(Xc, axis=1).astype(np.int64) if Xc else np.zeros((len(df), 0), np.int64)

        self.Xn = torch.from_numpy(Xn)
        self.Xc = torch.from_numpy(Xc)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.Xn[i], self.Xc[i], self.y[i]

# --------------------------
# Model
# --------------------------
class TabularModel(nn.Module):
    def __init__(self, num_in: int, cat_cardinalities: List[int], hidden: List[int], dropout: float = 0.25):
        super().__init__()
        self.has_cat = len(cat_cardinalities) > 0
        emb_dim = lambda card: min(50, (card + 1) // 2)  # simple rule
        if self.has_cat:
            self.embs = nn.ModuleList([nn.Embedding(card, emb_dim(card)) for card in cat_cardinalities])
            cat_dim = sum(emb_dim(card) for card in cat_cardinalities)
        else:
            self.embs = None
            cat_dim = 0

        layers = []
        in_dim = num_in + cat_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        if self.has_cat and x_cat.numel() > 0:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
            x = torch.cat([x_num] + embs, dim=1)
        else:
            x = x_num
        out = self.mlp(x).squeeze(-1)
        return out

# --------------------------
# Prep functions
# --------------------------
def build_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, PreprocessArtifacts]:
    # Form features (leak-safe)
    form = build_horse_form_features(df)
    df = pd.concat([df, form], axis=1)

    # Choose features that actually exist
    numeric_cols = [c for c in (NUMERIC_COLS_BASE + FORM_NUMERIC) if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    # Normalise key names
    if "runner_name" in df.columns:
        df["runner_name"] = df["runner_name"].map(_norm_name)

    # Compute numeric stats on TRAIN later; here just return df and placeholders
    art = PreprocessArtifacts(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        num_means={},
        num_stds={},
        cat_maps={},
        cat_cardinalities={}
    )
    return df, art

def fit_encoders(train_df: pd.DataFrame, art: PreprocessArtifacts) -> PreprocessArtifacts:
    # numeric
    num_means = {c: float(pd.to_numeric(train_df[c], errors="coerce").astype(float).mean()) for c in art.numeric_cols}
    num_stds  = {c: float(pd.to_numeric(train_df[c], errors="coerce").astype(float).std(ddof=0)) for c in art.numeric_cols}

    # categorical mappings (train only), include UNK
    cat_maps = {}
    cat_cards = {}
    for c in art.categorical_cols:
        vals = train_df[c].astype(str).fillna("UNK").replace({"nan": "UNK"})
        if c == "runner_name":
            vals = vals.map(_norm_name)
        uniq = ["UNK"] + sorted(set(vals) - {"UNK"})
        mapping = {v: i for i, v in enumerate(uniq)}
        cat_maps[c] = mapping
        cat_cards[c] = len(mapping)

    return PreprocessArtifacts(
        numeric_cols=art.numeric_cols,
        categorical_cols=art.categorical_cols,
        num_means=num_means,
        num_stds=num_stds,
        cat_maps=cat_maps,
        cat_cardinalities=cat_cards,
        target_name=art.target_name
    )

def make_splits(df: pd.DataFrame, art: PreprocessArtifacts, seed: int = 42):
    # ensure date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    n = len(df)
    i_tr = int(0.70 * n)
    i_va = int(0.85 * n)
    train_df = df.iloc[:i_tr].copy()
    valid_df = df.iloc[i_tr:i_va].copy()
    test_df  = df.iloc[i_va:].copy()

    # fit stats on TRAIN only
    art = fit_encoders(train_df, art)

    return train_df, valid_df, test_df, art

# --------------------------
# Train / Eval
# --------------------------
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

    # mask non-finite (prevents sklearn crash)
    mask = np.isfinite(preds) & np.isfinite(targs)
    if mask.sum() == 0:
        return float(np.mean(losses) if losses else math.nan), math.nan, math.nan, math.nan

    preds_m = preds[mask]
    targs_m = targs[mask]

    mae = float(mean_absolute_error(targs_m, preds_m))
    rmse = float(np.sqrt(mean_squared_error(targs_m, preds_m)))
    rho = float(spearmanr(targs_m, preds_m).correlation)
    return float(np.mean(losses)), mae, rmse, rho

def train_model(train_ds, valid_ds, art: PreprocessArtifacts,
                hidden: List[int], dropout: float, lr: float,
                weight_decay: float, epochs: int, batch_size: int,
                device: str = "cpu", patience: int = 5):

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)

    num_in = len(art.numeric_cols)
    cat_cards = [art.cat_cardinalities[c] for c in art.categorical_cols]
    model = TabularModel(num_in=num_in, cat_cardinalities=cat_cards, hidden=hidden, dropout=dropout).to(device)

    criterion = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = math.inf
    best_state = None
    patience_left = patience

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb_num, xb_cat, yb in train_loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            out = model(xb_num, xb_cat)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()
            tr_losses.append(loss.item())

        # eval
        train_loss = float(np.mean(tr_losses)) if tr_losses else math.nan
        val_loss, val_mae, val_rmse, val_spear = evaluate(model, valid_loader, criterion, device)

        print(f"Epoch {ep:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} val_spearman={val_spear:.4f}")

        # early stopping on val_loss
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

# --------------------------
# End-to-end train
# --------------------------
def run_training(data_path: str,
                 hidden: List[int], dropout: float,
                 lr: float, weight_decay: float,
                 epochs: int, batch_size: int,
                 device: Optional[str] = None):

    set_seed(42)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    df = pd.read_parquet(data_path)

    # Clean rows to avoid NaNs in targets
    df = clean_training_rows(df)

    # Build features + lists
    df, art = build_preprocess(df)

    # Keep only what we need
    needed = ["date", "race_id", "finish_rank"] + art.numeric_cols + art.categorical_cols
    df = df[needed].copy()

    # Splits + fit stats/mappings
    train_df, valid_df, test_df, art = make_splits(df, art)

    # Datasets
    train_ds = TabularDataset(train_df, art)
    valid_ds = TabularDataset(valid_df, art)
    test_ds  = TabularDataset(test_df, art)

    # Train
    model = train_model(train_ds, valid_ds, art, hidden, dropout, lr, weight_decay, epochs, batch_size, device)

    # Final eval
    crit = nn.MSELoss()
    test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False)
    test_loss, test_mae, test_rmse, test_spear = evaluate(model, test_loader, crit, device)
    print(f"TEST | loss={test_loss:.4f} MAE={test_mae:.4f} RMSE={test_rmse:.4f} Spearman={test_spear:.4f}")

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    model_path = "model_regression.pth"
    torch.save(model.state_dict(), model_path)

    with open("preprocess.pkl", "wb") as f:
        pickle.dump(art, f)

    with open("metrics.json", "w") as f:
        json.dump({
            "test_loss": float(test_loss),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "test_spearman": float(test_spear),
            "numeric_cols": art.numeric_cols,
            "categorical_cols": art.categorical_cols
        }, f, indent=2)

    return model, art

# --------------------------
# Inference helper
# --------------------------
def _prepare_new_frame(df_new: pd.DataFrame, art: PreprocessArtifacts) -> TabularDataset:
    """Prepare an arbitrary race dataframe for inference using saved artifacts."""
    # Ensure required columns exist
    work = df_new.copy()
    if "runner_name" in work.columns:
        work["runner_name"] = work["runner_name"].map(_norm_name)

    # Add missing columns
    for c in art.numeric_cols:
        if c not in work.columns:
            work[c] = np.nan
    for c in art.categorical_cols:
        if c not in work.columns:
            work[c] = "UNK"

    # Dummy target if missing
    if art.target_name not in work.columns:
        work[art.target_name] = np.nan

    return TabularDataset(work, art)

def load_model_and_predict(df_new: pd.DataFrame,
                           model_path: str = "model_regression.pth",
                           preprocess_path: str = "preprocess.pkl",
                           hidden: List[int] = [256, 128, 64],
                           dropout: float = 0.25,
                           device: Optional[str] = None) -> pd.DataFrame:
    """
    Load saved model + preprocess and return DataFrame with prediction + predicted rank.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(preprocess_path, "rb") as f:
        art: PreprocessArtifacts = pickle.load(f)

    num_in = len(art.numeric_cols)
    cat_cards = [art.cat_cardinalities[c] for c in art.categorical_cols]
    model = TabularModel(num_in=num_in, cat_cardinalities=cat_cards, hidden=hidden, dropout=dropout).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = _prepare_new_frame(df_new, art)
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    preds = []
    with torch.no_grad():
        for xb_num, xb_cat, _ in loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            out = model(xb_num, xb_cat)
            preds.append(out.detach().cpu().numpy().ravel())
    preds = np.concatenate(preds).astype(float) if preds else np.array([])

    # smaller predicted finish is better; derive ranks (1 = best)
    pred_rank = preds.argsort().argsort() + 1

    out = df_new.copy()
    out["predicted_finish"] = preds
    out["pred_rank"] = pred_rank
    return out

# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="five_year_dataset.parquet")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden", type=str, default="256,128,64", help="comma-separated hidden sizes")
    args = parser.parse_args()

    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    run_training(
        data_path=args.data,
        hidden=hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
