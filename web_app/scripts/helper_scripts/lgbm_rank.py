#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LightGBM for race ordering + inference from TAB schedule JSON.

Commands:
  - Train ranking:   python lgbm_rank.py train
  - Train regression:python lgbm_rank.py train_reg
  - Infer:           python lgbm_rank.py infer <meet_no> <race_no> [date=YYYY-MM-DD] [tau=0.4] [n_samples=4000] [alpha=0.6]

What it does:
  • Trains LambdaRank (optimises per-race ordering) with a chronological split (70/15/15)
  • Also includes a regression baseline on finish_rank (often solid for global Spearman)
  • Uses leak-safe per-horse form features
  • Saves artefacts to artifacts_gbm/ (model + encoder + medians + feature lists)
  • Inference: fetch schedule JSON, derive fav_rank + market implied probs, predict GBM scores,
    convert to position probabilities via PL, blend with market, and print win % + fair NZ odds
"""

import sys as _sys
_sys.modules['__main__'] = _sys.modules[__name__]

import argparse
import json
import math
import os
import pickle
import re

import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.preprocessing import OrdinalEncoder
import sys


# add near the top
import subprocess, pathlib

def train_pytorch_nn_cli(data_path, epochs, batch, lr, weight_decay, dropout, hidden):
    # resolve pytorch_pre.py next to this script
    script = pathlib.Path(__file__).with_name("pytorch_pre.py")
    if not script.exists():
        raise RuntimeError(f"Couldn't find pytorch_pre.py at {script}")
    cmd = [
        sys.executable, str(script),
        "--data", data_path,
        "--epochs", str(epochs),
        "--batch_size", str(batch),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--dropout", str(dropout),
        "--hidden", ",".join(map(str, hidden)),
    ]
    print("\n[TRAIN] PyTorch NN via CLI …")
    subprocess.run(cmd, check=True)

def _to_bool_mask_any(s: pd.Series) -> pd.Series:
    """Coerce any 'is_scratched' flavour to clean booleans."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    # strings/object: true/1/yes/y -> True
    return s.astype(str).str.strip().str.lower().isin({"true","t","1","yes","y"})


# -----------------------------
# Config
# -----------------------------
DATA_PATH = "five_year_dataset.parquet"
ART_DIR = "artifacts_gbm"
os.makedirs(ART_DIR, exist_ok=True)

# Features (mirror the PyTorch pipeline)
NUMERIC_COLS_BASE = [
    "meeting_number",
    "race_number",
    "race_distance_m",
    "stake",
    "fav_rank",
    "race_length",
    "race_number_sched",
    "entrant_weight",
]
FORM_NUMERIC = [
    "horse_starts_prior",
    "horse_win_rate_prior",
    "horse_top3_rate_prior",
    "horse_avg_finish_prior",
    "horse_last_finish",
    "horse_avg_fav_rank_prior",
    "horse_avg_margin_prior",
    "days_since_last_run",
]
CATEGORICAL_COLS = [
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

# -----------------------------
# Utils
# -----------------------------

def _norm_name(s: str) -> str:
    return " ".join(str(s).strip().upper().split())

def _to_float(x):
    try:
        return float(x) if x is not None and str(x).strip() != "" else np.nan
    except ValueError:
        return np.nan

def _parse_money_to_float(s):
    if s is None:
        return np.nan
    s = re.sub(r"[^0-9.]", "", str(s))
    try:
        return float(s) if s else np.nan
    except ValueError:
        return np.nan

def probs_to_decimal_odds(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return 1.0 / p

def logit(p, eps=1e-9):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))

def inv_logit(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))

def blend_with_market(win_model, win_market, alpha=0.6):
    """
    Logit blend model vs market, then renormalise so sum(p)=1 across the field.
    alpha in [0,1]: 1 trusts model most; 0 trusts market most.
    """
    win_model = np.asarray(win_model, dtype=float)
    win_market = np.asarray(win_market, dtype=float)
    z = alpha * logit(win_model) + (1 - alpha) * logit(win_market)
    p = inv_logit(z)
    s = p.sum()
    return p / s if s > 0 else p

# -----------------------------
# Form features (leak-safe)
# -----------------------------

def build_horse_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-horse lagged features using only prior races.
    Requires columns: runner_name, date, finish_rank, fav_rank, margin_len (optional).
    Returns frame aligned to df.index.
    """
    req = ["runner_name", "date", "finish_rank", "fav_rank", "margin_len"]
    tmp = df.copy()
    for c in req:
        if c not in tmp.columns:
            tmp[c] = np.nan
    work = tmp[req].copy()
    work["runner_name"] = work["runner_name"].map(_norm_name)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.sort_values(["runner_name", "date"]).reset_index()

    g = work.groupby("runner_name", sort=False)
    work["horse_starts_prior"] = g.cumcount()

    work["is_win"] = (work["finish_rank"] == 1).astype(float)
    work["is_top3"] = (work["finish_rank"] <= 3).astype(float)

    for col in ["is_win", "is_top3", "finish_rank", "fav_rank", "margin_len"]:
        s = work[col]
        work[f"cum_{col}_prior"] = g[col].cumsum() - s

    cnt = work["horse_starts_prior"].replace(0, np.nan)
    work["horse_win_rate_prior"] = work["cum_is_win_prior"] / cnt
    work["horse_top3_rate_prior"] = work["cum_is_top3_prior"] / cnt
    work["horse_avg_finish_prior"] = work["cum_finish_rank_prior"] / cnt
    work["horse_avg_fav_rank_prior"] = work["cum_fav_rank_prior"] / cnt
    work["horse_avg_margin_prior"] = work["cum_margin_len_prior"] / cnt

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

# -----------------------------
# Artefacts
# -----------------------------

@dataclass
class GBMArtifacts:
    numeric_cols: List[str]
    categorical_cols: List[str]
    medians: Dict[str, float]
    encoder: OrdinalEncoder  # fitted on TRAIN
    feature_order: List[str] # numeric + cat (encoded order)

# -----------------------------
# Training: ranking
# -----------------------------

def _prep_for_training():
    df = pd.read_parquet(DATA_PATH)

    # Basic filters / checks
    if "is_scratched" in df.columns:
        mask_scr = _to_bool_mask_any(df["is_scratched"])
        df = df[~mask_scr]

    for need in ["finish_rank", "race_id", "date"]:
        if need not in df.columns:
            raise ValueError(f"Training requires '{need}' in the dataset.")

    # Normalise names so train/infer match
    if "runner_name" in df.columns:
        df["runner_name"] = df["runner_name"].map(_norm_name)

    # Build leak-safe per-horse form features
    form_feat = build_horse_form_features(df)
    df = pd.concat([df, form_feat], axis=1)

    # Feature lists
    numeric_cols = [c for c in NUMERIC_COLS_BASE + FORM_NUMERIC if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    # Keep only what we need
    keep = ["race_id", "date", "finish_rank"] + numeric_cols + categorical_cols
    df = df[keep].copy()

    # Drop invalid finishes (NaN/<=0)
    df["finish_rank"] = pd.to_numeric(df["finish_rank"], errors="coerce")
    df = df[df["finish_rank"].notna() & (df["finish_rank"] > 0)].copy()

    # Chronological split, then within each split sort by race_id so groups are contiguous
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    n = len(df)
    i_tr = int(0.70 * n)
    i_va = int(0.85 * n)

    train_df = df.iloc[:i_tr].copy().sort_values("race_id").reset_index(drop=True)
    valid_df = df.iloc[i_tr:i_va].copy().sort_values("race_id").reset_index(drop=True)
    test_df  = df.iloc[i_va:].copy().sort_values("race_id").reset_index(drop=True)

    # Impute numerics with TRAIN medians
    medians = {c: float(pd.to_numeric(train_df[c], errors="coerce").median()) for c in numeric_cols}
    for part in (train_df, valid_df, test_df):
        for c in numeric_cols:
            part.loc[:, c] = pd.to_numeric(part[c], errors="coerce").fillna(medians[c])

    # Ordinal-encode categoricals (fit on TRAIN only)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if categorical_cols:
        train_cat = enc.fit_transform(train_df[categorical_cols].astype(str))
        valid_cat = enc.transform(valid_df[categorical_cols].astype(str))
        test_cat  = enc.transform(test_df[categorical_cols].astype(str))
    else:
        train_cat = np.empty((len(train_df), 0))
        valid_cat = np.empty((len(valid_df), 0))
        test_cat  = np.empty((len(test_df), 0))

    Xtr = np.hstack([train_df[numeric_cols].values, train_cat])
    Xva = np.hstack([valid_df[numeric_cols].values, valid_cat])
    Xte = np.hstack([test_df[numeric_cols].values,  test_cat])

    feat_names = [*numeric_cols, *categorical_cols]
    cat_idx = list(range(len(numeric_cols), len(feat_names)))  # categorical columns indices in X

    return (
        train_df, valid_df, test_df,
        Xtr, Xva, Xte,
        feat_names, cat_idx,
        enc, medians, numeric_cols, categorical_cols
    )

def train():
    print("Loading data…")
    (
        train_df, valid_df, test_df,
        Xtr, Xva, Xte,
        feat_names, cat_idx,
        enc, medians, numeric_cols, categorical_cols
    ) = _prep_for_training()

    # Build 0-based relevance and group sizes (frames are race_id-contiguous)
    def relevance_and_groups(frame: pd.DataFrame):
        grp = frame.groupby("race_id", sort=False)["finish_rank"]
        # winner => field_size-1, last => 0
        rel = grp.transform(lambda s: (len(s) - s).clip(lower=0)).astype(int).values
        groups = frame.groupby("race_id", sort=False).size().values
        return rel, groups

    ytr, gtr = relevance_and_groups(train_df)
    yva, gva = relevance_and_groups(valid_df)
    yte, gte = relevance_and_groups(test_df)

    # LightGBM datasets (mark categoricals)
    lgb_tr = lgb.Dataset(
        Xtr, label=ytr, group=gtr,
        feature_name=feat_names,
        categorical_feature=cat_idx
    )
    lgb_va = lgb.Dataset(
        Xva, label=yva, group=gva, reference=lgb_tr,
        feature_name=feat_names,
        categorical_feature=cat_idx
    )

    # Params (+ label_gain long enough for your biggest field)
    max_label = int(max(ytr.max(), yva.max(), yte.max()))
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_at=[1, 3, 5],
        learning_rate=0.05,
        num_leaves=127,              # more capacity for high-card cats
        min_data_in_leaf=60,
        feature_fraction=0.90,
        bagging_fraction=0.90,
        bagging_freq=1,
        max_depth=-1,
        verbose=-1,
        seed=42,
        max_bin=255,
        min_sum_hessian_in_leaf=1e-3,
        label_gain=list(range(max_label + 1)),
    )

    print("Training LightGBM LambdaRank…")
    callbacks = [
        lgb.early_stopping(stopping_rounds=200, verbose=True),
        lgb.log_evaluation(period=200),
    ]
    model = lgb.train(
        params,
        lgb_tr,
        valid_sets=[lgb_tr, lgb_va],
        valid_names=["train", "valid"],
        num_boost_round=6000,
        callbacks=callbacks,
    )

    # Evaluate on test (higher score = better; compare vs -score for rank)
    print("Evaluating on test…")
    score = model.predict(Xte, num_iteration=model.best_iteration)
    rho = spearmanr(test_df["finish_rank"].values, -score).correlation
    print(f"Test Spearman (higher is better): {rho:.3f}")

    # Save artefacts
    art = GBMArtifacts(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        medians=medians,
        encoder=enc,
        feature_order=[*numeric_cols, *categorical_cols],
    )
    model_path = os.path.join(ART_DIR, "model_lgbm.txt")
    artefacts_path = os.path.join(ART_DIR, "preprocess_gbm.pkl")

    model.save_model(model_path, num_iteration=model.best_iteration)
    with open(artefacts_path, "wb") as f:
        pickle.dump(art, f)
    with open(os.path.join(ART_DIR, "metrics_gbm.json"), "w") as f:
        json.dump({"test_spearman": float(rho)}, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved preprocess artefacts to {artefacts_path}")

# -----------------------------
# Training: regression baseline
# -----------------------------

def train_reg():
    print("Loading data (regression)…")
    (
        train_df, valid_df, test_df,
        Xtr, Xva, Xte,
        feat_names, cat_idx,
        enc, medians, numeric_cols, categorical_cols
    ) = _prep_for_training()

    ytr = train_df["finish_rank"].values.astype(float)
    yva = valid_df["finish_rank"].values.astype(float)
    yte = test_df["finish_rank"].values.astype(float)

    lgb_tr = lgb.Dataset(
        Xtr, label=ytr,
        feature_name=feat_names,
        categorical_feature=cat_idx
    )
    lgb_va = lgb.Dataset(
        Xva, label=yva, reference=lgb_tr,
        feature_name=feat_names,
        categorical_feature=cat_idx
    )

    params = dict(
        objective="regression",
        metric=["l2", "l1"],
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=60,
        feature_fraction=0.90,
        bagging_fraction=0.90,
        bagging_freq=1,
        max_depth=-1,
        verbose=-1,
        seed=42,
        max_bin=255,
        min_sum_hessian_in_leaf=1e-3,
    )

    print("Training LightGBM Regression…")
    callbacks = [
        lgb.early_stopping(stopping_rounds=200, verbose=True),
        lgb.log_evaluation(period=200),
    ]
    model = lgb.train(
        params, lgb_tr,
        valid_sets=[lgb_tr, lgb_va], valid_names=["train", "valid"],
        num_boost_round=6000,
        callbacks=callbacks,
    )

    # Evaluate with Spearman on test (lower pred => better rank)
    score = model.predict(Xte, num_iteration=model.best_iteration)
    rho = spearmanr(test_df["finish_rank"].values, score).correlation
    print(f"[REG] Test Spearman: {rho:.3f}")

    model.save_model(os.path.join(ART_DIR, "model_lgbm_reg.txt"), num_iteration=model.best_iteration)
    with open(os.path.join(ART_DIR, "metrics_gbm_reg.json"), "w") as f:
        json.dump({"test_spearman": float(rho)}, f, indent=2)

# -----------------------------
# Loading + scoring
# -----------------------------

@dataclass
class _LoadedArt:
    numeric_cols: List[str]
    categorical_cols: List[str]
    medians: Dict[str, float]
    encoder: OrdinalEncoder
    feature_order: List[str]
    booster: lgb.Booster

def _load_artifacts(model_name: str = "model_lgbm.txt") -> _LoadedArt:
    booster = lgb.Booster(model_file=os.path.join(ART_DIR, model_name))
    with open(os.path.join(ART_DIR, "preprocess_gbm.pkl"), "rb") as f:
        art: GBMArtifacts = pickle.load(f)
    return _LoadedArt(
        numeric_cols=art.numeric_cols,
        categorical_cols=art.categorical_cols,
        medians=art.medians,
        encoder=art.encoder,
        feature_order=art.feature_order,
        booster=booster
    )

def load_gbm_and_predict(df_new: pd.DataFrame, model_name: str = "model_lgbm_reg.txt"):
    """
    Returns dict with:
      - score: LightGBM raw scores
      - pred_rank: smaller = better (1 = best)
        * ranking model: higher score = better  -> sort by -score
        * regression model: lower pred = better -> sort by +score
    """
    la = _load_artifacts(model_name=model_name)

    # Ensure all features exist
    work = pd.DataFrame()
    # numeric
    for c in la.numeric_cols:
        if c in df_new.columns:
            col = pd.to_numeric(df_new[c], errors="coerce").fillna(la.medians[c])
        else:
            col = pd.Series([la.medians[c]] * len(df_new))
        work[c] = col
    # categorical
    for c in la.categorical_cols:
        if c in df_new.columns:
            col = df_new[c].astype(str).fillna("UNK").replace({"nan": "UNK"})
        else:
            col = pd.Series(["UNK"] * len(df_new))
        if c == "runner_name":
            col = col.map(_norm_name)
        work[c] = col

    # encode
    if la.categorical_cols:
        cat_arr = la.encoder.transform(work[la.categorical_cols].astype(str))
        X_num = work[la.numeric_cols].values
        X = np.hstack([X_num, cat_arr])
    else:
        X = work[la.numeric_cols].values

    scores = la.booster.predict(X, num_iteration=la.booster.best_iteration)

    # Determine if this booster is regression or ranking
    is_regression = False
    try:
        # LightGBM stores objective in params or dump_model
        obj = ""
        if hasattr(la.booster, "params") and isinstance(la.booster.params, dict):
            obj = la.booster.params.get("objective", "") or ""
        if not obj:
            md = la.booster.dump_model()
            obj = (md.get("objective") or "").lower()
        is_regression = "regression" in str(obj) or obj in {"l2", "l1", "huber", "quantile"}
    except Exception:
        # Fallback: infer from file name
        is_regression = "reg" in str(model_name).lower()

    # Rank: 1 = best
    if is_regression:
        order = (scores).argsort().argsort() + 1     # lower better
    else:
        order = (-scores).argsort().argsort() + 1    # higher better

    return {"score": scores, "pred_rank": order}

# -----------------------------
# Schedule fetch + parsing
# -----------------------------

def fetch_schedule_json(date_str: str, meet_no: int, race_no: int) -> dict:
    url = f"https://json.tab.co.nz/schedule/{date_str}/{meet_no}/{race_no}"
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))

def _extract_fixed_win(entry: dict):
    # Try common shapes for fixed odds
    for k in ("fixed_win", "win"):
        if k in entry:
            v = _to_float(entry.get(k))
            if not math.isnan(v):
                return v
    nests = [("fixedOdds", "win"), ("fixed", "win"), ("odds", "win"), ("prices", "win")]
    for a, b in nests:
        node = entry.get(a)
        if isinstance(node, dict) and b in node:
            v = _to_float(node.get(b))
            if not math.isnan(v):
                return v
    return np.nan

def schedule_json_to_df(obj: dict) -> pd.DataFrame:
    rows = []
    date = obj.get("date") or obj.get("day") or obj.get("meetingDate")
    for m in obj.get("meetings", []):
        meeting_number = m.get("number") or m.get("meetingNumber")
        meeting_country = m.get("country")
        meeting_venue = m.get("venue") or m.get("meetingName")
        meeting_id = m.get("id") or m.get("meetingId")

        races = m.get("races") or obj.get("races") or []
        for r in races:
            race_id = r.get("id") or r.get("raceId")
            race_number = r.get("number") or r.get("raceNumber")
            race_class = r.get("class")
            race_track = r.get("track") or r.get("trackCondition")
            race_weather = r.get("weather")
            race_name = r.get("name") or r.get("raceName")
            race_venue = r.get("venue") or meeting_venue
            race_distance_m = _to_float(r.get("length") or r.get("distanceMeters"))
            race_length = race_distance_m
            stake = _parse_money_to_float(r.get("stake"))

            entries = r.get("entries") or r.get("runners") or []
            tmp = []
            for e in entries:
                if e.get("scratched", False) or e.get("isScratched", False):
                    continue
                fixed_win = _extract_fixed_win(e)
                barrier = e.get("barrier") if e.get("barrier") is not None else e.get("draw")
                runner_no = e.get("number") or e.get("runnerNumber") or e.get("runner_number")
                jockey = e.get("jockey") or e.get("jockeyName") or "UNK"
                runner_name = e.get("name") or e.get("runnerName") or "UNK"

                tmp.append({
                    "date": date,
                    "meeting_id": meeting_id,
                    "race_id": race_id,
                    "race_name": race_name,
                    "meeting_number": meeting_number,
                    "race_number": race_number,
                    "race_distance_m": race_distance_m,
                    "race_length": race_length,
                    "stake": stake,
                    "entrant_weight": _to_float(e.get("weight")),
                    "race_class": race_class,
                    "race_track": race_track,
                    "race_weather": race_weather,
                    "meeting_country": meeting_country,
                    "meeting_venue": race_venue,
                    "race_class_sched": race_class,
                    "race_number_sched": race_number,
                    "entrant_barrier": str(barrier) if barrier is not None else "UNK",
                    "runner_number": int(runner_no) if runner_no is not None else np.nan,
                    "entrant_jockey": jockey,
                    "runner_name": _norm_name(runner_name),
                    "fixed_win": fixed_win,
                })

            if not tmp:
                continue

            df_tmp = pd.DataFrame(tmp)
            # market implied probs (strip overround so sum ≈ 1)
            if df_tmp["fixed_win"].notna().any():
                ip = 1.0 / df_tmp["fixed_win"]
                s = ip.sum()
                df_tmp["implied_p"] = (ip / s) if s > 0 else ip
                # fav rank: 1 = shortest odds
                df_tmp["fav_rank"] = df_tmp["fixed_win"].rank(ascending=True, method="min").astype(int)

            rows.extend(df_tmp.to_dict("records"))

    return pd.DataFrame(rows)

# -----------------------------
# Scores -> position probs (PL)
# -----------------------------

def _scores_to_pl_win_probs(scores: np.ndarray, tau: float = 0.4) -> np.ndarray:
    """
    For a single race: GBM scores (higher better) -> win probs via softmax with temperature tau.
    This equals the PL top-1 probability.
    """
    s = np.asarray(scores, dtype=float)
    ex = np.exp((s - s.mean()) / float(tau))
    return ex / ex.sum()

def _pl_sample_orders_from_scores(scores: np.ndarray, tau: float, n_samples: int, seed: int = 42):
    """
    Monte Carlo sampling of full finish orders from PL with skill derived from GBM scores.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, dtype=float)
    pl_scores = np.exp((s - s.mean()) / float(tau))  # positive skill
    n = len(s)
    pos_counts = np.zeros((n, n), dtype=np.int32)
    for _ in range(n_samples):
        alive = np.arange(n)
        weights = pl_scores.copy()
        for pos in range(n):
            p = weights / weights.sum()
            i = rng.choice(len(alive), p=p)
            idx = alive[i]
            pos_counts[idx, pos] += 1
            alive = np.delete(alive, i)
            weights = np.delete(weights, i)
    return pos_counts / float(n_samples)

# -----------------------------
# Inference from schedule
# -----------------------------

def infer_one_race(meet_no: int, race_no: int, date_str: str, tau: float = 0.4, n_samples: int = 4000, alpha: float = 0.6):
    obj = fetch_schedule_json(date_str, meet_no, race_no)
    race_df = schedule_json_to_df(obj)
    if race_df.empty:
        print("No eligible (unscratched) runners found in schedule JSON.")
        return

    # Predict GBM scores (ranking model by default)
    pred = load_gbm_and_predict(race_df, model_name="model_lgbm_reg.txt")
    scores = pred["score"]
    pred_rank = pred["pred_rank"]

    # Model-only probs
    win_model = _scores_to_pl_win_probs(scores, tau=tau)
    pos_probs = _pl_sample_orders_from_scores(scores, tau=tau, n_samples=n_samples, seed=42)
    top3_model = pos_probs[:, :min(3, len(scores))].sum(axis=1)

    # Confidence from sampling (90% CI on win prob)
    z = 1.645
    se = np.sqrt(np.clip(win_model * (1 - win_model) / n_samples, 0.0, 1.0))
    ci_low = np.clip(win_model - z * se, 0, 1)
    ci_high = np.clip(win_model + z * se, 0, 1)
    confidence_pct = np.clip(1.0 - (ci_high - ci_low), 0.0, 1.0) * 100.0

    # Market blend if available
    if "implied_p" in race_df.columns and race_df["implied_p"].notna().any():
        market_p = race_df["implied_p"].to_numpy()
        win_blend = blend_with_market(win_model, market_p, alpha=alpha)
        se_blend = se * alpha
        ci_low_b = np.clip(win_blend - z * se_blend, 0, 1)
        ci_high_b = np.clip(win_blend + z * se_blend, 0, 1)
        confidence_blend = np.clip(1.0 - (ci_high_b - ci_low_b), 0.0, 1.0) * 100.0
    else:
        win_blend = win_model
        ci_low_b = ci_low
        ci_high_b = ci_high
        confidence_blend = confidence_pct

    out = pd.DataFrame({
        "runner_number": race_df.get("runner_number", pd.Series([np.nan]*len(scores))).values,
        "runner_name": race_df.get("runner_name", pd.Series([None]*len(scores))).values,
        "entrant_jockey": race_df.get("entrant_jockey", pd.Series([None]*len(scores))).values,
        "fav_rank": race_df.get("fav_rank", pd.Series([np.nan]*len(scores))).values,
        "pred_rank": pred_rank,
        "win_prob": win_blend,
        "top3_prob_model": top3_model,
        "win_ci90_low": ci_low_b,
        "win_ci90_high": ci_high_b,
        "confidence_%": confidence_blend,
    })

    # Fair NZ odds (no margin) and pretty %
    out["win_%"] = (out["win_prob"] * 100).round(2)
    out["win_$fair"] = probs_to_decimal_odds(out["win_prob"]).round(2)
    out["top3_%"] = (out["top3_prob_model"] * 100).round(2)
    out["top3_$fair"] = probs_to_decimal_odds(out["top3_prob_model"]).round(2)
    out["win_%_90CI"] = (out["win_ci90_low"] * 100).round(1).astype(str) + "–" + (out["win_ci90_high"] * 100).round(1).astype(str)

    out = out.sort_values("win_prob", ascending=False).reset_index(drop=True)

    cols = [
        "runner_number","runner_name","entrant_jockey","fav_rank","pred_rank",
        "win_%","win_$fair","top3_%","top3_$fair","confidence_%","win_%_90CI"
    ]
    print(out[cols].to_string(index=False))

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train")
    sub.add_parser("train_reg")

    p_inf = sub.add_parser("infer")
    p_inf.add_argument("meet", type=int)
    p_inf.add_argument("race", type=int)
    p_inf.add_argument("date", nargs="?", default=datetime.now().strftime("%Y-%m-%d"))
    p_inf.add_argument("tau", nargs="?", type=float, default=0.4)
    p_inf.add_argument("n_samples", nargs="?", type=int, default=4000)
    p_inf.add_argument("alpha", nargs="?", type=float, default=0.6)

    args = parser.parse_args()

    if args.cmd == "train":
        train()
    elif args.cmd == "train_reg":
        train_reg()
    elif args.cmd == "infer":
        infer_one_race(args.meet, args.race, args.date, tau=args.tau, n_samples=args.n_samples, alpha=args.alpha)

if __name__ == "__main__":
    main()
