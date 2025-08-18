#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanity check: per-race hit@1 vs TAB results, using the live inference path.

Usage:
  # fetch results first (or pass --results-csv)
  python fetch_tab_results.py --date 2025-08-17

  # run sanity check
  python eval_holdout.py --date 2025-08-17
  python eval_holdout.py --date 2025-08-17 --meets 22 31
"""

import argparse
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from infer_from_schedule_json import fetch_schedule_json, schedule_json_to_df
from pytorch_pre import load_model_and_predict

# ---------- name normalisation (match model) ----------
import unicodedata
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
def _norm_name(s: str) -> str:
    return " ".join(_strip_accents(str(s)).strip().upper().split())

# ---------- NZ yesterday default ----------
def nz_yesterday_str() -> str:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Pacific/Auckland")
    except Exception:
        import pytz
        tz = pytz.timezone("Pacific/Auckland")
    return (datetime.now(tz) - timedelta(days=1)).date().isoformat()

# ---------- helpers ----------
ALT_NAME_FIELDS = ["runner_name", "name", "horse_name", "runner", "runnerName", "horseName"]

def _as_df(obj) -> pd.DataFrame:
    """Coerce load_model_and_predict output to DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        for k in ("df", "out", "pred", "pred_df", "predictions", "result", "results"):
            v = obj.get(k)
            if isinstance(v, pd.DataFrame):
                return v.copy()
            if isinstance(v, (list, tuple)) and v:
                try: return pd.DataFrame(v)
                except Exception: pass
        try: return pd.DataFrame(obj)
        except Exception: pass
    if isinstance(obj, (list, tuple)) and obj:
        if isinstance(obj[0], pd.DataFrame):
            return obj[0].copy()
        try: return pd.DataFrame(obj)
        except Exception: pass
    raise TypeError(f"Unsupported prediction return type: {type(obj)}")

def _ensure_runner_name(pred: pd.DataFrame, df_sched: pd.DataFrame) -> pd.DataFrame:
    """Guarantee pred['runner_name'] exists; fill from schedule if needed."""
    # 1) if any name-like col exists in pred, rename to runner_name
    for f in ALT_NAME_FIELDS:
        if f in pred.columns:
            if f != "runner_name":
                pred = pred.rename(columns={f: "runner_name"})
            # If empty strings present, fill from schedule where possible
            if pred["runner_name"].astype(str).str.strip().eq("").any() and "runner_name" in df_sched.columns and len(pred) == len(df_sched):
                mask = pred["runner_name"].astype(str).str.strip().eq("")
                pred.loc[mask, "runner_name"] = df_sched.loc[mask, "runner_name"].astype(str).values
            return pred

    # 2) merge from schedule on runner_number if we can
    if "runner_number" in pred.columns and "runner_number" in df_sched.columns:
        if "runner_name" in df_sched.columns:
            right = df_sched[["runner_number", "runner_name"]].copy()
        else:
            alt = next((c for c in ALT_NAME_FIELDS if c in df_sched.columns), None)
            right = df_sched[["runner_number", alt]].rename(columns={alt: "runner_name"}) if alt else None
        if right is not None:
            merged = pred.merge(right, on="runner_number", how="left", suffixes=("", "_sched"))
            if "runner_name" not in merged.columns and "runner_name_sched" in merged.columns:
                merged = merged.rename(columns={"runner_name_sched": "runner_name"})
            return merged

    # 3) last resort: index-align if lengths match
    if "runner_name" in df_sched.columns and len(pred) == len(df_sched):
        pred["runner_name"] = df_sched["runner_name"].astype(str).values
        return pred
    for alt in ALT_NAME_FIELDS:
        if alt in df_sched.columns and len(pred) == len(df_sched):
            pred["runner_name"] = df_sched[alt].astype(str).values
            return pred

    raise KeyError("runner_name not found in predictions and could not be resolved from schedule.")

def _inject_fav_rank(pred: pd.DataFrame, df_sched: pd.DataFrame) -> pd.DataFrame:
    """Ensure pred has fav_rank to use as a fallback signal."""
    if "fav_rank" in pred.columns and pd.to_numeric(pred["fav_rank"], errors="coerce").notna().any():
        return pred
    if "fav_rank" in df_sched.columns:
        # index-align if same length, else merge on runner_number
        if len(pred) == len(df_sched):
            pred["fav_rank"] = df_sched["fav_rank"].values
            return pred
        if "runner_number" in pred.columns and "runner_number" in df_sched.columns:
            pred = pred.merge(df_sched[["runner_number", "fav_rank"]], on="runner_number", how="left", suffixes=("", "_sched"))
            if "fav_rank" not in pred.columns and "fav_rank_sched" in pred.columns:
                pred = pred.rename(columns={"fav_rank_sched": "fav_rank"})
    return pred

def _pick_pos(series: pd.Series, higher_is_better: bool) -> Optional[int]:
    """Return an iloc-position (0..N-1) for the best row by this column, ignoring NaNs."""
    if series is None:
        return None
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)
    mask = ~np.isnan(arr)
    if not mask.any():
        return None
    # Use positions to avoid label lookups with NaN
    return int(np.nanargmax(arr) if higher_is_better else np.nanargmin(arr))

def predicted_winner_row(df_pred: pd.DataFrame, df_sched: pd.DataFrame) -> Optional[pd.Series]:
    """
    Robust top-pick selection:
      1) p_win (desc)
      2) pred_rank (asc)
      3) score (desc)
      4) prob/win_prob (desc)
      5) fav_rank (asc)  [injected from schedule if missing]
      6) fallback: first row
    Returns None if df is empty.
    """
    if df_pred is None or df_pred.empty:
        return None

    # Ensure we have fav_rank as a fallback
    df_pred = _inject_fav_rank(df_pred, df_sched)

    priorities = [
        ("p_win", True),
        ("pred_rank", False),
        ("score", True),
        ("prob", True),
        ("win_prob", True),
        ("fav_rank", False),
    ]
    for col, higher in priorities:
        if col in df_pred.columns:
            pos = _pick_pos(df_pred[col], higher)
            if pos is not None and 0 <= pos < len(df_pred):
                return df_pred.iloc[pos]

    # final fallback
    try:
        return df_pred.iloc[0]
    except Exception:
        return None

def actual_winner_name(df_results: pd.DataFrame) -> Optional[str]:
    w = df_results.loc[df_results["finish_rank"] == 1]
    if w.empty:
        return None
    return str(w.iloc[0]["runner_name"])

# ---------- main evaluation ----------
def evaluate_day(date_str: str, results_csv: Optional[str] = None, meets: Optional[list] = None, verbose: bool = True) -> Tuple[float, int]:
    # Load results
    if results_csv is None:
        results_csv = os.path.join("data", "results", date_str, "results_flat.csv")
    if not os.path.exists(results_csv):
        raise SystemExit(f"Results CSV not found: {results_csv}\nRun: python fetch_tab_results.py --date {date_str}")

    res = pd.read_csv(results_csv)
    need_cols = {"meeting_number", "race_number", "runner_name", "finish_rank"}
    missing = need_cols - set(res.columns)
    if missing:
        raise SystemExit(f"Results CSV missing columns: {missing}")

    # Completed only
    res = res.dropna(subset=["finish_rank"]).copy()
    res["finish_rank"] = pd.to_numeric(res["finish_rank"], errors="coerce").astype("Int64")
    if meets:
        res = res[res["meeting_number"].isin(meets)].copy()

    # Normalised for matching
    res["_rn_norm"] = res["runner_name"].map(_norm_name)

    # Pairs of races
    pairs = res[["meeting_number", "race_number"]].drop_duplicates().sort_values(["meeting_number", "race_number"]).values.tolist()

    hits = 0
    races = 0
    examples = []

    for meet_no, race_no in pairs:
        sched = fetch_schedule_json(date_str, int(meet_no), int(race_no))
        df_sched = schedule_json_to_df(sched)
        if df_sched is None or df_sched.empty:
            continue

        # Predict (robust to dict/tuple returns)
        pred_raw = load_model_and_predict(df_sched)
        pred = _as_df(pred_raw)

        # Attach identifiers if missing
        if "meeting_number" not in pred.columns:
            pred["meeting_number"] = int(meet_no)
        if "race_number" not in pred.columns:
            pred["race_number"] = int(race_no)

        # Ensure runner_name exists (fill from schedule if absent)
        pred = _ensure_runner_name(pred, df_sched)
        pred["_rn_norm"] = pred["runner_name"].map(_norm_name)

        # Choose model's top pick (robust)
        top = predicted_winner_row(pred, df_sched)
        if top is None:
            continue
        top_name = str(top.get("runner_name", ""))
        top_norm = _norm_name(top_name)

        # Actual winner
        r_slice = res[(res["meeting_number"] == meet_no) & (res["race_number"] == race_no)]
        actual = actual_winner_name(r_slice)
        if actual is None:
            continue
        actual_norm = _norm_name(actual)

        races += 1
        hit = int(top_norm == actual_norm)
        hits += hit

        if verbose and len(examples) < 15:
            examples.append(f"M{meet_no} R{race_no}: pick='{top_name}' | winner='{actual}' | {'HIT' if hit else 'MISS'}")

    hit_at1 = (hits / races) if races else 0.0

    if verbose:
        print(f"Per-race hit@1: {hit_at1:.3%}  (races={races}, hits={hits})")
        if examples:
            print("\nSamples:")
            for s in examples:
                print("  " + s)

    return hit_at1, races

def main():
    ap = argparse.ArgumentParser(description="Sanity check: per-race hit@1 vs TAB results.")
    ap.add_argument("--date", default=nz_yesterday_str(), help="YYYY-MM-DD (default: yesterday NZ)")
    ap.add_argument("--results-csv", default=None, help="Path to results_flat.csv (default under data/results/<date>/)")
    ap.add_argument("--meets", nargs="*", type=int, help="Optional meet filter, e.g. 22 31")
    args = ap.parse_args()

    evaluate_day(args.date, args.results_csv, args.meets, verbose=True)

if __name__ == "__main__":
    main()
