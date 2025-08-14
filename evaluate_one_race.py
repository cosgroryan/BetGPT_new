#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import urllib.request
import sys
from datetime import datetime
import numpy as np
import pandas as pd

# --- Make saved pickle artefacts loadable (maps __main__.PreprocessArtifacts) ---
import sys as _sys, pytorch_pre as _pp
_sys.modules['__main__'] = _pp  # pickle compat for preprocess.pkl

# Use your existing inference helpers
from infer_from_schedule_json import (
    fetch_schedule_json,
    schedule_json_to_df,
    estimate_position_probs_for_race_df,
)

# -----------------------------
# Utils
# -----------------------------

def _norm_name(s: str) -> str:
    return " ".join(str(s).strip().upper().split())

def fetch_results_json(date_str: str, meet_no: int, race_no: int) -> dict:
    url = f"https://json.tab.co.nz/results/{date_str}/{meet_no}/{race_no}"
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))

def results_to_df(obj: dict) -> pd.DataFrame:
    """
    Parse TAB results JSON into a DataFrame with:
      runner_name (normalised), finish_rank (int), runner_number (if present).
    Handles: placings[rank=1..], also_ran[finish_position], scratchings.
    """
    rows = []
    meetings = obj.get("meetings") or [obj]
    for m in meetings:
        races = m.get("races") or []
        for r in races:
            # Track scratchings (by runner number) to ignore DNFs reported as 0
            scratched = set()
            for s in (r.get("scratchings") or []):
                num = s.get("number") or s.get("runnerNumber")
                if num is not None:
                    try:
                        scratched.add(int(num))
                    except Exception:
                        pass

            # 1) Placings (usually top 3)
            for p in (r.get("placings") or []):
                name = p.get("name") or p.get("runnerName")
                if not name:
                    continue
                rn = _norm_name(name)
                num = p.get("number") or p.get("runnerNumber")
                try:
                    rank = int(p.get("rank") or p.get("placing") or p.get("position"))
                except Exception:
                    continue
                rows.append({
                    "runner_name": rn,
                    "finish_rank": rank,
                    "runner_number_res": num,
                })

            # 2) Also-rans (positions 4+ typically)
            for a in (r.get("also_ran") or []):
                name = a.get("name") or a.get("runnerName")
                if not name:
                    continue
                rn = _norm_name(name)
                num = a.get("number") or a.get("runnerNumber")
                try:
                    fin = int(
                        a.get("finish_position")
                        or a.get("placing")
                        or a.get("position")
                        or a.get("rank")
                        or 0
                    )
                except Exception:
                    fin = 0
                # ignore scratchings or zeros
                if fin and (num is None or int(num) not in scratched) and fin > 0:
                    rows.append({
                        "runner_name": rn,
                        "finish_rank": fin,
                        "runner_number_res": num,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # If duplicates, keep best (lowest) rank
    df = df.sort_values("finish_rank").drop_duplicates(
        subset=["runner_name"], keep="first"
    ).reset_index(drop=True)
    return df

def brier_win(probs: np.ndarray, winner_pos: int) -> float:
    y = np.zeros_like(probs, dtype=float)
    y[winner_pos] = 1.0
    return float(np.mean((probs - y) ** 2))

def log_loss_win(probs: np.ndarray, winner_pos: int, eps: float = 1e-12) -> float:
    p = float(np.clip(probs[winner_pos], eps, 1.0))
    return float(-np.log(p))

# -----------------------------
# Main
# -----------------------------

def main(meet_no: int, race_no: int, date_str: str):
    # 1) Predict from schedule (model+market blended, tau=0.4 inside helper)
    sched = fetch_schedule_json(date_str, meet_no, race_no)
    df_sched = schedule_json_to_df(sched)
    if df_sched.empty:
        print("No runners in schedule.")
        return

    preds = estimate_position_probs_for_race_df(df_sched)  # uses tau=0.4 and blend
    preds["runner_name"] = preds["runner_name"].map(_norm_name)

    # 2) Pull results and parse
    res = fetch_results_json(date_str, meet_no, race_no)
    df_res = results_to_df(res)
    if df_res.empty:
        print("No official results yet.")
        return

    # 3) Join on name
    df = preds.merge(df_res, on="runner_name", how="inner")
    if df.empty:
        print("No name matches between schedule and results.")
        return

    # Ensure numeric arrays for metrics
    finish_rank = df["finish_rank"].to_numpy()
    pred_rank = df["pred_rank"].to_numpy()
    win_prob = df["win_prob"].to_numpy()

    # 4) Metrics (use positional indices, not label indices)
    winner_pos = int(np.argmin(finish_rank))
    top_pick_pos = int(np.argmax(win_prob))
    winner_hit = int(winner_pos == top_pick_pos)

    top3_positions = np.argsort(-win_prob)[:3]
    top3_hit = int(winner_pos in top3_positions)

    from scipy.stats import spearmanr
    rho = float(spearmanr(finish_rank, pred_rank).correlation)

    brier = brier_win(win_prob, winner_pos)
    logloss = log_loss_win(win_prob, winner_pos)

    # 5) Print compact summary row
    top_pick_name = df.iloc[top_pick_pos]["runner_name"]
    winner_name = df.iloc[winner_pos]["runner_name"]
    top_pick_win_pct = round(float(df.iloc[top_pick_pos]["win_%"]), 2)

    summary = pd.DataFrame([{
        "date": date_str,
        "meet": meet_no,
        "race": race_no,
        "field": len(df),
        "winner_hit": winner_hit,
        "top3_hit": top3_hit,
        "spearman": round(rho, 3),
        "brier_win": round(brier, 4),
        "logloss_win": round(logloss, 4),
        "top_pick": top_pick_name,
        "top_pick_win_%": top_pick_win_pct,
        "winner": winner_name,
    }])

    print(summary.to_string(index=False))

    # 6) Nice tables
    show_cols = [
        "runner_number", "runner_name", "fav_rank",
        "finish_rank", "pred_rank", "win_%", "win_$fair", "top3_%", "top3_$fair"
    ]
    # The preds DF carries runner_number (from schedule); keep it if present
    for c in show_cols:
        if c not in df.columns:
            df[c] = np.nan

    print("\nActual order (by finish_rank):")
    print(df.sort_values("finish_rank")[show_cols].to_string(index=False))

    print("\nModel order (by win_prob desc):")
    print(df.sort_values("win_prob", ascending=False)[show_cols].to_string(index=False))

    # 7) Append to CSV log
    log_path = "eval_log.csv"
    header_needed = not os.path.exists(log_path)
    summary.to_csv(log_path, mode="a", header=header_needed, index=False)
    print(f"\nAppended to {log_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate_one_race.py <meet_no> <race_no> [date=YYYY-MM-DD]")
        sys.exit(1)
    meet = int(sys.argv[1])
    race = int(sys.argv[2])
    date = sys.argv[3] if len(sys.argv) > 3 else datetime.now().strftime("%Y-%m-%d")
    main(meet, race, date)
