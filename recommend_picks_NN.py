#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recommend picks for one meet / multiple races using the new NN model outputs.
Also saves results to a dayslips/ text file.

Examples:
  python recommend_races.py 2 1 3 4
  python recommend_races.py 5 2 --date 2025-08-18
"""

import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd

# pickle compat for preprocess.pkl
import sys as _sys, pytorch_pre as _pp
_sys.modules['__main__'] = _pp

from infer_from_schedule_json import (
    fetch_schedule_json,
    schedule_json_to_df,
)
from pytorch_pre import load_model_and_predict, _scores_from_pred_rank, _pl_sample_order


def _percent(x):
    return 100.0 * np.asarray(x, dtype=float)

def _fair_odds(p, eps: float = 1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    return 1.0 / p


def _nn_win_probs_for_race(race_df: pd.DataFrame, tau: float = 1.0, n_samples: int = 2000, seed: int = 42):
    out = load_model_and_predict(race_df)
    p_win = np.asarray(out.get("p_win_softmax")) if isinstance(out, dict) else None
    if p_win is not None and p_win.shape[0] == len(race_df) and np.isfinite(p_win).any():
        return p_win / (p_win.sum() if p_win.sum() > 0 else 1.0)
    pred_rank = np.asarray(out.get("pred_rank")) if isinstance(out, dict) else np.asarray(out)
    scores = _scores_from_pred_rank(pred_rank, tau=tau)
    rng = np.random.default_rng(seed)
    n = len(scores)
    pos_counts = np.zeros(n, dtype=np.int32)
    for _ in range(int(n_samples)):
        order = _pl_sample_order(scores, rng)
        pos_counts[order[0]] += 1
    return pos_counts / float(n_samples)


def recommend_for_race(meet_no: int, race_no: int, date_str: str):
    sched = fetch_schedule_json(date_str, meet_no, race_no)
    df = schedule_json_to_df(sched)
    if df.empty:
        return f"[meet {meet_no} race {race_no}] No runners found.\n"

    p_win = _nn_win_probs_for_race(df)

    try:
        nn_out = load_model_and_predict(df)
        new_horse = np.asarray(nn_out.get("new_horse"), dtype=bool) if isinstance(nn_out, dict) else np.zeros(len(df), dtype=bool)
    except Exception:
        new_horse = np.zeros(len(df), dtype=bool)

    out = pd.DataFrame({
        "runner_number": df.get("runner_number", pd.Series([None]*len(df))),
        "runner_name": df.get("runner_name", pd.Series([None]*len(df))),
        "fav_rank": df.get("fav_rank", pd.Series([None]*len(df))),
        "p_win": p_win,
        "new_horse": new_horse,
    })
    out["win_%"] = _percent(out["p_win"])
    out["$fair_win"] = _fair_odds(out["p_win"])
    out_sorted = out.sort_values("p_win", ascending=False).reset_index(drop=True)

    top = out_sorted.iloc[0]
    title = f"{date_str} | Meet {meet_no} Race {race_no}"
    lines = []
    lines.append("=" * len(title))
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(
        f"Top pick: #{int(top['runner_number']) if pd.notna(top['runner_number']) else '?'} "
        f"{str(top['runner_name'])}  |  Model win {top['win_%']:.1f}%  |  Fair ${top['$fair_win']:.2f}"
        f"{'  |  NEW HORSE' if bool(top['new_horse']) else ''}"
    )
    lines.append("\nModel order:")
    show_cols = ["runner_number", "runner_name", "fav_rank", "win_%", "$fair_win", "new_horse"]
    out_print = out_sorted[show_cols].copy()
    out_print["win_%"] = out_print["win_%"].map(lambda x: f"{float(x):.1f}")
    out_print["$fair_win"] = out_print["$fair_win"].map(lambda x: f"{float(x):.2f}")
    lines.append(out_print.to_string(index=False))
    lines.append("")  # blank line
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Recommend picks using the NN model.")
    parser.add_argument("meet", type=int, help="Meet number (e.g., 2)")
    parser.add_argument("races", nargs="+", type=int, help="Race numbers (e.g., 1 3 4)")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date (YYYY-MM-DD). Defaults to today.")
    args = parser.parse_args()

    meet_no = int(args.meet)
    date_str = args.date
    races = [int(r) for r in args.races]

    all_output = []
    for rno in races:
        all_output.append(recommend_for_race(meet_no, rno, date_str))

    text_out = "\n".join(all_output)

    # print to console
    print(text_out)

    # save to file
    os.makedirs("dayslips", exist_ok=True)
    race_min, race_max = min(races), max(races)
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    filename = f"dayslips/meet{meet_no} {race_min}-{race_max} {timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text_out)
    print(f"\nSaved recommendations to {filename}")


if __name__ == "__main__":
    main()
