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

# Use your existing inference helpers (blended probs/ranks)
from infer_from_schedule_json import (
    fetch_schedule_json,
    schedule_json_to_df,
    estimate_position_probs_for_race_df,  # returns blended win_prob/top3_prob/pred_rank
)

# New: pull NN-only probabilities directly
from pytorch_pre import load_model_and_predict

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

def _fair_odds(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return 1.0 / p

def _percent(p: np.ndarray) -> np.ndarray:
    return 100.0 * np.asarray(p, dtype=float)

# -----------------------------
# Main
# -----------------------------

def main(meet_no: int, race_no: int, date_str: str):
    # 1) Predict from schedule (blended model/market path)
    sched = fetch_schedule_json(date_str, meet_no, race_no)
    df_sched = schedule_json_to_df(sched)
    if df_sched.empty:
        print("No runners in schedule.")
        return

    # Keep a copy with normalised names for joins/printing
    df_sched["_runner_name_norm"] = df_sched["runner_name"].map(_norm_name)

    # Blend predictions (existing helper)
    preds_blend = estimate_position_probs_for_race_df(df_sched)  # includes: win_prob, top3_prob, pred_rank
    preds_blend["runner_name"] = preds_blend["runner_name"].map(_norm_name)
    preds_blend.rename(columns={
        "win_prob": "blend_win_prob",
        "top3_prob": "blend_top3_prob",
        "pred_rank": "blend_pred_rank",
    }, inplace=True)

    # 2) NN-only predictions (new path)
    nn_out = load_model_and_predict(df_sched)
    nn_win_prob = np.asarray(nn_out.get("p_win_softmax"))
    nn_plc_prob = np.asarray(nn_out.get("p_place_sigmoid"))
    nn_pred_rank = np.asarray(nn_out.get("pred_rank"))

    preds_nn = pd.DataFrame({
        "runner_name": df_sched["_runner_name_norm"],
        "nn_win_prob": nn_win_prob,
        "nn_top3_prob": np.minimum(1.0, np.asarray(nn_plc_prob, dtype=float)),  # leave as-is; you may sub a calibrated top3 if desired
        "nn_pred_rank": nn_pred_rank,
    })

    # 3) Merge blend + NN on normalised name
    preds = preds_blend.merge(preds_nn, on="runner_name", how="outer")
    # Add convenience percentages and fair odds for both
    for base in ("blend", "nn"):
        w = preds[f"{base}_win_prob"].to_numpy(dtype=float)
        t3 = preds[f"{base}_top3_prob"].to_numpy(dtype=float)
        preds[f"{base}_win_%"] = _percent(w)
        preds[f"{base}_win_$fair"] = _fair_odds(w)
        preds[f"{base}_top3_%"] = _percent(t3)
        preds[f"{base}_top3_$fair"] = _fair_odds(t3)

    # 4) Pull results and parse
    res = fetch_results_json(date_str, meet_no, race_no)
    df_res = results_to_df(res)
    if df_res.empty:
        print("No official results yet.")
        return

    # 5) Join on name
    df = preds.merge(df_res, on="runner_name", how="inner")
    if df.empty:
        print("No name matches between schedule and results.")
        return

    # Ensure numeric arrays for metrics (use NN by default)
    finish_rank = df["finish_rank"].to_numpy()
    nn_win_prob_arr = df["nn_win_prob"].to_numpy(dtype=float)
    blend_win_prob_arr = df["blend_win_prob"].to_numpy(dtype=float)
    nn_pred_rank_arr = df["nn_pred_rank"].to_numpy(dtype=float)
    blend_pred_rank_arr = df["blend_pred_rank"].to_numpy(dtype=float)

    # 6) Metrics (positional indices)
    winner_pos = int(np.argmin(finish_rank))

    # NN metrics
    nn_top_pick_pos = int(np.argmax(nn_win_prob_arr))
    nn_winner_hit = int(winner_pos == nn_top_pick_pos)
    nn_top3_positions = np.argsort(-nn_win_prob_arr)[:3]
    nn_top3_hit = int(winner_pos in nn_top3_positions)
    from scipy.stats import spearmanr
    nn_rho = float(spearmanr(finish_rank, nn_pred_rank_arr).correlation)
    nn_brier = brier_win(nn_win_prob_arr, winner_pos)
    nn_logloss = log_loss_win(nn_win_prob_arr, winner_pos)

    # Blend metrics (for comparison)
    bl_top_pick_pos = int(np.argmax(blend_win_prob_arr))
    bl_winner_hit = int(winner_pos == bl_top_pick_pos)
    bl_rho = float(spearmanr(finish_rank, blend_pred_rank_arr).correlation)
    bl_brier = brier_win(blend_win_prob_arr, winner_pos)
    bl_logloss = log_loss_win(blend_win_prob_arr, winner_pos)

    # 7) Print compact summary rows (NN first)
    nn_top_pick_name = df.iloc[nn_top_pick_pos]["runner_name"]
    bl_top_pick_name = df.iloc[bl_top_pick_pos]["runner_name"]
    winner_name = df.iloc[winner_pos]["runner_name"]

    nn_top_pick_win_pct = round(float(df.iloc[nn_top_pick_pos]["nn_win_%"]), 2)
    bl_top_pick_win_pct = round(float(df.iloc[bl_top_pick_pos]["blend_win_%"]), 2)

    summary = pd.DataFrame([
        {
            "date": date_str,
            "meet": meet_no,
            "race": race_no,
            "field": len(df),
            "winner": winner_name,
            # NN
            "nn_winner_hit": nn_winner_hit,
            "nn_spearman": round(nn_rho, 3),
            "nn_brier_win": round(nn_brier, 4),
            "nn_logloss_win": round(nn_logloss, 4),
            "nn_top_pick": nn_top_pick_name,
            "nn_top_pick_win_%": nn_top_pick_win_pct,
            # Blend
            "blend_winner_hit": bl_winner_hit,
            "blend_spearman": round(bl_rho, 3),
            "blend_brier_win": round(bl_brier, 4),
            "blend_logloss_win": round(bl_logloss, 4),
            "blend_top_pick": bl_top_pick_name,
            "blend_top_pick_win_%": bl_top_pick_win_pct,
        }
    ])
    print(summary.to_string(index=False))

    # 8) Nice tables
    show_cols = [
        "runner_number", "runner_name", "fav_rank",
        "finish_rank",
        # legacy/blend
        "blend_pred_rank", "blend_win_%", "blend_win_$fair", "blend_top3_%", "blend_top3_$fair",
        # new NN
        "nn_pred_rank", "nn_win_%", "nn_win_$fair", "nn_top3_%", "nn_top3_$fair",
    ]
    # Keep runner_number if present from schedule merge (not guaranteed post-merge)
    if "runner_number" not in df.columns and "runner_number" in df_sched.columns:
        df = df.merge(df_sched[["_runner_name_norm", "runner_number"]].rename(columns={"_runner_name_norm":"runner_name"}),
                      on="runner_name", how="left")

    for c in show_cols:
        if c not in df.columns:
            df[c] = np.nan

    print("\nActual order (by finish_rank):")
    print(df.sort_values("finish_rank")[show_cols].to_string(index=False))

    print("\nNN order (by nn_win_prob desc):")
    print(df.sort_values("nn_win_prob", ascending=False)[show_cols].to_string(index=False))

    print("\nBlended order (by blend_win_prob desc):")
    print(df.sort_values("blend_win_prob", ascending=False)[show_cols].to_string(index=False))

    # 9) Append to CSV log (NN-first summary)
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
