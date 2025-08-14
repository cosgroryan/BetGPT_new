#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

import numpy as np
import pandas as pd

# Local modules
import lgbm_rank as gbm
from infer_from_schedule_json import (
    estimate_position_probs_for_race_df as nn_estimate_for_race_df,
)

# -----------------------------
# Config
# -----------------------------
MAX_RACES_TO_PROBE = 30
GBM_RANK_MODEL_NAME = "model_lgbm.txt"
GBM_REG_MODEL_NAME  = "model_lgbm_reg.txt"
PL_TAU = 0.4
PL_NSAMPLES = 4000
BLEND_ALPHA = 0.6  # blend model with market if available

# -----------------------------
# Helpers
# -----------------------------
def today_nz_iso() -> str:
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo("Pacific/Auckland")).strftime("%Y-%m-%d")
    return datetime.now().strftime("%Y-%m-%d")

def parse_args(argv: List[str]) -> Tuple[List[int], str]:
    meets: List[int] = []
    date_str: Optional[str] = None

    if "--date" in argv:
        i = argv.index("--date")
        if i + 1 >= len(argv):
            print("Error: --date requires YYYY-MM-DD")
            sys.exit(1)
        date_str = argv[i + 1]
        argv = argv[:i] + argv[i+2:]

    date_pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    rest: List[str] = []
    for tok in argv:
        if date_str is None and date_pat.match(tok):
            date_str = tok
        else:
            rest.append(tok)

    for tok in rest:
        try:
            meets.append(int(tok))
        except ValueError:
            print(f"Warning: ignoring non-integer arg '{tok}'")

    if not meets:
        print("Usage: python dayslip.py <meet_no> [<meet_no> ...] [YYYY-MM-DD] [--date YYYY-MM-DD]")
        sys.exit(1)

    if date_str is None:
        date_str = today_nz_iso()

    return meets, date_str

def fetch_race_df(date_str: str, meet_no: int, race_no: int) -> pd.DataFrame:
    obj = gbm.fetch_schedule_json(date_str, meet_no, race_no)
    return gbm.schedule_json_to_df(obj)

def nn_top3_df(race_df: pd.DataFrame) -> pd.DataFrame:
    probs = nn_estimate_for_race_df(
        race_df, tau=PL_TAU, n_samples=PL_NSAMPLES, alpha_model_weight=BLEND_ALPHA
    )
    need = ["runner_name","runner_number","entrant_jockey","meeting_venue","race_number_sched","win_prob"]
    for c in need:
        if c not in probs.columns:
            probs[c] = None
    return probs[need].head(3).copy()

def gbm_top3_df(race_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    pred = gbm.load_gbm_and_predict(race_df, model_name=model_name)
    scores = np.asarray(pred["score"])
    pred_rank = np.asarray(pred["pred_rank"])

    win_model = gbm._scores_to_pl_win_probs(scores, tau=PL_TAU)
    if "implied_p" in race_df.columns and race_df["implied_p"].notna().any():
        win_blend = gbm.blend_with_market(win_model, race_df["implied_p"].to_numpy(), alpha=BLEND_ALPHA)
    else:
        win_blend = win_model

    out = pd.DataFrame({
        "runner_name": race_df.get("runner_name").values,
        "runner_number": race_df.get("runner_number").values,
        "entrant_jockey": race_df.get("entrant_jockey").values,
        "meeting_venue": race_df.get("meeting_venue").values if "meeting_venue" in race_df else "",
        "race_number_sched": race_df.get("race_number_sched").values if "race_number_sched" in race_df else race_df.get("race_number").values,
        "win_prob": win_blend,
        "pred_rank": pred_rank,
    }).sort_values(["win_prob"], ascending=[False]).reset_index(drop=True)
    return out.head(3)

def top3_tuples(df: pd.DataFrame) -> List[Tuple[str, Optional[int], str]]:
    out: List[Tuple[str, Optional[int], str]] = []
    if df is None or df.empty:
        return out
    for _, r in df.iterrows():
        out.append((str(r.get("runner_name") or ""), r.get("runner_number"), str(r.get("entrant_jockey") or "")))
    return out

def stars(name: str, membership_count: int) -> str:
    return name + ("*" * max(0, membership_count - 1))

def clean_num(n) -> str:
    if n is None:
        return "?"
    if isinstance(n, float) and math.isnan(n):
        return "?"
    try:
        return str(int(n))
    except Exception:
        return str(n)

# -----------------------------
# Main
# -----------------------------
def main():
    meets, date_str = parse_args(sys.argv[1:])

    # Per-model file buffers and a combined buffer
    nn_file: List[str]   = [f"NN DaySlip — {date_str}  (meets: {', '.join(map(str, meets))})"]
    rank_file: List[str] = [f"LGBM Rank DaySlip — {date_str}  (meets: {', '.join(map(str, meets))})"]
    reg_file: List[str]  = [f"LGBM Regression DaySlip — {date_str}  (meets: {', '.join(map(str, meets))})"]
    combined: List[str]  = [f"Combined DaySlip — {date_str}  (meets: {', '.join(map(str, meets))})"]

    # Stars-only file (**) with meet + race for placement
    starred_file: List[str] = [f"Starred (** only) — {date_str}  (meets: {', '.join(map(str, meets))})"]

    # Console header
    print("\n" + "="*72)
    print(f"DaySlips — {date_str}  (meets: {', '.join(map(str, meets))})")
    print("="*72)

    MEET_BREAK = "========="

    for idx_meet, meet_no in enumerate(meets):
        meet_header = f"\n=== Meet {meet_no} — {date_str} ==="
        print(meet_header)
        combined.append(meet_header)

        saw_any_race = False

        for race_no in range(1, MAX_RACES_TO_PROBE + 1):
            try:
                race_df = fetch_race_df(date_str, meet_no, race_no)
            except Exception:
                if saw_any_race:
                    break
                else:
                    continue

            if race_df is None or race_df.empty:
                if saw_any_race:
                    break
                else:
                    continue

            saw_any_race = True

            # Venue & race label
            venue = str(race_df.get("meeting_venue", pd.Series([""])).iloc[0] or
                        race_df.get("meeting_name", pd.Series([""])).iloc[0] or "")
            rnum  = int(pd.to_numeric(race_df.get("race_number", pd.Series([race_no])).iloc[0], errors="coerce") or race_no)
            race_hdr = f"{venue}  R{rnum}"
            print(f"\n{race_hdr}")
            combined.append(f"\n{race_hdr}")

            # Run models
            try:
                nn_df = nn_top3_df(race_df)
                nn_picks = top3_tuples(nn_df)
            except Exception as e:
                nn_picks = []
                print(f"  [NN] error: {e}")

            try:
                rnk_df = gbm_top3_df(race_df, GBM_RANK_MODEL_NAME)
                rnk_picks = top3_tuples(rnk_df)
            except Exception as e:
                rnk_picks = []
                print(f"  [LGBM Rank] error: {e}")

            try:
                reg_df = gbm_top3_df(race_df, GBM_REG_MODEL_NAME)
                reg_picks = top3_tuples(reg_df)
            except Exception as e:
                reg_picks = []
                print(f"  [LGBM Reg] error: {e}")

            # Agreement map
            sets = {
                "NN":  set(n.upper() for (n, _, __) in nn_picks),
                "RNK": set(n.upper() for (n, _, __) in rnk_picks),
                "REG": set(n.upper() for (n, _, __) in reg_picks),
            }
            def mark(picks: List[Tuple[str, Optional[int], str]]) -> List[Tuple[str, str]]:
                out: List[Tuple[str, str]] = []
                for (name, num, jky) in picks:
                    mcnt = int(name.upper() in sets["NN"]) + int(name.upper() in sets["RNK"]) + int(name.upper() in sets["REG"])
                    out.append((stars(name, mcnt), f"#{clean_num(num)} — {jky}"))
                return out

            nn_marked  = mark(nn_picks)
            rnk_marked = mark(rnk_picks)
            reg_marked = mark(reg_picks)

            # ** starred-only collector (dedup per race by upper name)
            starred_names_upper = {n.upper() for n in sets["NN"] & sets["RNK"] & sets["REG"]}
            # To fetch runner/jockey from whichever list contains it:
            def find_meta(name_upper: str) -> Tuple[str, str]:
                # search nn, then rank, then reg
                for lst in (nn_marked, rnk_marked, reg_marked):
                    for nm, meta in lst:
                        if nm.split('*')[0].upper() == name_upper:
                            return nm, meta
                return name_upper, "#? — ?"

            # append starred picks for this race
            for u in sorted(starred_names_upper):
                nm, meta = find_meta(u)
                starred_file.append(f"Meet {meet_no} | {venue}  R{rnum}: {nm} ({meta})")

            # Console pretty block (per race, grouped)
            def print_group(label: str, items: List[Tuple[str, str]]):
                if not items:
                    print(f"  {label}: (no picks)")
                else:
                    for idx, (nm, meta) in enumerate(items[:3], start=1):
                        prefix = "  " + (label if idx == 1 else " " * len(label))
                        print(f"{prefix}: {idx}) {nm} ({meta})")

            print_group("NN        ", nn_marked)
            print_group("LGBM-Rank ", rnk_marked)
            print_group("LGBM-Reg  ", reg_marked)

            # Files: per-model flat lines (keep SAME content, add blank line between races)
            def to_lines(items: List[Tuple[str, str]]) -> List[str]:
                if not items:
                    return [f"{race_hdr}: (no picks)"]
                return [f"{race_hdr}: {i}) {nm} ({meta})" for i, (nm, meta) in enumerate(items[:3], start=1)]

            nn_file.extend(to_lines(nn_marked))
            nn_file.append("")  # blank line between races

            rank_file.extend(to_lines(rnk_marked))
            rank_file.append("")

            reg_file.extend(to_lines(reg_marked))
            reg_file.append("")

            # Combined file (grouped like console)
            def combined_block(label: str, items: List[Tuple[str, str]]) -> List[str]:
                if not items:
                    return [f"  {label}: (no picks)"]
                rows = []
                for i, (nm, meta) in enumerate(items[:3], start=1):
                    prefix = "  " + (label if i == 1 else " " * len(label))
                    rows.append(f"{prefix}: {i}) {nm} ({meta})")
                return rows

            combined.extend(combined_block("NN        ", nn_marked))
            combined.extend(combined_block("LGBM-Rank ", rnk_marked))
            combined.extend(combined_block("LGBM-Reg  ", reg_marked))
            combined.append("")  # blank line between races in combined

        # Meet separator for the three per-model files (bigger break + blank line)
        if idx_meet < len(meets) - 1:
            for buf in (nn_file, rank_file, reg_file):
                buf.append(MEET_BREAK)
                buf.append("")

        import os

        # Ensure output folder exists
        out_dir = "dayslips"
        os.makedirs(out_dir, exist_ok=True)

        meets, date_str = parse_args(sys.argv[1:])

        # Update file paths to be inside dayslips/
        fn_nn   = os.path.join(out_dir, f"dayslip_nn_{date_str}.txt")
        fn_rank = os.path.join(out_dir, f"dayslip_lgbm_rank_{date_str}.txt")
        fn_reg  = os.path.join(out_dir, f"dayslip_lgbm_reg_{date_str}.txt")
        fn_all  = os.path.join(out_dir, f"dayslip_combined_{date_str}.txt")
        fn_star = os.path.join(out_dir, f"dayslip_starred_{date_str}.txt")

        with open(fn_nn, "w", encoding="utf-8") as f:
            f.write("\n".join(nn_file) + "\n")
        with open(fn_rank, "w", encoding="utf-8") as f:
            f.write("\n".join(rank_file) + "\n")
        with open(fn_reg, "w", encoding="utf-8") as f:
            f.write("\n".join(reg_file) + "\n")
        with open(fn_all, "w", encoding="utf-8") as f:
            f.write("\n".join(combined) + "\n")
        with open(fn_star, "w", encoding="utf-8") as f:
            f.write("\n".join(starred_file) + "\n")

        print("\nSaved:")
        print(f"  {fn_nn}")
        print(f"  {fn_rank}")
        print(f"  {fn_reg}")
        print(f"  {fn_all}")
        print(f"  {fn_star}")


if __name__ == "__main__":
    main()
