#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update dataset with new TAB results, then retrain models.

Changes in this version:
- PyTorch training now calls the pytorch_pre.py CLI via subprocess instead of
  expecting a `run_training` symbol. This makes it resilient to repo differences.
- GBM (LambdaRank + regression) paths are unchanged and still use lgbm_rank.py.

Usage:
  python update_data_retrain_model.py
  python update_data_retrain_model.py --retrain
  python update_data_retrain_model.py --start 2025-08-01 --end 2025-08-14 --retrain

Common flags:
  --data five_year_dataset.parquet
  --retrain                       # force retrain even if no new rows
  --skip-gbm                      # skip GBM retrain
  --skip-nn                       # skip NN retrain

PyTorch params (forwarded to pytorch_pre.py):
  --nn-epochs 20 --nn-batch 2048 --nn-lr 1e-3 --nn-wd 1e-4 --nn-dropout 0.25
  --nn-hidden "256,128,64"
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------
# Config
# ---------------------------
# Use data/parquet/five_year_dataset.parquet if it exists, otherwise fall back to current directory
if os.path.exists("data/parquet/five_year_dataset.parquet"):
    DATA_PATH_DEFAULT = "data/parquet/five_year_dataset.parquet"
else:
    DATA_PATH_DEFAULT = "five_year_dataset.parquet"

TAB_RESULTS_BASE = "https://json.tab.co.nz/results"  # results/{YYYY-MM-DD}/{meet}/{race}

BACKUP_DIR = "backups"
os.makedirs(BACKUP_DIR, exist_ok=True)

# For GBM retrain we import the module so we can call train/train_reg directly
try:
    import lgbm_rank as _lgbm
except Exception:
    _lgbm = None  # We’ll message if GBM is requested but module is missing


# ---------------------------
# Helpers
# ---------------------------

def _log(msg: str):
    print(msg, flush=True)

def load_parquet(path: str) -> pd.DataFrame:
    _log(f"[LOAD] {path}")
    return pd.read_parquet(path)

def _to_date(s) -> Optional[datetime.date]:
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None

def _daterange(start_date, end_date):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur += timedelta(days=1)

def _safe_request_json(url: str):
    import urllib.request, json as _json
    with urllib.request.urlopen(url, timeout=25) as resp:
        data = resp.read()
    return _json.loads(data.decode("utf-8"))

def _extract_rows_from_results_payload(obj: dict) -> pd.DataFrame:
    """
    Flatten results JSON into rows of runners with finish info.
    We only need a subset of columns; extra columns are harmless.
    """
    rows = []
    date = obj.get("date") or obj.get("day") or obj.get("meetingDate")
    meetings = obj.get("meetings") or []
    for m in meetings:
        meeting_id = m.get("id") or m.get("meetingId")
        meeting_number = m.get("number") or m.get("meetingNumber")
        meeting_country = m.get("country")
        meeting_venue = m.get("venue") or m.get("meetingName")

        for r in (m.get("races") or []):
            race_id = r.get("id") or r.get("raceId")
            race_number = r.get("number") or r.get("raceNumber")
            race_class = r.get("class")
            race_track = r.get("track") or r.get("trackCondition")
            race_weather = r.get("weather")
            race_name = r.get("name") or r.get("raceName")
            length_m = r.get("length") or r.get("distanceMeters")
            try:
                race_distance_m = float(length_m) if length_m not in (None, "") else np.nan
            except Exception:
                race_distance_m = np.nan

            # placings can be in explicit list or embedded per-runner
            placings_map = {}
            for p in (r.get("placings") or []):
                try:
                    num = int(p.get("number"))
                    rank = int(p.get("rank"))
                    placings_map[num] = rank
                except Exception:
                    pass

            entries = r.get("entries") or r.get("runners") or []
            for e in entries:
                scr = e.get("scr") or e.get("scratched") or e.get("isScratched")
                if scr:
                    continue
                try:
                    runner_no = int(e.get("number") or e.get("runnerNumber") or e.get("runner_number"))
                except Exception:
                    runner_no = None
                # rank: prefer placings map; else e.get("placing"/"position")
                rank = placings_map.get(runner_no)
                if rank is None:
                    for k in ("placing", "position", "finish"):
                        v = e.get(k)
                        try:
                            if v is not None:
                                rank = int(v)
                                break
                        except Exception:
                            pass

                rows.append({
                    "date": date,
                    "meeting_id": meeting_id,
                    "meeting_number": meeting_number,
                    "meeting_country": meeting_country,
                    "meeting_venue": meeting_venue,
                    "race_id": race_id,
                    "race_name": race_name,
                    "race_number": race_number,
                    "race_class": race_class,
                    "race_track": race_track,
                    "race_weather": race_weather,
                    "race_distance_m": race_distance_m,
                    "runner_number": runner_no,
                    "runner_name": (e.get("name") or e.get("runnerName") or "").strip(),
                    "finish_rank": (rank if rank is not None else np.nan),
                    "is_scratched": False,
                    # a few extra model-side fields if you use them
                    "fav_rank": np.nan,
                    "stake": np.nan,
                    "race_length": race_distance_m,
                    "race_number_sched": race_number,
                    "entrant_weight": np.nan,
                    "race_class_sched": race_class,
                    "entrant_barrier": str(e.get("barrier") or e.get("draw") or "") or "UNK",
                    "entrant_jockey": e.get("jockey") or e.get("jockeyName") or "UNK",
                    "source_section": "results",
                })
    return pd.DataFrame(rows)

def _to_bool_mask_any(s: pd.Series) -> pd.Series:
    """Coerce any 'is_scratched' flavour to clean booleans."""
    if s.dtype == bool or pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    # strings/object: true/1/yes/y => True
    return s.astype(str).str.strip().str.lower().isin({"true","t","1","yes","y"})


def fetch_missing_dates_and_append(df: pd.DataFrame,
                                   start_date: Optional[str],
                                   end_date: Optional[str]) -> Tuple[pd.DataFrame, int]:
    # Determine covered dates
    have_dates = pd.to_datetime(df["date"], errors="coerce").dt.date.dropna().unique()
    if len(have_dates) == 0:
        min_have = None
        max_have = None
    else:
        min_have = min(have_dates)
        max_have = max(have_dates)

    # Work out target window
    today_nz = datetime.now().date()
    default_start = (max_have + timedelta(days=1)) if max_have else (today_nz - timedelta(days=7))
    default_end = today_nz - timedelta(days=1)

    s = _to_date(start_date) or default_start
    e = _to_date(end_date) or default_end

    if s is None or e is None or s > e:
        _log(f"[SYNC] No missing dates. (start {s} > end {e})")
        return df, 0

    # Skip if our latest in data is already beyond e
    if max_have and max_have >= e:
        _log(f"[SYNC] No missing dates. (start {s} > end {e})")
        return df, 0

    _log(f"[FETCH] Missing dates: {s} … {e}")

    new_frames = []
    total_new = 0
    for d in _daterange(s, e):
        date_str = d.strftime("%Y-%m-%d")
        try:
            url = f"{TAB_RESULTS_BASE}/{date_str}"
            obj = _safe_request_json(url)
            df_day = _extract_rows_from_results_payload(obj)
            if not df_day.empty:
                _log(f"  - {date_str} … {len(df_day)} rows")
                new_frames.append(df_day)
                total_new += len(df_day)
        except Exception as ex:
            _log(f"  ! {date_str} fetch failed: {ex}")

    if not new_frames:
        _log("[APPEND] No new rows to append.")
        return df, 0

    # Backup original
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bkp = os.path.join(BACKUP_DIR, f"five_year_dataset_{ts}.parquet")
    shutil.copy2(DATA_PATH_DEFAULT, bkp)
    _log(f"[BACKUP] Saved original to {bkp}")

    # Append + save
    df_new = pd.concat([df] + new_frames, ignore_index=True)
    df_new.to_parquet(DATA_PATH_DEFAULT, index=False)
    _log(f"[SAVE] Appended {total_new} rows. New total: {len(df_new)}")

    return df_new, total_new


# ---------------------------
# Training wrappers
# ---------------------------

def train_gbm():
    if _lgbm is None:
        _log("[GBM] lgbm_rank.py not importable; skipping GBM.")
        return
    _log("\n[TRAIN] LightGBM LambdaRank …")
    _lgbm.train()
    _log("\n[TRAIN] LightGBM Regression …")
    _lgbm.train_reg()

def train_pytorch_via_cli(
    data_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    hidden: str,
):
    """
    Call pytorch_pre.py as a script so we don't rely on a specific exported API.
    Equivalent to:
      python pytorch_pre.py --data ... --epochs ... --batch_size ... --lr ... --weight_decay ... --dropout ... --hidden ...
    """
    _log("\n[TRAIN] PyTorch NN …")
    # Find the script next to this file or in PYTHONPATH
    script_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytorch_pre.py"),
        "pytorch_pre.py",
    ]
    script = None
    for c in script_candidates:
        if os.path.exists(c):
            script = c
            break
    if script is None:
        _log("[NN] Could not find pytorch_pre.py; skipping NN retrain.")
        return

    cmd = [
        sys.executable, script,
        "--data", data_path,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--dropout", str(dropout),
        "--hidden", hidden,
    ]
    _log(f"[NN] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        _log(f"[NN] Training failed (exit {e.returncode}). See output above.")


# ---------------------------
# CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA_PATH_DEFAULT)

    # Optional explicit window; default is: (max_date_in_data+1) .. (today-1)
    p.add_argument("--start", help="Start date YYYY-MM-DD (results)", default=None)
    p.add_argument("--end", help="End date YYYY-MM-DD (results)", default=None)

    # Control retrain
    p.add_argument("--retrain", action="store_true", help="Retrain even if no new rows")
    p.add_argument("--skip-gbm", action="store_true")
    p.add_argument("--skip-nn", action="store_true")

    # PyTorch hyperparams (forwarded to pytorch_pre.py)
    p.add_argument("--nn-epochs", type=int, default=20)
    p.add_argument("--nn-batch", type=int, default=2048)
    p.add_argument("--nn-lr", type=float, default=1e-3)
    p.add_argument("--nn-wd", type=float, default=1e-4)
    p.add_argument("--nn-dropout", type=float, default=0.25)
    p.add_argument("--nn-hidden", type=str, default="256,128,64")

    args = p.parse_args()

    # Load
    if not os.path.exists(args.data):
        sys.exit(f"Dataset not found: {args.data}")
    df = load_parquet(args.data)

    # Sync new results
    df, added = fetch_missing_dates_and_append(df, args.start, args.end)
    if added == 0 and not args.retrain:
        _log("\n[SKIP] No new data and --retrain not set. Skipping retrain.")
        return

    data_path = args.data

    # Retrain GBM
    if not args.skip_gbm:
        train_gbm()
    else:
        _log("[SKIP] GBM retrain skipped by flag.")

    # Retrain NN (PyTorch) via CLI wrapper
    if not args.skip_nn:
        train_pytorch_via_cli(
            data_path=data_path,
            epochs=args.nn_epochs,
            batch_size=args.nn_batch,
            lr=args.nn_lr,
            weight_decay=args.nn_wd,
            dropout=args.nn_dropout,
            hidden=args.nn_hidden,
        )
    else:
        _log("[SKIP] NN retrain skipped by flag.")

if __name__ == "__main__":
    main()
