#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

RUNNERS_CSV  = r'F:\db_hold\runners_flat.csv'
SCHED_CSV    = r'F:\db_hold\schedule_flat.csv'
SCHED_PQ     = Path(r'F:\db_hold\_sched_min.parquet')      # tiny column subset
OUT_DIR      = Path(r'F:\db_hold\merged_parquet'); OUT_DIR.mkdir(parents=True, exist_ok=True)

JOIN_KEYS = ["race_id","runner_number"]
CHUNK = 200_000

# 1) Build a small Parquet with schedule columns we actually need
sched_cols_all = pd.read_csv(SCHED_CSV, nrows=0).columns.tolist()
sched_keep = [c for c in [
    "race_id","runner_number","entrant_jockey","entrant_barrier","entrant_weight",
    "meeting_id","meeting_name","meeting_number","meeting_country","meeting_venue",
    "race_class","race_length","race_name","race_number","race_stake","race_status","race_track","race_weather"
] if c in sched_cols_all]

first = True
for chunk in pd.read_csv(SCHED_CSV, usecols=sched_keep, chunksize=CHUNK, low_memory=False):
    chunk["race_id"] = chunk["race_id"].astype(str)
    chunk["runner_number"] = pd.to_numeric(chunk["runner_number"], errors="coerce").astype("Int64")
    chunk = chunk.drop_duplicates(subset=JOIN_KEYS, keep="first")
    if first:
        chunk.to_parquet(SCHED_PQ, index=False)
        first = False
    else:
        pd.concat([pd.read_parquet(SCHED_PQ), chunk], ignore_index=True)\
          .drop_duplicates(subset=JOIN_KEYS, keep="first").to_parquet(SCHED_PQ, index=False)

print(f"✅ Wrote minimal schedule parquet → {SCHED_PQ}")

# 2) Stream runners and join each chunk to schedule parquet
sched = pd.read_parquet(SCHED_PQ).set_index(JOIN_KEYS)
part, total = 0, 0
runners_cols = pd.read_csv(RUNNERS_CSV, nrows=0).columns.tolist()
usecols = list(set(runners_cols) | set(JOIN_KEYS))

for chunk in pd.read_csv(RUNNERS_CSV, usecols=usecols, chunksize=CHUNK, low_memory=False):
    chunk["race_id"] = chunk["race_id"].astype(str)
    chunk["runner_number"] = pd.to_numeric(chunk["runner_number"], errors="coerce").astype("Int64")
    merged = chunk.join(sched, on=JOIN_KEYS, how="left", rsuffix="_sched")
    part += 1; total += len(merged)
    out = OUT_DIR / f"part_{part:04d}.parquet"
    merged.to_parquet(out, index=False)
    print(f"🧩 wrote {len(merged):,} rows → {out.name}  (total {total:,})")

print(f"✅ Done. Parts in {OUT_DIR}")
