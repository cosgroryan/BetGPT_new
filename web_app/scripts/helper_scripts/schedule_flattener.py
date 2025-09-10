#!/usr/bin/env python3
import argparse
import json
import csv
from typing import Any, Dict, Iterable, List, Tuple

# ---------- IO ----------
def read_ndjson(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# ---------- flatten helpers ----------
def flatten(obj: Any, prefix: str = "", sep: str = "_") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}{sep}{k}" if prefix else str(k)
            out.update(flatten(v, p, sep=sep))
    elif isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix] = "|".join("" if x is None else str(x) for x in obj)
        else:
            out[prefix] = json.dumps(obj, ensure_ascii=False)
    else:
        out[prefix] = obj
    return out

# Candidate keys
NUM_CANDIDATES  = ["number", "no", "num", "barrier", "box", "draw", "saddle", "saddlecloth"]
NAME_CANDIDATES = ["name", "runner_name", "dog_name", "horse_name"]
JOCKEY_CANDIDATES = ["jockey", "driver", "rider"]

def pick_first(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None

# ---------- core ----------
def gather_rows_and_header(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    header: set = set()
    entrant_count = 0

    for rec in read_ndjson(path):
        date = rec.get("date")
        payload = rec.get("payload", {})
        meetings = payload.get("meetings", []) or []

        for m in meetings:
            meeting_flat = flatten(m, "meeting")
            meeting_track   = m.get("track")
            meeting_weather = m.get("weather")

            for r in (m.get("races", []) or []):
                r_copy = dict(r)
                r_copy.setdefault("track", meeting_track)
                r_copy.setdefault("weather", meeting_weather)
                race_flat = flatten(r_copy, "race")

                entrants = (
                    r_copy.get("entrants")
                    or r_copy.get("entries")
                    or r_copy.get("runners")
                    or []
                )
                if not isinstance(entrants, list):
                    entrants = []

                if not entrants:
                    base = {"date": date}
                    base.update(meeting_flat)
                    base.update(race_flat)
                    rows.append(base)
                    header.update(base.keys())
                    continue

                for e in entrants:
                    entrant_flat = flatten(e, "entrant")
                    e_num  = pick_first(e, NUM_CANDIDATES)
                    e_name = pick_first(e, NAME_CANDIDATES)
                    e_jky  = pick_first(e, JOCKEY_CANDIDATES)

                    row = {"date": date}
                    row.update(meeting_flat)
                    row.update(race_flat)
                    row.update(entrant_flat)

                    if "race_id" in row:
                        row["race_id"] = str(row["race_id"])
                    row["entrant_number"] = e_num
                    row["entrant_name"]   = e_name
                    if e_jky is not None:
                        row["entrant_jockey"] = e_jky

                    row["runner_number"] = e_num
                    row["runner_name"]   = e_name

                    rows.append(row)
                    header.update(row.keys())

                    entrant_count += 1
                    if entrant_count % 1000 == 0:
                        print(f"Parsed {entrant_count:,} entrants...")

    meeting_cols = sorted([c for c in header if c.startswith("meeting_")])
    race_cols    = sorted([c for c in header if c.startswith("race_")])
    entrant_cols = sorted([c for c in header if c.startswith("entrant_")])
    alias_cols   = [c for c in ["runner_number","runner_name"] if c in header]
    other_cols   = sorted([c for c in header if c not in meeting_cols + race_cols + entrant_cols + alias_cols + ["date"]])

    fieldnames = ["date"] + meeting_cols + race_cols + entrant_cols + alias_cols + other_cols
    print(f"Finished parsing {entrant_count:,} entrants.")
    return rows, fieldnames

def write_csv(out_path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        written_count = 0
        for row in rows:
            for k in fieldnames:
                row.setdefault(k, "")
            w.writerow(row)
            written_count += 1
            if written_count % 10000 == 0:
                print(f"Wrote {written_count:,} rows...")
    print(f"Finished writing {written_count:,} rows.")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Flatten TAB schedule NDJSON to entrant-level CSV (with progress indicators).")
    ap.add_argument("--input", required=True, help="Path to NDJSON (one JSON record per line).")
    ap.add_argument("--out",   required=True, help="Output CSV path (entrant-level).")
    args = ap.parse_args()

    rows, fieldnames = gather_rows_and_header(args.input)
    write_csv(args.out, rows, fieldnames)
    print(f"✅ Wrote {len(rows)} rows with {len(fieldnames)} columns to {args.out}")

if __name__ == "__main__":
    main()
