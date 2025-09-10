#!/usr/bin/env python3
"""
Flatten race JSON into runner-level rows with streaming.

Supports:
- Single JSON with top-level "meetings"
- NDJSON (newline-delimited), where each line is a JSON object with "meetings"

Usage:
  python flatten_race_json_nd.py --input master.json --out runners_flat.csv
  python flatten_race_json_nd.py --input master.ndjson --out runners_flat.csv --ndjson
  python flatten_race_json_nd.py --input master.ndjson --out runners_flat.parquet --parquet

Requires:
  pip install ijson pandas pyarrow
"""

import argparse
import os
import sys
import json
from typing import Dict, Any, Iterable, List, Optional

try:
    import ijson
except Exception:
    ijson = None

try:
    import pandas as pd  # noqa: F401
except Exception as e:
    print("pandas is required. pip install pandas", file=sys.stderr)
    raise

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--parquet", action="store_true")
    p.add_argument("--ndjson", action="store_true", help="Treat input as newline-delimited JSON")
    p.add_argument("--chunk-size", type=int, default=100000)
    return p.parse_args()


def normalise_margin(margin: Optional[str]) -> Optional[float]:
    if margin is None:
        return None
    s = str(margin).strip().upper()
    if s in {"NSE", "NOSE"}:
        return 0.05
    if s in {"SH", "SHORT HALF HEAD"}:
        return 0.1
    if s in {"HD", "HEAD"}:
        return 0.2
    if s in {"NK", "NECK"}:
        return 0.3
    s = s.replace("L", "")
    try:
        if "-" in s:
            whole, frac = s.split("-", 1)
            whole_v = float(whole)
            frac_v = eval(frac) if "/" in frac else float(frac)
            return whole_v + float(frac_v)
        if "/" in s:
            return float(eval(s))
        return float(s)
    except Exception:
        return None


def parse_favouritism(fav: Optional[str]) -> Optional[float]:
    if not fav or not isinstance(fav, str):
        return None
    parts = fav.split("/")
    try:
        return float(parts[0])
    except Exception:
        return None


def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _from_pool(pmap: Dict[str, Dict[str, float]], key: str, rn: str) -> Optional[float]:
    try:
        return pmap.get(key, {}).get(rn)
    except Exception:
        return None


def pools_to_maps(pools: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for p in pools or []:
        ptype = p.get("type")
        amount = p.get("amount")
        if ptype is None or amount is None:
            continue
        nums = str(p.get("number") or "").split(",")
        if len(nums) == 1 and ":" in nums[0]:
            nums = nums[0].split(":")
        m = out.setdefault(ptype, {})
        for n in nums:
            n = n.strip()
            if not n:
                continue
            m[n] = max(m.get(n, float("-inf")), float(amount))
    return out


def flatten_one_day(obj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield runner rows for a single day's JSON object with 'meetings'."""
    # Some feeds may nest data under 'payload'
    day = obj
    if "payload" in obj and isinstance(obj["payload"], dict):
        day = obj["payload"]
    current_date = day.get("date")

    meetings = day.get("meetings") or []
    for meeting in meetings:
        meeting_id = meeting.get("id")
        meeting_number = meeting.get("number")
        for race in meeting.get("races") or []:
            race_id = race.get("id")
            race_number = race.get("number")
            race_class = race.get("class")
            race_distance = race.get("distance")
            race_name = race.get("name")
            stake = race.get("stake")
            status = race.get("status")
            scratchings = race.get("scratchings") or []
            scratched_numbers = {str(s.get("number")) for s in scratchings if s and s.get("number") is not None}

            pools_map = pools_to_maps(race.get("pools") or [])

            for p in race.get("placings") or []:
                runner_number = p.get("number")
                rn_str = str(runner_number) if runner_number is not None else ""
                yield {
                    "date": current_date,
                    "meeting_id": meeting_id,
                    "meeting_number": meeting_number,
                    "race_id": race_id,
                    "race_number": race_number,
                    "race_class": race_class,
                    "race_distance_m": _to_float(race_distance),
                    "race_name": race_name,
                    "stake": stake,
                    "status": status,
                    "runner_number": runner_number,
                    "runner_name": p.get("name"),
                    "finish_rank": p.get("rank"),
                    "margin_len": normalise_margin(p.get("margin")),
                    "fav_rank": parse_favouritism(p.get("favouritism")),
                    "is_scratched": rn_str in scratched_numbers,
                    "payout_win": _from_pool(pools_map, "WIN", rn_str),
                    "payout_plc": _from_pool(pools_map, "PLC", rn_str),
                    "payout_qla": _from_pool(pools_map, "QLA", rn_str),
                    "payout_tfa": _from_pool(pools_map, "TFA", rn_str),
                    "payout_ft4": _from_pool(pools_map, "FT4", rn_str),
                    "source_section": "placings",
                }

            for a in race.get("also_ran") or []:
                runner_number = a.get("number")
                rn_str = str(runner_number) if runner_number is not None else ""
                yield {
                    "date": current_date,
                    "meeting_id": meeting_id,
                    "meeting_number": meeting_number,
                    "race_id": race_id,
                    "race_number": race_number,
                    "race_class": race_class,
                    "race_distance_m": _to_float(race_distance),
                    "race_name": race_name,
                    "stake": stake,
                    "status": status,
                    "runner_number": runner_number,
                    "runner_name": a.get("name"),
                    "finish_rank": a.get("finish_position"),
                    "margin_len": _to_float(a.get("distance")),
                    "fav_rank": None,
                    "is_scratched": rn_str in scratched_numbers or a.get("finish_position") == 0,
                    "payout_win": _from_pool(pools_map, "WIN", rn_str),
                    "payout_plc": _from_pool(pools_map, "PLC", rn_str),
                    "payout_qla": _from_pool(pools_map, "QLA", rn_str),
                    "payout_tfa": _from_pool(pools_map, "TFA", rn_str),
                    "payout_ft4": _from_pool(pools_map, "FT4", rn_str),
                    "source_section": "also_ran",
                }


def write_csv_iter(out_path: str, rows_iter: Iterable[Dict[str, Any]]):
    import csv
    first = True
    cols = None
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = None
        for r in rows_iter:
            if first:
                cols = list(r.keys())
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                first = False
            writer.writerow(r)


def write_parquet_iter(out_path: str, rows_iter: Iterable[Dict[str, Any]]):
    if not HAVE_PARQUET:
        raise RuntimeError("pyarrow not available. pip install pyarrow")
    sink = None
    try:
        batch: List[Dict[str, Any]] = []
        for r in rows_iter:
            batch.append(r)
            if len(batch) >= 250_000:
                table = pa.Table.from_pylist(batch)
                if sink is None:
                    sink = pq.ParquetWriter(out_path, table.schema)
                sink.write_table(table)
                batch = []
        if batch:
            table = pa.Table.from_pylist(batch)
            if sink is None:
                sink = pq.ParquetWriter(out_path, table.schema)
            sink.write_table(table)
    finally:
        if sink is not None:
            sink.close()


def iter_single_json(path: str, chunk_size: int) -> Iterable[Dict[str, Any]]:
    # Stream meetings under top-level for a single big JSON
    if ijson is None:
        # Fallback to loading whole file (not ideal for 400MB)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        for row in flatten_one_day(obj):
            yield row
        return

    with open(path, "rb") as f:
        # read the whole document as one object and emit
        obj = None
        try:
            obj = next(ijson.items(f, ""))
        except Exception as e:
            raise
    for row in flatten_one_day(obj):
        yield row


def iter_ndjson(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                # give context for debugging
                msg = f"Bad NDJSON line at byte {f.tell()}: {e}"
                raise RuntimeError(msg) from e
            for row in flatten_one_day(obj):
                yield row


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.ndjson:
        rows_iter = iter_ndjson(args.input)
    else:
        rows_iter = iter_single_json(args.input, chunk_size=args.chunk_size)

    if args.parquet:
        write_parquet_iter(args.out, rows_iter)
    else:
        write_csv_iter(args.out, rows_iter)

    print(f"Done. Wrote {args.out}")


if __name__ == "__main__":
    main()
