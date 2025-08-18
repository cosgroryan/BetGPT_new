#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch TAB results for a given date and save to disk (raw JSON + flat CSV).
More robust flattener: supports shapes with separate runners/placings.

Usage:
  python fetch_tab_results.py                 # yesterday NZ
  python fetch_tab_results.py --date 2025-08-17
  python fetch_tab_results.py --date 2025-08-17 --meets 22 31 --debug
"""

import argparse, os, time, json, sys
from typing import Dict, Any, List, Iterable, Optional, Tuple, Union
import requests
import pandas as pd
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo
    TZ_AKL = ZoneInfo("Pacific/Auckland")
except Exception:
    import pytz
    TZ_AKL = pytz.timezone("Pacific/Auckland")

BASE_JSON = "https://json.tab.co.nz"
SCHEDULE_URL = f"{BASE_JSON}/schedule/{{date}}"
RESULTS_URL  = f"{BASE_JSON}/results/{{date}}"

HEADERS = {"Accept": "application/json", "User-Agent": "results-fetcher/1.1 (+betgpt)"}
TIMEOUT = 15

# ---------- helpers ----------
def nz_yesterday_str() -> str:
    now_akl = datetime.now(TZ_AKL)
    y = (now_akl - timedelta(days=1)).date()
    return y.isoformat()

def _get_json(url: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _norm(s: Any) -> str:
    return "" if s is None else str(s).strip()

# ----- flattening utilities -----
RunnerDict = Dict[str, Any]

# ----- replace helpers + flatten with the following -----
import unicodedata
# ---- drop-in replacement for placement helpers + flattener ----
from typing import Iterable, Dict, Any, List, Optional, Tuple
import unicodedata
import pandas as pd

PLAC_KEYS = ("rank","placing","position","finish","fin","finishPos","finish_position","finalPosition","place")
NUM_KEYS  = ("runner_number","number","runner","start","saddle","bib","runnerNumber","startNumber","horseNumber")
NAME_KEYS = ("runner_name","name","horse_name","horse","runner","horseName","runnerName")

def _norm_name_local(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return " ".join(str(s).strip().upper().split())

def _deep_lists_with_dicts(obj: Any) -> List[List[Dict[str, Any]]]:
    out: List[List[Dict[str, Any]]] = []
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        out.append(obj)  # type: ignore
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_deep_lists_with_dicts(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_deep_lists_with_dicts(v))
    return out

def _get_int_safe(d: Dict[str, Any], keys: Iterable[str]) -> Optional[int]:
    for k in keys:
        if k in d and d[k] is not None and str(d[k]).strip() != "":
            try:
                return int(str(d[k]).split()[0].replace("st","").replace("nd","").replace("rd","").replace("th",""))
            except Exception:
                try:
                    return int("".join(ch for ch in str(d[k]) if ch.isdigit()))
                except Exception:
                    pass
    return None

def _get_name_safe(d: Dict[str, Any]) -> str:
    for k in NAME_KEYS:
        if k in d and d[k]:
            if isinstance(d[k], dict) and "name" in d[k]:
                return str(d[k]["name"]).strip()
            return str(d[k]).strip()
    if isinstance(d.get("horse"), dict) and d["horse"].get("name"):
        return str(d["horse"]["name"]).strip()
    return ""

def _collect_runner_maps(payload: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return (num_map, name_map) from runners-like arrays."""
    num_map: Dict[int, Dict[str, Any]] = {}
    name_map: Dict[str, Dict[str, Any]] = {}

    # Prefer obvious "runners"; fall back to deep scan
    candidates: List[Iterable] = []
    for k in ("runners","Runners"):
        arr = payload.get(k)
        if isinstance(arr, list):
            candidates.append(arr)
    if not candidates:
        for arr in _deep_lists_with_dicts(payload):
            candidates.append(arr)

    for arr in candidates:
        for d in arr:
            if not isinstance(d, dict): 
                continue
            nm = _get_name_safe(d)
            num = _get_int_safe(d, NUM_KEYS)
            if nm:
                name_map.setdefault(_norm_name_local(nm), {"runner_name": nm, **d})
            if num is not None:
                base = {"runner_number": num}
                if nm:
                    base["runner_name"] = nm
                num_map.setdefault(num, {**base, **d})
    return num_map, name_map

def _collect_placings_and_also_ran(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Gather entries from both 'placings' (rank 1..3) and 'also_ran' (finish_position 4..N).
    Returns records with at least a place and (number or name).
    """
    out: List[Dict[str, Any]] = []

    # explicit top-3
    for k in ("placings","Placings"):
        arr = payload.get(k)
        if isinstance(arr, list):
            for d in arr:
                if not isinstance(d, dict): 
                    continue
                place = _get_int_safe(d, ("rank",))  # <<< key fix
                nm = _get_name_safe(d)
                num = _get_int_safe(d, NUM_KEYS)
                if place is not None and (num is not None or nm):
                    out.append({"_place": place, "_nm": nm, "_num": num, "_raw": d})

    # the rest of the field
    for k in ("also_ran","alsoRan","AlsoRan","alsoRanRunners"):
        arr = payload.get(k)
        if isinstance(arr, list):
            for d in arr:
                if not isinstance(d, dict): 
                    continue
                place = _get_int_safe(d, ("finish_position",))
                nm = _get_name_safe(d)
                num = _get_int_safe(d, NUM_KEYS)
                if place is not None and (num is not None or nm):
                    out.append({"_place": place, "_nm": nm, "_num": num, "_raw": d})

    # fallback: scan any list-of-dicts for PLAC_KEYS (covers other codes)
    if not out:
        for arr in _deep_lists_with_dicts(payload):
            for d in arr:
                if not isinstance(d, dict): 
                    continue
                place = _get_int_safe(d, PLAC_KEYS)
                nm = _get_name_safe(d)
                num = _get_int_safe(d, NUM_KEYS)
                if place is not None and (num is not None or nm):
                    out.append({"_place": place, "_nm": nm, "_num": num, "_raw": d})

    return out

def _dividend_maps(payload: Dict[str, Any]) -> Tuple[Dict[int,float], Dict[int,float]]:
    """Map runner_number -> dividend for WIN/PLC from 'pools'."""
    win_map: Dict[int,float] = {}
    plc_map: Dict[int,float] = {}
    pools = payload.get("pools") or []
    for p in pools:
        if not isinstance(p, dict):
            continue
        typ = str(p.get("type","")).upper()
        num = p.get("number")
        amt = p.get("amount")
        if num is None or amt is None:
            continue
        # simple case: single runner like "6"
        try:
            n = int(str(num).split(":")[0].split(",")[0])
            a = float(amt)
        except Exception:
            continue
        if typ == "WIN":
            win_map[n] = a
        elif typ in ("PLC","PLACE"):
            plc_map[n] = a
    return win_map, plc_map

def flatten_race_results(payload: Dict[str, Any],
                         date: str,
                         meet_no: int,
                         race_no: int,
                         debug: bool = False) -> List[Dict[str, Any]]:
    # Some responses are wrapped (meetings -> races). If so, drill to this race.
    if "meetings" in payload:
        try:
            meetings = payload["meetings"]
            for m in meetings:
                if int(m.get("number", 0)) != int(meet_no): 
                    continue
                for r in m.get("races", []):
                    if int(r.get("number", 0)) == int(race_no):
                        payload = r  # switch to the race node
                        raise StopIteration
        except StopIteration:
            pass

    num_map, name_map = _collect_runner_maps(payload)
    entries = _collect_placings_and_also_ran(payload)
    win_map, plc_map = _dividend_maps(payload)

    if debug:
        print(f"[debug] M{meet_no} R{race_no}: runners by num={len(num_map)} by name={len(name_map)} entries={len(entries)}")

    rows: List[Dict[str, Any]] = []
    seen = set()

    for e in entries:
        place = e["_place"]
        num = e["_num"]
        nm  = e["_nm"]
        src = e["_raw"]

        base = None
        if num is not None and num in num_map:
            base = num_map[num]
        elif nm:
            base = name_map.get(_norm_name_local(nm), {"runner_name": nm})

        if base is None:
            continue

        rn_num = base.get("runner_number") or num
        rn_nm  = base.get("runner_name") or nm or ""
        key = (rn_num, _norm_name_local(rn_nm), place)
        if key in seen:
            continue
        seen.add(key)

        payout_win = win_map.get(int(rn_num)) if rn_num is not None else None
        payout_plc = plc_map.get(int(rn_num)) if rn_num is not None else None
        # if not in maps, try direct extraction on the entry/base (other codes)
        if payout_win is None:
            payout_win = (src.get("win") if isinstance(src, dict) else None) or None
        if payout_plc is None:
            payout_plc = (src.get("place") if isinstance(src, dict) else None) or None

        rows.append({
            "date": date,
            "meeting_number": meet_no,
            "race_number": race_no,
            "runner_number": rn_num,
            "runner_name": rn_nm,
            "finish_rank": place,
            "payout_win": payout_win,
            "payout_plc": payout_plc,
        })

    if rows:
        df = pd.DataFrame(rows).drop_duplicates(
            subset=["meeting_number","race_number","runner_name","finish_rank"]
        ).sort_values(["meeting_number","race_number","finish_rank"])
        return df.to_dict(orient="records")
    return rows

# ---------- main fetch ----------
def main():
    ap = argparse.ArgumentParser(description="Fetch TAB results into raw JSON + flat CSV.")
    ap.add_argument("--date", default=nz_yesterday_str(), help="YYYY-MM-DD (default: yesterday NZ)")
    ap.add_argument("--meets", nargs="*", type=int, help="Optional meet numbers (e.g., 22 31)")
    ap.add_argument("--sleep", type=float, default=0.25, help="Delay between requests (s)")
    ap.add_argument("--debug", action="store_true", help="Print per-race debug counts")
    args = ap.parse_args()

    date = args.date
    meets_filter = set(args.meets) if args.meets else None

    sch_url = SCHEDULE_URL.format(date=date)
    sch = _get_json(sch_url)
    if sch is None:
        print(f"Failed to fetch schedule: {sch_url}", file=sys.stderr)
        sys.exit(2)

    # Discover meet/race pairs
    pairs: List[tuple] = []
    meetings = sch.get("meetings") or sch.get("Meetings") or sch.get("meeting") or []
    if not meetings and isinstance(sch, dict):
        for v in sch.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and ("meetno" in v[0] or "meeting_number" in v[0]):
                meetings = v
                break

    for m in meetings:
        if not isinstance(m, dict): continue
        meet_no = int(m.get("meetno") or m.get("meeting_number") or m.get("number") or 0)
        if meet_no <= 0: continue
        if meets_filter and meet_no not in meets_filter: continue
        races = m.get("races") or m.get("Races") or m.get("race") or []
        for r in races if isinstance(races, list) else []:
            if not isinstance(r, dict): continue
            race_no = int(r.get("raceno") or r.get("race_number") or r.get("number") or 0)
            if race_no > 0:
                pairs.append((meet_no, race_no))

    if not pairs:
        print("No (meet, race) pairs discovered from schedule; nothing to fetch.")
        sys.exit(0)

    base = os.path.join("data", "results", date)
    raw_dir = os.path.join(base, "raw")
    _ensure_dir(raw_dir)

    flat_rows: List[Dict[str, Any]] = []
    raw_saved = 0

    for meet_no, race_no in pairs:
        url = f"{RESULTS_URL.format(date=date)}/{meet_no}/{race_no}"
        payload = _get_json(url)
        if payload is None:
            continue

        out_path = os.path.join(raw_dir, f"meet{meet_no}_race{race_no}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            raw_saved += 1
        except Exception:
            pass

        rows = flatten_race_results(payload, date, meet_no, race_no, debug=args.debug)
        flat_rows.extend(rows)

        time.sleep(max(0.0, args.sleep))

    if flat_rows:
        df = pd.DataFrame(flat_rows)
        if "finish_rank" in df.columns:
            df = df[~df["finish_rank"].isna()].copy()
        csv_path = os.path.join(base, "results_flat.csv")
        _ensure_dir(base)
        df.to_csv(csv_path, index=False)
        print(f"Saved flat results: {csv_path}  (rows={len(df)}, raw_saved={raw_saved})")
    else:
        print(f"No result rows flattened. Raw JSON files saved: {raw_saved}. Try --debug to inspect shapes.")

if __name__ == "__main__":
    main()
