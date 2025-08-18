#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta, timezone
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- HTTP ----------------
TIMEOUT = 25

def make_session():
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def get_json(session: requests.Session, url: str) -> Optional[dict]:
    try:
        r = session.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

# ---------------- Try to reuse your helpers ----------------
try:
    # put benchmark_yesterday.py next to this script
    from benchmark_yesterday import is_gallops_meeting, fetch_results_race, actual_top3_from_results  # type: ignore
except Exception:
    def _text(o) -> str:
        return "" if o is None else str(o).strip().lower()

    def is_gallops_meeting(m: dict) -> bool:
        dom = str(m.get("domain") or "").strip().upper()
        evt = str(m.get("event_type") or "").strip().upper()
        if dom == "HORSE" and evt == "G":
            return True
        if dom in {"DOG", "GREYHOUND"} or evt in {"GR"}:
            return False
        if dom in {"HARNESS"} or evt in {"H"}:
            return False
        for k in ("section", "code", "category", "sport", "type", "discipline"):
            s = _text(m.get(k))
            if any(x in s for x in ("gallop", "thoroughbred", "tbred")):
                return True
            if any(x in s for x in ("dog", "grey", "greyhound", "harness", "trot", "pace")):
                return False
        return True

    def fetch_results_race(day: str, meet_no: int, race_no: int) -> Optional[dict]:
        import urllib.request, json as _json
        try:
            with urllib.request.urlopen(f"https://json.tab.co.nz/results/{day}/{meet_no}/{race_no}", timeout=20) as r:
                return _json.loads(r.read().decode("utf-8"))
        except Exception:
            return None

    def actual_top3_from_results(obj: dict) -> Tuple[List[int], Dict[int, int]]:
        finish_map: Dict[int, int] = {}
        top3: Dict[int, int] = {}
        meetings = obj.get("meetings") or [obj]
        for m in meetings:
            for r in m.get("races", []) or obj.get("races", []):
                for p in (r.get("placings") or []):
                    num = p.get("number") or p.get("runnerNumber")
                    rank = p.get("rank") or p.get("placing") or p.get("position")
                    try:
                        num, rank = int(num), int(rank)
                    except Exception:
                        continue
                    finish_map[num] = rank
                    if 1 <= rank <= 3:
                        top3[rank] = num
                for a in (r.get("also_ran") or []):
                    num = a.get("number") or a.get("runnerNumber")
                    fin = a.get("finish_position") or a.get("placing") or a.get("position") or a.get("rank")
                    try:
                        num, fin = int(num), int(fin)
                    except Exception:
                        continue
                    if fin > 0:
                        finish_map[num] = fin
        actual = [top3.get(1), top3.get(2), top3.get(3)]
        return [x for x in actual if x is not None], finish_map

# ---------------- Core pulls ----------------
SCHED = "https://json.tab.co.nz/schedule/{day}"
ODDS  = "https://json.tab.co.nz/odds/{day}/{meet}/{race}"

def extract_prices_from_odds(payload: dict, race_no: int) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}

    def _ins(entry: dict):
        try:
            num = int(entry.get("number") or entry.get("runner") or entry.get("runner_number"))
        except Exception:
            return
        rec = out.setdefault(num, {})
        if entry.get("ffwin") is not None: rec["win_fixed"]   = entry.get("ffwin")
        if entry.get("ffplc") is not None: rec["place_fixed"] = entry.get("ffplc")
        if entry.get("win")   is not None: rec["win_tote"]    = entry.get("win")
        if entry.get("plc")   is not None: rec["place_tote"]  = entry.get("plc")

    if not isinstance(payload, dict):
        return out
    meetings = payload.get("meetings") or []
    if meetings:
        for m in meetings:
            for r in m.get("races") or []:
                try:
                    rnum = int(r.get("number") or r.get("raceNumber"))
                except Exception:
                    continue
                if rnum == race_no:
                    for e in r.get("entries") or []:
                        _ins(e)
                    return out
        return out
    for r in payload.get("races") or []:
        try:
            rnum = int(r.get("number") or r.get("raceNumber"))
        except Exception:
            continue
        if rnum == race_no:
            for e in r.get("entries") or []:
                _ins(e)
            break
    return out

def iter_days(start: str, end: str):
    d0 = datetime.fromisoformat(start).date()
    d1 = datetime.fromisoformat(end).date()
    if d1 < d0:
        d0, d1 = d1, d0
    d = d0
    while d <= d1:
        yield d.isoformat()
        d += timedelta(days=1)

def safe_float(x):
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except Exception:
        return None

# ---------- logging ----------
def _ts():
    return datetime.now().strftime("%H:%M:%S")

def _log(msg: str, quiet: bool):
    if not quiet:
        print(f"[{_ts()}] {msg}", flush=True)

def pull_range(start: str, end: str, quiet: bool = False) -> pd.DataFrame:
    sess = make_session()
    rows = []

    _log(f"Starting pull from {start} to {end}", quiet)
    day_idx = 0
    total_meetings = total_races = total_rows = 0

    for day in iter_days(start, end):
        day_idx += 1
        _log(f"Day {day_idx}: fetching schedule for {day}", quiet)
        sched = get_json(sess, SCHED.format(day=day))
        if not sched or not sched.get("meetings"):
            _log(f"  No meetings for {day} or schedule fetch failed", quiet)
            continue

        meetings = sched["meetings"]
        _log(f"  {len(meetings)} meetings on schedule", quiet)

        gallops_meetings = 0
        for m in meetings:
            if not is_gallops_meeting(m):
                continue
            gallops_meetings += 1

            meet_no = m.get("number") or m.get("meetingNumber")
            venue   = m.get("venue") or m.get("meetingName")
            if meet_no is None:
                _log("  Skipping meeting with no number", quiet)
                continue
            try:
                meet_no = int(meet_no)
            except Exception:
                _log(f"  Skipping meeting with bad number {meet_no}", quiet)
                continue

            races = m.get("races") or []
            _log(f"  Meeting {meet_no} {venue}: {len(races)} races", quiet)
            total_meetings += 1

            for r in races:
                race_no   = r.get("number") or r.get("raceNumber")
                race_name = r.get("name") or r.get("raceName")
                if race_no is None:
                    _log("    Race with no number. Skipping", quiet)
                    continue
                try:
                    race_no = int(race_no)
                except Exception:
                    _log(f"    Bad race number {race_no}. Skipping", quiet)
                    continue

                total_races += 1
                _log(f"    Race {race_no}: odds fetch", quiet)
                odds = get_json(sess, ODDS.format(day=day, meet=meet_no, race=race_no)) or {}
                prices = extract_prices_from_odds(odds, race_no)
                _log(f"      odds ok. runners with prices {len(prices)}", quiet)

                _log(f"    Race {race_no}: results fetch", quiet)
                res = fetch_results_race(day, meet_no, race_no)
                if res is None:
                    _log("      results missing", quiet)
                top3, finish_map = actual_top3_from_results(res or {})
                _log(f"      results ok. finishers recorded {len(finish_map)}", quiet)

                dist      = r.get("distance") or r.get("raceDistance") or r.get("race_distance")
                track     = r.get("track_condition") or r.get("trackCondition")
                weather   = r.get("weather")
                start_ts  = r.get("advertised_start") or r.get("start_time") or r.get("tote_start_time")

                entries = r.get("runners") or r.get("entries") or []
                if not entries and isinstance(odds, dict):
                    if odds.get("meetings"):
                        for mm in odds["meetings"]:
                            for rr in (mm.get("races") or []):
                                rnum = safe_int(rr.get("number") or rr.get("raceNumber"))
                                if rnum == race_no:
                                    entries = rr.get("entries") or []
                                    break
                            if entries:
                                break
                    elif odds.get("races"):
                        for rr in (odds.get("races") or []):
                            rnum = safe_int(rr.get("number") or rr.get("raceNumber"))
                            if rnum == race_no:
                                entries = rr.get("entries") or []
                                break
                _log(f"    Race {race_no}: building rows from {len(entries)} entries", quiet)

                for e in entries:
                    num  = e.get("number") or e.get("runnerNumber") or e.get("runner")
                    name = e.get("name") or e.get("runnerName")
                    jockey = e.get("jockey") or e.get("rider")
                    barrier= e.get("barrier") or e.get("draw")
                    try:
                        num = int(num)
                    except Exception:
                        continue
                    p = prices.get(num, {})
                    rows.append({
                        "date": day,
                        "meeting_number": meet_no,
                        "meeting_venue": venue,
                        "race_number": race_no,
                        "race_name": race_name,
                        "distance": dist,
                        "track_condition": track,
                        "weather": weather,
                        "advertised_start": start_ts,
                        "runner_number": num,
                        "runner_name": name,
                        "barrier": barrier,
                        "jockey": jockey,
                        "win_fixed": safe_float(p.get("win_fixed")),
                        "place_fixed": safe_float(p.get("place_fixed")),
                        "win_tote": safe_float(p.get("win_tote")),
                        "place_tote": safe_float(p.get("place_tote")),
                        "finish_position": finish_map.get(num),
                    })
                total_rows = len(rows)

        _log(f"Day {day} complete. Gallops meetings {gallops_meetings}. Rows so far {total_rows}", quiet)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["race_number"] = pd.to_numeric(df["race_number"], errors="coerce").astype("Int64")
        df["runner_number"] = pd.to_numeric(df["runner_number"], errors="coerce").astype("Int64")

        def _norm_start(x):
            if x is None or x == "":
                return None
            try:
                if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()):
                    return datetime.fromtimestamp(int(x), tz=timezone.utc).isoformat()
                s = str(x)
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                if " " in s:
                    s = s.replace(" ", "T", 1)
                return datetime.fromisoformat(s).astimezone(timezone.utc).isoformat()
            except Exception:
                return None
        df["advertised_start_iso"] = df["advertised_start"].map(_norm_start)

    _log(f"Finished. Meetings {total_meetings}, races {total_races}, rows {len(df)}", quiet)
    return df

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Pull gallops schedule + odds + results into a parquet over a date range.")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--outdir", default=".", help="Output directory for parquet")
    ap.add_argument("--quiet", action="store_true", help="Silence console logging")
    args = ap.parse_args()

    quiet = bool(args.quiet)
    if not quiet:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Args: start={args.start} end={args.end} outdir={args.outdir}", flush=True)

    df = pull_range(args.start, args.end, quiet=quiet)
    out_name = f"gallops_{args.start}_to_{args.end}.parquet"
    out_path = os.path.join(args.outdir, out_name)
    if df.empty:
        if not quiet:
            print("[{}] No rows found for that range. Nothing written.".format(datetime.now().strftime('%H:%M:%S')), flush=True)
        return
    df.to_parquet(out_path, index=False)
    if not quiet:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] OK wrote {len(df)} rows to {out_path}", flush=True)

if __name__ == "__main__":
    main()
