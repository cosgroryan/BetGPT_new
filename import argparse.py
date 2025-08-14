import argparse
import datetime as dt
import json
import os
import sys
import time
from typing import Iterable, Optional

try:
    import requests
except Exception as e:
    print("Please install requests: pip install requests", file=sys.stderr)
    raise

DEFAULT_BASE_URL = "https://json.tab.co.nz/results"
DEFAULT_DAYS = 1825
DEFAULT_OUTFILE = "tab_results_master.ndjson"

def daterange(end_date: dt.date, days_back: int) -> Iterable[dt.date]:
    for i in range(days_back):
        yield end_date - dt.timedelta(days=i)

def load_existing_dates(path: str) -> set:
    dates = set()
    if not os.path.exists(path):
        return dates
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                d = rec.get("date")
                if d:
                    dates.add(d)
            except json.JSONDecodeError:
                # Skip any corrupt lines
                continue
    return dates

def fetch_results_for_date(base_url: str, d: dt.date, timeout: int = 60) -> Optional[dict]:
    url = f"{base_url}/{d.isoformat()}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "tab-results-scraper/1.0"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    if r.status_code == 404:
        # No results for this date
        return None
    r.raise_for_status()
    try:
        return r.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON for {d}: {e}")

def append_record(path: str, date_str: str, payload: dict) -> None:
    rec = {"date": date_str, "payload": payload}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False))
        f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Fetch TAB NZ results for the last N days and append to a master NDJSON file.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Number of past days to fetch, including today")
    parser.add_argument("--outfile", type=str, default=DEFAULT_OUTFILE, help="Path to master NDJSON file")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base URL for results endpoint")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests")
    args = parser.parse_args()

    end_date = dt.date.today()
    want_days = max(1, args.days)

    existing = load_existing_dates(args.outfile)
    print(f"Existing dates in master: {len(existing)}")

    wrote = 0
    skipped = 0
    errors = 0

    for d in daterange(end_date, want_days):
        ds = d.isoformat()
        if ds in existing:
            skipped += 1
            print(f"Skip {ds} already in file")
            continue
        try:
            payload = fetch_results_for_date(args.base_url, d)
            if payload is None:
                print(f"No results for {ds} (404)")
                continue
            append_record(args.outfile, ds, payload)
            wrote += 1
            print(f"Wrote {ds}")
        except requests.HTTPError as he:
            errors += 1
            print(f"HTTP error for {ds}: {he}", file=sys.stderr)
        except Exception as e:
            errors += 1
            print(f"Error for {ds}: {e}", file=sys.stderr)
        time.sleep(args.sleep)

    print(f"Done. Wrote {wrote}, skipped {skipped}, errors {errors}. Output: {args.outfile}")

if __name__ == "__main__":
    main()