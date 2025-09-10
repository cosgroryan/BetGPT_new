#!/usr/bin/env python3
# save_race_csv.py  —  odds + form + speedmap to CSV
import argparse, csv, re, sys
from datetime import datetime, timezone
from math import isfinite
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===== Config (mirrors your GUI) =====
NZ = ZoneInfo("Pacific/Auckland")

AFF_BASE  = "https://api.tab.co.nz/affiliates/v1/racing"
ODDS_BASE = "https://json.tab.co.nz/odds"
TIMEOUT   = 20

HEADERS = {
    "From": "r.cosgrove@hotmail.com",
    "X-Partner": "Personal use",
    "X-Partner-ID": "Personal use",
    "Accept": "application/json",
    "User-Agent": "RyanCosgrove/1.0",
}
# =====================================

def make_session():
    s = requests.Session()
    retry = Retry(
        total=3, backoff_factor=0.6,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["GET"], raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def today_nz():
    return datetime.now(NZ).strftime("%Y-%m-%d")

def safe_int(x, d=None):
    try: return int(x)
    except: return d

def safe_float(x, d=None):
    try:
        f = float(x)
        return f if isfinite(f) else d
    except:
        return d

def unwrap_meetings(obj):
    if isinstance(obj, dict):
        if "meetings" in obj: return obj["meetings"]
        d = obj.get("data")
        if isinstance(d, dict) and "meetings" in d: return d["meetings"]
    return []

def meeting_id(m):
    return m.get("id") or m.get("meeting_id") or m.get("uuid") or m.get("meeting")

def race_number(r):
    return r.get("race_number") or r.get("number")

def fmt_dt_nz(dt_utc):
    return dt_utc.astimezone(NZ).strftime("%Y-%m-%d %H:%M:%S") if dt_utc else ""

def parse_start_to_utc(race_obj):
    for k in ("advertised_start","start_time","start","scheduled_start"):
        v = race_obj.get(k)
        if isinstance(v, (int,float)) and v > 0:
            try: return datetime.fromtimestamp(float(v), tz=timezone.utc)
            except: pass
        if isinstance(v, str) and v:
            try: return datetime.fromisoformat(v.replace("Z","+00:00")).astimezone(timezone.utc)
            except: pass
    return None

# ---------- TAB odds ----------
def fetch_tab_odds(session, date_str, meetno, raceno):
    url = f"{ODDS_BASE}/{date_str}/{meetno}/{raceno}"
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()

    # Legacy markets/selections shape?
    if isinstance(data, dict) and data.get("markets"):
        return data["markets"]

    # Newer entries shape → convert to pseudo-markets
    try:
        entries = (data.get("meetings") or [])[0].get("races", [])[0].get("entries", [])
    except Exception:
        entries = []

    win_sel, plc_sel = [], []
    for e in entries:
        if e.get("scr") is True: 
            continue
        num   = safe_int(e.get("num"))
        name  = e.get("name") or ""
        ffwin = safe_float(e.get("ffwin"))
        ffplc = safe_float(e.get("ffplc"))
        win   = safe_float(e.get("win"))
        plc   = safe_float(e.get("plc"))
        flucs = e.get("flucs") if isinstance(e.get("flucs"), list) else []

        win_sel.append({"runner_number": num, "name": name, "fixed_price": ffwin, "tote_price": win, "flucs": flucs})
        plc_sel.append({"runner_number": num, "name": name, "fixed_price": ffplc, "tote_price": plc})

    return [{"name":"WIN","selections":win_sel},{"name":"PLACE","selections":plc_sel}]

def extract_prices_from_markets(markets):
    out = {}
    def ensure(n):
        if n not in out:
            out[n] = {"win_fixed":None,"place_fixed":None,"win_tote":None,"place_tote":None,"flucs":[]}
        return out[n]

    for m in markets or []:
        mname = (m.get("name") or "").upper()
        sels = m.get("selections") or []
        for s in sels:
            n = safe_int(s.get("runner_number") or s.get("number"))
            if n is None: 
                continue
            node = ensure(n)
            fx = safe_float(s.get("fixed_price"))
            tt = safe_float(s.get("tote_price"))
            if "WIN" in mname:
                if fx is not None: node["win_fixed"] = fx
                if tt is not None: node["win_tote"]  = tt
                if isinstance(s.get("flucs"), list): node["flucs"] = s["flucs"]
            elif "PLACE" in mname:
                if fx is not None: node["place_fixed"] = fx
                if tt is not None: node["place_tote"]  = tt
    return out

def flucs_disp(flucs):
    if not isinstance(flucs, list) or not flucs: return ""
    parts = []
    for x in flucs:
        fx = safe_float(x)
        if fx is not None: parts.append(f"{fx:.1f}")
    return "→".join(parts)

# ---------- Affiliates (schedule/form/speedmap) ----------
def find_meeting_id_for_meetno(session, date_str, meetno):
    url = f"{AFF_BASE}/meetings"
    params = {"date_from": date_str, "date_to": date_str, "enc": "json"}
    r = session.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    mtgs = unwrap_meetings(r.json()) or []
    chosen = None
    for m in mtgs:
        m_no = m.get("tote_meeting_number") or m.get("number") or m.get("meet_no")
        if safe_int(m_no) == safe_int(meetno):
            chosen = m; break
    if not chosen:
        return None, None
    mid = meeting_id(chosen)
    mname = chosen.get("name") or chosen.get("display_meeting_name") or chosen.get("venue") or ""
    return mid, mname

def fetch_meeting_detail(session, meeting_uuid):
    url = f"{AFF_BASE}/meetings/{meeting_uuid}"
    r = session.get(url, headers=HEADERS, params={"enc":"json"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def find_race_id(detail_payload, raceno):
    mtgs = unwrap_meetings(detail_payload)
    if not mtgs or not mtgs[0].get("races"): return None, None
    for rc in mtgs[0]["races"]:
        if safe_int(race_number(rc)) == safe_int(raceno):
            return rc.get("id"), rc
    return None, None

def fetch_event(session, race_id):
    url = f"{AFF_BASE}/events/{race_id}"
    r = session.get(url, headers=HEADERS, params={"enc":"json"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def extract_runners_from_event(payload):
    data = payload.get("data") or payload
    base = data.get("runners")
    if not base:
        mtgs = unwrap_meetings(payload)
        if mtgs:
            for race in mtgs[0].get("races", []):
                if race.get("runners"):
                    base = race["runners"]; break
    runners = []
    for r in base or []:
        num = safe_int(r.get("runner_number") or r.get("number"))
        bar = safe_int(r.get("barrier"))
        jockey = r.get("jockey") or ""
        w = r.get("weight") or {}
        weight = w.get("allocated") if isinstance(w, dict) else (str(w) if w else "")
        runners.append({
            "number": num,
            "name": r.get("name") or "",
            "barrier": bar,
            "jockey": jockey,
            "weight": weight,
            "form": r.get("last_twenty_starts") or "",
            "form_indicators": r.get("form_indicators") or [],
            "speedmap": r.get("speedmap") or {},  # expect {"label": "Frontrunner"} etc
            "win_fixed": None, "place_fixed": None, "win_tote": None, "place_tote": None,
            "flucs": r.get("flucs") or [], "flucs_disp": "",
        })
    return runners

def merge_prices_into_runners(runners, prices_by_num):
    for r in runners:
        n = r.get("number")
        p = prices_by_num.get(n, {}) if prices_by_num else {}
        r["win_fixed"]  = p.get("win_fixed")
        r["place_fixed"]= p.get("place_fixed")
        r["win_tote"]   = p.get("win_tote")
        r["place_tote"] = p.get("place_tote")
        if p.get("flucs"): r["flucs"] = p["flucs"]
        r["flucs_disp"] = flucs_disp(r.get("flucs"))

def speedmap_label(sm):
    return (sm.get("label") or "").strip() if isinstance(sm, dict) else ""

def sanitize_filename(s):
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "", s)

def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser(description="Save odds + form + speedmap to CSV")
    ap.add_argument("meet", type=int, help="Meet number")
    ap.add_argument("race", type=int, help="Race number")
    ap.add_argument("--date", default=today_nz(), help="YYYY-MM-DD (defaults to today NZ)")
    ap.add_argument("--out", default="", help="Optional explicit output path")
    args = ap.parse_args()

    s = make_session()

    # 1) Meeting UUID
    mid, meeting_name = find_meeting_id_for_meetno(s, args.date, args.meet)
    if not mid:
        print(f"Could not find meeting for date={args.date} meet={args.meet}", file=sys.stderr)
        sys.exit(2)

    # 2) Race id
    detail = fetch_meeting_detail(s, mid)
    race_id, race_node = find_race_id(detail, args.race)
    if not race_id:
        print(f"Could not find race #{args.race} in meeting {mid}", file=sys.stderr)
        sys.exit(3)

    # 3) Event → runners with form/speedmap
    event_payload = fetch_event(s, race_id)
    runners = extract_runners_from_event(event_payload)

    # 4) TAB odds merge
    markets = fetch_tab_odds(s, args.date, args.meet, args.race)
    prices  = extract_prices_from_markets(markets)
    merge_prices_into_runners(runners, prices)

    # 5) Build rows
    start_nz = fmt_dt_nz(parse_start_to_utc(race_node)) if race_node else ""
    rows = []
    for r in sorted(runners, key=lambda x: (x.get("number") or 999)):
        rows.append({
            "date_nz": args.date,
            "meeting_name": meeting_name,
            "meet_no": args.meet,
            "race_no": args.race,
            "race_id": race_id,
            "advertised_start_nz": start_nz,
            "number": r.get("number"),
            "runner": r.get("name"),
            "barrier": r.get("barrier"),
            "jockey": r.get("jockey"),
            "weight": r.get("weight"),
            "win_fx": r.get("win_fixed"),
            "place_fx": r.get("place_fixed"),
            "win_tote": r.get("win_tote"),
            "place_tote": r.get("place_tote"),
            "flucs": r.get("flucs_disp"),
            "form": r.get("form"),
            "speedmap": speedmap_label(r.get("speedmap")),
        })

    # 6) Save CSV
    if args.out:
        out_path = args.out
    else:
        stamp = datetime.now(NZ).strftime("%Y-%m-%d_%H-%M-%S")
        base  = f"M{args.meet}_{sanitize_filename(meeting_name or 'Meeting')}_R{args.race}_{stamp}.csv"
        out_path = base

    fields = [
        "date_nz","meeting_name","meet_no","race_no","race_id","advertised_start_nz",
        "number","runner","barrier","jockey","weight",
        "win_fx","place_fx","win_tote","place_tote","flucs",
        "form","speedmap",
    ]
    save_csv(out_path, rows, fields)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        body = ""
        try: body = e.response.text[:300]
        except: pass
        print(f"HTTP error: {e}\n{body}", file=sys.stderr); sys.exit(2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)
