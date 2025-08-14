
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TAB NZ affiliates GUI + TAB odds merge + Betting Assistant + Condition Edge

What’s inside (v4.2):
- Race Viewer with a top runners table and a Notebook: Signals, Staking, Notes.
- Signals: implied win %, firming %, each-way overlay, Kelly% (1x), pace label, condition edge.
- Staking: pick selections (from the table or auto-pick), compute Dutch (equal-profit) + Kelly stakes,
           show per-runner expected return & profit (from dutching) and a consolidated plan.
- Optional merge of TAB odds (json.tab.co.nz) to get Win/Place tote & flucs.
- Meeting/race browsing with your custom headers.
- Date defaults to *today* on startup.

This is an informational tool only. No betting advice is implied.
"""

import json
from math import isfinite
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

NZ = ZoneInfo("Pacific/Auckland")


# ================== Config ==================
AFF_BASE = "https://api.tab.co.nz/affiliates/v1/racing"

# <<< YOUR HEADERS (customise as needed) >>>
HEADERS = {
    "From": "r.cosgrove@hotmail.com",
    "X-Partner": "Personal use",
    "X-Partner-ID": "Personal use",
    "Accept": "application/json",
    "User-Agent": "RyanCosgrove/1.0",
}
# ^^^ add your real values above ^^^

TIMEOUT = 20
ODDS_BASE = "https://json.tab.co.nz/odds"
RESULTS_BASE = "https://json.tab.co.nz/results"

# ============================================


# ---------- HTTP session with retries ----------
def make_session():
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


# ---------- helpers for meeting/event shapes ----------
def unwrap_meetings(obj: dict):
    if not isinstance(obj, dict):
        return []
    if "meetings" in obj:
        return obj["meetings"]
    if "data" in obj and isinstance(obj["data"], dict):
        return obj["data"].get("meetings", [])
    return []

def meeting_id(m):
    return m.get("id") or m.get("meeting_id") or m.get("uuid") or m.get("meeting")

def race_number(r):
    return r.get("race_number") or r.get("number")

def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def safe_float(x, default=None):
    try:
        f = float(x)
        return f if isfinite(f) else default
    except Exception:
        return default



# ---------- TAB odds merge ----------
def fetch_tab_odds(session, date_str, meetno, raceno):
    """
    Returns a normalized 'markets' list so extract_prices_from_markets() works
    for both legacy 'markets/selections' payloads and the new 'entries' shape.

    - WIN market: fixed = ffwin, tote = win
    - PLACE market: fixed = ffplc, tote = plc
    - skips scratched runners (scr == true)
    """
    url = f"{ODDS_BASE}/{date_str}/{meetno}/{raceno}"
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()

    # If it's already in markets/selections shape, just return it.
    if isinstance(data, dict) and data.get("markets"):
        return data["markets"]

    # Try meetings -> races -> entries shape
    try:
        mtgs = data.get("meetings") or []
        races = mtgs[0].get("races") or []
        race = races[0]
        entries = race.get("entries") or []
    except Exception:
        entries = []

    if not entries:
        return []  # nothing we can use

    def build_market(bettype, fixed_key, tote_key):
        sels = []
        for e in entries:
            if e.get("scr"):
                continue
            num = safe_int(e.get("number"))
            if not num:
                continue
            sel = {
                "number": num,
                "fixed": e.get(fixed_key),
                "tote": e.get(tote_key),
                "flucs": [],  # none in this shape
            }
            sels.append(sel)
        return {"bettype": bettype, "selections": sels}

    markets = [
        build_market("WIN",   "ffwin", "win"),
        build_market("PLACE", "ffplc", "plc"),
    ]
    return markets

def extract_prices_from_markets(markets):
    """
    Build { runner_number: {win_fixed, place_fixed, win_tote, place_tote, flucs:[...] } }
    """
    out = {}

    def norm_bt(bt: str):
        if not bt:
            return ""
        bt = bt.upper().strip()
        if bt in {"WIN", "W"}:
            return "WIN"
        if bt in {"PLACE", "PLC", "PL"}:
            return "PLACE"
        return bt

    for mkt in markets or []:
        bt = norm_bt(mkt.get("bettype") or mkt.get("name") or "")
        for s in mkt.get("selections") or []:
            num = safe_int(s.get("number") or s.get("runner") or 0)
            if not num:
                continue
            fixed = (s.get("fixed") or s.get("price"))
            tote = (s.get("tote") or s.get("dividend") or s.get("pool_dividend"))
            flucs = s.get("flucs") or s.get("price_history") or []
            rec = out.setdefault(num, {"flucs": []})
            if bt == "WIN":
                if fixed is not None: rec["win_fixed"] = fixed
                if tote is not None:  rec["win_tote"] = tote
            elif bt == "PLACE":
                if fixed is not None: rec["place_fixed"] = fixed
                if tote is not None:  rec["place_tote"] = tote
            if flucs:
                rec["flucs"] = flucs
    return out

def merge_odds_into_event(event_payload, prices_by_num):
    """
    Adds .prices to each runner where we find a number/barrier.
    Supports shapes: data.runners OR meetings[0].races[...].runners
    """
    changed = False

    def merge_runner(r):
        nonlocal changed
        num = safe_int(r.get("runner_number") or r.get("number") or r.get("barrier") or 0)
        if num and num in prices_by_num:
            r.setdefault("prices", {}).update(prices_by_num[num])
            changed = True

    root = event_payload.get("data") or event_payload
    runners = root.get("runners")
    if isinstance(runners, list) and runners:
        for r in runners:
            merge_runner(r)

    mtgs = unwrap_meetings(event_payload)
    if mtgs:
        for race in mtgs[0].get("races", []):
            for r in race.get("runners", []) or race.get("entries", []):
                merge_runner(r)

    return changed

def fmt_money(x):
    return "" if x in (None, 0, 0.0) else f"${float(x):,.2f}"

def fetch_tab_race_node(session, date_str, meetno, raceno):
    """
    Returns the single race node from json.tab.co.nz/odds/{date}/{meet}/{race}
    (the shape has `meetings[0].races[0]` with `entries` and pool totals).
    """
    url = f"{ODDS_BASE}/{date_str}/{meetno}/{raceno}"
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        return {}
    # Prefer exact match on number if present
    mtgs = data.get("meetings") or []
    for mtg in mtgs:
        for race in mtg.get("races", []) or []:
            if safe_int(race.get("number")) == raceno:
                return race
        # fallback: first race if only one
        if mtgs and mtgs[0].get("races"):
            return mtgs[0]["races"][0]
    # some endpoints may surface races directly (rare)
    if data.get("races"):
        return data["races"][0]
    return {}

def extract_prices_from_tab_race(race_node):
    """
    Build { runner_number: {win_fixed, place_fixed, win_tote, place_tote} } from `entries`.
    Uses fields: ffwin, ffplc (fixed), win, plc (tote).
    """
    out = {}
    entries = race_node.get("entries") or []
    for e in entries:
        num = safe_int(e.get("number") or e.get("runner") or e.get("runner_number"))
        if not num:
            continue
        rec = out.setdefault(num, {})
        if e.get("ffwin") is not None: rec["win_fixed"]  = e.get("ffwin")
        if e.get("ffplc") is not None: rec["place_fixed"] = e.get("ffplc")
        if e.get("win")   is not None: rec["win_tote"]   = e.get("win")
        if e.get("plc")   is not None: rec["place_tote"] = e.get("plc")
    return out

def summarize_tab_race(race_node):
    """
    entrants: count non-scratched entries
    tote_avail: True if any tote odds present (win/plc) or pools > 0
    handle: sum of pool totals found on this race node
    """
    entries = race_node.get("entries") or []
    entrants = sum(1 for e in entries if not e.get("scr"))
    tote_avail = any(
        (safe_float(e.get("win")) or 0) > 0 or (safe_float(e.get("plc")) or 0) > 0
        for e in entries
    )
    handle = 0.0
    for k in ("winpool", "plcpool", "qlapool", "tfapool", "exapool", "quapool"):
        v = safe_float(race_node.get(k))
        if v:
            handle += v
    return (entrants or None), tote_avail, (handle if handle > 0 else None)


# ---------- metrics utilities ----------
def get_open_fluc(r):
    fwt = r.get("flucs_with_timestamp") or {}
    of = fwt.get("open", {}) if isinstance(fwt, dict) else {}
    val = of.get("fluc")
    if val is not None:
        return safe_float(val)
    fl = r.get("flucs")
    if isinstance(fl, list) and fl:
        return safe_float(fl[0])
    return None

def last_fluc(r):
    fl = r.get("flucs")
    if isinstance(fl, list) and fl:
        v = fl[-1]
        return safe_float(v)
    fwt = r.get("flucs_with_timestamp") or {}
    last6 = fwt.get("last_six") or []
    if last6 and isinstance(last6[-1], dict):
        return safe_float(last6[-1].get("fluc"))
    odds = r.get("odds") or {}
    if "fixed_win" in odds:
        return safe_float(odds.get("fixed_win"))
    return None

def fmt_price(x):
    return "—" if x in (None, 0, 0.0) else (f"{float(x):g}" if isinstance(x, (int,float)) else str(x))

import time



def parse_start_to_utc(rc: dict):
    """
    Return an aware UTC datetime for the race start, from any of:
    - integer epoch (advertised_start / start_time / tote_start_time)
    - ISO strings like '2025-08-11T04:02:00Z' or with +offset
    If a naive string is seen, assume UTC.
    """
    # numeric epoch?
    for k in ("advertised_start", "start_time", "tote_start_time"):
        ts = rc.get(k)
        if isinstance(ts, (int, float)) or (isinstance(ts, str) and ts.isdigit()):
            try:
                return datetime.fromtimestamp(int(ts), tz=timezone.utc)
            except Exception:
                pass

    # string forms
    s = rc.get("advertised_start_string") or rc.get("start_time") or rc.get("tote_start_time") or ""
    if isinstance(s, str) and s:
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"           # make 'Z' ISO-compatible
            s = s.replace(" ", "T", 1)           # tolerate "YYYY-MM-DD HH:MM:SS"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            return None
    return None

def fmt_dt_nz(dt_utc: datetime) -> str:
    if not dt_utc:
        return ""
    local = dt_utc.astimezone(NZ)
    return f"{local:%Y-%m-%d %H:%M} {local.tzname()}"

def minutes_to_jump_utc(dt_utc: datetime):
    if not dt_utc:
        return ""
    now_utc = datetime.now(timezone.utc)
    return int((dt_utc - now_utc).total_seconds() // 60)


def estimate_runners(rc):
    """
    Meeting-level estimator.
    Prefer explicit counts; else count non-scratched entries/runners; else max runner/barrier number.
    """
    n = safe_int(rc.get("field_size") or rc.get("entrant_count"))
    if n and n > 0:
        return n

    for key in ("runners", "entries"):
        arr = rc.get(key)
        if isinstance(arr, list) and arr:
            cnt, mx = 0, 0
            for it in arr:
                scr = it.get("is_scratched")
                if scr is None:
                    scr = it.get("scr")
                if not scr:
                    cnt += 1
                no = safe_int(it.get("runner_number") or it.get("number") or it.get("barrier"))
                if no:
                    mx = max(mx, no)
            return cnt or (mx if mx > 0 else None)
    return None

def estimate_runners_from_event(payload):
    """
    Event-level estimator (has full runner objects).
    """
    data = payload.get("data") or {}
    runners = data.get("runners") or []
    if runners:
        cnt = sum(1 for r in runners if not r.get("is_scratched"))
        if cnt:
            return cnt
        mx = 0
        for r in runners:
            no = safe_int(r.get("runner_number") or r.get("number") or r.get("barrier"))
            if no:
                mx = max(mx, no)
        return mx or None
    return None

def tote_available_in_event(payload):
    """
    Return True if we can see any tote (pool) pricing/pools on the event.
    """
    data = payload.get("data") or {}
    # pools list is the most reliable
    for p in (data.get("tote_pools") or []):
        if p.get("status") == "OPEN" and safe_float(p.get("total")) not in (None, 0.0):
            return True
    # or per-runner pool odds
    for r in (data.get("runners") or []):
        od = r.get("odds") or {}
        if safe_float(od.get("pool_win")) or safe_float(od.get("pool_place")):
            return True
    return False

def extract_handle_from_event(payload):
    """
    Rough handle = sum of all tote_pools totals on the event.
    """
    data = payload.get("data") or {}
    tot = 0.0
    for p in (data.get("tote_pools") or []):
        v = safe_float(p.get("total"))
        if v:
            tot += v
    return tot if tot > 0 else None


def pct(x):
    return "" if x is None else f"{x*100:.1f}"

def normalize_condition(s):
    s = (s or "").strip().lower()
    if not s:
        return ""
    if "firm" in s: return "firm"
    if "good" in s: return "good"
    if "soft" in s: return "soft"
    if "heavy" in s: return "heavy"
    if "synthetic" in s or "poly" in s or "tapeta" in s: return "synthetic"
    return s

def compute_condition_edge(r, race_condition: str):
    """
    If past_performances exist, compute a simple "same condition" top-3 rate vs overall.
    Return a string tag like 'Cond+ 58% top3' when we see a decent edge.
    """
    pps = r.get("past_performances")
    if not pps or not isinstance(pps, list):
        # fall back to form_indicators
        tags = []
        for fi in (r.get("form_indicators") or []):
            name = (fi.get("name") or "").lower()
            if "track / distance" in name or "track specialist" in name or "distance specialist" in name:
                tags.append("TD+")
        return " ".join(tags)

    this_cond = normalize_condition(race_condition)
    tot_starts = wins = top3 = 0
    c_starts = c_wins = c_top3 = 0

    for p in pps:
        tot_starts += 1
        pos = safe_int(p.get("position"))
        if pos and pos <= 3: top3 += 1
        if pos and pos == 1: wins += 1
        cond = normalize_condition(p.get("track_condition"))
        if this_cond and cond == this_cond:
            c_starts += 1
            if pos and pos <= 3: c_top3 += 1
            if pos and pos == 1: c_wins += 1

    if tot_starts == 0 or c_starts < 2:
        return ""

    # smoothed top3 rates
    overall_top3 = (top3 + 1) / (tot_starts + 2)
    cond_top3 = (c_top3 + 1) / (c_starts + 2)
    lift = cond_top3 - overall_top3

    if lift >= 0.15:  # +15% or more
        return f"Cond+ {cond_top3*100:.0f}% top3"
    return ""

# --- Edge tags from form_indicators ------------------------------------------
def edge_from_indicators(form_indicators):
    """
    Map affiliates form_indicators into short, readable tags for the Edge column.
    De-dupes and returns a single space-separated string like: "TD+ LSW HTR".
    """
    tags = []

    for ind in (form_indicators or []):
        g = (ind.get("group") or "").strip()
        n = (ind.get("name") or "").strip()
        neg = bool(ind.get("negative"))

        # Group-based positives
        if g in {"Track_Distance", "Course_Distance"} and not neg:
            tags.append("TD+")         # Track & Distance specialist
            continue
        if g == "Track" and not neg:
            tags.append("T+")          # Track specialist
            continue
        if g == "Distance" and not neg:
            tags.append("D+")          # Distance specialist
            continue
        if g == "Last_Start_Winner" and not neg:
            tags.append("LSW")         # Last-start winner
            continue

        # Name-based fallbacks / aliases (handles weird spacing/casing)
        low = n.lower()
        if low.startswith("track / distance") and not neg:
            tags.append("TD+")
            continue
        if low.startswith("course / distance") and not neg:
            tags.append("CD+")         # Some feeds say Course/Distance
            continue
        if low.startswith("going for the hat-trick") and not neg:
            tags.append("HTR")         # Going for the hat-trick
            continue

        # (Optional) show negatives if the feed ever sets negative=True
        if g == "Track_Distance" and neg:
            tags.append("TD-")
        elif g == "Track" and neg:
            tags.append("T-")
        elif g == "Distance" and neg:
            tags.append("D-")

    # De-dupe, keep order
    out, seen = [], set()
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return " ".join(out)

def extract_handle_from_event(payload):
    """
    Best-effort grab of total turnover/handle from event payload if present.
    Looks inside data.money_tracker for common keys.
    """
    try:
        mt = (payload.get("data") or {}).get("money_tracker") or {}
        # try a few likely fields
        for k in ("total_turnover", "total_staked", "total", "turnover", "sum", "pool_total"):
            v = mt.get(k)
            if isinstance(v, (int, float)) and v >= 0:
                return float(v)
        # sometimes nested by pool:
        if isinstance(mt, dict):
            tot = 0.0
            found = False
            for v in mt.values():
                if isinstance(v, dict):
                    for kk in ("total", "turnover", "pool_total"):
                        if isinstance(v.get(kk), (int, float)):
                            tot += float(v[kk]); found = True
            if found:
                return tot
    except Exception:
        pass
    return None


def compute_runner_metrics(r, race_condition=""):
    """Compute per-runner metrics from affiliates payload + merged prices.
    Returns:
      win_fx, place_fx, win_tote, place_tote,
      imp_win, firming_pct, ew_overlay_pct, pace, kelly_pct1x, edge_tag
    """
    odds = r.get("odds") or {}
    prices = r.get("prices") or {}
    win_fx   = safe_float(odds.get("fixed_win") or prices.get("win_fixed"))
    place_fx = safe_float(odds.get("fixed_place") or prices.get("place_fixed"))
    win_tote   = safe_float(prices.get("win_tote") or odds.get("win_tote"))
    place_tote = safe_float(prices.get("place_tote") or odds.get("place_tote"))

    imp_win = None
    if win_fx and win_fx > 1.0:
        imp_win = 1.0 / win_fx

    firming_pct = None
    fl = r.get("flucs") or []
    if isinstance(fl, list) and len(fl) >= 1:
        first = safe_float(fl[0])
        last  = safe_float(fl[-1])
        if first and last and first > 0:
            firming_pct = (first - last) / first

    ew_overlay_pct = None
    if place_tote and place_fx and place_fx > 0:
        ew_overlay_pct = (place_tote - place_fx) / place_fx

    # pace label from speedmap if present
    pace = ""
    sm = r.get("speedmap") or {}
    lbl = (sm.get("label") or "").strip()
    if lbl:
        pace = lbl

    # Kelly fraction (1x). Prefer WinTote as fair odds; else WinFx → edge≈0.
    kelly_pct1x = None
    if win_fx and win_fx > 1.0:
        b = win_fx - 1.0
        fair_p = None
        if win_tote and win_tote > 1.0:
            fair_p = 1.0 / win_tote
        elif win_fx and win_fx > 1.0:
            fair_p = 1.0 / win_fx
        if fair_p is not None and b > 0:
            q = 1.0 - fair_p
            f = (b * fair_p - q) / b
            if f is not None:
                kelly_pct1x = max(0.0, f)

    # Edge tags from form_indicators
    edge = edge_from_indicators(r.get("form_indicators"))
    
    return {
        "win_fx": win_fx,
        "place_fx": place_fx,
        "win_tote": win_tote,
        "place_tote": place_tote,
        "imp_win": imp_win,
        "firming_pct": firming_pct,
        "ew_overlay_pct": ew_overlay_pct,  # keep if you still use it elsewhere
        "pace": pace,
        "edge": edge,               # <— NEW
        "kelly_pct1x": kelly_pct1x,
    }

def fetch_event_for_dayslip(session, race_id: str):
    params = {"enc": "json"}
    r = session.get(f"{AFF_BASE}/events/{race_id}", headers=HEADERS, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def select_dutch_picks_from_payload(payload, max_picks: int):
    data = payload.get("data") or payload
    race_cond = (data.get("race") or {}).get("track_condition") or ""
    runners = extract_runners(payload)

    scored = []
    for r in runners:
        m = compute_runner_metrics(r, race_cond)
        O = m.get("win_fx")
        if not (O and O > 1.0):
            continue
        # same scoring as Recommend picks (overlay + firming + tiny edge bonus)
        overlay = (m.get("ew_overlay_pct") or 0.0)
        firm = m.get("firming_pct")
        firm_adj = -999 if firm is None else -firm   # prefer firming
        edge_bonus = 0.05 if m.get("edge") else 0.0
        score = overlay * 2.0 + firm_adj * 0.2 + edge_bonus
        scored.append((score, r, m))

    scored.sort(reverse=True, key=lambda x: x[0])
    picks = [(r, m) for _, r, m in scored[:max_picks]]

    # Fallback: if nothing scored (e.g. no overlay/firming), take up to N shortest odds
    if not picks:
        runners_with_O = []
        for r in runners:
            m = compute_runner_metrics(r, race_cond)
            O = m.get("win_fx")
            if O and O > 1.0:
                runners_with_O.append((O, r, m))
        runners_with_O.sort(key=lambda t: t[0])  # shortest odds first
        picks = [(r, m) for _, r, m in runners_with_O[:max_picks]]

    return picks


def extract_runners(payload):
    data = payload.get("data") or payload
    runners = []

    base = data.get("runners")
    if not base:
        mtgs = unwrap_meetings(payload)
        if mtgs:
            for race in mtgs[0].get("races", []):
                if race.get("runners"):
                    base = race["runners"]
                    break

    for r in base or []:
        num = safe_int(r.get("runner_number") or r.get("number"))
        bar = safe_int(r.get("barrier"))
        jockey = r.get("jockey") or ""
        weight = ""
        w = r.get("weight") or {}
        if isinstance(w, dict):
            weight = w.get("allocated") or w.get("total") or ""
        elif w:
            weight = str(w)

        odds = r.get("odds") or {}
        prices = r.get("prices") or {}
        win_fx  = safe_float(odds.get("fixed_win")  or prices.get("win_fixed"))
        pl_fx   = safe_float(odds.get("fixed_place") or prices.get("place_fixed"))
        win_tt  = safe_float(prices.get("win_tote"))
        pl_tt   = safe_float(prices.get("place_tote"))

        lf = last_fluc(r)
        fl_disp = "" if lf is None else str(lf)

        new_r = {
            "no": num,
            "name": r.get("name"),
            "barrier": bar,
            "jockey": jockey,
            "weight": weight,
            "win_fixed": win_fx,
            "place_fixed": pl_fx,
            "win_tote": win_tt,
            "place_tote": pl_tt,
            "flucs_disp": fl_disp,
            "form": r.get("last_twenty_starts") or "",
            "form_indicators": r.get("form_indicators") or [],
            "speedmap": r.get("speedmap") or {},
            "flucs": r.get("flucs"),
            "flucs_with_timestamp": r.get("flucs_with_timestamp"),
            "odds": r.get("odds") or {},
            "prices": r.get("prices") or {},
            "past_performances": r.get("past_performances"),
            "entrant_id": r.get("entrant_id") or r.get("horse_id") or r.get("competitor_id"),
        }
        runners.append(new_r)

    try:
        runners.sort(key=lambda x: (x["no"] is None, x["no"]))
    except Exception:
        pass
    return runners


# ---------- Raw JSON viewer ----------
class JsonViewer(tk.Toplevel):
    def __init__(self, parent, payload, title="Event JSON"):
        super().__init__(parent)
        self.title(title)
        self.geometry("900x640")
        self.payload = payload

        top = ttk.Frame(self, padding=8)
        top.pack(fill="both", expand=True)

        self.text = ScrolledText(top, wrap="none")
        self.text.pack(fill="both", expand=True)
        self.text.insert("1.0", json.dumps(payload, indent=2, sort_keys=True))
        self.text.configure(state="disabled")

        btns = ttk.Frame(self, padding=(0, 8, 0, 0))
        btns.pack(fill="x")

        ttk.Button(btns, text="Save JSON", command=self.save_json).pack(side="left")
        ttk.Button(btns, text="Copy to clipboard", command=self.copy_json).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Close", command=self.destroy).pack(side="right")


    def save_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save event JSON"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.payload, f, indent=2, sort_keys=True)
            messagebox.showinfo("Saved", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def copy_json(self):
        try:
            self.clipboard_clear()
            self.clipboard_append(json.dumps(self.payload, indent=2, sort_keys=True))
            self.update_idletasks()
            messagebox.showinfo("Copied", "JSON copied to clipboard")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy:\n{e}")




# ---------- Race Viewer ----------
class RaceViewer(tk.Toplevel):
    def __init__(self, parent, payload, title="Race"):
        super().__init__(parent)
        self.parent = parent
        self.payload = payload
        self.title(title)
        self.geometry("1180x900")

        data = payload.get("data") or payload
        self.race = data.get("race") or {}
        fav = data.get("favourite") or {}
        mover = data.get("mover") or {}

        race_cond = self.race.get("track_condition") or ""

        # runners + precompute metrics
        self.runners = extract_runners(payload)
        for r in self.runners:
            r["metrics"] = compute_runner_metrics(r, race_cond)

        # Header
        hdr = ttk.Frame(self, padding=10)
        hdr.pack(fill="x")

        head_title = f"{self.race.get('display_meeting_name') or self.race.get('meeting_name') or ''} — " \
                     f"R{self.race.get('race_number','?')}: {self.race.get('description','')}"
        ttk.Label(hdr, text=head_title, font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        
        # NZ-local start time
        start_utc = parse_start_to_utc(self.race) 
        start_str = fmt_dt_nz(start_utc)

        meta_line = " | ".join([
            f"Distance: {self.race.get('distance', '')}",
            f"Start: {start_str}",
            f"Status: {self.race.get('status', '')}",
            f"Track: {self.race.get('track_condition', '') or '-'}",
            f"Weather: {self.race.get('weather', '') or '-'}",
            f"Positions paid: {self.race.get('positions_paid', '') or '-'}",
        ])
        ttk.Label(hdr, text=meta_line).pack(anchor="w", pady=(4,0))

        chips = ttk.Frame(hdr)
        chips.pack(fill="x", pady=(6, 0))
        if fav.get("name"):
            ttk.Label(chips, text=f"Favourite: {fav.get('name')}  (#{fav.get('runner_number','?')}  "
                                  f"FW {fmt_price((fav.get('odds') or {}).get('fixed_win'))}, "
                                  f"FP {fmt_price((fav.get('odds') or {}).get('fixed_place'))})",
                      foreground="#0a7").pack(side="left")
        if mover.get("name"):
            ttk.Label(chips, text=f"  Mover: {mover.get('name')}",
                      foreground="#a70").pack(side="left")
        ttk.Button(chips, text="Refresh odds", command=self.refresh_odds).pack(side="right")
        ttk.Button(chips, text="Check results", command=self.check_results).pack(side="right", padx=(8, 0))
        # ... after the `chips` frame (right before you build the tables) add:

        # Top toolbar (always visible)
        toolbar = ttk.Frame(self, padding=(10, 6, 10, 6))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Raw JSON", command=self.show_raw).pack(side="left")
        ttk.Button(toolbar, text="Save JSON", command=self.save_json).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Check results", command=self.check_results).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Close", command=self.destroy).pack(side="right")

        # Handy hotkeys (Ctrl+S on Windows/Linux, Cmd+S on macOS)
        self.bind_all("<Control-s>", lambda e: self.save_json())
        self.bind_all("<Command-s>", lambda e: self.save_json())

        # Top runners table
        table_frame = ttk.Frame(self, padding=(10, 6, 10, 6))
        table_frame.pack(fill="both", expand=True)

        cols = ("no","name","barrier","jockey","weight","win_fx","pl_fx","win_tote","pl_tote","flucs","last20","edge")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="extended", height=12)
        headers = ["#","Runner","Barrier","Jockey","Wgt","WIFx","PIFx","WITote","PITote","Flucs","Form","Edge"]

        for c, h in zip(cols, headers):
            self.tree.heading(c, text=h)

        self.tree.column("no", width=50, anchor="center")
        self.tree.column("name", width=280, anchor="w")
        self.tree.column("barrier", width=70, anchor="center")
        self.tree.column("jockey", width=150, anchor="w")
        self.tree.column("weight", width=70, anchor="center")
        self.tree.column("win_fx", width=80, anchor="center")
        self.tree.column("pl_fx", width=80, anchor="center")
        self.tree.column("win_tote", width=80, anchor="center")
        self.tree.column("pl_tote", width=80, anchor="center")
        self.tree.column("flucs", width=120, anchor="center")
        self.tree.column("last20", width=180, anchor="w")
        self.tree.column("edge", width=120, anchor="w")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # tag for condition edge highlight
        self.tree.tag_configure("edge_row", background="#eaf7ea")
        # tag colours for 1-2-3
        self.tree.tag_configure("winner", background="#ffe680")   # gold-ish
        self.tree.tag_configure("second", background="#e0e0e0")   # silver
        self.tree.tag_configure("third",  background="#f7c6a3")   # bronze

        # clear any existing rows before refilling
        self.tree.delete(*self.tree.get_children())

        for r in self.runners:
            m = compute_runner_metrics(r)
            self.tree.insert(
                "",
                "end",
                iid=f"r{r.get('no')}",
                values=(
                    r.get("no") or "",
                    r.get("name") or "",
                    r.get("barrier") or "",
                    r.get("jockey") or "",
                    r.get("weight") or "",
                    fmt_price(r.get("win_fixed")),
                    fmt_price(r.get("place_fixed")),
                    fmt_price(r.get("win_tote")),
                    fmt_price(r.get("place_tote")),
                    r.get("flucs_disp") or "",
                    (r.get("form") or "")[:24],   # <<— your form guide (last_twenty_starts)
                    m.get("edge") or "",          # short “edge” tags
                ),
            )

        # ----- Notebook: Signals / Staking / Notes -----
        nb = ttk.Notebook(self)
        # IMPORTANT: make the notebook the top filler
        nb.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))



        # Signals tab
        sig_frame = ttk.Frame(nb, padding=10)
        nb.add(sig_frame, text="Signals")

        sig_cols = ("no","name","winfx","wintote","imp","firm","ewov","kelly","pace","edge")
        self.sig_tree = ttk.Treeview(sig_frame, columns=sig_cols, show="headings", height=10)
        for cid, label in zip(sig_cols, ["#","Runner","WinFx","WinTote","Imp%","Firm%","EWov%","Kelly%","Pace","Edge"]):
            self.sig_tree.heading(cid, text=label)

        for cid, w, anchor in [
            ("no",60,"center"),("name",260,"w"),("winfx",80,"center"),("wintote",80,"center"),
            ("imp",80,"center"),("firm",80,"center"),("ewov",80,"center"),("kelly",80,"center"),
            ("pace",120,"w"),("edge",160,"w")
        ]:
            self.sig_tree.column(cid, width=w, anchor=anchor)

        sig_vsb = ttk.Scrollbar(sig_frame, orient="vertical", command=self.sig_tree.yview)
        self.sig_tree.configure(yscroll=sig_vsb.set)
        self.sig_tree.pack(side="left", fill="both", expand=True)
        sig_vsb.pack(side="right", fill="y")

# Fill Signals grid (use precomputed metrics stored on each runner)
        for r in self.runners:
            m = r["metrics"]
            self.sig_tree.insert(
                "",
                "end",
                iid=f"r{r.get('no')}",  
                values=(
                    r.get("no") or "",
                    r.get("name") or "",
                    fmt_price(m.get("win_fx")),
                    fmt_price(m.get("win_tote")),
                    pct(m.get("imp_win")),         # implied win % from WinFx
                    pct(m.get("firming_pct")),     # price firming %
                    pct(m.get("ew_overlay_pct")),  # each-way overlay %
                    pct(m.get("kelly_pct1x")),     # Kelly (1x) fraction %
                    m.get("pace") or "",
                    m.get("edge") or "",
                ),
            )



        # Staking tab
        stk_frame = ttk.Frame(nb, padding=10)
        nb.add(stk_frame, text="Staking")

        ctrl = ttk.Frame(stk_frame)
        ctrl.pack(fill="x")

        ttk.Label(ctrl, text="Bankroll $").grid(row=0, column=0, sticky="w")
        self.bankroll_var = tk.StringVar(value="200")
        ttk.Entry(ctrl, textvariable=self.bankroll_var, width=10).grid(row=0, column=1, sticky="w", padx=(4,12))

        ttk.Label(ctrl, text="Kelly mult").grid(row=0, column=2, sticky="w")
        self.kelly_mult_var = tk.StringVar(value="0.50")
        ttk.Entry(ctrl, textvariable=self.kelly_mult_var, width=6).grid(row=0, column=3, sticky="w", padx=(4,12))

        ttk.Label(ctrl, text="Dutch profit $").grid(row=0, column=4, sticky="w")
        self.target_profit_var = tk.StringVar(value="20")
        ttk.Entry(ctrl, textvariable=self.target_profit_var, width=8).grid(row=0, column=5, sticky="w", padx=(4,12))

        self.require_kelly_pos = tk.BooleanVar(value=True)
        self.max_odds_var = tk.StringVar(value="12")  # cap at 12.0 by default

        ttk.Checkbutton(ctrl, text="Only pick value (Kelly>0)", variable=self.require_kelly_pos)\
            .grid(row=0, column=7, sticky="w", padx=(12,0))
        ttk.Label(ctrl, text="Max WinFx").grid(row=0, column=8, sticky="w", padx=(12,0))
        ttk.Entry(ctrl, textvariable=self.max_odds_var, width=6).grid(row=0, column=9, sticky="w")


        self.use_tote_as_fair = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Kelly uses WinTote as fair (fallback WinFx)", variable=self.use_tote_as_fair)\
            .grid(row=0, column=6, sticky="w")

        act = ttk.Frame(stk_frame)
        act.pack(fill="x", pady=(8,4))
        ttk.Button(act, text="Load selected picks", command=self.load_selected_picks).pack(side="left")
        ttk.Button(act, text="Recommend picks", command=self.recommend_picks).pack(side="left", padx=(8,0))
        ttk.Button(act, text="Compute stakes", command=self.compute_stakes).pack(side="left", padx=(8,0))
        ttk.Button(act, text="Copy plan", command=self.copy_plan).pack(side="right")
        ttk.Button(act, text="Save plan", command=self.save_plan).pack(side="right", padx=(8,0))

        picks_cols = ("no","name","winfx","wintote","kelly_pct","dutch","ret","profit","stake")
        self.picks_tree = ttk.Treeview(stk_frame, columns=picks_cols, show="headings", height=8)
        for cid, label in zip(picks_cols, ["#","Runner","WinFx","WinTote","Kelly%","Dutch$","Return$","Profit$","Use$"]):
            self.picks_tree.heading(cid, text=label)
        for cid, w, anchor in [
            ("no",60,"center"), ("name",260,"w"), ("winfx",80,"center"), ("wintote",80,"center"),
            ("kelly_pct",80,"center"), ("dutch",80,"center"), ("ret",90,"center"), ("profit",90,"center"),
            ("stake",90,"center")
        ]:
            self.picks_tree.column(cid, width=w, anchor=anchor)

        self.picks_tree.pack(fill="x", padx=0, pady=(0,8))

        self.plan_text = ScrolledText(stk_frame, height=10, wrap="word")
        self.plan_text.pack(fill="both", expand=True)

        # Notes tab
        notes = ScrolledText(nb, height=8, wrap="word")
        nb.add(notes, text="Notes")
        race_comment = (self.race.get("comment") or "").strip()
        if race_comment:
            notes.insert("1.0", race_comment + "\n")

        """        
        # Bottom buttons (pin to bottom so they’re always visible)
        btns = ttk.Frame(self, padding=10)
        btns.pack(side="bottom", fill="x")   # <-- was just pack(fill="x")

        ttk.Button(btns, text="Raw JSON", command=self.show_raw).pack(side="left")
        ttk.Button(btns, text="Save JSON", command=self.save_json).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Close", command=self.destroy).pack(side="right")
        """

    # ----- Staking helpers -----
    def _val_or_none(self, s):
        s = (s or "").strip()
        if s in {"", "—", "-"}:
            return None
        return safe_float(s)
    
    def _results_url_parts(self):
        """Build (date_str, meetno, raceno) for the results endpoint."""
        # Race date: prefer NZ date from the event; else GUI date field
        date_str = (self.race.get("race_date_nz") or "").strip()
        if not date_str:
            date_str = (self.parent.date_var.get() or "").strip()

        # Meeting number (TAB tote meeting number) from parent’s current meeting
        meetno = self.parent.current_meetno()
        # Race number from the event
        raceno = safe_int(self.race.get("race_number"))

        return date_str, meetno, raceno

    def _fetch_results_json(self):
        date_str, meetno, raceno = self._results_url_parts()
        if not (date_str and meetno and raceno):
            raise ValueError("Missing date/meet#/race# to fetch results.")

        url = f"{RESULTS_BASE}/{date_str}/{meetno}/{raceno}"
        r = self.parent.session.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def _extract_placings_map(self, res_json):
        """
        Return dict {1: runner_no, 2: runner_no, 3: runner_no}.
        Handles the typical results shape.
        """
        try:
            mtgs = res_json.get("meetings") or []
            races = mtgs[0].get("races") or []
            placings = races[0].get("placings") or []
        except Exception:
            placings = []

        places = {}
        for p in placings:
            rank = safe_int(p.get("rank"))
            num = safe_int(p.get("number"))
            if rank in (1, 2, 3) and num:
                places[rank] = num
        return places

    def _clear_result_tags(self):
        for iid in self.tree.get_children(""):
            # keep existing non-result tags if you use them; here we clear all result tags
            current = set(self.tree.item(iid, "tags"))
            current.difference_update({"winner", "second", "third"})
            self.tree.item(iid, tags=tuple(current))

    def _apply_result_tags(self, places):
        """Apply gold/silver/bronze to the matching runner rows."""
        # Build a quick map from runner number -> iid if you didn’t set iids
        def iid_for_runner(num):
            iid = f"r{num}"
            if iid in self.tree.get_children(""):
                return iid
            # fallback scan by value (in case ids differ)
            for it in self.tree.get_children(""):
                if safe_int(self.tree.set(it, "no")) == num:
                    return it
            return None

        self._clear_result_tags()

        for rank, tag in [(1, "winner"), (2, "second"), (3, "third")]:
            num = places.get(rank)
            if not num:
                continue
            iid = iid_for_runner(num)
            if not iid:
                continue
            current = set(self.tree.item(iid, "tags"))
            current.add(tag)
            self.tree.item(iid, tags=tuple(current))
            # Optionally scroll into view for the winner
            if rank == 1:
                self.tree.see(iid)

    def check_results(self):
        """Button handler: fetch results and highlight 1-2-3."""
        try:
            res = self._fetch_results_json()
            places = self._extract_placings_map(res)
            if not places:
                messagebox.showinfo("Results", "No placings found yet for this race.")
                return
            self._apply_result_tags(places)
            messagebox.showinfo(
                "Results",
                "Placings loaded.\n"
                f"1st: #{places.get(1, '?')}   2nd: #{places.get(2, '?')}   3rd: #{places.get(3, '?')}"
            )
        except Exception as e:
            messagebox.showerror("Results error", f"Could not fetch/apply results:\n{e}")


    def load_selected_picks(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select one or more runners in the top table first.")
            return
        existing = { self.picks_tree.set(i,"no") for i in self.picks_tree.get_children() }
        for iid in sel:
            num = self.tree.set(iid, "no")
            if num in existing:
                continue
            name = self.tree.set(iid, "name")
            winfx = self._val_or_none(self.tree.set(iid, "win_fx"))
            wintt = self._val_or_none(self.tree.set(iid, "win_tote"))
            self.picks_tree.insert("", "end", values=(num, name, fmt_price(winfx), fmt_price(wintt), "", "", "", "", ""))

    def recommend_picks(self):
        max_odds = safe_float(self.max_odds_var.get(), None)
        must_be_value = self.require_kelly_pos.get()

        candidates = []
        for r in self.runners:
            m = r["metrics"]
            O = m.get("win_fx")
            if not O or O <= 1.0:
                continue
            if max_odds and O > max_odds:
                continue
            if must_be_value and (m.get("kelly_pct1x") or 0) <= 0:
                continue

            # score: overlay first, then firming (more negative is better), tiny bonus for edge tags
            overlay = m.get("ew_overlay_pct") or 0.0
            firm = m.get("firming_pct")
            firm_adj = 0.0 if firm is None else -firm
            edge_bonus = 0.05 if m.get("edge") else 0.0
            score = overlay * 2.0 + firm_adj * 0.2 + edge_bonus
            candidates.append((score, r, m))

        candidates.sort(reverse=True, key=lambda x: x[0])
        picks = [ (r, m) for _, r, m in candidates[:3] ]

        self.picks_tree.delete(*self.picks_tree.get_children())
        for r, m in picks:
            self.picks_tree.insert("", "end",
                values=(r.get("no"), r.get("name"),
                        fmt_price(m["win_fx"]), fmt_price(m["win_tote"]), "", "", "", "", ""))

    def refresh_odds(self):
        """Fetch latest TAB odds for this race, merge, and refresh tables."""
        try:
            date_str = (self.parent.date_var.get() or "").strip()
            meetno = self.parent.current_meetno()
            raceno = safe_int(self.race.get("race_number"))

            if not (date_str and meetno and raceno):
                messagebox.showwarning("Can't refresh", "Need date, meeting, and race numbers.")
                return

            # Pull odds, merge into the current event payload
            markets = fetch_tab_odds(self.parent.session, date_str, meetno, raceno)
            prices_by_num = extract_prices_from_markets(markets)
            merge_odds_into_event(self.payload, prices_by_num)

            # Rebuild runners and metrics
            self.runners = extract_runners(self.payload)
            race_cond = self.race.get("track_condition") or ""
            for r in self.runners:
                r["metrics"] = compute_runner_metrics(r, race_cond)

            # Refill TOP runners table
            self.tree.delete(*self.tree.get_children())
            for r in self.runners:
                m = r.get("metrics", {})
                self.tree.insert(
                    "",
                    "end",
                    iid=f"r{r.get('no')}",
                    values=(
                        r.get("no") or "",
                        r.get("name") or "",
                        r.get("barrier") or "",
                        r.get("jockey") or "",
                        r.get("weight") or "",
                        fmt_price(r.get("win_fixed")),
                        fmt_price(r.get("place_fixed")),
                        fmt_price(r.get("win_tote")),
                        fmt_price(r.get("place_tote")),
                        r.get("flucs_disp") or "",
                        (r.get("form") or "")[:24],
                        m.get("edge") or "",
                    ),
                )

            # Refill SIGNALS table (if present)
            if hasattr(self, "sig_tree"):
                self.sig_tree.delete(*self.sig_tree.get_children())
                for r in self.runners:
                    m = r.get("metrics", {})
                    self.sig_tree.insert(
                        "",
                        "end",
                        values=(
                            r.get("no") or "",
                            r.get("name") or "",
                            fmt_price(m.get("win_fx")),
                            fmt_price(m.get("win_tote")),
                            pct(m.get("imp_win")),
                            pct(m.get("firming_pct")),
                            pct(m.get("ew_overlay_pct")),
                            pct(m.get("kelly_pct1x")),
                            m.get("pace") or "",
                            m.get("edge") or "",
                        ),
                    )

            # Update any already-loaded picks with the new WinFx/WinTote
            if hasattr(self, "picks_tree"):
                by_no = {str(r.get("no")): r.get("metrics", {}) for r in self.runners}
                for iid in self.picks_tree.get_children():
                    no_ = self.picks_tree.set(iid, "no")
                    m = by_no.get(no_ or "")
                    if m:
                        self.picks_tree.set(iid, "winfx", fmt_price(m.get("win_fx")))
                        self.picks_tree.set(iid, "wintote", fmt_price(m.get("win_tote")))

            messagebox.showinfo("Odds", "Odds refreshed.")
        except Exception as e:
            messagebox.showerror("Odds", f"Failed to refresh odds:\n{e}")


    def compute_stakes(self):
        bankroll = safe_float(self.bankroll_var.get(), 0.0) or 0.0
        k_mult = safe_float(self.kelly_mult_var.get(), 0.5) or 0.5
        target_profit = safe_float(self.target_profit_var.get(), 0.0) or 0.0
        use_tote = self.use_tote_as_fair.get()

        if bankroll <= 0:
            messagebox.showwarning("Bankroll?", "Enter a positive bankroll amount.")
            return

        rows = self.picks_tree.get_children()
        # clear computed cols
        for iid in rows:
            for col in ("kelly_pct", "stake", "dutch", "ret", "profit"):
                try:
                    self.picks_tree.set(iid, col, "")
                except tk.TclError:
                    pass

        if not rows:
            messagebox.showinfo("No picks", "Load picks from the runners table or use 'Recommend picks'.")
            return

        # Collect picks with usable WinFx
        picks = []
        for iid in rows:
            no_ = self.picks_tree.set(iid, "no")
            name = self.picks_tree.set(iid, "name")
            winfx = self._val_or_none(self.picks_tree.set(iid, "winfx"))
            wintt = self._val_or_none(self.picks_tree.set(iid, "wintote"))
            if winfx and winfx > 1.0:
                picks.append({"iid": iid, "no": no_, "name": name, "winfx": float(winfx), "wintt": wintt})

        plan_lines = []
        plan_lines.append(f"Bankroll: ${bankroll:.2f}  |  Kelly mult: {k_mult:.2f}  |  Dutch target profit: ${target_profit:.2f}")
        plan_lines.append("Assumptions: " + ("Kelly uses WinTote as fair probability when available; if missing, falls back to WinFx implied (edge≈0)." if use_tote else "Kelly uses WinFx implied probability only."))
        plan_lines.append("")

        if all((p["winfx"] and p["wintt"] and p["winfx"] <= p["wintt"]) or p["wintt"] is None for p in picks):
            plan_lines.append("Note: No measurable edge (Kelly<=0). Consider skipping this race.")


        # ---------- Kelly (per selection) ----------
        total_kelly = 0.0
        kelly_info = {}
        for p in picks:
            O = p["winfx"]
            b = O - 1.0
            if b <= 0:
                kelly_info[p["iid"]] = (0.0, 0.0)
                continue
            if use_tote and p["wintt"] and p["wintt"] > 1.0:
                fair_p = min(1.0, 1.0 / float(p["wintt"]))
            else:
                fair_p = min(1.0, 1.0 / O)  # no edge if using the same book line
            q = 1.0 - fair_p
            f = (b * fair_p - q) / b
            f = max(0.0, f)
            f *= k_mult
            stake = bankroll * f
            total_kelly += stake
            kelly_info[p["iid"]] = (f, stake)

        # ---------- Dutching (equal-profit with guardrails) ----------
        dutch_stakes = {}
        total_dutch = 0.0
        dutch_note = ""
        T = 0.0

        # Determine if any tote exists among picks to allow Kelly capping and EV filters
        any_tote = any(safe_float(self.picks_tree.set(iid, "wintote")) and safe_float(self.picks_tree.set(iid, "wintote")) > 1.0
                    for iid in rows if self._val_or_none(self.picks_tree.set(iid, "winfx")))

        # Build a small struct we can filter with
        pick_struct = []
        for iid in rows:
            winfx = self._val_or_none(self.picks_tree.set(iid, "winfx"))
            if not (winfx and winfx > 1.0):
                continue
            wintt = self._val_or_none(self.picks_tree.set(iid, "wintote"))
            # value guardrail when tote exists for this runner
            value_ratio = None
            if wintt and wintt > 1.0:
                try:
                    value_ratio = float(winfx) / float(wintt)
                except Exception:
                    value_ratio = None
            # require positive Kelly only if any tote exists in race
            kelly_f, kelly_stake = kelly_info.get(iid, (0.0, 0.0))
            if any_tote:
                if value_ratio is not None and value_ratio < 0.92:
                    continue
                if kelly_f <= 0.0:
                    continue
            pick_struct.append({"iid": iid, "O": float(winfx), "wintt": wintt, "kelly_f": kelly_f})

        # Replace rows list with filtered order:
        if any_tote:
            pick_struct.sort(key=lambda x: x["kelly_f"], reverse=True)
        else:
            pick_struct.sort(key=lambda x: x["O"])

        if target_profit > 0 and pick_struct:
            K = sum(1.0 / p["O"] for p in pick_struct)
            if K >= 0.98:
                dutch_note = f"Dutch skipped: book too tight (Σ1/O={K:.2f})."
                pick_struct = []
            elif K >= 0.85:
                dutch_note = f"Warning: tight market for dutching (Σ1/O={K:.2f})."
            if K < 1.0 and pick_struct:
                # equal-profit solution
                T = (K * target_profit) / (1.0 - K)
                total_dutch = T
                for p in pick_struct:
                    S_i = (T + target_profit) / p["O"]
                    dutch_stakes[p["iid"]] = S_i
                if not dutch_note:
                    dutch_note = "Dutch stakes computed with equal-profit method across picks."
            elif pick_struct:
                # per-runner fallback
                for p in pick_struct:
                    S_i = target_profit / max(1e-9, (p["O"] - 1.0))
                    dutch_stakes[p["iid"]] = S_i
                total_dutch = sum(dutch_stakes.values())
                if not dutch_note:
                    dutch_note = "Equal-profit dutching not feasible (Σ1/O ≥ 1). Used per-runner fallback."

        plan_lines.append(dutch_note)
        if target_profit > 0 and pick_struct:
            plan_lines.append(f"Dutch outlay total: ${total_dutch:.2f}")
        plan_lines.append("")


        # ---------- Per-line outputs & table updates ----------
        for iid in rows:
            # defaults for lines not in picks
            self.picks_tree.set(iid, "kelly_pct", "")
            self.picks_tree.set(iid, "dutch", "")
            self.picks_tree.set(iid, "ret", "")
            self.picks_tree.set(iid, "profit", "")
            self.picks_tree.set(iid, "stake", "")

        for p in picks:
            iid = p["iid"]
            O = p["winfx"]
            name = p["name"]
            no_ = p["no"]

            # Kelly
            kelly_f, kelly_stake = kelly_info.get(iid, (0.0, 0.0))

            # Dutch
            d_stake = dutch_stakes.get(iid, 0.0)

            # Expected return (gross payout) and profit for Dutching
            exp_return = (d_stake * O) if d_stake > 0 else 0.0
            exp_profit = (T + target_profit - T) if (d_stake > 0 and total_dutch > 0 and T > 0) else (exp_return - total_dutch if d_stake > 0 else 0.0)
            # If equal-profit dutching, exp_return should equal (T + P) for every selection and profit = P

            # Final "Use" stake = min(Kelly, Dutch) if both > 0, else whichever exists
            if kelly_stake > 0 and d_stake > 0:
                final_stake = min(kelly_stake, d_stake)
            elif d_stake > 0:
                final_stake = d_stake
            else:
                final_stake = kelly_stake

            # Update table
            self.picks_tree.set(iid, "kelly_pct", f"{kelly_f*100:.1f}")
            self.picks_tree.set(iid, "dutch", f"{d_stake:.2f}" if d_stake > 0 else "")
            self.picks_tree.set(iid, "ret", f"{exp_return:.2f}" if d_stake > 0 else "")
            self.picks_tree.set(iid, "profit", f"{(target_profit if T>0 and d_stake>0 else exp_profit):.2f}" if d_stake > 0 else "")
            self.picks_tree.set(iid, "stake", f"{final_stake:.2f}" if final_stake > 0 else "")

            plan_lines.append(
                f"#{no_:>2} {name:28s}  WinFx {O:g}  "
                f"Kelly% {kelly_f*100:.1f}  KellyStake ${kelly_stake:.2f}  "
                f"Dutch ${d_stake:.2f}  Return ${exp_return:.2f}  Profit ${((target_profit if T>0 and d_stake>0 else exp_profit)):.2f}  "
                f"=> Use ${final_stake:.2f}"
            )

        plan_lines.append("")
        plan_lines.append(f"Totals — Kelly stakes: ${total_kelly:.2f}   Dutch stakes: ${total_dutch:.2f}")

        self.plan_text.delete("1.0", "end")
        self.plan_text.insert("1.0", "\n".join(plan_lines))

    def copy_plan(self):
        try:
            txt = self.plan_text.get("1.0", "end-1c")
            self.clipboard_clear()
            self.clipboard_append(txt)
            self.update_idletasks()
            messagebox.showinfo("Copied", "Plan copied to clipboard")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy:\n{e}")

    def save_plan(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save staking plan"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.plan_text.get("1.0", "end-1c"))
            messagebox.showinfo("Saved", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

    # ----- Raw JSON / Save -----
    def show_raw(self):
        JsonViewer(self, self.payload, title="Raw Event JSON")
    


    def save_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save event JSON",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.payload, f, indent=2, sort_keys=True)
            messagebox.showinfo("Saved", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

class DaySlipDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("DaySlip settings")
        self.geometry("420x280")
        self.resizable(False, False)

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        # Strategy
        ttk.Label(frm, text="Strategy").grid(row=0, column=0, sticky="w")
        self.strategy = tk.StringVar(value="kelly")
        ttk.Radiobutton(frm, text="Kelly", variable=self.strategy, value="kelly").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(frm, text="Dutch", variable=self.strategy, value="dutch").grid(row=0, column=2, sticky="w")

        # Shared
        ttk.Label(frm, text="Bankroll $").grid(row=1, column=0, sticky="w", pady=(8,0))
        self.bankroll = tk.StringVar(value="200")
        ttk.Entry(frm, textvariable=self.bankroll, width=10).grid(row=1, column=1, sticky="w", pady=(8,0))

        self.use_tote = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Use Tote as fair (fallback FF)", variable=self.use_tote).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6,0))

        # Kelly-only
        self.k_mult = tk.StringVar(value="0.50")
        self.min_kelly_pct = tk.StringVar(value="1.0")
        kbox = ttk.LabelFrame(frm, text="Kelly", padding=8)
        kbox.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8,0))
        ttk.Label(kbox, text="Kelly mult").grid(row=0, column=0, sticky="w")
        ttk.Entry(kbox, textvariable=self.k_mult, width=6).grid(row=0, column=1, sticky="w", padx=(6,12))
        ttk.Label(kbox, text="Min Kelly %").grid(row=0, column=2, sticky="w")
        ttk.Entry(kbox, textvariable=self.min_kelly_pct, width=6).grid(row=0, column=3, sticky="w", padx=(6,0))

        # Dutch-only
        self.dutch_profit = tk.StringVar(value="20")
        self.max_picks = tk.StringVar(value="2")
        dbox = ttk.LabelFrame(frm, text="Dutch", padding=8)
        dbox.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8,0))
        ttk.Label(dbox, text="Target profit per race $").grid(row=0, column=0, sticky="w")
        ttk.Entry(dbox, textvariable=self.dutch_profit, width=8).grid(row=0, column=1, sticky="w", padx=(6,12))
        ttk.Label(dbox, text="Max picks / race").grid(row=0, column=2, sticky="w")
        ttk.Entry(dbox, textvariable=self.max_picks, width=6).grid(row=0, column=3, sticky="w", padx=(6,0))

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=3, sticky="e", pady=(12,0))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(btns, text="Build DaySlip", command=self.on_submit).pack(side="right", padx=(0,8))

        # --- after laying out all the dialog widgets ---
        self.update_idletasks()  # ensure sizes are computed

        # current size (fallbacks if not laid out yet)
        cur_w = self.winfo_width() or 480
        cur_h = self.winfo_height() or 360

        new_w = cur_w + 220
        new_h = cur_h + 100

        # center on parent
        px = self.master.winfo_rootx()
        py = self.master.winfo_rooty()
        pw = self.master.winfo_width()
        ph = self.master.winfo_height()
        x = px + max(0, (pw - new_w) // 2)
        y = py + max(0, (ph - new_h) // 2)

        self.geometry(f"{new_w}x{new_h}+{x}+{y}")


    def on_submit(self):
        self.parent._run_dayslip(
            strategy=self.strategy.get(),
            bankroll=self.bankroll.get(),
            k_mult=self.k_mult.get(),
            min_kelly_pct=self.min_kelly_pct.get(),
            dutch_profit=self.dutch_profit.get(),
            max_picks=self.max_picks.get(),
            use_tote=self.use_tote.get(),
        )
        self.destroy()

class DaySlipWindow(tk.Toplevel):
    def __init__(self, parent, text, title="DaySlip"):
        super().__init__(parent)
        self.title(title)
        self.geometry("680x520")
        from tkinter.scrolledtext import ScrolledText
        txt = ScrolledText(self, wrap="word")
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.insert("1.0", text)
        txt.configure(state="disabled")

        btns = ttk.Frame(self, padding=10)
        btns.pack(fill="x")
        def copy():
            self.clipboard_clear()
            self.clipboard_append(text)
        def save():
            from tkinter import filedialog, messagebox
            path = filedialog.asksaveasfilename(defaultextension=".txt",
                filetypes=[("Text files","*.txt"),("All files","*.*")],
                title="Save DaySlip")
            if not path: return
            try:
                with open(path, "w", encoding="utf-8") as f: f.write(text)
                messagebox.showinfo("Saved", f"Saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        ttk.Button(btns, text="Copy", command=copy).pack(side="right")
        ttk.Button(btns, text="Save", command=save).pack(side="right", padx=(0,8))


# ---------- Main GUI ----------
class RacingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TAB NZ affiliates helper")
        self.geometry("1180x820")

        self.session = make_session()

        # State
        self.current_meeting_detail = None
        self.current_event_payload = None
        self.odds_cache = {}    # {(date, meetno, raceno): prices_by_num}
        self.auto_job = None

        # Vars
        today = datetime.now().strftime("%Y-%m-%d")
        self.date_var = tk.StringVar(value=today)  # default to today
        self.meeting_display_to_id = {}
        self.meeting_var = tk.StringVar()
        self.race_id_var = tk.StringVar(value="")

        # Event query params
        self.enc_var = tk.StringVar(value="json")
        self.with_big_bets = tk.BooleanVar(value=False)
        self.with_biggest_bet = tk.BooleanVar(value=False)
        self.with_live_bets = tk.BooleanVar(value=False)
        self.with_money_tracker = tk.BooleanVar(value=False)
        self.with_tote_trends_data = tk.BooleanVar(value=False)
        self.with_will_pays = tk.BooleanVar(value=False)
        self.present_overlay = tk.BooleanVar(value=False)
        self.bet_type_filter = tk.StringVar(value="")

        # Odds options
        self.merge_odds = tk.BooleanVar(value=True)
        self.auto_refresh = tk.BooleanVar(value=False)
        self.refresh_secs = tk.IntVar(value=10)

        # Top bar
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Date YYYY-MM-DD:").pack(side="left")
        self.date_entry = ttk.Entry(top, textvariable=self.date_var, width=14)
        self.date_entry.pack(side="left", padx=(6, 12))

        ttk.Button(top, text="Fetch meetings", command=self.on_fetch_meetings).pack(side="left")

        ttk.Label(top, text="Meeting:").pack(side="left", padx=(18, 0))
        self.meeting_combo = ttk.Combobox(top, textvariable=self.meeting_var, width=72, state="readonly")
        self.meeting_combo.pack(side="left", padx=(6, 12), fill="x", expand=True)

        ttk.Button(top, text="Load races", command=self.on_load_races).pack(side="left")

        # Options row
        opts = ttk.Frame(self, padding=(10, 0, 10, 10))
        opts.pack(fill="x")

        ttk.Checkbutton(opts, text="Merge TAB odds", variable=self.merge_odds).pack(side="left")
        ttk.Checkbutton(opts, text="Auto refresh odds", variable=self.auto_refresh, command=self.on_toggle_auto).pack(side="left", padx=(12, 0))
        ttk.Label(opts, text="secs:").pack(side="left", padx=(6, 0))
        ttk.Spinbox(opts, from_=5, to=120, textvariable=self.refresh_secs, width=4).pack(side="left")
        ttk.Button(opts, text="Refresh odds now", command=self.on_refresh_odds).pack(side="left", padx=(12, 0))
        ttk.Button(opts, text="Build day slip", command=self.open_dayslip).pack(side="left", padx=(12, 0))


        # Races table (per-meeting)
        tree_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        tree_frame.pack(fill="both", expand=True)

        # New race-centric columns
        cols = ("race_number", "name", "klass", "race_id", "start_time",
                "entrants", "tote_avail", "handle", "t2j_min")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=18)

        heads = ["Race", "Name", "Klass", "Race ID", "Start",
                "#Runners", "Tote?", "Handle$", "T-2-J (min)"]
        for cid, label in zip(cols, heads):
            self.tree.heading(cid, text=label)

        self.tree.column("race_number", width=60, anchor="center")
        self.tree.column("name",        width=380, anchor="w")
        self.tree.column("klass",       width=90, anchor="center")
        self.tree.column("race_id",     width=330, anchor="w")
        self.tree.column("start_time",  width=150, anchor="center")
        self.tree.column("entrants",    width=90,  anchor="center")
        self.tree.column("tote_avail",  width=80,  anchor="center")
        self.tree.column("handle",      width=110, anchor="e")
        self.tree.column("t2j_min",     width=100, anchor="center")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")


        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Double-1>", self.on_tree_double_click)

        # Event controls
        event_frame = ttk.LabelFrame(self, text="Race event fetch", padding=10)
        event_frame.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(event_frame, text="Race ID:").grid(row=0, column=0, sticky="w")
        self.race_entry = ttk.Entry(event_frame, textvariable=self.race_id_var, width=50)
        self.race_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))

        ttk.Label(event_frame, text="enc:").grid(row=0, column=2, sticky="e")
        enc_combo = ttk.Combobox(event_frame, textvariable=self.enc_var, values=["json", "xml", "html"], width=7, state="readonly")
        enc_combo.grid(row=0, column=3, sticky="w", padx=(6, 12))

        ttk.Button(event_frame, text="Fetch event", command=self.on_fetch_event).grid(row=0, column=4, sticky="w")

        # Checkboxes row
        flags = ttk.Frame(event_frame)
        flags.grid(row=1, column=0, columnspan=5, sticky="w", pady=(8, 0))
        for i, (label, var) in enumerate([
            ("with_big_bets", self.with_big_bets),
            ("with_biggest_bet", self.with_biggest_bet),
            ("with_live_bets", self.with_live_bets),
            ("with_money_tracker", self.with_money_tracker),
            ("with_tote_trends_data", self.with_tote_trends_data),
            ("with_will_pays", self.with_will_pays),
            ("present_overlay", self.present_overlay),
        ]):
            ttk.Checkbutton(flags, text=label, variable=var).grid(row=0, column=i, sticky="w", padx=(0, 12))

        ttk.Label(flags, text="bet_type_filter:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(flags, textvariable=self.bet_type_filter, width=20).grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(8, 0))

        # Bottom status
        bottom = ttk.Frame(self, padding=10)
        bottom.pack(fill="x")
        ttk.Button(bottom, text="Copy selected race ID", command=self.copy_selected_id).pack(side="left")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var, anchor="w").pack(side="right")

    # ---------- helpers ----------
    def set_status(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

    def on_make_dayslip(self):
        if not self.current_meeting_detail:
            messagebox.showinfo("Pick a meeting", "Load a meeting (and its races) first.")
            return
        DaySlipDialog(self)

    def _run_dayslip(self, strategy, bankroll, k_mult, min_kelly_pct, dutch_profit, max_picks, use_tote):
        # assume meeting detail already loaded

        def _num(x, d=0.0):
            s = str(x).strip().replace(",", "")
            if s.startswith("$"):
                s = s[1:]
            try:
                return float(s)
            except Exception:
                return d

        bankroll       = _num(bankroll, 0.0)
        k_mult         = _num(k_mult, 0.5)
        min_kelly_pct  = _num(min_kelly_pct, 0.0)
        dutch_profit   = _num(dutch_profit, 0.0)
        try:
            max_picks = int(max_picks)
        except Exception:
            max_picks = 3
        use_tote = bool(use_tote)

        mtgs = unwrap_meetings(self.current_meeting_detail)
        if not mtgs or not mtgs[0].get("races"):
            messagebox.showinfo("No races", "Load a meeting first.")
            return

        out = []
        out.append(f"DaySlip — strategy: {strategy}, bankroll ${bankroll:.2f}")
        if strategy == "DUTCH":
            out.append(f"Dutch target profit per race: ${dutch_profit:.2f} | Max picks per race: {max_picks} | Fair for Kelly calc ignored")
        else:
            out.append(f"Kelly mult: {k_mult:.2f} | Min Kelly%: {min_kelly_pct:.1f} | Max picks per race: {max_picks} | Fair uses {'WinTote' if use_tote else 'WinFx'}")
        out.append("")

        # === Legend ===
        out.append("")
        out.append("Legend:")
        out.append("  EV $     = Expected Value in dollars (average profit/loss for this bet)")
        out.append("  ROI %    = Expected Return on Investment percentage")
        out.append("  P_fair   = Fair Probability of winning (model estimate)")
        out.append("  BE       = Break-Even Probability (needed to break even at given odds)")
        out.append("  payout_if_wins = Gross profit if bet wins (excludes stake)")
        out.append("  Kelly %  = Optimal stake fraction (adjusted by Kelly multiplier)")
        out.append("")

        for rc in mtgs[0]["races"]:
            rid = rc.get("id", "")
            row_info = {"id": rid,
                        "race_number": race_number(rc) or "?",
                        "name": rc.get("name","")}
            out.append(self._dayslip_for_event(
                row_info, strategy, bankroll, k_mult, min_kelly_pct,
                dutch_profit, max_picks, use_tote
            ))
            out.append("---------------")

        DaySlipWindow(self, "\n".join(out))






    def _dayslip_for_event(self, race_row_dict, strategy, bankroll, k_mult, min_kelly_pct,
                        dutch_profit, max_picks, use_tote):
        """
        Build one race block for the DaySlip with logic aligned to RaceViewer.

        Behaviours:
        • Always attempt to merge TAB tote odds before computing metrics (parity with RaceViewer).
        • Kelly% from compute_runner_metrics(); same logic as RaceViewer.
        • Kelly mode: if NO tote exists in the race, do NOT filter by min_kelly_pct.
            If tote exists but nothing passes, fallback to best score/shortest odds.
        • Dutch mode guardrails:
            - If runner has tote: skip when value_ratio = win_fx / win_tote < 0.92
            - If any tote exists in race: require Kelly>0 for picks
            - Book tightness checks using Σ(1/O)
            - When equal‑profit is feasible, cap each stake by (Kelly * bankroll * k_mult) if any tote exists
        • Output shows EV $, EV ROI %, P(fair) used, break-even P, and payout if wins.
        """
        rid   = race_row_dict["id"]
        rno   = race_row_dict.get("race_number", "?")
        rname = race_row_dict.get("name", "")
        title = f"Race #{rno}: {rname}"
        lines = [title]

        # 1) Fetch affiliates event
        try:
            payload = fetch_event_for_dayslip(self.session, rid)
        except Exception as e:
            lines.append(f"  (failed to fetch event: {e})")
            lines.append("--------------------------------")
            return "\n".join(lines)

        # 2) Try to merge TAB tote odds (non‑fatal if missing)
        try:
            date_str = (self.date_var.get() or "").strip()
            meetno   = self.current_meetno()
            raceno   = safe_int(rno)
            if date_str and meetno and raceno:
                markets = fetch_tab_odds(self.session, date_str, meetno, raceno)
                prices_by_num = extract_prices_from_markets(markets)
                merge_odds_into_event(payload, prices_by_num)
        except Exception:
            pass

        # 3) Extract runners and compute metrics (same as RaceViewer)
        data = payload.get("data") or {}
        race_cond = (data.get("race") or {}).get("track_condition") or ""
        runners = extract_runners(payload)

        cand = []
        any_tote = False
        for r in runners:
            m = compute_runner_metrics(r, race_cond)
            O = m.get("win_fx")
            if not (O and O > 1.0):
                continue
            if m.get("win_tote") and m["win_tote"] > 1.0:
                any_tote = True
            # precompute value_ratio only when runner has tote
            value_ratio = None
            if m.get("win_tote") and m["win_tote"] > 1.0:
                try:
                    value_ratio = float(O) / float(m["win_tote"])
                except Exception:
                    value_ratio = None
            kelly = m.get("kelly_pct1x") or 0.0
            cand.append((r, m, O, value_ratio, kelly))

        if not cand:
            lines.append("  (no priced runners)")
            lines.append("--------------------------------")
            return "\n".join(lines)

        
        # Parse numeric inputs
        try:
            max_p = int(max_picks) if max_picks else 0
        except Exception:
            max_p = 0

        bankroll = max(0.0, safe_float(bankroll, 0.0) or 0.0)
        km       = max(0.0, safe_float(k_mult, 0.5) or 0.5)
        # >>> interpret Min Kelly as a PERCENT (e.g. 1.0 = 1%)
        min_k_raw = safe_float(min_kelly_pct, 0.0) or 0.0
        min_k     = max(0.0, min_k_raw / 100.0)      # <-- key change
        P        = max(0.0, safe_float(dutch_profit, 0.0) or 0.0)


        strat = str(strategy).lower()
        picks = []

        # ---------- DUTCH MODE ----------
        if strat.startswith("dutch"):
            filtered = []
            for r, m, O, value_ratio, kelly in cand:
                # value guardrail only when this runner has tote
                if value_ratio is not None and value_ratio < 0.92:
                    continue
                # if race has any tote at all, require positive Kelly (EV>0)
                if any_tote and kelly <= 0:
                    continue
                filtered.append((r, m, O, value_ratio, kelly))

            if not filtered:
                lines.append("  (no selections passed dutch guardrails)")
                lines.append("--------------------------------")
                return "\n".join(lines)

            # Rank by Kelly when any tote; else by shortest odds
            if any_tote:
                filtered.sort(key=lambda t: t[4], reverse=True)
            else:
                filtered.sort(key=lambda t: t[2])

            picks = filtered[:max_p] if max_p > 0 else filtered

            # Book tightness checks
            Os = [O for (_, _, O, _, _) in picks]
            K = sum(1.0 / o for o in Os if o and o > 1.0)
            if K >= 0.98:
                lines.append(f"  Book too tight for dutching (Σ1/O={K:.2f}). Skipping race.")
                lines.append("--------------------------------")
                return "\n".join(lines)
            elif K >= 0.85:
                lines.append(f"  Caution: tight market for dutching (Σ1/O={K:.2f}).")

            # Stakes
            if P > 0 and K < 1.0:
                # Equal‑profit solution
                T = (K * P) / (1.0 - K)  # total outlay
                for (r, m, O, _vr, kelly) in picks:
                    S_dutch = (T + P) / O
                    # Kelly cap only meaningful when any tote exists
                    S_kelly = bankroll * (max(0.0, kelly) * km) if any_tote else float('inf')
                    S_use = min(S_dutch, S_kelly)
                    profit = P if S_use >= S_dutch - 1e-6 else max(0.0, S_use * O - T)
                    payout = S_use * (O - 1.0)
                    breakeven_p = 1.0 / O
                    # EV when dutching is not the goal (equal profit), so show win payout instead
                    lines.append(
                        f"  #{r.get('no')} {r.get('name')} @ ${O:g}  "
                        f"use ${S_use:.2f}  payout_if_wins ${payout:.2f}  "
                        f"(kelly_cap ${0.0 if not any_tote else S_kelly:.2f}, BE P {breakeven_p:.3f})"
                    )
            else:
                # Per‑runner fallback targeting P per runner
                for (r, m, O, _vr, kelly) in picks:
                    S_dutch = P / max(1e-9, (O - 1.0)) if P > 0 else 0.0
                    S_kelly = bankroll * (max(0.0, kelly) * km) if any_tote else float('inf')
                    S_use = min(S_dutch, S_kelly)
                    payout = S_use * (O - 1.0)
                    breakeven_p = 1.0 / O
                    lines.append(
                        f"  #{r.get('no')} {r.get('name')} @ ${O:g}  "
                        f"stake ${S_use:.2f}  payout_if_wins ${payout:.2f}  (BE P {breakeven_p:.3f})"
                    )

        # ---------- KELLY MODE ----------
        else:
            # Build filtered list
            filtered = []
            for r, m, O, _vr, kelly in cand:
                if any_tote:
                    # apply threshold only if at least one tote exists in the race
                    if kelly < min_k:
                        continue
                # when no tote, do NOT filter by Kelly (per your requirement)
                filtered.append((r, m, O, kelly))

            # If nothing passed and we do have tote somewhere, fallback to best score/shortest odds
            if not filtered and any_tote:
                tmp = []
                for r, m, O, _vr, _kelly in cand:
                    firm = m.get("firming_pct")
                    firm_adj = 0.0 if firm is None else -firm
                    edge_bonus = 0.05 if m.get("edge") else 0.0
                    overlay = m.get("ew_overlay_pct") or 0.0
                    score = overlay * 2.0 + firm_adj * 0.2 + edge_bonus
                    tmp.append((score, r, m, O))
                tmp.sort(key=lambda t: (t[0], -1.0 / t[3] if t[3] else 0.0), reverse=True)
                filtered = [(r, m, O, m.get("kelly_pct1x") or 0.0) for (score, r, m, O) in tmp]
                lines.append("  (no picks met Min Kelly%; fell back to best scores/shortest odds)")

            # Rank by Kelly when any tote; else by shorter odds
            if any_tote:
                filtered.sort(key=lambda t: t[3], reverse=True)
            else:
                filtered.sort(key=lambda t: t[2])

            picks = filtered[:max_p] if max_p > 0 else filtered

            # Stakes (Kelly fractional). Also show EV and payout if wins.
            for (r, m, O, _k) in picks:
                b = O - 1.0
                if use_tote and (m.get("win_tote") and m["win_tote"] > 1.0):
                    p = min(1.0, max(0.0, 1.0 / float(m["win_tote"])))
                else:
                    p = min(1.0, max(0.0, 1.0 / O))
                q = 1.0 - p
                f = max(0.0, (b * p - q) / b) * km
                S = bankroll * f
                roi_ev = (O * p) - 1.0              # per-$ expected return
                ev_profit = S * roi_ev              # expected profit (EV)
                payout_if_wins = S * (O - 1.0)      # gross profit if it wins
                breakeven_p = 1.0 / O

                lines.append(
                    f"  #{r.get('no')} {r.get('name')} @ ${O:g}  bet ${S:.2f}  "
                    f"EV ${ev_profit:.2f} (ROI {roi_ev*100:.1f}%, P_fair {p:.3f}, BE {breakeven_p:.3f})  "
                    f"payout_if_wins ${payout_if_wins:.2f}  (Kelly {f*100:.1f}%)"
                )

        lines.append("--------------------------------")
        return "\n".join(lines)

    def open_dayslip(self):
        if not self.current_meeting_detail:
            messagebox.showinfo("Day slip", "Load a meeting (Fetch meetings → Load races) first.")
            return
        DaySlipDialog(self)
    
    

    def build_day_slip(self, bankroll: float, min_kelly_frac: float, use_tote_fair: bool):
        """
        Walk all races in the loaded meeting, fetch event + merge TAB odds, compute Kelly,
        and produce a plain-text 'receipt' for bets with Kelly >= min_kelly_frac.
        """
        try:
            mtgs = unwrap_meetings(self.current_meeting_detail)
            if not mtgs or not mtgs[0].get("races"):
                messagebox.showinfo("Day slip", "No races found in this meeting.")
                return
            races = mtgs[0]["races"]
            meetno = self.current_meetno()
            date_str = (self.date_var.get() or "").strip()
            if not (meetno and date_str):
                messagebox.showwarning("Day slip", "Need date and meeting number (load races first).")
                return

            lines = []
            lines.append(f"DAY SLIP — {date_str}   Meet #{meetno}")
            lines.append("Assumptions: Kelly 1×, fair P from Tote when available (else FX).")
            lines.append(f"Bankroll ${bankroll:,.2f} | Min Kelly {min_kelly_frac*100:.2f}%")
            lines.append("-" * 48)

            # walk races in number order
            races_sorted = sorted(races, key=lambda rc: (safe_int(race_number(rc)) or 0))
            any_picks = False

            for rc in races_sorted:
                rid   = rc.get("id") or ""
                rno   = safe_int(race_number(rc)) or "?"
                rname = (rc.get("name") or rc.get("description") or "").strip()
                start_utc = parse_start_to_utc(rc)
                start_nz  = fmt_dt_nz(start_utc)

                # fetch affiliates event
                try:
                    url = f"{AFF_BASE}/events/{rid}"
                    params = {"enc": "json"}
                    r = self.session.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
                    r.raise_for_status()
                    payload = r.json()
                except Exception as e:
                    # still print header; note failure
                    lines.append(f"Race #{rno} — {rname}  ({start_nz})")
                    lines.append(f"  ! failed to load event: {e}")
                    lines.append("-" * 32)
                    continue

                # merge TAB odds for the exact race
                try:
                    raceno = safe_int(race_number(rc))
                    markets = fetch_tab_odds(self.session, date_str, meetno, raceno)
                    prices_by_num = extract_prices_from_markets(markets)
                    merge_odds_into_event(payload, prices_by_num)
                except Exception:
                    pass  # non-fatal

                # compute metrics
                data = payload.get("data") or {}
                race_cond = (data.get("race") or {}).get("track_condition") or ""
                runners = extract_runners(payload)
                for r in runners:
                    r["metrics"] = compute_runner_metrics(r, race_cond)

                # select picks meeting Kelly threshold
                picks = []
                for r in runners:
                    m = r.get("metrics") or {}
                    k = m.get("kelly_pct1x")
                    O = m.get("win_fx")
                    if (k or 0) >= min_kelly_frac and O and O > 1.0:
                        # fair probability p used for EV calc
                        if use_tote_fair and (m.get("win_tote") and m["win_tote"] > 1.0):
                            p = 1.0 / m["win_tote"]
                        else:
                            p = 1.0 / O
                        stake = bankroll * float(k)
                        ev_profit = stake * (O * p - 1.0)
                        picks.append({
                            "no": r.get("no"),
                            "name": (r.get("name") or "").strip(),
                            "O": O,
                            "stake": stake,
                            "ev": ev_profit
                        })

                if picks:
                    any_picks = True
                    lines.append(f"Race #{rno} — {rname}  ({start_nz})")
                    for p in sorted(picks, key=lambda x: (-x["ev"], x["O"])):
                        lines.append(f"  #{p['no']} {p['name']}  @ ${p['O']:g}   bet @ ${p['stake']:.2f}")
                        lines.append(f"     expected profit ${p['ev']:.2f}")
                    lines.append("-" * 32)

            if not any_picks:
                lines.append("No selections met the Kelly threshold.")

            # show in a simple popup with copy/save
            slip = tk.Toplevel(self)
            slip.title("Day slip")
            slip.geometry("760x520")
            top = ttk.Frame(slip, padding=10); top.pack(fill="both", expand=True)
            txt = ScrolledText(top, wrap="word"); txt.pack(fill="both", expand=True)
            txt.insert("1.0", "\n".join(lines))
            txt.configure(state="normal")

            btns = ttk.Frame(slip, padding=(0,8,0,0)); btns.pack(fill="x")
            def _copy():
                try:
                    self.clipboard_clear(); self.clipboard_append(txt.get("1.0","end-1c"))
                    self.update_idletasks(); messagebox.showinfo("Copied", "Day slip copied to clipboard")
                except Exception as e:
                    messagebox.showerror("Copy", f"Failed: {e}")
            def _save():
                path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                    filetypes=[("Text files","*.txt"),("All files","*.*")],
                                                    title="Save day slip")
                if not path: return
                try:
                    with open(path,"w",encoding="utf-8") as f:
                        f.write(txt.get("1.0","end-1c"))
                    messagebox.showinfo("Saved", f"Saved to:\n{path}")
                except Exception as e:
                    messagebox.showerror("Save", f"Failed: {e}")
            ttk.Button(btns, text="Copy", command=_copy).pack(side="left")
            ttk.Button(btns, text="Save", command=_save).pack(side="left", padx=(8,0))
            ttk.Button(btns, text="Close", command=slip.destroy).pack(side="right")

        except Exception as e:
            messagebox.showerror("Day slip", f"Failed to build day slip:\n{e}")

    def prime_meeting_rows(self, idx=0):
        """Sequentially refresh each row so the UI stays responsive."""
        rows = self.tree.get_children()
        if idx >= len(rows):
            self.set_status("Primed all races.")
            return
        row = rows[idx]
        try:
            self.refresh_row_snapshot(row)
        except Exception:
            # swallow errors per-row; keep going
            pass
        finally:
            # small delay to avoid hammering the endpoint and freezing the UI
            self.after(150, lambda: self.prime_meeting_rows(idx + 1))


    def refresh_row_snapshot(self, row_iid):
        """
        Populate the meeting table row with:
        - entrants (count of non-scratched entries from TAB odds feed)
        - tote_avail (Yes/No if any tote win/place numbers present)
        - handle (sum of pool totals on the odds node)
        - t2j (recomputed minutes to jump from meeting detail)
        """
        date_str = (self.date_var.get() or "").strip()
        meetno = self.current_meetno()
        raceno = safe_int(self.tree.set(row_iid, "race_number"))
        if not (date_str and meetno and raceno):
            return

        # TAB odds race node
        race_node = fetch_tab_race_node(self.session, date_str, meetno, raceno)

        # Entrants / Tote? / Handle
        entrants, tote_avail, handle = summarize_tab_race(race_node)
        if entrants is not None:
            self.tree.set(row_iid, "entrants", str(entrants))
        self.tree.set(row_iid, "tote_avail", "Yes" if tote_avail else "No")
        if handle is not None:
            self.tree.set(row_iid, "handle", fmt_money(handle))

        # Recompute minutes to jump using the meeting detail race object (more reliable for times)
        try:
            mtgs = unwrap_meetings(self.current_meeting_detail)
            rc = None
            if mtgs and mtgs[0].get("races"):
                for _rc in mtgs[0]["races"]:
                    if (_rc.get("id") or "") == row_iid:
                        rc = _rc
                        break
            if rc:
                start_utc = parse_start_to_utc(rc)           # you already have this helper
                t2j = minutes_to_jump_utc(start_utc)         # and this one
                self.tree.set(row_iid, "t2j", t2j)
        except Exception:
            pass


    def selected_race_row(self):
        sel = self.tree.selection()
        return sel[0] if sel else None

    def current_meetno(self):
        if not self.current_meeting_detail:
            return None
        mtgs = unwrap_meetings(self.current_meeting_detail)
        if not mtgs:
            return None
        return mtgs[0].get("tote_meeting_number")

    # ---------- actions ----------
    def on_fetch_meetings(self):
        date_str = self.date_var.get().strip()
        if not date_str:
            messagebox.showwarning("Missing date", "Please enter a date in YYYY-MM-DD format.")
            return
        self.set_status(f"Fetching meetings for {date_str}...")
        try:
            url = f"{AFF_BASE}/meetings"
            params = {"date_from": date_str, "date_to": date_str, "enc": "json"}
            r = self.session.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            meetings = unwrap_meetings(r.json())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch meetings:\n{e}")
            self.set_status("Error fetching meetings")
            return

        if not meetings:
            messagebox.showinfo("No meetings", f"No meetings found on {date_str}")
            self.meeting_combo["values"] = []
            self.meeting_var.set("")
            self.set_status("No meetings found")
            return

        values = []
        self.meeting_display_to_id.clear()
        for m in meetings:
            mid = meeting_id(m)
            # Location/track name (fallbacks included)
            loc = m.get("name") or m.get("display_meeting_name") or m.get("venue") or "-"
            # Meet number (tote or generic)
            meetno = (
                m.get("tote_meeting_number")
                or m.get("number")
                or m.get("meet_no")
                or "-"
            )
            disp = f"{loc}, Meet# {meetno}, ID {mid or '-'}"
            values.append(disp)
            if mid:
                self.meeting_display_to_id[disp] = mid


        self.meeting_combo["values"] = values
        if values:
            self.meeting_combo.current(0)
        self.set_status(f"Loaded {len(values)} meetings")

    def on_load_races(self):
        disp = self.meeting_var.get().strip()
        if not disp:
            messagebox.showwarning("No meeting selected", "Please select a meeting first.")
            return
        mid = self.meeting_display_to_id.get(disp)
        if not mid:
            messagebox.showwarning("Missing ID", "Selected meeting has no UUID in the list. Pick another.")
            return

        self.set_status("Fetching meeting detail...")
        try:
            url = f"{AFF_BASE}/meetings/{mid}"
            params = {"enc": "json"}
            r = self.session.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            self.current_meeting_detail = r.json()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch races:\n{e}")
            self.set_status("Error fetching races")
            return

        mtgs = unwrap_meetings(self.current_meeting_detail)
        if not mtgs or not mtgs[0].get("races"):
            messagebox.showinfo("No races", "No races found for this meeting.")
            self.tree.delete(*self.tree.get_children())
            self.set_status("No races found")
            return

       # Populate table. Clear first.
        self.tree.delete(*self.tree.get_children())

        for rc in mtgs[0]["races"]:
            rid   = rc.get("id", "")
            num   = race_number(rc) or ""
            name  = rc.get("name", "")
            klass = rc.get("class", "") or rc.get("grade", "") or rc.get("race_class", "")

            # #Runners: prefer explicit, else estimate from gates/runner numbers
            entrants = safe_int(rc.get("field_size") or rc.get("entrant_count"))
            if entrants is None:
                entrants = estimate_runners(rc)
            entrants_disp = "" if entrants in (None, 0) else str(entrants)

            # Start (NZ) + time to jump
            start_utc = parse_start_to_utc(rc)           # -> aware UTC dt or None
            start_str = fmt_dt_nz(start_utc) if start_utc else "—"
            t2j_min   = minutes_to_jump_utc(start_utc) if start_utc else None
            t2j_disp  = "" if t2j_min is None else f"{t2j_min}m"

            # Tote availability / Handle are filled later (after odds/event fetch)
            self.tree.insert(
                "", "end", iid=rid,
                values=(num, name, klass, rid, start_str, entrants_disp, "", "", t2j_disp)
            )
            # Kick off a background prime to fill each race once
            if self.merge_odds.get():
                self.set_status("Priming races with odds & fields…")
                self.after(100, self.prime_meeting_rows)




    def on_tree_select(self, _event):
        pass

    def on_tree_double_click(self, _event):
        row = self.selected_race_row()
        if not row:
            return
        rid = self.tree.set(row, "race_id")
        self.race_id_var.set(rid)
        self.on_fetch_event()

    def copy_selected_id(self):
        row = self.selected_race_row()
        if not row:
            return
        rid = self.tree.set(row, "race_id")
        if not rid:
            return
        self.clipboard_clear()
        self.clipboard_append(rid)
        self.update_idletasks()
        self.set_status("Race ID copied")

    def build_event_params(self):
        params = {"enc": self.enc_var.get() or "json"}
        if self.with_big_bets.get(): params["with_big_bets"] = "true"
        if self.with_biggest_bet.get(): params["with_biggest_bet"] = "true"
        if self.with_live_bets.get(): params["with_live_bets"] = "true"
        if self.with_money_tracker.get(): params["with_money_tracker"] = "true"
        if self.with_tote_trends_data.get(): params["with_tote_trends_data"] = "true"
        if self.with_will_pays.get(): params["with_will_pays"] = "true"
        if self.present_overlay.get(): params["present_overlay"] = "true"
        btf = self.bet_type_filter.get().strip()
        if btf: params["bet_type_filter"] = btf
        return params

    def on_fetch_event(self):
        rid = self.race_id_var.get().strip()
        if not rid:
            messagebox.showwarning("Missing race ID", "Please paste or select a race ID.")
            return
        params = self.build_event_params()
        self.set_status("Fetching event...")
        try:
            url = f"{AFF_BASE}/events/{rid}"
            r = self.session.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json() if params.get("enc") == "json" else {"raw": r.text}
            self.current_event_payload = payload

            # Optionally merge TAB odds
            if params.get("enc") == "json" and self.merge_odds.get():
                date_str = self.date_var.get().strip()
                meetno = self.current_meetno()
                raceno = None
                row = self.selected_race_row()
                if row:
                    raceno = safe_int(self.tree.set(row, "race_number"))
                if date_str and meetno and raceno:
                    key = (date_str, meetno, raceno)
                    prices_by_num = self.odds_cache.get(key)
                    if prices_by_num is None:
                        markets = fetch_tab_odds(self.session, date_str, meetno, raceno)
                        prices_by_num = extract_prices_from_markets(markets)
                        self.odds_cache[key] = prices_by_num
                    merge_odds_into_event(payload, prices_by_num)

            # Show formatted viewer (with staking, signals, etc.)
            RaceViewer(self, self.current_event_payload,
                       title=f"Race {rid} — runners & odds")
            
            row = self.selected_race_row()

            if row:
                # #Runners
                entrants = estimate_runners_from_event(self.current_event_payload)
                if entrants is not None:
                    self.tree.set(row, "entrants", str(entrants))

                # Tote available? (bool-like string)
                self.tree.set(row, "tote_avail", "Yes" if tote_available_in_event(self.current_event_payload) else "No")

                # Handle ($ total pools)
                handle = extract_handle_from_event(self.current_event_payload)
                if handle is not None:
                    # use your fmt_money if you have one; else just format
                    self.tree.set(row, "handle", f"${handle:,.2f}")

            if row:
                handle = extract_handle_from_event(self.current_event_payload)
                if handle is not None:
                    self.tree.set(row, "handle", fmt_money(handle))

            self.set_status("Event loaded")
        except requests.HTTPError as e:
            body = ""
            try: body = e.response.text[:600]
            except Exception: pass
            messagebox.showerror("HTTP error", f"{e}\n\n{body}")
            self.set_status("HTTP error")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch event:\n{e}")
            self.set_status("Error")

    # ---------- odds refresh ----------
    def on_toggle_auto(self):
        if self.auto_refresh.get():
            self.schedule_auto_refresh()
        else:
            if self.auto_job:
                self.after_cancel(self.auto_job)
                self.auto_job = None
                self.set_status("Auto refresh off")

    def schedule_auto_refresh(self):
        if not self.auto_refresh.get():
            return
        self.on_refresh_odds(silent=True)
        delay = max(5, int(self.refresh_secs.get() or 10)) * 1000
        self.auto_job = self.after(delay, self.schedule_auto_refresh)

    def on_refresh_odds(self, silent=False):
        row = self.selected_race_row()
        if not row:
            if not silent:
                self.set_status("Select a race row to refresh")
            return
        try:
            self.refresh_row_snapshot(row)
            if not silent:
                self.set_status("Row refreshed")
        except Exception as e:
            if not silent:
                self.set_status(f"Odds refresh failed: {e}")


# ---------- run ----------
if __name__ == "__main__":
    app = RacingGUI()
    app.mainloop()
