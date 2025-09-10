#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from datetime import datetime, timezone
import tkinter as tk
from tkinter import ttk, messagebox

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np

# place this near the top, after importing pytorch_pre
import __main__
import pytorch_pre as pp
# add near the imports at top of NEW_racing_GUI.py
from recommend_picks_NN import model_win_table  # uses your existing script as a library  # :contentReference[oaicite:1]{index=1}
# === add with the other imports ===
import re

# === add after make_session / HTTP helpers ===
SCHED_BASE = "https://json.tab.co.nz/schedule"

def _pretty_track(s: str) -> str:
    if not s: return "-"
    # e.g. "SOFT5" -> "Soft 5"
    m = re.match(r"([A-Z]+)\s*(\d+)?", str(s).strip().upper())
    if not m: return str(s).title()
    word = m.group(1).title()
    num  = (" " + m.group(2)) if m.group(2) else ""
    return word + num

def _pretty_weather(s: str) -> str:
    return "-" if not s else str(s).replace("_"," ").title()

def fetch_meeting_context(session, date_str: str, meetno: int):
    """
    Pull the meeting card + races so we can show a header and build race buttons.
    Uses: https://json.tab.co.nz/schedule/{date}/{meet}/1
    Returns:
      {
        "date": ..., "meet": meetno,
        "venue": "...", "country": "...",
        "races": [1,2,...],
        "race_meta_by_no": { race_no: {"track": "...", "weather": "...", "id": "..."} }
      }
    """
    url = f"{SCHED_BASE}/{date_str}/{meetno}/1"
    r = session.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json() if isinstance(r.json(), dict) else {}

    mtgs = (data.get("meetings") or [])
    if not mtgs:
        return {"date": date_str, "meet": meetno, "venue": "", "country": "", "races": [], "race_meta_by_no": {}}

    m = mtgs[0]
    venue   = m.get("venue") or m.get("name") or ""
    country = (m.get("country") or "").upper()
    races   = []
    race_meta_by_no = {}
    for rc in (m.get("races") or []):
        try:
            rn = int(rc.get("number"))
            races.append(rn)
            race_meta_by_no[rn] = {
                "id": rc.get("id"),
                "track": _pretty_track(rc.get("track")),
                "weather": _pretty_weather(rc.get("weather")),
                "name": rc.get("name") or "",
            }
        except Exception:
            pass

    races.sort()
    return {
        "date": date_str,
        "meet": meetno,
        "venue": venue,
        "country": country,
        "races": races,
        "race_meta_by_no": race_meta_by_no,
    }


# Make sure pickle can resolve classes saved under __main__
if not hasattr(__main__, "PreprocessArtifacts") and hasattr(pp, "PreprocessArtifacts"):
    __main__.PreprocessArtifacts = pp.PreprocessArtifacts

# If you have other helper classes that were pickled, alias them too
for cls_name in ["FeatureBuilder", "ModelBundle", "Normalizer"]:
    if hasattr(pp, cls_name) and not hasattr(__main__, cls_name):
        setattr(__main__, cls_name, getattr(pp, cls_name))


# === Reuse your endpoints and headers (from the old GUI) ===
AFF_BASE  = "https://api.tab.co.nz/affiliates/v1/racing"
ODDS_BASE = "https://json.tab.co.nz/odds"
HEADERS = {
    "From": "r.cosgrove@hotmail.com",
    "X-Partner": "Personal use",
    "X-Partner-ID": "Personal use",
    "Accept": "application/json",
    "User-Agent": "RyanCosgrove/1.0",
}

TIMEOUT = 20

# === Model side: use your pytorch_pre helpers exactly like your CLI does ===
# We mirror the small helper from recommend_picks_NN so we can get per-runner p(win)
from pytorch_pre import load_model_and_predict, _scores_from_pred_rank, _pl_sample_order  # :contentReference[oaicite:0]{index=0}

def _nn_win_probs_for_df(race_df: pd.DataFrame, tau: float = 1.0, n_samples: int = 2000, seed: int = 42) -> np.ndarray:
    out = load_model_and_predict(race_df)
    # Prefer direct softmax from the new model if present
    p_win = np.asarray(out.get("p_win_softmax")) if isinstance(out, dict) else None
    if p_win is not None and p_win.shape[0] == len(race_df) and np.isfinite(p_win).any():
        s = p_win.sum()
        return p_win / (s if s > 0 else 1.0)

    # Fallback to pred_rank sampling, same as your recommend script
    pred_rank = np.asarray(out.get("pred_rank")) if isinstance(out, dict) else np.asarray(out)
    scores = _scores_from_pred_rank(pred_rank, tau=tau)
    rng = np.random.default_rng(seed)
    n = len(scores)
    pos_counts = np.zeros(n, dtype=np.int32)
    for _ in range(int(n_samples)):
        order = _pl_sample_order(scores, rng)
        pos_counts[order[0]] += 1
    return pos_counts / float(n_samples)

# === Minimal helpers lifted from your old GUI shapes ===  :contentReference[oaicite:1]{index=1}
def make_session():
    s = requests.Session()
    retry = Retry(
        total=3, backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _safe_float(x, default=None):
    try:
        f = float(x)
        return f if np.isfinite(f) else default
    except Exception:
        return default

def fetch_tab_race_node(session, date_str, meetno, raceno):
    """json.tab.co.nz/odds/{date}/{meet}/{race} -> race node with entries and often id."""
    url = f"{ODDS_BASE}/{date_str}/{meetno}/{raceno}"
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        return {}
    mtgs = data.get("meetings") or []
    for mtg in mtgs:
        for race in mtg.get("races", []) or []:
            if _safe_int(race.get("number")) == raceno:
                return race
        if mtg.get("races"):
            return mtg["races"][0]
    if data.get("races"):
        return data["races"][0]
    return {}

def extract_prices_from_tab_race(race_node):
    """number -> {win_fixed, place_fixed, win_tote, place_tote}"""
    out = {}
    for e in race_node.get("entries") or []:
        num = _safe_int(e.get("number") or e.get("runner") or e.get("runner_number"))
        if not num: 
            continue
        rec = out.setdefault(num, {})
        if e.get("ffwin") is not None:  rec["win_fixed"]   = e.get("ffwin")
        if e.get("ffplc") is not None:  rec["place_fixed"] = e.get("ffplc")
        if e.get("win")   is not None:  rec["win_tote"]    = e.get("win")
        if e.get("plc")   is not None:  rec["place_tote"]  = e.get("plc")
    return out

def fetch_aff_event(session, race_id: str):
    url = f"{AFF_BASE}/events/{race_id}"
    r = session.get(url, headers=HEADERS, params={"enc": "json"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def merge_prices_into_event(payload, prices_by_number):
    """Attach .prices to each runner, by runner_number."""
    data = payload.get("data") or {}
    runners = data.get("runners") or []
    for r in runners:
        num = _safe_int(r.get("runner_number") or r.get("number") or r.get("barrier"))
        if num and num in prices_by_number:
            r.setdefault("prices", {}).update(prices_by_number[num])

def _norm_name_for_join(s):
    if s is None: return ""
    s = str(s).strip().lower()
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    return " ".join(s.split())


def build_df_for_model(payload) -> pd.DataFrame:
    """Build the feature set expected by pytorch_pre from the Affiliates event + merged odds."""
    data = payload.get("data") or {}
    race = data.get("race") or {}
    runners = data.get("runners") or []

    # Race-level fields
    venue   = race.get("display_meeting_name") or race.get("meeting_name") or ""
    country = race.get("meeting_country") or race.get("country") or ""
    rnum    = race.get("race_number") or race.get("number")
    rnum_s  = race.get("race_number_sched") or race.get("number")  # best effort
    dist    = race.get("distance") or race.get("race_distance") or race.get("race_distance_m")
    try:
        race_distance_m = float(dist)
    except Exception:
        race_distance_m = None
    stake   = race.get("stake") or race.get("prize_money") or None
    rclass  = race.get("class") or race.get("race_class") or ""
    rclass_s= race.get("race_class_sched") or ""
    track   = race.get("track_condition") or ""
    weather = race.get("weather") or ""
    date_s  = race.get("race_date") or race.get("date") or race.get("start_time") or ""
    # normalise to yyyy-mm-dd
    try:
        if isinstance(date_s, (int, float)) or (isinstance(date_s, str) and date_s.isdigit()):
            from datetime import datetime, timezone
            date_iso = datetime.fromtimestamp(int(date_s), tz=timezone.utc).date().isoformat()
        else:
            from datetime import datetime
            s = str(date_s)
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            s = s.replace(" ", "T", 1)
            dt = datetime.fromisoformat(s)
            date_iso = (dt.date() if dt else None).isoformat()
    except Exception:
        date_iso = None

    # Compute fav_rank from fixed win odds if available
    # Lower fixed_win means stronger favourite
    fixed_list = []
    for r in runners:
        num = r.get("runner_number") or r.get("number")
        fx  = None
        # prefer live fixed from odds object, else merged prices
        odds_obj = r.get("odds") or {}
        prices   = r.get("prices") or {}
        if odds_obj.get("fixed_win") is not None:
            fx = odds_obj.get("fixed_win")
        elif prices.get("win_fixed") is not None:
            fx = prices.get("win_fixed")
        try:
            fx_val = float(fx) if fx is not None else None
        except Exception:
            fx_val = None
        fixed_list.append((num, fx_val))

    # Rank by fixed_win ascending, tie-safe
    fx_rank = {}
    sorted_fx = sorted([(n, f) for (n, f) in fixed_list if f is not None], key=lambda t: t[1])
    for i, (n, _) in enumerate(sorted_fx, start=1):
        try:
            fx_rank[int(n)] = i
        except Exception:
            pass

    rows = []
    for r in runners:
        num = r.get("runner_number") or r.get("number")
        try:
            num_i = int(num)
        except Exception:
            num_i = None

        barrier = r.get("barrier") or ""
        jockey  = r.get("jockey") or ""
        weight  = r.get("weight") or r.get("handicap_weight") or r.get("entrant_weight") or None
        try:
            entrant_weight = float(weight) if weight is not None else None
        except Exception:
            entrant_weight = None

        rows.append({
            # minimal id/date for grouping
            "date": date_iso,
            "race_id": race.get("race_id") or race.get("id") or "",

            # numeric
            "meeting_number": race.get("meeting_number") or None,
            "race_number": rnum,
            "race_distance_m": race_distance_m,
            "stake": stake,
            "fav_rank": fx_rank.get(num_i),            # 1 is fave
            "race_length": race_distance_m,            # fall back if your model used this synonym
            "race_number_sched": rnum_s,
            "entrant_weight": entrant_weight,

            # categoricals
            "race_class": rclass,
            "race_track": track,
            "race_weather": weather,
            "meeting_country": country,
            "meeting_venue": venue,
            "source_section": "affiliates",
            "race_class_sched": rclass_s,
            "entrant_barrier": str(barrier) if barrier is not None else "",
            "entrant_jockey": jockey,
            "runner_name": r.get("name") or "",

            # convenience for the GUI table
            "runner_number": num,
        })

    return pd.DataFrame(rows)

def _parse_start_to_utc(rc: dict):
    # tolerate epoch or ISO strings
    for k in ("advertised_start", "start_time", "tote_start_time"):
        ts = rc.get(k)
        if isinstance(ts, (int, float)) or (isinstance(ts, str) and ts.isdigit()):
            try:
                return datetime.fromtimestamp(int(ts), tz=timezone.utc)
            except Exception:
                pass
    s = rc.get("advertised_start_string") or rc.get("start_time") or rc.get("tote_start_time") or ""
    if isinstance(s, str) and s:
        try:
            if s.endswith("Z"): s = s[:-1] + "+00:00"
            s = s.replace(" ", "T", 1)
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None

def _fmt(n):
    return "" if n in (None, 0, 0.0) else f"{float(n):g}"

def _pct(p):
    return "" if p in (None, "", 0) else f"{100*float(p):.1f}"

def _imp_from_fixed(win_fx):
    try:
        fx = float(win_fx)
        return 1.0 / fx if fx > 1.0 else None
    except Exception:
        return None
    


# === Simple GUI ===
class SimpleRaceGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simple Racing Viewer")
        self.geometry("1100x720")
        self.configure(bg="#f6fbff")  # very soft light blue

        # Inputs row
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="x")

        today = datetime.now().strftime("%Y-%m-%d")
        ttk.Label(frm, text="Date (YYYY-MM-DD)").grid(row=0, column=0, sticky="w")
        self.date_var = tk.StringVar(value=today)
        ttk.Entry(frm, textvariable=self.date_var, width=14).grid(row=0, column=1, sticky="w", padx=(6, 16))

        ttk.Label(frm, text="Meet #").grid(row=0, column=2, sticky="w")
        self.meet_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.meet_var, width=6).grid(row=0, column=3, sticky="w", padx=(6, 16))

        ttk.Label(frm, text="Race #").grid(row=0, column=4, sticky="w")
        self.race_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.race_var, width=6).grid(row=0, column=5, sticky="w", padx=(6, 16))

        ttk.Button(frm, text="Fetch", command=self.fetch_and_show).grid(row=0, column=6, sticky="w")
        ttk.Button(frm, text="Recommend…", command=self.open_recommendations).grid(row=0, column=7, sticky="w", padx=(12, 0))


        # Header info
        self.header = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.header, font=("TkDefaultFont", 11, "bold"), padding=(10, 6)).pack(anchor="w")

        self.subheader = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.subheader, padding=(10, 0)).pack(anchor="w")

        # Table
        table_frame = ttk.Frame(self, padding=(10, 6))
        table_frame.pack(fill="both", expand=True)

        cols = ("no","name","barrier","jockey",
                "win_fx","win_tote","imp_win","model_win",
                "pl_fx","pl_tote","pace","edge")
        headers = ["#","Runner","Barrier","Jockey",
                   "WinFx","WinTote","ImpWin%","ModelWin%",
                   "PlFx","PlTote","Pace","Edge"]
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=20)
        for c, h in zip(cols, headers):
            self.tree.heading(c, text=h)
        # widths
        self.tree.column("no", width=50, anchor="center")
        self.tree.column("name", width=260, anchor="w")
        self.tree.column("barrier", width=70, anchor="center")
        self.tree.column("jockey", width=160, anchor="w")
        self.tree.column("win_fx", width=80, anchor="center")
        self.tree.column("win_tote", width=80, anchor="center")
        self.tree.column("imp_win", width=90, anchor="center")
        self.tree.column("model_win", width=90, anchor="center")
        self.tree.column("pl_fx", width=80, anchor="center")
        self.tree.column("pl_tote", width=80, anchor="center")
        self.tree.column("pace", width=120, anchor="w")
        self.tree.column("edge", width=140, anchor="w")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # at the top of __init__
        self._last_event = None
        self._last_recs_data = None

    def _get_recs_for_race(self, date, meet_no, race_no, settings_dict):
        """
        Return a fresh recs_pack for (date, meet_no, race_no) using the same
        model + blending pipeline you used to build self._last_recs_data.
        """
        # Reuse your existing pipeline; below is a typical shape:
        return build_recs_pack(
            date=date,
            meet_no=meet_no,
            race_no=race_no,
            bet_type=settings_dict["bet_type"],
            market=settings_dict["market"],
            blend=settings_dict["blend"],
            w=settings_dict["w"],
            overround=settings_dict["overround"],
            bankroll=settings_dict["bankroll"],
            kelly_frac=settings_dict["kelly_frac"],
            min_edge=settings_dict["min_edge"],
            min_kelly=settings_dict["min_kelly"],
            max_picks=settings_dict["max_picks"],
            # any other knobs your builder expects…
        )

    def open_recommendations(self):
        if not self._last_recs_data:
            messagebox.showinfo("Recommendations", "Fetch a race first.")
            return

        # ensure we have meeting context (venue, weather, all race numbers)
        if not getattr(self, "_last_context", None):
            try:
                sess = make_session()
                self._last_context = fetch_meeting_context(
                    sess,
                    self.date_var.get().strip(),
                    _safe_int(self.meet_var.get().strip()),
                    loader=self._get_recs_for_race
                )
            except Exception:
                self._last_context = None

        RecommendationWindow(
            self,
            self._last_recs_data,
            context=self._last_context,     # include {date, meet_no, meeting_name, country, track, weather, available_races}
            loader=self._get_recs_for_race  # you implement this to call your existing model+blend pipeline
        )





    def fetch_and_show(self):
        # Local import so the GUI can run even if CLI script changes
        from recommend_picks_NN import model_win_table

        def _norm_name_for_join(s):
            if s is None:
                return ""
            s = str(s).strip().lower()
            # keep letters, numbers and spaces, collapse whitespace
            s = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
            return " ".join(s.split())

        date_str = self.date_var.get().strip()
        meetno   = _safe_int(self.meet_var.get().strip())
        raceno   = _safe_int(self.race_var.get().strip())
        if not (date_str and meetno and raceno):
            messagebox.showwarning("Input", "Please enter date, meet and race numbers.")
            return

        sess = make_session()
        try:
            # 1) Pull odds payload and extract race id + prices
            race_node = fetch_tab_race_node(sess, date_str, meetno, raceno)
            if not race_node or not isinstance(race_node, dict):
                raise RuntimeError("No race node found for that date, meet, race")

            prices_by_num = extract_prices_from_tab_race(race_node)
            race_id = str(
                race_node.get("id")
                or race_node.get("race_id")
                or race_node.get("raceId")
                or ""
            ).strip()
            if not race_id:
                raise RuntimeError("Race id not present on odds payload")

            # 2) Full event from Affiliates, then merge odds into runners
            event = fetch_aff_event(sess, race_id)
            if not isinstance(event, dict) or not event.get("data"):
                raise RuntimeError("Affiliates event payload is empty")
            merge_prices_into_event(event, prices_by_num)

            # 3) Build model probs via your recommend_picks_NN, then prep maps for join
            try:
                mod_df = model_win_table(meetno, raceno, date_str)  # runner_name, runner_number, p_win, win_%, new_horse
            except Exception:
                mod_df = pd.DataFrame(columns=["runner_number","runner_name","p_win","win_%","new_horse"])

            if not mod_df.empty:
                mod_df = mod_df.copy()
                mod_df["__key_name"] = mod_df["runner_name"].map(_norm_name_for_join)
                name2p = {k: float(p) for k, p in zip(mod_df["__key_name"], mod_df["p_win"]) if isinstance(k, str) and k}
                num2p  = {}
                for n, p in zip(mod_df["runner_number"], mod_df["p_win"]):
                    try:
                        num2p[int(n)] = float(p)
                    except Exception:
                        pass
            else:
                name2p, num2p = {}, {}

            # 4) Header text
            data = event.get("data") or {}
            race = data.get("race") or {}
            venue = race.get("display_meeting_name") or race.get("meeting_name") or ""
            desc  = race.get("description") or race.get("race_name") or ""
            start = _parse_start_to_utc(race)
            start_local = start.astimezone() if start else None
            start_str = start_local.strftime("%Y-%m-%d %H:%M %Z") if start_local else ""
            weather = race.get("weather") or "-"
            track   = race.get("track_condition") or "-"
            distance = race.get("distance") or race.get("race_distance") or ""
            positions_paid = race.get("positions_paid") or "-"

            self.header.set(f"{venue} | R{raceno} {desc}")
            self.subheader.set(
                f"Start: {start_str} | Distance: {distance} | Track: {track} | Weather: {weather} | Positions paid: {positions_paid}"
            )

            # 5) Fill rows using model probs from recommend_picks_NN
            self.tree.delete(*self.tree.get_children())
            runners = data.get("runners") or []
            for r in runners:
                num = _safe_int(r.get("runner_number") or r.get("number"))
                name = r.get("name") or ""
                jockey = r.get("jockey") or ""
                barrier = r.get("barrier") or ""

                prices = r.get("prices") or {}
                odds_obj = r.get("odds") or {}

                win_fx   = odds_obj.get("fixed_win",  prices.get("win_fixed"))
                pl_fx    = odds_obj.get("fixed_place", prices.get("place_fixed"))
                win_tote = prices.get("win_tote")
                pl_tote  = prices.get("place_tote")

                # implied win from fixed odds
                imp = _imp_from_fixed(win_fx)

                # model p via normalised name, fallback to runner_number
                key = _norm_name_for_join(name)
                model_p = name2p.get(key)
                if model_p is None and num is not None:
                    model_p = num2p.get(num)

                # speedmap and edge tags
                pace = ""
                sm = r.get("speedmap")
                if isinstance(sm, dict):
                    pace = (sm.get("label") or "").strip()

                edge_tags = []
                fi = r.get("form_indicators")
                if isinstance(fi, list):
                    for ind in fi:
                        grp = (ind.get("group") or "").strip()
                        nm  = (ind.get("name") or "").strip().lower()
                        neg = bool(ind.get("negative"))
                        if grp in {"Track_Distance", "Course_Distance"} and not neg:
                            edge_tags.append("TD+")
                        if grp == "Track" and not neg:
                            edge_tags.append("T+")
                        if grp == "Distance" and not neg:
                            edge_tags.append("D+")
                        if "hat-trick" in nm and not neg:
                            edge_tags.append("HTR")
                seen = set()
                edge = " ".join([t for t in edge_tags if not (t in seen or seen.add(t))])

                self.tree.insert(
                    "",
                    "end",
                    values=(
                        num or "",
                        name,
                        barrier or "",
                        jockey,
                        _fmt(win_fx),
                        _fmt(win_tote),
                        _pct(imp),
                        _pct(model_p),
                        _fmt(pl_fx),
                        _fmt(pl_tote),
                        pace,
                        edge,
                    ),
                )

            # cache once, after the table is built
            self._last_event = event
            # cache meeting context for race selector
            try:
                self._last_context = fetch_meeting_context(sess, date_str, meetno)
            except Exception:
                self._last_context = None

            rows_for_recs = []
            for r in runners:
                num = _safe_int(r.get("runner_number") or r.get("number"))
                name = r.get("name") or ""
                prices = r.get("prices") or {}
                odds_obj = r.get("odds") or {}

                win_fx   = odds_obj.get("fixed_win",  prices.get("win_fixed"))
                pl_fx    = odds_obj.get("fixed_place", prices.get("place_fixed"))
                win_tote = prices.get("win_tote")
                pl_tote  = prices.get("place_tote")

                imp_win = _imp_from_fixed(win_fx)
                key = _norm_name_for_join(name)
                model_p = name2p.get(key)
                if model_p is None and num is not None:
                    model_p = num2p.get(num)

                rows_for_recs.append({
                    "runner_number": num,
                    "runner_name": name,
                    "model_p": model_p if model_p is not None else 0.0,
                    "imp_p_fixed": imp_win if imp_win is not None else None,
                    "fixed_win": float(win_fx) if win_fx else None,
                    "fixed_place": float(pl_fx) if pl_fx else None,
                    "tote_win": float(win_tote) if win_tote else None,
                    "tote_place": float(pl_tote) if pl_tote else None,
                })

            self._last_recs_data = {
                "race_meta": {
                    "positions_paid": race.get("positions_paid") or 3,
                    "field_size": len(runners),
                },
                "rows": rows_for_recs,
            }


        except Exception as e:
            messagebox.showerror("Error", str(e))


class RecommendationWindow(tk.Toplevel):
    """
    A self-contained window that renders model/market-blended recommendations
    and lets you switch between races of the same meet via a tab-like button bar.

    Required constructor args:
        parent:  Tk root or another window
        recs_pack: dict of data to render (see _pack_to_rows for flexible schema)
        context:   dict with at least {date, meet_no, meeting_name, country, track, weather, available_races?}
        loader:    callable(date, meet_no, race_no, settings_dict) -> recs_pack (same shape as initial)
    """

    # ----------------------------
    # Construction / UI skeleton
    # ----------------------------
    def __init__(self, parent, recs_pack, context=None, loader=None):
        super().__init__(parent)
        self.title("Value Recommendations")
        self.configure(padx=8, pady=8)
        self.parent = parent

        # Core state
        self.recs_pack = recs_pack or {}
        self.context = context or {}
        self._loader = loader

        # Meta
        rm = (self.recs_pack.get("race_meta") or {})
        self.date = self.context.get("date") or rm.get("date")
        self.meet_no = int(self.context.get("meet_no") or rm.get("meet_no") or rm.get("meeting_number") or 0)
        self.meeting_name = self.context.get("meeting_name") or rm.get("meeting_name") or self.recs_pack.get("meeting_name") or ""
        self.country = (self.context.get("country") or rm.get("country") or "").upper() or "AUS"
        self.track = self.context.get("track") or rm.get("track") or ""
        self.weather = self.context.get("weather") or rm.get("weather") or ""
        self.current_race_no = int(rm.get("race_no") or self.context.get("race_no") or 1)

        # Available races
        avail = (
            self.context.get("available_races")
            or self.recs_pack.get("available_races")
            or list(range(1, int(rm.get("total_races") or 1) + 1))
        )
        self.available_races = [int(x) for x in avail] if avail else [self.current_race_no]

        # ----------------------------
        # Top header (meet line)
        # ----------------------------
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        meet_label_text = f"M{self.meet_no} {self.meeting_name} ({self.country}) —  Track: {self.track}   Weather: {self.weather}"
        self.meet_label = ttk.Label(header, text=meet_label_text, font=("SF Pro Display", 14, "bold"))
        self.meet_label.grid(row=0, column=0, sticky="w")

        # ----------------------------
        # Race selector (R1…RN)
        # ----------------------------
        self._init_race_styles()
        self.race_bar = ttk.Frame(self)
        self.race_bar.grid(row=1, column=0, sticky="w", pady=(6, 8))
        self._race_buttons = []
        self._build_race_buttons()

        # ----------------------------
        # Control bar (your knobs)
        # ----------------------------
        ctrl = ttk.Frame(self)
        ctrl.grid(row=2, column=0, sticky="ew")
        for c in range(12):
            ctrl.columnconfigure(c, weight=0)
        ctrl.columnconfigure(11, weight=1)

        # Variables (default from recs_pack.settings if present)
        settings = self.recs_pack.get("settings") or {}
        def _get(k, d):  # helper with safe defaults
            return settings.get(k, d)

        # Left block
        ttk.Label(ctrl, text="Bet type").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.bet_type = tk.StringVar(value=_get("bet_type", "place"))
        ttk.Combobox(ctrl, textvariable=self.bet_type, values=["win", "place"], width=8, state="readonly").grid(row=1, column=0, sticky="w", padx=(0, 12))

        ttk.Label(ctrl, text="Market").grid(row=0, column=1, sticky="w")
        self.win_source = tk.StringVar(value=_get("market", "fixed"))
        ttk.Combobox(ctrl, textvariable=self.win_source, values=["fixed", "tote"], width=8, state="readonly").grid(row=1, column=1, sticky="w", padx=(0, 12))

        ttk.Label(ctrl, text="Overround").grid(row=0, column=2, sticky="w")
        self.or_method = tk.StringVar(value=_get("overround", "proportional"))
        ttk.Combobox(ctrl, textvariable=self.or_method, values=["none", "proportional"], width=13, state="readonly").grid(row=1, column=2, sticky="w", padx=(0, 12))

        ttk.Label(ctrl, text="Blend mode").grid(row=0, column=3, sticky="w")
        self.blend_mode = tk.StringVar(value=_get("blend", "logit"))
        ttk.Combobox(ctrl, textvariable=self.blend_mode, values=["logit", "linear"], width=10, state="readonly").grid(row=1, column=3, sticky="w", padx=(0, 12))

        ttk.Label(ctrl, text="Model weight w").grid(row=0, column=4, sticky="w")
        self.model_w = tk.DoubleVar(value=float(_get("w", 0.2)))
        ttk.Entry(ctrl, textvariable=self.model_w, width=8).grid(row=1, column=4, sticky="w", padx=(0, 12))

        ttk.Label(ctrl, text="Bankroll").grid(row=0, column=5, sticky="w")
        self.bankroll = tk.DoubleVar(value=float(_get("bankroll", 100.0)))
        ttk.Entry(ctrl, textvariable=self.bankroll, width=10).grid(row=1, column=5, sticky="w", padx=(0, 12))

        ttk.Label(ctrl, text="Kelly fraction").grid(row=0, column=6, sticky="w")
        self.kelly_frac = tk.DoubleVar(value=float(_get("kelly_frac", 0.25)))
        ttk.Entry(ctrl, textvariable=self.kelly_frac, width=8).grid(row=1, column=6, sticky="w", padx=(0, 12))

        ttk.Label(ctrl, text="Min edge %").grid(row=0, column=7, sticky="w")
        self.min_edge_pct = tk.DoubleVar(value=float(_get("min_edge", 0.0)))
        ttk.Entry(ctrl, textvariable=self.min_edge_pct, width=8).grid(row=1, column=7, sticky="w", padx=(0, 12))

        ttk.Label(ctrl, text="Min Kelly %").grid(row=0, column=8, sticky="w")
        self.min_kelly_pct = tk.DoubleVar(value=float(_get("min_kelly", 0.0)))
        ttk.Entry(ctrl, textvariable=self.min_kelly_pct, width=8).grid(row=1, column=8, sticky="w", padx=(0, 12))

        ttk.Label(ctrl, text="Max picks (0=all)").grid(row=0, column=9, sticky="w")
        self.max_picks = tk.IntVar(value=int(_get("max_picks", 0)))
        ttk.Entry(ctrl, textvariable=self.max_picks, width=8).grid(row=1, column=9, sticky="w", padx=(0, 12))

        ttk.Button(ctrl, text="Refresh", command=self.refresh).grid(row=1, column=10, sticky="e")
        # Spacer
        ttk.Label(ctrl, text="").grid(row=1, column=11, sticky="ew")

        # ----------------------------
        # Table
        # ----------------------------
        self.tree = ttk.Treeview(self, show="headings", height=20)
        self.tree.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        self.rowconfigure(3, weight=1)
        self.columnconfigure(0, weight=1)

        columns = ("num", "runner", "market", "odds", "mktpct", "modelpct", "blendpct", "edgepct", "ev", "fair")
        self.tree["columns"] = columns

        self.tree.heading("num", text="#")
        self.tree.heading("runner", text="Runner")
        self.tree.heading("market", text="Market")
        self.tree.heading("odds", text="Odds")
        self.tree.heading("mktpct", text="Market%")
        self.tree.heading("modelpct", text="Model%")
        self.tree.heading("blendpct", text="Blend%")
        self.tree.heading("edgepct", text="Edge%")
        self.tree.heading("ev", text="EV")
        self.tree.heading("fair", text="Fair")

        # Column widths
        self.tree.column("num", width=50, anchor="center")
        self.tree.column("runner", width=320, anchor="w")
        self.tree.column("market", width=90, anchor="center")
        self.tree.column("odds", width=80, anchor="e")
        self.tree.column("mktpct", width=90, anchor="e")
        self.tree.column("modelpct", width=90, anchor="e")
        self.tree.column("blendpct", width=90, anchor="e")
        self.tree.column("edgepct", width=80, anchor="e")
        self.tree.column("ev", width=80, anchor="e")
        self.tree.column("fair", width=110, anchor="e")

        # Load initial rows
        self._reload_from_pack()

        # Nice look
        self._apply_theme_tweaks()

    # ----------------------------
    # Style helpers
    # ----------------------------
    def _init_race_styles(self):
        st = ttk.Style(self)
        base_font = ("SF Pro Text", 11)
        st.configure("Race.TButton", padding=(12, 4), font=base_font)
        st.configure("RaceSelected.TButton", padding=(12, 4), font=base_font)

    def _apply_theme_tweaks(self):
        st = ttk.Style(self)
        st.configure("Treeview", rowheight=26, font=("SF Pro Text", 12))
        st.configure("Treeview.Heading", font=("SF Pro Text", 12, "bold"))

    # ----------------------------
    # Race buttons
    # ----------------------------
    def _build_race_buttons(self):
        # Clear old
        for b in getattr(self, "_race_buttons", []):
            b.destroy()
        self._race_buttons = []

        for i, rno in enumerate(self.available_races):
            rno = int(rno)
            selected = (rno == int(self.current_race_no))
            style = "RaceSelected.TButton" if selected else "Race.TButton"
            btn = ttk.Button(self.race_bar, text=f"R{rno}", style=style,
                             command=(lambda rr=rno: self.on_change_race(rr)))
            if selected:
                btn.state(["disabled"])
            btn.grid(row=0, column=i, padx=(0 if i == 0 else 8, 0))
            self._race_buttons.append(btn)

    def on_change_race(self, race_no: int):
        """Switch to another race at the same meet/date, preserving settings."""
        if not callable(self._loader):
            messagebox.showwarning("No loader", "No loader callback was provided.")
            return
        try:
            race_no = int(race_no)
        except Exception:
            return

        # Fetch
        try:
            new_pack = self._loader(self.date, int(self.meet_no), race_no, self._current_settings())
        except Exception as e:
            messagebox.showerror("Fetch error", f"Could not load R{race_no} (M{self.meet_no} {self.date}).\n\n{e}")
            return

        # Update state
        self.recs_pack = new_pack or {}
        self.current_race_no = race_no

        # Prefer new available_races if provided
        avail = self.recs_pack.get("available_races")
        if avail:
            self.available_races = [int(x) for x in avail]

        # Redraw
        self._build_race_buttons()
        self._reload_from_pack()

    # ----------------------------
    # Controls -> settings sync
    # ----------------------------
    def _current_settings(self):
        # Snapshot all UI knobs so switching race preserves everything
        return dict(
            bet_type=(self.bet_type.get() or "place").lower(),
            market=(self.win_source.get() or "fixed").lower(),
            overround=(self.or_method.get() or "proportional").lower(),
            blend=(self.blend_mode.get() or "logit").lower(),
            w=float(self.model_w.get() or 0.2),
            bankroll=float(self.bankroll.get() or 100.0),
            kelly_frac=float(self.kelly_frac.get() or 0.25),
            min_edge=float(self.min_edge_pct.get() or 0.0),
            min_kelly=float(self.min_kelly_pct.get() or 0.0),
            max_picks=int(self.max_picks.get() or 0),
            race_no=int(self.current_race_no),
            meet_no=int(self.meet_no),
            date=self.date,
        )

    # ----------------------------
    # Refresh button action
    # ----------------------------
    def refresh(self):
        """Re-run the loader for the currently selected race with current knobs."""
        if not callable(self._loader):
            # If there's no loader, just re-render the existing pack (e.g. after edits)
            self._reload_from_pack()
            return
        try:
            new_pack = self._loader(self.date, int(self.meet_no), int(self.current_race_no), self._current_settings())
        except Exception as e:
            messagebox.showerror("Refresh error", f"Could not refresh R{self.current_race_no}.\n\n{e}")
            return
        self.recs_pack = new_pack or {}
        self._reload_from_pack()

    # ----------------------------
    # Render data
    # ----------------------------
    def _reload_from_pack(self):
        # Update header line if meta changed
        rm = (self.recs_pack.get("race_meta") or {})
        if rm:
            self.track = rm.get("track", self.track)
            self.weather = rm.get("weather", self.weather)
            self.meeting_name = rm.get("meeting_name", self.meeting_name)
            self.country = (rm.get("country", self.country) or "").upper() or self.country
            if rm.get("race_no"):
                self.current_race_no = int(rm.get("race_no"))
        self.meet_label.config(text=f"M{self.meet_no} {self.meeting_name} ({self.country}) —  Track: {self.track}   Weather: {self.weather}")

        # Table
        for item in self.tree.get_children():
            self.tree.delete(item)

        rows = self._pack_to_rows(self.recs_pack)
        for r in rows:
            self.tree.insert(
                "", "end",
                values=(
                    r.get("number", ""),
                    r.get("runner", ""),
                    r.get("market_label", r.get("market", "")),
                    self._fmt_odds(r.get("odds")),
                    self._fmt_pct(r.get("market_pct")),
                    self._fmt_pct(r.get("model_pct")),
                    self._fmt_pct(r.get("blend_pct")),
                    self._fmt_pct(r.get("edge_pct")),
                    self._fmt_float(r.get("ev")),
                    self._fmt_odds(r.get("fair")),
                )
            )

    # ----------------------------
    # Data normalization helpers
    # ----------------------------
    def _pack_to_rows(self, pack: dict):
        """
        Accepts flexible shapes. Expected keys (preferred first, fallbacks in []):
            rows: list of {
                number, runner, market_label[market],
                odds,
                market_pct[market_prob, mkt_pct, mkt_prob],
                model_pct[model_prob, mdl_pct],
                blend_pct[blend_prob, bln_pct],
                edge_pct[edge],
                ev,
                fair[fair_odds]
            }
        """
        rows = pack.get("rows") or pack.get("data") or []
        norm = []
        for it in rows:
            norm.append(dict(
                number=it.get("number") or it.get("no") or it.get("runner_no"),
                runner=it.get("runner") or it.get("name"),
                market_label=it.get("market_label") or it.get("market") or ("Fixed P" if (pack.get("settings") or {}).get("market","fixed")=="fixed" else "Tote P"),
                odds=it.get("odds") or it.get("price"),
                market_pct=(
                    it.get("market_pct") or it.get("market_prob") or it.get("mkt_pct") or it.get("mkt_prob")
                ),
                model_pct=(
                    it.get("model_pct") or it.get("model_prob") or it.get("mdl_pct")
                ),
                blend_pct=(
                    it.get("blend_pct") or it.get("blend_prob") or it.get("bln_pct")
                ),
                edge_pct=it.get("edge_pct") or it.get("edge"),
                ev=it.get("ev") or it.get("expected_value"),
                fair=it.get("fair") or it.get("fair_odds"),
            ))
        return norm

    @staticmethod
    def _fmt_pct(x):
        try:
            return f"{float(x)*100:.1f}"
        except Exception:
            try:
                return f"{float(x):.1f}"
            except Exception:
                return ""

    @staticmethod
    def _fmt_odds(x):
        try:
            return f"{float(x):g}"
        except Exception:
            return ""

    @staticmethod
    def _fmt_float(x):
        try:
            return f"{float(x):.2f}"
        except Exception:
            return ""

class DayPicksWindow(tk.Toplevel):
    """Pop up that prints day picks grouped by meet, then race.
    Uses the same blending, overround and thresholds as the Recommendation window.
    """
    SCHED_URL = "https://json.tab.co.nz/schedule/{day}"
    ODDS_URL  = "https://json.tab.co.nz/odds/{day}/{meet}/{race}"

    def __init__(self, parent, date_str, settings):
        super().__init__(parent)
        self.parent = parent
        self.title(f"Day picks for {date_str}")
        self.geometry("1024x720")
        self.configure(bg="#f6fbff")
        self.date_str = date_str
        self.settings = settings

        # Controls
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text=f"Date: {date_str}").pack(side="left")
        ttk.Button(top, text="Refresh", command=self._run).pack(side="right")



        # Text area
        body = ttk.Frame(self, padding=6)
        body.pack(fill="both", expand=True)
        self.txt = tk.Text(body, wrap="word", font=("Menlo", 11))
        y = ttk.Scrollbar(body, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=y.set)
        self.txt.pack(side="left", fill="both", expand=True)
        y.pack(side="right", fill="y")

        self._run()



    # ---------- HTTP helpers ----------
    def _make_session(self):
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        s = requests.Session()
        retry = Retry(total=4, backoff_factor=0.5,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET"], raise_on_status=False)
        s.mount("https://", HTTPAdapter(max_retries=retry))
        return s

    def _get_json(self, s, url):
        try:
            r = s.get(url, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    # ---------- maths reused from RecommendationWindow ----------
    @staticmethod
    def _imp(od):
        try:
            od = float(od)
            return 1.0/od if od and od > 1.0 else None
        except Exception:
            return None

    @staticmethod
    def _deoverround_proportional(p):
        s = np.nansum(p)
        return p/s if s and np.isfinite(s) and s > 0 else p

    @staticmethod
    def _deoverround_power(p, alpha=0.9):
        q = np.power(np.clip(p, 1e-12, 1.0), alpha)
        s = np.nansum(q)
        return q/s if s and s > 0 else p

    @staticmethod
    def _logit(x):
        x = np.clip(x, 1e-12, 1-1e-12)
        return np.log(x/(1-x))

    @staticmethod
    def _sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))

    @staticmethod
    def _kelly_pct(p, odds_dec):
        try:
            if p is None or odds_dec is None or odds_dec <= 1.0:
                return 0.0
            b = float(odds_dec) - 1.0
            q = 1.0 - float(p)
            k = (b*float(p) - q) / b
            return max(0.0, 100.0*k)
        except Exception:
            return 0.0

    @staticmethod
    def _positions_paid(field_size):
        return 2 if field_size <= 7 else 3

    # ---------- odds parsing ----------
    def _extract_prices(self, race_node):
        out = {}
        for e in (race_node.get("entries") or []):
            try:
                num = int(e.get("number") or e.get("runner") or e.get("runner_number"))
            except Exception:
                continue
            rec = out.setdefault(num, {})
            if e.get("ffwin") is not None: rec["win_fixed"]   = e.get("ffwin")
            if e.get("ffplc") is not None: rec["place_fixed"] = e.get("ffplc")
            if e.get("win")   is not None: rec["win_tote"]    = e.get("win")
            if e.get("plc")   is not None: rec["place_tote"]  = e.get("plc")
            name = e.get("name") or e.get("runnerName") or e.get("runner_name")
            if name:
                rec["runner_name"] = name
        return out

    # ---------- model probs ----------
    def _model_probs_from_entries(self, venue, rnum, entries_map):
        # Minimal DF like your GUI uses for the NN
        rows = []
        for num, p in entries_map.items():
            rows.append({
                "runner_name": p.get("runner_name", ""),
                "runner_number": num,
                "meeting_venue": venue,
                "race_number": rnum,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return {}
        try:
            out = load_model_and_predict(df)
            p = np.asarray(out.get("p_win_softmax"), dtype=float)
            if p.shape[0] != len(df) or not np.isfinite(p).any():
                # fallback softmax around pred_rank if present
                if isinstance(out, dict) and "pred_rank" in out:
                    r = -np.asarray(out["pred_rank"], dtype=float)
                    r = r - np.max(r)
                    ex = np.exp(r)
                    p = ex / np.sum(ex)
                else:
                    p = np.ones(len(df)) / float(len(df))
        except Exception:
            p = np.ones(len(df)) / float(len(df))
        res = {}
        for i, num in enumerate(df["runner_number"].tolist()):
            res[int(num)] = float(p[i])
        return res

    # ---------- core run ----------
    def _run(self):
        s = self._make_session()
        sched = self._get_json(s, self.SCHED_URL.format(day=self.date_str)) or {}
        meetings = [m for m in sched.get("meetings") or []
                    if str(m.get("domain") or "").upper()=="HORSE"
                    and str(m.get("event_type") or "").upper()=="G"]
        meetings.sort(key=lambda m: int(m.get("number") or 0))

        lines = []
        total_recs = 0

        # unpack settings
        bet_type   = self.settings.get("bet_type", "win")
        market_src = self.settings.get("market", "fixed")
        overround  = self.settings.get("overround", "proportional")
        blend      = self.settings.get("blend", "logit")
        w          = float(self.settings.get("w", 0.5))
        min_edge   = float(self.settings.get("min_edge", 1.0))
        min_kelly  = float(self.settings.get("min_kelly", 0.3))
        kelly_frac = float(self.settings.get("kelly_frac", 0.25))
        bankroll   = float(self.settings.get("bankroll", 1000.0))
        max_picks  = int(self.settings.get("max_picks", 0))

        for m in meetings:
            meet_no = int(m.get("number") or 0)
            venue   = m.get("venue") or m.get("meetingName") or ""
            races   = m.get("races") or []
            races.sort(key=lambda r: int(r.get("number") or 0))
            lines.append(f"\n== {venue} (meet {meet_no}) ==\n")

            for r in races:
                race_no = int(r.get("number") or 0)
                race_node = self._get_json(s, self.ODDS_URL.format(day=self.date_str, meet=meet_no, race=race_no)) or {}
                # find the sub-node with entries
                sub = None
                for mm in race_node.get("meetings") or []:
                    for rr in mm.get("races") or []:
                        if int(rr.get("number") or 0) == race_no:
                            sub = rr; break
                    if sub: break
                if not sub and race_node.get("races"):
                    for rr in race_node.get("races") or []:
                        if int(rr.get("number") or 0) == race_no:
                            sub = rr; break
                if not sub:
                    continue

                prices = self._extract_prices(sub)
                if not prices:
                    continue

                # model probs keyed by runner_number
                p_mod_map = self._model_probs_from_entries(venue, race_no, prices)

                # arrays in runner order
                order_nums = sorted(prices.keys())
                odds_win = []
                odds_pl  = []
                p_mkt_win = []
                for num in order_nums:
                    rec = prices[num]
                    od_w = rec.get("win_fixed") if market_src == "fixed" else rec.get("win_tote")
                    od_p = rec.get("place_fixed") if market_src == "fixed" else rec.get("place_tote")
                    odds_win.append(float(od_w) if od_w is not None else np.nan)
                    odds_pl.append(float(od_p) if od_p is not None else np.nan)
                    p_mkt_win.append(self._imp(od_w))
                odds_win = np.array(odds_win, dtype=float)
                odds_pl  = np.array(odds_pl, dtype=float)
                p_mkt_win = np.array(p_mkt_win, dtype=float)

                # deoverround market win probs
                if overround == "proportional":
                    p_mkt_win = self._deoverround_proportional(p_mkt_win)
                elif overround.startswith("power"):
                    p_mkt_win = self._deoverround_power(p_mkt_win, alpha=0.9)

                # model win and place probs
                p_mod_win = np.array([p_mod_map.get(int(n), 0.0) for n in order_nums], dtype=float)
                p_mod_win = np.clip(p_mod_win, 1e-6, 1-1e-6)
                positions_paid = self._positions_paid(len(order_nums))
                p_mod_pl = np.clip(p_mod_win * positions_paid, 1e-6, 1-1e-6)

                # market place prob if we have place odds
                p_mkt_pl_raw = np.array([self._imp(x) for x in odds_pl], dtype=float)
                if overround == "proportional":
                    p_mkt_pl = self._deoverround_proportional(p_mkt_pl_raw) if np.isfinite(p_mkt_pl_raw).any() else p_mkt_pl_raw
                elif overround.startswith("power"):
                    p_mkt_pl = self._deoverround_power(p_mkt_pl_raw, alpha=0.9) if np.isfinite(p_mkt_pl_raw).any() else p_mkt_pl_raw
                else:
                    p_mkt_pl = p_mkt_pl_raw

                # blend
                if blend == "linear":
                    p_bl_win = w*p_mod_win + (1.0-w)*p_mkt_win
                    p_bl_pl  = w*p_mod_pl  + (1.0-w)*p_mkt_pl
                else:
                    p_bl_win = self._sigmoid(w*self._logit(p_mod_win) + (1.0-w)*self._logit(p_mkt_win))
                    if np.isfinite(p_mkt_pl).any():
                        p_bl_pl = self._sigmoid(w*self._logit(p_mod_pl) + (1.0-w)*self._logit(p_mkt_pl))
                    else:
                        p_bl_pl = p_mod_pl

                # choose picks
                idxs = list(range(len(order_nums)))
                idxs.sort(key=lambda i: -p_bl_win[i])  # order by win blend
                picks = []
                for i in idxs:
                    od_w = odds_win[i]
                    od_p = odds_pl[i]
                    imp_w = self._imp(od_w)
                    imp_p = self._imp(od_p)
                    edge_w = 100.0*((p_bl_win[i] - (imp_w or 0.0)) if imp_w is not None else 0.0)
                    edge_p = 100.0*((p_bl_pl[i]  - (imp_p or 0.0)) if imp_p is not None else 0.0)
                    k_w = self._kelly_pct(p_bl_win[i], od_w) if np.isfinite(od_w) else 0.0
                    k_p = self._kelly_pct(p_bl_pl[i],  od_p) if np.isfinite(od_p) else 0.0

                    pass_w = np.isfinite(od_w) and od_w > 1.0 and edge_w >= min_edge and k_w >= min_kelly
                    pass_p = np.isfinite(od_p) and od_p > 1.0 and edge_p >= min_edge and k_p >= min_kelly
                    passed = pass_w if bet_type == "win" else pass_p if bet_type == "place" else (pass_w or pass_p)
                    if not passed:
                        continue

                    picks.append({
                        "idx": i,
                        "num": order_nums[i],
                        "name": prices[order_nums[i]].get("runner_name", f"#{order_nums[i]}"),
                        "odds_win": od_w, "odds_pl": od_p,
                        "p_bl_win": float(p_bl_win[i]), "p_bl_pl": float(p_bl_pl[i]),
                        "kelly_win": float(k_w), "kelly_pl": float(k_p),
                        "edge_win": float(edge_w), "edge_pl": float(edge_p),
                    })
                    if max_picks and len(picks) >= max_picks:
                        break

                # write race block
                lines.append(f"Race {race_no}")
                if not picks:
                    lines.append("  no bet")
                else:
                    for p in picks:
                        if bet_type == "win":
                            stake = bankroll * (p["kelly_win"]/100.0) * kelly_frac
                            lines.append(
                                f"  WIN  ${stake:.2f} on #{p['num']} {p['name']} at {market_src} {p['odds_win']}  "
                                f"blend {p['p_bl_win']*100:.1f}% edge {p['edge_win']:.1f}% kelly {p['kelly_win']:.1f}%"
                            )
                        elif bet_type == "place":
                            stake = bankroll * (p["kelly_pl"]/100.0) * kelly_frac
                            lines.append(
                                f"  PLACE  ${stake:.2f} on #{p['num']} {p['name']} at {market_src} {p['odds_pl']}  "
                                f"blend {p['p_bl_pl']*100:.1f}% edge {p['edge_pl']:.1f}% kelly {p['kelly_pl']:.1f}%"
                            )
                        else:
                            stake_w = bankroll * (p["kelly_win"]/100.0) * (kelly_frac*0.5)
                            stake_p = bankroll * (p["kelly_pl"]/100.0) * (kelly_frac*0.5)
                            lines.append(
                                f"  EACH WAY  W ${stake_w:.2f} at {market_src} {p['odds_win']} | P ${stake_p:.2f} at {market_src} {p['odds_pl']} "
                                f"on #{p['num']} {p['name']}  (blend W {p['p_bl_win']*100:.1f}% P {p['p_bl_pl']*100:.1f}%)"
                            )
                total_recs += len(picks)

        # display
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, "\n".join(lines))
        self.txt.insert(tk.END, f"\n\nTotal picks: {total_recs}\n")
        self.txt.see(tk.END)


# ---- hook this into RecommendationWindow ----


if __name__ == "__main__":
    app = SimpleRaceGUI()
    app.mainloop()
