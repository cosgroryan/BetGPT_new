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

    def open_recommendations(self):
        if not self._last_recs_data:
            messagebox.showinfo("Recommendations", "Fetch a race first.")
            return
        RecommendationWindow(self, self._last_recs_data)



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

                # cache for recommender popout
                self._last_event = event
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
    Value overlay popout with market-model blending.

    p_mkt is de-overrounded market probability from selected market.
    p_blend = w * p_model + (1-w) * p_mkt      [linear]
           or sigmoid( w*logit(p_model) + (1-w)*logit(p_mkt) )  [logit]
    Bets and Kelly are computed using p_blend against the chosen market odds.
    """
    def __init__(self, parent, recs_pack):
        super().__init__(parent)
        self.parent = parent
        self.title("Value Recommendations")
        self.geometry("1040x680")
        self.configure(bg="#f6fbff")

        self.recs_pack = recs_pack  # {"race_meta": {...}, "rows": [...]}

        # ===== Controls =====
        ctrl = ttk.Frame(self, padding=10)
        ctrl.pack(fill="x")

        # Markets
        ttk.Label(ctrl, text="Win market").grid(row=0, column=0, sticky="w")
        self.win_source = tk.StringVar(value="fixed")
        ttk.Combobox(ctrl, textvariable=self.win_source, width=8, state="readonly",
                     values=["fixed", "tote"]).grid(row=0, column=1, sticky="w", padx=(6, 16))

        ttk.Label(ctrl, text="Overround").grid(row=0, column=2, sticky="w")
        self.or_method = tk.StringVar(value="proportional")
        ttk.Combobox(ctrl, textvariable=self.or_method, width=13, state="readonly",
                     values=["proportional", "power(α=0.9)", "none"]).grid(row=0, column=3, sticky="w", padx=(6, 16))

        # Blend
        ttk.Label(ctrl, text="Blend mode").grid(row=0, column=4, sticky="w")
        self.blend_mode = tk.StringVar(value="logit")
        ttk.Combobox(ctrl, textvariable=self.blend_mode, width=8, state="readonly",
                     values=["linear", "logit"]).grid(row=0, column=5, sticky="w", padx=(6, 16))

        ttk.Label(ctrl, text="Model weight w").grid(row=0, column=6, sticky="w")
        self.model_w = tk.DoubleVar(value=0.4)  # start conservative to favour market
        ttk.Entry(ctrl, textvariable=self.model_w, width=6).grid(row=0, column=7, sticky="w", padx=(6, 16))

        # Bankroll & Kelly
        ttk.Label(ctrl, text="Bankroll").grid(row=1, column=0, sticky="w", pady=(8,0))
        self.bankroll = tk.DoubleVar(value=1000.0)
        ttk.Entry(ctrl, textvariable=self.bankroll, width=10).grid(row=1, column=1, sticky="w", padx=(6, 16), pady=(8,0))

        ttk.Label(ctrl, text="Kelly fraction").grid(row=1, column=2, sticky="w", pady=(8,0))
        self.kelly_frac = tk.DoubleVar(value=0.25)
        ttk.Entry(ctrl, textvariable=self.kelly_frac, width=6).grid(row=1, column=3, sticky="w", padx=(6, 16), pady=(8,0))

        # Thresholds
        ttk.Label(ctrl, text="Min edge %").grid(row=1, column=4, sticky="w", pady=(8,0))
        self.min_edge_pct = tk.DoubleVar(value=1.0)  # softer now we shrink to market
        ttk.Entry(ctrl, textvariable=self.min_edge_pct, width=8).grid(row=1, column=5, sticky="w", padx=(6, 16), pady=(8,0))

        ttk.Label(ctrl, text="Min Kelly %").grid(row=1, column=6, sticky="w", pady=(8,0))
        self.min_kelly_pct = tk.DoubleVar(value=0.3)
        ttk.Entry(ctrl, textvariable=self.min_kelly_pct, width=8).grid(row=1, column=7, sticky="w", padx=(6, 16), pady=(8,0))

        ttk.Label(ctrl, text="Max picks").grid(row=1, column=8, sticky="w", pady=(8,0))
        self.max_picks = tk.IntVar(value=4)
        ttk.Entry(ctrl, textvariable=self.max_picks, width=6).grid(row=1, column=9, sticky="w", padx=(6, 16), pady=(8,0))

        ttk.Button(ctrl, text="Refresh", command=self.refresh).grid(row=1, column=10, sticky="e")

        # ===== Table =====
        table_frame = ttk.Frame(self, padding=(10, 6))
        table_frame.pack(fill="both", expand=True)

        cols = ("no","name","mkt","odds","mkt%","model%","blend%","edge%","fair","kelly%","stake")
        headers = ["#","Runner","Market","Odds","Market%","Model%","Blend%","Edge%","Fair","Kelly%","Stake"]
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=20)
        for c, h in zip(cols, headers):
            self.tree.heading(c, text=h)
        widths = [50, 240, 80, 70, 80, 80, 80, 70, 70, 70, 90]
        anchors = ["center","w","center","center","center","center","center","center","center","center","center"]
        for (c, w, a) in zip(cols, widths, anchors):
            self.tree.column(c, width=w, anchor=a)
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.refresh()

    # ===== maths helpers =====
    @staticmethod
    def _implied_from_odds(od):
        try:
            od = float(od)
            return 1.0/od if od and od > 1.0 else None
        except Exception:
            return None

    @staticmethod
    def _deoverround_proportional(probs):
        s = np.nansum(probs)
        if not np.isfinite(s) or s <= 0:
            return probs
        return probs / s

    @staticmethod
    def _deoverround_power(probs, alpha=0.9):
        # power method often shrinks longshots toward zero a touch more
        p = np.power(np.clip(probs, 1e-12, 1.0), alpha)
        s = np.nansum(p)
        return p / s if s > 0 else probs

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-12, 1-1e-12)
        return np.log(p/(1-p))

    @staticmethod
    def _sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def _kelly_pct(p, odds_dec):
        try:
            if p is None or odds_dec is None or odds_dec <= 1.0:
                return 0.0
            b = float(odds_dec) - 1.0
            p = float(p)
            q = 1.0 - p
            k = (b*p - q) / b
            return max(0.0, 100.0 * k)
        except Exception:
            return 0.0

    # ===== core =====
    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        rows = self.recs_pack.get("rows") or []

        # read controls
        use_fixed = (self.win_source.get() == "fixed")
        bankroll = max(0.0, float(self.bankroll.get() or 0.0))
        frac = max(0.0, min(1.0, float(self.kelly_frac.get() or 0.0)))
        min_edge = float(self.min_edge_pct.get() or 0.0)
        min_kelly = float(self.min_kelly_pct.get() or 0.0)
        max_picks = int(self.max_picks.get() or 9999)
        w = max(0.0, min(1.0, float(self.model_w.get() or 0.0)))
        blend_mode = self.blend_mode.get()
        or_method = self.or_method.get()

        # Build market implied vector, then de-overround
        odds_list = []
        model_p_list = []
        for r in rows:
            odds = r["fixed_win"] if use_fixed else r["tote_win"]
            odds_list.append(odds)
            model_p_list.append(float(r.get("model_p") or 0.0))
        mkt_probs = np.array([self._implied_from_odds(od) for od in odds_list], dtype=float)

        if or_method == "proportional":
            mkt_probs = self._deoverround_proportional(mkt_probs)
        elif or_method.startswith("power"):
            mkt_probs = self._deoverround_power(mkt_probs, alpha=0.9)
        # else "none" keeps raw implied 1/odds

        model_probs = np.array(model_p_list, dtype=float)
        # normalise tiny negatives
        model_probs = np.clip(model_probs, 0.0, 1.0)

        # Blend
        if blend_mode == "linear":
            blend_probs = w*model_probs + (1.0-w)*mkt_probs
        else:
            # logit-space blend
            z = w*self._logit(model_probs) + (1.0-w)*self._logit(mkt_probs)
            blend_probs = self._sigmoid(z)

        # compute outputs per runner
        recs = []
        for idx, r in enumerate(rows):
            num  = r["runner_number"]
            name = r["runner_name"]
            odds = r["fixed_win"] if use_fixed else r["tote_win"]
            mkt = "Fixed" if use_fixed else "Tote"

            p_mkt = mkt_probs[idx]
            p_mod = model_probs[idx]
            p_bl  = blend_probs[idx]

            imp = self._implied_from_odds(odds)
            if imp is None or odds is None or p_bl <= 0.0:
                edge_pct = 0.0
                kelly_pct = 0.0
                fair = ""
            else:
                # edge vs chosen market odds using blended probability
                edge_pct = 100.0 * (p_bl - imp)
                kelly_pct = self._kelly_pct(p_bl, odds)
                fair = f"{1.0/max(p_bl,1e-9):.2f}"

            if edge_pct >= min_edge and kelly_pct >= min_kelly:
                stake = bankroll * (kelly_pct/100.0) * frac
                recs.append((
                    edge_pct,          # sort key
                    num, name, mkt, odds,
                    100.0*p_mkt,       # show market%
                    100.0*p_mod,       # model%
                    100.0*p_bl,        # blend%
                    fair,
                    kelly_pct,
                    stake
                ))

        recs.sort(key=lambda t: (t[0], t[9]), reverse=True)
        recs = recs[:max_picks]

        for item in recs:
            (_, num, name, mkt, odds, mkt_pct, mod_pct, bl_pct, fair, kelly_pct, stake) = item
            self.tree.insert(
                "",
                "end",
                values=(
                    num or "",
                    name,
                    mkt,
                    f"{odds:g}" if odds else "",
                    f"{mkt_pct:.1f}",
                    f"{mod_pct:.1f}",
                    f"{bl_pct:.1f}",
                    f"{item[0]:.1f}",      # edge%
                    fair,
                    f"{kelly_pct:.1f}",
                    f"{stake:.2f}",
                ),
            )

if __name__ == "__main__":
    app = SimpleRaceGUI()
    app.mainloop()
