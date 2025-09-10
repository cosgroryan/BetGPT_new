#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
  python recs_by_meet.py <MEET_NO> <RACE...> [options]

Examples:
  python recs_by_meet.py 12 1 2 3 --date 2025-08-20 --bet_type place --market fixed \
    --blend logit --w 0.2 --overround proportional --min_edge 0.6 --min_kelly 1.0 \
    --kelly_frac 0.25 --bankroll 100 --max_picks 1 --tote_fallback --debug
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------- Required headers (as requested) ----------
HEADERS = {
    "User-Agent": "Ryan Cosgrove",
    "Accept": "application/json",
    "Origin": "https://www.tab.co.nz",
    "Referer": "https://www.tab.co.nz/",
}

ODDS_BASE = "https://json.tab.co.nz/odds"
SCHED_BASE = "https://json.tab.co.nz/schedule"

TIMEOUT = 20


# ---------- Optional: help PyTorch pickles resolve classes ----------
try:
    import __main__
    import pytorch_pre as pp  # your helpers used by recommend_picks_NN
    for cls_name in ["PreprocessArtifacts", "FeatureBuilder", "ModelBundle", "Normalizer"]:
        if hasattr(pp, cls_name) and not hasattr(__main__, cls_name):
            setattr(__main__, cls_name, getattr(pp, cls_name))
    from pytorch_pre import load_model_and_predict
except Exception:
    load_model_and_predict = None  # will be unused if model_win_table works


# ---------- Prefer your existing model entrypoint ----------
def _import_model_win_table():
    try:
        from recommend_picks_NN import model_win_table
        return model_win_table
    except Exception as e:
        return None


# ---------- Session / HTTP ----------
def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def get_json(s: requests.Session, url: str, debug: bool = False):
    if debug:
        print(f"[http] GET {url}")
    r = s.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ---------- Odds + schedule parsing ----------
def find_meeting_node(schedule: dict, meet_no: int) -> Optional[dict]:
    for m in (schedule.get("meetings") or []):
        try:
            if int(m.get("number") or 0) == int(meet_no):
                return m
        except Exception:
            continue
    return None


def extract_race_node(odds_payload: dict, race_no: int) -> Optional[dict]:
    # odds payload shape:
    # { date, meetings: [ { number, races: [ { number, entries:[...] } ] } ] }
    for m in (odds_payload.get("meetings") or []):
        for r in (m.get("races") or []):
            try:
                if int(r.get("number") or 0) == int(race_no):
                    return r
            except Exception:
                pass
    # fallback if "races" is at root
    for r in (odds_payload.get("races") or []):
        try:
            if int(r.get("number") or 0) == int(race_no):
                return r
        except Exception:
            pass
    return None


def extract_prices_map(race_node: dict) -> Dict[int, dict]:
    """
    Returns {runner_number: {
        'runner_name', 'win_fixed','place_fixed','win_tote','place_tote'
    }}
    """
    out: Dict[int, dict] = {}
    for e in (race_node.get("entries") or []):
        n = e.get("number") or e.get("runner") or e.get("runner_number")
        try:
            rn = int(n)
        except Exception:
            continue
        rec = out.setdefault(rn, {})
        # names can live on entries for /odds endpoint; preserve if present
        nm = e.get("name") or e.get("runnerName") or e.get("runner_name")
        if nm:
            rec["runner_name"] = str(nm)
        # fixed prices (as decimals)
        if e.get("ffwin") is not None:
            rec["win_fixed"] = e.get("ffwin")
        if e.get("ffplc") is not None:
            rec["place_fixed"] = e.get("ffplc")
        # tote prices (as decimals)
        if e.get("win") is not None:
            rec["win_tote"] = e.get("win")
        if e.get("plc") is not None:
            rec["place_tote"] = e.get("plc")
    return out


# ---------- Maths helpers (same as GUI logic) ----------
def implied_from_odds(od: Optional[float]) -> Optional[float]:
    try:
        od = float(od)
        return 1.0 / od if od and od > 1.0 else None
    except Exception:
        return None


def deoverround_proportional(p: np.ndarray) -> np.ndarray:
    s = np.nansum(p)
    if not np.isfinite(s) or s <= 0:
        return p
    return p / s


def deoverround_power(p: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    q = np.power(np.clip(p, 1e-12, 1.0), alpha)
    s = np.nansum(q)
    return q / s if s > 0 else p


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return np.log(x / (1 - x))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def kelly_pct(p: Optional[float], odds_dec: Optional[float]) -> float:
    try:
        if p is None or odds_dec is None or odds_dec <= 1.0:
            return 0.0
        b = float(odds_dec) - 1.0
        p = float(p)
        q = 1.0 - p
        k = (b * p - q) / b
        return max(0.0, 100.0 * k)
    except Exception:
        return 0.0


def positions_paid_default(field_size: int) -> int:
    return 2 if field_size <= 7 else 3


# ---------- Model probabilities (use your model) ----------
def model_probs_for(meet_no: int, race_no: int, date_str: str, entries: Dict[int, dict], debug=False) -> Dict[int, float]:
    """
    Try your recommend_picks_NN.model_win_table first (preferred),
    else fall back to pytorch_pre.load_model_and_predict on a minimal DF.
    Returns {runner_number: p_win}
    """
    model_win_table = _import_model_win_table()
    # try #1: your recommend_picks_NN path
    if model_win_table is not None:
        try:
            df = model_win_table(meet_no, race_no, date_str)  # expects your local code
            if isinstance(df, pd.DataFrame) and not df.empty and "runner_number" in df and "p_win" in df:
                mp = {}
                for rn, p in zip(df["runner_number"].tolist(), df["p_win"].tolist()):
                    try:
                        mp[int(rn)] = float(p)
                    except Exception:
                        pass
                if debug:
                    print(f"[model] model_win_table found {len(mp)} runners")
                if mp:
                    return mp
        except Exception as e:
            if debug:
                print(f"[model] model_win_table errored, falling back: {e}")

    # try #2: pytorch_pre direct (softmax)
    if load_model_and_predict is not None and entries:
        rows = []
        for rn, rec in entries.items():
            rows.append({
                "runner_number": rn,
                "runner_name": rec.get("runner_name", f"#{rn}"),
                "race_number": race_no,
                "meeting_number": meet_no,
                "meeting_venue": "",  # optional; your loader ignores if absent
            })
        df2 = pd.DataFrame(rows)
        try:
            out = load_model_and_predict(df2)
            if isinstance(out, dict) and "p_win_softmax" in out:
                p = np.asarray(out["p_win_softmax"], dtype=float)
                # safety normalise
                if np.isfinite(p).any():
                    s = float(np.nansum(p))
                    if s > 0:
                        p = p / s
                res = {int(rn): float(p[i]) for i, rn in enumerate(df2["runner_number"].tolist())}
                if debug:
                    print(f"[model] load_model_and_predict produced {len(res)} probs")
                return res
        except Exception as e:
            if debug:
                print(f"[model] load_model_and_predict errored: {e}")

    # final fallback: uniform
    n = len(entries)
    if n > 0:
        if debug:
            print("[model] fallback to uniform probs")
        return {int(rn): 1.0 / n for rn in entries.keys()}
    return {}


# ---------- CLI core ----------
@dataclass
class Settings:
    bet_type: str         # 'win' | 'place' | 'each_way'
    market: str           # 'fixed' | 'tote'
    overround: str        # 'proportional' | 'power' | 'none'
    blend: str            # 'linear' | 'logit'
    w: float
    min_edge: float       # percent points
    min_kelly: float      # percent
    kelly_frac: float     # fraction of bankroll
    bankroll: float
    max_picks: int
    tote_fallback: bool
    alpha_power: float    # for overround 'power'
    positions_paid: Optional[int]
    debug: bool


def run_day_for_meet(date_str: str, meet_no: int, race_list: List[int], cfg: Settings):
    s = make_session()
    # schedule (for pretty names & to confirm meet exists)
    sched_url = f"{SCHED_BASE}/{date_str}"
    try:
        sched = get_json(s, sched_url, cfg.debug)
    except Exception as e:
        print(f"[sched] fetch failed: {e}")
        sched = {}

    mnode = find_meeting_node(sched, meet_no)
    if not mnode:
        print(f"[sched] meeting {meet_no} not found for {date_str}")
        return

    venue = mnode.get("venue") or mnode.get("name") or mnode.get("meetingName") or f"Meet {meet_no}"
    mtype = mnode.get("type") or mnode.get("event_type") or "?"
    print(f"\nDay Recs — {date_str} | M{meet_no} {venue}")
    print(f"bet_type={cfg.bet_type} market={cfg.market} blend={cfg.blend} w={cfg.w} overround={cfg.overround}")
    print(f"bankroll={cfg.bankroll:.2f} kelly_frac={cfg.kelly_frac} min_edge={cfg.min_edge} min_kelly={cfg.min_kelly} max_picks={cfg.max_picks}")
    print("-" * 72)

    total_picks = 0

    # if races not passed, try all available from schedule
    if not race_list:
        race_list = sorted([int(r.get("number") or 0) for r in (mnode.get("races") or []) if r.get("number")])

    for raceno in race_list:
        odds_url = f"{ODDS_BASE}/{date_str}/{meet_no}/{raceno}"
        try:
            odds_payload = get_json(s, odds_url, cfg.debug)
        except Exception as e:
            print(f"[race] R{raceno}: odds fetch failed: {e}")
            print("-" * 72)
            continue

        rnode = extract_race_node(odds_payload, raceno)
        if not rnode:
            print(f"[race] R{raceno}: no race node in odds payload")
            print("-" * 72)
            continue

        # print NZ-normalised time if present
        nz_time = rnode.get("norm_time")  # often "YYYY-MM-DD HH:MM:SS" NZT
        rname = (rnode.get("name") or "").strip()
        rstatus = rnode.get("status") or "?"
        print(f"[race] consider R{raceno} {rname} status={rstatus} nz_time={nz_time}")

        prices = extract_prices_map(rnode)
        if not prices:
            print(f"[race] R{raceno}: no prices")
            print("-" * 72)
            continue

        # model p(win) per runner_number
        p_mod_map = model_probs_for(meet_no, raceno, date_str, prices, debug=cfg.debug)

        # assemble aligned arrays
        order = sorted(prices.keys())
        odds_win = []
        odds_pl = []
        names = []
        for rn in order:
            rec = prices[rn]
            # choose market leg
            od_w = rec.get("win_fixed") if cfg.market == "fixed" else rec.get("win_tote")
            od_p = rec.get("place_fixed") if cfg.market == "fixed" else rec.get("place_tote")
            # optional tote fallback if fixed missing / invalid
            if cfg.tote_fallback and cfg.market == "fixed":
                if od_w is None or (isinstance(od_w, (int, float)) and od_w <= 1.0):
                    od_w = rec.get("win_tote", od_w)
                if od_p is None or (isinstance(od_p, (int, float)) and od_p <= 1.0):
                    od_p = rec.get("place_tote", od_p)
            # coerce floats or NaN
            def _f(x):
                try:
                    return float(x)
                except Exception:
                    return float("nan")
            odds_win.append(_f(od_w))
            odds_pl.append(_f(od_p))
            names.append(rec.get("runner_name", f"#{rn}"))

        odds_win = np.array(odds_win, dtype=float)
        odds_pl = np.array(odds_pl, dtype=float)

        # market implied WIN probs, with overround adjustment
        p_mkt_win = np.array([implied_from_odds(x) for x in odds_win], dtype=float)
        if cfg.overround == "proportional":
            p_mkt_win = deoverround_proportional(p_mkt_win)
        elif cfg.overround == "power":
            p_mkt_win = deoverround_power(p_mkt_win, alpha=cfg.alpha_power)

        # model WIN probs
        p_mod_win = np.array([float(p_mod_map.get(int(n), 0.0)) for n in order], dtype=float)
        p_mod_win = np.clip(p_mod_win, 1e-6, 1 - 1e-6)

        # model PLACE proxy (positions paid * p_win), clipped to [0,1]
        field_size = len(order)
        pos_paid = cfg.positions_paid if cfg.positions_paid else positions_paid_default(field_size)
        p_mod_pl = np.clip(p_mod_win * float(pos_paid), 1e-6, 1 - 1e-6)

        # market implied PLACE probs if we have place odds
        p_mkt_pl_raw = np.array([implied_from_odds(x) for x in odds_pl], dtype=float)
        if cfg.overround == "proportional":
            p_mkt_pl = deoverround_proportional(p_mkt_pl_raw) if np.isfinite(p_mkt_pl_raw).any() else p_mkt_pl_raw
        elif cfg.overround == "power":
            p_mkt_pl = deoverround_power(p_mkt_pl_raw, alpha=cfg.alpha_power) if np.isfinite(p_mkt_pl_raw).any() else p_mkt_pl_raw
        else:
            p_mkt_pl = p_mkt_pl_raw

        # blend (same shapes as GUI)
        w = float(max(0.0, min(1.0, cfg.w)))
        if cfg.blend == "linear":
            p_bl_win = w * p_mod_win + (1.0 - w) * p_mkt_win
            p_bl_pl = w * p_mod_pl + (1.0 - w) * p_mkt_pl
        else:
            # logit blend
            p_bl_win = sigmoid(w * logit(p_mod_win) + (1.0 - w) * logit(p_mkt_win))
            if np.isfinite(p_mkt_pl).any():
                p_bl_pl = sigmoid(w * logit(p_mod_pl) + (1.0 - w) * logit(p_mkt_pl))
            else:
                p_bl_pl = p_mod_pl

        # compute edges, kelly, EV, etc. and pick
        out_rows = []
        for i, rn in enumerate(order):
            imp_w = implied_from_odds(odds_win[i])
            imp_p = implied_from_odds(odds_pl[i])

            edge_w = 100.0 * ((p_bl_win[i] - (imp_w or 0.0)) if imp_w is not None else 0.0)
            edge_p = 100.0 * ((p_bl_pl[i] - (imp_p or 0.0)) if imp_p is not None else 0.0)

            k_w = kelly_pct(p_bl_win[i], odds_win[i]) if np.isfinite(odds_win[i]) else 0.0
            k_p = kelly_pct(p_bl_pl[i], odds_pl[i]) if np.isfinite(odds_pl[i]) else 0.0

            ev_w = p_bl_win[i] * odds_win[i] - 1.0 if np.isfinite(odds_win[i]) else 0.0
            ev_p = p_bl_pl[i] * odds_pl[i] - 1.0 if np.isfinite(odds_pl[i]) else 0.0
            ev_ew = 0.5 * ev_w + 0.5 * ev_p

            # pass tests
            pass_w = np.isfinite(odds_win[i]) and odds_win[i] > 1.0 and edge_w >= cfg.min_edge and k_w >= cfg.min_kelly
            pass_p = np.isfinite(odds_pl[i]) and odds_pl[i] > 1.0 and edge_p >= cfg.min_edge and k_p >= cfg.min_kelly
            passed = pass_w if cfg.bet_type == "win" else pass_p if cfg.bet_type == "place" else (pass_w or pass_p)

            out_rows.append({
                "sort": float(p_bl_win[i]),
                "passed": passed,
                "rn": int(rn),
                "name": names[i],
                "odds_w": float(odds_win[i]) if np.isfinite(odds_win[i]) else None,
                "odds_p": float(odds_pl[i]) if np.isfinite(odds_pl[i]) else None,
                "p_bl_w": float(p_bl_win[i]),
                "p_bl_p": float(p_bl_pl[i]),
                "edge_w": float(edge_w),
                "edge_p": float(edge_p),
                "kelly_w": float(k_w),
                "kelly_p": float(k_p),
                "ev_w": float(ev_w),
                "ev_p": float(ev_p),
                "ev_ew": float(ev_ew),
            })

        # order by blended win% desc (same as GUI)
        out_rows.sort(key=lambda d: d["sort"], reverse=True)

        picks = []
        for r in out_rows:
            if not r["passed"]:
                continue
            picks.append(r)
            if cfg.max_picks and len(picks) >= cfg.max_picks:
                break

        print(f"R{raceno} — {rname}")
        if not picks:
            print("  no bet")
        else:
            for p in picks:
                if cfg.bet_type == "win":
                    stake = cfg.bankroll * (p["kelly_w"] / 100.0) * cfg.kelly_frac
                    print(
                        f"  WIN   ${stake:.2f} on #{p['rn']} {p['name']}  @ {cfg.market} {p['odds_w']:.2f}  "
                        f"blend {p['p_bl_w']*100:.1f}%  edge {p['edge_w']:.1f}%  kelly {p['kelly_w']:.1f}%  "
                        f"fair {1.0/max(p['p_bl_w'],1e-9):.2f}  EV {p['ev_w']:.2f}"
                    )
                elif cfg.bet_type == "place":
                    stake = cfg.bankroll * (p["kelly_p"] / 100.0) * cfg.kelly_frac
                    odds_p = p["odds_p"]
                    odds_p_s = f"{odds_p:.2f}" if odds_p is not None else "-"
                    fair_p = (1.0 / max(p["p_bl_p"], 1e-9)) if p["p_bl_p"] > 0 else float("nan")
                    print(
                        f"  PLACE ${stake:.2f} on #{p['rn']} {p['name']}  @ {cfg.market} {odds_p_s}  "
                        f"blend {p['p_bl_p']*100:.1f}%  edge {p['edge_p']:.1f}%  kelly {p['kelly_p']:.1f}%  "
                        f"fair {fair_p:.2f}  EV {p['ev_p']:.2f}"
                    )
                else:
                    # each-way: half on win, half on place
                    stake_w = cfg.bankroll * (p["kelly_w"] / 100.0) * (cfg.kelly_frac * 0.5)
                    stake_p = cfg.bankroll * (p["kelly_p"] / 100.0) * (cfg.kelly_frac * 0.5)
                    ow = f"{p['odds_w']:.2f}" if p["odds_w"] is not None else "-"
                    op = f"{p['odds_p']:.2f}" if p["odds_p"] is not None else "-"
                    print(
                        f"  EACH WAY  W ${stake_w:.2f} @ {cfg.market} {ow} | P ${stake_p:.2f} @ {cfg.market} {op}  "
                        f"#{p['rn']} {p['name']}  (blend W {p['p_bl_w']*100:.1f}%  P {p['p_bl_p']*100:.1f}%)"
                    )
            total_picks += len(picks)

        print("-" * 72)

    print(f"Total picks: {total_picks}")


def nz_today_iso() -> str:
    # Default to NZ date (Pacific/Auckland) for schedule alignment
    try:
        from zoneinfo import ZoneInfo
        nz = ZoneInfo("Pacific/Auckland")
        return datetime.now(tz=nz).date().isoformat()
    except Exception:
        # fallback to local date
        return datetime.now().date().isoformat()


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model–market blended racing recommendations by meet/race.")
    p.add_argument("meet", type=int, help="Meeting number (e.g., 12)")
    p.add_argument("races", type=int, nargs="*", help="Race numbers (e.g., 1 2 3). If empty, all available races in meet.")
    p.add_argument("--date", type=str, default=nz_today_iso(), help="YYYY-MM-DD (defaults to NZ date today).")
    p.add_argument("--bet_type", choices=["win", "place", "each_way"], default="place")
    p.add_argument("--market", choices=["fixed", "tote"], default="fixed")
    p.add_argument("--overround", choices=["proportional", "power", "none"], default="proportional")
    p.add_argument("--alpha", type=float, default=0.9, help="alpha for overround=power (default 0.9)")
    p.add_argument("--blend", choices=["linear", "logit"], default="logit")
    p.add_argument("--w", type=float, default=0.2, help="Model weight in blend (0..1)")
    p.add_argument("--bankroll", type=float, default=100.0)
    p.add_argument("--kelly_frac", type=float, default=0.25)
    p.add_argument("--min_edge", type=float, default=0.6, help="Minimum edge percentage points to qualify (e.g., 0.6)")
    p.add_argument("--min_kelly", type=float, default=1.0, help="Minimum Kelly percent to qualify (e.g., 1.0)")
    p.add_argument("--max_picks", type=int, default=1, help="Max picks per race (0 = no limit)")
    p.add_argument("--tote_fallback", action="store_true", help="When market=fixed, fall back to tote odds if fixed missing/invalid.")
    p.add_argument("--positions_paid", type=int, default=None, help="Override places paid (default: 2 if <=7 runners else 3).")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = Settings(
        bet_type=args.bet_type,
        market=args.market,
        overround=args.overround,
        blend=args.blend,
        w=float(args.w),
        min_edge=float(args.min_edge),
        min_kelly=float(args.min_kelly),
        kelly_frac=float(args.kelly_frac),
        bankroll=float(args.bankroll),
        max_picks=int(args.max_picks),
        tote_fallback=bool(args.tote_fallback),
        alpha_power=float(args.alpha),
        positions_paid=args.positions_paid,
        debug=bool(args.debug),
    )
    run_day_for_meet(args.date, int(args.meet), list(args.races), cfg)


if __name__ == "__main__":
    main()
