#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtest blended model+market recommendations on historical races.

- Reads a Parquet or CSV with past races (works with your big parquet).
- Computes market implied from fixed or tote odds, de-overrounds per race.
- Calls your NN once per race to get model p(win).
- Blends p(model) and p(market) by either linear or logit blend.
- Applies Min edge %, Min Kelly %, Max picks, Kelly fraction and bankroll.
- Tracks realised P&L and equity curve, plus summary metrics.
- Can grid search over w, blend mode, overround, edge thresholds etc.

Example:
  python backtest_blend.py \
      --data five_year_dataset.parquet \
      --market fixed \
      --start 2024-01-01 --end 2025-08-18 \
      --blend linear --w 0.4 --overround proportional \
      --min_edge 1.0 --min_kelly 0.3 --kelly_frac 0.25 \
      --bankroll 1000 --max_picks 4 --dynamic_bankroll \
      --bet_type win
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# --- import your NN loader and shim the __main__ pickle classes if needed
import __main__
import pytorch_pre as pp

if not hasattr(__main__, "PreprocessArtifacts") and hasattr(pp, "PreprocessArtifacts"):
    __main__.PreprocessArtifacts = pp.PreprocessArtifacts

load_model_and_predict = pp.load_model_and_predict  # returns dict with p_win_softmax

# -----------------------
# Helpers
# -----------------------
ODDS_FIXED_CANDIDATES = ["ffwin", "fixed_win", "win_fixed", "win_fx", "odds_fixed_win"]
ODDS_TOTE_CANDIDATES  = ["win", "tote_win", "win_tote"]

# new: place odds candidates
ODDS_PLACE_FIXED_CANDIDATES = ["ffplc", "place_fixed", "pl_fixed", "plc_fixed", "odds_fixed_place"]
ODDS_PLACE_TOTE_CANDIDATES  = ["plc", "tote_place", "place_tote"]

RESULT_COLS           = ["finish_position", "finish_rank", "placing", "position", "result_pos"]


def choose_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
        # allow case-insensitive hits
        for cc in df.columns:
            if cc.lower() == c.lower():
                return cc
    return None

def implied_from_odds(od):
    try:
        od = float(od)
        return 1.0/od if od and od > 1.0 else np.nan
    except Exception:
        return np.nan

def deoverround_proportional(p: np.ndarray) -> np.ndarray:
    s = np.nansum(p)
    return p / s if s and np.isfinite(s) and s > 0 else p

def deoverround_power(p: np.ndarray, alpha=0.9) -> np.ndarray:
    q = np.power(np.clip(p, 1e-12, 1.0), alpha)
    s = np.nansum(q)
    return q / s if s and s > 0 else p

def logit(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return np.log(p/(1-p))

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def kelly_pct(p_blend: float, odds_dec: float) -> float:
    if not odds_dec or odds_dec <= 1.0 or p_blend <= 0.0:
        return 0.0
    b = float(odds_dec) - 1.0
    q = 1.0 - float(p_blend)
    k = (b*float(p_blend) - q) / b
    return max(0.0, 100.0*k)

def race_key_from_df(g: pd.DataFrame) -> str:
    # prefer race_id if present, else venue|race_number|date
    if "race_id" in g.columns and g["race_id"].notna().any():
        return str(g["race_id"].astype(str).iloc[0])
    mv = g["meeting_venue"].astype(str) if "meeting_venue" in g.columns else pd.Series([""], index=g.index)
    rn = g["race_number"].astype(str)    if "race_number" in g.columns else pd.Series([""], index=g.index)
    dt = pd.to_datetime(g["date"], errors="coerce").dt.strftime("%Y-%m-%d") if "date" in g.columns else pd.Series([""], index=g.index)
    return f"{mv.iloc[0]}|{rn.iloc[0]}|{dt.iloc[0]}"

def default_positions_paid(runners: int) -> int:
    # simple default, adjust if you later want jurisdiction rules
    if runners <= 7:
        return 2
    return 3

# -----------------------
# Backtest core
# -----------------------
@dataclass
class Params:
    market: str = "fixed"          # fixed or tote
    blend: str = "linear"          # linear or logit
    w: float = 0.4                 # model weight
    overround: str = "proportional"  # proportional, power, none
    min_edge: float = 1.0
    min_kelly: float = 0.3
    kelly_frac: float = 0.25
    max_picks: int = 4
    dynamic_bankroll: bool = True
    bankroll0: float = 1000.0
    bet_type: str = "win"          # win, place, each_way

@dataclass
class Metrics:
    total_profit: float
    roi: float
    picks: int
    hit_rate: float
    kelly_avg: float
    max_drawdown: float
    sharpe: float
    final_bankroll: float

def compute_model_probs_for_race(g: pd.DataFrame) -> np.ndarray:
    """Call your NN on the race subset and return p_win per row order."""
    out = load_model_and_predict(g)
    if isinstance(out, dict) and "p_win_softmax" in out:
        p = np.asarray(out["p_win_softmax"], dtype=float)
        if p.shape[0] == len(g):
            return p
    # fallback: softmax by -pred_rank
    if isinstance(out, dict) and "pred_rank" in out:
        r = -np.asarray(out["pred_rank"], dtype=float)
    else:
        r = -np.asarray(out, dtype=float)
    r = r - np.max(r)
    ex = np.exp(r)
    return ex / np.sum(ex)

def backtest(df: pd.DataFrame,
             params: Params,
             odds_col_fixed: str,
             odds_col_tote: Optional[str],
             result_col: str,
             verbose: bool = True
             ) -> Tuple[Metrics, pd.DataFrame]:
    # sort and prep
    if "date" in df.columns:
        df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["_date"] = pd.Timestamp("1970-01-01")
    if "race_number" in df.columns:
        df["_rnum"] = pd.to_numeric(df["race_number"], errors="coerce")
    else:
        df["_rnum"] = 0
    df = df.sort_values(["_date", "_rnum"]).copy()

    bankroll = params.bankroll0
    equity_rows: List[dict] = []
    pick_rows: List[dict] = []

    # choose group columns safely
    group_cols = [c for c in ["_date", "meeting_venue", "race_number"] if c in df.columns]
    if not group_cols:
        group_cols = ["_date"]

    total_groups = 0

    for _, g in df.groupby(group_cols, dropna=False):
        g = g.copy()
        total_groups += 1
        key = race_key_from_df(g)
        n = len(g)

        # runners and places
        runners = int(n)
        positions_paid = default_positions_paid(runners)
        if "positions_paid" in g.columns:
            try:
                # prefer dataset value if present
                ppd = int(pd.to_numeric(g["positions_paid"], errors="coerce").dropna().iloc[0])
                if ppd > 0:
                    positions_paid = ppd
            except Exception:
                pass

        # pick market odds series for win and place
        fixed_win = pd.to_numeric(g.get(odds_col_fixed), errors="coerce") if odds_col_fixed in g.columns else pd.Series([np.nan]*n, index=g.index)
        tote_win  = pd.to_numeric(g.get(odds_col_tote),  errors="coerce") if odds_col_tote and odds_col_tote in g.columns else pd.Series([np.nan]*n, index=g.index)

        # detect place columns
        place_fixed_col = choose_col(g, ODDS_PLACE_FIXED_CANDIDATES)
        place_tote_col  = choose_col(g, ODDS_PLACE_TOTE_CANDIDATES)
        fixed_place = pd.to_numeric(g.get(place_fixed_col), errors="coerce") if place_fixed_col else pd.Series([np.nan]*n, index=g.index)
        tote_place  = pd.to_numeric(g.get(place_tote_col),  errors="coerce") if place_tote_col  else pd.Series([np.nan]*n, index=g.index)

        # primary and fallback per chosen market
        if params.market == "fixed":
            odds_win  = np.where(np.isfinite(fixed_win) & (fixed_win > 1.0), fixed_win, np.where(np.isfinite(tote_win) & (tote_win > 1.0), tote_win, np.nan))
            odds_place= np.where(np.isfinite(fixed_place) & (fixed_place > 1.0), fixed_place, np.where(np.isfinite(tote_place) & (tote_place > 1.0), tote_place, np.nan))
        else:
            odds_win  = np.where(np.isfinite(tote_win) & (tote_win > 1.0), tote_win, np.where(np.isfinite(fixed_win) & (fixed_win > 1.0), fixed_win, np.nan))
            odds_place= np.where(np.isfinite(tote_place) & (tote_place > 1.0), tote_place, np.where(np.isfinite(fixed_place) & (fixed_place > 1.0), fixed_place, np.nan))

        # count valid odds depending on bet type
        valid_win_cnt   = int(np.isfinite(odds_win).sum())
        valid_place_cnt = int(np.isfinite(odds_place).sum())

        if params.bet_type == "place":
            valid_odds = valid_place_cnt
        elif params.bet_type == "each_way":
            # both legs need usable odds, use the limiting leg
            valid_odds = min(valid_win_cnt, valid_place_cnt)
        else:
            valid_odds = valid_win_cnt

        if verbose:
            print(f"[bt] race={key} runners={n} valid_odds={valid_odds}", flush=True)

        # if no odds at all for the selected bet type, just carry equity forward
        if (params.bet_type == "win" and valid_odds == 0) or \
           (params.bet_type == "place" and not np.isfinite(odds_place).any()) or \
           (params.bet_type == "each_way" and (not np.isfinite(odds_win).any() or not np.isfinite(odds_place).any())):
            equity_rows.append({"date": g["_date"].iloc[0], "race_key": key, "bankroll": bankroll})
            continue

        # market implied probs
        p_mkt_win = np.array([implied_from_odds(x) for x in odds_win], dtype=float)
        p_mkt_place_raw = np.array([implied_from_odds(x) for x in odds_place], dtype=float)

        # de-overround per race
        if params.overround == "proportional":
            p_mkt_win = deoverround_proportional(p_mkt_win)
            # place market is not a true multinomial, but normalise anyway for a stable blend
            if np.isfinite(p_mkt_place_raw).any():
                p_mkt_place = deoverround_proportional(p_mkt_place_raw)
            else:
                p_mkt_place = p_mkt_place_raw
        elif params.overround.startswith("power"):
            p_mkt_win = deoverround_power(p_mkt_win, alpha=0.9)
            if np.isfinite(p_mkt_place_raw).any():
                p_mkt_place = deoverround_power(p_mkt_place_raw, alpha=0.9)
            else:
                p_mkt_place = p_mkt_place_raw
        else:
            p_mkt_place = p_mkt_place_raw

        # model p(win)
        p_mod_win = compute_model_probs_for_race(g)
        p_mod_win = np.clip(p_mod_win, 1e-6, 1 - 1e-6)

        # rough model p(place): scale p(win) by positions paid
        # this preserves ranking and gives a higher place probability for favourites
        p_mod_place = np.clip(p_mod_win * float(positions_paid), 1e-6, 1 - 1e-6)

        # blend
        w = max(0.0, min(1.0, float(params.w)))
        if params.blend == "linear":
            p_bl_win   = w*p_mod_win   + (1.0-w)*p_mkt_win
            p_bl_place = w*p_mod_place + (1.0-w)*p_mkt_place
        else:
            p_bl_win   = sigmoid(w*logit(p_mod_win)   + (1.0-w)*logit(p_mkt_win))
            # guard against all-nan place probs
            if np.isfinite(p_mkt_place).any():
                p_bl_place = sigmoid(w*logit(p_mod_place) + (1.0-w)*logit(p_mkt_place))
            else:
                p_bl_place = p_mod_place  # fallback to model-only if no market place odds

        # selection metrics per leg
        imp_win   = np.array([implied_from_odds(x) for x in odds_win], dtype=float)
        edge_win  = p_bl_win - imp_win
        kelly_win = np.array([kelly_pct(p_bl_win[i], odds_win[i]) if np.isfinite(odds_win[i]) else 0.0
                              for i in range(len(odds_win))], dtype=float)

        imp_place   = np.array([implied_from_odds(x) for x in odds_place], dtype=float)
        edge_place  = p_bl_place - imp_place
        kelly_place = np.array([kelly_pct(p_bl_place[i], odds_place[i]) if np.isfinite(odds_place[i]) else 0.0
                                for i in range(len(odds_place))], dtype=float)

        # order by blended win prob, consistent with earlier behaviour
        order = np.argsort(-p_bl_win)

        chosen = []
        for i in order:
            if params.bet_type == "win":
                ok_odds = np.isfinite(odds_win[i]) and odds_win[i] > 1.0
                ok_edge = 100.0*edge_win[i] >= params.min_edge
                ok_kelly = kelly_win[i] >= params.min_kelly
                if ok_odds and ok_edge and ok_kelly:
                    chosen.append(("win", i))
            elif params.bet_type == "place":
                ok_odds = np.isfinite(odds_place[i]) and odds_place[i] > 1.0
                ok_edge = 100.0*edge_place[i] >= params.min_edge
                ok_kelly = kelly_place[i] >= params.min_kelly
                if ok_odds and ok_edge and ok_kelly:
                    chosen.append(("place", i))
            else:  # each_way
                ok_win   = np.isfinite(odds_win[i])   and odds_win[i]   > 1.0 and \
                           (100.0*edge_win[i]   >= params.min_edge or kelly_win[i]   >= params.min_kelly)
                ok_place = np.isfinite(odds_place[i]) and odds_place[i] > 1.0 and \
                           (100.0*edge_place[i] >= params.min_edge or kelly_place[i] >= params.min_kelly)
                if ok_win or ok_place:
                    chosen.append(("each_way", i))

            if params.max_picks and len(chosen) >= params.max_picks:
                break

        race_pnl = 0.0

        # outcome measurement helpers
        fin_ser = pd.to_numeric(g[result_col], errors="coerce")
        def _won_win(ix):   # winner only
            try:
                return int(fin_ser.iloc[ix]) == 1
            except Exception:
                return False
        def _won_place(ix): # placed within paid positions
            try:
                fp = int(fin_ser.iloc[ix])
                return 1 <= fp <= positions_paid
            except Exception:
                return False

        for bet_kind, i in chosen:
            stake_base = bankroll if params.dynamic_bankroll else params.bankroll0

            if bet_kind == "win":
                k_pct = float(kelly_win[i])
                stake = stake_base * (k_pct/100.0) * params.kelly_frac
                won = _won_win(i)
                pnl = stake*(odds_win[i]-1.0) if won else -stake
                race_pnl += pnl

                pick_rows.append({
                    "date": g["_date"].iloc[0].date() if pd.notna(g["_date"].iloc[0]) else None,
                    "venue": g["meeting_venue"].iloc[0] if "meeting_venue" in g.columns else "",
                    "race_number": g["race_number"].iloc[0] if "race_number" in g.columns else None,
                    "runner_name": g["runner_name"].iloc[i] if "runner_name" in g.columns else "",
                    "bet_type": "win",
                    "odds": float(odds_win[i]) if np.isfinite(odds_win[i]) else np.nan,
                    "p_mkt": float(p_mkt_win[i]),
                    "p_mod": float(p_mod_win[i]),
                    "p_blend": float(p_bl_win[i]),
                    "edge_pct": float(100.0*edge_win[i]),
                    "kelly_pct": float(kelly_win[i]),
                    "stake": float(stake),
                    "won": int(won),
                    "pnl": float(pnl),
                    "race_key": key,
                })

            elif bet_kind == "place":
                k_pct = float(kelly_place[i])
                stake = stake_base * (k_pct/100.0) * params.kelly_frac
                won = _won_place(i)
                pnl = stake*(odds_place[i]-1.0) if won else -stake
                race_pnl += pnl

                pick_rows.append({
                    "date": g["_date"].iloc[0].date() if pd.notna(g["_date"].iloc[0]) else None,
                    "venue": g["meeting_venue"].iloc[0] if "meeting_venue" in g.columns else "",
                    "race_number": g["race_number"].iloc[0] if "race_number" in g.columns else None,
                    "runner_name": g["runner_name"].iloc[i] if "runner_name" in g.columns else "",
                    "bet_type": "place",
                    "odds": float(odds_place[i]) if np.isfinite(odds_place[i]) else np.nan,
                    "p_mkt": float(p_mkt_place[i]) if np.isfinite(p_mkt_place[i]) else np.nan,
                    "p_mod": float(p_mod_place[i]),
                    "p_blend": float(p_bl_place[i]) if np.isfinite(p_bl_place[i]) else float(p_mod_place[i]),
                    "edge_pct": float(100.0*edge_place[i]) if np.isfinite(edge_place[i]) else np.nan,
                    "kelly_pct": float(kelly_place[i]),
                    "stake": float(stake),
                    "won": int(won),
                    "pnl": float(pnl),
                    "race_key": key,
                })

            else:  # each_way: half win leg, half place leg
                # stakes per leg
                k_win = float(kelly_win[i])
                k_plc = float(kelly_place[i])

                stake_win  = stake_base * (k_win/100.0) * params.kelly_frac * 0.5
                stake_plc  = stake_base * (k_plc/100.0) * params.kelly_frac * 0.5

                won_win  = _won_win(i)
                won_plc  = _won_place(i)

                pnl_win  = stake_win*(odds_win[i]-1.0)   if won_win else -stake_win
                pnl_plc  = stake_plc*(odds_place[i]-1.0) if won_plc else -stake_plc
                pnl = pnl_win + pnl_plc
                race_pnl += pnl

                pick_rows.append({
                    "date": g["_date"].iloc[0].date() if pd.notna(g["_date"].iloc[0]) else None,
                    "venue": g["meeting_venue"].iloc[0] if "meeting_venue" in g.columns else "",
                    "race_number": g["race_number"].iloc[0] if "race_number" in g.columns else None,
                    "runner_name": g["runner_name"].iloc[i] if "runner_name" in g.columns else "",
                    "bet_type": "each_way",
                    "odds_win": float(odds_win[i]) if np.isfinite(odds_win[i]) else np.nan,
                    "odds_place": float(odds_place[i]) if np.isfinite(odds_place[i]) else np.nan,
                    "p_blend_win": float(p_bl_win[i]),
                    "p_blend_place": float(p_bl_place[i]) if np.isfinite(p_bl_place[i]) else float(p_mod_place[i]),
                    "edge_pct_win": float(100.0*edge_win[i]),
                    "edge_pct_place": float(100.0*edge_place[i]) if np.isfinite(edge_place[i]) else np.nan,
                    "kelly_pct_win": float(k_win),
                    "kelly_pct_place": float(k_plc),
                    "stake_win": float(stake_win),
                    "stake_place": float(stake_plc),
                    "won_win": int(won_win),
                    "won_place": int(won_plc),
                    "pnl": float(pnl),
                    "pnl_win": float(pnl_win),
                    "pnl_place": float(pnl_plc),
                    "race_key": key,
                })

        bankroll += race_pnl
        equity_rows.append({"date": g["_date"].iloc[0], "race_key": key, "bankroll": bankroll})

        if verbose:
            print(f"[bt]   picks={len(chosen)} race_pnl={race_pnl:.2f} bankroll={bankroll:.2f}", flush=True)

    # build outputs
    picks_df = pd.DataFrame(pick_rows)
    eq = pd.DataFrame(equity_rows).sort_values("date")

    if picks_df.empty:
        metrics = Metrics(
            total_profit=0.0,
            roi=0.0,
            picks=0,
            hit_rate=0.0,
            kelly_avg=0.0,
            max_drawdown=0.0,
            sharpe=0.0,
            final_bankroll=float(bankroll),
        )
        if verbose:
            print(f"[bt] done. races={total_groups} picks=0 final_bankroll={bankroll:.2f}", flush=True)
        return metrics, picks_df

    total_profit = float(picks_df["pnl"].sum())
    # helper to safely sum a column that may not exist
    def _col_sum(df: pd.DataFrame, col: str) -> float:
        return float(df[col].sum()) if col in df.columns else 0.0

    total_staked = (
        _col_sum(picks_df, "stake")
        + _col_sum(picks_df, "stake_win")
        + _col_sum(picks_df, "stake_place")
    )

    roi = (total_profit / total_staked) if total_staked > 0 else 0.0

    # define hit_rate as proportion of bets with positive pnl
    hit_rate = float((picks_df["pnl"] > 0).mean())
    # average of available Kelly cols
    if "kelly_pct" in picks_df.columns:
        kelly_avg = float(picks_df["kelly_pct"].mean())
    else:
        kcols = [c for c in ["kelly_pct_win","kelly_pct_place"] if c in picks_df.columns]
        kelly_avg = float(pd.concat([picks_df[c] for c in kcols], axis=1).mean(axis=1).mean()) if kcols else 0.0

    eq["ret"] = eq["bankroll"].pct_change().fillna(0.0)
    sharpe = 0.0
    if eq["ret"].std(ddof=1) > 0:
        sharpe = float(eq["ret"].mean() / eq["ret"].std(ddof=1) * np.sqrt(252))

    eq["peak"] = eq["bankroll"].cummax()
    drawdown = (eq["bankroll"] - eq["peak"]) / eq["peak"]
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    metrics = Metrics(
        total_profit=total_profit,
        roi=float(roi),
        picks=int(len(picks_df)),
        hit_rate=hit_rate,
        kelly_avg=kelly_avg,
        max_drawdown=max_dd,
        sharpe=sharpe,
        final_bankroll=float(bankroll),
    )
    if verbose:
        print(f"[bt] done. races={total_groups} picks={len(picks_df)} final_bankroll={bankroll:.2f}", flush=True)
    return metrics, picks_df

# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Parquet or CSV with historical races")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--market", choices=["fixed","tote"], default="fixed")
    ap.add_argument("--blend", choices=["linear","logit"], default="linear")
    ap.add_argument("--w", type=float, default=0.4)
    ap.add_argument("--overround", choices=["proportional","power","none"], default="proportional")
    ap.add_argument("--min_edge", type=float, default=1.0)
    ap.add_argument("--min_kelly", type=float, default=0.3)
    ap.add_argument("--kelly_frac", type=float, default=0.25)
    ap.add_argument("--max_picks", type=int, default=4)
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--dynamic_bankroll", action="store_true")
    ap.add_argument("--bet_type", choices=["win","place","each_way"], default="win")

    # grid search lists
    ap.add_argument("--grid_w", nargs="*", type=float, default=[])
    ap.add_argument("--grid_blend", nargs="*", default=[])
    ap.add_argument("--grid_overround", nargs="*", default=[])
    ap.add_argument("--grid_min_edge", nargs="*", type=float, default=[])
    ap.add_argument("--grid_min_kelly", nargs="*", type=float, default=[])
    return ap.parse_args()

def main():
    args = parse_args()

    # load data
    if args.data.lower().endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    # date filter
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        if args.start:
            df = df[df["date"] >= pd.to_datetime(args.start).date()]
        if args.end:
            df = df[df["date"] <= pd.to_datetime(args.end).date()]

    # choose odds and result columns
    fixed_col = choose_col(df, ODDS_FIXED_CANDIDATES)
    tote_col  = choose_col(df, ODDS_TOTE_CANDIDATES)
    result_col= choose_col(df, RESULT_COLS)
    print(f"Using cols -> fixed: {fixed_col} | tote: {tote_col} | result: {result_col}")

    if not fixed_col and not tote_col:
        raise SystemExit("Could not find any odds column like ffwin/fixed_win or win/tote_win")
    if not result_col:
        raise SystemExit("Could not find a finish result column like finish_rank")

    params = Params(
        market=args.market,
        blend=args.blend,
        w=args.w,
        overround=args.overround,
        min_edge=args.min_edge,
        min_kelly=args.min_kelly,
        kelly_frac=args.kelly_frac,
        max_picks=args.max_picks,
        dynamic_bankroll=args.dynamic_bankroll,
        bankroll0=args.bankroll,
        bet_type=args.bet_type,
    )

    # grid search or single run
    grid_w          = args.grid_w or [params.w]
    grid_blend      = args.grid_blend or [params.blend]
    grid_overround  = args.grid_overround or [params.overround]
    grid_min_edge   = args.grid_min_edge or [params.min_edge]
    grid_min_kelly  = args.grid_min_kelly or [params.min_kelly]

    results = []
    best = None
    best_key = None

    for w in grid_w:
        for blend in grid_blend:
            for orr in grid_overround:
                for me in grid_min_edge:
                    for mk in grid_min_kelly:
                        p = Params(
                            market=params.market,
                            blend=blend,
                            w=w,
                            overround=orr,
                            min_edge=me,
                            min_kelly=mk,
                            kelly_frac=params.kelly_frac,
                            max_picks=params.max_picks,
                            dynamic_bankroll=params.dynamic_bankroll,
                            bankroll0=params.bankroll0,
                            bet_type=params.bet_type,
                        )
                        metrics, picks_df = backtest(df, p, fixed_col or tote_col, tote_col, result_col)
                        row = {
                            "bet_type": params.bet_type,
                            "w": w, "blend": blend, "overround": orr,
                            "min_edge": me, "min_kelly": mk,
                            "picks": metrics.picks,
                            "profit": metrics.total_profit,
                            "roi": metrics.roi,
                            "hit_rate": metrics.hit_rate,
                            "kelly_avg": metrics.kelly_avg,
                            "max_drawdown": metrics.max_drawdown,
                            "sharpe": metrics.sharpe,
                            "final_bankroll": metrics.final_bankroll,
                        }
                        results.append(row)
                        # save per-run picks
                        safe_name = f"bet{params.bet_type}_w{w}_blend{blend}_orr{orr}_me{me}_mk{mk}".replace(".","p")
                        outdir = "backtest_logs"
                        os.makedirs(outdir, exist_ok=True)
                        picks_path = os.path.join(outdir, f"picks_{safe_name}.csv")
                        picks_df.to_csv(picks_path, index=False)
                        # pick best by final bankroll then Sharpe
                        score = (metrics.final_bankroll, metrics.sharpe)
                        if best is None or score > best:
                            best = score
                            best_key = row

    res_df = pd.DataFrame(results).sort_values(["final_bankroll","sharpe"], ascending=[False, False])
    print("\n=== Top settings ===")
    print(res_df.head(10).to_string(index=False))

    out_csv = "backtest_grid_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved full grid results to {out_csv}")

if __name__ == "__main__":
    main()
