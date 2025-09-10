#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import os
from datetime import datetime
import importlib.util
import inspect

import numpy as np
import pandas as pd

def import_from_path(path, mod_name="backtest_blend"):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def append_row_csv(out_path: str, row: dict, header_fields: list):
    # create dir if needed
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    df = pd.DataFrame([row], columns=header_fields)
    df.to_csv(out_path, mode="a", index=False, header=write_header)

def main():
    ap = argparse.ArgumentParser(description="Sweep backtest_blend.py settings and log each result as it finishes.")
    ap.add_argument("--backtest_path", default="backtest_blend.py", help="Path to backtest_blend.py")
    ap.add_argument("--data", required=True, help="CSV or Parquet of races with odds and results")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD filter start")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD filter end")

    # grids. pass space-separated values
    ap.add_argument("--markets", nargs="*", default=["tote", "fixed"])
    ap.add_argument("--blends", nargs="*", default=["logit", "linear"])
    ap.add_argument("--weights", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--overrounds", nargs="*", default=["proportional", "power", "none"])
    ap.add_argument("--min_edges", nargs="*", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--min_kellys", nargs="*", type=float, default=[0.0, 0.3, 0.5])
    ap.add_argument("--kelly_fracs", nargs="*", type=float, default=[0.1, 0.2, 0.25])
    ap.add_argument("--max_picks_list", nargs="*", type=int, default=[2, 3, 4])
    ap.add_argument("--dynamic_bankroll", action="store_true")

    # new: bet types
    ap.add_argument("--bet_types", nargs="*", default=["win"], help="win place eachway (depends on backtest support)")

    ap.add_argument("--out", default=None, help="Output CSV. Default auto-named with timestamp.")
    args = ap.parse_args()

    bb = import_from_path(args.backtest_path, "backtest_blend")

    # load data once
    if args.data.lower().endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    # date filter if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        if args.start:
            df = df[df["date"] >= pd.to_datetime(args.start).date()]
        if args.end:
            df = df[df["date"] <= pd.to_datetime(args.end).date()]

    # pick columns using constants inside backtest_blend
    fixed_col  = bb.choose_col(df, bb.ODDS_FIXED_CANDIDATES)
    tote_col   = bb.choose_col(df, bb.ODDS_TOTE_CANDIDATES)
    result_col = bb.choose_col(df, bb.RESULT_COLS)
    if not fixed_col and not tote_col:
        raise SystemExit("No odds columns found. Expected one of: " + ", ".join(bb.ODDS_FIXED_CANDIDATES + bb.ODDS_TOTE_CANDIDATES))
    if not result_col:
        raise SystemExit("No result column found. Expected one of: " + ", ".join(bb.RESULT_COLS))

    print(f"Using cols -> fixed: {fixed_col} | tote: {tote_col} | result: {result_col}", flush=True)

    # set output path
    out_path = args.out or f"sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Writing incremental results to {out_path}", flush=True)

    # header fields for the csv
    header_fields = [
        "market","blend","w","overround","min_edge","min_kelly","kelly_frac","max_picks","bet_type",
        "picks","profit","roi","hit_rate","kelly_avg","max_drawdown","sharpe","final_bankroll"
    ]

    # work out if the backtest() supports bet_type kwarg
    backtest_sig = inspect.signature(bb.backtest)
    supports_bet_type = "bet_type" in backtest_sig.parameters

    # build combos
    combos = list(itertools.product(
        args.markets, args.blends, args.weights, args.overrounds,
        args.min_edges, args.min_kellys, args.kelly_fracs, args.max_picks_list, args.bet_types
    ))
    print(f"Total combos: {len(combos)}", flush=True)

    # run
    for i, (market, blend, w, orr, me, mk, kf, mp, bt) in enumerate(combos, 1):
        params = bb.Params(
            market=market,
            blend=blend,
            w=float(w),
            overround=orr,
            min_edge=float(me),
            min_kelly=float(mk),
            kelly_frac=float(kf),
            max_picks=int(mp),
            dynamic_bankroll=bool(args.dynamic_bankroll),
            bankroll0=1000.0,
        )
        try:
            # build kwargs safely
            kwargs = dict(
                df=df.copy(),
                params=params,
                odds_col_fixed=fixed_col or tote_col,
                odds_col_tote=tote_col,
                result_col=result_col,
                verbose=False,
            )
            if supports_bet_type:
                kwargs["bet_type"] = bt  # win/place/eachway if your backtest supports it

            metrics, picks_df = bb.backtest(**kwargs)

            row = {
                "market": market,
                "blend": blend,
                "w": w,
                "overround": orr,
                "min_edge": me,
                "min_kelly": mk,
                "kelly_frac": kf,
                "max_picks": mp,
                "bet_type": bt,
                "picks": metrics.picks,
                "profit": metrics.total_profit,
                "roi": metrics.roi,
                "hit_rate": metrics.hit_rate,
                "kelly_avg": metrics.kelly_avg,
                "max_drawdown": metrics.max_drawdown,
                "sharpe": metrics.sharpe,
                "final_bankroll": metrics.final_bankroll,
            }

            append_row_csv(out_path, row, header_fields)
            print(f"[{i}/{len(combos)}] {row}", flush=True)

        except Exception as e:
            # log failure as a row too, so you can spot patterns later
            row = {
                "market": market,
                "blend": blend,
                "w": w,
                "overround": orr,
                "min_edge": me,
                "min_kelly": mk,
                "kelly_frac": kf,
                "max_picks": mp,
                "bet_type": bt,
                "picks": np.nan,
                "profit": np.nan,
                "roi": np.nan,
                "hit_rate": np.nan,
                "kelly_avg": np.nan,
                "max_drawdown": np.nan,
                "sharpe": np.nan,
                "final_bankroll": np.nan,
            }
            append_row_csv(out_path, row, header_fields)
            print(f"[{i}/{len(combos)}] FAILED market={market} blend={blend} w={w} overround={orr} "
                  f"min_edge={me} min_kelly={mk} kelly_frac={kf} max_picks={mp} bet_type={bt} :: {e}", flush=True)

    print(f"\nAll done. Results appended to {out_path}", flush=True)

if __name__ == "__main__":
    main()
