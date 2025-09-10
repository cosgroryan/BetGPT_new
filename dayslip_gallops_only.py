#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, os, math, json, time, pathlib, textwrap
from typing import Dict, Any, List, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

BASE = "https://json.tab.co.nz"

def http_get_json(url: str, debug: bool=False) -> Dict[str, Any]:
    try:
        if debug: print(f"[http] GET {url} -> ", end="", flush=True)
        with urlopen(Request(url, headers={"User-Agent":"dayslip/1.0"}), timeout=20) as r:
            data = r.read()
        obj = json.loads(data.decode("utf-8"))
        if debug: print("OK")
        return obj
    except HTTPError as e:
        if debug: print(f"HTTP {e.code}")
        raise
    except URLError as e:
        if debug: print(f"URLError {e.reason}")
        raise

def dump_json(obj: Any, path: str, debug: bool=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    if debug: print(f"[dump] wrote {path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD (NZ time)")
    p.add_argument("--bet_type", default="place", choices=["win","place"])
    p.add_argument("--market", type=str, default="tote",
                    choices=["tote", "fixed"], help="Market type (tote or fixed)")
# keeping as-is for your workflow
    p.add_argument("--blend", default="logit", choices=["logit","linear","none"])
    p.add_argument("--w", type=float, default=0.6)
    p.add_argument("--overround", default="none", choices=["none","proportional"])

    p.add_argument("--min_edge", type=float, default=0.0, help="minimum edge (fraction), e.g. 0.02 = 2%")
    p.add_argument("--min_kelly", type=float, default=0.0, help="minimum Kelly fraction threshold")
    p.add_argument("--kelly_frac", type=float, default=0.25, help="fraction of Kelly to bet")
    p.add_argument("--bankroll", type=float, default=100.0)
    p.add_argument("--max_picks", type=int, default=1)

    # NEW: model flags
    p.add_argument("--model", default="none", choices=["none","shadow"],
                   help="none = skip model; shadow = market-prob shadow with boost")
    p.add_argument("--shadow_edge", type=float, default=0.02,
                   help="shadow model boost (e.g. 0.03 = +3% on fair prob before re-normalizing)")

    # debug / dumps
    p.add_argument("--debug", action="store_true")
    p.add_argument("--dump-dir", default=None)

    # expert: allow dogs/harness fallback if you want
    p.add_argument("--allow_non_gallops", action="store_true",
                   help="If set, will include non-G meetings only when no G meetings available")
    return p.parse_args()

def header():
    print("------------------------------------------------------------------------")

def normalize_proportional(ps: List[float]) -> List[float]:
    s = sum(p for p in ps if p is not None)
    return [ (p/s if s > 0 and p is not None else 0.0) for p in ps ]

def implied_prob_from_odds(odds: float) -> float:
    # Tote dividends are typically net-of-takeout. We’ll treat odds as decimal returns
    # and derive a naive implied prob = 1 / odds.
    return 1.0 / odds if odds and odds > 0 else 0.0

def apply_overround(probs: List[float], mode: str) -> List[float]:
    if mode == "none":  # do nothing
        return probs
    if mode == "proportional":
        return normalize_proportional(probs)
    return probs

def kelly_fraction(p: float, d: float) -> float:
    # d = decimal odds; edge = p*d - 1; full Kelly = (d*p - 1)/(d - 1)
    if not d or d <= 1.0: return 0.0
    return max(0.0, (d*p - 1.0) / (d - 1.0))

def shadow_boost(probs: List[float], boost: float) -> List[float]:
    # Multiply by (1+boost) then renormalize, to create a mild “belief” edge vs market
    boosted = [(p * (1.0 + boost)) for p in probs]
    return normalize_proportional(boosted)

def choose_price(entry: Dict[str, Any], bet_type: str) -> float:
    # exact API field names from your samples
    # - WIN price is at entry["win"]
    # - PLACE price is at entry["plc"]
    # ignore fixed/futures (ffwin/ffplc), and skip scratched runners (scr=True)
    if entry.get("scr"):
        return 0.0
    if bet_type == "win":
        return entry.get("win") or 0.0
    else:
        return entry.get("plc") or 0.0

def collect_market_probs(race: Dict[str, Any], bet_type: str, overround: str, debug: bool=False) -> Tuple[List[float], List[float]]:
    prices = []
    for e in race.get("entries", []):
        price = choose_price(e, bet_type)
        prices.append(price if price else 0.0)
    probs = [implied_prob_from_odds(x) for x in prices]
    probs = apply_overround(probs, overround)
    if debug:
        print(f"[odds] prices={prices}")
        print(f"[odds] market-implied probs (after '{overround}')={['{:.3f}'.format(p) for p in probs]}")
    return probs, prices

def model_predict(probs_mkt: List[float], model: str, shadow_edge: float, debug: bool=False) -> List[float]:
    if model == "none":
        # You can still bet using raw market probs; or return empty to only run with your external model
        return probs_mkt
    if model == "shadow":
        preds = shadow_boost(probs_mkt, shadow_edge)
        if debug:
            print(f"[model] shadow_edge={shadow_edge:.4f}; preds={['{:.3f}'.format(p) for p in preds]}")
        return preds
    return probs_mkt

def main():
    args = parse_args()

    # Fetch schedule (live) and filter to Gallops
    sched_url = f"{BASE}/schedule/{args.date}"
    schedule = http_get_json(sched_url, debug=args.debug)
    if args.dump_dir:
        dump_json(schedule, os.path.join(args.dump_dir, f"schedule_{args.date}.json"), debug=args.debug)

    # meetings have exact header "type": "G" (gallops), "GR" (greyhounds), "H" (harness)
    meetings = schedule.get("meetings", [])
    accepted_types = ["G"]
    g_meetings = [m for m in meetings if (m.get("type") in accepted_types)]
    if args.debug:
        for m in meetings:
            print(f"[filter] meeting name='{m.get('name')}' type='{m.get('type')}'")
        print(f"[sched] total meetings: {len(meetings)}; accepted (G only): {len(g_meetings)}")

    if not g_meetings and args.allow_non_gallops:
        g_meetings = meetings[:]  # fallback if requested

    # Fetch odds (live) for the day; exact sample schema = {"date": "...", "meetings":[{ "id", "number", "races":[{...}]}]}
    odds_url = f"{BASE}/odds/{args.date}"
    odds_doc = http_get_json(odds_url, debug=args.debug)
    if args.dump_dir:
        dump_json(odds_doc, os.path.join(args.dump_dir, f"odds_{args.date}.json"), debug=args.debug)

    # Build race-id -> race odds map for fast lookup
    odds_by_race: Dict[str, Dict[str, Any]] = {}
    for mtg in odds_doc.get("meetings", []):
        for r in mtg.get("races", []):
            rid = r.get("id")
            if rid: odds_by_race[rid] = r

    # Iterate gallops and score entries
    picks: List[Dict[str, Any]] = []
    for m in g_meetings:
        for race in m.get("races", []):
            rid = race.get("id")
            rnum = race.get("number")
            rname = race.get("name") or f"Race {rnum}"
            rstate = race.get("status") or "OK"
            if args.debug:
                print(f"[race] consider {m.get('name')} R{rnum} {rname} status={rstate}")

            # pull odds block for this race id
            r_odds = odds_by_race.get(rid)
            if not r_odds:
                if args.debug: print(f"[skip] no odds block for race_id={rid}")
                continue

            # copy entries from odds (they contain 'win','plc','scr', etc)
            entries = r_odds.get("entries", [])
            if not entries:
                if args.debug: print(f"[skip] race has no entries with odds yet")
                continue

            # market probs & prices
            probs_mkt, prices = collect_market_probs(r_odds, args.bet_type, args.overround, debug=args.debug)

            # Model predictions
            preds = model_predict(probs_mkt, args.model, args.shadow_edge, debug=args.debug)

            # Compute edges and Kelly
            race_rows = []
            for idx, e in enumerate(entries):
                if e.get("scr"):  # scratched
                    if args.debug: print(f"[runner] #{e.get('number')} scratched")
                    continue
                price = prices[idx]
                if price <= 0:
                    if args.debug: print(f"[runner] #{e.get('number')} has no {args.bet_type} price yet")
                    continue

                p = preds[idx]
                d = price
                edge = p * d - 1.0
                k_full = kelly_fraction(p, d)
                k_size = k_full * args.kelly_frac
                reasons = []
                if edge < args.min_edge:
                    reasons.append(f"edge {edge:.3f} < min_edge {args.min_edge:.3f}")
                if k_full < args.min_kelly:
                    reasons.append(f"kelly {k_full:.3f} < min_kelly {args.min_kelly:.3f}")

                if args.debug:
                    print(f"[runner] #{e.get('number')}: price={d:.3f} p={p:.3f} edge={edge:.3f} kelly={k_full:.3f} -> stake={k_size*args.bankroll:.2f}"
                          + ("" if not reasons else f"  |  drop: {', '.join(reasons)}"))

                if not reasons:
                    race_rows.append({
                        "meeting": m.get("name"),
                        "race_no": rnum,
                        "race_name": rname,
                        "race_id": rid,
                        "runner_no": e.get("number"),
                        "bet_type": args.bet_type,
                        "price": d,
                        "prob_pred": p,
                        "edge": edge,
                        "kelly": k_full,
                        "stake": k_size * args.bankroll
                    })

            # keep the best candidate(s) in this race
            race_rows.sort(key=lambda r: (r["edge"], r["kelly"]), reverse=True)
            if race_rows:
                picks.extend(race_rows[: args.max_picks])

    # Finalize picks list (cap to global max_picks if you prefer)
    # (Keeping per-race cap as requested; if you want global cap, uncomment below)
    # picks = sorted(picks, key=lambda r: (r["edge"], r["kelly"]), reverse=True)[: args.max_picks]

    # Output
    print(f"\nDaySlip — {args.date} (Gallops only)")
    print(f"bet_type={args.bet_type} market={args.market} blend={args.blend} w={args.w} overround={args.overround}")
    print(f"min_edge={args.min_edge*100:.1f}% min_kelly={args.min_kelly*100:.1f}% kelly_frac={args.kelly_frac} bankroll={args.bankroll:.1f} max_picks={args.max_picks}")
    print(f"Model={args.model}" + (f" (shadow_edge={args.shadow_edge:.3f})" if args.model=='shadow' else ""))

    if not picks:
        print("Total picks: 0")
        header()
        return

    # Pretty print & save files
    import csv
    os.makedirs("dayslips", exist_ok=True)
    txt_path = f"dayslips/dayslip_{args.date}_gallops.txt"
    csv_path = f"dayslips/dayslip_{args.date}_gallops.csv"

    with open(txt_path, "w", encoding="utf-8") as ftxt, open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(picks[0].keys()))
        w.writeheader()
        for p in picks:
            line = (f"{p['meeting']} R{p['race_no']} #{p['runner_no']} "
                    f"{p['bet_type']} @ {p['price']:.2f} | p={p['prob_pred']:.3f} "
                    f"edge={p['edge']:.3f} kelly={p['kelly']:.3f} stake={p['stake']:.2f}")
            print(line)
            ftxt.write(line + "\n")
            w.writerow(p)

    header()
    print(f"Saved: {txt_path}")
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()
