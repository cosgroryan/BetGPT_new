#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import sys
import math
import urllib.request
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# --- Make saved pickle artefacts loadable (maps __main__.PreprocessArtifacts) ---
import sys as _sys, pytorch_pre as _pp
_sys.modules['__main__'] = _pp
from pytorch_pre import load_model_and_predict

# =========================
# Web fetch
# =========================

def fetch_schedule_json(date_str: str, meet_no: int, race_no: int) -> dict:
    url = f"https://json.tab.co.nz/schedule/{date_str}/{meet_no}/{race_no}"
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))

# =========================
# Helpers
# =========================

def _parse_money_to_float(s):
    if s is None:
        return np.nan
    s = re.sub(r"[^0-9.]", "", str(s))
    try:
        return float(s) if s else np.nan
    except ValueError:
        return np.nan

def _to_float(x):
    try:
        return float(x) if x is not None and str(x).strip() != "" else np.nan
    except ValueError:
        return np.nan

def _norm_name(s: str) -> str:
    return " ".join(str(s).strip().upper().split())

def logit(p, eps=1e-9):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))

def inv_logit(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))

def blend_with_market(win_model, win_market, alpha=0.6):
    """
    Logit blend model vs market, then renormalise so sum(p)=1 across the field.
    alpha in [0,1]: 1 trusts the model most; 0 trusts the market most.
    """
    win_model = np.asarray(win_model, dtype=float)
    win_market = np.asarray(win_market, dtype=float)
    z = alpha * logit(win_model) + (1 - alpha) * logit(win_market)
    p = inv_logit(z)
    s = p.sum()
    return p / s if s > 0 else p

def probs_to_decimal_odds(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return 1.0 / p

# =========================
# JSON -> DataFrame
# =========================

def _extract_fixed_win(entry: dict):
    """
    Try several common shapes to find fixed win odds in schedule JSON.
    Returns float or NaN.
    """
    # direct
    for k in ("fixed_win", "win"):
        if k in entry:
            v = _to_float(entry.get(k))
            if not math.isnan(v):
                return v
    # nested guesses
    nests = [
        ("fixedOdds", "win"),
        ("fixed", "win"),
        ("odds", "win"),
        ("prices", "win"),
    ]
    for a, b in nests:
        node = entry.get(a)
        if isinstance(node, dict) and b in node:
            v = _to_float(node.get(b))
            if not math.isnan(v):
                return v
    return np.nan

def schedule_json_to_df(obj: dict) -> pd.DataFrame:
    """
    Flatten schedule JSON for ONE RACE into rows=runners with model features.
    If fixed win odds are present, derive market implied probabilities (overround-stripped)
    and fav_rank (1 = shortest).
    """
    rows = []
    date = obj.get("date") or obj.get("day") or obj.get("meetingDate")
    for m in obj.get("meetings", []):
        meeting_number = m.get("number") or m.get("meetingNumber")
        meeting_country = m.get("country")
        meeting_venue = m.get("venue") or m.get("meetingName")
        meeting_id = m.get("id") or m.get("meetingId")

        # single race endpoint usually has exactly one race
        races = m.get("races") or obj.get("races") or []
        for r in races:
            race_id = r.get("id") or r.get("raceId")
            race_number = r.get("number") or r.get("raceNumber")
            race_class = r.get("class")
            race_track = r.get("track") or r.get("trackCondition")
            race_weather = r.get("weather")
            race_name = r.get("name") or r.get("raceName")
            race_venue = r.get("venue") or meeting_venue
            race_distance_m = _to_float(r.get("length") or r.get("distanceMeters"))
            race_length = race_distance_m
            stake = _parse_money_to_float(r.get("stake"))

            tmp = []
            entries = r.get("entries") or r.get("runners") or []
            for e in entries:
                if e.get("scratched", False) or e.get("isScratched", False):
                    continue
                fixed_win = _extract_fixed_win(e)
                barrier = e.get("barrier") if e.get("barrier") is not None else e.get("draw")
                runner_no = e.get("number") or e.get("runnerNumber") or e.get("runner_number")
                jockey = e.get("jockey") or e.get("jockeyName") or "UNK"
                runner_name = e.get("name") or e.get("runnerName") or "UNK"

                row = {
                    "date": date,
                    "meeting_id": meeting_id,
                    "race_id": race_id,
                    "race_name": race_name,
                    "meeting_number": meeting_number,
                    "race_number": race_number,
                    "race_distance_m": race_distance_m,
                    "race_length": race_length,
                    "stake": stake,
                    "entrant_weight": _to_float(e.get("weight")),
                    "race_class": race_class,
                    "race_track": race_track,
                    "race_weather": race_weather,
                    "meeting_country": meeting_country,
                    "meeting_venue": race_venue,
                    "race_class_sched": race_class,
                    "race_number_sched": race_number,
                    "entrant_barrier": str(barrier) if barrier is not None else "UNK",
                    "runner_number": int(runner_no) if runner_no is not None else np.nan, 
                    "entrant_jockey": jockey,
                    "runner_name": _norm_name(runner_name),
                    "fixed_win": fixed_win,
                }
                tmp.append(row)

            if not tmp:
                continue

            df_tmp = pd.DataFrame(tmp)
            # market implied probs (strip overround so sum ≈ 1)
            if df_tmp["fixed_win"].notna().any():
                ip = 1.0 / df_tmp["fixed_win"]
                s = ip.sum()
                df_tmp["implied_p"] = (ip / s) if s > 0 else ip
                # 1 = shortest odds
                df_tmp["fav_rank"] = df_tmp["fixed_win"].rank(ascending=True, method="min").astype(int)

            rows.extend(df_tmp.to_dict("records"))

    return pd.DataFrame(rows)

# =========================
# Rank -> Position probabilities (PL)
# =========================

def _scores_from_pred_rank(pred_rank: np.ndarray, tau: float = 0.4) -> np.ndarray:
    """Lower predicted rank = stronger; convert to positive PL scores."""
    return np.exp(-np.asarray(pred_rank, dtype=float) / float(tau))

def _pl_sample_order(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scores = scores.astype(float).copy()
    n = len(scores)
    order = np.empty(n, dtype=int)
    alive = np.arange(n)
    s = scores.copy()
    for pos in range(n):
        p = s / s.sum()
        i = rng.choice(len(alive), p=p)
        order[pos] = alive[i]
        alive = np.delete(alive, i)
        s = np.delete(s, i)
    return order

def estimate_position_probs_for_race_df(
    race_df: pd.DataFrame,
    tau: float = 0.4,
    n_samples: int = 4000,
    alpha_model_weight: float = 0.6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - model-only win/top3 probs + 90% CI + confidence %
      - blended win prob (model vs market) + fair NZ decimal odds
      - fav_rank if available
    """
    # 1) Predict finish ranks with your trained model
    pred_out = load_model_and_predict(race_df)
    if isinstance(pred_out, dict) and "pred_rank" in pred_out:
        pred_rank = np.asarray(pred_out["pred_rank"])
        new_horse = np.asarray(pred_out.get("new_horse", [False]*len(pred_rank)))
    else:
        pred_rank = np.asarray(pred_out)
        new_horse = np.array([False]*len(pred_rank))

    # 2) Monte-Carlo PL sampling for position distribution
    scores = _scores_from_pred_rank(pred_rank, tau=tau)
    rng = np.random.default_rng(seed)
    n = len(scores)
    pos_counts = np.zeros((n, n), dtype=np.int32)
    for _ in range(n_samples):
        order = _pl_sample_order(scores, rng)
        for pos, idx in enumerate(order):
            pos_counts[idx, pos] += 1
    pos_probs = pos_counts / float(n_samples)
    win_model = pos_probs[:, 0]
    top3_model = pos_probs[:, :min(3, n)].sum(axis=1)

    # 3) Confidence (sampling) for win prob: 90% CI from binomial
    z = 1.645  # 90%
    se_model = np.sqrt(np.clip(win_model * (1 - win_model) / n_samples, 0.0, 1.0))
    ci_low_model = np.clip(win_model - z * se_model, 0, 1)
    ci_high_model = np.clip(win_model + z * se_model, 0, 1)
    confidence_pct = np.clip(1.0 - (ci_high_model - ci_low_model), 0.0, 1.0) * 100.0

    # 4) Market blend for win prob (if implied_p present)
    if "implied_p" in race_df.columns and race_df["implied_p"].notna().any():
        market_p = race_df["implied_p"].to_numpy()
        win_blend = blend_with_market(win_model, market_p, alpha=alpha_model_weight)
        # crude SE propagation (market treated as noiseless)
        se_blend = se_model * alpha_model_weight
        ci_low_blend = np.clip(win_blend - z * se_blend, 0, 1)
        ci_high_blend = np.clip(win_blend + z * se_blend, 0, 1)
        confidence_blend_pct = np.clip(1.0 - (ci_high_blend - ci_low_blend), 0.0, 1.0) * 100.0
    else:
        win_blend = win_model.copy()
        ci_low_blend = ci_low_model.copy()
        ci_high_blend = ci_high_model.copy()
        confidence_blend_pct = confidence_pct.copy()

    # 5) Assemble output
    out = pd.DataFrame({
        "runner_name": race_df.get("runner_name", pd.Series([None]*n)).values,
        "runner_number": race_df.get("runner_number", pd.Series([np.nan]*n)).values,
        "entrant_barrier": race_df.get("entrant_barrier", pd.Series([None]*n)).values,
        "entrant_jockey": race_df.get("entrant_jockey", pd.Series([None]*n)).values,
        "fav_rank": race_df.get("fav_rank", pd.Series([np.nan]*n)).values,
        "pred_rank": pred_rank,
        # model-only
        "win_prob_model": win_model,
        "top3_prob_model": top3_model,
        "win_ci90_low_model": ci_low_model,
        "win_ci90_high_model": ci_high_model,
        "confidence_model_%": confidence_pct,
        # blended (use this)
        "win_prob": win_blend,
        "win_ci90_low": ci_low_blend,
        "win_ci90_high": ci_high_blend,
        "confidence_%": confidence_blend_pct,
        "new_horse": new_horse,
    })

    # Fair NZ decimal odds (no margin)
    out["win_%"] = (out["win_prob"] * 100.0).round(2)
    out["win_$fair"] = probs_to_decimal_odds(out["win_prob"]).round(2)
    out["top3_%"] = (out["top3_prob_model"] * 100.0).round(2)
    out["top3_$fair"] = probs_to_decimal_odds(out["top3_prob_model"]).round(2)

    # Pretty CI strings
    out["win_%_90CI"] = (out["win_ci90_low"] * 100).round(1).astype(str) + "–" + (out["win_ci90_high"] * 100).round(1).astype(str)

    # Order by blended win prob
    out = out.sort_values("win_prob", ascending=False).reset_index(drop=True)
    return out

# =========================
# CLI
# =========================

def main(meet_no: int, race_no: int, date_str: str, tau: float = 0.4, n_samples: int = 4000, alpha_model_weight: float = 0.6):
    obj = fetch_schedule_json(date_str, meet_no, race_no)
    race_df = schedule_json_to_df(obj)
    if race_df.empty:
        print("No eligible (unscratched) runners found in schedule JSON.")
        return

    probs = estimate_position_probs_for_race_df(
        race_df,
        tau=tau,
        n_samples=n_samples,
        alpha_model_weight=alpha_model_weight,
    )

    cols = [
        "runner_number", "runner_name","entrant_barrier","entrant_jockey","fav_rank","pred_rank",
        "win_%","win_$fair","top3_%","top3_$fair","confidence_%","win_%_90CI"
    ]
    print(probs[cols].to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python infer_from_schedule_json.py <meet_no> <race_no> [date=YYYY-MM-DD(todays local)] [tau=0.4] [n_samples=4000] [alpha_model_weight=0.6]")
        sys.exit(1)
    meet_no = int(sys.argv[1])
    race_no = int(sys.argv[2])
    date_str = sys.argv[3] if len(sys.argv) > 3 else datetime.now().strftime("%Y-%m-%d")
    tau = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
    n_samples = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
    alpha = float(sys.argv[6]) if len(sys.argv) > 6 else 0.6
    main(meet_no, race_no, date_str, tau=tau, n_samples=n_samples, alpha_model_weight=alpha)
