#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, urllib.request, contextlib
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---- Local modules (must live alongside this script) ----
import pytorch_pre as nnmodel          # expects: model_regression.pth + preprocess.pkl
import lgbm_rank as gbm                # expects: artifacts_gbm/{model_lgbm_reg.txt, model_lgbm.txt, preprocess_gbm.pkl}

ART_GBM_DIR    = getattr(gbm, "ART_DIR", "artifacts_gbm")
GBM_REG_NAME   = "model_lgbm_reg.txt"
GBM_RANK_NAME  = "model_lgbm.txt"
GBM_REG_PATH   = os.path.join(ART_GBM_DIR, GBM_REG_NAME)
GBM_RANK_PATH  = os.path.join(ART_GBM_DIR, GBM_RANK_NAME)

# ---------------- Time (NZ) ----------------
def today_nz() -> date:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Pacific/Auckland")
    except Exception:
        import pytz
        tz = pytz.timezone("Pacific/Auckland")
    return datetime.now(tz).date()

def default_day_str() -> str:
    return (today_nz() - timedelta(days=1)).isoformat()

# ---------------- HTTP ----------------
def get_json(url: str, timeout: int = 20) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

# ---------------- Filters ----------------
DOG_KEYS = ("dog", "grey", "greyhound")
HARNESS_KEYS = ("harness", "trot", "pace")
GALLOPS_KEYS = ("gallop", "thoroughbred", "tbred")

def _text(o) -> str:
    if o is None: return ""
    return str(o).strip().lower()

def is_gallops_meeting(m: dict) -> bool:
    # Prefer explicit section/code/category
    for k in ("section","code","category","sport","type","discipline"):
        s = _text(m.get(k))
        if any(x in s for x in GALLOPS_KEYS): return True
        if any(x in s for x in DOG_KEYS + HARNESS_KEYS): return False
    # Fallback: scan races for harness/grey keywords
    for r in (m.get("races") or []):
        name = _text(r.get("name") or r.get("raceName"))
        klass = _text(r.get("class"))
        if any(x in name or x in klass for x in HARNESS_KEYS + DOG_KEYS):
            return False
    return True  # default assume gallops

# ---------------- Results parsing ----------------
def fetch_results_race(day: str, meet_no: int, race_no: int) -> Optional[dict]:
    return get_json(f"https://json.tab.co.nz/results/{day}/{meet_no}/{race_no}")

def actual_top3_from_results(obj: dict) -> Tuple[List[int], Dict[int, int]]:
    finish_map: Dict[int, int] = {}
    top3: Dict[int, int] = {}
    meetings = obj.get("meetings") or [obj]
    for m in meetings:
        for r in m.get("races", []) or obj.get("races", []):
            for p in (r.get("placings") or []):
                num = p.get("number") or p.get("runnerNumber")
                rank = p.get("rank") or p.get("placing") or p.get("position")
                try: num, rank = int(num), int(rank)
                except Exception: continue
                finish_map[num] = rank
                if 1 <= rank <= 3: top3[rank] = num
            for a in (r.get("also_ran") or []):
                num = a.get("number") or a.get("runnerNumber")
                fin = a.get("finish_position") or a.get("placing") or a.get("position") or a.get("rank")
                try: num, fin = int(num), int(fin)
                except Exception: continue
                if fin > 0: finish_map[num] = fin
    actual = [top3.get(1), top3.get(2), top3.get(3)]
    return [x for x in actual if x is not None], finish_map

# ---------------- sys.modules['__main__'] alias helpers ----------------
@contextlib.contextmanager
def main_alias(mod):
    """Temporarily alias pickle's '__main__' to a given module (fixes artifact unpickling)."""
    prev = sys.modules.get("__main__")
    sys.modules["__main__"] = mod
    try:
        yield
    finally:
        if prev is not None:
            sys.modules["__main__"] = prev
        else:
            del sys.modules["__main__"]

# ---------------- Predictions ----------------

def predict_top3_nn(race_df: pd.DataFrame) -> Tuple[List[int], Optional[str]]:
    """
    Handle both return shapes from pytorch_pre.load_model_and_predict:
      - DataFrame with 'pred_rank'
      - dict with one of: 'pred_rank', 'predicted_finish', or 'score'
    """
    try:
        with main_alias(nnmodel):
            out = nnmodel.load_model_and_predict(race_df)

        # Normalise to a DataFrame with 'pred_rank'
        if isinstance(out, pd.DataFrame):
            dfp = out.copy()
            if "pred_rank" not in dfp.columns:
                # derive from predicted_finish if present
                if "predicted_finish" in dfp.columns:
                    s = pd.to_numeric(dfp["predicted_finish"], errors="coerce").values
                    dfp["pred_rank"] = s.argsort().argsort() + 1  # lower is better
                else:
                    raise ValueError("DataFrame lacks 'pred_rank' (and no 'predicted_finish').")
        elif isinstance(out, dict):
            dfp = race_df.copy()
            if "pred_rank" in out:
                dfp["pred_rank"] = np.asarray(out["pred_rank"]).ravel()
            elif "predicted_finish" in out:
                s = np.asarray(out["predicted_finish"]).ravel().astype(float)
                dfp["pred_rank"] = s.argsort().argsort() + 1  # lower is better
            elif "score" in out:
                s = np.asarray(out["score"]).ravel().astype(float)
                # For our NN this is a finish-score, so lower is better
                dfp["pred_rank"] = s.argsort().argsort() + 1
            else:
                raise ValueError("dict returned without 'pred_rank'/'predicted_finish'/'score'.")
        else:
            raise TypeError(f"Unexpected return type from NN: {type(out)}")

        top = dfp.sort_values("pred_rank").head(3)
        return [int(x) for x in top["runner_number"].tolist() if pd.notna(x)], None

    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def _predict_top3_gbm_with(race_df: pd.DataFrame, model_name: str) -> Tuple[List[int], Optional[str]]:
    """Shared GBM predictor for a specific model file."""
    try:
        model_path = os.path.join(ART_GBM_DIR, model_name)
        if not os.path.exists(model_path):
            return [], f"GBM model missing: {model_name}"
        with main_alias(gbm):  # handle preprocess_gbm.pkl pickled under __main__
            pred = gbm.load_gbm_and_predict(race_df, model_name=model_name)
        out = race_df.copy()
        out["pred_rank"] = pred["pred_rank"]
        top = out.sort_values("pred_rank").head(3)
        return [int(x) for x in top["runner_number"].tolist() if pd.notna(x)], None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"

def predict_top3_gbm_reg(race_df: pd.DataFrame) -> Tuple[List[int], Optional[str]]:
    return _predict_top3_gbm_with(race_df, GBM_REG_NAME)

def predict_top3_gbm_rank(race_df: pd.DataFrame) -> Tuple[List[int], Optional[str]]:
    return _predict_top3_gbm_with(race_df, GBM_RANK_NAME)

def codebreaker_row(pred_top3: List[int], finish_map: Dict[int,int]) -> Tuple[List[str], int, int, int]:
    codes = []; g=y=r=0
    for pos, rn in enumerate(pred_top3[:3], start=1):
        fin = finish_map.get(rn)
        if fin is None or fin <= 0: codes.append("R"); r+=1
        elif fin == pos:           codes.append("G"); g+=1
        elif 1 <= fin <= 3:        codes.append("Y"); y+=1
        else:                      codes.append("R"); r+=1
    while len(codes) < 3: codes.append("R"); r+=1
    return codes, g, y, r

# ---------------- Benchmark day (Gallops only) ----------------
def benchmark_day_gallops(day: str) -> pd.DataFrame:
    sched = get_json(f"https://json.tab.co.nz/schedule/{day}")
    if not sched or not sched.get("meetings"):
        print(f"[info] No schedule for {day}")
        return pd.DataFrame()

    rows = []
    market_rows = []  # one row per race with favourite outcomes

    for m in sched["meetings"]:
        if not is_gallops_meeting(m):
            continue

        meet_no = m.get("number") or m.get("meetingNumber")
        venue   = m.get("venue") or m.get("meetingName")
        if meet_no is None:
            continue
        try:
            meet_no = int(meet_no)
        except Exception:
            continue

        for r in (m.get("races") or []):
            race_no   = r.get("number") or r.get("raceNumber")
            race_name = r.get("name") or r.get("raceName")
            if race_no is None:
                continue
            try:
                race_no = int(race_no)
            except Exception:
                continue

            # Results
            res = fetch_results_race(day, meet_no, race_no)
            if not res:
                continue
            actual_top3, finish_map = actual_top3_from_results(res)
            if not finish_map or not actual_top3:
                continue

            # Schedule -> df for this race
            try:
                one = gbm.fetch_schedule_json(day, meet_no, race_no)
                race_df = gbm.schedule_json_to_df(one)
                race_df = race_df[pd.notna(race_df["runner_number"])].copy()
                race_df["runner_number"] = race_df["runner_number"].astype(int, errors="ignore")
                if race_df.empty:
                    raise ValueError("empty race_df after filtering")
            except Exception:
                continue

            # ---------- Market favourite outcomes ----------
            # Use fav_rank computed in schedule_json_to_df when fixed odds exist
            fav_win_hit = fav_place_hit = sec_place_hit = third_place_hit = None
            if "fav_rank" in race_df.columns and race_df["fav_rank"].notna().any():
                # Find runner_numbers for 1st, 2nd, 3rd favourites if present
                def _rn_for_rank(k):
                    s = race_df.loc[race_df["fav_rank"] == k, "runner_number"]
                    return int(s.iloc[0]) if len(s) else None

                fav_rn   = _rn_for_rank(1)
                sec_rn   = _rn_for_rank(2)
                third_rn = _rn_for_rank(3)

                def _placed(rn):
                    fin = finish_map.get(rn)
                    return 1 if fin is not None and 1 <= fin <= 3 else 0

                def _won(rn):
                    fin = finish_map.get(rn)
                    return 1 if fin == 1 else 0

                if fav_rn is not None:
                    fav_win_hit = _won(fav_rn)
                    fav_place_hit = _placed(fav_rn)
                if sec_rn is not None:
                    sec_place_hit = _placed(sec_rn)
                if third_rn is not None:
                    third_place_hit = _placed(third_rn)

                market_rows.append(dict(
                    date=day, meeting_number=meet_no, race_number=race_no, meeting_venue=venue,
                    fav_win_hit=fav_win_hit if fav_win_hit is not None else 0,
                    fav_place_hit=fav_place_hit if fav_place_hit is not None else 0,
                    sec_place_hit=sec_place_hit if sec_place_hit is not None else 0,
                    third_place_hit=third_place_hit if third_place_hit is not None else 0,
                    has_market=int(race_df["fav_rank"].notna().any())
                ))

            # ---------- Models ----------
            pred_nn,   err_nn   = predict_top3_nn(race_df)
            codes_nn,  g_nn,  y_nn,  r_nn  = codebreaker_row(pred_nn,  finish_map)

            pred_gbr,  err_gbr  = predict_top3_gbm_reg(race_df)   # regression
            codes_gbr, g_gbr, y_gbr, r_gbr = codebreaker_row(pred_gbr, finish_map)

            pred_gbk,  err_gbk  = predict_top3_gbm_rank(race_df)  # ranking
            codes_gbk, g_gbk, y_gbk, r_gbk = codebreaker_row(pred_gbk, finish_map)

            rows.append(dict(
                date=day, meeting_number=meet_no, meeting_venue=venue,
                race_number=race_no, race_name=race_name,
                actual_top3=",".join(map(str, actual_top3)),
                model="nn",
                pred_top3=",".join(map(str, pred_nn)) if pred_nn else "",
                code_pos1=codes_nn[0], code_pos2=codes_nn[1], code_pos3=codes_nn[2],
                greens=g_nn, yellows=y_nn, reds=r_nn,
                placed_hits=g_nn + y_nn, exact_hits=g_nn,
                model_error=err_nn or ""
            ))
            rows.append(dict(
                date=day, meeting_number=meet_no, meeting_venue=venue,
                race_number=race_no, race_name=race_name,
                actual_top3=",".join(map(str, actual_top3)),
                model="gbm_reg",
                pred_top3=",".join(map(str, pred_gbr)) if pred_gbr else "",
                code_pos1=codes_gbr[0], code_pos2=codes_gbr[1], code_pos3=codes_gbr[2],
                greens=g_gbr, yellows=y_gbr, reds=r_gbr,
                placed_hits=g_gbr + y_gbr, exact_hits=g_gbr,
                model_error=err_gbr or ""
            ))
            rows.append(dict(
                date=day, meeting_number=meet_no, meeting_venue=venue,
                race_number=race_no, race_name=race_name,
                actual_top3=",".join(map(str, actual_top3)),
                model="gbm_rank",
                pred_top3=",".join(map(str, pred_gbk)) if pred_gbk else "",
                code_pos1=codes_gbk[0], code_pos2=codes_gbk[1], code_pos3=codes_gbk[2],
                greens=g_gbk, yellows=y_gbk, reds=r_gbk,
                placed_hits=g_gbk + y_gbk, exact_hits=g_gbk,
                model_error=err_gbk or ""
            ))

    df = pd.DataFrame(rows).sort_values(["date","meeting_number","race_number","model"]).reset_index(drop=True)
    # Attach market summary as an attribute to avoid changing callers
    df.market_summary = pd.DataFrame(market_rows)
    return df

# ---------------- CLI ----------------
def main():
    day = sys.argv[1] if len(sys.argv) > 1 else default_day_str()
    print(f"[plan] Benchmarking Gallops on {day} (NZ)")
    df = benchmark_day_gallops(day)
    if df.empty:
        print("[info] No gallops races evaluated.")
        return

    out = f"benchmark_gallops_{day}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"[ok] Wrote {len(df)} rows to {out}")

    # Count unique races analysed
    races_analysed = df.groupby(["date","meeting_number","race_number"]).ngroups
    print(f"\nRaces analysed: {races_analysed}")

    # Model favourite strategies using model’s own top3
    use_cols = ["date","meeting_number","race_number","model","actual_top3","pred_top3"]
    if not set(use_cols).issubset(df.columns):
        print("\n[warn] Missing expected columns for model-favourite summary.")
    else:
        dd = df[use_cols].drop_duplicates(["date","meeting_number","race_number","model"]).copy()

        def _to_list_int(s):
            if not isinstance(s, str) or not s.strip():
                return []
            return [int(x) for x in str(s).split(",") if str(x).strip().isdigit()]

        fav_win = []
        fav_place = []
        sec_place = []
        third_place = []

        for _, row in dd.iterrows():
            actual = _to_list_int(row["actual_top3"])   # ordered top 3
            preds  = _to_list_int(row["pred_top3"])     # model top 3 prediction

            winner = actual[0] if len(actual) >= 1 else None
            actual_set = set(actual)

            fav   = preds[0] if len(preds) >= 1 else None
            sec   = preds[1] if len(preds) >= 2 else None
            third = preds[2] if len(preds) >= 3 else None

            fav_win.append(1 if (fav is not None and winner is not None and fav == winner) else 0)
            fav_place.append(1 if (fav is not None and fav in actual_set) else 0)
            sec_place.append(1 if (sec is not None and sec in actual_set) else 0)
            third_place.append(1 if (third is not None and third in actual_set) else 0)

        dd["fav_win_hit"] = fav_win
        dd["fav_place_hit"] = fav_place
        dd["sec_place_hit"] = sec_place
        dd["third_place_hit"] = third_place

        # For denominators, require the selection to exist
        dd["has_fav"] = dd["pred_top3"].apply(lambda s: len(_to_list_int(s)) >= 1)
        dd["has_sec"] = dd["pred_top3"].apply(lambda s: len(_to_list_int(s)) >= 2)
        dd["has_third"] = dd["pred_top3"].apply(lambda s: len(_to_list_int(s)) >= 3)

        def _pct_str(x, d):
            return f"{x} of {d} ({(100.0*x/d):.1f}%)" if d else "0 of 0 (0.0%)"

        print("\nModel favourite strategies:")
        for model, grp in dd.groupby("model"):
            n_fav = int(grp["has_fav"].sum())
            n_sec = int(grp["has_sec"].sum())
            n_th  = int(grp["has_third"].sum())

            c_fav_win   = int(grp["fav_win_hit"].sum())
            c_fav_place = int(grp["fav_place_hit"].sum())
            c_sec_place = int(grp["sec_place_hit"].sum())
            c_th_place  = int(grp["third_place_hit"].sum())

            print(f"\n[{model}]")
            print(f"if bet on favourite to win -> {_pct_str(c_fav_win, n_fav)}")
            print(f"if bet on favourite to place -> {_pct_str(c_fav_place, n_fav)}")
            print(f"if bet on second to place -> {_pct_str(c_sec_place, n_sec)}")
            print(f"if bet on third to place -> {_pct_str(c_th_place, n_th)}")

    # Per-model hit summary
    summ = (df.groupby("model")[["greens","yellows","reds","placed_hits","exact_hits"]]
              .sum().reset_index())
    print("\nModel summary:")
    print(summ.to_string(index=False))

    # Market strategy summary
    ms = getattr(df, "market_summary", pd.DataFrame())
    if not ms.empty:
        total_mkt = int(ms["has_market"].sum())
        def pct(x, d):
            return f"{x} of {d} ({(100.0*x/d):.1f}%)" if d else "0 of 0 (0.0%)"

        fav_win = int(ms["fav_win_hit"].sum())
        fav_plc = int(ms["fav_place_hit"].sum())
        sec_plc = int(ms["sec_place_hit"].sum())
        thr_plc = int(ms["third_place_hit"].sum())

        print("\nMarket strategies:")
        print(f"- If bet on fave to win -> {pct(fav_win, total_mkt)}")
        print(f"- If bet on fave to place -> {pct(fav_plc, total_mkt)}")
        print(f"- If bet on second to place -> {pct(sec_plc, total_mkt)}")
        print(f"- If bet on third to place -> {pct(thr_plc, total_mkt)}")
    else:
        print("\nMarket strategies: no races with fixed odds available to derive favourites.")

    errs = df[df["model_error"] != ""]
    if not errs.empty:
        print("\nErrors (first 10):")
        print(errs[["meeting_venue","race_number","model","model_error"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
