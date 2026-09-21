"""Phase 15C: observational error/event analysis of frozen Model B replay.

Does not train, retune, or modify Phase 13A–15B implementations.
Reads Phase 15B ``replay_predictions.csv`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PRED_PATH = BASE_DIR / "outputs" / "v2_replay" / "phase15b" / "replay_predictions.csv"
QUALITY_PATH = BASE_DIR / "outputs" / "v2_replay" / "phase15b" / "replay_data_quality.json"
OUT_DIR = BASE_DIR / "outputs" / "v2_replay" / "phase15c"
THRESHOLD = 0.065
LEADS = (1, 2, 3)
STATIONS = ("VOTV", "VECC", "VIDP", "VOCI", "VABB")
BIN_EDGES = [0.0, 0.025, 0.05, 0.065, 0.10, 0.20, 0.50, 1.0000001]
BIN_LABELS = [
    "[0.000, 0.025)",
    "[0.025, 0.050)",
    "[0.050, 0.065)",
    "[0.065, 0.100)",
    "[0.100, 0.200)",
    "[0.200, 0.500)",
    "[0.500, 1.000]",
]
NEAR_EPS = 0.02
MODEL_PATHS = [
    BASE_DIR / "models/v2/nwp_experiment/B_atmospheric_nwp_target_1h.joblib",
    BASE_DIR / "models/v2/nwp_experiment/B_atmospheric_nwp_target_2h.joblib",
    BASE_DIR / "models/v2/nwp_experiment/B_atmospheric_nwp_target_3h.joblib",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions(path: Path | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else PRED_PATH
    df = pd.read_csv(p, low_memory=False)
    df["prediction_timestamp_utc"] = pd.to_datetime(df["prediction_timestamp_utc"], utc=True)
    return df


def labeled(df: pd.DataFrame, lead: int) -> pd.DataFrame:
    """COMPLETE rows with finite probability, alert, and target for one lead."""
    pcol = f"lead_{lead}h_probability"
    acol = f"lead_{lead}h_alert"
    ycol = f"target_{lead}h"
    sub = df.loc[df["data_status"] == "COMPLETE"].copy()
    mask = sub[pcol].notna() & sub[acol].notna() & sub[ycol].notna()
    out = sub.loc[mask].copy()
    out["_p"] = out[pcol].astype(float)
    out["_a"] = out[acol].astype(int)
    out["_y"] = out[ycol].astype(int)
    out["_cls"] = np.where(
        (out["_a"] == 1) & (out["_y"] == 1),
        "TP",
        np.where(
            (out["_a"] == 1) & (out["_y"] == 0),
            "FP",
            np.where((out["_a"] == 0) & (out["_y"] == 1), "FN", "TN"),
        ),
    )
    return out


def confusion_counts(lab: pd.DataFrame) -> dict[str, int]:
    return {
        "TP": int((lab["_cls"] == "TP").sum()),
        "FP": int((lab["_cls"] == "FP").sum()),
        "FN": int((lab["_cls"] == "FN").sum()),
        "TN": int((lab["_cls"] == "TN").sum()),
        "n": int(len(lab)),
    }


def _rate(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return float(num / den)


def _prec_rec_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return float(prec), float(rec), float(f1)


def _dist_counts(s: pd.Series, keys: list[Any]) -> dict[str, int]:
    vc = s.value_counts()
    return {str(k): int(vc.get(k, 0)) for k in keys}


def run_analysis(
    *,
    pred_path: Path | None = None,
    out_dir: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    df = load_predictions(pred_path)
    complete = df.loc[df["data_status"] == "COMPLETE"]
    labs = {lead: labeled(df, lead) for lead in LEADS}

    # 1. confusion
    conf_rows = []
    for lead in LEADS:
        c = confusion_counts(labs[lead])
        conf_rows.append(
            {
                "lead": f"{lead}h",
                "threshold": THRESHOLD,
                "n": c["n"],
                "TP": c["TP"],
                "FP": c["FP"],
                "FN": c["FN"],
                "TN": c["TN"],
                "sum_equals_n": c["TP"] + c["FP"] + c["FN"] + c["TN"] == c["n"],
            }
        )
    confusion_df = pd.DataFrame(conf_rows)

    # 2. bins
    bin_rows = []
    for lead in LEADS:
        lab = labs[lead]
        b = pd.cut(lab["_p"], bins=BIN_EDGES, labels=BIN_LABELS, right=False, include_lowest=True)
        for label in BIN_LABELS:
            part = lab.loc[b == label]
            bin_rows.append(
                {
                    "lead": f"{lead}h",
                    "bin": label,
                    "n": int(len(part)),
                    "positive_count": int(part["_y"].sum()) if len(part) else 0,
                    "positive_rate": _rate(int(part["_y"].sum()) if len(part) else 0, len(part)),
                    "alert_count": int(part["_a"].sum()) if len(part) else 0,
                }
            )
    bins_df = pd.DataFrame(bin_rows)

    # 3–4 FP / FN
    years = list(range(2021, 2026))
    months = list(range(1, 13))

    def error_summary(kind: str) -> pd.DataFrame:
        rows = []
        for lead in LEADS:
            lab = labs[lead]
            part = lab.loc[lab["_cls"] == kind]
            ts = part["prediction_timestamp_utc"]
            p = part["_p"]
            rec: dict[str, Any] = {
                "lead": f"{lead}h",
                "class": kind,
                "count": int(len(part)),
                "prob_min": float(p.min()) if len(part) else None,
                "prob_median": float(p.median()) if len(part) else None,
                "prob_mean": float(p.mean()) if len(part) else None,
                "prob_max": float(p.max()) if len(part) else None,
            }
            if kind == "FN" and len(part):
                rec["median_distance_below_threshold"] = float(THRESHOLD - p.median())
                rec["mean_distance_below_threshold"] = float(THRESHOLD - p.mean())
                rec["n_within_0.02_of_threshold"] = int((THRESHOLD - p <= NEAR_EPS).sum())
            if kind == "FP" and len(part):
                rec["median_distance_above_threshold"] = float(p.median() - THRESHOLD)
                rec["mean_distance_above_threshold"] = float(p.mean() - THRESHOLD)
            for st in STATIONS:
                rec[f"n_{st}"] = int((part["station_id"] == st).sum())
            for y in years:
                rec[f"n_year_{y}"] = int((ts.dt.year == y).sum()) if len(part) else 0
            for m in months:
                rec[f"n_month_{m:02d}"] = int((ts.dt.month == m).sum()) if len(part) else 0
            rows.append(rec)
        return pd.DataFrame(rows)

    fp_df = error_summary("FP")
    fn_df = error_summary("FN")

    # 5 + 11 alert patterns
    three = complete.loc[
        complete["lead_1h_alert"].notna()
        & complete["lead_2h_alert"].notna()
        & complete["lead_3h_alert"].notna()
    ].copy()
    three["pattern"] = (
        three["lead_1h_alert"].astype(int).astype(str)
        + three["lead_2h_alert"].astype(int).astype(str)
        + three["lead_3h_alert"].astype(int).astype(str)
    )
    n3 = len(three)
    patt_rows = []
    for code in [f"{i:03b}" for i in range(8)]:
        k = int((three["pattern"] == code).sum())
        patt_rows.append(
            {
                "pattern": code,
                "count": k,
                "percentage": _rate(k, n3) * 100 if n3 else None,
            }
        )
    patt_rows.append(
        {
            "pattern": "N_comparable",
            "count": n3,
            "percentage": 100.0 if n3 else None,
        }
    )
    a1 = int((three["lead_1h_alert"] == 1).sum())
    a2 = int((three["lead_2h_alert"] == 1).sum())
    a3 = int((three["lead_3h_alert"] == 1).sum())
    t12 = int(((three["lead_1h_alert"] == 1) != (three["lead_2h_alert"] == 1)).sum())
    t23 = int(((three["lead_2h_alert"] == 1) != (three["lead_3h_alert"] == 1)).sum())
    patt_rows.extend(
        [
            {"pattern": "alert_freq_1h", "count": a1, "percentage": _rate(a1, n3) * 100 if n3 else None},
            {"pattern": "alert_freq_2h", "count": a2, "percentage": _rate(a2, n3) * 100 if n3 else None},
            {"pattern": "alert_freq_3h", "count": a3, "percentage": _rate(a3, n3) * 100 if n3 else None},
            {"pattern": "transition_1h_ne_2h", "count": t12, "percentage": _rate(t12, n3) * 100 if n3 else None},
            {"pattern": "transition_2h_ne_3h", "count": t23, "percentage": _rate(t23, n3) * 100 if n3 else None},
        ]
    )
    patterns_df = pd.DataFrame(patt_rows)

    # 6 monthly month-of-year + year-month
    month_rows = []
    for lead in LEADS:
        lab = labs[lead]
        lab = lab.copy()
        lab["month"] = lab["prediction_timestamp_utc"].dt.month
        lab["year_month"] = lab["prediction_timestamp_utc"].dt.strftime("%Y-%m")
        for grain, col in (("month_of_year", "month"), ("year_month", "year_month")):
            for key, part in lab.groupby(col, sort=True):
                c = confusion_counts(part)
                month_rows.append(
                    {
                        "lead": f"{lead}h",
                        "grain": grain,
                        "period": str(key) if grain == "year_month" else f"{int(key):02d}",
                        "valid_cases": c["n"],
                        "positive_cases": int(part["_y"].sum()),
                        "positive_rate": _rate(int(part["_y"].sum()), c["n"]),
                        "alerts": int(part["_a"].sum()),
                        "alert_rate": _rate(int(part["_a"].sum()), c["n"]),
                        **{k: c[k] for k in ("TP", "FP", "FN", "TN")},
                    }
                )
    monthly_df = pd.DataFrame(month_rows)

    # 7 stations
    st_rows = []
    for st in STATIONS:
        for lead in LEADS:
            part = labs[lead].loc[labs[lead]["station_id"] == st]
            c = confusion_counts(part)
            prec, rec, f1 = _prec_rec_f1(c["TP"], c["FP"], c["FN"])
            st_rows.append(
                {
                    "station_id": st,
                    "lead": f"{lead}h",
                    "valid_sample_count": c["n"],
                    "positive_count": int(part["_y"].sum()) if len(part) else 0,
                    "positive_rate": _rate(int(part["_y"].sum()) if len(part) else 0, c["n"]),
                    "alert_count": int(part["_a"].sum()) if len(part) else 0,
                    "alert_rate": _rate(int(part["_a"].sum()) if len(part) else 0, c["n"]),
                    "TP": c["TP"],
                    "FP": c["FP"],
                    "FN": c["FN"],
                    "TN": c["TN"],
                    "precision": prec,
                    "recall": rec,
                    "F1": f1,
                    "threshold": THRESHOLD,
                }
            )
    station_df = pd.DataFrame(st_rows)

    # 8 yearly
    year_rows = []
    for lead in LEADS:
        lab = labs[lead]
        yy = lab["prediction_timestamp_utc"].dt.year
        for y in years:
            part = lab.loc[yy == y]
            c = confusion_counts(part)
            year_rows.append(
                {
                    "year": y,
                    "lead": f"{lead}h",
                    "valid_cases": c["n"],
                    "positive_cases": int(part["_y"].sum()) if len(part) else 0,
                    "positive_rate": _rate(int(part["_y"].sum()) if len(part) else 0, c["n"]),
                    "alerts": int(part["_a"].sum()) if len(part) else 0,
                    "alert_rate": _rate(int(part["_a"].sum()) if len(part) else 0, c["n"]),
                    **{k: c[k] for k in ("TP", "FP", "FN", "TN")},
                }
            )
    yearly_df = pd.DataFrame(year_rows)

    # 9 representative cases
    case_rows = []
    for lead in LEADS:
        lab = labs[lead].sort_values(["station_id", "prediction_timestamp_utc"])
        for kind in ("TP", "FP", "FN", "TN"):
            part = lab.loc[lab["_cls"] == kind]
            if part.empty:
                case_rows.append(
                    {
                        "lead": f"{lead}h",
                        "classification": kind,
                        "station_id": None,
                        "prediction_timestamp_utc": None,
                        "probability": None,
                        "threshold": THRESHOLD,
                        "alert": None,
                        "target": None,
                        "distance_from_threshold": None,
                        "status": "unavailable",
                    }
                )
                continue
            if kind in ("FP", "FN"):
                dist = (part["_p"] - THRESHOLD).abs()
                idx = dist.idxmin()
                row = part.loc[idx]
            else:
                row = part.iloc[0]
            p = float(row["_p"])
            case_rows.append(
                {
                    "lead": f"{lead}h",
                    "classification": kind,
                    "station_id": row["station_id"],
                    "prediction_timestamp_utc": pd.Timestamp(row["prediction_timestamp_utc"]).isoformat(),
                    "probability": p,
                    "threshold": THRESHOLD,
                    "alert": int(row["_a"]),
                    "target": int(row["_y"]),
                    "distance_from_threshold": p - THRESHOLD if kind in ("FP", "FN") else None,
                    "status": "ok",
                }
            )
    cases_df = pd.DataFrame(case_rows)

    # 10 near threshold
    near_rows = []
    for lead in LEADS:
        lab = labs[lead]
        part = lab.loc[(lab["_p"] - THRESHOLD).abs() <= NEAR_EPS]
        c = confusion_counts(part)
        near_rows.append(
            {
                "lead": f"{lead}h",
                "distance_max": NEAR_EPS,
                "count": c["n"],
                "positive_count": int(part["_y"].sum()) if len(part) else 0,
                "positive_rate": _rate(int(part["_y"].sum()) if len(part) else 0, c["n"]),
                "TP": c["TP"],
                "FP": c["FP"],
                "FN": c["FN"],
                "TN": c["TN"],
            }
        )
    near_df = pd.DataFrame(near_rows)

    # 12 validation
    dup = int(df.duplicated(["station_id", "prediction_timestamp_utc"]).sum())
    alerts_ok = True
    probs_ok = True
    for lead in LEADS:
        lab = labs[lead]
        if ((lab["_p"] < 0) | (lab["_p"] > 1)).any():
            probs_ok = False
        expected = (lab["_p"] >= THRESHOLD).astype(int)
        if not (expected == lab["_a"]).all():
            alerts_ok = False
        # missing targets not in labeled set
    missing_not_zero = True
    for lead in LEADS:
        ycol = f"target_{lead}h"
        pcol = f"lead_{lead}h_probability"
        csub = complete.loc[complete[pcol].notna() & complete[ycol].isna()]
        if len(csub) and f"target_{lead}h" in complete.columns:
            # those rows must not appear in labeled n
            if len(labs[lead].merge(csub[["station_id", "prediction_timestamp_utc"]], how="inner")):
                missing_not_zero = False

    quality = {}
    if QUALITY_PATH.is_file():
        quality = json.loads(QUALITY_PATH.read_text(encoding="utf-8"))

    validation = {
        "phase": "15C",
        "input_rows": int(len(df)),
        "complete_rows": int(len(complete)),
        "unavailable_rows": int((df["data_status"] == "UNAVAILABLE").sum()),
        "duplicate_station_timestamp": dup,
        "probabilities_in_unit_interval": probs_ok,
        "alerts_match_threshold_0.065": alerts_ok,
        "missing_targets_excluded_not_zero": missing_not_zero,
        "unavailable_not_in_labeled_metrics": all(
            (labs[lead]["data_status"] == "COMPLETE").all() for lead in LEADS
        ),
        "valid_labeled_n": {f"{lead}h": int(len(labs[lead])) for lead in LEADS},
        "confusion": {r["lead"]: {k: r[k] for k in ("TP", "FP", "FN", "TN", "n")} for r in conf_rows},
        "stations_in_complete": sorted(complete["station_id"].dropna().unique().tolist()),
        "all_five_stations_present": set(complete["station_id"].dropna().unique()) >= set(STATIONS),
        "bin_coverage_ok": all(
            int(bins_df.loc[bins_df["lead"] == f"{lead}h", "n"].sum()) == len(labs[lead]) for lead in LEADS
        ),
        "phase15b_eval_n": quality.get("evaluation_sample_count_per_lead"),
        "confusion_matches_15b_n": {
            f"{lead}h": int(len(labs[lead])) == int(quality.get("evaluation_sample_count_per_lead", {}).get(f"{lead}h", -1))
            if quality
            else None
            for lead in LEADS
        },
        "deterministic": True,
        "threshold": THRESHOLD,
        "note": "Phase 15C is an observational analysis of the frozen V2 Model B replay outputs. It does not modify the model.",
    }

    tables = {
        "confusion_summary.csv": confusion_df,
        "probability_bins.csv": bins_df,
        "false_positive_summary.csv": fp_df,
        "false_negative_summary.csv": fn_df,
        "lead_alert_patterns.csv": patterns_df,
        "monthly_event_analysis.csv": monthly_df,
        "station_error_analysis.csv": station_df,
        "yearly_error_analysis.csv": yearly_df,
        "representative_cases.csv": cases_df,
        "near_threshold_analysis.csv": near_df,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, tbl in tables.items():
            tbl.to_csv(out_dir / name, index=False)
        (out_dir / "analysis_validation.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )
    return {"tables": tables, "validation": validation, "labeled": labs, "predictions": df}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 15C observational replay analysis.")
    p.add_argument("--pred", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    run_analysis(
        pred_path=Path(args.pred) if args.pred else None,
        out_dir=Path(args.out) if args.out else None,
        write=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
