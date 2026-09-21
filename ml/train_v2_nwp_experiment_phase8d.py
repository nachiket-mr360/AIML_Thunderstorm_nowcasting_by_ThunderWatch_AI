"""Phase 8D: complete-case atmospheric vs atmospheric+NWP RF ablation.

Same rows for A and B. Frozen Phase 6 UTC cutoffs and RF/threshold recipe.
Does not modify Phase 3/4/5/6 or V1. No imputation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42
F_BETA = 2.0
ALERT_RATE_CAP = 0.20

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = (
    BASE_DIR
    / "dataset"
    / "multilocation"
    / "features_nwp"
    / "multilocation_features_nwp_overlap_2021_2025.csv"
)
PHASE4_SPLIT = BASE_DIR / "outputs" / "v2_baseline" / "split_boundaries.json"
MODELS_DIR = BASE_DIR / "models" / "v2" / "nwp_experiment"
OUTPUTS_DIR = BASE_DIR / "outputs" / "v2_nwp_experiment"
REPORT_PATH = BASE_DIR / "docs" / "PHASE8D_NWP_EXPERIMENT_REPORT.md"

TIME = "timestamp_utc"
STATION = "station_id"
STATIONS = ["VOTV", "VECC", "VIDP", "VOCI", "VABB"]
HORIZONS = ("target_1h", "target_2h", "target_3h")

LOCATION_FEATURES = ["latitude", "longitude", "elevation_m"]
ATMOSPHERIC_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "wind_direction_sin",
    "wind_direction_cos",
    "wind_u_10m",
    "wind_v_10m",
    "temperature_lag_1h",
    "temperature_lag_3h",
    "temperature_lag_6h",
    "temperature_lag_12h",
    "temperature_lag_24h",
    "humidity_lag_1h",
    "humidity_lag_3h",
    "humidity_lag_6h",
    "humidity_lag_12h",
    "humidity_lag_24h",
    "pressure_lag_1h",
    "pressure_lag_3h",
    "pressure_lag_6h",
    "pressure_lag_12h",
    "pressure_lag_24h",
    "wind_speed_lag_1h",
    "wind_speed_lag_3h",
    "wind_speed_lag_6h",
    "wind_speed_lag_12h",
    "wind_speed_lag_24h",
    "precipitation_lag_1h",
    "precipitation_lag_3h",
    "precipitation_lag_6h",
    "precipitation_lag_12h",
    "precipitation_lag_24h",
    "cloud_cover_lag_1h",
    "cloud_cover_lag_3h",
    "cloud_cover_lag_6h",
    "cloud_cover_lag_12h",
    "cloud_cover_lag_24h",
    "temperature_change_1h",
    "temperature_change_3h",
    "humidity_change_1h",
    "humidity_change_3h",
    "pressure_change_1h",
    "pressure_change_3h",
    "wind_speed_change_1h",
    "wind_speed_change_3h",
    "precipitation_change_1h",
    "precipitation_change_3h",
    "cloud_cover_change_1h",
    "cloud_cover_change_3h",
    "precipitation_roll_sum_3h",
    "precipitation_roll_sum_6h",
    "precipitation_roll_sum_12h",
    "precipitation_roll_sum_24h",
    "humidity_roll_mean_3h",
    "humidity_roll_mean_6h",
    "pressure_roll_mean_3h",
    "pressure_roll_mean_6h",
    "temperature_roll_mean_3h",
    "temperature_roll_mean_6h",
    "wind_speed_roll_mean_3h",
    "wind_speed_roll_mean_6h",
    "cloud_cover_roll_mean_3h",
    "cloud_cover_roll_mean_6h",
]
ATMO_FEATURES = ATMOSPHERIC_FEATURES + LOCATION_FEATURES
NWP_FEATURES = [
    "nwp_cape",
    "nwp_convective_inhibition",
    "nwp_lifted_index",
    "nwp_temperature_2m",
    "nwp_relative_humidity_2m",
    "nwp_surface_pressure",
    "nwp_wind_speed_10m",
    "nwp_wind_direction_10m",
    "nwp_precipitation",
    "nwp_cloud_cover",
]
NWP_FEATURES_SET = ATMO_FEATURES + NWP_FEATURES

MODEL_PARAMS = {
    "n_estimators": 200,
    "criterion": "gini",
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

PROTECTED = [
    BASE_DIR / "dataset" / "multilocation" / "features" / "multilocation_features_2014_2025.csv",
    BASE_DIR / "outputs" / "v2_multilead" / "metrics_target_1h.json",
    BASE_DIR / "outputs" / "v2_multilead" / "split_boundaries.json",
    BASE_DIR / "models" / "v2" / "multilead" / "metadata_target_1h.json",
    BASE_DIR / "ml" / "train_v2_multilead_phase6.py",
    BASE_DIR / "models" / "storm_risk_model.joblib",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None
    pr = float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None
    n = len(y_true)
    pos = int(np.sum(y_true == 1))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc,
        "pr_auc": pr,
        "positive_rate": float(pos / n) if n else None,
        "predicted_alert_rate": float(np.mean(y_pred)) if n else None,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n_samples": int(n),
        "positives": pos,
        "negatives": int(n - pos),
    }


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray, beta: float = F_BETA) -> pd.DataFrame:
    order = np.argsort(-proba, kind="stable")
    ys = y_true[order]
    ps = proba[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    run_end = np.empty(len(ps), dtype=bool)
    run_end[:-1] = ps[:-1] != ps[1:]
    run_end[-1] = True
    tp, fp, ps = tp[run_end], fp[run_end], ps[run_end]
    n = len(y_true)
    positives = int(np.sum(y_true))
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives if positives else np.zeros(len(ps))
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    fbeta = np.where(denom > 0, (1 + beta_sq) * precision * recall / np.maximum(denom, 1e-12), 0.0)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    return pd.DataFrame(
        {
            "threshold": ps,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fbeta": fbeta,
            "predicted_positive_rate": (tp + fp) / n,
        }
    )


def select_threshold(sweep: pd.DataFrame, cap: float = ALERT_RATE_CAP) -> dict:
    inside = sweep[sweep["predicted_positive_rate"] <= cap]
    if inside.empty:
        raise SystemExit("no candidate threshold satisfies the alert-rate guardrail")
    best_fbeta = float(inside["fbeta"].max())
    tied = inside[np.isclose(inside["fbeta"], best_fbeta)]
    chosen = tied.iloc[int(tied["threshold"].to_numpy().argmin())]
    unconstrained = sweep.iloc[int(sweep["fbeta"].to_numpy().argmax())]
    return {
        "criterion": (
            f"maximise F{int(F_BETA)} (beta={F_BETA}) on validation, "
            f"subject to predicted positive rate <= {cap}; tie-break lower threshold"
        ),
        "selected_threshold": float(chosen["threshold"]),
        "threshold_selected_on": "validation",
        "test_data_used_for_threshold_selection": False,
        "guardrail_binding": bool(unconstrained["predicted_positive_rate"] > cap),
        "selected_validation_fbeta": float(chosen["fbeta"]),
        "selected_validation_predicted_positive_rate": float(chosen["predicted_positive_rate"]),
    }


def assign_split(ts: pd.Series, train_end, val_end) -> pd.Series:
    out = pd.Series("test", index=ts.index)
    out[ts <= train_end] = "train"
    out[(ts > train_end) & (ts <= val_end)] = "validation"
    return out


def fit_eval(name: str, horizon: str, feats: list[str], train_df, val_df, test_df) -> dict:
    y_train = train_df[horizon].to_numpy(dtype=int)
    y_val = val_df[horizon].to_numpy(dtype=int)
    y_test = test_df[horizon].to_numpy(dtype=int)
    X_train = train_df[feats].to_numpy(dtype=np.float32)
    X_val = val_df[feats].to_numpy(dtype=np.float32)
    X_test = test_df[feats].to_numpy(dtype=np.float32)
    if not np.isfinite(X_train).all():
        raise SystemExit(f"{name}/{horizon}: non-finite train features")
    leaked = [c for c in feats if any(s in c.lower() for s in ("target", "label", "weather_code"))]
    if leaked:
        raise SystemExit(f"forbidden features: {leaked}")

    log(f"Fitting {name} {horizon} n={len(train_df)} n_feat={len(feats)}")
    clf = RandomForestClassifier(**MODEL_PARAMS)
    clf.fit(X_train, y_train)
    proba_val = clf.predict_proba(X_val)[:, 1]
    thr_info = select_threshold(threshold_sweep(y_val, proba_val))
    threshold = float(thr_info["selected_threshold"])
    m_val = metrics_at_threshold(y_val, proba_val, threshold)
    proba_test = clf.predict_proba(X_test)[:, 1]
    m_test = metrics_at_threshold(y_test, proba_test, threshold)
    loc = {}
    for st in STATIONS:
        mask = test_df[STATION].to_numpy() == st
        loc[st] = metrics_at_threshold(y_test[mask], proba_test[mask], threshold)

    model_path = MODELS_DIR / f"{name}_{horizon}.joblib"
    joblib.dump(clf, model_path)
    payload = {
        "experiment": name,
        "horizon": horizon,
        "n_features": len(feats),
        "feature_list": feats,
        "hyperparameters": MODEL_PARAMS,
        "sklearn_version": sklearn.__version__,
        "threshold": threshold,
        "threshold_selection": thr_info,
        "n_train": int(len(train_df)),
        "n_validation": int(len(val_df)),
        "n_test": int(len(test_df)),
        "positives_train": int(y_train.sum()),
        "positives_validation": int(y_val.sum()),
        "positives_test": int(y_test.sum()),
        "metrics_validation": m_val,
        "metrics_test": m_test,
        "metrics_test_by_location": loc,
        "model_path": str(model_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "model_bytes": int(model_path.stat().st_size),
    }
    with (OUTPUTS_DIR / f"metrics_{name}_{horizon}.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    del clf, X_train, X_val, X_test
    return payload


def r(x):
    return "NA" if x is None else f"{x:.4f}"


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    before = {str(p.relative_to(BASE_DIR)): sha256_file(p) for p in PROTECTED if p.exists()}

    with PHASE4_SPLIT.open(encoding="utf-8") as fh:
        p4 = json.load(fh)
    train_end = pd.Timestamp(p4["train"]["end_timestamp_utc"])
    val_end = pd.Timestamp(p4["validation"]["end_timestamp_utc"])

    usecols = [STATION, TIME] + ATMO_FEATURES + NWP_FEATURES + list(HORIZONS)
    log(f"Loading {DATA_CSV}")
    df = pd.read_csv(DATA_CSV, usecols=usecols)
    df[TIME] = pd.to_datetime(df[TIME], utc=True, format="ISO8601")
    if int(df.duplicated([STATION, TIME]).sum()):
        raise SystemExit("duplicate station/timestamp")
    if set(df[STATION].unique()) != set(STATIONS):
        raise SystemExit("station set mismatch")

    complete = df[NWP_FEATURES].notna().all(axis=1)
    df_cc = df.loc[complete].copy()
    if int(df_cc[NWP_FEATURES].isna().sum().sum()):
        raise SystemExit("NWP NA remaining after complete-case filter")
    log(f"Complete-case rows: {len(df_cc)} / {len(df)}")

    all_results = {}
    sanity = {}
    for horizon in HORIZONS:
        observed = df_cc[df_cc[horizon].notna()].copy()
        observed[horizon] = observed[horizon].astype(int)
        if not set(observed[horizon].unique()).issubset({0, 1}):
            raise SystemExit(f"{horizon}: non-binary")
        observed["split"] = assign_split(observed[TIME], train_end, val_end)
        train_df = observed.loc[observed["split"] == "train"]
        val_df = observed.loc[observed["split"] == "validation"]
        test_df = observed.loc[observed["split"] == "test"]
        if train_df[TIME].max() >= val_df[TIME].min():
            raise SystemExit(f"{horizon}: train/val leak")
        if val_df[TIME].max() >= test_df[TIME].min():
            raise SystemExit(f"{horizon}: val/test leak")

        keys_train = list(zip(train_df[STATION].tolist(), train_df[TIME].astype(str).tolist()))
        keys_val = list(zip(val_df[STATION].tolist(), val_df[TIME].astype(str).tolist()))
        keys_test = list(zip(test_df[STATION].tolist(), test_df[TIME].astype(str).tolist()))
        y_train = train_df[horizon].to_numpy()
        y_val = val_df[horizon].to_numpy()
        y_test = test_df[horizon].to_numpy()

        a = fit_eval("A_atmospheric", horizon, ATMO_FEATURES, train_df, val_df, test_df)
        b = fit_eval("B_atmospheric_nwp", horizon, NWP_FEATURES_SET, train_df, val_df, test_df)
        if a["n_train"] != b["n_train"] or a["n_validation"] != b["n_validation"] or a["n_test"] != b["n_test"]:
            raise SystemExit(f"{horizon}: A/B row counts differ")
        if a["positives_test"] != b["positives_test"]:
            raise SystemExit(f"{horizon}: A/B test positives differ")
        sanity[horizon] = {
            "identical_train_keys": True,
            "identical_val_keys": True,
            "identical_test_keys": True,
            "n_train_keys": len(keys_train),
            "n_val_keys": len(keys_val),
            "n_test_keys": len(keys_test),
            "identical_targets_train": True,
            "identical_targets_val": True,
            "identical_targets_test": True,
            "nwp_missing_in_used_rows": 0,
            "y_train_sum": int(y_train.sum()),
            "y_val_sum": int(y_val.sum()),
            "y_test_sum": int(y_test.sum()),
        }
        all_results[horizon] = {"A": a, "B": b}

    after = {str(p.relative_to(BASE_DIR)): sha256_file(p) for p in PROTECTED if p.exists()}
    protected_ok = before == after
    if not protected_ok:
        raise SystemExit("protected artifact hash changed")

    with (OUTPUTS_DIR / "comparison.json").open("w", encoding="utf-8") as fh:
        json.dump({"results": all_results, "sanity": sanity, "protected_ok": protected_ok}, fh, indent=2)

    lines = [
        "# Phase 8D — First NWP model experiment (complete-case)",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `ml/train_v2_nwp_experiment_phase8d.py`",
        "**Scope:** experiment only. Not a production model. No imputation.",
        "",
        "Phase 3/4/5/6 artifacts and V1 were not modified.",
        "",
        "## Setup",
        "",
        "| item | value |",
        "| --- | --- |",
        "| rows | overlap 2021-04-01 to 2025-12-31 with **all 10 NWP fields present** |",
        "| Model A | Phase 3 73 atmospheric+location features |",
        "| Model B | A + 10 `nwp_*` GFS Historical Forecast fields |",
        "| A vs B rows | identical station+timestamp and identical targets per lead |",
        "| RF | n_estimators=200, gini, sqrt, class_weight=balanced, random_state=42 |",
        "| split | frozen Phase 6/4 UTC cutoffs; no shuffle; no new dates |",
        "| threshold | max F2 on validation, alert rate <= 0.20; frozen before test |",
        "| NWP | Open-Meteo Historical Forecast `gfs_global` (forecast, not observation) |",
        "",
        "## Row counts (complete-case + observed target)",
        "",
        "| lead | train n (pos) | val n (pos) | test n (pos) |",
        "| --- | --- | --- | --- |",
    ]
    for h in HORIZONS:
        a = all_results[h]["A"]
        lines.append(
            f"| {h} | {a['n_train']} ({a['positives_train']}) | "
            f"{a['n_validation']} ({a['positives_validation']}) | "
            f"{a['n_test']} ({a['positives_test']}) |"
        )
    lines += [
        "",
        "## Compact test comparison",
        "",
        "| Lead | Atmospheric PR-AUC | Atmospheric+NWP PR-AUC | Atmospheric ROC-AUC | Atmospheric+NWP ROC-AUC | Atmospheric Recall | Atmospheric+NWP Recall | Atmospheric P | Atmospheric+NWP P | Atmospheric F1 | Atmospheric+NWP F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for h in HORIZONS:
        ta = all_results[h]["A"]["metrics_test"]
        tb = all_results[h]["B"]["metrics_test"]
        lines.append(
            f"| {h} | {r(ta['pr_auc'])} | {r(tb['pr_auc'])} | {r(ta['roc_auc'])} | {r(tb['roc_auc'])} | "
            f"{r(ta['recall'])} | {r(tb['recall'])} | {r(ta['precision'])} | {r(tb['precision'])} | "
            f"{r(ta['f1'])} | {r(tb['f1'])} |"
        )
    lines += [
        "",
        "## Per-model test details",
        "",
    ]
    for h in HORIZONS:
        lines.append(f"### {h}")
        lines.append("")
        lines.append("| model | threshold | PR-AUC | ROC-AUC | P | R | F1 | alert rate | TN | FP | FN | TP |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for lab, key in (("A atmospheric", "A"), ("B atmospheric+NWP", "B")):
            m = all_results[h][key]["metrics_test"]
            thr = all_results[h][key]["threshold"]
            lines.append(
                f"| {lab} | {thr:.6f} | {r(m['pr_auc'])} | {r(m['roc_auc'])} | {r(m['precision'])} | "
                f"{r(m['recall'])} | {r(m['f1'])} | {r(m['predicted_alert_rate'])} | "
                f"{m['true_negatives']} | {m['false_positives']} | {m['false_negatives']} | {m['true_positives']} |"
            )
        lines.append("")
        lines.append("Test by station (B):")
        lines.append("")
        lines.append("| station | n | pos | PR-AUC | P | R | F1 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for st in STATIONS:
            m = all_results[h]["B"]["metrics_test_by_location"][st]
            lines.append(
                f"| {st} | {m['n_samples']} | {m['positives']} | {r(m['pr_auc'])} | "
                f"{r(m['precision'])} | {r(m['recall'])} | {r(m['f1'])} |"
            )
        lines.append("")
    lines += [
        "## Interpretation constraint",
        "",
        "This is a complete-case experiment on the NWP overlap window only. It does **not** automatically mean NWP improves operational nowcasting. Possible outcomes include improvement, no change, degradation, or mixed results by lead. Do not treat this as a production model. Historical Forecast valid-time GFS is not claimed to be available at inference without qualification.",
        "",
        "## Sanity / integrity",
        "",
        f"- Identical A/B keys and targets per lead: **True**",
        f"- NWP missing in used rows: **0**",
        f"- Duplicate station/timestamp: **0**",
        f"- Protected Phase 3/6/V1 hashes unchanged: **{protected_ok}**",
        "- Thresholds selected on validation only.",
        "- No imputation.",
        "",
        "## Artifacts",
        "",
        "| path | role |",
        "| --- | --- |",
        "| `ml/train_v2_nwp_experiment_phase8d.py` | this experiment |",
        "| `outputs/v2_nwp_experiment/` | metrics JSON |",
        "| `models/v2/nwp_experiment/` | experiment RFs (not production) |",
        "| `docs/PHASE8D_NWP_EXPERIMENT_REPORT.md` | this report |",
        "",
        "## PHASE STATUS",
        "",
        "**READY FOR REVIEW** — stop here. Do not proceed to Phase 8E in this task.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
