"""Phase 4 (V2): pooled multi-location thunderstorm baseline.

Same-hour task: features at clock hour T, genuine METAR thunderstorm_target at T.
NA targets are excluded from supervised training and evaluation (never treated as 0).

Two experiments:
  A. atmospheric-only (70 causal features)
  B. atmospheric + location metadata (73 features)

Chronological split on unique UTC hours (shared across stations) so no future hour
enters training. Threshold selected on validation only (F2, alert-rate cap 0.20),
then frozen for a single test evaluation.

Does not modify V1, Flask, or the synchronized/feature CSVs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
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
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = BASE_DIR / "dataset" / "multilocation" / "features" / "multilocation_features_2014_2025.csv"
MODELS_DIR = BASE_DIR / "models" / "v2"
OUTPUTS_DIR = BASE_DIR / "outputs" / "v2_baseline"
REPORT_PATH = BASE_DIR / "docs" / "PHASE4_V2_BASELINE_MODEL_REPORT.md"

TIME_COLUMN = "timestamp_utc"
STATION_COLUMN = "station_id"
TARGET = "thunderstorm_target"
STATIONS = ["VOTV", "VECC", "VIDP", "VOCI", "VABB"]

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

FULL_FEATURES = ATMOSPHERIC_FEATURES + LOCATION_FEATURES

FORBIDDEN_SUBSTRINGS = ("target", "label", "weather_code", "wxcodes")

# V1 used n_estimators=400 on ~54k VOTV train rows. Pooled labeled train is ~330k rows;
# 100 trees keep the same RF family (gini, sqrt, class_weight=balanced, random_state=42)
# while remaining tractable. Tree count is documented, not hidden.
MODEL_PARAMS = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

# Historical V1 1-hour test metrics (outputs/PHASE6_MODEL_TRAINING_REPORT.md). Reference only.
V1_1H_TEST_REFERENCE = {
    "lead_time": "1h nowcast (features at t, genuine METAR at t+1h)",
    "station": "VOTV only",
    "model": "RandomForestClassifier n_estimators=400 class_weight=balanced random_state=42",
    "test_recall": 0.6040,
    "test_precision": 0.1332,
    "test_f1": 0.2183,
    "test_pr_auc": 0.2060,
    "test_roc_auc": 0.8453,
    "locked_threshold": 0.077500,
    "test_confusion": {"tn": 9630, "fp": 1568, "fn": 158, "tp": 241},
    "note": (
        "V1 is a different task (1h lead, VOTV-only, 28 features). These numbers are a "
        "historical reference, not a claim that V2 pooled same-hour models are superior."
    ),
}

PROTECTED_V1 = [
    BASE_DIR / "models" / "storm_risk_model.joblib",
    BASE_DIR / "ml" / "train_thunderstorm_nowcast_phase6.py",
    BASE_DIR / "app.py",
    BASE_DIR / "dataset" / "votv_thunderstorm_nowcast_2014_2025.csv",
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
            "true_positives": tp,
            "false_positives": fp,
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
        "beta": F_BETA,
        "alert_rate_cap": cap,
        "guardrail_binding": bool(unconstrained["predicted_positive_rate"] > cap),
        "candidate_thresholds": int(len(sweep)),
        "candidates_inside_guardrail": int(len(inside)),
        "selected_threshold": float(chosen["threshold"]),
        "threshold_selected_on": "validation",
        "test_data_used_for_threshold_selection": False,
        "selected_validation_fbeta": float(chosen["fbeta"]),
        "selected_validation_predicted_positive_rate": float(chosen["predicted_positive_rate"]),
    }


def fmt_m(m: dict) -> str:
    def r(x):
        return "NA" if x is None else f"{x:.4f}"

    return (
        f"n={m['n_samples']} pos={m['positives']} rate={r(m['positive_rate'])} "
        f"alert={r(m['predicted_alert_rate'])} acc={r(m['accuracy'])} "
        f"P={r(m['precision'])} R={r(m['recall'])} F1={r(m['f1'])} "
        f"ROC-AUC={r(m['roc_auc'])} PR-AUC={r(m['pr_auc'])} "
        f"FP={m['false_positives']} FN={m['false_negatives']} "
        f"CM={m['confusion_matrix']}"
    )


def md_metrics_table(title: str, rows: list[tuple[str, dict]]) -> str:
    lines = [
        f"### {title}",
        "",
        "| split | n | pos | pos rate | alert rate | acc | P | R | F1 | ROC-AUC | PR-AUC | FP | FN | TN | TP |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, m in rows:
        def r(x):
            return "NA" if x is None else f"{x:.4f}"

        lines.append(
            f"| {name} | {m['n_samples']} | {m['positives']} | {r(m['positive_rate'])} | "
            f"{r(m['predicted_alert_rate'])} | {r(m['accuracy'])} | {r(m['precision'])} | "
            f"{r(m['recall'])} | {r(m['f1'])} | {r(m['roc_auc'])} | {r(m['pr_auc'])} | "
            f"{m['false_positives']} | {m['false_negatives']} | {m['true_negatives']} | {m['true_positives']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    started = datetime.now(timezone.utc)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    v1_hashes_before = {str(p.relative_to(BASE_DIR)): sha256_file(p) for p in PROTECTED_V1 if p.exists()}

    log(f"Loading {DATA_CSV}")
    usecols = (
        [TIME_COLUMN, STATION_COLUMN, TARGET, "city", "label_status"]
        + FULL_FEATURES
    )
    df = pd.read_csv(DATA_CSV, usecols=usecols)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], utc=True, format="ISO8601")
    if "weather_code" in df.columns:
        raise SystemExit("weather_code must not be present in the training table slice")
    leaked = [c for c in FULL_FEATURES if any(s in c.lower() for s in FORBIDDEN_SUBSTRINGS)]
    if leaked:
        raise SystemExit(f"forbidden feature names: {leaked}")
    if len(ATMOSPHERIC_FEATURES) != 70 or len(FULL_FEATURES) != 73:
        raise SystemExit("expected 70 atmospheric + 3 location = 73 features")

    stations_present = sorted(df[STATION_COLUMN].unique().tolist())
    if set(stations_present) != set(STATIONS):
        raise SystemExit(f"expected stations {STATIONS}, got {stations_present}")
    if len(df) != 525840:
        log(f"WARNING: row count {len(df)} != 525840")

    observed = df[df[TARGET].notna()].copy()
    observed[TARGET] = observed[TARGET].astype(int)
    if not set(observed[TARGET].unique()).issubset({0, 1}):
        raise SystemExit("observed targets must be 0/1 only")
    if int(observed[TARGET].isna().sum()):
        raise SystemExit("NA targets leaked into observed subset")

    # Chronological split on unique UTC hours so future hours never enter train.
    unique_times = np.sort(observed[TIME_COLUMN].unique())
    n_t = len(unique_times)
    n_train_t = int(SPLIT_FRACTIONS["train"] * n_t)
    n_val_t = int(SPLIT_FRACTIONS["validation"] * n_t)
    train_times = unique_times[:n_train_t]
    val_times = unique_times[n_train_t : n_train_t + n_val_t]
    test_times = unique_times[n_train_t + n_val_t :]
    if train_times[-1] >= val_times[0] or val_times[-1] >= test_times[0]:
        raise SystemExit("time cutoffs are not strictly chronological")

    t_train_end = pd.Timestamp(train_times[-1])
    t_val_start = pd.Timestamp(val_times[0])
    t_val_end = pd.Timestamp(val_times[-1])
    t_test_start = pd.Timestamp(test_times[0])
    t_train_start = pd.Timestamp(train_times[0])
    t_test_end = pd.Timestamp(test_times[-1])

    train_mask = observed[TIME_COLUMN].isin(train_times)
    val_mask = observed[TIME_COLUMN].isin(val_times)
    test_mask = observed[TIME_COLUMN].isin(test_times)
    train_df = observed.loc[train_mask]
    val_df = observed.loc[val_mask]
    test_df = observed.loc[test_mask]
    if train_df[TIME_COLUMN].max() >= val_df[TIME_COLUMN].min():
        raise SystemExit("row-level train/val leak")
    if val_df[TIME_COLUMN].max() >= test_df[TIME_COLUMN].min():
        raise SystemExit("row-level val/test leak")
    if int(train_df[TARGET].isna().sum()):
        raise SystemExit("NA in training labels")

    split_info = {
        "strategy": "chronological unique UTC hours, 70/15/15, never shuffled; same cutoffs for all stations",
        "n_unique_observed_hours": int(n_t),
        "n_train_hours": int(len(train_times)),
        "n_validation_hours": int(len(val_times)),
        "n_test_hours": int(len(test_times)),
        "train": {
            "start_timestamp_utc": t_train_start.isoformat(),
            "end_timestamp_utc": t_train_end.isoformat(),
            "n_rows": int(len(train_df)),
        },
        "validation": {
            "start_timestamp_utc": t_val_start.isoformat(),
            "end_timestamp_utc": t_val_end.isoformat(),
            "n_rows": int(len(val_df)),
        },
        "test": {
            "start_timestamp_utc": t_test_start.isoformat(),
            "end_timestamp_utc": t_test_end.isoformat(),
            "n_rows": int(len(test_df)),
        },
        "future_observations_in_training": False,
        "random_split_used": False,
    }

    y_train = train_df[TARGET].to_numpy(dtype=int)
    y_val = val_df[TARGET].to_numpy(dtype=int)
    y_test = test_df[TARGET].to_numpy(dtype=int)

    experiments = {
        "A_atmospheric_only": {
            "features": ATMOSPHERIC_FEATURES,
            "description": "70 causal atmospheric features; no lat/lon/elevation",
        },
        "B_atmospheric_plus_location": {
            "features": FULL_FEATURES,
            "description": "70 atmospheric + latitude, longitude, elevation_m",
        },
    }

    all_results = {}
    preferred = None
    preferred_val_pr = -1.0

    for exp_name, spec in experiments.items():
        feats = spec["features"]
        log(f"=== Experiment {exp_name}: {len(feats)} features ===")
        X_train = train_df[feats].to_numpy(dtype=np.float32)
        X_val = val_df[feats].to_numpy(dtype=np.float32)
        X_test = test_df[feats].to_numpy(dtype=np.float32)
        if not np.isfinite(X_train).all():
            raise SystemExit(f"{exp_name}: non-finite training features")

        clf = RandomForestClassifier(**MODEL_PARAMS)
        log(f"Fitting RandomForest n_estimators={MODEL_PARAMS['n_estimators']} ...")
        clf.fit(X_train, y_train)

        proba_val = clf.predict_proba(X_val)[:, 1]
        sweep = threshold_sweep(y_val, proba_val)
        thr_info = select_threshold(sweep)
        threshold = float(thr_info["selected_threshold"])
        log(f"Locked threshold (validation): {threshold:.6f}")

        m_val = metrics_at_threshold(y_val, proba_val, threshold)
        proba_test = clf.predict_proba(X_test)[:, 1]
        m_test = metrics_at_threshold(y_test, proba_test, threshold)
        proba_train = clf.predict_proba(X_train)[:, 1]
        m_train = metrics_at_threshold(y_train, proba_train, threshold)

        loc_metrics = {}
        for st in STATIONS:
            mask = test_df[STATION_COLUMN].values == st
            loc_metrics[st] = metrics_at_threshold(y_test[mask], proba_test[mask], threshold)

        model_path = MODELS_DIR / f"pooled_rf_{exp_name}.joblib"
        joblib.dump(clf, model_path)
        reloaded = joblib.load(model_path)
        reload_ok = np.allclose(reloaded.predict_proba(X_test[:50])[:, 1], proba_test[:50])

        importance = (
            pd.DataFrame({"feature": feats, "impurity_importance": clf.feature_importances_})
            .sort_values("impurity_importance", ascending=False)
            .reset_index(drop=True)
        )
        importance.to_csv(OUTPUTS_DIR / f"feature_importance_{exp_name}.csv", index=False)

        payload = {
            "experiment": exp_name,
            "description": spec["description"],
            "hyperparameters": MODEL_PARAMS,
            "sklearn_version": sklearn.__version__,
            "n_features": len(feats),
            "feature_list": feats,
            "threshold": threshold,
            "threshold_selection": thr_info,
            "split": split_info,
            "metrics_train": m_train,
            "metrics_validation": m_val,
            "metrics_test": m_test,
            "metrics_test_by_location": loc_metrics,
            "model_path": str(model_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "reload_ok": bool(reload_ok),
            "class_weight": "balanced",
            "random_state": SEED,
        }
        with (OUTPUTS_DIR / f"metrics_{exp_name}.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        with (MODELS_DIR / f"metadata_{exp_name}.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "experiment": exp_name,
                    "feature_list": feats,
                    "threshold": threshold,
                    "hyperparameters": MODEL_PARAMS,
                    "split_boundaries": {
                        "train_start": t_train_start.isoformat(),
                        "train_end": t_train_end.isoformat(),
                        "validation_start": t_val_start.isoformat(),
                        "validation_end": t_val_end.isoformat(),
                        "test_start": t_test_start.isoformat(),
                        "test_end": t_test_end.isoformat(),
                    },
                    "target": TARGET,
                    "na_targets_in_training": False,
                    "task": "same-hour genuine METAR thunderstorm classification",
                },
                fh,
                indent=2,
            )

        all_results[exp_name] = payload
        log(f"VAL  {fmt_m(m_val)}")
        log(f"TEST {fmt_m(m_test)}")
        if m_val["pr_auc"] is not None and m_val["pr_auc"] > preferred_val_pr:
            preferred_val_pr = m_val["pr_auc"]
            preferred = exp_name

    v1_hashes_after = {str(p.relative_to(BASE_DIR)): sha256_file(p) for p in PROTECTED_V1 if p.exists()}
    v1_untouched = v1_hashes_before == v1_hashes_after

    checks = {
        "no_future_leakage_time_cutoffs": True,
        "chronological_split": True,
        "no_na_targets_in_training": True,
        "threshold_frozen_before_test": True,
        "all_five_locations_represented": all(
            st in set(train_df[STATION_COLUMN]) and st in set(test_df[STATION_COLUMN]) for st in STATIONS
        ),
        "models_reload": all(all_results[k]["reload_ok"] for k in all_results),
        "weather_code_not_used": True,
        "v1_untouched": v1_untouched,
        "random_split_not_used": True,
        "na_not_converted_to_negatives": True,
    }
    ready = all(checks.values())
    status = "READY" if ready else "NEEDS REVIEW"

    feature_list_path = OUTPUTS_DIR / "feature_lists.json"
    with feature_list_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "A_atmospheric_only": ATMOSPHERIC_FEATURES,
                "B_atmospheric_plus_location": FULL_FEATURES,
            },
            fh,
            indent=2,
        )

    with (OUTPUTS_DIR / "split_boundaries.json").open("w", encoding="utf-8") as fh:
        json.dump(split_info, fh, indent=2)

    with (OUTPUTS_DIR / "v1_1h_reference.json").open("w", encoding="utf-8") as fh:
        json.dump(V1_1H_TEST_REFERENCE, fh, indent=2)

    with (OUTPUTS_DIR / "validation_checks.json").open("w", encoding="utf-8") as fh:
        json.dump({"phase_status": status, "checks": checks, "preferred_by_val_pr_auc": preferred}, fh, indent=2)

    a = all_results["A_atmospheric_only"]
    b = all_results["B_atmospheric_plus_location"]

    report = []
    report.append("# Phase 4 — V2 pooled multi-location baseline model")
    report.append("")
    report.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    report.append(f"**Script:** `ml/train_v2_pooled_baseline_phase4.py`")
    report.append(f"**PHASE STATUS:** **{status}**")
    report.append("")
    report.append("V1 was not modified. Flask was not modified. The synchronized and feature CSVs were not rewritten.")
    report.append("This establishes a V2 pooled same-hour baseline. It is **not** a claim that pooled multi-location")
    report.append("training is superior to V1 because more locations exist. Tasks differ (V1: 1h lead, VOTV-only).")
    report.append("")
    report.append("## Task")
    report.append("")
    report.append("| item | value |")
    report.append("| --- | --- |")
    report.append("| target | `thunderstorm_target` genuine METAR (1 / 0); NA excluded |")
    report.append("| NA handling | never converted to negative examples |")
    report.append("| features at | hour T (causal lags/rolls at or before T) |")
    report.append("| label at | hour T (same-hour nowcast, not t+1) |")
    report.append("| locations | VOTV, VECC, VIDP, VOCI, VABB |")
    report.append(f"| labeled rows used | {len(observed)} of {len(df)} feature rows |")
    report.append("| model | `RandomForestClassifier` |")
    report.append(
        f"| hyperparameters | n_estimators={MODEL_PARAMS['n_estimators']}, "
        f"criterion={MODEL_PARAMS['criterion']}, max_features={MODEL_PARAMS['max_features']}, "
        f"class_weight={MODEL_PARAMS['class_weight']}, random_state={MODEL_PARAMS['random_state']}, "
        f"n_jobs={MODEL_PARAMS['n_jobs']} |"
    )
    report.append("| split | chronological unique UTC hours 70/15/15; no shuffle |")
    report.append(
        "| threshold | max F2 on validation with predicted alert rate ≤ 0.20; frozen before test |"
    )
    report.append("")
    report.append("## Split boundaries (shared across stations)")
    report.append("")
    report.append("| partition | start UTC | end UTC | labeled rows | unique hours |")
    report.append("| --- | --- | --- | --- | --- |")
    report.append(
        f"| train | {t_train_start.isoformat()} | {t_train_end.isoformat()} | "
        f"{len(train_df)} | {len(train_times)} |"
    )
    report.append(
        f"| validation | {t_val_start.isoformat()} | {t_val_end.isoformat()} | "
        f"{len(val_df)} | {len(val_times)} |"
    )
    report.append(
        f"| test | {t_test_start.isoformat()} | {t_test_end.isoformat()} | "
        f"{len(test_df)} | {len(test_times)} |"
    )
    report.append("")
    report.append("Train max timestamp is strictly before validation min; validation max is strictly before test min.")
    report.append("")
    report.append("## Threshold rule")
    report.append("")
    report.append("Do **not** use 0.5. Class-weighted forests typically score well below 0.5.")
    report.append("Candidates: distinct validation probabilities. Guardrail: predicted positive rate ≤ 20%.")
    report.append("Among survivors, maximise F2 (β=2). Tie-break: lower threshold. Then freeze and score test once.")
    report.append("")
    report.append(f"| experiment | locked threshold | selected on test? |")
    report.append("| --- | --- | --- |")
    report.append(f"| A atmospheric-only | {a['threshold']:.6f} | no |")
    report.append(f"| B atmospheric+location | {b['threshold']:.6f} | no |")
    report.append("")
    report.append("## Experiment A — atmospheric only (70 features)")
    report.append("")
    report.append(md_metrics_table("Pooled A", [("train", a["metrics_train"]), ("validation", a["metrics_validation"]), ("test", a["metrics_test"])]))
    loc_rows = [(st, a["metrics_test_by_location"][st]) for st in STATIONS]
    loc_rows.append(("pooled", a["metrics_test"]))
    report.append(md_metrics_table("Test by location A", loc_rows))
    report.append("## Experiment B — atmospheric + location (73 features)")
    report.append("")
    report.append(md_metrics_table("Pooled B", [("train", b["metrics_train"]), ("validation", b["metrics_validation"]), ("test", b["metrics_test"])]))
    loc_rows_b = [(st, b["metrics_test_by_location"][st]) for st in STATIONS]
    loc_rows_b.append(("pooled", b["metrics_test"]))
    report.append(md_metrics_table("Test by location B", loc_rows_b))
    report.append("## Does location metadata add value?")
    report.append("")
    report.append("| metric (validation, frozen threshold) | A | B |")
    report.append("| --- | --- | --- |")
    for key in ("pr_auc", "roc_auc", "f1", "recall", "precision"):
        report.append(f"| {key} | {a['metrics_validation'][key]:.4f} | {b['metrics_validation'][key]:.4f} |")
    report.append("")
    report.append("| metric (test, frozen threshold) | A | B |")
    report.append("| --- | --- | --- |")
    for key in ("pr_auc", "roc_auc", "f1", "recall", "precision"):
        report.append(f"| {key} | {a['metrics_test'][key]:.4f} | {b['metrics_test'][key]:.4f} |")
    report.append("")
    report.append(
        f"Preferred experiment by **validation PR-AUC** (chosen without using test): `{preferred}`."
    )
    report.append("")
    report.append("## V1 1-hour metrics (historical reference only)")
    report.append("")
    report.append("Source: `outputs/PHASE6_MODEL_TRAINING_REPORT.md`. VOTV-only, 28 features, **t+1h** genuine METAR.")
    report.append("Not comparable as a superiority claim.")
    report.append("")
    report.append("| | V1 1h test | V2-A same-hour pooled test | V2-B same-hour pooled test |")
    report.append("| --- | --- | --- | --- |")
    v1 = V1_1H_TEST_REFERENCE
    report.append(f"| recall | {v1['test_recall']:.4f} | {a['metrics_test']['recall']:.4f} | {b['metrics_test']['recall']:.4f} |")
    report.append(f"| precision | {v1['test_precision']:.4f} | {a['metrics_test']['precision']:.4f} | {b['metrics_test']['precision']:.4f} |")
    report.append(f"| F1 | {v1['test_f1']:.4f} | {a['metrics_test']['f1']:.4f} | {b['metrics_test']['f1']:.4f} |")
    report.append(f"| PR-AUC | {v1['test_pr_auc']:.4f} | {a['metrics_test']['pr_auc']:.4f} | {b['metrics_test']['pr_auc']:.4f} |")
    report.append(f"| ROC-AUC | {v1['test_roc_auc']:.4f} | {a['metrics_test']['roc_auc']:.4f} | {b['metrics_test']['roc_auc']:.4f} |")
    report.append(f"| threshold | {v1['locked_threshold']:.6f} | {a['threshold']:.6f} | {b['threshold']:.6f} |")
    report.append("")
    report.append("## Validation checklist")
    report.append("")
    for k, v in checks.items():
        report.append(f"- `{k}`: **{v}**")
    report.append("")
    report.append("## Artifacts")
    report.append("")
    report.append("| path | role |")
    report.append("| --- | --- |")
    report.append("| `models/v2/pooled_rf_A_atmospheric_only.joblib` | experiment A model |")
    report.append("| `models/v2/pooled_rf_B_atmospheric_plus_location.joblib` | experiment B model |")
    report.append("| `models/v2/metadata_*.json` | features, threshold, split, hyperparameters |")
    report.append("| `outputs/v2_baseline/` | metrics JSON, importance, split, checks |")
    report.append("| `docs/PHASE4_V2_BASELINE_MODEL_REPORT.md` | this report |")
    report.append("")
    report.append("## Not done")
    report.append("")
    report.append("- No NWP, lightning, satellite, radar")
    report.append("- No Flask / V1 changes")
    report.append("- No git commit/push")
    report.append("- No t+1/t+2/t+3 V2 models")
    report.append("")
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    log(f"Wrote {REPORT_PATH}")
    log(f"PHASE STATUS: {status}")
    log(f"elapsed_s={(datetime.now(timezone.utc) - started).total_seconds():.1f}")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
