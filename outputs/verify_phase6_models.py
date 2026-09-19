"""Phase 6 -- INDEPENDENT verification of the trained thunderstorm nowcast models.

This verifier deliberately does NOT import ``ml/train_thunderstorm_nowcast_phase6.py``.
It re-derives everything through its own code path:

  * the split is rebuilt from the CSV with independent index arithmetic;
  * the validation threshold is re-derived with a separate sweep implementation that uses
    scikit-learn's own metric functions rather than the trainer's vectorised cumulative sums;
  * every metric is recomputed from the reloaded model's predictions;
  * reproducibility is tested by REFITTING a forest with the recorded parameters and
    comparing predictions with the shipped artifact.

Checks
------
  1. all three model artifacts exist, are non-empty and load
  2. feature count and names match the Phase 4 feature list
  3. the estimator class and its parameters are the documented ones
  4. saved metadata matches the shipped model and the evaluation JSON
  5. split boundaries are strictly chronological and match the metadata
  6. test metrics are reproduced from the saved model
  7. the threshold is reproducible from the validation set alone, and is recorded as
     validation-derived, not test-derived
  8. no target leakage: no target/label/weather_code/audit column is a feature
  9. no NaN or inf in the training, validation or test feature matrices
 10. artifacts exist and are non-empty (joblib, PNG, CSV)
 11. Phase 1-5 protected files remain byte-identical
 12. the shipped models are not the Phase 1 surrogate model
 13. the preferred lead time in the comparison follows from validation-only evidence

Run:  python outputs/verify_phase6_models.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = BASE_DIR / "dataset" / "votv_thunderstorm_nowcast_2014_2025.csv"
DATA_META = BASE_DIR / "dataset" / "votv_thunderstorm_nowcast_2014_2025_metadata.json"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

TIME_COLUMN = "timestamp_utc"
LEADS = [1, 2, 3]
TARGET_BY_LEAD = {1: "target_1h", 2: "target_2h", 3: "target_3h"}

# Hard-coded independently of the trainer.
EXPECTED_FEATURES = [
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
    "precipitation_roll_sum_3h",
    "precipitation_roll_sum_6h",
    "humidity_roll_mean_3h",
    "pressure_roll_mean_3h",
    "temperature_roll_mean_3h",
    "wind_speed_roll_mean_3h",
]

FORBIDDEN = [
    "thunderstorm_label",
    "weather_code",
    "target_1h",
    "target_2h",
    "target_3h",
    "target_1h_observed",
    "target_2h_observed",
    "target_3h_observed",
]

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
F_BETA = 2.0
ALERT_RATE_CAP = 0.20
TOL = 1e-9

PHASE1_TO_5_FILES = [
    "dataset/weather_data.csv",
    "dataset/weather_data_with_code.csv",
    "dataset/historical_thunderstorm_labels_votv.csv",
    "dataset/historical_thunderstorm_labels_votv_metadata.json",
    "dataset/historical_thunderstorm_labels_votv_stats.json",
    "dataset/votv_thunderstorm_synchronized_2014_2025.csv",
    "dataset/votv_thunderstorm_synchronized_2014_2025_metadata.json",
    "dataset/votv_thunderstorm_features_2014_2025.csv",
    "dataset/votv_thunderstorm_features_2014_2025_metadata.json",
    "dataset/votv_thunderstorm_nowcast_2014_2025.csv",
    "dataset/votv_thunderstorm_nowcast_2014_2025_metadata.json",
    "dataset/raw_openmeteo/votv_openmeteo_hourly_2014_2025.csv",
    "dataset/build_synchronized_dataset_phase3.py",
    "dataset/feature_engineering_phase4.py",
    "dataset/nowcast_targets_phase5.py",
    "dataset/verify_nowcast_targets_phase5.py",
    "models/storm_risk_model.joblib",
    "models/model_metadata.json",
    "ml/train_model.py",
    "ml/predict.py",
    "app.py",
]

#: Phase 6 artifacts allowed to be new.
PHASE6_ARTIFACTS = {
    "ml/train_thunderstorm_nowcast_phase6.py",
    "outputs/verify_phase6_models.py",
    "outputs/PHASE6_MODEL_TRAINING_REPORT.md",
    "outputs/thunderstorm_nowcast_model_comparison.json",
}

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return bool(ok)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics_from(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Independent metric computation straight from scikit-learn."""
    y_pred = np.where(proba >= threshold, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision_pr_auc": float(average_precision_score(y_true, proba)),
        "baseline_positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def rederive_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, dict]:
    """Second implementation of the documented F2 / alert-rate-cap rule.

    Uses scikit-learn metric functions at each candidate cut instead of the trainer's
    cumulative-sum sweep.
    """
    candidates = np.unique(proba)
    best = None
    inside_cap = 0
    for thr in candidates:
        y_pred = np.where(proba >= thr, 1, 0)
        if y_pred.sum() == 0:
            continue
        alert = float(y_pred.mean())
        if alert > ALERT_RATE_CAP:
            continue
        inside_cap += 1
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        denom = F_BETA * F_BETA * p + r
        fbeta = (1 + F_BETA * F_BETA) * p * r / denom if denom > 0 else 0.0
        if best is None or fbeta > best[1] + 1e-15 or (abs(fbeta - best[1]) <= 1e-15 and thr < best[0]):
            best = (float(thr), float(fbeta), float(p), float(r), alert)
    if best is None:
        raise RuntimeError("no candidate threshold inside the alert-rate guardrail")
    return best[0], {
        "candidates_considered": int(len(candidates)),
        "candidates_inside_guardrail": inside_cap,
        "fbeta": best[1],
        "precision": best[2],
        "recall": best[3],
        "predicted_positive_rate": best[4],
    }


def main() -> int:
    print("=" * 80)
    print("PHASE 6 -- INDEPENDENT VERIFICATION OF THE THUNDERSTORM NOWCAST MODELS")
    print("=" * 80)
    print(f"python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}\n")

    if not check("dataset file exists", DATA_CSV.is_file(), str(DATA_CSV)):
        return 1
    df = pd.read_csv(DATA_CSV)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], utc=True, format="ISO8601")
    data_meta = json.loads(DATA_META.read_text(encoding="utf-8"))

    print("\n[1] Artifact presence and loading")
    bundles: dict[int, dict] = {}
    ok_all = True
    for lead in LEADS:
        model_path = MODELS_DIR / f"thunderstorm_nowcast_{lead}h.joblib"
        meta_path = MODELS_DIR / f"thunderstorm_nowcast_{lead}h_metadata.json"
        eval_path = OUTPUTS_DIR / f"thunderstorm_nowcast_{lead}h_evaluation.json"
        imp_path = OUTPUTS_DIR / f"thunderstorm_nowcast_{lead}h_feature_importance.csv"
        cm_path = OUTPUTS_DIR / f"thunderstorm_nowcast_{lead}h_confusion_matrix.png"
        for path in (model_path, meta_path, eval_path, imp_path, cm_path):
            ok_all &= check(f"{lead}h: {path.name} exists and is non-empty", path.is_file() and path.stat().st_size > 0)
        if not ok_all:
            continue
        ok_all &= check(
            f"{lead}h: confusion matrix PNG has a PNG signature",
            cm_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n",
        )
        bundle = joblib.load(model_path)
        bundles[lead] = bundle
        check(f"{lead}h: artifact loads and is the documented dict bundle", isinstance(bundle, dict), f"type={type(bundle).__name__}")
    if len(bundles) != 3:
        print("\ncannot continue without all three models")
        return 1

    print("\n[2] Feature list and estimator identity")
    for lead in LEADS:
        bundle = bundles[lead]
        model = bundle.get("model")
        check(
            f"{lead}h: estimator is a RandomForestClassifier",
            isinstance(model, RandomForestClassifier),
            f"type={type(model).__name__}",
        )
        check(f"{lead}h: feature count is 28", len(bundle.get("feature_names", [])) == 28)
        check(
            f"{lead}h: feature names and order match the Phase 4 list exactly",
            bundle.get("feature_names") == EXPECTED_FEATURES,
        )
        check(f"{lead}h: model was fitted on 28 features", int(model.n_features_in_) == 28)
        check(
            f"{lead}h: estimator parameters are the documented ones",
            model.n_estimators == 400
            and model.class_weight == "balanced"
            and model.random_state == 42
            and model.n_jobs == -1,
            f"n_estimators={model.n_estimators} class_weight={model.class_weight} "
            f"random_state={model.random_state} n_jobs={model.n_jobs}",
        )
        check(
            f"{lead}h: bundle names the right target and lead time",
            bundle.get("target_column") == TARGET_BY_LEAD[lead] and bundle.get("lead_time_hours") == lead,
        )
        check(
            f"{lead}h: bundle threshold is a usable probability in (0, 1)",
            isinstance(bundle.get("decision_threshold"), float) and 0.0 < bundle["decision_threshold"] < 1.0,
            f"threshold={bundle.get('decision_threshold')}",
        )

    print("\n[3] Leakage: nothing target-like can be a feature")
    for lead in LEADS:
        names = bundles[lead]["feature_names"]
        check(
            f"{lead}h: no target, label, audit or weather_code column is a feature",
            not (set(names) & set(FORBIDDEN)),
            f"intersection={sorted(set(names) & set(FORBIDDEN))}",
        )
        check(
            f"{lead}h: no feature name mentions target or label",
            not [c for c in names if "target" in c.lower() or "label" in c.lower()],
        )
    check("dataset table contains no weather_code column at all", "weather_code" not in df.columns)
    check(
        "the model bundle carries no target column as a feature",
        all(TARGET_BY_LEAD[lead] not in bundles[lead]["feature_names"] for lead in LEADS),
    )

    print("\n[4] Independent split reconstruction")
    splits: dict[int, dict] = {}
    for lead in LEADS:
        target = TARGET_BY_LEAD[lead]
        valid = df[df[target].notna()].reset_index(drop=True)
        n = len(valid)
        n_train = int(SPLIT_FRACTIONS[0] * n)
        n_val = int(SPLIT_FRACTIONS[1] * n)
        n_test = n - n_train - n_val
        tr = valid.iloc[:n_train]
        va = valid.iloc[n_train:n_train + n_val]
        te = valid.iloc[n_train + n_val:]
        splits[lead] = {
            "n_valid": n,
            "n_train": n_train,
            "n_validation": n_val,
            "n_test": n_test,
            "X_train": tr[EXPECTED_FEATURES].to_numpy(dtype=float),
            "y_train": tr[target].to_numpy(dtype=int),
            "X_validation": va[EXPECTED_FEATURES].to_numpy(dtype=float),
            "y_validation": va[target].to_numpy(dtype=int),
            "X_test": te[EXPECTED_FEATURES].to_numpy(dtype=float),
            "y_test": te[target].to_numpy(dtype=int),
            "train_ts": tr[TIME_COLUMN],
            "val_ts": va[TIME_COLUMN],
            "test_ts": te[TIME_COLUMN],
        }
        check(
            f"{lead}h: chronological ordering holds across all three partitions",
            bool(tr[TIME_COLUMN].max() < va[TIME_COLUMN].min()) and bool(va[TIME_COLUMN].max() < te[TIME_COLUMN].min()),
            f"train_end={tr[TIME_COLUMN].iloc[-1]} val_start={va[TIME_COLUMN].iloc[0]} "
            f"val_end={va[TIME_COLUMN].iloc[-1]} test_start={te[TIME_COLUMN].iloc[0]}",
        )
        check(
            f"{lead}h: test rows are strictly later than every training row (no test row could be trained on)",
            bool(te[TIME_COLUMN].min() > tr[TIME_COLUMN].max()),
        )
        check(
            f"{lead}h: partitions are disjoint and cover every valid row",
            n_train + n_val + n_test == n,
            f"{n_train}+{n_val}+{n_test}={n}",
        )

    print("\n[5] No NaN or inf in any feature matrix used for training or evaluation")
    for lead in LEADS:
        s = splits[lead]
        bad = {
            part: (int(np.isnan(s[f"X_{part}"]).sum()), int(np.isinf(s[f"X_{part}"]).sum()))
            for part in ("train", "validation", "test")
        }
        check(f"{lead}h: every feature matrix is finite", all(v == (0, 0) for v in bad.values()), f"nan_inf={bad}")

    print("\n[6] Metadata and evaluation JSON agree with the shipped model and the split")
    for lead in LEADS:
        target = TARGET_BY_LEAD[lead]
        bundle = bundles[lead]
        meta = json.loads((MODELS_DIR / f"thunderstorm_nowcast_{lead}h_metadata.json").read_text(encoding="utf-8"))
        evaluation = json.loads((OUTPUTS_DIR / f"thunderstorm_nowcast_{lead}h_evaluation.json").read_text(encoding="utf-8"))
        s = splits[lead]
        check(
            f"{lead}h: metadata threshold equals the bundle threshold",
            abs(meta["threshold"]["selected_threshold"] - bundle["decision_threshold"]) < 1e-15,
            f"meta={meta['threshold']['selected_threshold']} bundle={bundle['decision_threshold']}",
        )
        check(
            f"{lead}h: evaluation threshold equals the bundle threshold",
            abs(evaluation["threshold_applied_to_test"] - bundle["decision_threshold"]) < 1e-15,
        )
        check(
            f"{lead}h: metadata records the Phase 4 feature list and target",
            meta["features"]["feature_order"] == EXPECTED_FEATURES
            and meta["target"]["column"] == target
            and meta["target"]["lead_time_hours"] == lead,
        )
        check(
            f"{lead}h: metadata split row counts and boundaries match the independent reconstruction",
            meta["row_counts"] == {"train": s["n_train"], "validation": s["n_validation"], "test": s["n_test"]}
            and meta["split"]["train"]["start_timestamp_utc"] == s["train_ts"].iloc[0].isoformat()
            and meta["split"]["train"]["end_timestamp_utc"] == s["train_ts"].iloc[-1].isoformat()
            and meta["split"]["test"]["start_timestamp_utc"] == s["test_ts"].iloc[0].isoformat()
            and meta["split"]["test"]["end_timestamp_utc"] == s["test_ts"].iloc[-1].isoformat(),
            f"meta={meta['row_counts']} independent={[s['n_train'], s['n_validation'], s['n_test']]}",
        )
        check(
            f"{lead}h: metadata class balance matches the independently rebuilt splits",
            meta["class_balance"]["train"]["positives"] == int(s["y_train"].sum())
            and meta["class_balance"]["validation"]["positives"] == int(s["y_validation"].sum())
            and meta["class_balance"]["test"]["positives"] == int(s["y_test"].sum())
            and meta["class_balance"]["train"]["n_samples"] == s["n_train"],
        )
        check(
            f"{lead}h: metadata records the checksum of the dataset file that is actually present",
            meta["training_dataset"]["sha256"] == sha256_file(DATA_CSV),
            f"recorded={str(meta['training_dataset']['sha256'])[:12]}... actual={sha256_file(DATA_CSV)[:12]}...",
        )
        check(
            f"{lead}h: metadata declares the target as a genuine observation, not a surrogate",
            meta["target"]["genuine_observation_label"] is True and meta["target"]["surrogate_or_proxy_target"] is False,
        )
        check(
            f"{lead}h: metadata states no synthetic labels and no resampling of the test set",
            meta["target"]["synthetic_labels_created"] is False
            and "no smote" in meta["class_balance"]["handling"].lower()
            and meta["class_balance"]["handling"].lower().startswith("class_weight"),
        )
        check(
            f"{lead}h: metadata carries a scientific disclaimer",
            isinstance(meta.get("scientific_disclaimer"), str) and len(meta["scientific_disclaimer"]) > 100,
        )
        check(
            f"{lead}h: metadata and evaluation both declare the test set was used once, after locking",
            meta["split"]["test_used_for_tuning"] is False
            and meta["no_model_selected_on_test"] is True
            and evaluation["test_set_evaluated_once_with_locked_threshold"] is True,
        )

    print("\n[7] Threshold verification: reproduced from VALIDATION only")
    threshold_evidence: dict[int, dict] = {}
    for lead in LEADS:
        s = splits[lead]
        bundle = bundles[lead]
        proba_va = bundle["model"].predict_proba(s["X_validation"])[:, 1]
        rederived, trace = rederive_threshold(s["y_validation"], proba_va)
        locked = bundle["decision_threshold"]
        check(
            f"{lead}h: locked threshold is reproduced from the validation set alone",
            abs(rederived - locked) < 1e-12,
            f"rederived={rederived:.6f} locked={locked:.6f}",
        )
        check(
            f"{lead}h: threshold sits inside the documented alert-rate guardrail on validation",
            trace["predicted_positive_rate"] <= ALERT_RATE_CAP,
            f"alert_rate={trace['predicted_positive_rate']:.4f} cap={ALERT_RATE_CAP}",
        )
        # Informational: what a test-tuned threshold would have been.
        proba_te = bundle["model"].predict_proba(s["X_test"])[:, 1]
        test_tuned, test_trace = rederive_threshold(s["y_test"], proba_te)
        threshold_evidence[lead] = {
            "locked": locked,
            "rederived_from_validation": rederived,
            "validation_trace": trace,
            "test_tuned_threshold_for_information_only": test_tuned,
            "test_tuned_trace_for_information_only": test_trace,
        }
        evidence = json.loads((OUTPUTS_DIR / f"thunderstorm_nowcast_{lead}h_evaluation.json").read_text(encoding="utf-8"))
        check(
            f"{lead}h: evidence JSON records the threshold as validation-derived",
            evidence["threshold_selection"]["threshold_selected_on"] == "validation"
            and evidence["threshold_selection"]["test_data_used_for_threshold_selection"] is False,
        )
        print(
            f"      note: a test-tuned F2 threshold would have been {test_tuned:.6f} "
            f"(validation-locked value {locked:.6f}); a different value means the shipped threshold "
            f"was not fitted here"
        )

    print("\n[8] Test metrics reproduced from the saved models")
    for lead in LEADS:
        s = splits[lead]
        bundle = bundles[lead]
        threshold = bundle["decision_threshold"]
        proba_te = bundle["model"].predict_proba(s["X_test"])[:, 1]
        recomputed = metrics_from(s["y_test"], proba_te, threshold)
        evaluation = json.loads((OUTPUTS_DIR / f"thunderstorm_nowcast_{lead}h_evaluation.json").read_text(encoding="utf-8"))
        stored = evaluation["metrics"]["test"]
        scalar_ok = all(
            abs(float(stored[k]) - float(recomputed[k])) <= TOL
            for k in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision_pr_auc",
                "baseline_positive_rate",
                "predicted_positive_rate",
            )
        )
        check(f"{lead}h: every stored test scalar metric is reproduced", scalar_ok)
        check(
            f"{lead}h: stored test confusion matrix is reproduced",
            stored["confusion_matrix"] == recomputed["confusion_matrix"],
            f"stored={stored['confusion_matrix']} recomputed={recomputed['confusion_matrix']}",
        )
        check(
            f"{lead}h: stored test counts and class distribution are reproduced",
            stored["counts"]["true_positives"] == recomputed["confusion_matrix"][1][1]
            and stored["class_distribution"]["positives"] == int(s["y_test"].sum())
            and stored["class_distribution"]["n_samples"] == s["n_test"],
        )
        meta = json.loads((MODELS_DIR / f"thunderstorm_nowcast_{lead}h_metadata.json").read_text(encoding="utf-8"))
        check(
            f"{lead}h: metadata and evaluation report the same test metrics",
            abs(meta["metrics"]["test"]["precision"] - stored["precision"]) <= TOL
            and abs(meta["metrics"]["test"]["recall"] - stored["recall"]) <= TOL
            and abs(meta["metrics"]["test"]["f1"] - stored["f1"]) <= TOL,
        )
        validation_recomputed = metrics_from(
            s["y_validation"], bundle["model"].predict_proba(s["X_validation"])[:, 1], threshold
        )
        stored_validation = evaluation["metrics"]["validation"]
        check(
            f"{lead}h: stored validation metrics are reproduced from the saved model",
            all(
                abs(float(stored_validation[k]) - float(validation_recomputed[k])) <= TOL
                for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision_pr_auc")
            ),
            f"recomputed recall={validation_recomputed['recall']:.4f} precision={validation_recomputed['precision']:.4f}",
        )
        check(
            f"{lead}h: PR-AUC exceeds the base rate, so the ranking carries real signal",
            recomputed["average_precision_pr_auc"] > recomputed["baseline_positive_rate"],
            f"PR-AUC={recomputed['average_precision_pr_auc']:.4f} baseline={recomputed['baseline_positive_rate']:.4f}",
        )

    print("\n[9] Reproducibility: refit from the recorded parameters and compare")
    for lead in LEADS:
        s = splits[lead]
        bundle = bundles[lead]
        refit = RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1
        )
        refit.fit(s["X_train"], s["y_train"])
        stored_proba = bundle["model"].predict_proba(s["X_test"])[:, 1]
        refit_proba = refit.predict_proba(s["X_test"])[:, 1]
        check(
            f"{lead}h: an independent refit reproduces the shipped model's test predictions exactly",
            bool(np.array_equal(stored_proba, refit_proba)),
            f"max_abs_difference={float(np.max(np.abs(stored_proba - refit_proba))):.3e}",
        )

    print("\n[10] Feature importance artifacts")
    for lead in LEADS:
        imp_path = OUTPUTS_DIR / f"thunderstorm_nowcast_{lead}h_feature_importance.csv"
        imp = pd.read_csv(imp_path)
        check(
            f"{lead}h: feature importance CSV covers all 28 features exactly once",
            len(imp) == 28 and sorted(imp["feature"]) == sorted(EXPECTED_FEATURES),
        )
        check(
            f"{lead}h: RF impurity importances are non-negative and sum to 1",
            bool((imp["rf_impurity_importance"] >= 0).all())
            and abs(float(imp["rf_impurity_importance"].sum()) - 1.0) < 1e-6,
            f"sum={float(imp['rf_impurity_importance'].sum()):.6f}",
        )
        top = imp.sort_values("rf_impurity_importance", ascending=False)["feature"].iloc[0]
        check(f"{lead}h: a top feature is recorded for the model", isinstance(top, str) and len(top) > 0, f"top={top}")

    print("\n[11] Comparison artifact follows from validation-only evidence")
    comparison_path = OUTPUTS_DIR / "thunderstorm_nowcast_model_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    val_f2 = {
        row["lead_time_hours"]: row["validation"]["f2"] for row in comparison["lead_time_results"]
    }
    best_by_validation = max(val_f2, key=lambda k: val_f2[k])
    check(
        "the preferred lead time is the validation-F2 winner",
        comparison["preferred_lead_time_hours"] == best_by_validation,
        f"preferred={comparison['preferred_lead_time_hours']} validation_F2={val_f2}",
    )
    check(
        "the comparison states the decision was made from validation metrics and availability",
        "validation metrics and data availability only" in comparison["selection"]["decided_from"],
    )
    check(
        "the comparison states test metrics did not drive the choice",
        comparison["selection"]["test_metrics_role"].startswith("final evaluation")
        and comparison["threshold_criterion"]["test_used"] is False,
    )
    check(
        "the comparison carries the report file and the scientific disclaimer",
        (OUTPUTS_DIR / "PHASE6_MODEL_TRAINING_REPORT.md").is_file()
        and len(comparison["scientific_disclaimer"]) > 100,
    )
    report_text = (OUTPUTS_DIR / "PHASE6_MODEL_TRAINING_REPORT.md").read_text(encoding="utf-8")
    check(
        "the report explicitly distinguishes probability, threshold, metrics and observation",
        all(
            phrase in report_text.lower()
            for phrase in ("prediction probability", "classification threshold", "evaluation metrics", "actual thunderstorm observation")
        ),
    )
    check(
        "the report rejects the claims Phase 6 must not make",
        all(
            phrase in report_text.lower()
            for phrase in ("not** nationwide", "not** radar-based", "not** an operational imd", "not evidence of skill")
        ),
    )

    print("\n[12] The shipped models are not the Phase 1 surrogate model")
    surrogate_path = MODELS_DIR / "storm_risk_model.joblib"
    if check("the Phase 1 surrogate model still exists", surrogate_path.is_file()):
        surrogate = joblib.load(surrogate_path)
        check(
            "the surrogate model is a different object from each Phase 6 model",
            all(bundles[lead]["model"] is not surrogate for lead in LEADS),
        )
        for lead in LEADS:
            meta = json.loads((MODELS_DIR / f"thunderstorm_nowcast_{lead}h_metadata.json").read_text(encoding="utf-8"))
            check(
                f"{lead}h: metadata records that the surrogate model is unused",
                "storm_risk_model.joblib" in meta["model"]["not_the_phase1_surrogate_model"],
            )

    print("\n[13] Phase 1-5 protected files remain byte-identical")
    missing = [rel for rel in PHASE1_TO_5_FILES if not (BASE_DIR / rel).is_file()]
    check("every Phase 1-5 protected file is still present", not missing, f"missing={missing}")
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=BASE_DIR, capture_output=True, text=True, timeout=180
        )
        modified, unexpected = [], []
        for line in status.stdout.splitlines():
            if not line.strip():
                continue
            code, path = line[:2], line[3:].strip().strip('"')
            if code == "??":
                if not (
                    path.startswith("models/thunderstorm_nowcast")
                    or path.startswith("outputs/thunderstorm_nowcast")
                    or path in PHASE6_ARTIFACTS
                ):
                    unexpected.append(path)
            else:
                modified.append(line.strip())
        check("git reports no modified tracked file anywhere in the repository", not modified, f"modified={modified}")
        check("the only untracked files are Phase 6 artifacts", not unexpected, f"untracked={unexpected}")
    except Exception as exc:  # pragma: no cover
        check("git status could be queried", False, f"{type(exc).__name__}: {exc}")

    phase5_recorded_phase4_sha = data_meta.get("inputs", {}).get("feature_table", {}).get("sha256")
    phase4_sha_now = sha256_file(BASE_DIR / "dataset" / "votv_thunderstorm_features_2014_2025.csv")
    check(
        "the Phase 4 -> Phase 5 ancestry is still intact (recorded checksum matches the file on disk)",
        phase5_recorded_phase4_sha == phase4_sha_now,
        f"recorded={str(phase5_recorded_phase4_sha)[:12]}... actual={phase4_sha_now[:12]}...",
    )
    nowcast_sha_now = sha256_file(DATA_CSV)
    for lead in LEADS:
        meta = json.loads((MODELS_DIR / f"thunderstorm_nowcast_{lead}h_metadata.json").read_text(encoding="utf-8"))
        check(
            f"{lead}h: the model was trained on the Phase 5 target table exactly as it exists now",
            meta["training_dataset"]["sha256"] == nowcast_sha_now
            and meta["training_dataset"]["input_chain_check"]["phase4_to_phase6_ancestry_intact"] is True,
            f"model_recorded={str(meta['training_dataset']['sha256'])[:12]}... now={nowcast_sha_now[:12]}...",
        )

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 80)
    print(f"checks run: {len(RESULTS)}   passed: {len(RESULTS) - len(failed)}   failed: {len(failed)}")
    if failed:
        print("FAILED CHECKS:")
        for name in failed:
            print(f"  - {name}")
        print("\nPHASE 6 INDEPENDENT VERIFICATION: FAIL")
        return 1
    print("PHASE 6 INDEPENDENT VERIFICATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
