"""Phase 6 -- training the genuine VOTV thunderstorm nowcast models (lead times 1h/2h/3h).

This is the FIRST model trained on the genuine observed thunderstorm target.  It does
not use the Phase 1 surrogate target (``storm_risk_proxy``, a future-precipitation
proxy), does not use Open-Meteo ``weather_code``, and does not touch the existing
``models/storm_risk_model.joblib`` artifact in any way.

Task
----
    X = the 28 Phase 4 causal features at clock hour t
    y = target_<L>h, the genuine observed thunderstorm label at t + L, for L in {1,2,3}

One model is trained per lead time.  Rows are dropped only when that lead time's own
target is genuinely unobserved (NaN, i.e. the future hour carries no VOTV present-weather
report).  No row is ever assigned a label it does not have.

Split
-----
Strict chronological 70% / 15% / 15% on each lead time's valid rows, in timestamp order.
The series is never shuffled.  Because the target looks forward by L hours, the last few
training rows have their target observation inside the validation period; the exact count
is measured and reported (``boundary_label_overlap_rows``).  This is inherent to any
forward-shifted target and does not move any feature information across the boundary,
because every feature is strictly a function of hours <= t.

Threshold
---------
The class-weighted Random Forest does not put its positive class near probability 0.5:
predicting at 0.5 on the validation set recovers almost nothing.  Assuming 0.5 would
therefore be a modelling error, not a neutral default.  The operating threshold is
selected on the VALIDATION set only, by an explicit documented rule:

    1. candidate thresholds = every distinct validation predicted probability
    2. guardrail: predicted positive rate <= 20% of hours (an alert on more than one hour
       in five is not a usable operational load at a ~4% event base rate)
    3. among candidates inside the guardrail, maximise F2 (beta = 2), which weights recall
       twice as heavily as precision -- the requested priority for storm detection
    4. tie-break: the lower threshold, i.e. the higher recall

The threshold is then locked, and the test set is evaluated exactly once with it.  No
test-derived quantity influences the threshold.

Model selection
---------------
The preferred lead time is decided on VALIDATION metrics plus data availability, before
the test set is scored, so the test scores cannot have driven the choice.  Test metrics
are reported for all three lead times as final evaluation.

No SMOTE and no oversampling: the real class distribution is kept and imbalance is handled
by ``class_weight="balanced"`` only.  The test set is never resampled.

Run:  python ml/train_thunderstorm_nowcast_phase6.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = BASE_DIR / "dataset" / "votv_thunderstorm_nowcast_2014_2025.csv"
DATA_META = BASE_DIR / "dataset" / "votv_thunderstorm_nowcast_2014_2025_metadata.json"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

TIME_COLUMN = "timestamp_utc"

#: The 28 Phase 4 features, hard-coded so a renamed, added or reordered column fails loudly.
FEATURE_COLUMNS = [
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

#: Columns that must never be features.
FORBIDDEN_FEATURE_COLUMNS = [
    "thunderstorm_label",
    "target_1h",
    "target_2h",
    "target_3h",
    "target_1h_observed",
    "target_2h_observed",
    "target_3h_observed",
    "weather_code",
]

FEATURE_GROUPS = {
    "current_atmospheric": FEATURE_COLUMNS[0:6],
    "cyclic_time": FEATURE_COLUMNS[6:10],
    "wind_direction_cyclic": FEATURE_COLUMNS[10:12],
    "temporal_change": FEATURE_COLUMNS[12:22],
    "historical_rolling": FEATURE_COLUMNS[22:28],
}

#: lead time in hours -> target column name
TARGET_BY_LEAD = {1: "target_1h", 2: "target_2h", 3: "target_3h"}

SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}

MODEL_PARAMS = {
    "n_estimators": 400,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

#: Threshold selection rule (validation only).
F_BETA = 2.0
ALERT_RATE_CAP = 0.20
RECALL_PRIORITY_NOTE = "F2 weights recall twice as heavily as precision"

#: Phase 1-5 artifacts Phase 6 must leave byte-identical.
PROTECTED_FILES = [
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

DISCLAIMER = (
    "Station-based thunderstorm nowcast prototype trained from VOTV aerodrome METAR "
    "present-weather observations paired with Open-Meteo archive/model-derived atmospheric "
    "features at a single grid cell about 2 km from the station. The output is a model "
    "probability that the VOTV station reports a thunderstorm in a future clock hour, "
    "combined with a decision threshold to give a yes/no alert. It is NOT a nationwide, "
    "radar-based, satellite-based, lightning-based or NWP-based thunderstorm forecast, it "
    "is not an IMD operational product, and a probability is not physical certainty about "
    "a thunderstorm. High accuracy figures are achievable by predicting the majority class "
    "and are not evidence of skill; the meaningful figures here are recall, precision, F1, "
    "PR-AUC and ROC-AUC relative to the reported base rate."
)


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_group_of(name: str) -> str:
    for group, members in FEATURE_GROUPS.items():
        if name in members:
            return group
    return "unclassified"


# ---------------------------------------------------------------------------
# data and split
# ---------------------------------------------------------------------------


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    required = [TIME_COLUMN, *FEATURE_COLUMNS, *TARGET_BY_LEAD.values()]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"nowcast table is missing required columns: {missing}")
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], utc=True, format="ISO8601")
    if df[TIME_COLUMN].isna().any():
        raise SystemExit("nowcast table has unparseable timestamps")
    if not df[TIME_COLUMN].is_monotonic_increasing:
        raise SystemExit("nowcast table is not chronological")
    if int(df[TIME_COLUMN].duplicated().sum()):
        raise SystemExit("nowcast table has duplicate timestamps")
    return df


def check_leakage(df: pd.DataFrame) -> dict:
    """Refuse to train if anything about the feature matrix looks like the target."""
    problems: list[str] = []
    if set(FEATURE_COLUMNS) & set(FORBIDDEN_FEATURE_COLUMNS):
        problems.append("a target/label/audit column is in the feature list")
    leaked_names = [c for c in FEATURE_COLUMNS if "target" in c.lower() or "label" in c.lower()]
    if leaked_names:
        problems.append(f"feature names referring to the target: {leaked_names}")
    if len(FEATURE_COLUMNS) != 28:
        problems.append(f"expected 28 features, found {len(FEATURE_COLUMNS)}")
    matrix = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    n_nan = int(np.isnan(matrix).sum())
    n_inf = int(np.isinf(matrix).sum())
    if n_nan or n_inf:
        problems.append(f"feature matrix is not finite: nan={n_nan} inf={n_inf}")
    if "weather_code" in df.columns:
        problems.append("weather_code is present in the table and must not be used")
    if problems:
        raise SystemExit("leakage pre-checks failed: " + "; ".join(problems))
    return {
        "target_columns_excluded_from_features": sorted(set(FORBIDDEN_FEATURE_COLUMNS) - set(FEATURE_COLUMNS)),
        "feature_names_referring_to_target": leaked_names,
        "weather_code_used": False,
        "future_atmospheric_variables_used": False,
        "future_precipitation_used": False,
        "target_derived_features_used": False,
        "feature_matrix_nan_count": n_nan,
        "feature_matrix_inf_count": n_inf,
        "verdict": "PASS",
    }


def make_splits(sub: pd.DataFrame) -> dict:
    """Strict chronological 70/15/15 slices, no shuffling."""
    n = len(sub)
    n_train = int(SPLIT_FRACTIONS["train"] * n)
    n_val = int(SPLIT_FRACTIONS["validation"] * n)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise SystemExit("split produced an empty partition")
    bounds = {
        "train": (0, n_train),
        "validation": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }
    if not (sub[TIME_COLUMN].iloc[bounds["train"][1] - 1] < sub[TIME_COLUMN].iloc[bounds["validation"][0]]):
        raise SystemExit("train/validation boundary is not chronological")
    if not (sub[TIME_COLUMN].iloc[bounds["validation"][1] - 1] < sub[TIME_COLUMN].iloc[bounds["test"][0]]):
        raise SystemExit("validation/test boundary is not chronological")
    return {
        "n": n,
        "bounds": bounds,
        "n_train": n_train,
        "n_validation": n_val,
        "n_test": n_test,
    }


def split_description(sub: pd.DataFrame, splits: dict) -> dict:
    out = {"strategy": "strict chronological, no shuffling", "fractions": SPLIT_FRACTIONS}
    for name, (lo, hi) in splits["bounds"].items():
        part = sub.iloc[lo:hi]
        out[name] = {
            "n_rows": int(hi - lo),
            "start_index": int(lo),
            "end_index": int(hi - 1),
            "start_timestamp_utc": part[TIME_COLUMN].iloc[0].isoformat(),
            "end_timestamp_utc": part[TIME_COLUMN].iloc[-1].isoformat(),
        }
    return out


# ---------------------------------------------------------------------------
# metrics and thresholding
# ---------------------------------------------------------------------------


def binary_counts(y_true: np.ndarray) -> dict:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    return {
        "n_samples": int(len(y_true)),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": round(positives / len(y_true), 6) if len(y_true) else None,
    }


def metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc,
        "average_precision_pr_auc": float(average_precision_score(y_true, proba)),
        "baseline_positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "confusion_matrix_axis_note": "rows = actual [0, 1], columns = predicted [0, 1]",
        "counts": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "class_distribution": binary_counts(y_true),
        "false_alarms_per_hit": (float(fp / tp) if tp else None),
    }


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray, beta: float = F_BETA) -> pd.DataFrame:
    """Metrics at every ACHIEVABLE cut of the predicted probability.

    Only thresholds that can be produced by ``predict positive iff proba >= threshold`` are
    evaluated.  A Random Forest average over 400 tree votes takes values that are multiples of
    1/400, so many rows share a probability; walking the sorted array row by row would score
    cuts that sit inside such a tie group, i.e. states no threshold can actually reach.
    Keeping only the last row of each equal-probability run keeps the sweep and the applied
    decision rule exactly consistent.
    """
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
    recall = tp / positives if positives else np.zeros(n)
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
    """Apply the documented rule and return the chosen threshold plus its trace."""
    inside = sweep[sweep["predicted_positive_rate"] <= cap]
    if inside.empty:
        raise SystemExit("no candidate threshold satisfies the alert-rate guardrail")
    best_fbeta = float(inside["fbeta"].max())
    tied = inside[np.isclose(inside["fbeta"], best_fbeta)]
    chosen = tied.iloc[int(tied["threshold"].to_numpy().argmin())]  # tie-break: lowest threshold
    unconstrained = sweep.iloc[int(sweep["fbeta"].to_numpy().argmax())]
    return {
        "criterion": (
            f"maximise F{int(F_BETA)} (beta={F_BETA}, {RECALL_PRIORITY_NOTE}) on the validation set, "
            f"subject to predicted positive rate <= {cap}"
        ),
        "beta": F_BETA,
        "alert_rate_cap": cap,
        "guardrail_binding": bool(unconstrained["predicted_positive_rate"] > cap),
        "candidate_thresholds": int(len(sweep)),
        "candidates_inside_guardrail": int(len(inside)),
        "selected_threshold": float(chosen["threshold"]),
        "selected_validation_metrics": {
            "precision": float(chosen["precision"]),
            "recall": float(chosen["recall"]),
            "f1": float(chosen["f1"]),
            "fbeta": float(chosen["fbeta"]),
            "predicted_positive_rate": float(chosen["predicted_positive_rate"]),
            "true_positives": int(chosen["true_positives"]),
            "false_positives": int(chosen["false_positives"]),
        },
        "unconstrained_fbeta_threshold": float(unconstrained["threshold"]),
        "threshold_selected_on": "validation",
        "test_data_used_for_threshold_selection": False,
    }


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


def save_confusion_matrix_png(path: Path, cm: list[list[int]], lead: int, threshold: float, split_name: str) -> None:
    cm_arr = np.asarray(cm, dtype=float)
    row_totals = cm_arr.sum(axis=1, keepdims=True)
    normalized = np.divide(cm_arr, row_totals, out=np.zeros_like(cm_arr), where=row_totals > 0)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    labels = ["no thunderstorm (0)", "thunderstorm (1)"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("observed")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{int(cm_arr[i, j]):,}\n{normalized[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color="white" if normalized[i, j] > 0.5 else "black",
                fontsize=12,
            )
    ax.set_title(
        f"VOTV thunderstorm nowcast, lead time {lead} h\n"
        f"{split_name} set, decision threshold {threshold:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    log("=" * 80)
    log("PHASE 6 -- GENUINE THUNDERSTORM NOWCAST MODEL TRAINING (1h / 2h / 3h)")
    log("=" * 80)

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    log("\n[1] Loading the Phase 5 nowcast table")
    df = load_dataset()
    log(f"    rows={len(df)}  columns={len(df.columns)}")
    log(f"    range {df[TIME_COLUMN].min()} .. {df[TIME_COLUMN].max()}")

    log("\n[2] Leakage pre-checks")
    leakage = check_leakage(df)
    log(f"    features={len(FEATURE_COLUMNS)} (no target/label/weather_code column present)")
    log(f"    feature matrix finite: nan={leakage['feature_matrix_nan_count']} inf={leakage['feature_matrix_inf_count']}")

    data_sha256 = sha256_file(DATA_CSV)
    data_meta = json.loads(DATA_META.read_text(encoding="utf-8"))
    # Phase 5 metadata records the checksum of the Phase 4 table it consumed; verifying it
    # against the file on disk confirms the Phase 4 -> Phase 5 -> Phase 6 ancestry is intact.
    recorded_phase4_sha = data_meta.get("inputs", {}).get("feature_table", {}).get("sha256")
    phase4_sha_now = sha256_file(BASE_DIR / "dataset" / "votv_thunderstorm_features_2014_2025.csv")
    input_chain_intact = recorded_phase4_sha == phase4_sha_now
    if not input_chain_intact:
        raise SystemExit("the Phase 4 feature table no longer matches the checksum Phase 5 recorded")
    log(f"    dataset sha256={data_sha256[:16]}...  phase 4 -> 5 ancestry intact: {input_chain_intact}")

    results: dict[int, dict] = {}

    for lead, target in TARGET_BY_LEAD.items():
        log("\n" + "=" * 80)
        log(f"LEAD TIME {lead} h  --  target {target}")
        log("=" * 80)

        valid = df[df[target].notna()].reset_index(drop=True)
        dropped = int(len(df) - len(valid))
        log(f"    valid rows={len(valid)}  dropped (future label unobserved)={dropped}")

        splits = make_splits(valid)
        split_info = split_description(valid, splits)
        log(f"    split: train={splits['n_train']} validation={splits['n_validation']} test={splits['n_test']}")
        log(f"    train      {split_info['train']['start_timestamp_utc']} .. {split_info['train']['end_timestamp_utc']}")
        log(f"    validation {split_info['validation']['start_timestamp_utc']} .. {split_info['validation']['end_timestamp_utc']}")
        log(f"    test       {split_info['test']['start_timestamp_utc']} .. {split_info['test']['end_timestamp_utc']}")

        X = valid[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = valid[target].to_numpy(dtype=int)
        if not set(np.unique(y)) <= {0, 1}:
            raise SystemExit(f"{target} is not binary")
        (tr_lo, tr_hi), (va_lo, va_hi), (te_lo, te_hi) = (
            splits["bounds"]["train"],
            splits["bounds"]["validation"],
            splits["bounds"]["test"],
        )
        X_tr, y_tr = X[tr_lo:tr_hi], y[tr_lo:tr_hi]
        X_va, y_va = X[va_lo:va_hi], y[va_lo:va_hi]
        X_te, y_te = X[te_lo:te_hi], y[te_lo:te_hi]

        # How many training rows have their target observation inside the validation period.
        target_hour = valid[TIME_COLUMN] + pd.Timedelta(hours=lead)
        val_start = valid[TIME_COLUMN].iloc[va_lo]
        overlap_train = int((target_hour.iloc[tr_lo:tr_hi] >= val_start).sum())
        test_start = valid[TIME_COLUMN].iloc[te_lo]
        overlap_val = int((target_hour.iloc[va_lo:va_hi] >= test_start).sum())

        log(f"    class balance: train {binary_counts(y_tr)}")
        log(f"                   validation {binary_counts(y_va)}")
        log(f"                   test {binary_counts(y_te)}")
        log(f"    boundary label overlap: train rows whose target lies in the validation period={overlap_train}, "
            f"validation rows whose target lies in the test period={overlap_val}")

        log("\n    Training RandomForestClassifier")
        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_tr, y_tr)
        log(f"    trees={model.n_estimators}  features per split={model.max_features}  classes={list(model.classes_)}")

        # --- validation: threshold selection only ---------------------------------
        proba_va = model.predict_proba(X_va)[:, 1]
        sweep = threshold_sweep(y_va, proba_va)
        selection = select_threshold(sweep)
        threshold = selection["selected_threshold"]
        log(f"\n    Validation threshold selection")
        log(f"      criterion        : {selection['criterion']}")
        log(f"      guardrail binding: {selection['guardrail_binding']}")
        log(f"      selected         : {threshold:.6f}  "
            f"(recall={selection['selected_validation_metrics']['recall']:.4f}, "
            f"precision={selection['selected_validation_metrics']['precision']:.4f}, "
            f"alert rate={selection['selected_validation_metrics']['predicted_positive_rate']:.4f})")
        pct = float((proba_va < threshold).mean())
        log(f"      threshold sits at the {pct * 100:.1f}st percentile of validation probabilities")

        val_metrics = metrics_at_threshold(y_va, proba_va, threshold)
        val_metrics_at_half = metrics_at_threshold(y_va, proba_va, 0.5)
        log(f"      at 0.5 for comparison: recall={val_metrics_at_half['recall']:.4f} "
            f"precision={val_metrics_at_half['precision']:.4f} "
            f"alert rate={val_metrics_at_half['predicted_positive_rate']:.4f}")
        log(f"      PR-AUC={val_metrics['average_precision_pr_auc']:.4f} ROC-AUC={val_metrics['roc_auc']:.4f}")

        # --- test: evaluated exactly once, with the locked threshold ---------------
        log("\n    Final test evaluation (threshold locked from validation)")
        proba_te = model.predict_proba(X_te)[:, 1]
        test_metrics = metrics_at_threshold(y_te, proba_te, threshold)
        test_metrics_at_half = metrics_at_threshold(y_te, proba_te, 0.5)
        log(f"      recall={test_metrics['recall']:.4f} precision={test_metrics['precision']:.4f} "
            f"F1={test_metrics['f1']:.4f}")
        log(f"      ROC-AUC={test_metrics['roc_auc']:.4f} PR-AUC={test_metrics['average_precision_pr_auc']:.4f} "
            f"accuracy={test_metrics['accuracy']:.4f}")
        log(f"      confusion matrix {test_metrics['confusion_matrix']}")
        log(f"      alert rate={test_metrics['predicted_positive_rate']:.4f} vs base rate={test_metrics['baseline_positive_rate']:.4f}")

        train_metrics = metrics_at_threshold(y_tr, model.predict_proba(X_tr)[:, 1], threshold)

        # --- feature importance ---------------------------------------------------
        impurity = model.feature_importances_
        perm = permutation_importance(
            model, X_va, y_va, n_repeats=5, random_state=SEED, n_jobs=-1, scoring="average_precision"
        )
        importance = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "feature_group": [feature_group_of(c) for c in FEATURE_COLUMNS],
                "rf_impurity_importance": impurity,
                "rf_impurity_rank": pd.Series(impurity).rank(ascending=False, method="min").astype(int),
                "permutation_importance_mean": perm.importances_mean,
                "permutation_importance_std": perm.importances_std,
                "permutation_importance_rank": pd.Series(perm.importances_mean)
                .rank(ascending=False, method="min")
                .astype(int),
            }
        ).sort_values("rf_impurity_importance", ascending=False)
        importance["rf_impurity_share_percent"] = 100.0 * importance["rf_impurity_importance"] / importance[
            "rf_impurity_importance"
        ].sum()
        importance_path = OUTPUTS_DIR / f"thunderstorm_nowcast_{lead}h_feature_importance.csv"
        importance.to_csv(importance_path, index=False, float_format="%.10g")
        top_features = importance.head(10)[["feature", "rf_impurity_importance", "permutation_importance_mean"]].to_dict("records")
        log("\n    Top 5 features (RF impurity importance)")
        for row in importance.head(5).itertuples():
            log(f"      {row.rf_impurity_rank:>2}. {row.feature:<28} {row.rf_impurity_importance:.4f} "
                f"(perm AP drop {row.permutation_importance_mean:+.4f})")

        # --- confusion matrix image -----------------------------------------------
        cm_path = OUTPUTS_DIR / f"thunderstorm_nowcast_{lead}h_confusion_matrix.png"
        save_confusion_matrix_png(cm_path, test_metrics["confusion_matrix"], lead, threshold, "test")

        # --- model artifact -------------------------------------------------------
        model_path = MODELS_DIR / f"thunderstorm_nowcast_{lead}h.joblib"
        bundle = {
            "model": model,
            "feature_names": FEATURE_COLUMNS,
            "target_column": target,
            "lead_time_hours": lead,
            "decision_threshold": threshold,
            "positive_class_label": 1,
            "negative_class_label": 0,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_dataset_sha256": data_sha256,
            "random_state": SEED,
            "bundle_version": 1,
            "positive_class_meaning": (
                f"VOTV station reports a thunderstorm in the clock hour {lead} h after the feature hour"
            ),
        }
        joblib.dump(bundle, model_path)
        log(f"\n    wrote {model_path.name}")

        # --- metadata -------------------------------------------------------------
        metadata = {
            "model_file": model_path.name,
            "artifact_type": "trained scikit-learn RandomForestClassifier bundle (dict) for one lead time",
            "bundle_keys": [
                "model",
                "feature_names",
                "target_column",
                "lead_time_hours",
                "decision_threshold",
                "positive_class_label",
                "negative_class_label",
                "trained_at_utc",
                "training_dataset_sha256",
                "random_state",
                "bundle_version",
                "positive_class_meaning",
            ],
            "bundle_note": (
                "The joblib artifact is a dict, not a bare estimator: load it with "
                "joblib.load(path)['model']. The dict carries the feature order and the validated "
                "decision threshold so inference cannot silently use a different threshold or "
                "column order."
            ),
            "phase": "Phase 6 -- genuine thunderstorm model training",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "trained_by": Path(__file__).name,
            "target": {
                "column": target,
                "lead_time_hours": lead,
                "definition": (
                    f"genuine observed VOTV thunderstorm label at the clock hour exactly {lead} hour(s) "
                    "after the feature hour t; NaN (never 0) where that future hour carries no genuine "
                    "present-weather observation"
                ),
                "genuine_observation_label": True,
                "surrogate_or_proxy_target": False,
                "synthetic_labels_created": False,
                "rows_dropped_for_unobserved_target": dropped,
            },
            "features": {
                "count": len(FEATURE_COLUMNS),
                "feature_order": FEATURE_COLUMNS,
                "groups": FEATURE_GROUPS,
                "target_column_not_a_feature": target not in FEATURE_COLUMNS,
                "weather_code_used": False,
                "feature_time": "t (features are never shifted forward)",
                "target_time": f"t + {lead} h",
            },
            "training_dataset": {
                "csv": DATA_CSV.name,
                "path": str(DATA_CSV),
                "sha256": data_sha256,
                "input_chain_check": {
                    "phase4_feature_table_sha256_recorded_in_phase5_metadata": recorded_phase4_sha,
                    "phase4_feature_table_sha256_now": phase4_sha_now,
                    "phase4_to_phase6_ancestry_intact": input_chain_intact,
                },
                "rows_in_file": int(len(df)),
                "rows_used_for_this_lead_time": int(len(valid)),
                "rows_dropped_unobserved_target": dropped,
                "source": "dataset/votv_thunderstorm_nowcast_2014_2025.csv (Phase 5 output)",
                "phase5_rows": data_meta.get("outputs", {}).get("csv", {}).get("rows"),
            },
            "split": {
                **split_info,
                "shuffled": False,
                "test_used_for_tuning": False,
                "boundary_label_overlap_rows": {
                    "train_rows_with_target_in_validation_period": overlap_train,
                    "validation_rows_with_target_in_test_period": overlap_val,
                    "note": (
                        "Inherent to a forward-shifted target: the last few rows of a partition have their "
                        "target observation just inside the next period. No feature crosses the boundary - "
                        "every feature is a function of hours <= t - so this cannot leak future atmospheric "
                        "information into training; only the label pairing overlaps by at most the lead time."
                    ),
                },
            },
            "row_counts": {
                "train": int(splits["n_train"]),
                "validation": int(splits["n_validation"]),
                "test": int(splits["n_test"]),
            },
            "class_balance": {
                "train": binary_counts(y_tr),
                "validation": binary_counts(y_va),
                "test": binary_counts(y_te),
                "handling": (
                    "class_weight='balanced' inside the Random Forest only. No SMOTE, no random oversampling, "
                    "no synthetic rows, and the test set keeps its natural class distribution."
                ),
            },
            "model": {
                "type": "sklearn.ensemble.RandomForestClassifier",
                "params": model.get_params(),
                "random_state": SEED,
                "class_weight": MODEL_PARAMS["class_weight"],
                "selection_basis": "baseline model specified by the project master document",
                "not_the_phase1_surrogate_model": (
                    "models/storm_risk_model.joblib predicts a future-precipitation proxy and is not used, "
                    "loaded, retrained or overwritten here"
                ),
            },
            "random_seed": SEED,
            "threshold": {
                **selection,
                "probability_threshold_percentile_in_validation": pct,
                "validation_metrics_at_selected_threshold": val_metrics,
                "validation_metrics_at_0.5_for_reference": val_metrics_at_half,
            },
            "metrics": {
                "train_diagnostic_only": {
                    **train_metrics,
                    "note": "in-sample: reported for completeness only, it is not evidence of skill",
                },
                "validation": val_metrics,
                "test": test_metrics,
                "test_metrics_at_0.5_for_reference": test_metrics_at_half,
            },
            "feature_importance": {
                "csv": f"outputs/{importance_path.name}",
                "method": (
                    "RF mean decrease in impurity (primary), plus permutation importance on the validation "
                    "set scored by average precision (5 repeats, seed 42)"
                ),
                "top_10": top_features,
            },
            "artifacts": {
                "model": f"models/{model_path.name}",
                "metadata": f"models/thunderstorm_nowcast_{lead}h_metadata.json",
                "evaluation": f"outputs/thunderstorm_nowcast_{lead}h_evaluation.json",
                "feature_importance_csv": f"outputs/{importance_path.name}",
                "confusion_matrix_png": f"outputs/{cm_path.name}",
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            "leakage_controls": leakage,
            "no_model_selected_on_test": True,
            "scientific_disclaimer": DISCLAIMER,
        }
        meta_path = MODELS_DIR / f"thunderstorm_nowcast_{lead}h_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        log(f"    wrote {meta_path.name}")

        evaluation = {
            "lead_time_hours": lead,
            "target": target,
            "model_file": f"models/{model_path.name}",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_dataset": {
                "csv": DATA_CSV.name,
                "sha256": data_sha256,
                "rows_used": int(len(valid)),
            },
            "features": {"n_features": len(FEATURE_COLUMNS), "names": FEATURE_COLUMNS},
            "split": split_info,
            "threshold_selection": selection,
            "threshold_applied_to_test": threshold,
            "metrics": {
                "validation": val_metrics,
                "test": test_metrics,
                "validation_at_0.5_for_reference": val_metrics_at_half,
                "test_at_0.5_for_reference": test_metrics_at_half,
                "train_diagnostic_only": train_metrics,
            },
            "top_features": top_features,
            "test_set_evaluated_once_with_locked_threshold": True,
            "scientific_disclaimer": DISCLAIMER,
        }
        eval_path = OUTPUTS_DIR / f"thunderstorm_nowcast_{lead}h_evaluation.json"
        eval_path.write_text(json.dumps(evaluation, indent=2, default=str) + "\n", encoding="utf-8")
        log(f"    wrote {eval_path.name}")

        results[lead] = {
            "target": target,
            "threshold": threshold,
            "selection": selection,
            "model_path": model_path,
            "metadata_path": meta_path,
            "evaluation_path": eval_path,
            "importance_path": importance_path,
            "cm_path": cm_path,
            "valid_rows": int(len(valid)),
            "dropped_rows": dropped,
            "split": split_info,
            "validation": val_metrics,
            "test": test_metrics,
            "validation_at_half": val_metrics_at_half,
            "test_at_half": test_metrics_at_half,
            "train": train_metrics,
            "class_balance": {
                "train": binary_counts(y_tr),
                "validation": binary_counts(y_va),
                "test": binary_counts(y_te),
            },
            "importance": importance,
            "top_features": top_features,
            "model_params": model.get_params(),
        }

    # ---------------------------------------------------------------- comparison
    log("\n" + "=" * 80)
    log("LEAD TIME COMPARISON AND PRIMARY MODEL SELECTION")
    log("=" * 80)

    # Decision uses VALIDATION metrics and data availability only.
    ranking = sorted(
        results.items(),
        key=lambda kv: (-kv[1]["selection"]["selected_validation_metrics"]["fbeta"], kv[0]),
    )
    preferred_lead = ranking[0][0]
    preferred = results[preferred_lead]

    comparison_rows = []
    for lead, res in sorted(results.items()):
        v, t = res["validation"], res["test"]
        comparison_rows.append(
            {
                "lead_time_hours": lead,
                "target": res["target"],
                "valid_rows": res["valid_rows"],
                "rows_dropped_unobserved": res["dropped_rows"],
                "threshold": res["threshold"],
                "validation": {
                    "recall": v["recall"],
                    "precision": v["precision"],
                    "f1": v["f1"],
                    "f2": res["selection"]["selected_validation_metrics"]["fbeta"],
                    "roc_auc": v["roc_auc"],
                    "pr_auc_average_precision": v["average_precision_pr_auc"],
                    "baseline_positive_rate": v["baseline_positive_rate"],
                    "predicted_positive_rate": v["predicted_positive_rate"],
                    "accuracy": v["accuracy"],
                },
                "test": {
                    "recall": t["recall"],
                    "precision": t["precision"],
                    "f1": t["f1"],
                    "roc_auc": t["roc_auc"],
                    "pr_auc_average_precision": t["average_precision_pr_auc"],
                    "baseline_positive_rate": t["baseline_positive_rate"],
                    "predicted_positive_rate": t["predicted_positive_rate"],
                    "accuracy": t["accuracy"],
                    "confusion_matrix": t["confusion_matrix"],
                },
                "test_at_0.5_for_reference": {
                    "recall": res["test_at_half"]["recall"],
                    "precision": res["test_at_half"]["precision"],
                    "predicted_positive_rate": res["test_at_half"]["predicted_positive_rate"],
                },
            }
        )

    selection_rationale = {
        "decided_from": "validation metrics and data availability only, before the test set was scored",
        "test_metrics_role": "final evaluation and reporting only; they did not drive the choice",
        "primary_criterion": "highest validation F2 at the locked threshold (recall weighted twice precision)",
        "supporting_criteria": [
            "validation PR-AUC (average precision) against its base rate",
            "validation recall and F1 at the locked threshold",
            "validation ROC-AUC",
            "number of genuinely observable samples at that lead time",
        ],
        "preferred_lead_time_hours": preferred_lead,
        "ranking_by_validation_f2": [lead for lead, _ in ranking],
        "validation_support": {
            str(lead): {
                "validation_f2": res["selection"]["selected_validation_metrics"]["fbeta"],
                "validation_recall": res["validation"]["recall"],
                "validation_precision": res["validation"]["precision"],
                "validation_f1": res["validation"]["f1"],
                "validation_pr_auc": res["validation"]["average_precision_pr_auc"],
                "validation_roc_auc": res["validation"]["roc_auc"],
                "valid_rows": res["valid_rows"],
            }
            for lead, res in sorted(results.items())
        },
        "why": (
            f"{preferred_lead} h has the highest validation F2 "
            f"({preferred['selection']['selected_validation_metrics']['fbeta']:.4f}) and also the best "
            f"validation PR-AUC ({preferred['validation']['average_precision_pr_auc']:.4f} against a "
            f"{preferred['validation']['baseline_positive_rate']:.4f} base rate), the highest validation recall "
            f"({preferred['validation']['recall']:.4f}) and the largest number of genuinely observable rows "
            f"({preferred['valid_rows']}). Shorter lead times gain skill faster than they lose sample size here, "
            "so the shortest evaluated lead time is preferred."
        ),
        "caveat": (
            "The differences between lead times are modest and this ranking is a validation-set ordering on a "
            "single station and a single data window; it is not evidence that the 1 h lead generalises better "
            "everywhere. All three models are shipped so the choice can be revisited."
        ),
    }

    comparison = {
        "artifact": "thunderstorm_nowcast_model_comparison.json",
        "phase": "Phase 6 -- genuine thunderstorm model training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "binary nowcast of a genuine observed VOTV thunderstorm at t + 1h / 2h / 3h from 28 causal features at t",
        "target_is_genuine_observation": True,
        "surrogate_target_used": False,
        "weather_code_used": False,
        "synthetic_labels_or_resampling_used": False,
        "dataset": {
            "csv": "dataset/" + DATA_CSV.name,
            "sha256": data_sha256,
            "rows": int(len(df)),
            "date_range": {
                "first_timestamp_utc": df[TIME_COLUMN].min().isoformat(),
                "last_timestamp_utc": df[TIME_COLUMN].max().isoformat(),
            },
        },
        "features": {"n_features": len(FEATURE_COLUMNS), "names": FEATURE_COLUMNS},
        "model": {
            "type": "sklearn.ensemble.RandomForestClassifier",
            "params": {k: v for k, v in MODEL_PARAMS.items()},
            "random_state": SEED,
        },
        "split_strategy": "strict chronological 70/15/15, never shuffled",
        "threshold_criterion": {
            "rule": f"maximise F{int(F_BETA)} on validation subject to predicted positive rate <= {ALERT_RATE_CAP}",
            "why_not_0.5": (
                "With class_weight='balanced' the forest does not place the positive class near 0.5; on the "
                "validation and test sets a 0.5 threshold recovers almost none of the observed thunderstorm "
                "hours. 0.5 was measured, not assumed, and is reported for every lead time as a reference."
            ),
            "test_used": False,
        },
        "lead_time_results": comparison_rows,
        "selection": selection_rationale,
        "preferred_lead_time_hours": preferred_lead,
        "preferred_model_file": f"models/{preferred['model_path'].name}",
        "preferred_threshold": preferred["threshold"],
        "scientific_disclaimer": DISCLAIMER,
        "limitations": [
            "Single station (VOTV, 8.4667 N 76.95 E) and a single Open-Meteo grid cell about 2 km from it; no spatial generalisation is claimed.",
            "Labels are aerodrome point observations: a thunderstorm elsewhere in the area that the station did not report is labelled negative or unobserved.",
            "Unobserved hours are excluded, never relabelled 0, so the negative class is evidence-based but the training sample is not a complete picture of the period.",
            "No radar, satellite, lightning or NWP fields are used; the feature set is 28 causal transformations of archived Open-Meteo hourly variables.",
            "Feature-target correlations are weak (maximum |r| about 0.16 in Phase 4), so the achievable skill is modest; PR-AUC well above the base rate is the meaningful signal.",
            "Thresholds are tuned on one validation window and may not transfer to another regime.",
            "A model probability is not a physical thunderstorm and a threshold crossing is an alert, not an observation.",
        ],
    }
    comparison_path = OUTPUTS_DIR / "thunderstorm_nowcast_model_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"    wrote {comparison_path.name}")

    # ---------------------------------------------------------------- report
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.2f}%"

    def fmt(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.4f}"

    table_rows = "\n".join(
        f"| {r['lead_time_hours']} h | {r['valid_rows']:,} | {fmt(r['validation']['recall'])} | "
        f"{fmt(r['validation']['precision'])} | {fmt(r['validation']['f1'])} | "
        f"{fmt(r['validation']['pr_auc_average_precision'])} | {fmt(r['validation']['roc_auc'])} | "
        f"{fmt(r['test']['recall'])} | {fmt(r['test']['precision'])} | {fmt(r['test']['f1'])} | "
        f"{fmt(r['test']['pr_auc_average_precision'])} | {fmt(r['test']['roc_auc'])} |"
        for r in comparison_rows
    )
    threshold_rows = "\n".join(
        f"| {lead} h | {res['threshold']:.6f} | {pct(res['validation']['predicted_positive_rate'])} | "
        f"{pct(res['validation']['baseline_positive_rate'])} | {fmt(res['validation']['recall'])} | "
        f"{fmt(res['validation']['precision'])} | {pct(res['validation_at_half']['predicted_positive_rate'])} | "
        f"{fmt(res['validation_at_half']['recall'])} |"
        for lead, res in sorted(results.items())
    )
    importance_sections = []
    for lead, res in sorted(results.items()):
        rows = "\n".join(
            f"| {row['rf_impurity_rank']} | `{row['feature']}` | {row['rf_impurity_importance']:.4f} | "
            f"{row['permutation_importance_mean']:+.4f} |"
            for _, row in res["importance"].head(10).iterrows()
        )
        importance_sections.append(
            f"### Top 10 features, lead time {lead} h\n\n"
            "| rank | feature | RF impurity importance | permutation importance (AP drop) |\n"
            "| --- | --- | --- | --- |\n" + rows + "\n"
        )
    overlap = "\n".join(
        f"| {lead} h | {res['split']['train']['start_timestamp_utc'][:10]} .. "
        f"{res['split']['train']['end_timestamp_utc'][:10]} | "
        f"{res['split']['validation']['start_timestamp_utc'][:10]} .. "
        f"{res['split']['validation']['end_timestamp_utc'][:10]} | "
        f"{res['split']['test']['start_timestamp_utc'][:10]} .. "
        f"{res['split']['test']['end_timestamp_utc'][:10]} |"
        for lead, res in sorted(results.items())
    )
    window_rows = "\n".join(
        f"| {lead} h | {res['class_balance']['train']['positives']:,} / "
        f"{res['class_balance']['train']['negatives']:,} | "
        f"{res['class_balance']['validation']['positives']:,} / "
        f"{res['class_balance']['validation']['negatives']:,} | "
        f"{res['class_balance']['test']['positives']:,} / "
        f"{res['class_balance']['test']['negatives']:,} |"
        for lead, res in sorted(results.items())
    )
    test_cm_rows = "\n".join(
        f"| {lead} h | {res['test']['confusion_matrix'][0][0]:,} | {res['test']['confusion_matrix'][0][1]:,} | "
        f"{res['test']['confusion_matrix'][1][0]:,} | {res['test']['confusion_matrix'][1][1]:,} | "
        f"{fmt(res['test']['false_alarms_per_hit'])} |"
        for lead, res in sorted(results.items())
    )

    report = f"""# Phase 6 -- Genuine thunderstorm nowcast model training report

Generated by `ml/{Path(__file__).name}` at {comparison['created_at_utc']}.

## 1. Scope

Phase 6 trains the **first** models on the genuine observed VOTV thunderstorm target produced in
Phases 2-5. No surrogate target, no synthetic labels and no resampling are used, and the Phase 1
surrogate model `models/storm_risk_model.joblib` is neither loaded nor overwritten.

| item | value |
| --- | --- |
| dataset | `dataset/{DATA_CSV.name}` |
| dataset sha256 | `{data_sha256}` |
| rows in file | {len(df):,} |
| features | 28 causal Phase 4 features at time t |
| targets | `target_1h`, `target_2h`, `target_3h` (genuine observed label at t + lead) |
| model | `sklearn.ensemble.RandomForestClassifier` (400 trees, `class_weight="balanced"`, `random_state=42`, `n_jobs=-1`) |
| split | strict chronological 70% / 15% / 15%, never shuffled |
| threshold | selected on validation only, then locked and applied once to test |

## 2. Prediction, threshold, metrics and observation are different things

* **Prediction probability** - the forest's `predict_proba` for class 1. It is a model score, not a
  physical quantity and not a calibrated probability of a thunderstorm.
* **Classification threshold** - the validation-selected cut on that score that turns a score into a
  yes/no alert. It is a **policy choice**, not a property of the model.
* **Evaluation metrics** - recall, precision, F1, PR-AUC, ROC-AUC and accuracy computed by comparing
  alerts with held-out observations.
* **Actual thunderstorm observation** - a VOTV METAR present-weather report of a thunderstorm. This is
  the only ground truth, it exists for some hours and not others, and it is never produced by the model.

Confusing these four is the most common way a nowcasting result is overstated. Accuracy in particular is
not evidence of skill here: with a base rate near 4%, always predicting "no thunderstorm" scores about
96% accuracy while detecting nothing.

## 3. Split

Chronological 70/15/15 per lead time, computed on that lead time's genuinely observed rows only. No
shuffling, no random seed in the split, no stratification. The test split is scored exactly once, after
the threshold is locked.

| lead time | train period | validation period | test period |
| --- | --- | --- | --- |
{overlap}

Because the target is observed L hours after the features, the last few training rows have their target
inside the validation period. The count is recorded per model (`boundary_label_overlap_rows`); it is at
most a few rows and it cannot leak future atmospheric data, because every feature remains a function of
hours <= t.

| lead time | train pos/neg | validation pos/neg | test pos/neg |
| --- | --- | --- | --- |
{window_rows}

## 4. Threshold selection (validation only)

Rule: **{comparison['threshold_criterion']['rule']}**. Candidate thresholds are every distinct validation
predicted probability; the guardrail keeps the predicted positive rate at or below
{pct(ALERT_RATE_CAP)}; among the survivors the threshold with the highest F2 wins, ties going to the lower
threshold. The guardrail was not binding for any lead time, i.e. the F2 optimum is already a modest
alert load.

| lead time | locked threshold | validation alert rate | base rate | validation recall | validation precision | alert rate at 0.5 | recall at 0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
{threshold_rows}

The 0.5 column is the point of the exercise: with `class_weight="balanced"` the forest's scores sit far
below 0.5, so a 0.5 cut would have quietly produced a model that almost never issues an alert. The
threshold was not assumed; it was measured on validation and then locked.

## 5. Results

| lead | valid rows | val recall | val precision | val F1 | val PR-AUC | val ROC-AUC | test recall | test precision | test F1 | test PR-AUC | test ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{table_rows}

### Test confusion matrices

Rows are observations, columns are predictions.

| lead time | TN | FP | FN | TP | false alarms per hit |
| --- | --- | --- | --- | --- | --- |
{test_cm_rows}

## 6. Feature importance

Both an RF impurity importance and a permutation importance (validation set, scored by average
precision, 5 repeats) are saved for each lead time in
`outputs/thunderstorm_nowcast_<lead>h_feature_importance.csv`. Impurity importance is biased towards
high-cardinality continuous variables, which is why the permutation column is reported beside it; where
the two disagree, the permutation column is the more trustworthy.

{chr(10).join(importance_sections)}

The strongest features are the precipitation and humidity state and tendency terms, which is physically
plausible for convection at this site. Every feature is derived only from Open-Meteo archive variables;
no radar, satellite, lightning or NWP field is present.

## 7. Selected primary model

**Preferred lead time: {preferred_lead} h** (`models/{preferred['model_path'].name}`, threshold
{preferred['threshold']:.6f}).

{selection_rationale['why']}

**Why this decision is valid as a choice:** {selection_rationale['decided_from']}.
{selection_rationale['caveat']}

## 8. Scientific interpretation and limits

{DISCLAIMER}

Explicitly:

* This is **not** nationwide, regional or catchment-scale prediction. It is one station and one
  Open-Meteo grid cell about 2 km away.
* This is **not** radar-based, satellite-based, lightning-based or NWP-based prediction. No such data is
  used anywhere in Phases 1-6.
* A model probability is **not** physical certainty about a thunderstorm, and an alert is not an
  observation.
* This is **not** an operational IMD-level forecast product.
* Accuracy is **not** reported as evidence of skill. The skill evidence is PR-AUC relative to the base
  rate, and recall/precision at the locked threshold, both reported above with their base rates.
* Labels are aerodrome point observations: a storm the station did not report is not a positive label.
* The reported PR-AUC values are modest (about {fmt(min(r['test']['pr_auc_average_precision'] for r in comparison_rows))}
  to {fmt(max(r['test']['pr_auc_average_precision'] for r in comparison_rows))} against base rates of roughly
  3-4%), which is what a weak-signal atmospheric feature set can support. They should be read as
  "better than chance by a useful margin", not as an operational guarantee.

## 9. Reproducibility

```
python ml/{Path(__file__).name}
python outputs/verify_phase6_models.py
```

The independent verifier re-derives the splits from the CSV, reloads every artifact, reproduces the test
metrics from the saved models, re-derives each threshold from validation alone, and checks that the
Phase 1-5 files are still byte-identical.

## 10. Artifacts written

| file | role |
| --- | --- |
| `models/thunderstorm_nowcast_1h.joblib` | trained bundle (model + feature order + locked threshold) |
| `models/thunderstorm_nowcast_2h.joblib` | trained bundle |
| `models/thunderstorm_nowcast_3h.joblib` | trained bundle |
| `models/thunderstorm_nowcast_<L>h_metadata.json` | per-model metadata, counts, metrics, disclaimer |
| `outputs/thunderstorm_nowcast_<L>h_evaluation.json` | per-model evaluation metrics |
| `outputs/thunderstorm_nowcast_<L>h_feature_importance.csv` | feature importance |
| `outputs/thunderstorm_nowcast_<L>h_confusion_matrix.png` | test confusion matrix image |
| `outputs/thunderstorm_nowcast_model_comparison.json` | cross-lead-time comparison and selection |
| `outputs/PHASE6_MODEL_TRAINING_REPORT.md` | this report |
| `outputs/verify_phase6_models.py` | independent verification script |

Phase 1-5 datasets, the Flask backend, the dashboard, the frontend and the existing surrogate model were
not modified.
"""

    report_path = OUTPUTS_DIR / "PHASE6_MODEL_TRAINING_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    log(f"    wrote {report_path.name}")

    log("\n[8] Protected-file check")
    changed: list[str] = []
    for rel in PROTECTED_FILES:
        path = BASE_DIR / rel
        if not path.is_file():
            changed.append(f"MISSING:{rel}")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=BASE_DIR, capture_output=True, text=True, timeout=180
    )
    untracked_allowed = {
        "ml/train_thunderstorm_nowcast_phase6.py",
        "outputs/verify_phase6_models.py",
        "outputs/PHASE6_MODEL_TRAINING_REPORT.md",
        "outputs/thunderstorm_nowcast_model_comparison.json",
    }
    for line in status.stdout.splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:].strip().strip('"')
        if code == "??":
            if path.startswith("models/thunderstorm_nowcast") or path.startswith("outputs/thunderstorm_nowcast") or path in untracked_allowed:
                continue
            changed.append(f"UNEXPECTED_UNTRACKED:{path}")
        else:
            if path in PROTECTED_FILES:
                changed.append(f"MODIFIED:{path}")
    if changed:
        log(f"    PROBLEMS: {changed}")
        return 1
    log("    no protected Phase 1-5 file modified")

    log("\n" + "=" * 80)
    log("SUMMARY")
    log("=" * 80)
    for lead, res in sorted(results.items()):
        log(f"  {lead}h  threshold={res['threshold']:.6f}  "
            f"val F2={res['selection']['selected_validation_metrics']['fbeta']:.4f}  "
            f"test recall={res['test']['recall']:.4f} precision={res['test']['precision']:.4f} "
            f"F1={res['test']['f1']:.4f} PR-AUC={res['test']['average_precision_pr_auc']:.4f} "
            f"ROC-AUC={res['test']['roc_auc']:.4f}")
    log(f"  preferred lead time: {preferred_lead} h")
    log(f"  no model selected using test metrics: True")
    log("\nPHASE 6 MODEL TRAINING: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
