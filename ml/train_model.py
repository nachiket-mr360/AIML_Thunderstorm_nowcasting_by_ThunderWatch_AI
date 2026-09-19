"""Phase 1 ML training pipeline: convective-risk surrogate nowcasting.

SIH 2026 prototype -- "AIML based Nowcasting of thunderstorm and lightning
using atmospheric observation including multiple radars, satellite, lightning
and model data."

SCIENTIFIC DISCLAIMER
---------------------
This prototype uses a future high-impact precipitation event as a surrogate
target because verified historical lightning/thunderstorm labels were not
available in the current dataset. It demonstrates the ML nowcasting pipeline
and should not be interpreted as an operational IMD thunderstorm or lightning
forecast.

Why a surrogate target is required
----------------------------------
The available Open-Meteo extract contains only WMO ``weather_code`` values
0, 1, 2, 3, 51, 53, 55, 61, 63 and 65. It contains NO thunderstorm codes
(95, 96, 99) and no lightning or radar observations. There is therefore no
genuine historical thunderstorm/lightning label to learn from, and inventing
one would be scientifically dishonest. Instead we derive a clearly named
surrogate: a *future high-impact precipitation / convective-risk proxy*.

Surrogate target definition
---------------------------
1. Sort observations chronologically by ``date``.
2. For every timestamp ``t`` compute the accumulated precipitation over the
   NEXT 3 HOURS:
       future_precip_3h = precipitation(t+1) + precipitation(t+2) + precipitation(t+3)
3. Compute the 90th percentile of ``future_precip_3h`` using ONLY the training
   portion of the data (prevents temporal leakage).
4. ``storm_risk_proxy = 1`` if ``future_precip_3h >= training_90th_percentile``
   else ``0``.
5. Drop rows where the next 3 hours are unavailable (end of record, or a gap in
   the hourly series).
6. ``future_precip_3h`` is NEVER used as a model feature.

Leakage guards
--------------
* ``weather_code`` is excluded from the features entirely: it encodes the
  present-weather event definition and would leak the target.
* Every feature is computable at prediction time ``t`` (current values, past
  lags, past rolling windows, cyclical clock/calendar encodings). No feature
  reads a future timestep.
* The decision threshold is fitted on the training window only.

Usage
-----
    python ml/train_model.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DISCLAIMER = (
    "This prototype uses a future high-impact precipitation event as a "
    "surrogate target because verified historical lightning/thunderstorm "
    "labels were not available in the current dataset. It demonstrates the ML "
    "nowcasting pipeline and should not be interpreted as an operational IMD "
    "thunderstorm or lightning forecast."
)

DATASET_FILENAME = "weather_data_with_code.csv"

#: Raw observation columns that must be present in the source CSV.
RAW_COLUMNS = [
    "date",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]

#: Columns that must exist in the CSV but are deliberately NOT used as features.
UNUSED_COLUMNS = ["weather_code"]

#: Columns the pipeline requires to exist in the source CSV.
REQUIRED_COLUMNS = RAW_COLUMNS + UNUSED_COLUMNS

#: Names that must never appear among the model features.
FORBIDDEN_FEATURE_NAMES = {
    "weather_code",  # leaks the event definition
    "future_precip_3h",  # the future outcome itself
    "storm_risk_proxy",  # the target itself
}

TARGET_NAME = "storm_risk_proxy"
FUTURE_PRECIP_NAME = "future_precip_3h"
HORIZON_HOURS = 3
QUANTILE = 0.90
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# TEST_FRACTION is the remainder (0.15).

RANDOM_STATE = 42

#: Number of past hourly rows needed to evaluate every lag/rolling feature.
#: ``diff(3)`` needs 3 prior hours; ``shift(1).rolling(6)`` needs 6 prior hours
#: (t-6 .. t-1), which is the binding constraint.
MIN_HISTORY_ROWS = 6

RISK_LABELS = {0: "Low Risk", 1: "Elevated Risk"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# --------------------------------------------------------------------------
# Dataset location and loading
# --------------------------------------------------------------------------


def resolve_dataset_path(filename: str = DATASET_FILENAME) -> Path:
    """Locate the observation CSV robustly, independent of the cwd.

    Search order:
      1. ``SIH2_WEATHER_CSV`` environment variable (explicit override).
      2. ``<project root>/dataset/<filename>``.
      3. ``<cwd>/dataset/<filename>``.
      4. ``<project root>/<filename>``, ``<project root>/../dataset/<filename>``.
      5. A bounded recursive search from the project root and the cwd.

    Raises
    ------
    FileNotFoundError
        If no candidate exists. The searched locations are listed in the
        message so the failure is actionable.
    """
    searched: list[Path] = []
    candidates: list[Path] = []

    env_override = os.environ.get("SIH2_WEATHER_CSV")
    if env_override:
        candidates.append(Path(env_override).expanduser())

    anchor_dirs = [PROJECT_ROOT, Path.cwd().resolve()]
    for anchor in anchor_dirs:
        candidates.append(anchor / "dataset" / filename)
        candidates.append(anchor / filename)
        candidates.append(anchor.parent / "dataset" / filename)

    for candidate in candidates:
        searched.append(candidate)
        if candidate.is_file():
            return candidate.resolve()

    # Bounded recursive fallback (project root then cwd).
    for anchor in anchor_dirs:
        if not anchor.is_dir():
            continue
        try:
            for match in sorted(anchor.glob(f"**/{filename}")):
                searched.append(match)
                if match.is_file():
                    return match.resolve()
        except OSError:
            continue

    listing = "\n".join(f"  - {p}" for p in searched)
    raise FileNotFoundError(
        f"Could not locate '{filename}'. Searched:\n{listing}\n"
        "Set the SIH2_WEATHER_CSV environment variable to the CSV path to override."
    )


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise a clear error if any required column is missing."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required column(s): "
            f"{missing}. Present columns: {list(df.columns)}"
        )


def load_observations(csv_path: Path) -> pd.DataFrame:
    """Load the CSV, parse/validate ``date`` and return it sorted chronologically."""
    df = pd.read_csv(csv_path)
    validate_columns(df, REQUIRED_COLUMNS)

    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    if df["date"].isna().any():
        bad = int(df["date"].isna().sum())
        raise ValueError(f"{bad} row(s) have an unparseable 'date' value.")

    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    if df["date"].duplicated().any():
        duplicates = int(df["date"].duplicated().sum())
        raise ValueError(f"Found {duplicates} duplicate timestamp(s); refusing to proceed.")

    # All meteorological fields must be numeric for feature engineering.
    for column in RAW_COLUMNS[1:]:
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def _json_default(value: Any) -> Any:
    """Make numpy scalars/arrays JSON-serializable (pandas-derived values)."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file (dataset provenance)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Feature engineering -- every feature is available at prediction time t
# --------------------------------------------------------------------------


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model feature matrix from raw observations.

    Parameters
    ----------
    df:
        Chronologically sorted observations containing ``RAW_COLUMNS``.

    Returns
    -------
    pandas.DataFrame
        Feature frame aligned to ``df.index``. Rows without enough history are
        left as NaN and are dropped by :func:`prepare_supervised_frame`.

    Notes
    -----
    Rolling windows are computed over the hours strictly *before* ``t``
    (``shift(1)`` then ``rolling(...)``) and reduced with ``sum`` for
    precipitation and ``mean`` for humidity/pressure.
    """
    features = pd.DataFrame(index=df.index)

    # --- instantaneous observations at time t -----------------------------
    features["temperature_2m"] = df["temperature_2m"].astype(float)
    features["relative_humidity_2m"] = df["relative_humidity_2m"].astype(float)
    features["surface_pressure"] = df["surface_pressure"].astype(float)
    features["wind_speed_10m"] = df["wind_speed_10m"].astype(float)
    features["precipitation"] = df["precipitation"].astype(float)
    features["cloud_cover"] = df["cloud_cover"].astype(float)

    # --- wind direction as cyclic sin/cos components ----------------------
    direction_rad = np.deg2rad(df["wind_direction_10m"].astype(float))
    features["wind_direction_sin"] = np.sin(direction_rad)
    features["wind_direction_cos"] = np.cos(direction_rad)

    # --- change over the previous 1h and 3h ------------------------------
    for column, stem in (
        ("temperature_2m", "temperature_2m"),
        ("relative_humidity_2m", "relative_humidity_2m"),
        ("surface_pressure", "surface_pressure"),
        ("wind_speed_10m", "wind_speed_10m"),
    ):
        series = df[column].astype(float)
        features[f"{stem}_change_1h"] = series.diff(1)
        features[f"{stem}_change_3h"] = series.diff(3)

    # --- rolling aggregates over previous hours (strictly before t) -------
    features["precip_roll_3h"] = df["precipitation"].astype(float).shift(1).rolling(3).sum()
    features["precip_roll_6h"] = df["precipitation"].astype(float).shift(1).rolling(6).sum()
    features["humidity_roll_3h"] = (
        df["relative_humidity_2m"].astype(float).shift(1).rolling(3).mean()
    )
    features["pressure_roll_3h"] = (
        df["surface_pressure"].astype(float).shift(1).rolling(3).mean()
    )

    # --- cyclical time encodings -----------------------------------------
    hour = df["date"].dt.hour.astype(float)
    month = df["date"].dt.month.astype(float)
    features["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    features["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    features["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)

    return features


def build_future_precipitation(df: pd.DataFrame, horizon: int = HORIZON_HOURS) -> pd.Series:
    """Accumulated precipitation over the next ``horizon`` hours.

    Rows where the future window is unavailable (end of record, or a missing /
    non-hourly-spaced observation inside the window) are returned as ``NaN``.
    """
    precipitation = df["precipitation"].astype(float)
    date = df["date"]

    total = pd.Series(0.0, index=df.index)
    complete = pd.Series(True, index=df.index)

    for step in range(1, horizon + 1):
        future_value = precipitation.shift(-step)
        # Require an actual observation and an exactly-step-hour offset so that
        # "next 3 hours" really means three consecutive hourly timestamps.
        expected_gap = pd.Timedelta(hours=step)
        future_date = date.shift(-step)
        is_contiguous = (future_date - date) == expected_gap

        total = total + future_value.fillna(0.0)
        complete = complete & future_value.notna() & is_contiguous

    return total.where(complete, np.nan)


def prepare_supervised_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return features, target (unthresholded future precipitation) and validity.

    Returns
    -------
    (features, future_precip_3h, dropped_rows)
        ``features`` and ``future_precip_3h`` are restricted to rows where both
        the feature vector and the future window are fully available. The third
        value is a small accounting frame describing which rows were removed.
    """
    features = build_features(df)
    future_precip = build_future_precipitation(df)

    feature_ok = features.notna().all(axis=1) & np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    future_ok = future_precip.notna()

    keep = feature_ok & future_ok

    dropped = pd.DataFrame(
        {
            "total_rows": [len(df)],
            "dropped_missing_future_window": [int((~future_ok).sum())],
            "dropped_insufficient_feature_history": [int((future_ok & ~feature_ok).sum())],
            "dropped_total": [int((~keep).sum())],
            "usable_rows": [int(keep.sum())],
            "first_usable_timestamp": [
                None if not keep.any() else df.loc[keep, "date"].iloc[0].isoformat()
            ],
            "last_usable_timestamp": [
                None if not keep.any() else df.loc[keep, "date"].iloc[-1].isoformat()
            ],
        }
    )

    return features.loc[keep].reset_index(drop=True), future_precip.loc[keep].reset_index(drop=True), dropped


# --------------------------------------------------------------------------
# Target, split and evaluation
# --------------------------------------------------------------------------


def chronological_split_indices(n_rows: int) -> dict[str, np.ndarray]:
    """Index arrays for the 70% / 15% / 15% chronological split."""
    n_train = int(n_rows * TRAIN_FRACTION)
    n_val = int(n_rows * VAL_FRACTION)
    return {
        "train": np.arange(0, n_train),
        "validation": np.arange(n_train, n_train + n_val),
        "test": np.arange(n_train + n_val, n_rows),
    }


def evaluate_split(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Compute classification metrics for one split from real predictions."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y).astype(int)
    y_pred = model.predict(X).astype(int)

    class_counts = {str(label): int((y_true == label).sum()) for label in (0, 1)}
    both_classes_present = class_counts["0"] > 0 and class_counts["1"] > 0

    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "class_distribution": {
            "count_0_low_risk": class_counts["0"],
            "count_1_elevated_risk": class_counts["1"],
            "fraction_1_elevated_risk": (
                float((y_true == 1).mean()) if len(y_true) else None
            ),
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # ``pos_label=1`` keeps precision/recall/F1 anchored to the event class.
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": ["actual_0_low_risk", "actual_1_elevated_risk"],
        "confusion_matrix_axis_note": "rows = actual, columns = predicted, labels ordered [0, 1]",
        "roc_auc": None,
    }

    if both_classes_present and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["roc_auc_note"] = "ROC-AUC not defined: only one class present in this split."

    return metrics


def save_confusion_matrix_plot(cm: list[list[int]], path: Path, title: str) -> None:
    """Render the test-set confusion matrix to a PNG file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(cm, dtype=int)
    labels = [RISK_LABELS[0], RISK_LABELS[1]]

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)

    threshold_value = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            ax.text(
                column,
                row,
                f"{value}",
                ha="center",
                va="center",
                color="white" if value > threshold_value else "black",
                fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Training entry point
# --------------------------------------------------------------------------


def train(
    csv_path: Path | None = None,
    models_dir: Path | None = None,
    outputs_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full pipeline and return the evaluation report."""
    from sklearn.ensemble import RandomForestClassifier

    models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
    outputs_dir = Path(outputs_dir or DEFAULT_OUTPUTS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(csv_path) if csv_path else resolve_dataset_path()

    print("=" * 74)
    print("SIH 2026 -- convective-risk surrogate nowcasting (Phase 1 training)")
    print("=" * 74)
    print(DISCLAIMER)
    print()

    # 1-4. Load, validate, parse dates, sort chronologically.
    df = load_observations(csv_path)
    print(f"Dataset        : {csv_path}")
    print(f"Raw shape      : {df.shape[0]} rows x {df.shape[1]} columns")
    print(
        "Period         : "
        f"{df['date'].iloc[0].isoformat()} -> {df['date'].iloc[-1].isoformat()}"
    )
    observed_codes = sorted(int(code) for code in df["weather_code"].dropna().unique())
    thunderstorm_codes = sorted(set(observed_codes) & {95, 96, 99})
    print(f"weather_code   : {observed_codes} (excluded from features)")
    print(
        "Thunderstorm codes 95/96/99 present : "
        f"{thunderstorm_codes if thunderstorm_codes else 'NONE -> surrogate target required'}"
    )
    print()

    # 6-7. Features and the future 3-hour precipitation series.
    features, future_precip, dropped = prepare_supervised_frame(df)

    feature_names = list(features.columns)
    leaked = FORBIDDEN_FEATURE_NAMES.intersection(feature_names)
    if leaked:
        raise AssertionError(f"Forbidden feature(s) leaked into the model input: {sorted(leaked)}")

    print("Row accounting")
    for column in dropped.columns:
        print(f"  {column:<38}: {dropped[column].iloc[0]}")
    print(f"  feature_count                         : {len(feature_names)}")
    print()

    # 8. Chronological split -- never random.
    n_rows = len(features)
    if n_rows < 100:
        raise ValueError(f"Only {n_rows} usable rows; too few to train a meaningful model.")

    splits = chronological_split_indices(n_rows)
    X = features.to_numpy(dtype=float)

    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).sum())
        raise ValueError(f"{bad} non-finite value(s) reached the model matrix.")

    # 3. Threshold from the TRAINING portion only (no temporal leakage).
    future_values = future_precip.to_numpy(dtype=float)
    train_idx, val_idx, test_idx = splits["train"], splits["validation"], splits["test"]
    threshold = float(np.quantile(future_values[train_idx], QUANTILE))

    # 4. Surrogate target.
    target = (future_values >= threshold).astype(int)
    y = pd.Series(target, name=TARGET_NAME)

    # Recover the timestamps that survived the cleaning step for reporting.
    feature_ok = build_features(df).notna().all(axis=1)
    future_series = build_future_precipitation(df)
    keep_mask = feature_ok & future_series.notna()
    kept_dates = df.loc[keep_mask, "date"].reset_index(drop=True)
    if len(kept_dates) != n_rows:
        raise AssertionError("Internal error: usable-row bookkeeping is inconsistent.")

    def split_bounds(idx: np.ndarray) -> dict[str, Any]:
        return {
            "n_rows": int(len(idx)),
            "start_index": int(idx[0]),
            "end_index": int(idx[-1]),
            "start_timestamp": kept_dates.iloc[idx[0]].isoformat(),
            "end_timestamp": kept_dates.iloc[idx[-1]].isoformat(),
        }

    split_info = {name: split_bounds(idx) for name, idx in splits.items()}

    # Hard guarantee: test is strictly after validation, which is strictly
    # after training.
    ordered = (
        kept_dates.iloc[train_idx[-1]] < kept_dates.iloc[val_idx[0]]
        and kept_dates.iloc[val_idx[-1]] < kept_dates.iloc[test_idx[0]]
    )
    if not ordered:
        raise AssertionError("Chronological split violated: splits are not time-ordered.")

    print("Chronological split (70 / 15 / 15, no shuffling)")
    for name in ("train", "validation", "test"):
        info = split_info[name]
        positives = int(y.iloc[splits[name]].sum())
        share = positives / info["n_rows"] if info["n_rows"] else 0.0
        print(
            f"  {name:<11} {info['n_rows']:>7} rows  "
            f"{info['start_timestamp'][:19]} -> {info['end_timestamp'][:19]}  "
            f"positives={positives} ({share:.3%})"
        )
    print()

    print("Surrogate target")
    print(f"  definition : {FUTURE_PRECIP_NAME} = precipitation(t+1)+precipitation(t+2)+precipitation(t+3)")
    print(f"  threshold  : train-only {QUANTILE:.0%} percentile = {threshold:.4f} mm / 3h")
    print(f"  train positives : {int(y.iloc[train_idx].sum())}")
    if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
        raise ValueError(
            "Target has a single class in train and/or test; refusing to report degenerate metrics."
        )
    print()

    # 9. Train the RandomForest.
    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X[train_idx], y.iloc[train_idx].to_numpy())
    print(f"Trained RandomForestClassifier with {model.n_estimators} trees")
    print()

    # 10. Evaluate.
    metrics = {
        "train": evaluate_split(model, X[train_idx], y.iloc[train_idx].to_numpy()),
        "validation": evaluate_split(model, X[val_idx], y.iloc[val_idx].to_numpy()),
        "test": evaluate_split(model, X[test_idx], y.iloc[test_idx].to_numpy()),
    }
    test_metrics = metrics["test"]

    print("Test-set metrics (untouched hold-out, chronological)")
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  F1-score  : {test_metrics['f1_score']:.4f}")
    roc_auc = test_metrics["roc_auc"]
    print(f"  ROC-AUC   : {roc_auc:.4f}" if roc_auc is not None else "  ROC-AUC   : n/a")
    print(f"  Class distribution (actual): {test_metrics['class_distribution']}")
    print(f"  Confusion matrix [ [TN, FP], [FN, TP] ]: {test_metrics['confusion_matrix']}")
    print()

    # 11. Save the trained model.
    model_path = models_dir / "storm_risk_model.joblib"
    joblib.dump(model, model_path, compress=3)

    # 12-14. Save evaluation, confusion-matrix image and feature importance.
    import sklearn

    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_file": model_path.name,
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "task": "binary classification -- surrogate convective/high-impact precipitation risk",
        "target_name": TARGET_NAME,
        "target_definition": (
            f"{FUTURE_PRECIP_NAME}(t) = precipitation(t+1) + precipitation(t+2) + "
            f"precipitation(t+3); {TARGET_NAME} = 1 if {FUTURE_PRECIP_NAME}(t) >= "
            f"train-only {QUANTILE:.2f} percentile, else 0"
        ),
        "horizon_hours": HORIZON_HOURS,
        "decision_threshold_mm_per_3h": threshold,
        "threshold_quantile": QUANTILE,
        "threshold_fitted_on": "training split only",
        "positive_class_baseline_rate_in_train": float(y.iloc[train_idx].mean()),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "risk_labels": {str(key): value for key, value in RISK_LABELS.items()},
        "excluded_columns": {
            "weather_code": "present-weather code; excluded to avoid leaking the event definition",
            FUTURE_PRECIP_NAME: "future outcome; never a feature",
        },
        "min_history_rows_required_at_inference": MIN_HISTORY_ROWS,
        "raw_columns_expected_by_predictor": RAW_COLUMNS,
        "split": {
            "strategy": "chronological (no shuffling)",
            "fractions": {"train": TRAIN_FRACTION, "validation": VAL_FRACTION, "test": 0.15},
            **split_info,
        },
        "model_params": model.get_params(),
        "random_state": RANDOM_STATE,
        "training_data": {
            "csv": csv_path.name,
            "path": str(csv_path),
            "sha256": sha256_of_file(csv_path),
            "raw_rows": int(len(df)),
            "usable_rows": int(n_rows),
            "first_timestamp": kept_dates.iloc[0].isoformat(),
            "last_timestamp": kept_dates.iloc[-1].isoformat(),
            "row_accounting": {
                key: (None if pd.isna(dropped[key].iloc[0]) else dropped[key].iloc[0])
                for key in dropped.columns
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "created_at_utc": created_at,
        "disclaimer": DISCLAIMER,
    }

    metadata_path = models_dir / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
    )

    report = {
        "generated_at_utc": created_at,
        "disclaimer": DISCLAIMER,
        "dataset": metadata["training_data"],
        "target": {
            "name": TARGET_NAME,
            "surrogate": True,
            "definition": metadata["target_definition"],
            "horizon_hours": HORIZON_HOURS,
            "threshold_mm_per_3h": threshold,
            "threshold_quantile": QUANTILE,
            "threshold_fitted_on": "training split only",
        },
        "features": {
            "n_features": len(feature_names),
            "names": feature_names,
            "excluded_as_predictors": sorted(FORBIDDEN_FEATURE_NAMES),
        },
        "split": metadata["split"],
        "model": {
            "type": metadata["model_type"],
            "params": metadata["model_params"],
            "random_state": RANDOM_STATE,
        },
        "metrics": metrics,
        "test_set_is_untouched_holdout": True,
        "notes": [
            "All metrics are computed from real predictions on the held-out test split.",
            "The decision threshold uses the training split only, so no future information "
            "influences the target.",
            "ROC-AUC is reported only when both classes exist in the evaluated split.",
        ],
    }

    evaluation_path = outputs_dir / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )

    confusion_path = outputs_dir / "confusion_matrix.png"
    save_confusion_matrix_plot(
        test_metrics["confusion_matrix"],
        confusion_path,
        title="Test-set confusion matrix\n(convective-risk surrogate target)",
    )

    importance_path = outputs_dir / "feature_importance.csv"
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    importance.to_csv(importance_path, index=False)

    # 15. Concise summary.
    print("Artifacts")
    for path in (model_path, metadata_path, evaluation_path, confusion_path, importance_path):
        size_kb = path.stat().st_size / 1024.0
        print(f"  {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}  ({size_kb:.1f} KB)")
    print()
    print("Top 10 features by importance")
    for row in importance.head(10).itertuples(index=False):
        print(f"  {row.feature:<28} {row.importance:.4f}")
    print()
    print("Reminder: this is a surrogate convective-risk prototype, not an operational")
    print("IMD thunderstorm or lightning forecast.")
    print("=" * 74)

    return report


def main() -> int:
    try:
        train()
    except Exception as error:  # noqa: BLE001 - surface a clear CLI failure
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
