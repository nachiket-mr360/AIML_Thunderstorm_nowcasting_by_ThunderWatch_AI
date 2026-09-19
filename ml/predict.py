"""Inference API for the Phase 1 convective-risk surrogate prototype.

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

What the output means
---------------------
``risk_label`` is one of ``"Low Risk"`` / ``"Elevated Risk"`` for the surrogate
*convective / high-impact precipitation risk* over the next 3 hours. It is NOT,
and must never be relabelled as, a confirmed thunderstorm or lightning
occurrence.

Usage
-----
>>> from ml.predict import predict_risk
>>> result = predict_risk(observation, history=recent_hourly_rows)
>>> result["risk_label"]
'Elevated Risk'

``observation`` is one current/latest row (a mapping, ``Series``, or one-row
``DataFrame``) holding the raw observed fields the model needs plus ``date``.
``history`` holds the immediately preceding hourly rows (oldest first) so the
lag/rolling features can be reconstructed exactly as during training.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

try:  # executed as part of the ``ml`` package
    from .train_model import (
        DISCLAIMER,
        MIN_HISTORY_ROWS,
        RAW_COLUMNS,
        RISK_LABELS,
        build_features,
        build_future_precipitation,  # noqa: F401  (re-exported for callers)
        resolve_dataset_path,
    )
except ImportError:  # executed as a plain script: ``python ml/predict.py``
    from train_model import (  # type: ignore[no-redef]
        DISCLAIMER,
        MIN_HISTORY_ROWS,
        RAW_COLUMNS,
        RISK_LABELS,
        build_features,
        build_future_precipitation,  # noqa: F401
        resolve_dataset_path,
    )

__all__ = [
    "RISK_LABELS",
    "DISCLAIMER",
    "load_artifacts",
    "predict_risk",
    "predict_dataframe",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "storm_risk_model.joblib"
METADATA_FILENAME = "model_metadata.json"

#: Raw fields an observation must carry.
OBSERVATION_FIELDS = [column for column in RAW_COLUMNS if column != "date"]


def load_artifacts(models_dir: str | Path | None = None) -> tuple[Any, dict[str, Any]]:
    """Load the trained model and its metadata from ``models/``.

    Raises
    ------
    FileNotFoundError
        If the artifacts are missing; run ``python ml/train_model.py`` first.
    """
    directory = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    model_path = directory / MODEL_FILENAME
    metadata_path = directory / METADATA_FILENAME

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. Run 'python ml/train_model.py' first."
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found at '{metadata_path}'. Run 'python ml/train_model.py' first."
        )

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def _as_single_row(observation: Any) -> pd.Series:
    """Normalise an observation mapping / Series / one-row frame into a Series."""
    if isinstance(observation, pd.DataFrame):
        if len(observation) != 1:
            raise ValueError(
                f"Expected exactly one observation row, received {len(observation)} rows."
            )
        return observation.iloc[0]
    if isinstance(observation, pd.Series):
        return observation
    if isinstance(observation, Mapping):
        return pd.Series(dict(observation))
    raise TypeError(
        "observation must be a mapping, a pandas Series, or a one-row DataFrame; "
        f"received {type(observation).__name__}."
    )


def _frames_from_history(history: Any) -> pd.DataFrame:
    """Normalise the history argument into a DataFrame."""
    if history is None:
        return pd.DataFrame()
    if isinstance(history, pd.Series):
        frame = history.to_frame().T
    elif isinstance(history, pd.DataFrame):
        frame = history.copy()
    elif isinstance(history, Iterable):
        frame = pd.DataFrame(list(history))
    else:
        raise TypeError(
            "history must be a DataFrame, a list of mappings, or None; "
            f"received {type(history).__name__}."
        )
    return frame.reset_index(drop=True)


def _coerce_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Could not parse observation 'date' value: {value!r}")
    return timestamp


def _build_feature_frame(observation: Any, history: Any, feature_names: list[str]) -> pd.DataFrame:
    """Reconstruct the model feature vector for one observation.

    Two paths are supported:

    * **Pre-computed features** -- the observation already carries every column
      in ``feature_names``; they are used as-is.
    * **Raw observations** -- the observation carries the raw meteorological
      fields and ``date``, and ``history`` supplies the preceding hourly rows.
      The exact training-time feature engineering is then replayed.
    """
    row = _as_single_row(observation)
    missing_features = [name for name in feature_names if name not in row.index]

    if not missing_features:
        return pd.DataFrame([[float(row[name]) for name in feature_names]], columns=feature_names)

    missing_raw = [field for field in OBSERVATION_FIELDS if field not in row.index]
    if missing_raw:
        raise ValueError(
            "Observation is missing field(s) required to build features: "
            f"{missing_raw}. Provide either all pre-computed feature columns "
            f"({len(feature_names)} of them) or the raw fields {OBSERVATION_FIELDS}."
        )
    if "date" not in row.index:
        raise ValueError(
            "Observation is missing 'date'; it is required for the cyclical hour/month features."
        )

    history_frame = _frames_from_history(history)
    if len(history_frame) < MIN_HISTORY_ROWS:
        raise ValueError(
            f"Need at least {MIN_HISTORY_ROWS} preceding hourly observation(s) to compute the "
            f"lag/rolling features; received {len(history_frame)}. Pass the most recent hours "
            "as 'history' (oldest first)."
        )

    missing_history = [column for column in RAW_COLUMNS if column not in history_frame.columns]
    if missing_history:
        raise ValueError(f"History rows are missing required column(s): {missing_history}.")

    current = pd.DataFrame([{column: row[column] for column in RAW_COLUMNS}])
    combined = pd.concat(
        [history_frame[RAW_COLUMNS].copy(), current],
        ignore_index=True,
    )
    combined["date"] = pd.to_datetime(combined["date"], utc=True, errors="raise")
    combined = combined.sort_values("date", kind="mergesort").reset_index(drop=True)
    if combined["date"].duplicated().any():
        raise ValueError("History and observation contain duplicate timestamps.")
    if combined["date"].iloc[-1] != _coerce_timestamp(row["date"]):
        raise ValueError("Observation timestamp must be later than every history timestamp.")

    for column in RAW_COLUMNS[1:]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    if combined[RAW_COLUMNS[1:]].isna().any().any():
        raise ValueError("History/observation contain non-numeric or missing meteorological values.")

    engineered = build_features(combined)
    latest = engineered.iloc[[-1]].reset_index(drop=True)

    if latest.isna().any().any():
        bad = [name for name in feature_names if latest[name].isna().iloc[0]]
        raise ValueError(
            f"Feature(s) could not be computed from the supplied history: {bad}. "
            "Provide a longer contiguous hourly history."
        )

    return latest[feature_names].astype(float)


def predict_risk(
    observation: Any,
    history: Any = None,
    models_dir: str | Path | None = None,
    model: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one current/latest observation row.

    Parameters
    ----------
    observation:
        The current/latest observation. Either a mapping/``Series``/one-row
        ``DataFrame`` containing the raw observed fields plus ``date``, or one
        already containing every pre-computed feature column.
    history:
        The preceding hourly rows (oldest first) needed for the lag and rolling
        features. Not required when the observation already contains every
        feature column.
    models_dir:
        Directory holding the trained artifacts; defaults to ``<project>/models``.
    model, metadata:
        Pre-loaded artifacts, to avoid re-reading them per call. When omitted
        they are loaded from ``models_dir``.

    Returns
    -------
    dict
        ``probability`` -- P(Elevated Risk) from ``predict_proba``
        ``predicted_class`` -- 0 or 1
        ``risk_label`` -- ``"Low Risk"`` or ``"Elevated Risk"``
        plus ``decision_threshold_mm_per_3h``, ``target``, ``surrogate`` and
        ``disclaimer`` for traceability.
    """
    if model is None or metadata is None:
        model, loaded_metadata = load_artifacts(models_dir)
        metadata = metadata or loaded_metadata

    feature_names = list(metadata["feature_names"])
    feature_frame = _build_feature_frame(observation, history, feature_names)

    values = feature_frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Refusing to predict: feature vector contains NaN or infinite values.")

    probabilities = model.predict_proba(values)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Loaded model does not expose the positive class 1; classes={classes}")
    positive_index = classes.index(1)

    probability = float(probabilities[0, positive_index])
    predicted_class = int(classes[int(np.argmax(probabilities[0]))])

    return {
        "probability": probability,
        "predicted_class": predicted_class,
        "risk_label": RISK_LABELS[predicted_class],
        "decision_threshold_mm_per_3h": metadata.get("decision_threshold_mm_per_3h"),
        "target": metadata.get("target_name", "storm_risk_proxy"),
        "surrogate": True,
        "horizon_hours": metadata.get("horizon_hours"),
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
    }


def predict_dataframe(
    frame: pd.DataFrame,
    models_dir: str | Path | None = None,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Batch-score a frame that already contains every pre-computed feature column."""
    model, metadata = load_artifacts(models_dir)
    names = feature_names or list(metadata["feature_names"])

    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"Input frame is missing feature column(s): {missing}")

    values = frame[names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Refusing to predict: input frame contains NaN or infinite values.")

    probabilities = model.predict_proba(values)[:, list(model.classes_).index(1)]
    predicted = model.predict(values).astype(int)

    return pd.DataFrame(
        {
            "probability": probabilities,
            "predicted_class": predicted,
            "risk_label": [RISK_LABELS[int(label)] for label in predicted],
        },
        index=frame.index,
    )


def _demo(rows: int = 24) -> int:
    """Score the latest observation in the local dataset as a smoke test."""
    csv_path = resolve_dataset_path()
    frame = pd.read_csv(csv_path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.sort_values("date").reset_index(drop=True)

    if len(frame) < rows + 1:
        print("Not enough rows in the dataset for a demo.", file=sys.stderr)
        return 1

    history = frame.iloc[-(rows + 1):-1]
    observation = frame.iloc[-1]

    result = predict_risk(observation, history=history)
    print(f"Dataset        : {csv_path}")
    print(f"Observation at : {result['scored_at_utc']} (scored at)")
    print(f"Timestamp      : {observation['date'].isoformat()}")
    print(f"Probability    : {result['probability']:.4f}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Risk label     : {result['risk_label']}")
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
