"""Phase 8 prediction backend: Flask API over the verified Phase 7 nowcast engine.

SIH 2026 (SIH26072) -- "AIML based nowcasting of thunderstorm and lightning
using atmospheric observation including multiple radars, satellite, lightning
and model data."

WHAT THIS APPLICATION SERVES
-----------------------------
The primary product is a genuine **1-hour thunderstorm nowcast** for the VOTV
station, produced end-to-end by the Phase 7 engine
``ml/predict_thunderstorm_nowcast.py``:

    latest available atmospheric hour t
        -> the 28 Phase 4 causal features
        -> probability that the station reports a thunderstorm in hour t + 1 h
        -> a yes/no alert against the locked Phase 6 threshold 0.0775

``GET /api/prediction`` returns that engine's payload. This module performs no
inference, no feature engineering, no thresholding and no data fetching of its
own -- see ``backend/nowcast_service.py``, which is a thin transport layer.

The Phase 2 application that used to occupy this file is retained, not
replaced. Its surrogate high-precipitation proxy endpoints are unchanged and
now live alongside the nowcast:

    GET /api/prediction/proxy   Phase 1 high-precipitation risk proxy (surrogate)
    GET /api/history            recent hourly atmospheric data (Phase 1 site)
    GET /api/evaluation         Phase 1 metrics + feature importance (read-only)
    GET /api/artifacts/<file>   whitelisted Phase 1 image artifact (read-only)
    GET /api/scenario           Phase 1 historical test-set demonstration

WHAT THE TWO MODELS MEAN (they are different things)
----------------------------------------------------
``/api/prediction`` -- a real thunderstorm nowcast. Its target ``target_1h`` is a
genuine VOTV METAR present-weather observation one hour ahead. Its probability
is a model score, NOT a confidence and NOT a calibrated thunderstorm frequency:
the forest was trained with ``class_weight='balanced'``, which shifts scores
towards the rare class on purpose. Only the ordering of scores and the locked
threshold have been validated.

``/api/prediction/proxy`` -- an *unrelated* surrogate. Its target is not a
thunderstorm and not lightning; it is ``1`` when accumulated precipitation over
the next three hours reaches a training-only 90th percentile. Its Phase 1 test
metrics (ROC-AUC 0.8897, F1 0.4863) describe that surrogate only and must never
be presented as thunderstorm or lightning detection accuracy, nor as an official
IMD warning.

SCIENTIFIC TERMS THIS API KEEPS STRAIGHT
----------------------------------------
* probability != confidence
* latest available atmospheric data != guaranteed direct observation (the seven
  inputs come from a model-derived provider grid cell, not the aerodrome
  ground observation)
* prediction != actual future thunderstorm observation (the observation for hour
  t + 1 h does not exist at prediction time, and the payload says so explicitly)

Data handling guarantees
------------------------
* No weather value and no prediction is ever invented, imputed, defaulted or
  carried over from a previous request.
* A failure never returns a probability. Both endpoints answer with a JSON error
  body whose prediction fields are explicitly ``null``.
* The primary 1-hour nowcast bundle is read once per process at startup (the
  artifact is ~144 MB) and is only ever read, never written.
* The legacy Phase 1 surrogate bundle is read **lazily**, on the first request
  that actually needs it (``/api/prediction/proxy``, ``/api/scenario``). Its
  artifact is ~130 MB on disk but ~440 MB once resident, so loading it at
  import would make one process hold both forests simultaneously and exceed a
  512 MiB hosting instance. See ``ensure_legacy_model`` for the deployment
  consequence.

Usage
-----
    python app.py
    # or
    flask --app app run

Environment variables (all optional):
    HOST                  bind address (default 127.0.0.1)
    PORT                  bind port (default 5000)
    FLASK_DEBUG           "1" to enable the debug reloader (default off)
    CORS_ALLOWED_ORIGINS  comma-separated origin allow-list, or "*" (default)
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------
# Phase 8: the primary product. The prediction engine is imported through the
# Phase 8 service layer, which owns the one-time model load and the
# engine-error -> HTTP translation. Nothing in ml/ is modified by this module.
# --------------------------------------------------------------------------
from backend import nowcast_service  # noqa: E402
from ml.predict_thunderstorm_nowcast import (  # noqa: E402
    DISCLAIMER as NOWCAST_DISCLAIMER,
    TRAINING_SERVED_GRID_CELL,
    ThunderstormNowcastError,
)
from ml.predict import (  # noqa: E402
    DISCLAIMER as SURROGATE_TARGET_DISCLAIMER,
    MIN_HISTORY_ROWS,
    RISK_LABELS,
    load_artifacts,
    predict_risk,
)

# Phase 1 training helpers, imported read-only so the historical scenario uses
# byte-identical feature engineering and the byte-identical chronological split.
from ml import train_model as phase1  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Origin allow-list for the read-only API. This is a local prototype with no
#: authentication, no cookies and no write endpoints, so "*" is a defensible
#: default; set CORS_ALLOWED_ORIGINS to a comma-separated list to restrict it.
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

#: Header value used for ``Access-Control-Allow-Origin``. ``*`` cannot be
#: combined with credentials, and this API never uses them, so the wildcard is
#: only ever emitted for the "allow everything" configuration.
CORS_ALLOW_ANY_ORIGIN = CORS_ALLOWED_ORIGINS == ["*"]

TARGET_LOCATION = {
    "location": "Thiruvananthapuram, Kerala",
    "latitude": 8.4855,
    "longitude": 76.9492,
}

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

#: The exact hourly variables the Phase 1 model consumes. These names match the
#: training data columns one-for-one, so no renaming is needed downstream.
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]

#: Recent history pulled from the API. Needs at least MIN_HISTORY_ROWS (6) valid
#: contiguous hours; 3 past days gives a large safety margin against nulls.
PAST_DAYS = 3
FORECAST_DAYS = 1
REQUEST_TIMEOUT_SECONDS = 20

#: Units the Phase 1 model was trained on. The training notebook called the
#: Open-Meteo archive API with no unit overrides, so the provider defaults apply.
#: Any deviation is treated as an error rather than silently rescaled input.
EXPECTED_UNITS = {
    "temperature_2m": "\u00b0C",
    "relative_humidity_2m": "%",
    "surface_pressure": "hPa",
    "wind_speed_10m": "km/h",
    "wind_direction_10m": "\u00b0",
    "precipitation": "mm",
    "cloud_cover": "%",
}

PROXY_NAME = "AI High-Precipitation Risk Proxy"

DISCLAIMER_TEXT = (
    "This prototype estimates the risk of a high-precipitation event during the "
    "next 3 hours using a machine-learning model trained on historical "
    "atmospheric data. It is not an official IMD thunderstorm or lightning forecast."
)

METRIC_CAVEAT = (
    "Phase 1 test metrics (ROC-AUC 0.8897, F1 0.4863) describe the surrogate "
    "high-precipitation proxy target only. They are NOT thunderstorm or "
    "lightning detection accuracy and must not be relabelled as such."
)

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVALUATION_PATH = OUTPUTS_DIR / "evaluation.json"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"

#: How many recent hourly observations /api/history returns for the trend chart.
HISTORY_MAX_HOURS = 48

#: Only these Phase 1 artifacts may be served verbatim, and only read-only.
ARTIFACT_WHITELIST = {
    "confusion_matrix.png": (CONFUSION_MATRIX_PATH, "image/png"),
}

#: Wording shown next to the evaluation numbers, so they are never mistaken for
#: thunderstorm or lightning detection skill.
EVALUATION_SECTION_TITLE = "Model Evaluation \u2014 High-Precipitation Risk Proxy"
EVALUATION_CAVEAT = (
    "These metrics evaluate the surrogate high-precipitation target and are NOT "
    "thunderstorm or lightning prediction accuracy."
)

#: Wording for the historical demonstration scenario.
SCENARIO_EXPLANATION = (
    "Historical test-set scenario. Model prediction uses only information available "
    "at the selected historical timestamp; the actual next-3-hour precipitation is "
    "shown separately as the observed outcome."
)
SCENARIO_CAVEAT = (
    "This is a historical high-precipitation-risk proxy scenario, not a replay of a "
    "verified thunderstorm or lightning event."
)


class OpenMeteoError(Exception):
    """Raised when live weather cannot be obtained or cannot be trusted."""

    def __init__(self, code: str, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


class ArtifactsUnavailableError(Exception):
    """Raised when a read-only Phase 1 evaluation artifact cannot be read."""

    def __init__(self, code: str, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


class ScenarioUnavailableError(Exception):
    """Raised when no legitimate historical demonstration row can be produced."""

    def __init__(self, code: str, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


class ModelUnavailableError(Exception):
    """Raised when the Phase 1 model artifacts cannot be loaded."""

    code = "model_unavailable"


# --------------------------------------------------------------------------
# Application and model wiring
# --------------------------------------------------------------------------

app = Flask(__name__)

#: ``load_error`` keeps the failure reason so /api/health can report it and
#: /api/prediction can refuse to serve.
#:
#: ``_MODEL`` is the Phase 1 surrogate proxy (legacy, served at
#: /api/prediction/proxy). The Phase 8 primary model -- the Phase 6 1-hour
#: thunderstorm nowcast bundle -- is owned by ``backend.nowcast_service`` and
#: loaded once there, at import.
#:
#: DEPLOYMENT NOTE (512 MiB hosts): the legacy surrogate bundle is deliberately
#: NOT loaded at import. It is not used by the primary dashboard, which calls
#: only /api/prediction, /api/history and /api/evaluation. Nothing about the
#: endpoint changes -- only *when* the artifact is read.
_MODEL: Any = None
_METADATA: dict[str, Any] | None = None
_LOAD_ERROR: str | None = None
_EVALUATION: dict[str, Any] | None = None

#: Guards the lazy legacy load. ``_MODEL_ATTEMPTED`` makes the load single-shot:
#: a failure is recorded once and re-reported on every later call rather than
#: being retried, so a broken artifact stays visibly broken.
_MODEL_LOCK = threading.Lock()
_MODEL_ATTEMPTED = False

#: Cached `models/model_metadata.json` contents, used only so /api/health can
#: report the legacy model type and feature count while the bundle is unloaded.
_LEGACY_METADATA_PREVIEW: dict[str, Any] | None = None

#: Names of the legacy Phase 1 artifacts, used only to answer "can this endpoint
#: serve?" without reading ~130 MB into memory.
PROXY_MODEL_FILENAME = "storm_risk_model.joblib"
PROXY_METADATA_FILENAME = "model_metadata.json"


def _load_model() -> None:
    """Load the Phase 1 artifacts exactly once, recording any failure."""
    global _MODEL, _METADATA, _LOAD_ERROR, _MODEL_ATTEMPTED
    _MODEL_ATTEMPTED = True
    try:
        _MODEL, _METADATA = load_artifacts(MODELS_DIR)
        _LOAD_ERROR = None
    except Exception as exc:  # noqa: BLE001 - surfaced through /api/health
        _MODEL, _METADATA = None, None
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"


def ensure_legacy_model() -> None:
    """Load the legacy Phase 1 surrogate bundle on first use, at most once.

    The lock makes the load single-flight, and ``_MODEL_ATTEMPTED`` makes it
    single-shot, so concurrent first requests produce one read rather than N.

    Deployment consequence (documented deliberately): the first request to
    ``/api/prediction/proxy`` or ``/api/scenario`` pays the artifact read and
    the ~440 MB resident cost, and that memory then stays resident for the life
    of the process, exactly as it did when the load happened at import. Every
    other endpoint -- including the whole primary dashboard -- never triggers
    it.
    """
    if _MODEL_ATTEMPTED:
        return
    with _MODEL_LOCK:
        if _MODEL_ATTEMPTED:
            return
        _load_model()


def legacy_artifacts_present() -> bool:
    """True when both legacy Phase 1 artifact files exist, without reading them."""
    return (MODELS_DIR / PROXY_MODEL_FILENAME).is_file() and (
        MODELS_DIR / PROXY_METADATA_FILENAME
    ).is_file()


def legacy_metadata_preview() -> dict[str, Any]:
    """Read the small legacy metadata JSON without reading the ~130 MB model.

    /api/health reports the legacy model type and feature count. Those two
    values live in `models/model_metadata.json` (a few KB), so they can be
    reported truthfully while the bundle itself stays unloaded. Returns an
    empty mapping if the file is unreadable.
    """
    global _LEGACY_METADATA_PREVIEW
    if _LEGACY_METADATA_PREVIEW is None:
        try:
            _LEGACY_METADATA_PREVIEW = json.loads(
                (MODELS_DIR / PROXY_METADATA_FILENAME).read_text(encoding="utf-8")
            )
        except Exception:  # noqa: BLE001 - reported as missing values, never fatal
            _LEGACY_METADATA_PREVIEW = {}
    return _LEGACY_METADATA_PREVIEW


def legacy_model_loaded() -> bool:
    """True when the legacy Phase 1 bundle is resident in memory."""
    return _MODEL is not None and _METADATA is not None


def _load_evaluation() -> None:
    """Read the Phase 1 evaluation report for reference, if it is present."""
    global _EVALUATION
    try:
        _EVALUATION = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - optional metadata, never fatal
        _EVALUATION = None


#: The legacy surrogate bundle is intentionally not loaded here; see
#: ``ensure_legacy_model``. Only its (cheap) CSV-shaped evaluation metadata is
#: read at import.
_load_evaluation()

#: Phase 8: load the primary (Phase 6 nowcast) bundle once, at import, so the
#: very first /api/health or /api/prediction call does not pay the ~144 MB read.
#: A failure is recorded inside the service layer and reported by /api/health;
#: it never stops the application from starting, and it never yields a
#: substituted prediction.
nowcast_service.warm_up()


def load_evaluation_report() -> dict[str, Any]:
    """Read ``outputs/evaluation.json`` fresh on every call (read-only).

    The file is never modified or rewritten; values are passed through exactly
    as Phase 1 produced them.
    """
    if not EVALUATION_PATH.is_file():
        raise ArtifactsUnavailableError(
            "artifacts_unavailable",
            "outputs/evaluation.json is missing, so the Phase 1 evaluation report "
            "cannot be displayed.",
            str(EVALUATION_PATH),
        )
    try:
        report = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactsUnavailableError(
            "artifacts_unavailable",
            "outputs/evaluation.json could not be read or parsed.",
            f"{type(exc).__name__}: {exc}",
        ) from exc

    if not isinstance(report, dict) or "metrics" not in report:
        raise ArtifactsUnavailableError(
            "artifacts_unavailable",
            "outputs/evaluation.json does not contain the expected 'metrics' block.",
        )
    return report


def load_feature_importance() -> list[dict[str, Any]]:
    """Read ``outputs/feature_importance.csv`` fresh (read-only).

    Importance values are returned exactly as stored; nothing is recomputed,
    rescaled or hard-coded here.
    """
    if not FEATURE_IMPORTANCE_PATH.is_file():
        raise ArtifactsUnavailableError(
            "artifacts_unavailable",
            "outputs/feature_importance.csv is missing, so feature importance "
            "cannot be displayed.",
            str(FEATURE_IMPORTANCE_PATH),
        )
    try:
        frame = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    except (OSError, ValueError) as exc:
        raise ArtifactsUnavailableError(
            "artifacts_unavailable",
            "outputs/feature_importance.csv could not be read or parsed.",
            f"{type(exc).__name__}: {exc}",
        ) from exc

    missing = [column for column in ("feature", "importance") if column not in frame.columns]
    if missing:
        raise ArtifactsUnavailableError(
            "artifacts_unavailable",
            f"outputs/feature_importance.csv is missing column(s): {missing}.",
        )

    return [
        {"feature": str(row.feature), "importance": float(row.importance)}
        for row in frame.itertuples(index=False)
    ]


def finite_or_none(value: Any) -> float | None:
    """Return a JSON-safe float, or ``None`` for NaN/inf.

    ``None`` is used deliberately instead of substituting a placeholder number:
    a missing observation must stay visibly missing rather than become a value.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


#: Guards the lazily computed historical scenario.
_SCENARIO_LOCK = threading.Lock()
_SCENARIO_CACHE: dict[str, Any] | None = None


def compute_historical_scenario() -> dict[str, Any]:
    """Select a real predicted-positive row from the Phase 1 chronological test split.

    The scenario is derived entirely from the immutable Phase 1 dataset and the
    already-trained model:

    * features come from ``phase1.build_features`` -- the identical Phase 1
      engineering, so the 24 predictors are computed exactly as in training;
    * the split comes from ``phase1.chronological_split_indices`` -- the same
      final-15% test window;
    * the prediction is ``model.predict_proba`` over the 24 features only.

    The future 3-hour precipitation is computed solely to display the observed
    outcome afterwards. It is never placed in the feature matrix, so it cannot
    reach ``predict_proba``.
    """
    model, metadata = require_model()
    csv_path = phase1.resolve_dataset_path()

    frame = phase1.load_observations(csv_path)

    # Identical Phase 1 feature engineering. `feature_frame` contains exactly the
    # 24 model predictors and nothing else.
    feature_frame = phase1.build_features(frame)
    # Surrogate target, used only for display and for the row-availability mask.
    future_precip_all = phase1.build_future_precipitation(frame)

    keep = feature_frame.notna().all(axis=1) & future_precip_all.notna()
    features = feature_frame.loc[keep].reset_index(drop=True)
    future_precip = future_precip_all.loc[keep].reset_index(drop=True)
    timestamps = frame.loc[keep, "date"].reset_index(drop=True)
    raw_observations = frame.loc[keep, list(phase1.RAW_COLUMNS)].reset_index(drop=True)

    feature_names = list(metadata["feature_names"])

    # Cross-check against the immutable Phase 1 artifact: if the re-derived row
    # count or feature list disagrees with model_metadata.json, refuse to serve a
    # scenario rather than silently showing something inconsistent.
    expected_rows = (metadata.get("training_data") or {}).get("usable_rows")
    if expected_rows is not None and len(features) != expected_rows:
        raise ScenarioUnavailableError(
            "scenario_inconsistent",
            "Re-derived feature matrix does not match the Phase 1 artifact, so no "
            "scenario can be shown.",
            {"expected_usable_rows": expected_rows, "derived_rows": int(len(features))},
        )
    if list(features.columns) != feature_names:
        raise ScenarioUnavailableError(
            "scenario_inconsistent",
            "Derived feature columns do not match the model's expected feature order.",
            {"derived": list(features.columns), "expected": feature_names},
        )

    leaked = sorted(set(feature_names) & {
        phase1.FUTURE_PRECIP_NAME, phase1.TARGET_NAME, "weather_code"
    })
    if leaked:
        raise ScenarioUnavailableError(
            "scenario_inconsistent",
            "The model feature vector contains a target or leaking column.",
            leaked,
        )

    matrix = features.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ScenarioUnavailableError(
            "scenario_inconsistent",
            "The feature matrix contains NaN or infinite values.",
        )

    # Same chronological split as Phase 1: final 15% is the untouched test window.
    test_index = phase1.chronological_split_indices(len(features))["test"]
    test_matrix = matrix[test_index]

    classes = list(model.classes_)
    if 1 not in classes:
        raise ScenarioUnavailableError(
            "scenario_unavailable", "The loaded model does not expose the positive class.", classes
        )
    positive_column = classes.index(1)

    probabilities = model.predict_proba(test_matrix)[:, positive_column]
    predicted = model.predict(test_matrix).astype(int)

    positives = np.flatnonzero(predicted == 1)
    if positives.size == 0:
        raise ScenarioUnavailableError(
            "scenario_unavailable",
            "The model predicts Elevated Risk for no row in the chronological test "
            "split, so no demonstration scenario exists. Nothing has been fabricated.",
            {"test_rows": int(len(test_index))},
        )

    # Deterministic: highest probability among predicted-positive rows, ties go
    # to the earliest test row.
    best_local = int(positives[int(np.argmax(probabilities[positives]))])
    row = int(test_index[best_local])

    probability = float(probabilities[best_local])
    predicted_class = int(predicted[best_local])
    threshold = metadata.get("decision_threshold_mm_per_3h")
    observed_future = float(future_precip.iloc[row])
    actual_class = int(observed_future >= float(threshold))

    raw = raw_observations.iloc[row]
    conditions = {
        variable: {
            "value": finite_or_none(raw[variable]),
            "unit": EXPECTED_UNITS[variable],
        }
        for variable in HOURLY_VARIABLES
    }

    feature_vector = {
        name: float(features.iloc[row][name]) for name in feature_names
    }

    lag_features = {
        name: feature_vector[name]
        for name in feature_names
        if name.endswith(("_change_1h", "_change_3h", "_roll_3h", "_roll_6h"))
    }

    test_start = timestamps.iloc[int(test_index[0])]
    test_end = timestamps.iloc[int(test_index[-1])]

    return {
        "scenario_type": "historical_test_scenario",
        "mode": "historical_scenario",
        "historical_timestamp": timestamps.iloc[row].isoformat(),
        "input_atmospheric_conditions": conditions,
        "historical_lag_and_rolling_features": lag_features,
        "predicted_class": predicted_class,
        "probability": probability,
        "risk_label": RISK_LABELS[predicted_class],
        "horizon_hours": metadata.get("horizon_hours", 3),
        "actual_proxy_outcome": {
            "future_precip_3h": observed_future,
            "actual_class": actual_class,
            "actual_label": RISK_LABELS[actual_class],
            "threshold_mm_per_3h": threshold,
            "definition": (
                "Accumulated precipitation over the three hours following the "
                "historical timestamp, compared with the training-only 90th "
                "percentile threshold."
            ),
            "note": (
                "Displayed separately for validation context only. This value is not "
                "part of the model input."
            ),
        },
        "explanation": SCENARIO_EXPLANATION,
        "scenario_caveat": SCENARIO_CAVEAT,
        "model": {
            "type": metadata.get("model_type"),
            "feature_count": metadata.get("n_features"),
            "features_used_for_prediction": feature_names,
            "feature_vector_used_for_prediction": feature_vector,
            "decision_threshold_mm_per_3h": threshold,
            "trained_at_utc": metadata.get("created_at_utc"),
        },
        "scenario_selection": {
            "criterion": (
                "Highest predicted probability among chronological test-set rows the "
                "model classifies as Elevated Risk (ties go to the earliest row)."
            ),
            "test_rows_predicted_positive": int(positives.size),
            "selected_test_row_index": int(best_local),
        },
        "dataset_period": {
            "source": csv_path.name,
            "start": timestamps.iloc[0].isoformat(),
            "end": timestamps.iloc[-1].isoformat(),
            "usable_rows": int(len(features)),
        },
        "test_set_only": True,
        "test_set_period": {
            "start": test_start.isoformat(),
            "end": test_end.isoformat(),
            "rows": int(len(test_index)),
        },
        "future_values_used_for_prediction": False,
        "future_precip_3h_is_input_feature": phase1.FUTURE_PRECIP_NAME in feature_names,
        "future_precip_3h": observed_future,
        "actual_class": actual_class,
        "risk_labels": {str(key): value for key, value in RISK_LABELS.items()},
        "location": TARGET_LOCATION["location"],
        "latitude": TARGET_LOCATION["latitude"],
        "longitude": TARGET_LOCATION["longitude"],
        "disclaimer": DISCLAIMER_TEXT,
        "surrogate_target_disclaimer": SURROGATE_TARGET_DISCLAIMER,
        "metric_caveat": METRIC_CAVEAT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def require_model() -> tuple[Any, dict[str, Any]]:
    """Return the loaded model and metadata, or raise ``ModelUnavailableError``.

    This is the only entry point to the legacy bundle: it triggers the deferred
    load on first use and then behaves exactly as it did when the load was
    eager.
    """
    ensure_legacy_model()
    if _MODEL is None or _METADATA is None:
        raise ModelUnavailableError(
            _LOAD_ERROR or "Model artifacts are not available in models/."
        )
    return _MODEL, _METADATA


# --------------------------------------------------------------------------
# Open-Meteo access
# --------------------------------------------------------------------------


def fetch_hourly_weather() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch recent hourly weather for the target location from Open-Meteo.

    Returns
    -------
    (frame, provenance)
        ``frame`` has one row per returned hour with ``date`` (UTC) plus the
        seven model variables. ``provenance`` records the raw request/response
        facts used to prove a real network call was made.

    Raises
    ------
    OpenMeteoError
        On any network failure, non-200 status, malformed body, missing
        variable, unexpected units, or length mismatch. No value is ever
        fabricated to work around a failure.
    """
    params = {
        "latitude": TARGET_LOCATION["latitude"],
        "longitude": TARGET_LOCATION["longitude"],
        "hourly": ",".join(HOURLY_VARIABLES),
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "timezone": "UTC",
    }

    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.Timeout as exc:
        raise OpenMeteoError(
            "upstream_timeout",
            f"Open-Meteo did not respond within {REQUEST_TIMEOUT_SECONDS}s.",
            str(exc),
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise OpenMeteoError(
            "upstream_unreachable",
            "Could not reach the Open-Meteo forecast API.",
            str(exc),
        ) from exc

    if response.status_code != 200:
        raise OpenMeteoError(
            "upstream_status",
            f"Open-Meteo returned HTTP {response.status_code}.",
            response.text[:500],
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenMeteoError(
            "malformed_response",
            "Open-Meteo returned a body that is not valid JSON.",
            str(exc),
        ) from exc

    if not isinstance(payload, dict):
        raise OpenMeteoError(
            "malformed_response",
            "Open-Meteo returned an unexpected JSON structure.",
            type(payload).__name__,
        )

    if payload.get("error"):
        raise OpenMeteoError(
            "upstream_error",
            "Open-Meteo reported an error for this request.",
            payload.get("reason") or payload.get("error"),
        )

    # We request timezone=UTC; a non-zero offset would misalign every timestamp.
    offset = payload.get("utc_offset_seconds")
    if offset not in (0, None):
        raise OpenMeteoError(
            "unexpected_timezone_offset",
            f"Expected UTC timestamps but Open-Meteo reported utc_offset_seconds={offset}.",
        )

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise OpenMeteoError(
            "missing_hourly_block",
            "Open-Meteo response contains no 'hourly' block.",
        )

    if "time" not in hourly or not isinstance(hourly["time"], list) or not hourly["time"]:
        raise OpenMeteoError(
            "missing_hourly_block",
            "Open-Meteo response contains no hourly timestamps.",
        )

    n_hours = len(hourly["time"])

    units = payload.get("hourly_units") or {}
    validate_units(units)

    series: dict[str, Any] = {}
    missing_variables: list[str] = []
    for variable in HOURLY_VARIABLES:
        values = hourly.get(variable)
        if not isinstance(values, list):
            missing_variables.append(variable)
            continue
        if len(values) != n_hours:
            raise OpenMeteoError(
                "length_mismatch",
                f"Variable '{variable}' returned {len(values)} values but there are "
                f"{n_hours} timestamps.",
            )
        series[variable] = values

    if missing_variables:
        raise OpenMeteoError(
            "missing_variables",
            f"Open-Meteo did not return required variable(s): {missing_variables}.",
        )

    frame = pd.DataFrame({"date": pd.to_datetime(hourly["time"], utc=True, errors="coerce")})
    if frame["date"].isna().any():
        raise OpenMeteoError(
            "malformed_response",
            "Open-Meteo returned an unparseable hourly timestamp.",
        )

    for variable in HOURLY_VARIABLES:
        # Non-numeric or null entries stay as NaN; they are never filled in.
        frame[variable] = pd.to_numeric(pd.Series(series[variable]), errors="coerce")

    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)

    provenance = {
        "provider": "Open-Meteo",
        "endpoint": OPEN_METEO_URL,
        "request_parameters": {**params, "hourly": HOURLY_VARIABLES},
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        "hours_returned": int(len(frame)),
        "window_start_utc": frame["date"].iloc[0].isoformat(),
        "window_end_utc": frame["date"].iloc[-1].isoformat(),
        "grid_latitude": payload.get("latitude"),
        "grid_longitude": payload.get("longitude"),
        "elevation_m": payload.get("elevation"),
        "hourly_units": units,
        "upstream_generation_time_ms": payload.get("generationtime_ms"),
    }
    return frame, provenance


def validate_units(units: Any) -> None:
    """Reject responses whose units do not match the model's training units.

    Feeding a differently-scaled variable (for example wind speed in m/s instead
    of km/h) would silently corrupt the prediction, so this is a hard failure.
    """
    if not isinstance(units, dict):
        raise OpenMeteoError(
            "missing_units",
            "Open-Meteo response did not include 'hourly_units'.",
        )

    mismatches = {}
    for variable, expected in EXPECTED_UNITS.items():
        actual = units.get(variable)
        if actual is None or str(actual).strip() != expected:
            mismatches[variable] = {"expected": expected, "received": actual}

    if mismatches:
        raise OpenMeteoError(
            "unexpected_units",
            "Open-Meteo returned units that differ from the units the model was "
            "trained on; refusing to predict on rescaled inputs.",
            mismatches,
        )


def select_latest_observation(
    frame: pd.DataFrame, now: pd.Timestamp | None = None
) -> tuple[pd.Series, pd.DataFrame]:
    """Choose the most recent usable observation and its preceding hourly history.

    Only hours with all seven variables present are usable, and only hours at or
    before ``now`` are considered, so no future weather value can influence the
    features.

    Returns
    -------
    (observation, history)
        ``observation`` is the latest usable hour; ``history`` holds the
        ``MIN_HISTORY_ROWS`` contiguous hours immediately before it, oldest
        first, which is what the Phase 1 lag/rolling features require.
    """
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    needed = MIN_HISTORY_ROWS + 1

    # Drop every hour after the current instant up front. The API also returns
    # hours later today; none of them may influence the features, and keeping
    # them would otherwise make the newest contiguous run look "future-ending".
    total_hours_returned = int(len(frame))
    future_hours_discarded = int((frame["date"] > now).sum())
    frame = frame[frame["date"] <= now].reset_index(drop=True)

    if frame.empty:
        raise OpenMeteoError(
            "no_recent_hours",
            "Open-Meteo returned no hourly observation at or before the current time.",
            {
                "hours_returned": total_hours_returned,
                "latest_returned_hour_utc": None,
            },
        )

    values_present = frame[HOURLY_VARIABLES].notna().all(axis=1).to_numpy()
    times = frame["date"].to_numpy()
    one_hour = np.timedelta64(1, "h")

    # Group rows into runs of consecutive, contiguous, fully-populated hours.
    run_ids = np.zeros(len(frame), dtype=int)
    run = 0
    for index in range(len(frame)):
        if index > 0 and not (
            values_present[index]
            and values_present[index - 1]
            and (times[index] - times[index - 1]) == one_hour
        ):
            run += 1
        run_ids[index] = run

    # Every run now ends at or before ``now``, so the newest qualifying run is
    # simply the most recent usable history.
    candidates: list[tuple[int, int]] = []
    for run_id in np.unique(run_ids):
        positions = np.flatnonzero((run_ids == run_id) & values_present)
        if positions.size >= needed:
            candidates.append((int(positions[0]), int(positions[-1])))

    if not candidates:
        raise OpenMeteoError(
            "insufficient_recent_history",
            "Open-Meteo did not return a run of at least "
            f"{needed} consecutive recent hours, at or before the current time, "
            "with all seven required variables present.",
            {
                "required_consecutive_hours": needed,
                "hours_at_or_before_now": int(len(frame)),
                "hours_with_all_variables": int(np.count_nonzero(values_present)),
                "hours_returned_total": total_hours_returned,
                "future_hours_discarded": future_hours_discarded,
                "latest_usable_hour_utc": frame["date"].iloc[-1].isoformat(),
            },
        )

    start, end = max(candidates, key=lambda bounds: bounds[1])
    window = frame.iloc[end - needed + 1 : end + 1].reset_index(drop=True)

    observation = window.iloc[-1]
    history = window.iloc[:-1].reset_index(drop=True)
    return observation, history


def build_prediction() -> dict[str, Any]:
    """Fetch live weather and run the Phase 1 model on it.

    Raises
    ------
    OpenMeteoError
        If trustworthy live weather could not be obtained.
    ModelUnavailableError
        If the Phase 1 artifacts are not loadable.
    """
    model, metadata = require_model()
    frame, provenance = fetch_hourly_weather()

    # A single reference instant is used for both row selection and reporting.
    now = pd.Timestamp.now(tz="UTC")
    observation, history = select_latest_observation(frame, now=now)
    future_hours_discarded = int((frame["date"] > now).sum())

    try:
        result = predict_risk(observation, history=history, model=model, metadata=metadata)
    except ValueError as exc:
        # ml.predict refuses to score NaN/inf or incomplete history; surface it
        # as an upstream data problem rather than guessing a value.
        raise OpenMeteoError(
            "features_unavailable",
            "The model's features could not be computed from the live weather "
            f"returned by Open-Meteo: {exc}",
        ) from exc

    prediction_timestamp = datetime.now(timezone.utc).isoformat()
    observation_timestamp = observation["date"].isoformat()

    latest_weather = {
        variable: {
            "value": float(observation[variable]),
            "unit": EXPECTED_UNITS[variable],
        }
        for variable in HOURLY_VARIABLES
    }

    return {
        "proxy_name": PROXY_NAME,
        "location": TARGET_LOCATION["location"],
        "latitude": TARGET_LOCATION["latitude"],
        "longitude": TARGET_LOCATION["longitude"],
        "prediction_timestamp": prediction_timestamp,
        "observation_timestamp_utc": observation_timestamp,
        "current_weather": latest_weather,
        "predicted_class": int(result["predicted_class"]),
        "probability": float(result["probability"]),
        "risk_label": result["risk_label"],
        "risk_labels": {str(key): value for key, value in RISK_LABELS.items()},
        "horizon_hours": metadata.get("horizon_hours"),
        "model_target_description": metadata.get("target_definition"),
        "model": {
            "name": (metadata.get("model_file") or "storm_risk_model.joblib").replace(
                ".joblib", ""
            ),
            "type": metadata.get("model_type"),
            "feature_count": metadata.get("n_features"),
            "feature_names": metadata.get("feature_names"),
            "decision_threshold_mm_per_3h": metadata.get("decision_threshold_mm_per_3h"),
            "threshold_fitted_on": metadata.get("threshold_fitted_on"),
            "trained_at_utc": metadata.get("created_at_utc"),
            "trained_on_location": "Open-Meteo archive, 8.4855 N / 76.9492 E",
        },
        "model_metrics_reference": _model_metrics_reference(),
        "metric_caveat": METRIC_CAVEAT,
        "features": {
            "history_hours_used": int(len(history)),
            "history_start_utc": history["date"].iloc[0].isoformat(),
            "history_end_utc": history["date"].iloc[-1].isoformat(),
            "hours_returned_by_api": int(len(frame)),
            "future_hours_discarded": future_hours_discarded,
            "future_values_used": False,
        },
        "data_source": provenance,
        "disclaimer": DISCLAIMER_TEXT,
        "surrogate_target_disclaimer": SURROGATE_TARGET_DISCLAIMER,
    }


def _model_metrics_reference() -> dict[str, Any] | None:
    """Surface the Phase 1 test metrics verbatim, if the report is available."""
    if not _EVALUATION:
        return None
    test = (_EVALUATION.get("metrics") or {}).get("test")
    if not test:
        return None
    return {
        "split": "chronological hold-out test",
        "n_samples": test.get("n_samples"),
        "accuracy": test.get("accuracy"),
        "precision": test.get("precision"),
        "recall": test.get("recall"),
        "f1_score": test.get("f1_score"),
        "roc_auc": test.get("roc_auc"),
        "note": "Metrics for the surrogate high-precipitation proxy target only.",
    }


def _error_response(error: Exception):
    """Render a 503 JSON payload for an upstream, artifact or model failure."""
    code = getattr(error, "code", "internal_error")
    detail = getattr(error, "detail", str(error))

    return (
        jsonify(
            {
                "error": True,
                "status": "error",
                "code": code,
                "message": str(error),
                "detail": detail,
                "location": TARGET_LOCATION["location"],
                "latitude": TARGET_LOCATION["latitude"],
                "longitude": TARGET_LOCATION["longitude"],
                "predicted_class": None,
                "probability": None,
                "risk_label": None,
                "threshold": None,
                "no_prediction_produced": True,
                "note": (
                    "No prediction was produced. No weather or prediction value "
                    "has been fabricated."
                ),
                "disclaimer": DISCLAIMER_TEXT,
            }
        ),
        503,
    )


# --------------------------------------------------------------------------
# Cross-origin access and response hardening
# --------------------------------------------------------------------------

#: API responses are never cacheable. A cached prediction would be presented as
#: current while describing an hour that has already passed, which is exactly
#: the failure mode the freshness policy exists to prevent.
_NO_STORE_PATHS_PREFIXES = ("/api/",)


@app.after_request
def _apply_response_headers(response):
    """Attach CORS and hardening headers to every response.

    Implemented directly rather than by adding ``flask-cors``: the requirement
    is a fixed, read-only header set on a local prototype, and keeping the
    dependency list unchanged is worth more here than the library.
    """
    origin = request.headers.get("Origin")
    if origin:
        if CORS_ALLOW_ANY_ORIGIN or origin in CORS_ALLOWED_ORIGINS:
            response.headers["Access-Control-Allow-Origin"] = origin if not CORS_ALLOW_ANY_ORIGIN else "*"
            # The response varies by Origin once a specific origin is echoed.
            response.headers.add("Vary", "Origin")
    elif CORS_ALLOW_ANY_ORIGIN:
        response.headers["Access-Control-Allow-Origin"] = "*"

    # Read-only surface: only these methods are advertised, so a preflight for
    # POST/PUT/DELETE cannot succeed.
    response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Accept, Content-Type"
    response.headers["Access-Control-Max-Age"] = "600"

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"

    if request.path.startswith(_NO_STORE_PATHS_PREFIXES):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"

    return response


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.route("/")
def dashboard():
    """Phase 9 decision-support dashboard over the Phase 8 prediction API."""
    # Display the Open-Meteo cell the model was trained on / is scored at.
    # Rounded to the same public figures used in Phase 7/8 validation.
    served_lat = round(float(TRAINING_SERVED_GRID_CELL["latitude"]), 6)
    served_lon = round(float(TRAINING_SERVED_GRID_CELL["longitude"]), 5)
    return render_template(
        "index.html",
        page_title="Thunderstorm Nowcast Decision Support",
        proxy_name=PROXY_NAME,
        location=TARGET_LOCATION["location"],
        latitude=TARGET_LOCATION["latitude"],
        longitude=TARGET_LOCATION["longitude"],
        served_grid_latitude=served_lat,
        served_grid_longitude=served_lon,
        disclaimer=NOWCAST_DISCLAIMER,
        metric_caveat=METRIC_CAVEAT,
        evaluation_title=EVALUATION_SECTION_TITLE,
        evaluation_caveat=EVALUATION_CAVEAT,
        imd_disclaimer=(
            "This prototype provides AI-assisted short-horizon thunderstorm risk "
            "information and is not a replacement for official IMD warnings."
        ),
    )


@app.route("/api/health")
def api_health():
    """Phase 8 health check: application, primary model, engine, live data.

    Four things are actually verified rather than assumed:

    * the application is running -- it answered this request;
    * the primary model is available -- the Phase 6 nowcast bundle is loaded
      (the Phase 7 loader re-validates feature order, threshold, lead time and
      target against the metadata before returning it);
    * the prediction engine is available -- the Phase 7 callable exists and the
      Phase 4 feature module it depends on imports and yields 28 features;
    * the live-data dependency is reachable -- an explicit, bounded request to
      Open-Meteo. Pass ``?live=0`` to skip that probe when offline.

    ``status`` is ``ok`` when all of the above hold, ``degraded`` when the model
    and engine work but the live provider is unreachable (a prediction request
    would currently fail with 503), and ``unavailable`` when no prediction can
    be served at all. HTTP 200 is returned only for ``ok``; everything else is
    503, so a monitor that only reads the status code still sees the problem.
    The body always states which check failed.
    """
    probe_upstream = request.args.get("live", "1").lower() not in ("0", "false", "no")
    snapshot = nowcast_service.health_snapshot(probe_upstream=probe_upstream)

    # The legacy Phase 1 surrogate proxy is reported separately, so its status
    # can never be confused with the health of the thunderstorm nowcast.
    #
    # A health probe is deliberately NOT allowed to trigger the lazy load: a
    # monitor polling this endpoint would otherwise pull ~440 MB into memory
    # just by looking. ``available`` therefore reports whether the endpoint can
    # serve (its artifacts are present), and ``loaded_into_memory`` reports
    # whether the bundle happens to be resident right now.
    proxy_loaded = legacy_model_loaded()
    proxy_metadata = _METADATA or legacy_metadata_preview()
    snapshot["legacy_proxy_model"] = {
        "available": legacy_artifacts_present(),
        "loaded_into_memory": proxy_loaded,
        "load_policy": (
            "lazy: the bundle is read on the first request to "
            "/api/prediction/proxy or /api/scenario, never at import"
        ),
        "error": _LOAD_ERROR,
        "model_type": proxy_metadata.get("model_type"),
        "feature_count": proxy_metadata.get("n_features"),
        "target": "high-precipitation risk proxy (surrogate, NOT a thunderstorm label)",
        "served_at": "/api/prediction/proxy",
        "note": (
            "Reported for completeness only. The health of this surrogate model "
            "does not affect the thunderstorm nowcast served at /api/prediction."
        ),
    }
    snapshot["live_probe_performed"] = probe_upstream

    status_code = 200 if snapshot["status"] == "ok" else 503
    return jsonify(snapshot), status_code


@app.route("/api/prediction")
def api_prediction():
    """Phase 8 primary endpoint: a real 1-hour thunderstorm nowcast.

    The response is the Phase 7 engine payload (``probability``,
    ``predicted_class``, ``risk_label``, ``threshold``, ``lead_time_hours``,
    ``prediction_timestamp``, ``feature_timestamp``, ``source.provenance`` with
    the served grid cell, the model identifier and ``feature_completeness``),
    wrapped in a thin API envelope. This module performs no inference itself.

    If the live atmospheric data cannot be obtained -- provider unreachable,
    unexpected units, a missing variable, no contiguous usable window, data
    older than the freshness limit -- the Phase 7 engine refuses and this
    endpoint answers with a JSON error whose prediction fields are ``null``.
    No value is ever substituted, and an error must never be read as "no
    thunderstorm".
    """
    try:
        return jsonify(nowcast_service.live_nowcast()), 200
    except ThunderstormNowcastError as error:
        return (
            jsonify(nowcast_service.engine_error_payload(error)),
            nowcast_service.http_status_for_engine_error(error.code),
        )
    except Exception as error:  # noqa: BLE001 - never leak an HTML 500 traceback
        payload = nowcast_service.engine_error_payload(
            ThunderstormNowcastError(
                "internal_error",
                "The prediction endpoint failed unexpectedly. No prediction has "
                "been fabricated.",
                f"{type(error).__name__}: {error}",
            )
        )
        return jsonify(payload), 500


@app.route("/api/prediction/proxy")
def api_prediction_proxy():
    """Phase 2 endpoint, retained: the Phase 1 surrogate risk proxy.

    Unchanged from Phase 2. Its target is a high-precipitation proxy, NOT a
    thunderstorm or lightning label, and it is a different product from
    ``/api/prediction``.
    """
    try:
        return jsonify(build_prediction()), 200
    except (OpenMeteoError, ModelUnavailableError) as error:
        return _error_response(error)


@app.route("/api/history")
def api_history():
    """Return recent REAL hourly observations for the dashboard trend chart.

    Only hours at or before the current UTC time are returned, so the chart
    cannot show future weather as if it were observed. Missing values are
    returned as JSON ``null`` and are never substituted.
    """
    try:
        frame, provenance = fetch_hourly_weather()
        now = pd.Timestamp.now(tz="UTC")
        recent = frame[frame["date"] <= now].tail(HISTORY_MAX_HOURS).reset_index(drop=True)
        if recent.empty:
            raise OpenMeteoError(
                "no_recent_hours",
                "Open-Meteo returned no hourly observation at or before the current time.",
            )
    except OpenMeteoError as error:
        return _error_response(error)

    hours: list[dict[str, Any]] = []
    for row in recent.itertuples(index=False):
        entry: dict[str, Any] = {"timestamp_utc": row.date.isoformat()}
        for variable in HOURLY_VARIABLES:
            entry[variable] = finite_or_none(getattr(row, variable))
        hours.append(entry)

    return (
        jsonify(
            {
                "location": TARGET_LOCATION["location"],
                "latitude": TARGET_LOCATION["latitude"],
                "longitude": TARGET_LOCATION["longitude"],
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "hours_returned": len(hours),
                "window_start_utc": hours[0]["timestamp_utc"],
                "window_end_utc": hours[-1]["timestamp_utc"],
                "units": EXPECTED_UNITS,
                "hours": hours,
                "future_values_used": False,
                "note": (
                    "Recent observed/short-range hours at or before the current UTC "
                    "time. No future hours and no fabricated values."
                ),
                "data_source": provenance,
                "disclaimer": DISCLAIMER_TEXT,
            }
        ),
        200,
    )


@app.route("/api/evaluation")
def api_evaluation():
    """Serve the existing Phase 1 evaluation metrics and feature importance.

    Values are read from ``outputs/evaluation.json`` and
    ``outputs/feature_importance.csv`` exactly as Phase 1 wrote them. Nothing is
    recomputed, rounded or adjusted here.
    """
    try:
        report = load_evaluation_report()
        importance = load_feature_importance()
    except ArtifactsUnavailableError as error:
        return _error_response(error)

    metrics = report.get("metrics") or {}
    test = metrics.get("test")
    if not isinstance(test, dict):
        return _error_response(
            ArtifactsUnavailableError(
                "artifacts_unavailable",
                "outputs/evaluation.json has no test-set metrics to display.",
            )
        )

    return (
        jsonify(
            {
                "section_title": EVALUATION_SECTION_TITLE,
                "caveat": EVALUATION_CAVEAT,
                "metrics": {
                    "split": "chronological hold-out test",
                    "n_samples": test.get("n_samples"),
                    "accuracy": test.get("accuracy"),
                    "precision": test.get("precision"),
                    "recall": test.get("recall"),
                    "f1_score": test.get("f1_score"),
                    "roc_auc": test.get("roc_auc"),
                    "class_distribution": test.get("class_distribution"),
                },
                "all_splits": {
                    name: {
                        "n_samples": block.get("n_samples"),
                        "accuracy": block.get("accuracy"),
                        "precision": block.get("precision"),
                        "recall": block.get("recall"),
                        "f1_score": block.get("f1_score"),
                        "roc_auc": block.get("roc_auc"),
                    }
                    for name, block in metrics.items()
                    if isinstance(block, dict)
                },
                "target": report.get("target"),
                "confusion_matrix": {
                    "image_url": "/api/artifacts/confusion_matrix.png",
                    "matrix": test.get("confusion_matrix"),
                    "labels": [RISK_LABELS[0], RISK_LABELS[1]],
                    "axis_note": test.get("confusion_matrix_axis_note"),
                },
                "feature_importance": importance,
                "sources": {
                    "evaluation": "outputs/evaluation.json",
                    "feature_importance": "outputs/feature_importance.csv",
                    "confusion_matrix": "outputs/confusion_matrix.png",
                },
                "disclaimer": DISCLAIMER_TEXT,
            }
        ),
        200,
    )


@app.route("/api/artifacts/<path:filename>")
def api_artifact(filename: str):
    """Serve a whitelisted, already-generated Phase 1 artifact verbatim (read-only)."""
    entry = ARTIFACT_WHITELIST.get(filename)
    if entry is None:
        return (
            jsonify(
                {
                    "error": True,
                    "status": "error",
                    "code": "artifact_not_allowed",
                    "message": f"Artifact '{filename}' is not exposed by this API.",
                    "allowed": sorted(ARTIFACT_WHITELIST),
                }
            ),
            404,
        )

    path, mimetype = entry
    if not path.is_file():
        return _error_response(
            ArtifactsUnavailableError(
                "artifacts_unavailable",
                f"{path.name} is missing from outputs/.",
                str(path),
            )
        )

    # max_age=0 so a refreshed dashboard never shows a cached image.
    return send_file(path, mimetype=mimetype, max_age=0)


@app.route("/api/scenario")
def api_scenario():
    """Replay a real predicted-positive row from the Phase 1 chronological test split.

    This is a demonstration of model behaviour on historical data. It does not
    touch, alter or replace the live prediction path.
    """
    global _SCENARIO_CACHE
    try:
        require_model()
        if _SCENARIO_CACHE is None:
            with _SCENARIO_LOCK:
                if _SCENARIO_CACHE is None:
                    _SCENARIO_CACHE = compute_historical_scenario()
        return jsonify(_SCENARIO_CACHE), 200
    except (ScenarioUnavailableError, ModelUnavailableError) as error:
        return _error_response(error)
    except Exception as error:  # noqa: BLE001 - never leak an HTML 500 to the demo UI
        return _error_response(
            ScenarioUnavailableError(
                "scenario_error",
                "The historical scenario could not be prepared. No scenario has been "
                "fabricated.",
                f"{type(error).__name__}: {error}",
            )
        )


@app.errorhandler(404)
def not_found(_error):
    """JSON 404 so API clients never receive an HTML error page."""
    return (
        jsonify(
            {
                "error": True,
                "status": "error",
                "code": "not_found",
                "message": "No such endpoint.",
            }
        ),
        404,
    )


@app.errorhandler(405)
def method_not_allowed(_error):
    """JSON 405: this API surface is read-only (GET/HEAD/OPTIONS)."""
    return (
        jsonify(
            {
                "error": True,
                "status": "error",
                "code": "method_not_allowed",
                "message": "This endpoint is read-only; use GET.",
            }
        ),
        405,
    )


@app.errorhandler(500)
def internal_error(_error):
    """JSON 500 that never leaks a traceback, and never implies a prediction."""
    return (
        jsonify(
            {
                "error": True,
                "status": "error",
                "code": "internal_error",
                "message": "The server hit an unexpected error. No prediction has been fabricated.",
                "probability": None,
                "predicted_class": None,
                "risk_label": None,
                "no_prediction_produced": True,
            }
        ),
        500,
    )


def main() -> int:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    # Report the Phase 8 primary model and the retained Phase 2 surrogate
    # separately, so the startup log cannot be misread as one model.
    nowcast_ok = nowcast_service.model_available()
    print(
        f"[primary ] thunderstorm nowcast (Phase 6/7) loaded: {nowcast_ok}"
        + (
            ""
            if nowcast_ok
            else f" -- /api/prediction will refuse to serve"
        )
    )
    print(
        "[legacy  ] high-precipitation proxy (Phase 1) artifacts present: "
        f"{legacy_artifacts_present()} (loaded lazily, on first use of "
        "/api/prediction/proxy or /api/scenario)"
        + (f" ({_LOAD_ERROR})" if _LOAD_ERROR else "")
    )
    print(f"Serving on http://{host}:{port}/")
    print(f"  GET /                dashboard")
    print(f"  GET /api/health      health (add ?live=0 to skip the provider probe)")
    print(f"  GET /api/prediction  1-hour thunderstorm nowcast (Phase 7 engine)")
    app.run(host=host, port=port, debug=debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
