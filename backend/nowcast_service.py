"""Phase 8 -- Flask service layer over the verified Phase 7 prediction engine.

SIH26072 -- "AIML based nowcasting of thunderstorm and lightning".

RESPONSIBILITY
--------------
This module contains **no** science. It does not fetch weather, build features,
score a model, choose a threshold or define a risk label. All of that lives in
``ml/predict_thunderstorm_nowcast.py`` (Phase 7), which was validated 74/74 and
is imported here read-only. This module's entire job is:

    * load the Phase 6 bundle **once** and keep it in memory (the artifact is
      ~144 MB, so re-reading it per request would be wasteful and slow),
    * call the Phase 7 entry point and hand its payload to the caller
      unchanged, apart from a small, explicitly-marked API envelope,
    * translate ``ThunderstormNowcastError`` into an HTTP status and a JSON
      body that carries **no** probability,
    * answer the health question -- is the app up, are the model and the
      prediction engine usable, is the live-data provider reachable.

WHAT IS DELIBERATELY NOT DONE HERE
----------------------------------
No value is ever invented, defaulted, cached from a previous call, averaged or
rounded. If the Phase 7 engine refuses -- because the provider is unreachable,
the units changed, the newest hour is stale, or a feature came out non-finite --
this module propagates that refusal. A failed request never returns a
probability, and a cached prediction is never served in place of a fresh one.

SCIENTIFIC VOCABULARY THAT THIS LAYER PRESERVES
-----------------------------------------------
The Phase 7 payload keeps four quantities apart, and this layer does not merge
them: ``probability`` (a model score, not a confidence and not a calibrated
thunderstorm frequency), ``threshold`` (the locked 0.0775 cut-off),
``risk_label`` (the result of comparing the two) and ``observed_outcome``
(unavailable by construction, because it belongs to a future hour). The seven
atmospheric inputs are "latest available atmospheric data" from a provider grid
cell, not a station observation. None of that wording is softened here.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests

# --------------------------------------------------------------------------
# The Phase 7 engine: the single source of prediction truth.
# --------------------------------------------------------------------------

from ml.predict_thunderstorm_nowcast import (  # noqa: E402
    DEFAULT_MAX_DATA_AGE_HOURS,
    DISCLAIMER as NOWCAST_DISCLAIMER,
    LEAD_TIME_HOURS,
    LIVE_VARIABLES,
    MODEL_FILENAME,
    MODEL_NAME,
    OPEN_METEO_URL,
    REQUEST_TIMEOUT_SECONDS,
    RISK_LABELS as NOWCAST_RISK_LABELS,
    TRAINING_REQUEST_LATITUDE,
    TRAINING_REQUEST_LONGITUDE,
    TRAINING_SERVED_GRID_CELL,
    ThunderstormNowcastError,
    load_model_bundle,
    measured_skill_reference,
    phase4_feature_module,
    predict_current_thunderstorm_risk,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

#: Identifies the engine this API serves, so a response can always be traced to
#: the module and callable that produced it.
ENGINE_MODULE = "ml.predict_thunderstorm_nowcast"
ENGINE_FUNCTION = "predict_current_thunderstorm_risk"
API_PHASE = "Phase 8 (Flask backend)"

#: The one lead time this API serves. Read from the engine rather than restated,
#: so a future engine change cannot leave two disagreeing copies behind.
EXPECTED_LEAD_TIME_HOURS = LEAD_TIME_HOURS


# --------------------------------------------------------------------------
# Error translation
# --------------------------------------------------------------------------

#: Codes that describe a fault on *our* side -- a missing, unreadable or
#: self-inconsistent artifact, or a missing library. These are 500s: the client
#: did nothing wrong and retrying will not help.
_SERVER_FAULT_CODES = frozenset(
    {
        "sklearn_unavailable",
        "requests_unavailable",
        "model_missing",
        "metadata_missing",
        "model_unloadable",
        "model_bundle_unexpected",
        "model_bundle_incomplete",
        "model_not_probabilistic",
        "model_type_unexpected",
        "feature_names_mismatch",
        "feature_count_unexpected",
        "feature_count_mismatch",
        "leaking_feature_in_model",
        "lead_time_unexpected",
        "target_unexpected",
        "threshold_invalid",
        "metadata_unreadable",
        "threshold_disagreement",
        "metadata_model_type_unexpected",
        "positive_class_missing",
    }
)


def http_status_for_engine_error(code: str) -> int:
    """Map a Phase 7 error code onto an HTTP status.

    Everything that is not a fault on our own side is treated as a dependence
    that is momentarily unavailable -- the upstream provider, the freshness
    window, or the data itself -- and therefore a 503. That default is
    deliberate: it means an unrecognised code degrades to "cannot serve this
    right now" rather than to a 500 that looks like a crash, and never to a
    200 that would imply a prediction exists.
    """
    return 500 if code in _SERVER_FAULT_CODES else 503


def engine_error_payload(error: ThunderstormNowcastError) -> dict[str, Any]:
    """Build the JSON body for a refused prediction.

    The prediction fields are present and explicitly ``null``: a consumer that
    blindly reads ``probability`` gets ``None`` rather than a missing key or,
    worse, a plausible-looking number. ``message`` is the engine's own wording,
    passed through unedited.
    """
    return {
        "error": True,
        "status": "error",
        "code": error.code,
        "message": error.message,
        "detail": error.detail,
        "http_status": http_status_for_engine_error(error.code),
        "prediction": {
            "probability": None,
            "predicted_class": None,
            "risk_label": None,
            "threshold": None,
            "lead_time_hours": None,
        },
        # Top-level mirrors, so both a strict and a permissive client see the
        # same absence of a prediction.
        "probability": None,
        "predicted_class": None,
        "risk_label": None,
        "threshold": None,
        "no_prediction_produced": True,
        "note": (
            "No prediction was produced and no atmospheric value or probability "
            "has been fabricated. The live-data dependence or an artifact is "
            "unavailable; retry when it is reachable."
        ),
        "live_data_failure_is_not_a_no_alert": (
            "An error here must not be read as 'no thunderstorm'. A missing "
            "answer and a negative answer are different things."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------
# Model bundle: loaded once, reused for every request
# --------------------------------------------------------------------------

_MODEL_LOCK = threading.Lock()
_MODEL_BUNDLE: Any = None
_MODEL_ERROR: ThunderstormNowcastError | None = None
_MODEL_LOAD_SECONDS: float | None = None


def load_nowcast_model(*, force_reload: bool = False) -> Any:
    """Return the cached Phase 6 bundle, loading it at most once.

    The lock makes the load single-flight: under a threaded server, N concurrent
    first requests produce one 144 MB read rather than N of them. A failed load
    is recorded and re-raised on every later call rather than being retried
    silently, so a broken artifact stays visibly broken instead of half-working.

    Raises
    ------
    ThunderstormNowcastError
        Propagated from :func:`load_model_bundle` and re-raised on every call.
    """
    global _MODEL_BUNDLE, _MODEL_ERROR, _MODEL_LOAD_SECONDS

    if _MODEL_BUNDLE is not None and not force_reload:
        return _MODEL_BUNDLE
    if _MODEL_ERROR is not None and not force_reload:
        raise _MODEL_ERROR

    with _MODEL_LOCK:
        if _MODEL_BUNDLE is not None and not force_reload:
            return _MODEL_BUNDLE
        if _MODEL_ERROR is not None and not force_reload:
            raise _MODEL_ERROR

        started = time.perf_counter()
        try:
            bundle = load_model_bundle(MODELS_DIR)
        except ThunderstormNowcastError as exc:
            _MODEL_ERROR = exc
            _MODEL_BUNDLE = None
            raise
        except Exception as exc:  # noqa: BLE001 - normalised into the engine's type
            normalised = ThunderstormNowcastError(
                "model_unloadable",
                f"The model artifact could not be loaded: {type(exc).__name__}: {exc}",
                str(MODELS_DIR / MODEL_FILENAME),
            )
            _MODEL_ERROR = normalised
            _MODEL_BUNDLE = None
            raise normalised from exc

        _MODEL_LOAD_SECONDS = time.perf_counter() - started
        _MODEL_BUNDLE = bundle
        _MODEL_ERROR = None
        return bundle


def model_available() -> bool:
    """True when the Phase 6 bundle is loaded and usable."""
    try:
        load_nowcast_model()
    except ThunderstormNowcastError:
        return False
    return True


def warm_up() -> None:
    """Attempt the one-time load, swallowing the failure for startup logging.

    Called once at import of the Flask app so ``/api/health`` can answer
    truthfully on the very first request instead of paying (or timing out on)
    the first load.
    """
    try:
        load_nowcast_model()
    except ThunderstormNowcastError:
        pass


# --------------------------------------------------------------------------
# Live-data dependency probe (health only)
# --------------------------------------------------------------------------

#: The probe is a real network call, so it is cached briefly: a dashboard that
#: polls health every few seconds must not turn into a load generator aimed at
#: a public API.
PROBE_CACHE_SECONDS = 60.0
PROBE_TIMEOUT_SECONDS = 6.0

_PROBE_LOCK = threading.Lock()
_PROBE_CACHE: dict[str, Any] | None = None
_PROBE_CACHE_AT: float = 0.0


def _probe_live_provider_uncached() -> dict[str, Any]:
    """Ask Open-Meteo for the smallest possible response, and report the facts.

    Only reachability and protocol health are checked. The result is never used
    as atmospheric input: the prediction endpoint always performs its own full
    request through the Phase 7 engine, so a warm probe can never be mistaken
    for the data a prediction was built on.
    """
    params = {
        "latitude": TRAINING_REQUEST_LATITUDE,
        "longitude": TRAINING_REQUEST_LONGITUDE,
        "hourly": "temperature_2m",
        "forecast_days": 1,
    }
    started = time.perf_counter()
    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=PROBE_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - any failure is simply "unreachable"
        return {
            "status": "unreachable",
            "reachable": False,
            "endpoint": OPEN_METEO_URL,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Reachability check only. A prediction is never built from this "
                "probe response."
            ),
        }

    body_is_json = False
    try:
        body_is_json = isinstance(response.json(), Mapping)
    except ValueError:
        body_is_json = False

    ok = response.status_code == 200 and body_is_json
    return {
        "status": "reachable" if ok else "unhealthy",
        "reachable": ok,
        "endpoint": OPEN_METEO_URL,
        "http_status": response.status_code,
        "body_is_json_object": body_is_json,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Reachability check only. A prediction is never built from this "
            "probe response."
        ),
    }


def probe_live_provider(*, use_cache: bool = True) -> dict[str, Any]:
    """Cached reachability check for the live atmospheric data dependency."""
    global _PROBE_CACHE, _PROBE_CACHE_AT

    if use_cache:
        with _PROBE_LOCK:
            fresh = _PROBE_CACHE is not None and (time.monotonic() - _PROBE_CACHE_AT) < PROBE_CACHE_SECONDS
            if fresh:
                cached = dict(_PROBE_CACHE or {})
                cached["cached"] = True
                return cached

    result = _probe_live_provider_uncached()

    with _PROBE_LOCK:
        _PROBE_CACHE = dict(result)
        _PROBE_CACHE_AT = time.monotonic()

    result = dict(result)
    result["cached"] = False
    return result


# --------------------------------------------------------------------------
# Prediction engine availability
# --------------------------------------------------------------------------


def engine_status() -> dict[str, Any]:
    """Report whether the Phase 7 engine and its Phase 4 feature source load.

    The check is a real one: the Phase 4 module is actually imported and its
    feature list actually read, because a Phase 7 that cannot rebuild the
    training features is not usable even though the model file may be present.
    """
    status: dict[str, Any] = {
        "available": callable(predict_current_thunderstorm_risk),
        "module": ENGINE_MODULE,
        "function": ENGINE_FUNCTION,
        "lead_time_hours": LEAD_TIME_HOURS,
        "live_variables": list(LIVE_VARIABLES),
        "live_provider": OPEN_METEO_URL,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "max_data_age_hours": DEFAULT_MAX_DATA_AGE_HOURS,
        "feature_module": "dataset/feature_engineering_phase4.py",
        "feature_module_imported": False,
        "feature_count": None,
        "feature_source_error": None,
    }

    try:
        module = phase4_feature_module()
        columns = list(module.FEATURE_COLUMNS)
        status["feature_module_imported"] = True
        status["feature_count"] = len(columns)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal to the probe
        status["available"] = False
        status["feature_source_error"] = f"{type(exc).__name__}: {exc}"

    return status


# --------------------------------------------------------------------------
# The prediction itself
# --------------------------------------------------------------------------


def _envelope(result: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap the Phase 7 payload in a thin, explicitly-labelled API envelope.

    The engine's own keys are copied through untouched. Two additions are made
    and both are *aliases of values the engine already produced*, never new
    numbers:

    * ``served_grid_cell`` -- hoisted from ``source.provenance`` so the grid cell
      the provider actually served is readable without walking the tree;
    * ``model_identifier`` -- hoisted from ``model.name``.

    ``prediction_engine`` names the module and callable that produced the
    payload, so the provenance of a score is auditable from the response alone.
    """
    payload = dict(result)

    provenance = (result.get("source") or {}).get("provenance") or {}
    model = result.get("model") or {}

    payload["served_grid_cell"] = provenance.get("served_grid_cell")
    payload["model_identifier"] = model.get("name")

    # Thin aliases for the Phase 2 dashboard, which still reads these keys.
    # Values are copies of engine fields -- never invented weather numbers.
    payload["observation_timestamp_utc"] = result.get("feature_timestamp")
    conditions = result.get("input_atmospheric_conditions") or {}
    payload["current_weather"] = {
        name: {"value": entry.get("value"), "unit": entry.get("unit")}
        for name, entry in conditions.items()
        if isinstance(entry, Mapping)
    }

    payload["prediction_engine"] = {
        "module": ENGINE_MODULE,
        "function": ENGINE_FUNCTION,
        "phase": "Phase 7 (real-time prediction engine)",
        "validated": "Phase 7 validator: 74/74 PASS",
        "model_artifact": MODEL_FILENAME,
    }
    payload["api"] = {
        "phase": API_PHASE,
        "endpoint": "/api/prediction",
        "method": "GET",
        "engine_delegation": (
            "This response is the Phase 7 engine payload. The backend performed "
            "no inference, feature engineering, thresholding or data fetching of "
            "its own."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return payload


def live_nowcast() -> dict[str, Any]:
    """Produce one real 1-hour thunderstorm nowcast from live atmospheric data.

    The cached Phase 6 bundle is passed into the Phase 7 engine explicitly, so
    the 144 MB artifact is read once for the lifetime of the process rather than
    once per request. Everything else -- the network call, the freshness check,
    the 28 features, the score, the threshold comparison -- is the engine's.

    Raises
    ------
    ThunderstormNowcastError
        On any missing, malformed, stale or inconsistent input. Never returns a
        partial or substituted prediction.
    """
    bundle = load_nowcast_model()
    result = predict_current_thunderstorm_risk(model=bundle)
    if not isinstance(result, Mapping):
        raise ThunderstormNowcastError(
            "engine_payload_unexpected",
            "The prediction engine returned a payload that is not a mapping.",
            type(result).__name__,
        )
    return _envelope(result)


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


def health_snapshot(*, probe_upstream: bool = True) -> dict[str, Any]:
    """Assemble the health view: app, model, engine, live-data dependency.

    ``status`` is one of:

    ``ok``
        Model and engine usable, and the live provider reachable (or the probe
        skipped).
    ``degraded``
        The application is running and the model still loads, but a dependence
        is momentarily unavailable -- typically the live provider. A prediction
        request would currently fail with 503, so this is *not* reported as ok.
    ``unavailable``
        The primary model or the prediction engine is unusable. No prediction
        can be served at all.

    HTTP status is chosen by the caller from this value; see ``app.py``.
    """
    model_error: ThunderstormNowcastError | None
    try:
        bundle = load_nowcast_model()
        model_error = None
    except ThunderstormNowcastError as exc:
        bundle = None
        model_error = exc

    engine = engine_status()

    model: dict[str, Any] = {
        "available": bundle is not None,
        "name": MODEL_NAME,
        "artifact": MODEL_FILENAME,
        "bundle_version": getattr(bundle, "bundle_version", None),
        "estimator_class": type(bundle.model).__name__ if bundle is not None else None,
        "n_features": len(bundle.feature_names) if bundle is not None else None,
        "lead_time_hours": getattr(bundle, "lead_time_hours", None),
        "decision_threshold": getattr(bundle, "threshold", None),
        "target_column": getattr(bundle, "target_column", None),
        "trained_at_utc": getattr(bundle, "trained_at_utc", None),
        "training_dataset_sha256": getattr(bundle, "training_dataset_sha256", None),
        "load_seconds": _MODEL_LOAD_SECONDS,
        "error_code": getattr(model_error, "code", None),
        "error": getattr(model_error, "message", None),
        "loaded_once_at_startup": _MODEL_BUNDLE is not None,
    }

    if probe_upstream:
        live_dependency = probe_live_provider()
    else:
        live_dependency = {
            "status": "not_checked",
            "reachable": None,
            "note": "The live-data reachability probe was skipped for this request.",
        }

    live_ok = live_dependency.get("reachable") is None or live_dependency.get("reachable") is True

    if model["available"] and engine["available"]:
        status = "ok" if live_ok else "degraded"
    else:
        status = "unavailable"

    return {
        "status": status,
        "application": "SIH 2026 thunderstorm nowcast -- prediction backend",
        "phase": API_PHASE,
        "checks": {
            "application_running": {
                "status": "ok",
                "detail": "The Flask application answered this request.",
            },
            "primary_model": model,
            "prediction_engine": engine,
            "live_data_dependency": live_dependency,
        },
        "prediction_endpoint": {
            "path": "/api/prediction",
            "engine": f"{ENGINE_MODULE}.{ENGINE_FUNCTION}",
            "lead_time_hours": LEAD_TIME_HOURS,
            "risk_labels": {str(key): value for key, value in NOWCAST_RISK_LABELS.items()},
            "served_when": "only when the Phase 7 engine returns a payload",
        },
        "legacy_proxy_endpoint": {
            "path": "/api/prediction/proxy",
            "model": "Phase 1 high-precipitation risk proxy (surrogate target)",
            "note": (
                "Retained unchanged from Phase 2. It is a different, surrogate "
                "target and is NOT a thunderstorm forecast."
            ),
        },
        "measured_skill_reference": measured_skill_reference(OUTPUTS_DIR),
        "served_grid": {
            "requested_grid_point": {
                "latitude": TRAINING_REQUEST_LATITUDE,
                "longitude": TRAINING_REQUEST_LONGITUDE,
            },
            "training_grid_cell": TRAINING_SERVED_GRID_CELL,
        },
        "no_fabrication_guarantee": (
            "No atmospheric value or prediction is invented, imputed, defaulted "
            "or carried over from a previous request. A failure never returns a "
            "probability."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": NOWCAST_DISCLAIMER,
    }
