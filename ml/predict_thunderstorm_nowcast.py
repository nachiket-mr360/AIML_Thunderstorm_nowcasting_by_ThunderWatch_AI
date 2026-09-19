"""Phase 7 -- real-time prediction engine for the 1-hour thunderstorm nowcast.

SIH26072 -- "Nowcasting of thunderstorm and lightning".

WHAT THIS MODULE DOES
---------------------
One complete production step:

    latest available atmospheric hour t
        -> the same 28 causal features the model was trained on
        -> probability that the VOTV station reports a thunderstorm in the
           clock hour t + 1
        -> a yes/no alert by comparing that probability with the locked
           decision threshold stored inside the model artifact

It is deliberately the *only* thing this module does. It does not retrain,
recalibrate, re-tune or replace anything: the estimator, the feature order and
the decision threshold are read out of
``models/thunderstorm_nowcast_1h.joblib`` exactly as Phase 6 wrote them.

FEATURES ARE NOT REIMPLEMENTED HERE
-----------------------------------
Requirement for this phase is that production features are *identical* to the
Phase 4 features, so this module does not contain a second copy of the feature
mathematics. It imports ``dataset/feature_engineering_phase4.py`` by file path
and calls its ``add_features()`` on the live window. Every column is therefore
produced by the same code that produced the training table, and
``FEATURE_COLUMNS`` is taken from that same module rather than restated. A
second implementation that merely "looks equivalent" is exactly the failure
mode this design removes.

TEMPORAL SEMANTICS (read this before reading the numbers)
---------------------------------------------------------
The model is a 1-hour nowcast, so the feature hour and the predicted hour are
different hours:

    feature hour t      the latest hour for which the atmospheric inputs are
                        available. All 28 features are functions of hours <= t.
    predicted hour t+1  the clock hour whose VOTV thunderstorm observation the
                        probability refers to. It is one hour after t.

``target_1h`` in Phase 5 is a genuine METAR present-weather observation at
t + 1 h. That observation does not exist yet at prediction time, which is the
whole point of nowcasting. The engine therefore reports the four quantities
separately and never conflates them:

    probability        the model score for hour t+1  (NOT a confidence, and NOT
                       a calibrated frequency of thunderstorms -- the forest was
                       trained with class_weight="balanced", which deliberately
                       distorts the raw scores towards the minority class)
    threshold          the operational cut-off, 0.0775, chosen in Phase 6 on the
                       validation split only and LOCKED there
    risk_label         the human-readable alert produced by comparing the two
    actual observation NOT AVAILABLE for hour t+1 by construction; the engine
                       says so explicitly instead of implying an outcome

WHAT AN ALERT IS, AND IS NOT
----------------------------
A thunderstorm at one station is a rare event (about 3-4% of hours), and the
Phase 6 threshold was chosen to protect recall, so on the held-out test split
the model reached recall 0.604 at precision 0.133, i.e. roughly 6.5 false
alarms per correctly alerted event. An alert is a screening signal, not a
forecast of a definite thunderstorm. Those measured numbers are carried in the
payload (``measured_skill_reference``) so a consumer never sees the probability
without the skill context.

This is a single-station, single-grid-cell prototype. It is not a nationwide
product and it uses no radar, satellite, lightning or NWP fields.

WHERE THE LIVE INPUT COMES FROM
-------------------------------
Open-Meteo's forecast endpoint, at the Phase 3 training request point
(8.482 N, 76.920 E), requesting the same seven variables in the same units the
model was trained on. Two honesty notes:

* The endpoint returns *model-derived latest available* hourly values. These are
  "latest available atmospheric data", NOT the VOTV aerodrome ground
  observation.
* Phase 6 trained on the Open-Meteo *archive* endpoint while this engine reads
  the *forecast* endpoint. The two products are not guaranteed to be
  distributionally identical, so a live score carries an additional,
  unquantified domain-shift caveat. It is stated rather than hidden.

FAILURE POLICY
--------------
No atmospheric value is ever invented, imputed, defaulted or carried forward.
If the provider is unreachable, returns unexpected units, omits a variable,
returns no contiguous fully-populated run of at least
``MIN_PRECEDING_HOURS`` hours, leaves the data staler than the caller's
freshness limit, or if any feature comes out NaN/inf, the engine raises
``ThunderstormNowcastError`` with a machine-readable ``code``. It never returns
a prediction built from substituted data.

One behaviour is worth stating because it is a deliberate choice rather than a
failure: if the newest hour at or before the reference instant is incomplete or
there is a gap, the engine does NOT fill the hole. It moves back to the newest
hour that is complete and continues the hourly sequence, and reports that it did
so in ``selection.trailing_hours_skipped`` together with the resulting
``temporal_semantics.data_age_hours``. If the newest complete hour is further
back than the freshness limit, the run fails with ``stale_data`` rather than
presenting an old hour as current.

USAGE
-----
    from ml.predict_thunderstorm_nowcast import predict_current_thunderstorm_risk

    result = predict_current_thunderstorm_risk()
    result["probability"], result["risk_label"]

    # replay/deterministic mode: score a supplied window ending at a chosen hour
    result = predict_current_thunderstorm_risk(frame=window, now="2025-06-01T12:00:00Z")

CLI
---
    python ml/predict_thunderstorm_nowcast.py
    python ml/predict_thunderstorm_nowcast.py --frame-csv <csv> --now <iso>

SCOPE NOTE
----------
Phase 7 only. No Phase 1-6 dataset, model artifact or evaluation output is
written, moved or modified by this module; ``app.py``, the templates and the
static frontend are untouched.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths and artifact names
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_DIR = PROJECT_ROOT / "dataset"

PHASE4_FEATURE_SCRIPT = DATASET_DIR / "feature_engineering_phase4.py"

MODEL_NAME = "thunderstorm_nowcast_1h"
MODEL_FILENAME = "thunderstorm_nowcast_1h.joblib"
METADATA_FILENAME = "thunderstorm_nowcast_1h_metadata.json"
EVALUATION_FILENAME = "thunderstorm_nowcast_1h_evaluation.json"

#: The one lead time this engine serves. Guarded against the artifact below.
LEAD_TIME_HOURS = 1
EXPECTED_TARGET_COLUMN = "target_1h"
#: The estimator Phase 6 trained. The check is an isinstance test rather than a
#: string compare, because the class's ``__module__`` is the private
#: ``sklearn.ensemble._forest`` path while Phase 6 records the public
#: ``sklearn.ensemble`` path; comparing strings would reject a correct artifact.
EXPECTED_ESTIMATOR_CLASS = "RandomForestClassifier"
EXPECTED_MODEL_TYPE_RECORDED = "sklearn.ensemble.RandomForestClassifier"

# --------------------------------------------------------------------------
# Live source configuration
# --------------------------------------------------------------------------

#: Phase 3 requested the atmospheric series at the VOTV aerodrome reference
#: point 8.482 N / 76.920 E; Open-Meteo served the archive grid cell
#: 8.4710016 N / 76.9329834 E at 4 m. The live request must use the *same*
#: request point as training so the engine reads the same grid cell the model
#: learned on. (This deliberately differs from app.py's Phase 1 site
#: 8.4855 N / 76.9492 E, which belongs to the surrogate model's own pipeline.)
TRAINING_REQUEST_LATITUDE = 8.482
TRAINING_REQUEST_LONGITUDE = 76.920
TRAINING_SERVED_GRID_CELL = {
    "latitude": 8.471001625061035,
    "longitude": 76.9329833984375,
    "elevation_m": 4.0,
}

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

#: The seven variables the 28 features are derived from. ``weather_code`` is
#: absent on purpose: Phase 3 found zero hours with thunderstorm codes and it is
#: excluded from the model, so it is never requested here either.
LIVE_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]

#: Units the model was trained on. A differently-scaled variable (wind speed in
#: m/s instead of km/h, say) would silently corrupt every derived feature, so a
#: mismatch is a hard failure rather than something to rescale.
EXPECTED_UNITS = {
    "temperature_2m": "\u00b0C",
    "relative_humidity_2m": "%",
    "surface_pressure": "hPa",
    "wind_speed_10m": "km/h",
    "wind_direction_10m": "\u00b0",
    "precipitation": "mm",
    "cloud_cover": "%",
}

PAST_DAYS = 3
FORECAST_DAYS = 1
REQUEST_TIMEOUT_SECONDS = 20.0

#: Hours of history required *before* the feature hour. Phase 4 kept only rows
#: at or after window_start + MAX_HISTORY_HOURS, so the shortest history any
#: training row ever had is 6 preceding hours (the 6-hour precipitation
#: accumulation needs t-5 .. t). Requiring the same here means a live feature
#: vector is never computed from less history than any training row.
MIN_PRECEDING_HOURS = 6

#: Freshness policy. A nowcast presented as "current" must not be built on data
#: that has stopped arriving; beyond this age the engine refuses (codes
#: ``stale_data``). Pass ``max_data_age_hours=None`` to disable the guard for a
#: historical replay, where staleness is expected and legitimate.
DEFAULT_MAX_DATA_AGE_HOURS = 3.0

# --------------------------------------------------------------------------
# Output vocabulary
# --------------------------------------------------------------------------

#: Two levels only, tied directly to the locked threshold. No finer grading is
#: offered because the score is neither calibrated nor validated for band
#: cut-offs, and inventing bands would imply precision the evidence lacks.
RISK_LABELS = {
    0: "No thunderstorm alert",
    1: "Thunderstorm alert",
}
RISK_LEVELS = {
    0: "no_alert",
    1: "alert",
}

DISCLAIMER = (
    "Station-based thunderstorm nowcast prototype. The probability refers to the "
    "VOTV station reporting a thunderstorm in one specific future clock hour "
    "(t + 1 h) and is produced from a single Open-Meteo grid cell about 2 km from "
    "the aerodrome. It is not an official IMD product, not a nationwide forecast, "
    "and it uses no radar, satellite, lightning or NWP fields. A probability is "
    "not certainty, and an alert is a screening signal limited by the measured "
    "false-alarm burden reported alongside it."
)

PROBABILITY_SEMANTICS = (
    "Model score for the positive class (thunderstorm reported at the station in "
    "hour t + 1 h). It is NOT a confidence, NOT a probability of being correct, "
    "and NOT a calibrated frequency of thunderstorms: the forest was trained with "
    "class_weight='balanced', which shifts scores towards the rare class on "
    "purpose. Only the order of scores and the locked threshold have been "
    "validated."
)

THRESHOLD_SEMANTICS = (
    "Operational cut-off selected in Phase 6 on the validation split only "
    "(maximise F2 subject to a predicted-positive rate <= 0.20) and then locked. "
    "The test split was scored once with this value. It has not been adjusted, "
    "re-tuned or re-optimised in Phase 7."
)

RISK_LABEL_SEMANTICS = (
    "Human-readable alert obtained by comparing the probability with the locked "
    "threshold. It reports a threshold crossing, not an observed or physically "
    "certain thunderstorm."
)

ACTUAL_OBSERVATION_SEMANTICS = (
    "The genuine VOTV METAR observation for the predicted hour t + 1 h does not "
    "exist at prediction time, so no observed outcome is reported here. Phase 6 "
    "states the measured skill of exactly this model/threshold pair separately."
)

EXTERNAL_DOMAIN_SHIFT_NOTE = (
    "Phase 6 trained on the Open-Meteo archive endpoint; this engine reads the "
    "Open-Meteo forecast endpoint, whose latest hours are model-derived. The two "
    "products are not guaranteed to be distributionally identical, so a live "
    "score carries an additional, unquantified domain-shift caveat."
)

GROUND_TRUTH_NOTE = (
    "The seven atmospheric inputs are 'latest available atmospheric data' from a "
    "model-derived provider grid cell. They are NOT the VOTV aerodrome ground "
    "observation, and they are not presented as one."
)


class ThunderstormNowcastError(Exception):
    """Raised when no trustworthy prediction can be produced.

    ``code`` is a stable machine-readable identifier and ``detail`` carries the
    supporting facts (counts, missing columns, upstream text). No caller should
    ever receive a probability alongside this error.
    """

    def __init__(self, code: str, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


# --------------------------------------------------------------------------
# Phase 4 feature code, loaded by path (single source of truth)
# --------------------------------------------------------------------------

_PHASE4_MODULE: Any = None


def phase4_feature_module() -> Any:
    """Import ``dataset/feature_engineering_phase4.py`` and return the module.

    Loading by file path (rather than duplicating the maths) is what guarantees
    the production feature vector is bit-identical to the training table. The
    Phase 4 script has a ``__main__`` guard, so importing it only defines
    constants and functions; it does not build or write anything.
    """
    global _PHASE4_MODULE
    if _PHASE4_MODULE is None:
        if not PHASE4_FEATURE_SCRIPT.is_file():
            raise ThunderstormNowcastError(
                "phase4_module_missing",
                "The Phase 4 feature engineering script is missing, so production "
                "features cannot be guaranteed identical to the training features.",
                str(PHASE4_FEATURE_SCRIPT),
            )
        spec = importlib.util.spec_from_file_location(
            "_phase4_feature_engineering", PHASE4_FEATURE_SCRIPT
        )
        if spec is None or spec.loader is None:
            raise ThunderstormNowcastError(
                "phase4_module_unloadable",
                "The Phase 4 feature engineering script could not be loaded.",
                str(PHASE4_FEATURE_SCRIPT),
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _PHASE4_MODULE = module
    return _PHASE4_MODULE


def feature_columns() -> list[str]:
    """The 28 modelled features, in order, as Phase 4 defines them."""
    return list(phase4_feature_module().FEATURE_COLUMNS)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def as_utc_timestamp(value: Any, *, field: str) -> pd.Timestamp:
    """Parse a timestamp-like value into a tz-aware UTC ``pd.Timestamp``."""
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if stamp is None or pd.isna(stamp):
        raise ThunderstormNowcastError(
            "bad_timestamp",
            f"Could not interpret {field} as a UTC timestamp.",
            {"field": field, "value": repr(value)},
        )
    return stamp


def _iso(stamp: pd.Timestamp) -> str:
    return stamp.isoformat()


# --------------------------------------------------------------------------
# Model artifact access (read-only)
# --------------------------------------------------------------------------


def _expected_estimator_class() -> Any:
    """The public ``RandomForestClassifier`` class, imported lazily."""
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ThunderstormNowcastError(
            "sklearn_unavailable",
            "scikit-learn is required to load the Phase 6 model artifact.",
            str(exc),
        ) from exc
    return RandomForestClassifier


@dataclasses.dataclass(frozen=True)
class NowcastModel:
    """The loaded 1-hour nowcast artifact plus everything needed to trust it."""

    model: Any
    feature_names: tuple[str, ...]
    threshold: float
    lead_time_hours: int
    target_column: str
    bundle_version: Any
    trained_at_utc: Any
    training_dataset_sha256: Any
    positive_class_index: int
    model_path: Path
    metadata_path: Path
    metadata: Mapping[str, Any]

    @property
    def model_type(self) -> str:
        return f"{type(self.model).__module__}.{type(self.model).__name__}"


def load_model_bundle(models_dir: str | Path | None = None) -> NowcastModel:
    """Load and validate the 1-hour artifact. Never writes anything.

    Every claim this engine makes downstream (feature order, threshold, lead
    time, target) is taken from the artifact itself and cross-checked against
    the Phase 6 metadata and the Phase 4 feature list. A disagreement is a hard
    error: guessing which of two disagreeing sources is right is how a silent
    feature-order bug reaches production.
    """
    directory = Path(models_dir) if models_dir else MODELS_DIR
    model_path = directory / MODEL_FILENAME
    metadata_path = directory / METADATA_FILENAME

    if not model_path.is_file():
        raise ThunderstormNowcastError(
            "model_missing",
            f"The Phase 6 model artifact is missing: {model_path.name}",
            str(model_path),
        )
    if not metadata_path.is_file():
        raise ThunderstormNowcastError(
            "metadata_missing",
            f"The Phase 6 model metadata is missing: {metadata_path.name}",
            str(metadata_path),
        )

    try:
        bundle = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed error
        raise ThunderstormNowcastError(
            "model_unloadable",
            f"The model artifact could not be loaded: {type(exc).__name__}: {exc}",
            str(model_path),
        ) from exc

    if not isinstance(bundle, Mapping):
        raise ThunderstormNowcastError(
            "model_bundle_unexpected",
            "The model artifact is not the Phase 6 dict bundle. Loading a bare "
            "estimator would lose the validated threshold and feature order.",
            {"type": type(bundle).__name__, "path": str(model_path)},
        )

    required_keys = {
        "model",
        "feature_names",
        "target_column",
        "lead_time_hours",
        "decision_threshold",
        "bundle_version",
    }
    missing = sorted(required_keys - set(bundle.keys()))
    if missing:
        raise ThunderstormNowcastError(
            "model_bundle_incomplete",
            f"The model bundle is missing required key(s): {missing}.",
            sorted(bundle.keys()),
        )

    estimator = bundle["model"]
    if not hasattr(estimator, "predict_proba"):
        raise ThunderstormNowcastError(
            "model_not_probabilistic",
            "The stored estimator does not expose predict_proba.",
            type(estimator).__name__,
        )

    model_type = f"{type(estimator).__module__}.{type(estimator).__name__}"
    if not isinstance(estimator, _expected_estimator_class()):
        raise ThunderstormNowcastError(
            "model_type_unexpected",
            "Expected a RandomForestClassifier but the artifact holds "
            f"{model_type}.",
            model_type,
        )

    # -- feature contract ---------------------------------------------------
    phase4_features = feature_columns()
    stored_features = list(bundle["feature_names"])
    if stored_features != phase4_features:
        raise ThunderstormNowcastError(
            "feature_names_mismatch",
            "The model's feature order does not match the Phase 4 feature list; "
            "refusing to build a vector whose column order could be wrong.",
            {
                "model": stored_features,
                "phase4": phase4_features,
                "same_set_different_order": sorted(stored_features) == sorted(phase4_features),
            },
        )
    if len(phase4_features) != 28:
        raise ThunderstormNowcastError(
            "feature_count_unexpected",
            f"Expected 28 Phase 4 features, found {len(phase4_features)}.",
            phase4_features,
        )
    forbidden = [
        name
        for name in stored_features
        if "target" in name.lower() or "label" in name.lower() or name == "weather_code"
    ]
    if forbidden:
        raise ThunderstormNowcastError(
            "leaking_feature_in_model",
            "The model's feature list contains a target, label or weather_code column.",
            forbidden,
        )
    fitted_on = getattr(estimator, "n_features_in_", None)
    if fitted_on != len(stored_features):
        raise ThunderstormNowcastError(
            "feature_count_mismatch",
            "The estimator was fitted on a different number of features than the "
            "bundle declares.",
            {"n_features_in_": fitted_on, "declared": len(stored_features)},
        )

    # -- lead time / target -------------------------------------------------
    lead_time = bundle["lead_time_hours"]
    if lead_time != LEAD_TIME_HOURS:
        raise ThunderstormNowcastError(
            "lead_time_unexpected",
            f"This engine serves only the {LEAD_TIME_HOURS} h lead time, but the "
            f"artifact declares {lead_time} h.",
            lead_time,
        )
    if bundle["target_column"] != EXPECTED_TARGET_COLUMN:
        raise ThunderstormNowcastError(
            "target_unexpected",
            f"Expected target column {EXPECTED_TARGET_COLUMN!r} but the artifact "
            f"declares {bundle['target_column']!r}.",
            bundle["target_column"],
        )

    # -- threshold ----------------------------------------------------------
    try:
        threshold = float(bundle["decision_threshold"])
    except (TypeError, ValueError) as exc:
        raise ThunderstormNowcastError(
            "threshold_invalid",
            "The artifact's decision threshold is not a usable number.",
            repr(bundle["decision_threshold"]),
        ) from exc
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ThunderstormNowcastError(
            "threshold_invalid",
            "The artifact's decision threshold is not a probability in (0, 1).",
            threshold,
        )

    metadata: Mapping[str, Any] = {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ThunderstormNowcastError(
            "metadata_unreadable",
            f"Could not read the model metadata: {type(exc).__name__}: {exc}",
            str(metadata_path),
        ) from exc

    recorded_threshold = (
        (metadata.get("threshold") or {}).get("selected_threshold")
        if isinstance(metadata.get("threshold"), Mapping)
        else None
    )
    if recorded_threshold is None or abs(float(recorded_threshold) - threshold) > 1e-12:
        raise ThunderstormNowcastError(
            "threshold_disagreement",
            "The bundle threshold and the metadata threshold disagree; refusing to "
            "pick one silently.",
            {"bundle": threshold, "metadata": recorded_threshold},
        )

    recorded_model_type = (
        (metadata.get("model") or {}).get("type")
        if isinstance(metadata.get("model"), Mapping)
        else None
    )
    if recorded_model_type is not None and not str(recorded_model_type).endswith(
        EXPECTED_ESTIMATOR_CLASS
    ):
        raise ThunderstormNowcastError(
            "metadata_model_type_unexpected",
            "The metadata describes a different estimator than the one loaded.",
            {"metadata": recorded_model_type, "loaded": model_type},
        )

    classes = list(getattr(estimator, "classes_", []))
    if 1 not in classes:
        raise ThunderstormNowcastError(
            "positive_class_missing",
            "The estimator does not expose the positive (thunderstorm) class.",
            classes,
        )

    return NowcastModel(
        model=estimator,
        feature_names=tuple(stored_features),
        threshold=threshold,
        lead_time_hours=int(lead_time),
        target_column=str(bundle["target_column"]),
        bundle_version=bundle.get("bundle_version"),
        trained_at_utc=bundle.get("trained_at_utc"),
        training_dataset_sha256=bundle.get("training_dataset_sha256"),
        positive_class_index=int(classes.index(1)),
        model_path=model_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def measured_skill_reference(outputs_dir: str | Path | None = None) -> dict[str, Any] | None:
    """Read the Phase 6 test metrics for this exact model/threshold, read-only.

    Returned so a consumer cannot see a probability without the measured skill
    that qualifies it. ``None`` if the file is unavailable -- the engine still
    works, but it then reports no skill context rather than inventing any.
    """
    directory = Path(outputs_dir) if outputs_dir else OUTPUTS_DIR
    path = directory / EVALUATION_FILENAME
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    test = ((report.get("metrics") or {}).get("test")) or {}
    validation = ((report.get("metrics") or {}).get("validation")) or {}
    if not test:
        return None

    return {
        "source": f"outputs/{EVALUATION_FILENAME}",
        "split": "chronological hold-out test (scored once with the locked threshold)",
        "threshold": test.get("threshold"),
        "test_rows": test.get("n_samples"),
        "base_positive_rate": test.get("baseline_positive_rate"),
        "predicted_alert_rate": test.get("predicted_positive_rate"),
        "recall": test.get("recall"),
        "precision": test.get("precision"),
        "f1": test.get("f1"),
        "roc_auc": test.get("roc_auc"),
        "pr_auc_average_precision": test.get("average_precision_pr_auc"),
        "accuracy": test.get("accuracy"),
        "confusion_matrix": test.get("confusion_matrix"),
        "false_alarms_per_hit": test.get("false_alarms_per_hit"),
        "validation_recall": validation.get("recall"),
        "caveat": (
            "Measured on held-out data for this model and threshold. Precision is low "
            "because the event is rare and the threshold protects recall: most alerts "
            "will not be followed by a thunderstorm. Accuracy is high because the "
            "majority class dominates, and it is NOT evidence of skill."
        ),
    }


# --------------------------------------------------------------------------
# Live source
# --------------------------------------------------------------------------


def _validate_units(units: Any) -> None:
    if not isinstance(units, Mapping):
        raise ThunderstormNowcastError(
            "missing_units",
            "Open-Meteo did not report the units of the returned series.",
        )
    mismatches = {}
    for variable, expected in EXPECTED_UNITS.items():
        actual = units.get(variable)
        if actual is None or str(actual).strip() != expected:
            mismatches[variable] = {"expected": expected, "received": actual}
    if mismatches:
        raise ThunderstormNowcastError(
            "unexpected_units",
            "Open-Meteo returned units that differ from the units the model was "
            "trained on; refusing to predict on rescaled inputs.",
            mismatches,
        )


def fetch_live_atmospheric_data(
    *,
    latitude: float = TRAINING_REQUEST_LATITUDE,
    longitude: float = TRAINING_REQUEST_LONGITUDE,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    past_days: int = PAST_DAYS,
    forecast_days: int = FORECAST_DAYS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch recent hourly atmospheric data for the training grid cell.

    Returns ``(frame, provenance)``. ``frame`` has ``timestamp_utc`` plus the
    seven model variables, ascending and de-duplicated. ``provenance`` records
    the raw request/response facts, including the grid cell the provider says it
    served, so a value can always be traced to the cell it belongs to.

    Raises ``ThunderstormNowcastError`` on any network failure, non-200 status,
    malformed body, missing variable, unit mismatch or length mismatch. Nothing
    is ever substituted for a failed request.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ThunderstormNowcastError(
            "requests_unavailable",
            "The 'requests' package is required to reach Open-Meteo.",
            str(exc),
        ) from exc

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(LIVE_VARIABLES),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }

    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - network errors are all fatal here
        name = type(exc).__name__
        code = "upstream_timeout" if "Timeout" in name else "upstream_unreachable"
        raise ThunderstormNowcastError(
            code,
            f"Could not obtain live atmospheric data from Open-Meteo: {name}: {exc}",
            {"endpoint": OPEN_METEO_URL, "request_parameters": params},
        ) from exc

    if response.status_code != 200:
        raise ThunderstormNowcastError(
            "upstream_status",
            f"Open-Meteo returned HTTP {response.status_code}.",
            response.text[:500],
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise ThunderstormNowcastError(
            "malformed_response",
            "Open-Meteo returned a body that is not valid JSON.",
            str(exc),
        ) from exc

    if not isinstance(payload, Mapping):
        raise ThunderstormNowcastError(
            "malformed_response",
            "Open-Meteo returned an unexpected JSON structure.",
            type(payload).__name__,
        )
    if payload.get("error"):
        raise ThunderstormNowcastError(
            "upstream_error",
            "Open-Meteo reported an error for this request.",
            payload.get("reason") or payload.get("error"),
        )

    offset = payload.get("utc_offset_seconds")
    if offset not in (0, None):
        raise ThunderstormNowcastError(
            "unexpected_timezone_offset",
            f"Expected UTC timestamps but Open-Meteo reported utc_offset_seconds={offset}.",
            offset,
        )

    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping) or not hourly.get("time"):
        raise ThunderstormNowcastError(
            "missing_hourly_block",
            "Open-Meteo returned no hourly block or no hourly timestamps.",
        )

    n_hours = len(hourly["time"])
    units = payload.get("hourly_units") or {}
    _validate_units(units)

    missing_variables: list[str] = []
    series: dict[str, Any] = {}
    for variable in LIVE_VARIABLES:
        values = hourly.get(variable)
        if values is None:
            missing_variables.append(variable)
            continue
        if len(values) != n_hours:
            raise ThunderstormNowcastError(
                "length_mismatch",
                f"Variable '{variable}' returned {len(values)} values for "
                f"{n_hours} timestamps.",
                {"variable": variable, "values": len(values), "timestamps": n_hours},
            )
        series[variable] = values
    if missing_variables:
        raise ThunderstormNowcastError(
            "missing_variables",
            f"Open-Meteo did not return required variable(s): {missing_variables}.",
            missing_variables,
        )

    frame = pd.DataFrame(
        {"timestamp_utc": pd.to_datetime(hourly["time"], utc=True, errors="coerce")}
    )
    if frame["timestamp_utc"].isna().any():
        raise ThunderstormNowcastError(
            "malformed_response",
            "Open-Meteo returned an unparseable hourly timestamp.",
        )

    for variable in LIVE_VARIABLES:
        # Non-numeric and null entries stay NaN. They are never filled in.
        frame[variable] = pd.to_numeric(pd.Series(series[variable]), errors="coerce")

    frame = frame.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)

    provenance = {
        "provider": "Open-Meteo",
        "product": "forecast endpoint, latest available hours",
        "endpoint": OPEN_METEO_URL,
        "request_parameters": {**params, "hourly": LIVE_VARIABLES},
        "requested_at_utc": utc_now().isoformat(),
        "hours_returned": int(len(frame)),
        "window_start_utc": _iso(frame["timestamp_utc"].iloc[0]),
        "window_end_utc": _iso(frame["timestamp_utc"].iloc[-1]),
        "requested_grid_point": {"latitude": latitude, "longitude": longitude},
        "served_grid_cell": {
            "latitude": payload.get("latitude"),
            "longitude": payload.get("longitude"),
            "elevation_m": payload.get("elevation"),
        },
        "training_grid_cell": TRAINING_SERVED_GRID_CELL,
        "hourly_units": dict(units),
        "ground_truth_note": GROUND_TRUTH_NOTE,
        "domain_shift_note": EXTERNAL_DOMAIN_SHIFT_NOTE,
    }
    return frame, provenance


# --------------------------------------------------------------------------
# Window selection
# --------------------------------------------------------------------------


def normalise_live_frame(frame: Any) -> pd.DataFrame:
    """Normalise a caller-supplied frame to the live-frame contract.

    Accepts ``timestamp_utc`` or Open-Meteo's ``date`` as the time column, and
    requires the seven variables. The time column is parsed to timezone-aware UTC
    and the variables to numeric: a frame read from CSV arrives with string
    timestamps and would otherwise sort lexicographically and misalign every lag.
    Anything that cannot be parsed raises rather than being guessed at.
    """
    if frame is None:
        raise ThunderstormNowcastError("no_frame", "No atmospheric frame was supplied.")
    if not isinstance(frame, pd.DataFrame):
        raise ThunderstormNowcastError(
            "frame_not_a_dataframe",
            "The supplied atmospheric frame is not a pandas DataFrame.",
            type(frame).__name__,
        )

    working = frame.copy()
    if "timestamp_utc" not in working.columns:
        if "date" in working.columns:
            working = working.rename(columns={"date": "timestamp_utc"})
        else:
            raise ThunderstormNowcastError(
                "frame_missing_timestamp",
                "The supplied frame has no 'timestamp_utc' (or 'date') column.",
                list(working.columns),
            )

    missing = [name for name in LIVE_VARIABLES if name not in working.columns]
    if missing:
        raise ThunderstormNowcastError(
            "frame_missing_variables",
            f"The supplied frame is missing required variable(s): {missing}.",
            {"missing": missing, "present": list(working.columns)},
        )

    # Strings become tz-aware UTC timestamps, and values that cannot be parsed
    # become NaT/NaN so the completeness checks reject them. Nothing is filled in.
    working["timestamp_utc"] = pd.to_datetime(
        working["timestamp_utc"], utc=True, errors="coerce", format="ISO8601"
    )
    for variable in LIVE_VARIABLES:
        working[variable] = pd.to_numeric(working[variable], errors="coerce")

    if working["timestamp_utc"].isna().any():
        raise ThunderstormNowcastError(
            "frame_bad_timestamp",
            "The supplied frame contains an unparseable timestamp.",
        )

    working = working.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
    if working["timestamp_utc"].duplicated().any():
        raise ThunderstormNowcastError(
            "frame_duplicate_timestamps",
            "The supplied frame contains duplicate hourly timestamps, so a lag "
            "would silently read the wrong hour.",
            int(working["timestamp_utc"].duplicated().sum()),
        )
    return working


def select_latest_feature_window(
    frame: pd.DataFrame,
    *,
    now: pd.Timestamp,
    min_preceding_hours: int = MIN_PRECEDING_HOURS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pick the newest usable hour t and the contiguous history ending at it.

    Only hours at or before ``now`` are considered, so no future hour can reach
    a feature. A usable hour has all seven variables present; the window is the
    trailing run of consecutive, fully-populated hours, and that run must supply
    at least ``min_preceding_hours`` hours before t -- the shortest history any
    Phase 4 training row ever had.
    """
    needed = min_preceding_hours + 1

    total_hours_returned = int(len(frame))
    future_hours_discarded = int((frame["timestamp_utc"] > now).sum())
    usable_by_time = frame[frame["timestamp_utc"] <= now].reset_index(drop=True)

    if usable_by_time.empty:
        raise ThunderstormNowcastError(
            "no_recent_hours",
            "No hourly atmospheric data at or before the current time was available.",
            {
                "hours_returned": total_hours_returned,
                "future_hours_discarded": future_hours_discarded,
            },
        )

    values_present = usable_by_time[LIVE_VARIABLES].notna().all(axis=1).to_numpy()
    times = usable_by_time["timestamp_utc"].to_numpy()
    one_hour = np.timedelta64(1, "h")

    # Group rows into runs of consecutive, contiguous, fully-populated hours.
    run_ids = np.zeros(len(usable_by_time), dtype=int)
    run = 0
    for index in range(len(usable_by_time)):
        if index > 0 and not (
            values_present[index]
            and values_present[index - 1]
            and (times[index] - times[index - 1]) == one_hour
        ):
            run += 1
        run_ids[index] = run

    candidates: list[tuple[int, int]] = []
    for run_id in np.unique(run_ids):
        positions = np.flatnonzero((run_ids == run_id) & values_present)
        if positions.size >= needed:
            candidates.append((int(positions[0]), int(positions[-1])))

    if not candidates:
        raise ThunderstormNowcastError(
            "insufficient_recent_history",
            "Open-Meteo did not supply a run of "
            f"{needed} consecutive recent hours (at least {min_preceding_hours} hours "
            "before the feature hour) with all seven variables present. Every "
            "Phase 4 feature needs that history, and none of it is imputed.",
            {
                "required_consecutive_hours": needed,
                "required_preceding_hours": min_preceding_hours,
                "hours_at_or_before_now": int(len(usable_by_time)),
                "hours_with_all_variables": int(np.count_nonzero(values_present)),
                "hours_returned_total": total_hours_returned,
                "future_hours_discarded": future_hours_discarded,
            },
        )

    start, end = max(candidates, key=lambda bounds: bounds[1])
    window = usable_by_time.iloc[start : end + 1].reset_index(drop=True)

    feature_timestamp = window["timestamp_utc"].iloc[-1]
    # Hours newer than the chosen feature hour are dropped, either because they
    # are incomplete or because there is a gap. They are counted and named so a
    # caller can always see that the engine stepped back instead of filling in.
    trailing_skipped = int(len(usable_by_time) - 1 - end)
    # NaN counts are reported per variable for the hours that were considered,
    # so a caller can see what was missing without it being silently dropped.
    nulls = {
        variable: int(frame[variable].isna().sum())
        for variable in LIVE_VARIABLES
    }
    report = {
        "feature_timestamp_utc": _iso(feature_timestamp),
        "history_hours_used": int(len(window) - 1),
        "required_preceding_hours": int(min_preceding_hours),
        "history_window_start_utc": _iso(window["timestamp_utc"].iloc[0]),
        "hours_returned_by_source": total_hours_returned,
        "hours_considered_at_or_before_now": int(len(usable_by_time)),
        "future_hours_discarded": future_hours_discarded,
        "trailing_hours_skipped": trailing_skipped,
        "trailing_hours_skipped_reason": (
            None
            if trailing_skipped == 0
            else (
                "the newest hour(s) at or before the reference time were incomplete "
                "(a required variable was missing) or did not continue the hourly "
                "sequence; the engine moved back to the newest complete hour rather "
                "than filling any value in"
            )
        ),
        "null_counts_in_raw_frame": nulls,
        "contiguity": "window rows are exactly one hour apart, verified",
        "values_imputed": 0,
    }
    return window, report


# --------------------------------------------------------------------------
# Feature construction (delegated to Phase 4 code)
# --------------------------------------------------------------------------


def build_feature_row(window: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """Compute the 28 features for the newest hour in ``window``.

    Calls the Phase 4 ``add_features`` on the live window, then takes its last
    row. A second, independent check recomputes the same features from a
    truncated window containing only the minimum required history and requires
    bit-equality: that proves the value at t depends on the last
    ``MIN_PRECEDING_HOURS + 1`` hours and on nothing else -- in particular not on
    any hour after t and not on how much extra history happened to be returned.
    """
    phase4 = phase4_feature_module()
    columns = list(phase4.FEATURE_COLUMNS)

    computed = phase4.add_features(window)
    row = computed.iloc[-1]
    vector = pd.Series(
        {name: float(row[name]) for name in columns},
        index=columns,
        dtype=float,
    )

    tail = window.tail(MIN_PRECEDING_HOURS + 1).reset_index(drop=True)
    tail_computed = phase4.add_features(tail)
    tail_row = tail_computed.iloc[-1]
    deltas = {name: abs(float(tail_row[name]) - float(vector[name])) for name in columns}
    worst = max(deltas.values()) if deltas else 0.0
    history_independent = worst <= 1e-12

    non_finite = [name for name in columns if not np.isfinite(vector[name])]
    report = {
        "feature_source": "dataset/feature_engineering_phase4.py::add_features (imported, not reimplemented)",
        "n_features": len(columns),
        "non_finite_features": non_finite,
        "values_imputed": 0,
        "history_independence_check": {
            "method": (
                "recompute the feature row from only the last "
                f"{MIN_PRECEDING_HOURS + 1} hours and compare with the row built from "
                "the full returned history"
            ),
            "max_abs_difference": float(worst),
            "tolerance": 1e-12,
            "verdict": "PASS" if history_independent else "FAIL",
        },
    }
    if non_finite:
        raise ThunderstormNowcastError(
            "non_finite_features",
            "Feature construction produced NaN or infinite values, so no honest "
            "prediction can be made from this input.",
            {"features": non_finite, "window_rows": int(len(window))},
        )
    if not history_independent:
        raise ThunderstormNowcastError(
            "feature_history_dependence",
            "The feature row at hour t changed when earlier hours were removed, so "
            "it does not depend on the trailing window alone.",
            {"max_abs_difference": float(worst)},
        )
    return vector, report


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def classify(probability: float, threshold: float) -> int:
    """Apply the locked threshold. ``>=`` matches the Phase 6 convention."""
    return int(probability >= threshold)


def predict_current_thunderstorm_risk(
    *,
    frame: pd.DataFrame | None = None,
    now: Any = None,
    model: NowcastModel | None = None,
    models_dir: str | Path | None = None,
    outputs_dir: str | Path | None = None,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    max_data_age_hours: float | None = DEFAULT_MAX_DATA_AGE_HOURS,
    latitude: float = TRAINING_REQUEST_LATITUDE,
    longitude: float = TRAINING_REQUEST_LONGITUDE,
) -> dict[str, Any]:
    """Produce one 1-hour thunderstorm nowcast for the latest usable hour.

    Parameters
    ----------
    frame
        Atmospheric frame with ``timestamp_utc`` (or ``date``) plus the seven
        variables. ``None`` fetches live data from Open-Meteo. Supplying a frame
        is the supported way to replay a historical hour or to run without
        network access; the feature, threshold and scoring path is identical.
    now
        Reference instant. Hours after it are discarded, so it is the guard that
        keeps future data out of the features. Defaults to the current UTC time.
        Supplying it also pins ``prediction_timestamp``, which makes the whole
        result deterministic for repeated identical input.
    model
        A bundle returned by :func:`load_model_bundle`. Pass it to serve many
        predictions without re-reading the 144 MB artifact per call; ``None``
        loads it from ``models_dir``. Either way the artifact is read-only.
    max_data_age_hours
        Refuse to present a nowcast built on data older than this many hours
        (``stale_data``). ``None`` disables the guard, which is the correct
        setting for a historical replay.

    Returns
    -------
    dict
        Probability, predicted class, risk label, threshold, lead time,
        prediction and feature timestamps, source provenance, feature
        completeness and the explicit separation of probability / threshold /
        label / unavailable observation.

    Raises
    ------
    ThunderstormNowcastError
        On any missing, malformed, unit-inconsistent, stale or non-finite input.
        A failure never returns a probability.
    """
    reference = as_utc_timestamp(now, field="now") if now is not None else utc_now()
    replay_mode = now is not None

    artifact = model if model is not None else load_model_bundle(models_dir)

    if frame is None:
        live_frame, provenance = fetch_live_atmospheric_data(
            latitude=latitude, longitude=longitude, timeout_seconds=timeout_seconds
        )
        frame_origin = "live_open_meteo_forecast_endpoint"
    else:
        live_frame = normalise_live_frame(frame)
        frame_is_empty = bool(live_frame.empty)
        provenance = {
            "provider": "caller-supplied frame",
            "product": "not a live request",
            "endpoint": None,
            "requested_at_utc": None,
            "hours_returned": int(len(live_frame)),
            # An empty frame has no window bounds; the window is reported as null
            # rather than indexed, so the failure below stays a typed error.
            "window_start_utc": None if frame_is_empty else _iso(live_frame["timestamp_utc"].iloc[0]),
            "window_end_utc": None if frame_is_empty else _iso(live_frame["timestamp_utc"].iloc[-1]),
            "requested_grid_point": {"latitude": latitude, "longitude": longitude},
            "served_grid_cell": None,
            "training_grid_cell": TRAINING_SERVED_GRID_CELL,
            "hourly_units": dict(EXPECTED_UNITS),
            "ground_truth_note": GROUND_TRUTH_NOTE,
            "domain_shift_note": EXTERNAL_DOMAIN_SHIFT_NOTE,
        }
        frame_origin = "caller_supplied_frame"

    window, selection = select_latest_feature_window(live_frame, now=reference)

    feature_timestamp = as_utc_timestamp(
        selection["feature_timestamp_utc"], field="feature_timestamp_utc"
    )
    target_timestamp = feature_timestamp + pd.Timedelta(hours=artifact.lead_time_hours)

    data_age_hours = float((reference - feature_timestamp) / pd.Timedelta(hours=1))
    # The target hour begins at t + lead. If that instant has already passed,
    # the hour being predicted is no longer wholly in the future.
    target_hour_has_begun = bool(reference >= target_timestamp)

    if max_data_age_hours is not None and data_age_hours > float(max_data_age_hours):
        raise ThunderstormNowcastError(
            "stale_data",
            f"The newest usable atmospheric hour is {data_age_hours:.2f} h behind the "
            f"reference time, beyond the {float(max_data_age_hours):.2f} h freshness "
            "limit. Refusing to present stale data as a current nowcast.",
            {
                "feature_timestamp_utc": _iso(feature_timestamp),
                "reference_time_utc": _iso(reference),
                "data_age_hours": data_age_hours,
                "max_data_age_hours": float(max_data_age_hours),
            },
        )

    vector, feature_report = build_feature_row(window)

    matrix = vector.to_numpy(dtype=float).reshape(1, -1)
    if not np.isfinite(matrix).all():
        raise ThunderstormNowcastError(
            "non_finite_features",
            "The feature vector passed to the model contains NaN or infinite values.",
        )

    probabilities = artifact.model.predict_proba(matrix)
    probability = float(probabilities[0, artifact.positive_class_index])
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ThunderstormNowcastError(
            "probability_out_of_range",
            "The model returned a value outside [0, 1].",
            probability,
        )

    predicted_class = classify(probability, artifact.threshold)

    feature_completeness = {
        "n_features_expected": len(artifact.feature_names),
        "n_features_provided": int(vector.shape[0]),
        "n_features_missing": 0,
        "missing_feature_names": [],
        "all_features_finite": bool(np.isfinite(matrix).all()),
        "feature_order_matches_training_metadata": list(vector.index) == list(artifact.feature_names),
        "history_hours_used": selection["history_hours_used"],
        "required_preceding_hours": selection["required_preceding_hours"],
        "history_sufficient": selection["history_hours_used"] >= selection["required_preceding_hours"],
        "raw_values_missing_in_window": 0,
        "values_imputed": 0,
    }

    feature_vector = {name: float(vector[name]) for name in artifact.feature_names}
    raw_conditions = {
        variable: float(window[variable].iloc[-1]) for variable in LIVE_VARIABLES
    }

    return {
        # -- the four quantities, kept apart ------------------------------
        "probability": probability,
        "probability_semantics": PROBABILITY_SEMANTICS,
        "threshold": artifact.threshold,
        "threshold_semantics": THRESHOLD_SEMANTICS,
        "predicted_class": predicted_class,
        "risk_label": RISK_LABELS[predicted_class],
        "risk_level": RISK_LEVELS[predicted_class],
        "risk_labels": {str(key): value for key, value in RISK_LABELS.items()},
        "risk_label_semantics": RISK_LABEL_SEMANTICS,
        "observed_outcome": {
            "available": False,
            "target_timestamp_utc": _iso(target_timestamp),
            "reason": ACTUAL_OBSERVATION_SEMANTICS,
        },
        "quantity_separation": {
            "probability": "model score for hour t + 1 h (not confidence, not calibrated)",
            "threshold": f"locked operational cut-off ({artifact.threshold})",
            "risk_label": f"alert from probability >= threshold ({RISK_LABELS[predicted_class]})",
            "actual_observation": "unavailable by construction for a future hour",
        },
        # -- timing -------------------------------------------------------
        "lead_time_hours": artifact.lead_time_hours,
        "prediction_timestamp": _iso(reference),
        "feature_timestamp": _iso(feature_timestamp),
        "target_timestamp": _iso(target_timestamp),
        "temporal_semantics": {
            "feature_hour": "t -- latest hour whose atmospheric inputs are available",
            "predicted_hour": "t + 1 h -- the clock hour the probability refers to",
            "lead_time_hours": artifact.lead_time_hours,
            "features_use_only_hours_at_or_before": "t",
            "future_atmospheric_values_used": False,
            "future_hours_discarded": selection["future_hours_discarded"],
            "target_hour_is_in_the_future": bool(reference < target_timestamp),
            "target_hour_has_begun": target_hour_has_begun,
            "data_age_hours": data_age_hours,
            "note": (
                "The model is a lead-1 h nowcast. The atmospheric inputs describe "
                "hour t; the thunderstorm observation being predicted belongs to "
                "hour t + 1 h and does not exist yet."
            ),
        },
        # -- trust / provenance -------------------------------------------
        "model": {
            "name": MODEL_NAME,
            "version": artifact.bundle_version,
            "file": artifact.model_path.name,
            "type": (
                (artifact.metadata.get("model") or {}).get("type")
                if isinstance(artifact.metadata.get("model"), Mapping)
                else None
            )
            or EXPECTED_MODEL_TYPE_RECORDED,
            "estimator_class": type(artifact.model).__name__,
            "estimator_qualified_name": artifact.model_type,
            "n_features": len(artifact.feature_names),
            "feature_names": list(artifact.feature_names),
            "lead_time_hours": artifact.lead_time_hours,
            "target_column": artifact.target_column,
            "decision_threshold": artifact.threshold,
            "trained_at_utc": artifact.trained_at_utc,
            "training_dataset_sha256": artifact.training_dataset_sha256,
            "random_state": artifact.metadata.get("random_seed"),
            "phase": artifact.metadata.get("phase"),
            "positive_class_meaning": artifact.metadata.get("target", {}).get("definition")
            if isinstance(artifact.metadata.get("target"), Mapping)
            else None,
        },
        "source": {
            "frame_origin": frame_origin,
            "replay_mode": replay_mode,
            "provenance": provenance,
        },
        "input_atmospheric_conditions": {
            variable: {
                "value": raw_conditions[variable],
                "unit": EXPECTED_UNITS[variable],
                "timestamp_utc": _iso(feature_timestamp),
            }
            for variable in LIVE_VARIABLES
        },
        "feature_completeness": feature_completeness,
        "features_used_for_prediction": feature_vector,
        "features": feature_report,
        "selection": selection,
        "measured_skill_reference": measured_skill_reference(outputs_dir),
        "leakage_guard": {
            "weather_code_requested": False,
            "weather_code_used_as_feature": False,
            "target_columns_used_as_features": False,
            "future_atmospheric_variables_used": False,
            "target_derived_statistics_used": False,
        },
        "freshness_policy": {
            "max_data_age_hours": max_data_age_hours,
            "data_age_hours": data_age_hours,
            "status": "within_limit" if max_data_age_hours is None else (
                "within_limit" if data_age_hours <= float(max_data_age_hours) else "too_stale"
            ),
        },
        "location": {
            "requested_grid_point": {"latitude": latitude, "longitude": longitude},
            "training_grid_cell": TRAINING_SERVED_GRID_CELL,
        },
        "disclaimer": DISCLAIMER,
        "scientific_disclaimer": artifact.metadata.get("scientific_disclaimer"),
        "ground_truth_note": GROUND_TRUTH_NOTE,
        "domain_shift_note": EXTERNAL_DOMAIN_SHIFT_NOTE,
        "generated_at_utc": utc_now().isoformat(),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_frame_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ThunderstormNowcastError(
            "frame_file_missing",
            f"The supplied frame CSV does not exist: {path}",
            str(path),
        )
    try:
        return pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise ThunderstormNowcastError(
            "frame_file_unreadable",
            f"The supplied frame CSV could not be read: {type(exc).__name__}: {exc}",
            str(path),
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Produce one 1-hour thunderstorm nowcast from the latest available "
            "atmospheric hour, using the locked Phase 6 model and threshold."
        )
    )
    parser.add_argument(
        "--frame-csv",
        type=Path,
        default=None,
        help=(
            "Score a supplied CSV instead of fetching live data. Needs "
            "timestamp_utc (or date) plus the seven model variables."
        ),
    )
    parser.add_argument(
        "--now",
        default=None,
        help="Reference instant in UTC (ISO 8601). Hours after it are discarded.",
    )
    parser.add_argument(
        "--max-data-age-hours",
        type=float,
        default=DEFAULT_MAX_DATA_AGE_HOURS,
        help=(
            "Refuse to predict when the newest usable hour is older than this "
            f"(default {DEFAULT_MAX_DATA_AGE_HOURS}). Use -1 to disable."
        ),
    )
    parser.add_argument("--models-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    max_age: float | None = args.max_data_age_hours
    if max_age is not None and max_age < 0:
        max_age = None

    frame = _load_frame_csv(args.frame_csv) if args.frame_csv else None

    try:
        result = predict_current_thunderstorm_risk(
            frame=frame,
            now=args.now,
            models_dir=args.models_dir,
            max_data_age_hours=max_age,
        )
    except ThunderstormNowcastError as error:
        log(
            json.dumps(
                {
                    "error": True,
                    "status": "error",
                    "code": error.code,
                    "message": error.message,
                    "detail": error.detail,
                    "note": (
                        "No prediction was produced. No atmospheric value or "
                        "probability has been fabricated."
                    ),
                },
                indent=2,
                default=str,
            )
        )
        return 1

    log(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
