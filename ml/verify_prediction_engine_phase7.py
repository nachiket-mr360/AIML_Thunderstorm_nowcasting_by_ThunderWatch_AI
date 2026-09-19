"""Phase 7 -- INDEPENDENT VERIFICATION OF THE REAL-TIME PREDICTION ENGINE.

This script does not import the engine's own test helpers or trust its
self-description. It re-derives what it can from the Phase 1-6 artifacts and
checks the engine against them:

  * the 28 production features are compared, value by value, against the shipped
    Phase 4 feature table, which was built before this engine existed;
  * the engine's probability is compared against ``model.predict_proba`` applied
    to the Phase 5 training table's own feature columns for the same hour, which
    cross-validates Phase 4 -> Phase 5 -> engine feature ordering at once;
  * the locked threshold is re-read from the Phase 6 artifact and the Phase 6
    evaluation JSON, not from this script;
  * every failure path is exercised with deliberately broken input to prove the
    engine refuses rather than fabricating a value;
  * the Phase 1-6 protected artifacts are hashed before and after the engine
    runs, so "the engine is read-only" is measured rather than asserted.

Run:  python ml/verify_prediction_engine_phase7.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml import predict_thunderstorm_nowcast as engine  # noqa: E402
from ml.predict_thunderstorm_nowcast import (  # noqa: E402
    LIVE_VARIABLES,
    RISK_LABELS,
    RISK_LEVELS,
    ThunderstormNowcastError,
    classify,
    load_model_bundle,
    predict_current_thunderstorm_risk,
)

DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

GRID_CSV = DATASET_DIR / "raw_openmeteo" / "votv_openmeteo_hourly_2014_2025.csv"
PHASE4_CSV = DATASET_DIR / "votv_thunderstorm_features_2014_2025.csv"
PHASE4_META = DATASET_DIR / "votv_thunderstorm_features_2014_2025_metadata.json"
PHASE5_CSV = DATASET_DIR / "votv_thunderstorm_nowcast_2014_2025.csv"
MODEL_FILE = MODELS_DIR / "thunderstorm_nowcast_1h.joblib"
MODEL_META = MODELS_DIR / "thunderstorm_nowcast_1h_metadata.json"
EVAL_JSON = OUTPUTS_DIR / "thunderstorm_nowcast_1h_evaluation.json"

#: The Phase 1-6 artifacts the engine must leave byte-identical. This is the same
#: protected set Phase 6 defines, so "protected" means one thing in this project.
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
    "dataset/feature_engineering_phase4.py",
    "dataset/nowcast_targets_phase5.py",
    "dataset/verify_nowcast_targets_phase5.py",
    "dataset/build_synchronized_dataset_phase3.py",
    "models/storm_risk_model.joblib",
    "models/model_metadata.json",
    "models/thunderstorm_nowcast_1h.joblib",
    "models/thunderstorm_nowcast_1h_metadata.json",
    "models/thunderstorm_nowcast_2h.joblib",
    "models/thunderstorm_nowcast_2h_metadata.json",
    "models/thunderstorm_nowcast_3h.joblib",
    "models/thunderstorm_nowcast_3h_metadata.json",
    "outputs/thunderstorm_nowcast_1h_evaluation.json",
    "outputs/thunderstorm_nowcast_2h_evaluation.json",
    "outputs/thunderstorm_nowcast_3h_evaluation.json",
    "outputs/thunderstorm_nowcast_1h_feature_importance.csv",
    "outputs/thunderstorm_nowcast_2h_feature_importance.csv",
    "outputs/thunderstorm_nowcast_3h_feature_importance.csv",
    "outputs/thunderstorm_nowcast_1h_confusion_matrix.png",
    "outputs/thunderstorm_nowcast_2h_confusion_matrix.png",
    "outputs/thunderstorm_nowcast_3h_confusion_matrix.png",
    "outputs/thunderstorm_nowcast_model_comparison.json",
    "outputs/PHASE6_MODEL_TRAINING_REPORT.md",
    "ml/train_model.py",
    "ml/predict.py",
    "ml/train_thunderstorm_nowcast_phase6.py",
    "app.py",
]

SEED = 20250915
SAMPLE_ROWS = 10
WINDOW_HOURS = 240

CHECKS: list[tuple[str, bool, str]] = []


def check(ok: bool, label: str, note: str = "") -> bool:
    CHECKS.append((label, bool(ok), note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {note}" if note else ""), flush=True)
    return bool(ok)


def section(title: str) -> None:
    print(f"\n{title}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(paths: list[str]) -> dict[str, str]:
    out = {}
    for rel in paths:
        path = PROJECT_ROOT / rel
        out[rel] = sha256_file(path) if path.is_file() else "MISSING"
    return out


def expect_error(code_expected: str | None, label: str, **kwargs) -> bool:
    """The engine must raise a typed error and must not return a probability."""
    try:
        result = predict_current_thunderstorm_risk(**kwargs)
    except ThunderstormNowcastError as error:
        ok_codes = code_expected is None or error.code == code_expected
        leaked = "probability" in (error.detail or {}) if isinstance(error.detail, dict) else False
        return check(
            ok_codes and not leaked,
            label,
            f"code={error.code} (expected {code_expected})",
        )
    except Exception as error:  # noqa: BLE001
        return check(False, label, f"wrong exception type {type(error).__name__}: {error}")
    return check(
        False,
        label,
        f"engine returned a prediction instead of failing: probability="
        f"{result.get('probability')!r}",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid = pd.read_csv(GRID_CSV)
    grid = grid.rename(columns={"date": "timestamp_utc"})
    grid["timestamp_utc"] = pd.to_datetime(grid["timestamp_utc"], utc=True, format="ISO8601")
    grid = grid.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)

    phase4 = pd.read_csv(PHASE4_CSV)
    phase4["timestamp_utc"] = pd.to_datetime(
        phase4["timestamp_utc"], utc=True, format="ISO8601"
    )

    phase5 = pd.read_csv(PHASE5_CSV)
    phase5["timestamp_utc"] = pd.to_datetime(
        phase5["timestamp_utc"], utc=True, format="ISO8601"
    )
    return grid, phase4, phase5


def main() -> int:
    print("=" * 80)
    print("PHASE 7 -- INDEPENDENT VERIFICATION OF THE REAL-TIME PREDICTION ENGINE")
    print("=" * 80)
    print(f"python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}")

    grid, phase4, phase5 = load_inputs()
    phase4_meta = json.loads(PHASE4_META.read_text(encoding="utf-8"))
    model_meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
    evaluation = json.loads(EVAL_JSON.read_text(encoding="utf-8"))

    before_protected = snapshot(PROTECTED_FILES)

    # -----------------------------------------------------------------------
    section("[1] Model artifact loads and the engine agrees with Phase 6 metadata")
    bundle = load_model_bundle(MODELS_DIR)
    check(bundle is not None, "the 1 h model artifact loads through the engine loader")
    check(
        type(bundle.model).__name__ == "RandomForestClassifier",
        "the loaded estimator is a RandomForestClassifier",
        type(bundle.model).__name__,
    )
    params = (model_meta.get("model") or {}).get("params") or {}
    check(
        bundle.model.n_estimators == params.get("n_estimators") == 400,
        "n_estimators is 400 as Phase 6 recorded",
        str(bundle.model.n_estimators),
    )
    check(
        bundle.model.class_weight == params.get("class_weight") == "balanced",
        "class_weight is 'balanced' as Phase 6 recorded",
        str(bundle.model.class_weight),
    )
    check(
        bundle.model.random_state == params.get("random_state") == 42,
        "random_state is 42 as Phase 6 recorded",
        str(bundle.model.random_state),
    )
    check(
        bundle.lead_time_hours == 1,
        "the artifact declares a 1-hour lead time",
        str(bundle.lead_time_hours),
    )
    check(
        bundle.target_column == "target_1h",
        "the artifact targets target_1h",
        bundle.target_column,
    )
    check(
        bundle.threshold == 0.0775,
        "the locked threshold is 0.0775, unchanged",
        repr(bundle.threshold),
    )
    check(
        abs(float(evaluation["metrics"]["test"]["threshold"]) - bundle.threshold) < 1e-12,
        "the locked threshold equals the threshold used in the Phase 6 test evaluation",
        f"bundle={bundle.threshold} evaluation={evaluation['metrics']['test']['threshold']}",
    )

    section("[2] Exactly 28 features, in the Phase 4 order")
    phase4_features = list(phase4_meta["features"]["feature_order"])
    check(
        len(bundle.feature_names) == 28,
        "the model uses exactly 28 features",
        str(len(bundle.feature_names)),
    )
    check(
        list(bundle.feature_names) == engine.feature_columns(),
        "engine feature list equals dataset/feature_engineering_phase4.py FEATURE_COLUMNS",
    )
    check(
        list(bundle.feature_names) == phase4_features,
        "feature order matches the Phase 4 metadata feature_order exactly",
    )
    check(
        list(bundle.feature_names) == list(model_meta["features"]["feature_order"]),
        "feature order matches the Phase 6 metadata feature_order exactly",
    )
    leaked = [
        name
        for name in bundle.feature_names
        if name == "weather_code" or "target" in name.lower() or "label" in name.lower()
    ]
    check(not leaked, "no weather_code, target or label column is a feature", str(leaked))

    # -----------------------------------------------------------------------
    section("[3] Production features reproduce the Phase 4 table exactly")
    rng = np.random.default_rng(SEED)
    candidates = phase4.index.to_numpy()
    picks = np.sort(rng.choice(candidates, size=SAMPLE_ROWS, replace=False))

    worst_overall = 0.0
    worst_name = ""
    worst_time = None
    mismatched_rows = 0
    probabilities: list[float] = []

    for position in picks:
        row = phase4.iloc[int(position)]
        stamp = row["timestamp_utc"]
        window = grid[grid["timestamp_utc"] <= stamp].tail(WINDOW_HOURS).reset_index(drop=True)
        result = predict_current_thunderstorm_risk(
            frame=window, now=stamp, model=bundle, max_data_age_hours=None
        )
        probabilities.append(result["probability"])
        produced = result["features_used_for_prediction"]
        row_worst = 0.0
        for name in phase4_features:
            delta = abs(float(produced[name]) - float(row[name]))
            if delta > row_worst:
                row_worst = delta
            if delta > worst_overall:
                worst_overall, worst_name, worst_time = delta, name, stamp
        if row_worst > 1e-12:
            mismatched_rows += 1

    check(
        mismatched_rows == 0,
        f"all 28 features match the shipped Phase 4 table on {SAMPLE_ROWS} sampled hours",
        f"max abs difference={worst_overall:.3e} (worst: {worst_name} @ {worst_time})",
    )
    check(
        worst_overall <= 1e-12,
        "the largest feature difference is within the 1e-12 tolerance Phase 4 itself uses",
        f"{worst_overall:.3e}",
    )

    # The features must not depend on how much history is supplied, which is what
    # makes a live 3-day fetch equivalent to the multi-year training grid.
    stamp = phase4.iloc[int(picks[0])]["timestamp_utc"]
    full_window = grid[grid["timestamp_utc"] <= stamp].tail(WINDOW_HOURS).reset_index(drop=True)
    short = predict_current_thunderstorm_risk(
        frame=grid[grid["timestamp_utc"] <= stamp].tail(7).reset_index(drop=True),
        now=stamp,
        model=bundle,
        max_data_age_hours=None,
    )
    full = predict_current_thunderstorm_risk(
        frame=full_window,
        now=stamp,
        model=bundle,
        max_data_age_hours=None,
    )
    same = all(
        abs(short["features_used_for_prediction"][n] - full["features_used_for_prediction"][n]) <= 1e-12
        for n in phase4_features
    )
    check(
        same and abs(short["probability"] - full["probability"]) <= 1e-15,
        "7 hours of history and 240 hours of history give the same features and probability",
    )

    # A frame read from CSV carries string timestamps. If that path were not
    # parsed, sorting would be lexicographic and every lag would silently
    # misalign, so the CSV round-trip must reproduce the in-memory result.
    import io

    csv_buffer = io.StringIO()
    full_window.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    from_csv = pd.read_csv(csv_buffer)
    csv_result = predict_current_thunderstorm_risk(
        frame=from_csv, now=stamp, model=bundle, max_data_age_hours=None
    )
    in_memory = predict_current_thunderstorm_risk(
        frame=full_window, now=stamp, model=bundle, max_data_age_hours=None
    )
    check(
        csv_result["probability"] == in_memory["probability"]
        and csv_result["feature_timestamp"] == in_memory["feature_timestamp"],
        "a CSV round-trip (string timestamps) gives the same prediction as the in-memory frame",
        f"csv={csv_result['probability']:.6f} memory={in_memory['probability']:.6f}",
    )

    # -----------------------------------------------------------------------
    section("[4] The engine reproduces the Phase 6 model on the Phase 5 table")
    phase5_by_time = phase5.set_index("timestamp_utc")
    agree = 0
    compared = 0
    prob_worst = 0.0
    class_mismatch = 0
    for position in picks:
        stamp = phase4.iloc[int(position)]["timestamp_utc"]
        if stamp not in phase5_by_time.index:
            continue
        reference_row = phase5_by_time.loc[stamp]
        reference_matrix = np.array(
            [[float(reference_row[name]) for name in phase4_features]], dtype=float
        )
        reference_probability = float(
            bundle.model.predict_proba(reference_matrix)[0, bundle.positive_class_index]
        )
        window = grid[grid["timestamp_utc"] <= stamp].tail(WINDOW_HOURS).reset_index(drop=True)
        result = predict_current_thunderstorm_risk(
            frame=window, now=stamp, model=bundle, max_data_age_hours=None
        )
        compared += 1
        delta = abs(result["probability"] - reference_probability)
        prob_worst = max(prob_worst, delta)
        if delta <= 1e-15:
            agree += 1
        if result["predicted_class"] != classify(reference_probability, bundle.threshold):
            class_mismatch += 1

    check(
        compared > 0 and agree == compared,
        "engine probability equals predict_proba on the Phase 5 feature columns, for every sampled hour",
        f"{agree}/{compared} exact, max difference={prob_worst:.3e}",
    )
    check(
        class_mismatch == 0,
        "engine class assignment matches the locked-threshold rule on the Phase 5 features",
    )

    # -----------------------------------------------------------------------
    section("[5] Probability range, threshold rule and risk labels")
    check(
        all(0.0 <= p <= 1.0 and np.isfinite(p) for p in probabilities),
        "every produced probability lies in [0, 1] and is finite",
        f"n={len(probabilities)} min={min(probabilities):.6f} max={max(probabilities):.6f}",
    )
    boundary = [
        (bundle.threshold - 1e-9, 0),
        (bundle.threshold, 1),
        (bundle.threshold + 1e-9, 1),
        (0.0, 0),
        (1.0, 1),
    ]
    check(
        all(classify(p, bundle.threshold) == expected for p, expected in boundary),
        "classify() applies probability >= threshold consistently at the boundary",
        str([(round(p, 6), classify(p, bundle.threshold)) for p, _ in boundary]),
    )
    sample = predict_current_thunderstorm_risk(
        frame=grid[grid["timestamp_utc"] <= phase4.iloc[int(picks[0])]["timestamp_utc"]]
        .tail(WINDOW_HOURS)
        .reset_index(drop=True),
        now=phase4.iloc[int(picks[0])]["timestamp_utc"],
        model=bundle,
        max_data_age_hours=None,
    )
    check(
        sample["predicted_class"] == classify(sample["probability"], bundle.threshold),
        "predicted_class equals probability compared with the locked threshold",
        f"p={sample['probability']:.6f} -> {sample['predicted_class']}",
    )
    check(
        sample["risk_label"] == RISK_LABELS[sample["predicted_class"]]
        and sample["risk_level"] == RISK_LEVELS[sample["predicted_class"]],
        "risk_label and risk_level follow from predicted_class",
        f"{sample['risk_level']} / {sample['risk_label']}",
    )

    # -----------------------------------------------------------------------
    section("[6] No future information reaches the features")
    stamp = phase4.iloc[int(picks[1])]["timestamp_utc"]
    base_window = grid[grid["timestamp_utc"] <= stamp].tail(WINDOW_HOURS).reset_index(drop=True)
    base = predict_current_thunderstorm_risk(
        frame=base_window, now=stamp, model=bundle, max_data_age_hours=None
    )

    # Same frame, but extended with hours strictly after t whose values are wildly
    # different. If anything at t read a later hour, these would move it.
    future = grid[
        (grid["timestamp_utc"] > stamp) & (grid["timestamp_utc"] <= stamp + pd.Timedelta(hours=48))
    ].copy()
    for variable in LIVE_VARIABLES:
        future[variable] = 999.0
    extended = pd.concat([base_window, future], ignore_index=True)
    extended_result = predict_current_thunderstorm_risk(
        frame=extended, now=stamp, model=bundle, max_data_age_hours=None
    )
    identical = all(
        abs(extended_result["features_used_for_prediction"][n] - base["features_used_for_prediction"][n])
        <= 1e-15
        for n in phase4_features
    )
    check(
        identical and extended_result["probability"] == base["probability"],
        "corrupting all hours after t leaves the feature vector and probability unchanged",
        f"future hours injected={len(future)}",
    )
    check(
        extended_result["selection"]["future_hours_discarded"] == len(future),
        "hours after the reference instant are discarded, not used",
        f"discarded={extended_result['selection']['future_hours_discarded']}",
    )
    check(
        base["temporal_semantics"]["future_atmospheric_values_used"] is False
        and base["leakage_guard"]["future_atmospheric_variables_used"] is False,
        "the engine declares no future atmospheric values were used",
    )
    check(
        base["features"]["history_independence_check"]["verdict"] == "PASS",
        "the engine's own truncation check confirms features depend only on the trailing window",
        f"max_abs_difference={base['features']['history_independence_check']['max_abs_difference']:.3e}",
    )
    check(
        base["leakage_guard"]["weather_code_used_as_feature"] is False
        and "weather_code" not in base["features_used_for_prediction"],
        "weather_code is neither requested nor used as a feature",
    )

    # -----------------------------------------------------------------------
    section("[7] Missing, malformed or stale live data fails safely")
    healthy = grid[grid["timestamp_utc"] <= stamp].tail(WINDOW_HOURS).reset_index(drop=True)

    expect_error(
        "no_recent_hours",
        "an empty frame fails",
        frame=healthy.iloc[0:0],
        now=stamp,
        model=bundle,
    )
    expect_error(
        "insufficient_recent_history",
        "fewer than 7 contiguous hours fails",
        frame=healthy.tail(5).reset_index(drop=True),
        now=stamp,
        model=bundle,
        max_data_age_hours=None,
    )

    # A hole in the newest hour must not be filled in. The documented behaviour is
    # to step back to the newest *complete* hour and say so, so that is asserted
    # directly: an imputed value would instead keep the feature hour at t.
    holed = healthy.copy()
    holed.loc[holed.index[-1], "temperature_2m"] = np.nan
    stepped_back = predict_current_thunderstorm_risk(
        frame=holed, now=stamp, model=bundle, max_data_age_hours=None
    )
    check(
        pd.Timestamp(stepped_back["feature_timestamp"]) == stamp - pd.Timedelta(hours=1),
        "a NaN in the newest hour is not imputed: the engine uses the newest complete hour",
        f"feature hour moved {stamp.isoformat()} -> {stepped_back['feature_timestamp']}",
    )
    check(
        stepped_back["selection"]["trailing_hours_skipped"] == 1
        and stepped_back["selection"]["values_imputed"] == 0
        and stepped_back["feature_completeness"]["values_imputed"] == 0,
        "the skipped incomplete hour is reported and nothing was imputed",
        f"trailing_hours_skipped={stepped_back['selection']['trailing_hours_skipped']}",
    )
    check(
        abs(stepped_back["temporal_semantics"]["data_age_hours"] - 1.0) < 1e-9,
        "stepping back one hour is reflected in the reported data age",
        f"{stepped_back['temporal_semantics']['data_age_hours']}",
    )
    gapped = healthy.copy()
    gapped.loc[gapped.index[-1], "temperature_2m"] = np.nan
    expect_error(
        "stale_data",
        "an incomplete newest hour combined with a tight freshness limit fails",
        frame=gapped,
        now=stamp + pd.Timedelta(hours=4),
        model=bundle,
        max_data_age_hours=3.0,
    )
    expect_error(
        "frame_missing_variables",
        "a frame missing a required variable fails",
        frame=healthy.drop(columns=["cloud_cover"]),
        now=stamp,
        model=bundle,
        max_data_age_hours=None,
    )
    expect_error(
        "frame_duplicate_timestamps",
        "duplicate hourly timestamps fail",
        frame=pd.concat([healthy, healthy.tail(1)], ignore_index=True),
        now=stamp,
        model=bundle,
        max_data_age_hours=None,
    )
    expect_error(
        "no_recent_hours",
        "a frame entirely in the future fails",
        frame=healthy.assign(timestamp_utc=healthy["timestamp_utc"] + pd.Timedelta(days=365)),
        now=stamp,
        model=bundle,
        max_data_age_hours=None,
    )
    expect_error(
        "stale_data",
        "data older than the freshness limit fails",
        frame=healthy,
        now=stamp + pd.Timedelta(hours=48),
        model=bundle,
        max_data_age_hours=3.0,
    )
    expect_error(
        "frame_not_a_dataframe",
        "a non-frame object fails",
        frame="not a dataframe",
        now=stamp,
        model=bundle,
        max_data_age_hours=None,
    )
    expect_error(
        "frame_missing_timestamp",
        "a frame with no time column fails",
        frame=healthy.drop(columns=["timestamp_utc"]),
        now=stamp,
        model=bundle,
        max_data_age_hours=None,
    )

    # Unit validation and upstream failures are tested without needing the network.
    try:
        engine._validate_units({k: "bogus" for k in engine.EXPECTED_UNITS})
        check(False, "a unit mismatch is rejected", "no error raised")
    except ThunderstormNowcastError as error:
        check(
            error.code == "unexpected_units",
            "a unit mismatch is rejected rather than silently rescaled",
            f"code={error.code}",
        )

    real_url = engine.OPEN_METEO_URL
    engine.OPEN_METEO_URL = "https://invalid.invalid/nonexistent-open-meteo-endpoint"
    try:
        expect_error(
            None,
            "an unreachable provider raises a typed upstream error",
            frame=None,
            now=None,
            model=bundle,
            max_data_age_hours=None,
            timeout_seconds=5.0,
        )
    finally:
        engine.OPEN_METEO_URL = real_url

    # -----------------------------------------------------------------------
    section("[8] Repeated identical input produces identical output")
    repeats = [
        predict_current_thunderstorm_risk(
            frame=base_window, now=stamp, model=bundle, max_data_age_hours=None
        )
        for _ in range(3)
    ]
    stable = all(
        json.dumps({k: v for k, v in r.items() if k != "generated_at_utc"}, sort_keys=True, default=str)
        == json.dumps(
            {k: v for k, v in repeats[0].items() if k != "generated_at_utc"},
            sort_keys=True,
            default=str,
        )
        for r in repeats[1:]
    )
    check(stable, "three calls with identical input are byte-identical apart from generated_at_utc")

    # -----------------------------------------------------------------------
    section("[9] Lead-time and observation semantics")
    check(base["lead_time_hours"] == 1, "lead_time_hours is 1")
    feature_ts = pd.Timestamp(base["feature_timestamp"])
    target_ts = pd.Timestamp(base["target_timestamp"])
    prediction_ts = pd.Timestamp(base["prediction_timestamp"])
    check(
        target_ts - feature_ts == pd.Timedelta(hours=1),
        "target_timestamp is exactly one hour after feature_timestamp",
        f"{feature_ts.isoformat()} -> {target_ts.isoformat()}",
    )
    check(
        feature_ts <= prediction_ts,
        "the feature hour is at or before the prediction instant",
    )
    check(
        base["predicted_class"] == classify(base["probability"], bundle.threshold)
        and bundle.target_column == "target_1h",
        "the probability refers to the target_1h hour (t + 1 h), not to hour t",
    )
    check(
        base["observed_outcome"]["available"] is False,
        "no observed outcome is reported for the future hour",
    )
    check(
        base["observed_outcome"]["target_timestamp_utc"] == base["target_timestamp"],
        "the unavailable observation is identified by the predicted hour",
    )
    check(
        base["temporal_semantics"]["target_hour_is_in_the_future"] is True
        and base["temporal_semantics"]["target_hour_has_begun"] is False,
        "the predicted hour is correctly recognised as still in the future",
    )
    check(
        base["temporal_semantics"]["data_age_hours"] >= 0.0
        and abs(base["temporal_semantics"]["data_age_hours"]) < 1e-9,
        "data age is reported non-negative and is zero when the latest hour is the reference",
        f"{base['temporal_semantics']['data_age_hours']}",
    )
    check(
        "not a confidence" in base["probability_semantics"].lower()
        or "NOT a confidence" in base["probability_semantics"],
        "the payload states that the probability is not a confidence",
    )
    check(
        json.dumps(base, default=str).count('"confidence"') == 0
        and '"confidence"' not in json.dumps(base, default=str),
        "no output field is named 'confidence'",
    )
    payload_text = json.dumps(base, default=str)
    # The Phase 6 scientific disclaimer does mention high accuracy, but only in a
    # sentence that rejects it. So the requirement is that every such mention
    # carries its negation, not that the words never appear.
    sentences = [s.strip() for s in payload_text.replace("\\n", " ").split(".")]
    unnegated = [
        s
        for s in sentences
        if "high accuracy" in s.lower() and "not evidence of skill" not in s.lower()
    ]
    check(
        not unnegated,
        "no sentence presents high accuracy as evidence of skill",
        f"{len(unnegated)} unnegated mention(s)",
    )
    check(
        "not evidence of skill" in payload_text.lower(),
        "the payload states explicitly that high accuracy is not evidence of skill",
    )
    check(
        "not an official IMD" in base["disclaimer"]
        and "uses no radar" in base["disclaimer"]
        and "not a nationwide forecast" in base["disclaimer"],
        "the disclaimer denies IMD authority, radar/satellite/lightning/NWP inputs and nationwide scope",
    )

    # -----------------------------------------------------------------------
    section("[10] Output contract")
    required = {
        "probability",
        "predicted_class",
        "risk_label",
        "threshold",
        "lead_time_hours",
        "prediction_timestamp",
        "feature_timestamp",
        "source",
        "model",
        "feature_completeness",
    }
    check(required <= set(base), "every required output field is present", str(sorted(required - set(base))))
    completeness = base["feature_completeness"]
    check(
        completeness["n_features_expected"] == 28
        and completeness["n_features_provided"] == 28
        and completeness["n_features_missing"] == 0,
        "feature_completeness reports 28 of 28 features present",
    )
    check(
        completeness["all_features_finite"] is True
        and completeness["history_sufficient"] is True
        and completeness["values_imputed"] == 0,
        "feature_completeness reports finite features, sufficient history and no imputation",
    )
    check(
        all(
            np.isfinite(float(v)) for v in base["features_used_for_prediction"].values()
        ),
        "no NaN or inf appears in the produced feature vector",
    )
    check(
        base["source"]["provenance"] is not None and base["model"]["name"] == "thunderstorm_nowcast_1h",
        "source provenance and model name/version are reported",
        f"name={base['model']['name']} version={base['model']['version']}",
    )
    skill = base["measured_skill_reference"] or {}
    check(
        abs(float(skill.get("recall", -1)) - float(evaluation["metrics"]["test"]["recall"])) < 1e-12
        and abs(float(skill.get("precision", -1)) - float(evaluation["metrics"]["test"]["precision"])) < 1e-12,
        "the payload carries the Phase 6 measured skill of this model/threshold pair",
        f"recall={skill.get('recall'):.4f} precision={skill.get('precision'):.4f}",
    )

    # -----------------------------------------------------------------------
    section("[11] Live provider integration")
    try:
        live = predict_current_thunderstorm_risk(
            model=bundle, max_data_age_hours=engine.DEFAULT_MAX_DATA_AGE_HOURS
        )
    except ThunderstormNowcastError as error:
        check(
            True,
            "a live call either succeeds or fails safely with a typed error",
            f"offline/upstream failure handled: code={error.code}",
        )
    else:
        live_cell = live["source"]["provenance"].get("served_grid_cell") or {}
        training_cell = engine.TRAINING_SERVED_GRID_CELL
        near = (
            live_cell.get("latitude") is not None
            and abs(float(live_cell["latitude"]) - training_cell["latitude"]) < 1e-3
            and abs(float(live_cell["longitude"]) - training_cell["longitude"]) < 1e-3
        )
        check(
            live["source"]["frame_origin"] == "live_open_meteo_forecast_endpoint",
            "a live call reaches the Open-Meteo forecast endpoint",
            f"hours_returned={live['source']['provenance']['hours_returned']}",
        )
        check(
            near,
            "the live request is served the same grid cell the model was trained on",
            f"served={live_cell.get('latitude')},{live_cell.get('longitude')} "
            f"training={training_cell['latitude']},{training_cell['longitude']}",
        )
        check(
            0.0 <= live["probability"] <= 1.0
            and live["predicted_class"] == classify(live["probability"], bundle.threshold)
            and live["feature_completeness"]["values_imputed"] == 0,
            "the live prediction is internally consistent and imputes nothing",
            f"p={live['probability']:.6f} class={live['predicted_class']} "
            f"label={live['risk_label']}",
        )
        check(
            live["temporal_semantics"]["future_hours_discarded"] > 0,
            "the live call discarded the future hours the provider returned",
            f"discarded={live['temporal_semantics']['future_hours_discarded']}",
        )

    # -----------------------------------------------------------------------
    section("[12] Phase 1-6 protected files remain unchanged")
    after_protected = snapshot(PROTECTED_FILES)
    changed = sorted(
        rel
        for rel in PROTECTED_FILES
        if before_protected[rel] != after_protected[rel]
    )
    check(not changed, "the engine left every protected Phase 1-6 artifact byte-identical", str(changed))
    check(
        all(value != "MISSING" for value in before_protected.values()),
        "every protected Phase 1-6 artifact is present",
        str([r for r, v in before_protected.items() if v == "MISSING"]),
    )

    phase4_meta_now = json.loads(PHASE4_META.read_text(encoding="utf-8"))
    check(
        phase4_meta_now["protected_files"]["changed_files"] == [],
        "the Phase 4 metadata still reports that no Phase 1-3 artifact changed during Phase 4",
        str(phase4_meta_now["protected_files"]["changed_files"]),
    )
    supervision_sha = phase4_meta_now["inputs"]["supervision_source"]["sha256"]
    actual_sync_sha = sha256_file(DATASET_DIR / "votv_thunderstorm_synchronized_2014_2025.csv")
    check(
        supervision_sha == actual_sync_sha,
        "the Phase 3 synchronization table still matches the hash Phase 4 recorded for it",
        f"{supervision_sha[:12]}... vs {actual_sync_sha[:12]}...",
    )

    model_meta_now = json.loads(MODEL_META.read_text(encoding="utf-8"))
    training = model_meta_now.get("training_dataset") or {}
    chain = training.get("input_chain_check") or {}
    check(
        chain.get("phase4_feature_table_sha256_recorded_in_phase5_metadata")
        == chain.get("phase4_feature_table_sha256_now"),
        "the Phase 4 -> Phase 5 ancestry is still intact (recorded feature-table checksum matches disk)",
        f"recorded={str(chain.get('phase4_feature_table_sha256_recorded_in_phase5_metadata'))[:12]}...",
    )
    now_sha = sha256_file(PHASE5_CSV)
    check(
        training.get("sha256") == now_sha,
        "the 1 h model was trained on the Phase 5 target table exactly as it exists now",
        f"recorded={str(training.get('sha256'))[:12]}... now={now_sha[:12]}...",
    )

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    modified_protected = []
    added_protected = []
    other_modified = []
    for line in status.stdout.splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:].strip().strip('"')
        if path not in PROTECTED_FILES:
            if code != "??":
                other_modified.append(line.strip())
            continue
        if "??" in code or "A" in code:
            # A protected artifact appearing for the first time is not a change to
            # it; Phase 6 artifacts are still uncommitted by design.
            added_protected.append(line.strip())
        elif any(flag in code for flag in ("M", "D", "R", "T")):
            modified_protected.append(line.strip())
    check(
        not modified_protected,
        "git shows no content change to any protected Phase 1-6 artifact",
        str(modified_protected),
    )
    print(
        f"      note: protected artifacts staged for addition (awaiting commit, "
        f"unchanged by Phase 7): {len(added_protected)}",
        flush=True,
    )
    print(
        f"      note: modified/added tracked files outside the protected set: "
        f"{other_modified or 'none'}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    failed = [label for label, ok, _ in CHECKS if not ok]
    print(f"checks run: {len(CHECKS)}   passed: {len(CHECKS) - len(failed)}   failed: {len(failed)}")
    if failed:
        print("FAILED CHECKS:")
        for label in failed:
            print(f"  - {label}")
        print("\nPHASE 7 INDEPENDENT VERIFICATION: FAIL")
        return 1
    print("\nPHASE 7 INDEPENDENT VERIFICATION: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(2)
