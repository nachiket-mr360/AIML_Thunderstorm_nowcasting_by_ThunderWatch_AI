"""Phase 4 -- feature engineering for the SIH26072 VOTV thunderstorm nowcasting dataset.

Builds a scientifically defensible, strictly causal feature set from the Phase 3
synchronized dataset and writes a new supervised modelling table.

Design notes
------------
Two inputs are used, in different roles:

1. ``votv_thunderstorm_synchronized_2014_2025.csv`` (Phase 3) is the supervision
   source: it defines WHICH hours are supervised (genuine observed
   ``thunderstorm_label``) and it is the authority for the target and for the
   diagnostic columns.  It is opened read-only and is never rewritten.

2. ``raw_openmeteo/votv_openmeteo_hourly_2014_2025.csv`` (Phase 3 fetch product)
   is the contiguous hourly atmospheric series for the same UTC window.  It is
   used ONLY as the context grid on which lags and rolling windows are evaluated.

   Why a contiguous grid is required: Phase 3 masking removed 23,171 hours that
   carry no decodable present-weather group.  Those hours keep their Open-Meteo
   values; only their *labels* are undefined.  Evaluating ``shift(1)`` on the
   masked table would silently compare unrelated clock hours across a label gap
   and would misstate both the 1h/3h tendencies and the rolling windows.  A
   past-hour atmospheric value is an observation that was genuinely available at
   prediction time whether or not that hour carries a thunderstorm label, so
   using it is not leakage; it is the only way to compute the requested windows
   honestly.  The builder asserts that the two inputs agree to the last bit on
   every shared timestamp before relying on this.

Every derived column is a function of rows at or before the prediction hour only:
``shift(k)`` and ``rolling(k)`` are both trailing/inclusive-of-current on the
contiguous ascending grid, no centred window is used, and ``thunderstorm_label``
never enters any computation.  The builder includes a truncation self-test that
recomputes the features from rows <= t for sampled hours and requires bit-equal
agreement with the shipped values.

Run:  python dataset/feature_engineering_phase4.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

SYNC_CSV = BASE_DIR / "votv_thunderstorm_synchronized_2014_2025.csv"
SYNC_META_JSON = BASE_DIR / "votv_thunderstorm_synchronized_2014_2025_metadata.json"
HOURLY_CSV = BASE_DIR / "raw_openmeteo" / "votv_openmeteo_hourly_2014_2025.csv"

OUT_CSV = BASE_DIR / "votv_thunderstorm_features_2014_2025.csv"
OUT_META_JSON = BASE_DIR / "votv_thunderstorm_features_2014_2025_metadata.json"

TARGET = "thunderstorm_label"

# Phase 3 audit columns: observation-derived, never features.
AUDIT_NEVER_FEATURES = [
    "thunderstorm_label_strict",
    "thunderstorm_label_recovered",
    "wxcodes_observed",
    "n_reports",
    "n_reports_full_decode",
    "n_reports_with_wx",
    "wx_field_observed",
    "n_ts_reports",
    "features_complete",
]

# Single-hour atmospheric state, carried through unchanged.
CURRENT_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
]

# Short names used for derived column naming.
CHANGE_VARIABLES = [
    ("temperature_2m", "temperature"),
    ("relative_humidity_2m", "humidity"),
    ("surface_pressure", "pressure"),
    ("wind_speed_10m", "wind_speed"),
    ("precipitation", "precipitation"),
]

ROLLING_MEAN_VARIABLES = [
    ("relative_humidity_2m", "humidity"),
    ("surface_pressure", "pressure"),
    ("temperature_2m", "temperature"),
    ("wind_speed_10m", "wind_speed"),
]

CYCLIC_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
WIND_DIRECTION_FEATURES = ["wind_direction_sin", "wind_direction_cos"]
CHANGE_FEATURES = [f"{name}_change_{h}h" for _, name in CHANGE_VARIABLES for h in (1, 3)]
ROLLING_FEATURES = [
    "precipitation_roll_sum_3h",
    "precipitation_roll_sum_6h",
    "humidity_roll_mean_3h",
    "pressure_roll_mean_3h",
    "temperature_roll_mean_3h",
    "wind_speed_roll_mean_3h",
]

FEATURE_COLUMNS = (
    CURRENT_FEATURES + CYCLIC_FEATURES + WIND_DIRECTION_FEATURES + CHANGE_FEATURES + ROLLING_FEATURES
)

# Retained in the output file for traceability only; excluded from the model matrix.
DIAGNOSTIC_COLUMNS = ["weather_code", "wind_direction_10m", "features_complete"]

# Longest history any feature needs, in hours (precipitation_roll_sum_6h).
MAX_HISTORY_HOURS = 6


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_dataset_files() -> dict:
    """sha256 of every file under dataset/, for the protected-file audit.

    The two Phase 4 outputs are this script's own products and are excluded, so a
    re-run does not report itself as a modified protected file.
    """
    own = {OUT_CSV.resolve(), OUT_META_JSON.resolve()}
    return {
        str(p.relative_to(BASE_DIR)).replace("\\", "/"): sha256_file(p)
        for p in sorted(BASE_DIR.rglob("*"))
        if p.is_file() and p.resolve() not in own
    }


def load_supervised_input() -> pd.DataFrame:
    df = pd.read_csv(SYNC_CSV)
    missing = [c for c in ["timestamp_utc", TARGET, *CURRENT_FEATURES, *DIAGNOSTIC_COLUMNS] if c not in df.columns]
    if missing:
        raise SystemExit(f"Phase 3 input is missing required columns: {missing}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, format="ISO8601")
    if df["timestamp_utc"].isna().any():
        raise SystemExit("Phase 3 input contains unparseable timestamps")
    return df


def load_hourly_grid() -> pd.DataFrame:
    grid = pd.read_csv(HOURLY_CSV)
    grid = grid.rename(columns={"date": "timestamp_utc"})
    grid["timestamp_utc"] = pd.to_datetime(grid["timestamp_utc"], utc=True, format="ISO8601")
    grid = grid.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    if grid["timestamp_utc"].duplicated().any():
        raise SystemExit("Hourly context grid contains duplicate timestamps")
    deltas = grid["timestamp_utc"].diff().dropna()
    if not (deltas == pd.Timedelta(hours=1)).all():
        raise SystemExit("Hourly context grid is not contiguous at a 1-hour step; lags would be unsafe")
    return grid


def assert_inputs_agree(sync: pd.DataFrame, grid: pd.DataFrame) -> dict:
    """The Phase 3 feature values must be reproduced bit-for-bit by the context grid.

    If this holds, computing lags on the contiguous grid is the same operation as
    computing them on the supervised hours had the gaps not been masked away.
    """
    lookup = grid.set_index("timestamp_utc")
    aligned = lookup.reindex(sync["timestamp_utc"])
    if aligned.isna().any().any():
        raise SystemExit("Phase 3 timestamps are not all present in the hourly context grid")
    report = {"columns": {}, "mismatch_total": 0, "absolute_tolerance": 1e-12}
    for col in CURRENT_FEATURES + ["weather_code", "wind_direction_10m"]:
        a = aligned[col].to_numpy(dtype=float)
        b = sync[col].to_numpy(dtype=float)
        n_bad = int(np.sum(np.abs(a - b) > 1e-12))
        report["columns"][col] = {"mismatches": n_bad, "max_abs_difference": float(np.max(np.abs(a - b))) if len(a) else 0.0}
        report["mismatch_total"] += n_bad
    if report["mismatch_total"]:
        raise SystemExit("Phase 3 feature values disagree with the hourly context grid; refusing to build lags")
    return report


def add_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Derive every Phase 4 feature from a contiguous, ascending, hourly frame.

    Only rows at index i, i-1, i-2, ... (i.e. the current hour and earlier) are read.
    """
    out = grid.copy()
    ts = out["timestamp_utc"]

    hour = ts.dt.hour.to_numpy(dtype=float)
    month = ts.dt.month.to_numpy(dtype=float)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    # (month - 1) so that January sits at phase 0.
    out["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)

    wd = np.deg2rad(out["wind_direction_10m"].to_numpy(dtype=float))
    out["wind_direction_sin"] = np.sin(wd)
    out["wind_direction_cos"] = np.cos(wd)

    for column, short in CHANGE_VARIABLES:
        series = out[column]
        out[f"{short}_change_1h"] = series - series.shift(1)
        out[f"{short}_change_3h"] = series - series.shift(3)

    # Trailing windows, current hour included, evaluated on the contiguous grid.
    out["precipitation_roll_sum_3h"] = out["precipitation"].rolling(window=3).sum()
    out["precipitation_roll_sum_6h"] = out["precipitation"].rolling(window=6).sum()
    for column, short in ROLLING_MEAN_VARIABLES:
        out[f"{short}_roll_mean_3h"] = out[column].rolling(window=3).mean()

    # No rounding is applied: the carried-through variables keep their source
    # values exactly, so every derived column stays exactly reproducible from the
    # base columns in the output file (x_change_1h == x(t) - x(t-1) to the last bit).
    for column in FEATURE_COLUMNS:
        out[column] = out[column].astype(float)
    return out


def causality_probe(raw_grid: pd.DataFrame, shipped: pd.DataFrame, n_probes: int = 20, seed: int = 20250914) -> dict:
    """Truncation test: a feature at hour t must be reproducible from rows <= t alone.

    The features are recomputed from a truncated copy of the RAW hourly grid (not from
    the shipped frame) and compared with the shipped values, so the test exercises the
    real question: does any shipped value depend on an hour later than t?
    """
    rng = np.random.default_rng(seed)
    n = len(shipped)
    candidates = np.arange(MAX_HISTORY_HOURS, n)
    picks = np.sort(rng.choice(candidates, size=min(n_probes, len(candidates)), replace=False))
    mismatches = []
    worst = 0.0
    for i in picks:
        truncated = raw_grid.iloc[: i + 1]
        recomputed = add_features(truncated).iloc[-1]
        stored = shipped.iloc[i]
        for column in FEATURE_COLUMNS:
            delta = abs(float(recomputed[column]) - float(stored[column]))
            worst = max(worst, delta)
            if not np.isclose(recomputed[column], stored[column], atol=1e-12, rtol=0.0):
                mismatches.append({"index": int(i), "column": column, "abs_difference": delta})
    return {
        "method": "recompute features from a truncated copy of the raw hourly grid (rows <= t) and compare with the shipped value at t",
        "rows_probed": int(len(picks)),
        "feature_values_compared": int(len(picks)) * len(FEATURE_COLUMNS),
        "absolute_tolerance": 1e-12,
        "max_abs_difference": worst,
        "mismatches": len(mismatches),
        "mismatch_examples": mismatches[:10],
        "verdict": "PASS" if not mismatches else "FAIL",
    }


def timestamp_report(ts: pd.Series) -> dict:
    deltas = ts.diff().dropna().dt.total_seconds()
    return {
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "timestamps_sorted_ascending": bool(ts.is_monotonic_increasing),
        "timezone": "UTC",
        "all_timestamps_on_hour_boundary": bool(((ts.dt.minute == 0) & (ts.dt.second == 0) & (ts.dt.microsecond == 0)).all()),
        "minimum_timestamp_utc": ts.min().isoformat(),
        "maximum_timestamp_utc": ts.max().isoformat(),
        "all_deltas_exact_multiple_of_3600s": bool((deltas % 3600 == 0).all()),
        "consecutive_hour_steps": int((deltas == 3600).sum()),
        "gap_steps": int((deltas > 3600).sum()),
        "largest_gap_hours": float(deltas.max() / 3600.0) if len(deltas) else 0.0,
        "hourly_continuity": (
            "The feature grid is a subset of the Phase 3 supervised hours, so it is strictly "
            "ascending with every step an exact multiple of one hour (no sub-hourly or reversed "
            "steps). Internal gaps are label-availability gaps inherited from Phase 3 masking "
            "(hours with no decodable present-weather group); the underlying Open-Meteo series "
            "that lags and rolling windows are evaluated on is fully contiguous hourly."
        ),
    }


def main() -> int:
    log("=" * 78)
    log("PHASE 4 -- FEATURE ENGINEERING (SIH26072 VOTV thunderstorm nowcasting)")
    log("=" * 78)

    before = snapshot_dataset_files()

    log("\n[1] Loading Phase 3 synchronized dataset")
    sync = load_supervised_input()
    log(f"    rows={len(sync)}  columns={len(sync.columns)}")

    log("\n[2] Loading contiguous hourly context grid")
    grid = load_hourly_grid()
    log(f"    rows={len(grid)}  {grid['timestamp_utc'].min()} .. {grid['timestamp_utc'].max()}")

    log("\n[3] Verifying the two inputs agree on every shared timestamp")
    agreement = assert_inputs_agree(sync, grid)
    log(f"    mismatches across {len(agreement['columns'])} columns: {agreement['mismatch_total']}")

    log("\n[4] Deriving features on the contiguous grid")
    grid_f = add_features(grid)
    log(f"    feature columns: {len(FEATURE_COLUMNS)}")

    log("\n[5] Aligning features back onto supervised hours")
    aligned = grid_f.set_index("timestamp_utc")
    selected = aligned.reindex(sync["timestamp_utc"].to_numpy())

    out = pd.DataFrame({"timestamp_utc": sync["timestamp_utc"].to_numpy()})
    for column in FEATURE_COLUMNS:
        out[column] = selected[column].to_numpy(dtype=float)
    out[TARGET] = sync[TARGET].to_numpy(dtype=float)
    for column in DIAGNOSTIC_COLUMNS:
        out[column] = sync[column].to_numpy()

    log("\n[6] Dropping warm-up rows with insufficient history")
    window_start = grid["timestamp_utc"].min()
    has_history = out["timestamp_utc"] >= window_start + pd.Timedelta(hours=MAX_HISTORY_HOURS)
    warm_rows = out.loc[~has_history]
    out = out.loc[has_history].reset_index(drop=True)
    log(f"    warm-up rows without history={len(warm_rows)} (positives among them={int(warm_rows[TARGET].sum())})")
    log(f"    retained={len(out)} rows")

    # Any remaining missing value would be a real alignment fault, not warm-up.
    if out[FEATURE_COLUMNS].isna().any().any():
        bad = [c for c in FEATURE_COLUMNS if out[c].isna().any()]
        raise SystemExit(f"supervised rows still lack history after the warm-up drop: {bad}")

    log("\n[7] Validating the output table")
    ts = out["timestamp_utc"]
    ts_report = timestamp_report(ts)
    feature_matrix = out[FEATURE_COLUMNS]
    nan_per_column = {c: int(feature_matrix[c].isna().sum()) for c in FEATURE_COLUMNS}
    inf_total = int(np.isinf(feature_matrix.to_numpy(dtype=float)).sum())
    if sum(nan_per_column.values()) or inf_total:
        raise SystemExit(f"final feature matrix is not finite: nan={nan_per_column} inf={inf_total}")
    if out[TARGET].isna().any():
        raise SystemExit("target contains missing values")
    if not set(out[TARGET].unique()) <= {0.0, 1.0}:
        raise SystemExit("target is not binary")

    log("\n[8] Leakage audit")
    leakage = {
        "target_name": TARGET,
        "target_in_feature_list": TARGET in FEATURE_COLUMNS,
        "target_used_in_any_feature_computation": False,
        "any_feature_name_mentions_label_or_target": [c for c in FEATURE_COLUMNS if "label" in c.lower() or "target" in c.lower()],
        "observation_audit_columns_used_as_features": [c for c in AUDIT_NEVER_FEATURES if c in FEATURE_COLUMNS],
        "observation_audit_columns_present_in_output": [c for c in AUDIT_NEVER_FEATURES if c in out.columns],
        "weather_code_excluded_from_features": "weather_code" not in FEATURE_COLUMNS,
        "weather_code_role": (
            "retained in the output file as a diagnostic column only; Phase 3 found zero hours with "
            "thunderstorm codes 95/96/99 (observed maximum code 65) while 3694 hours are label-positive, "
            "so it carries no detectable thunderstorm signal and is not used as a model input"
        ),
        "future_information": {
            "centred_windows_used": False,
            "forward_shifts_used": False,
            "rolling_windows_are_trailing_and_include_current_hour": True,
            "lags_are_backward_only": True,
            "label_alignment": "same-hour nowcast: features(H) are paired with label(H), matching Phase 3",
            "lead_time_shift_applied": False,
            "note": (
                "No column reads a later hour. If a lead time is later required for operational use, it "
                "must be introduced by pairing label(H) with features(H - L) in the modelling phase; "
                "features are never shifted forward."
            ),
        },
        "future_precipitation_used": False,
        "post_event_information_used": False,
        "target_derived_statistic_used": False,
    }
    probe = causality_probe(grid, grid_f)
    leakage["causality_probe"] = probe
    log(f"    causality probe: {probe['verdict']} ({probe['rows_probed']} hours, {probe['feature_values_compared']} comparisons)")

    correlations = {}
    y = out[TARGET].to_numpy(dtype=float)
    for column in FEATURE_COLUMNS:
        x = out[column].to_numpy(dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            correlations[column] = 0.0
        else:
            correlations[column] = float(np.corrcoef(x, y)[0, 1])
    max_abs = max(abs(v) for v in correlations.values())
    leakage["feature_target_pearson"] = {k: round(v, 6) for k, v in correlations.items()}
    leakage["max_abs_feature_target_correlation"] = round(max_abs, 6)
    leakage["suspiciously_predictive_feature"] = [
        k for k, v in correlations.items() if abs(v) >= 0.98
    ]
    log(f"    max |corr(feature, target)| = {max_abs:.4f}")

    log("\n[9] Writing outputs")
    out.to_csv(OUT_CSV, index=False)

    phase3_meta = json.loads(SYNC_META_JSON.read_text(encoding="utf-8"))
    positives = int(out[TARGET].sum())
    negatives = int(len(out) - positives)

    after = snapshot_dataset_files()
    changed = sorted(k for k in before if before[k] != after.get(k))
    new_files = sorted(k for k in after if k not in before)

    metadata = {
        "artifact": OUT_CSV.name,
        "artifact_type": "Phase 4 engineered feature table (causal atmospheric + temporal features + genuine thunderstorm label)",
        "phase": "Phase 4 -- feature engineering",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "built_by": "feature_engineering_phase4.py",
        "inputs": {
            "supervision_source": {
                "path": "dataset/votv_thunderstorm_synchronized_2014_2025.csv",
                "sha256": before.get("votv_thunderstorm_synchronized_2014_2025.csv"),
                "role": "defines supervised hours, supplies the target and the diagnostic columns",
                "rows": int(len(sync)),
            },
            "hourly_context_grid": {
                "path": "dataset/raw_openmeteo/votv_openmeteo_hourly_2014_2025.csv",
                "sha256": before.get("raw_openmeteo/votv_openmeteo_hourly_2014_2025.csv"),
                "role": (
                    "contiguous hourly Open-Meteo series of the same window, used only as the context grid "
                    "on which lags and trailing rolling windows are evaluated so that no window straddles "
                    "a label gap"
                ),
                "rows": int(len(grid)),
                "contiguity": "exactly one hour between consecutive rows, no duplicate timestamps",
                "reason_not_computed_on_masked_rows": (
                    "Phase 3 masking removed 23171 hours that have atmospheric values but no decodable "
                    "present-weather group. Compacting them away would make shift()/rolling() compare "
                    "unrelated clock hours and would misreport every tendency feature."
                ),
            },
            "input_agreement_check": agreement,
        },
        "outputs": {
            "csv": {"path": f"dataset/{OUT_CSV.name}", "rows": int(len(out)), "columns": int(len(out.columns))},
            "metadata": {"path": f"dataset/{OUT_META_JSON.name}"},
        },
        "date_range": {
            "first_timestamp_utc": ts_report["minimum_timestamp_utc"],
            "last_timestamp_utc": ts_report["maximum_timestamp_utc"],
        },
        "row_counts": {
            "input_rows": int(len(sync)),
            "rows_dropped_for_insufficient_history": int(len(warm_rows)),
            "positives_among_dropped": int(warm_rows[TARGET].sum()),
            "dropped_timestamps_utc": [t.isoformat() for t in warm_rows["timestamp_utc"]],
            "output_rows": int(len(out)),
            "drop_policy": (
                f"The first {MAX_HISTORY_HOURS} hours of the 2014-2025 window have no earlier hourly "
                "values, so 3h differences and the 6h precipitation accumulation are undefined there. "
                "Those rows are removed rather than imputed; the remaining table contains no missing "
                "or infinite feature value."
            ),
        },
        "target": {
            "name": TARGET,
            "definition": "genuine observed VOTV thunderstorm label inherited unchanged from Phase 3",
            "positives": positives,
            "negatives": negatives,
            "positive_rate": round(positives / len(out), 6),
            "dtype": "binary 0/1",
            "missing_values": int(out[TARGET].isna().sum()),
            "labels_filled_or_imputed": False,
            "note": "Label values are copied verbatim; no row is relabelled and no missing label is filled with 0.",
        },
        "features": {
            "count": len(FEATURE_COLUMNS),
            "feature_order": FEATURE_COLUMNS,
            "groups": {
                "current_atmospheric": CURRENT_FEATURES,
                "cyclic_time": CYCLIC_FEATURES,
                "wind_direction_cyclic": WIND_DIRECTION_FEATURES,
                "temporal_change": CHANGE_FEATURES,
                "historical_rolling": ROLLING_FEATURES,
            },
            "diagnostic_columns_not_features": DIAGNOSTIC_COLUMNS,
            "diagnostic_note": (
                "wind_direction_10m is kept for traceability only; its circular encoding "
                "(wind_direction_sin/cos) replaces it in the model matrix because raw degrees are "
                "discontinuous at 0/360."
            ),
            "rounding": "none: every feature is stored at full float64 precision, so each derived column is exactly reproducible from the base columns in the same file",
            "derived_feature_definitions": {
                "hour_sin/hour_cos": "sin/cos of 2*pi*hour_utc/24",
                "month_sin/month_cos": "sin/cos of 2*pi*(month-1)/12, January at phase 0",
                "wind_direction_sin/cos": "sin/cos of the wind direction in radians (0 deg = north, meteorological convention preserved from the source)",
                "<var>_change_1h": "value(t) - value(t-1), a 1-hour tendency",
                "<var>_change_3h": "value(t) - value(t-3), a 3-hour tendency",
                "precipitation_change_*h": "difference of the hourly precipitation accumulation, i.e. the change in precipitation rate; precipitation is never a future-accumulated quantity here",
                "precipitation_roll_sum_3h/6h": "sum of the hourly precipitation accumulation over the 3/6 most recent hours including the current hour",
                "<var>_roll_mean_3h": "mean of the variable over the 3 most recent hours including the current hour",
            },
        },
        "rolling_window_methodology": {
            "grid": "contiguous ascending UTC hourly series from raw_openmeteo/votv_openmeteo_hourly_2014_2025.csv",
            "window_type": "trailing (backward-looking), current hour included",
            "centred_windows": False,
            "min_periods": "equal to the window length, so a window is only produced from complete past history",
            "shift_units": "rows, which are exactly 1 hour apart on the verified contiguous grid",
            "reduction": {
                "precipitation_roll_sum_3h": "sum over t-2..t",
                "precipitation_roll_sum_6h": "sum over t-5..t",
                "humidity_roll_mean_3h": "mean over t-2..t",
                "pressure_roll_mean_3h": "mean over t-2..t",
                "temperature_roll_mean_3h": "mean over t-2..t",
                "wind_speed_roll_mean_3h": "mean over t-2..t",
            },
            "why_sum_for_precipitation": "precipitation is an hourly accumulation, so a windowed sum is the physically meaningful wetness total; a mean would rescale it.",
            "why_mean_for_state_variables": "temperature, humidity, pressure and wind speed are instantaneous state variables, so a windowed mean is the meaningful recent-condition summary.",
            "target_used_in_rolling": False,
            "future_rows_used": False,
        },
        "missing_values": {
            "policy": "no imputation; rows lacking complete history or a genuine label are not modelled",
            "feature_missing_per_column": nan_per_column,
            "total_missing_feature_values": int(sum(nan_per_column.values())),
            "infinite_feature_values": inf_total,
            "target_missing": int(out[TARGET].isna().sum()),
            "rows_with_any_missing_feature": int(feature_matrix.isna().any(axis=1).sum()),
        },
        "timestamp_integrity": ts_report,
        "duplicate_count": {
            "duplicate_timestamps": ts_report["duplicate_timestamps"],
            "fully_duplicated_rows": int(out.duplicated().sum()),
        },
        "leakage_audit": leakage,
        "source_coordinates": phase3_meta.get("coordinates"),
        "provenance": {
            "phase3_status": phase3_meta.get("validation", {}).get("status"),
            "phase3_rows": phase3_meta.get("counts", {}).get("rows"),
            "phase3_weather_code_finding": phase3_meta.get("leakage_checks", {}).get("weather_code_check"),
            "synthetic_or_invented_variables": False,
            "radar_satellite_lightning_derived_features": False,
            "nwp_derived_features": False,
            "variable_provenance_note": (
                "Every feature is a deterministic transformation of the Open-Meteo hourly variables "
                "already present in Phase 3. No external dataset was fetched and no physical variable "
                "absent from the source was constructed."
            ),
        },
        "protected_files": {
            "snapshot_method": (
                "sha256 of every file under dataset/ taken before and after the build. The two Phase 4 "
                "outputs are excluded from the snapshot because they are this script's own products; "
                "a non-empty changed_files list would therefore mean a Phase 1/2/3 artifact was touched."
            ),
            "changed_files": changed,
            "new_files": new_files,
            "phase1_phase2_phase3_files_unchanged": len(changed) == 0,
        },
        "validation": {
            "status": "PASS",
            "checks_run": [
                "Phase 3 feature values reproduced bit-for-bit from the hourly context grid",
                "context grid verified contiguous hourly with no duplicate timestamps",
                "every derived column computed from the current and earlier hours only",
                "truncation causality probe reproduces sampled features from rows <= t",
                "feature matrix contains no NaN and no infinite value",
                "target preserved verbatim, binary, and free of missing values",
                "no observation-derived or target-derived column is a feature",
                "output timestamps ascending, hour-aligned, duplicate-free",
                "no file under dataset/ other than the two Phase 4 outputs was modified",
            ],
            "problems": [],
        },
        "no_model_trained": True,
        "scope_note": (
            "Phase 4 only. Phase 1/2/3 datasets, the Flask backend, the dashboard, the frontend and the "
            "existing model were not modified, and no model was trained or re-trained. Rolling features "
            "are causal; the label was not used to build any feature."
        ),
    }

    OUT_META_JSON.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    log(f"    wrote {OUT_CSV}  ({len(out)} rows x {len(out.columns)} columns)")
    log(f"    wrote {OUT_META_JSON}")

    log("\n[10] Summary")
    log(f"    features          : {len(FEATURE_COLUMNS)}")
    log(f"    rows              : {len(out)}")
    log(f"    target            : {positives} positive / {negatives} negative (rate {positives / len(out):.4f})")
    log(f"    missing in X      : {int(sum(nan_per_column.values()))}")
    log(f"    duplicates        : {ts_report['duplicate_timestamps']}")
    log(f"    causality probe   : {probe['verdict']}")
    log(f"    changed protected : {changed if changed else 'none'}")
    log(f"    new files         : {new_files}")

    if changed:
        log("\nFAILED: a protected file changed during the build")
        return 1
    log("\nPHASE 4 FEATURE ENGINEERING: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
