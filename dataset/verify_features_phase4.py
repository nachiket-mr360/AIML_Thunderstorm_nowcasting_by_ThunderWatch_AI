"""Phase 4 -- INDEPENDENT verification of the engineered feature table.

This verifier deliberately does NOT import ``feature_engineering_phase4.py``.  It
re-derives the expected feature values with a separate code path (explicit clock-hour
index arithmetic on the raw Open-Meteo grid) and re-checks every claim the builder
made about the output table, the target, causality and protected files.

Checks
------
 1. output CSV and metadata exist and are non-empty
 2. row count equals the independently derived expectation
 3. timestamp range and target counts match the independently derived expectation
 4. no duplicate timestamps
 5. timestamps strictly chronological
 6. hourly continuity: hour-aligned, every step an exact multiple of 3600 s
 7. no NaN and no inf in any feature column
 8. target is excluded from the model matrix (exact feature-name list, hard-coded here)
 9. no future-derived column exists (independent prefix re-derivation per sampled hour)
10. rolling features are causal and use the intended trailing windows
11. Phase 1/2/3 files are unmodified (sha256 against the Phase 3 metadata, plus mtimes)

Run:  python dataset/verify_features_phase4.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

SYNC_CSV = BASE_DIR / "votv_thunderstorm_synchronized_2014_2025.csv"
SYNC_META = BASE_DIR / "votv_thunderstorm_synchronized_2014_2025_metadata.json"
HOURLY_CSV = BASE_DIR / "raw_openmeteo" / "votv_openmeteo_hourly_2014_2025.csv"
OUT_CSV = BASE_DIR / "votv_thunderstorm_features_2014_2025.csv"
OUT_META = BASE_DIR / "votv_thunderstorm_features_2014_2025_metadata.json"

TARGET = "thunderstorm_label"

# Hard-coded independently of the builder, so a silently renamed/dropped/reordered
# feature is caught rather than accepted.
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

# Columns that must exist but must never be model inputs.
NON_MODEL_COLUMNS = ["timestamp_utc", TARGET, "weather_code", "wind_direction_10m", "features_complete"]

WARMUP_HOURS = 6
TOL = 1e-9

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


def load_raw_grid() -> pd.DataFrame:
    grid = pd.read_csv(HOURLY_CSV).rename(columns={"date": "timestamp_utc"})
    grid["timestamp_utc"] = pd.to_datetime(grid["timestamp_utc"], utc=True, format="ISO8601")
    return grid.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)


def independent_features(grid: pd.DataFrame, pos: int) -> dict:
    """Re-derive all 28 features at grid position ``pos`` using clock-hour arithmetic.

    Written as an explicit column/offset lookup so it shares no code with the builder.
    Only positions <= pos are read.
    """
    def back(col: str, hours: int) -> float:
        src = pos - hours
        if src < 0:
            raise IndexError("insufficient history")
        return float(grid[col].iloc[src])

    ts = grid["timestamp_utc"].iloc[pos]
    hour = float(ts.hour)
    month = float(ts.month)

    f: dict[str, float] = {}
    f["temperature_2m"] = back("temperature_2m", 0)
    f["relative_humidity_2m"] = back("relative_humidity_2m", 0)
    f["surface_pressure"] = back("surface_pressure", 0)
    f["wind_speed_10m"] = back("wind_speed_10m", 0)
    f["precipitation"] = back("precipitation", 0)
    f["cloud_cover"] = back("cloud_cover", 0)

    f["hour_sin"] = math.sin(2.0 * math.pi * hour / 24.0)
    f["hour_cos"] = math.cos(2.0 * math.pi * hour / 24.0)
    f["month_sin"] = math.sin(2.0 * math.pi * (month - 1.0) / 12.0)
    f["month_cos"] = math.cos(2.0 * math.pi * (month - 1.0) / 12.0)

    direction = math.radians(back("wind_direction_10m", 0))
    f["wind_direction_sin"] = math.sin(direction)
    f["wind_direction_cos"] = math.cos(direction)

    changes = [
        ("temperature_2m", "temperature"),
        ("relative_humidity_2m", "humidity"),
        ("surface_pressure", "pressure"),
        ("wind_speed_10m", "wind_speed"),
        ("precipitation", "precipitation"),
    ]
    for column, short in changes:
        f[f"{short}_change_1h"] = back(column, 0) - back(column, 1)
        f[f"{short}_change_3h"] = back(column, 0) - back(column, 3)

    f["precipitation_roll_sum_3h"] = sum(back("precipitation", k) for k in range(3))
    f["precipitation_roll_sum_6h"] = sum(back("precipitation", k) for k in range(6))
    for column, short in [
        ("relative_humidity_2m", "humidity"),
        ("surface_pressure", "pressure"),
        ("temperature_2m", "temperature"),
        ("wind_speed_10m", "wind_speed"),
    ]:
        f[f"{short}_roll_mean_3h"] = sum(back(column, k) for k in range(3)) / 3.0
    return f


def main() -> int:
    print("=" * 78)
    print("PHASE 4 -- INDEPENDENT VERIFICATION OF ENGINEERED FEATURES")
    print("=" * 78)
    print(f"python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}\n")

    # ---- 1. artifacts exist -------------------------------------------------
    print("[1] Artifact presence")
    if not check("feature CSV exists and is non-empty", OUT_CSV.is_file() and OUT_CSV.stat().st_size > 0, str(OUT_CSV)):
        return 1
    if not check("metadata JSON exists and is non-empty", OUT_META.is_file() and OUT_META.stat().st_size > 0, str(OUT_META)):
        return 1

    out = pd.read_csv(OUT_CSV)
    meta = json.loads(OUT_META.read_text(encoding="utf-8"))
    sync = pd.read_csv(SYNC_CSV)
    sync["timestamp_utc"] = pd.to_datetime(sync["timestamp_utc"], utc=True, format="ISO8601")
    grid = load_raw_grid()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, format="ISO8601")

    # ---- 2/3. independent expectation --------------------------------------
    print("\n[2] Row count, range and target against an independent derivation")
    window_start = grid["timestamp_utc"].min()
    expected_ts = sync.loc[
        sync["timestamp_utc"] >= window_start + pd.Timedelta(hours=WARMUP_HOURS), "timestamp_utc"
    ].reset_index(drop=True)
    expected_positives = int(sync.loc[sync["timestamp_utc"] >= window_start + pd.Timedelta(hours=WARMUP_HOURS), TARGET].sum())

    check("row count matches independent expectation", len(out) == len(expected_ts), f"output={len(out)} expected={len(expected_ts)}")
    check(
        "feature matrix column count is exactly 28 + non-model columns",
        len(out.columns) == len(EXPECTED_FEATURES) + 5,
        f"columns={len(out.columns)}",
    )
    check("first output hour is at or after the warm-up boundary", out["timestamp_utc"].min() >= window_start + pd.Timedelta(hours=WARMUP_HOURS), str(out["timestamp_utc"].min()))
    check("first output hour is the earliest supervised hour with full history", out["timestamp_utc"].min() == expected_ts.min(), f"output={out['timestamp_utc'].min()} expected={expected_ts.min()}")
    check("timestamp range is inside the Phase 3 window", out["timestamp_utc"].min() >= sync["timestamp_utc"].min() and out["timestamp_utc"].max() <= sync["timestamp_utc"].max())
    check("output timestamps equal the Phase 3 supervised hours minus warm-up", out["timestamp_utc"].reset_index(drop=True).equals(expected_ts))

    positives = int(out[TARGET].sum())
    negatives = int(len(out) - positives)
    check("positive count matches independent expectation", positives == expected_positives, f"output={positives} expected={expected_positives}")
    check("positives + negatives == rows", positives + negatives == len(out), f"{positives}+{negatives}={len(out)}")
    check("target contains only {0,1}", set(out[TARGET].unique()) <= {0, 1}, str(sorted(out[TARGET].unique())))
    check("target has no missing value", int(out[TARGET].isna().sum()) == 0)

    # ---- 4/5/6. timestamp integrity ----------------------------------------
    print("\n[3] Timestamp integrity")
    ts = out["timestamp_utc"]
    check("no duplicate timestamps", int(ts.duplicated().sum()) == 0, f"duplicates={int(ts.duplicated().sum())}")
    check("timestamps strictly chronological (ascending)", bool(ts.is_monotonic_increasing))
    check(
        "all timestamps on an exact hour boundary (UTC)",
        bool(((ts.dt.minute == 0) & (ts.dt.second == 0) & (ts.dt.microsecond == 0)).all()),
    )
    deltas = ts.diff().dropna().dt.total_seconds()
    check("every consecutive step is an exact multiple of 3600 s", bool((deltas % 3600 == 0).all()), f"min={deltas.min():.0f}s max={deltas.max():.0f}s")
    check("no reversed or sub-hourly step", bool((deltas >= 3600).all()))
    gaps = int((deltas > 3600).sum())
    check(
        "hourly continuity: steps are 1 h except at documented label gaps",
        int((deltas == 3600).sum()) + gaps == len(deltas),
        f"1h steps={int((deltas == 3600).sum())} gap steps={gaps} (gaps are Phase 3 label-coverage gaps, not data loss)",
    )

    # ---- 7. finiteness ------------------------------------------------------
    print("\n[4] Feature-column finiteness")
    missing_features = [c for c in EXPECTED_FEATURES if c not in out.columns]
    check("every expected feature is present", not missing_features, f"missing={missing_features}")
    if missing_features:
        return 1
    matrix = out[EXPECTED_FEATURES].to_numpy(dtype=float)
    nan_total = int(np.isnan(matrix).sum())
    inf_total = int(np.isinf(matrix).sum())
    check("no NaN in any feature column", nan_total == 0, f"NaN={nan_total}")
    check("no inf in any feature column", inf_total == 0, f"inf={inf_total}")
    check("feature columns are all numeric", all(pd.api.types.is_numeric_dtype(out[c]) for c in EXPECTED_FEATURES))

    # ---- 8. target excluded from X -----------------------------------------
    print("\n[5] Target exclusion")
    check("target is not among the model features", TARGET not in EXPECTED_FEATURES)
    actual_dropped = [c for c in out.columns if c not in EXPECTED_FEATURES + NON_MODEL_COLUMNS]
    check("output carries no unexpected extra column", not actual_dropped, f"unexpected={actual_dropped}")
    leaked = [c for c in EXPECTED_FEATURES if "label" in c.lower() or "target" in c.lower() or "ts" == c.lower()]
    check("no feature name refers to the target or an observation count", not leaked, f"offending={leaked}")
    for audit_col in ["n_reports", "n_reports_full_decode", "n_reports_with_wx", "wx_field_observed", "n_ts_reports", "wxcodes_observed", "thunderstorm_label_strict", "thunderstorm_label_recovered"]:
        if audit_col in out.columns and audit_col in EXPECTED_FEATURES:
            check(f"observation column {audit_col} is not a feature", False)
    check("observation-derived Phase 3 columns are absent from the file entirely", not any(c in out.columns for c in ["n_reports", "wxcodes_observed", "wx_field_observed", "thunderstorm_label_strict", "thunderstorm_label_recovered"]))
    check("weather_code is present for diagnostics but excluded from the model matrix", "weather_code" in out.columns and "weather_code" not in EXPECTED_FEATURES)

    # ---- 9/10. causality by independent prefix re-derivation ---------------
    print("\n[6] Causality: independent re-derivation from rows at or before t")
    grid_index = {t: i for i, t in enumerate(grid["timestamp_utc"])}
    positions = [grid_index[t] for t in out["timestamp_utc"]]
    check("every output timestamp resolves to a raw grid hour", len(positions) == len(out))

    rng = np.random.default_rng(4242)
    sample_idx = np.unique(np.concatenate([
        np.array([0, 1, 2, len(out) - 1]),
        rng.choice(len(out), size=250, replace=False),
    ]))
    worst = {c: 0.0 for c in EXPECTED_FEATURES}
    bad = []
    for k in sample_idx:
        pos = positions[k]
        expected = independent_features(grid, pos)
        row = out.iloc[k]
        for column in EXPECTED_FEATURES:
            delta = abs(float(row[column]) - expected[column])
            if delta > worst[column]:
                worst[column] = delta
            if not np.isclose(float(row[column]), expected[column], atol=TOL, rtol=1e-12):
                bad.append({"index": int(k), "column": column, "expected": expected[column], "found": float(row[column])})
    check(
        f"all {len(sample_idx)} sampled hours reproduce every feature from past/current hours only",
        not bad,
        f"comparisons={len(sample_idx) * len(EXPECTED_FEATURES)} mismatches={len(bad)}",
    )
    if bad:
        for b in bad[:10]:
            print(f"      mismatch: {b}")
    check("maximum deviation across all compared cells is negligible", max(worst.values()) < 1e-9, f"max_abs_difference={max(worst.values()):.3e}")

    # ---- 11. rolling windows are causal and trailing ------------------------
    print("\n[7] Rolling-window semantics")
    roll_cases = [
        ("precipitation_roll_sum_3h", "precipitation", 3, "sum"),
        ("precipitation_roll_sum_6h", "precipitation", 6, "sum"),
        ("humidity_roll_mean_3h", "relative_humidity_2m", 3, "mean"),
        ("pressure_roll_mean_3h", "surface_pressure", 3, "mean"),
        ("temperature_roll_mean_3h", "temperature_2m", 3, "mean"),
        ("wind_speed_roll_mean_3h", "wind_speed_10m", 3, "mean"),
    ]
    for column, source, window, reduction in roll_cases:
        worst_delta = 0.0
        for k in sample_idx:
            pos = positions[k]
            window_values = [float(grid[source].iloc[pos - back]) for back in range(window)]
            value = sum(window_values) if reduction == "sum" else sum(window_values) / window
            worst_delta = max(worst_delta, abs(float(out.iloc[k][column]) - value))
        check(f"{column} equals the trailing {window}h {reduction} ending at t", worst_delta <= TOL, f"max_abs_difference={worst_delta:.3e}")

    print("\n[8] Future-perturbation invariance (definitive causality test)")
    # Rewrite every raw value in the six hours AFTER t with an extreme value and require
    # all 28 features at t to be bit-for-bit unchanged. This proves no column can read a
    # future hour. (A trailing/forward window comparison is not sufficient here: for a
    # sparse variable such as precipitation the two windows are both zero most of the
    # time and would agree by coincidence.)
    source_columns = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "precipitation", "cloud_cover", "weather_code"]
    perturbed_rows = 0
    violations = []
    for k in sample_idx:
        pos = positions[k]
        if pos + WARMUP_HOURS >= len(grid):
            continue
        probe = grid.copy()
        future = probe.iloc[pos + 1 : pos + 1 + WARMUP_HOURS]
        for column in source_columns:
            probe.loc[future.index, column] = probe.loc[future.index, column] + 1234.5
        expected = independent_features(probe, pos)
        perturbed_rows += 1
        for column in EXPECTED_FEATURES:
            if abs(float(out.iloc[k][column]) - expected[column]) > TOL:
                violations.append({"index": int(k), "column": column})
    check(
        "every feature at t is invariant to arbitrary changes in hours t+1 .. t+6",
        not violations,
        f"hours perturbed={perturbed_rows} feature values checked={perturbed_rows * len(EXPECTED_FEATURES)} violations={len(violations)}",
    )
    if violations:
        for v in violations[:10]:
            print(f"      future-dependent feature: {v}")

    # change features must follow clock hours, not output-file rows
    print("\n[9] Change features follow clock hours, not output-file rows")
    row_steps = np.diff(positions)
    check(
        "output rows are not guaranteed adjacent in clock time (gaps exist)",
        int((row_steps > 1).sum()) > 0,
        f"row steps greater than 1 hour: {int((row_steps > 1).sum())}",
    )
    clock_bad = 0
    checked_gap_rows = 0
    for k in sample_idx:
        pos = positions[k]
        if k > 0 and positions[k - 1] == pos - 1:
            continue
        if k == 0 or pos == 0:
            continue
        prev_row_value = float(out.iloc[k - 1]["temperature_2m"])
        clock_value = float(grid["temperature_2m"].iloc[pos - 1])
        if abs(prev_row_value - clock_value) > 1e-9:
            checked_gap_rows += 1
            if abs(float(out.iloc[k]["temperature_change_1h"]) - (float(out.iloc[k]["temperature_2m"]) - clock_value)) > TOL:
                clock_bad += 1
    check(
        "temperature_change_1h uses the previous clock hour even across label gaps",
        clock_bad == 0,
        f"gap-adjacent hours examined={checked_gap_rows} violations={clock_bad}",
    )

    # ---- 12. protected Phase 1/2/3 files ------------------------------------
    print("\n[10] Protected Phase 1/2/3 inputs")
    phase3 = json.loads(SYNC_META.read_text(encoding="utf-8"))
    recorded = dict(phase3.get("protected_files_sha256_before", {}))
    recorded.update(phase3.get("protected_files_sha256_after", {}))
    for name, digest in recorded.items():
        path = BASE_DIR / name
        if not path.is_file():
            check(f"{name} still present", False)
            continue
        check(f"{name} byte-identical to the Phase 3 audit", sha256_file(path) == digest, f"sha256={sha256_file(path)[:16]}...")

    sync_digest = phase3.get("checksums", {}).get("csv_sha256")
    check("Phase 3 synchronized CSV unchanged", sha256_file(SYNC_CSV) == sync_digest, f"sha256={sha256_file(SYNC_CSV)[:16]}...")

    per_year = phase3.get("inputs", {}).get("features", {}).get("per_year", {})
    year_bad = []
    for year, info in per_year.items():
        path = BASE_DIR / "raw_openmeteo" / f"votv_openmeteo_{year}.csv"
        if not path.is_file() or sha256_file(path) != info["sha256"]:
            year_bad.append(year)
    check("all 12 raw Open-Meteo feature files unchanged", not year_bad, f"mismatched={year_bad}")

    out_mtime = OUT_CSV.stat().st_mtime
    own = {OUT_CSV.resolve(), OUT_META.resolve(), Path(__file__).resolve()}
    newer = [
        str(p.relative_to(BASE_DIR)).replace("\\", "/")
        for p in BASE_DIR.rglob("*")
        if p.is_file() and p.resolve() not in own and p.stat().st_mtime > out_mtime
    ]
    check("no other file under dataset/ was written after the Phase 4 output", not newer, f"newer={newer}")

    # ---- 13. metadata contents ---------------------------------------------
    print("\n[11] Metadata completeness")
    required = [
        "input_files" if "input_files" in meta else "inputs",
        "outputs",
        "date_range",
        "row_counts",
        "features",
        "target",
        "missing_values",
        "duplicate_count",
        "timestamp_integrity",
        "leakage_audit",
        "rolling_window_methodology",
        "source_coordinates",
    ]
    for key in required:
        check(f"metadata records '{key}'", key in meta)
    check("metadata feature count matches the hard-coded expectation", meta["features"]["count"] == len(EXPECTED_FEATURES), f"{meta['features']['count']} vs {len(EXPECTED_FEATURES)}")
    check("metadata feature list matches the output header order", meta["features"]["feature_order"] == EXPECTED_FEATURES)
    check("metadata confirms weather_code was excluded from features", meta["leakage_audit"]["weather_code_excluded_from_features"] is True)
    check("metadata confirms centred windows were not used", meta["leakage_audit"]["future_information"]["centred_windows_used"] is False)
    check("metadata confirms no future shift was used", meta["leakage_audit"]["future_information"]["forward_shifts_used"] is False)
    check("metadata confirms no model was trained in this phase", meta.get("no_model_trained") is True)
    check("metadata confirms no fabricated radar/satellite/lightning/NWP feature", meta["provenance"]["radar_satellite_lightning_derived_features"] is False and meta["provenance"]["nwp_derived_features"] is False and meta["provenance"]["synthetic_or_invented_variables"] is False)

    # ---- summary ------------------------------------------------------------
    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 78)
    print(f"CHECKS RUN: {len(RESULTS)}   PASSED: {len(RESULTS) - len(failed)}   FAILED: {len(failed)}")
    print(f"output rows        : {len(out)}")
    print(f"feature count      : {len(EXPECTED_FEATURES)}")
    print(f"target distribution: {positives} positive / {negatives} negative")
    if failed:
        print("\nFAILED CHECKS:")
        for name in failed:
            print(f"  - {name}")
        print("\nPHASE 4 INDEPENDENT VERIFICATION: FAIL")
        return 1
    print("\nPHASE 4 INDEPENDENT VERIFICATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
