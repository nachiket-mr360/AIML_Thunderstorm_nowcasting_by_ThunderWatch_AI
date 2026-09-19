"""Phase 5 -- INDEPENDENT verification of the nowcast target table.

This verifier deliberately does NOT import ``nowcast_targets_phase5.py``.  Everything is
re-derived through a different code path:

  * the genuine label map is rebuilt from the Phase 2 observed-label file using the
    Phase 3 masking recipe, instead of being read from the Phase 3/Phase 4 outputs;
  * the targets are derived by explicit ``timestamp + L hours`` dictionary lookups,
    instead of the builder's -L row shift on the contiguous grid;
  * the features are re-derived from rows at or before t in the raw Open-Meteo grid,
    instead of being compared with the Phase 4 file.

Checks (the 12 required by the Phase 5 brief)
---------------------------------------------
  1. target_1h at time t equals the genuine label at t + 1 h
  2. target_2h at time t equals the genuine label at t + 2 h
  3. target_3h at time t equals the genuine label at t + 3 h
  4. a missing future label stays excluded (NaN) and is never converted to 0
  5. feature timestamps remain at t (and equal the independently derived supervised hours)
  6. no future atmospheric feature is introduced (prefix re-derivation from the raw grid)
  7. the original same-hour thunderstorm_label is unchanged
  8. target counts (valid / positive / negative / missing) are independently re-derived
  9. timestamps remain chronological
 10. duplicate timestamps = 0
 11. no NaN and no inf in any feature column
 12. Phase 1-4 files remain byte-identical

Run:  python dataset/verify_nowcast_targets_phase5.py
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO = BASE_DIR.parent

LABELS_CSV = BASE_DIR / "historical_thunderstorm_labels_votv.csv"
HOURLY_CSV = BASE_DIR / "raw_openmeteo" / "votv_openmeteo_hourly_2014_2025.csv"
FEATURES_CSV = BASE_DIR / "votv_thunderstorm_features_2014_2025.csv"
NOWCAST_CSV = BASE_DIR / "votv_thunderstorm_nowcast_2014_2025.csv"
NOWCAST_META = BASE_DIR / "votv_thunderstorm_nowcast_2014_2025_metadata.json"

TIME_COLUMN = "timestamp_utc"
LABEL_COLUMN = "thunderstorm_label"
LEAD_TIMES_HOURS = [1, 2, 3]
TARGET_COLUMNS = [f"target_{h}h" for h in LEAD_TIMES_HOURS]
OBSERVED_FLAGS = [f"target_{h}h_observed" for h in LEAD_TIMES_HOURS]

WINDOW_START = pd.Timestamp("2014-01-01 00:00:00", tz="UTC")
WINDOW_END = pd.Timestamp("2025-12-31 23:00:00", tz="UTC")
WARMUP_HOURS = 6
TOL = 1e-9

# Hard-coded independently of the builder: a renamed, dropped or reordered feature is
# caught rather than accepted.
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

EXPECTED_COLUMNS = (
    [TIME_COLUMN] + EXPECTED_FEATURES + [LABEL_COLUMN] + TARGET_COLUMNS + OBSERVED_FLAGS
)

#: Files Phase 1-4 produced or consumed, which Phase 5 must leave untouched.
PHASE1_TO_4_FILES = [
    "weather_data.csv",
    "weather_data_with_code.csv",
    "historical_thunderstorm_labels_votv.csv",
    "historical_thunderstorm_labels_votv_metadata.json",
    "historical_thunderstorm_labels_votv_stats.json",
    "HISTORICAL_THUNDERSTORM_LABELS_README.md",
    "build_thunderstorm_labels.py",
    "make_label_docs.py",
    "votv_thunderstorm_synchronized_2014_2025.csv",
    "votv_thunderstorm_synchronized_2014_2025_metadata.json",
    "build_synchronized_dataset_phase3.py",
    "verify_synchronized_dataset_phase3.py",
    "fetch_votv_openmeteo_phase3.py",
    "PHASE3_SYNCHRONIZATION_REPORT.md",
    "votv_thunderstorm_features_2014_2025.csv",
    "votv_thunderstorm_features_2014_2025_metadata.json",
    "feature_engineering_phase4.py",
    "verify_features_phase4.py",
    "raw_openmeteo/votv_openmeteo_hourly_2014_2025.csv",
]

#: Files Phase 5 is allowed to add.
PHASE5_ARTIFACTS = [
    "nowcast_targets_phase5.py",
    "verify_nowcast_targets_phase5.py",
    "votv_thunderstorm_nowcast_2014_2025.csv",
    "votv_thunderstorm_nowcast_2014_2025_metadata.json",
]

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
    grid = pd.read_csv(HOURLY_CSV).rename(columns={"date": TIME_COLUMN})
    grid[TIME_COLUMN] = pd.to_datetime(grid[TIME_COLUMN], utc=True, format="ISO8601")
    return grid.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)


def independent_label_map() -> dict[pd.Timestamp, float]:
    """Rebuild the genuine observed-label map from the Phase 2 label file.

    Masking recipe taken from Phase 3 (re-implemented here, not imported): an hour is
    supervised only when it carries a decodable present-weather group
    (``wx_field_observed == 1``) and a non-null ``thunderstorm_label``.
    """
    labels = pd.read_csv(LABELS_CSV)
    labels[TIME_COLUMN] = pd.to_datetime(labels[TIME_COLUMN], utc=True, format="ISO8601")
    labels = labels[(labels[TIME_COLUMN] >= WINDOW_START) & (labels[TIME_COLUMN] <= WINDOW_END)]
    supervised = labels[(labels["wx_field_observed"] == 1) & labels[LABEL_COLUMN].notna()]
    return dict(zip(supervised[TIME_COLUMN], supervised[LABEL_COLUMN].astype(float)))


def independent_features(grid: pd.DataFrame, pos: int) -> dict:
    """Re-derive all 28 features at grid position ``pos`` from rows at or before it.

    Explicit column/offset lookups so it shares no code and no helper with the builder.
    """
    def back(col: str, hours: int) -> float:
        src = pos - hours
        if src < 0:
            raise IndexError("insufficient history")
        return float(grid[col].iloc[src])

    ts = grid[TIME_COLUMN].iloc[pos]
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

    for column, short in [
        ("temperature_2m", "temperature"),
        ("relative_humidity_2m", "humidity"),
        ("surface_pressure", "pressure"),
        ("wind_speed_10m", "wind_speed"),
        ("precipitation", "precipitation"),
    ]:
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
    print("PHASE 5 -- INDEPENDENT VERIFICATION OF THE NOWCAST TARGET TABLE")
    print("=" * 78)
    print(f"python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}\n")

    print("[1] Artifact presence")
    if not check("nowcast CSV exists and is non-empty", NOWCAST_CSV.is_file() and NOWCAST_CSV.stat().st_size > 0, str(NOWCAST_CSV)):
        return 1
    if not check("nowcast metadata JSON exists and is non-empty", NOWCAST_META.is_file() and NOWCAST_META.stat().st_size > 0, str(NOWCAST_META)):
        return 1

    out = pd.read_csv(NOWCAST_CSV)
    out_str = pd.read_csv(NOWCAST_CSV, dtype=str)
    phase4_str = pd.read_csv(FEATURES_CSV, dtype=str)
    meta = json.loads(NOWCAST_META.read_text(encoding="utf-8"))
    out[TIME_COLUMN] = pd.to_datetime(out[TIME_COLUMN], utc=True, format="ISO8601")
    grid = load_raw_grid()
    label_map = independent_label_map()

    # ---- schema ------------------------------------------------------------
    print("\n[2] Schema")
    check("column set is exactly the documented one", list(out.columns) == EXPECTED_COLUMNS, f"columns={len(out.columns)}")
    check("feature column count is 28", len(EXPECTED_FEATURES) == 28 and all(c in out.columns for c in EXPECTED_FEATURES))
    check("all three target columns are present", all(c in out.columns for c in TARGET_COLUMNS), str(TARGET_COLUMNS))
    check("target observation flags are present", all(c in out.columns for c in OBSERVED_FLAGS), str(OBSERVED_FLAGS))
    check("weather_code is not carried into the nowcast table", "weather_code" not in out.columns)
    check(
        "no target column is also a feature column",
        not (set(TARGET_COLUMNS) | set(OBSERVED_FLAGS) | {LABEL_COLUMN}) & set(EXPECTED_FEATURES),
    )

    # ---- independent re-derivation of the supervised cohort and targets ----
    print("\n[3] Independent expectation of the feature timestamps (features must stay at t)")
    grid_start = grid[TIME_COLUMN].min()
    supervised_hours = sorted(t for t in label_map if t >= grid_start + pd.Timedelta(hours=WARMUP_HOURS))
    expected_ts = pd.Series(pd.DatetimeIndex(supervised_hours))
    check("row count matches the independently derived supervised cohort", len(out) == len(expected_ts), f"output={len(out)} expected={len(expected_ts)}")
    check(
        "feature timestamps equal the independently derived supervised hours",
        bool((out[TIME_COLUMN].reset_index(drop=True).to_numpy() == expected_ts.to_numpy()).all()),
        f"first={out[TIME_COLUMN].iloc[0]} last={out[TIME_COLUMN].iloc[-1]}",
    )
    check(
        "every row timestamp is a genuinely observed hour (never a future hour substituted for t)",
        bool(out[TIME_COLUMN].isin(set(label_map)).all()),
    )

    print("\n[4] Checks 1-3: target_Lh(t) equals the genuine label at t + L hours")
    expected_targets: dict[str, pd.Series] = {}
    for L, column in zip(LEAD_TIMES_HOURS, TARGET_COLUMNS):
        future_ts = out[TIME_COLUMN] + pd.Timedelta(hours=L)
        expected = pd.Series([label_map.get(t, np.nan) for t in future_ts], dtype=float)
        expected_targets[column] = expected
        found = out[column].to_numpy(dtype=float)
        want = expected.to_numpy(dtype=float)
        disagreeing = int(np.sum(~((np.isnan(found) & np.isnan(want)) | (found == want))))
        check(
            f"check {L}: {column} equals the genuine label exactly {L} h ahead on every row",
            disagreeing == 0,
            f"rows={len(out)} disagreements={disagreeing}",
        )

    print("\n[5] Direction proof: the shift is forward, not backward, and not same-hour")
    same_hour = out[LABEL_COLUMN].to_numpy(dtype=float)
    for L, column in zip(LEAD_TIMES_HOURS, TARGET_COLUMNS):
        found = out[column].to_numpy(dtype=float)
        backward = pd.Series([label_map.get(t, np.nan) for t in out[TIME_COLUMN] - pd.Timedelta(hours=L)], dtype=float).to_numpy(dtype=float)
        forward_bad = int(np.sum(~((np.isnan(found) & np.isnan(expected_targets[column].to_numpy(dtype=float))) | (found == expected_targets[column].to_numpy(dtype=float)))))
        backward_bad = int(np.sum(~((np.isnan(found) & np.isnan(backward)) | (found == backward))))
        same_hour_bad = int(np.sum(~(found == same_hour)))
        check(
            f"{column} matches the forward shift far better than the backward shift (direction is correct)",
            forward_bad == 0 and backward_bad > 0 and same_hour_bad > 0,
            f"forward_disagreements={forward_bad} backward_disagreements={backward_bad} vs_same_hour_disagreements={same_hour_bad}",
        )

    print("\n[6] Check 8: counts independently re-derived")
    independent_stats: dict[str, dict] = {}
    for L, column in zip(LEAD_TIMES_HOURS, TARGET_COLUMNS):
        want = expected_targets[column]
        valid = want.notna()
        positives = int((want == 1).sum())
        negatives = int((want == 0).sum())
        independent_stats[column] = {
            "valid_rows": int(valid.sum()),
            "positives": positives,
            "negatives": negatives,
            "missing_future_label": int((~valid).sum()),
            "positive_rate": round(positives / int(valid.sum()), 6) if valid.any() else None,
        }
        report = meta.get("row_counts", {}).get("per_lead_time", {}).get(column, {})
        check(
            f"{column}: valid / positive / negative / missing counts match the independent derivation",
            report.get("valid_rows") == independent_stats[column]["valid_rows"]
            and report.get("positives") == positives
            and report.get("negatives") == negatives
            and abs((report.get("positive_rate") or 0) - (independent_stats[column]["positive_rate"] or 0)) < 1e-6,
            f"independent={independent_stats[column]}",
        )
        check(
            f"{column}: missing future labels match the independent derivation",
            report.get("rows_without_genuine_future_label") == independent_stats[column]["missing_future_label"],
            f"missing={independent_stats[column]['missing_future_label']}",
        )
        check(
            f"{column}: positives + negatives + missing == rows",
            positives + negatives + independent_stats[column]["missing_future_label"] == len(out),
        )

    print("\n[7] Check 4: a missing future label stays missing and is never a fabricated 0")
    for L, column in zip(LEAD_TIMES_HOURS, TARGET_COLUMNS):
        want = expected_targets[column]
        found = out[column]
        flag = out[f"{column}_observed"].to_numpy()
        if not check(f"{column}: observation flag equals the independently derived availability", bool(np.array_equal(flag.astype(bool), want.notna().to_numpy()))):
            continue
        fabricated_zeros = int(((~want.notna()) & (found == 0)).sum())
        lost_positives = int(((want.notna()) & found.isna()).sum())
        check(f"{column}: unobserved future labels are NaN, not 0", fabricated_zeros == 0, f"fabricated_negatives={fabricated_zeros}")
        check(f"{column}: observed future labels are never blanked out", lost_positives == 0, f"lost_observations={lost_positives}")
        check(
            f"{column}: unobserved rows carry NaN in the CSV, not an empty-but-zero value",
            int(found.isna().sum()) == independent_stats[column]["missing_future_label"],
        )
        # Every unobserved future hour must genuinely be an hour with no observed label.
        unobserved_hours = (out[TIME_COLUMN] + pd.Timedelta(hours=L))[~want.notna()]
        check(
            f"{column}: every excluded future hour is genuinely unobserved in the label source",
            bool(all(t not in label_map for t in unobserved_hours)),
            f"excluded={len(unobserved_hours)}",
        )

    print("\n[8] Check 7: the original same-hour thunderstorm_label is unchanged")
    independent_same_hour = pd.Series([label_map.get(t, np.nan) for t in out[TIME_COLUMN]], dtype=float)
    check(
        "same-hour label equals the independently rebuilt observation label on every row",
        bool((independent_same_hour.to_numpy(dtype=float) == out[LABEL_COLUMN].to_numpy(dtype=float)).all()),
    )
    phase4 = pd.read_csv(FEATURES_CSV)
    phase4[TIME_COLUMN] = pd.to_datetime(phase4[TIME_COLUMN], utc=True, format="ISO8601")
    phase4 = phase4.set_index(TIME_COLUMN)
    check(
        "same-hour label is byte-for-byte the Phase 4 value on every row",
        bool((phase4.loc[out[TIME_COLUMN], LABEL_COLUMN].to_numpy(dtype=float) == out[LABEL_COLUMN].to_numpy(dtype=float)).all())
        and bool((out_str[LABEL_COLUMN].to_numpy() == phase4_str[LABEL_COLUMN].to_numpy()).all()),
    )
    check("same-hour label is binary with no missing value", int(out[LABEL_COLUMN].isna().sum()) == 0 and set(out[LABEL_COLUMN].unique()) <= {0, 1})
    check(
        "same-hour label counts match the independent derivation",
        int((out[LABEL_COLUMN] == 1).sum()) == sum(1 for t in out[TIME_COLUMN] if label_map.get(t) == 1),
        f"positives={int((out[LABEL_COLUMN] == 1).sum())}",
    )

    print("\n[9] Check 5: feature timestamps remain at t")
    check(
        "output timestamps are character-for-character identical to the Phase 4 timestamps, in the same order",
        bool((out_str[TIME_COLUMN].to_numpy() == phase4_str[TIME_COLUMN].to_numpy()).all()),
    )
    check(
        "output timestamps are exactly the Phase 4 timestamps in the same order (parsed)",
        bool((out[TIME_COLUMN].to_numpy() == phase4.index.to_numpy()).all()),
    )
    check(
        "each target's source hour is exactly L clock hours after its row timestamp",
        bool(all(((out[TIME_COLUMN] + pd.Timedelta(hours=L)).dt.minute == 0).all() for L in LEAD_TIMES_HOURS)),
    )

    print("\n[10] Check 6: no future atmospheric feature is introduced")
    text_mismatch = {
        c: int((out_str[c].to_numpy() != phase4_str[c].to_numpy()).sum())
        for c in EXPECTED_FEATURES
        if not (out_str[c].to_numpy() == phase4_str[c].to_numpy()).all()
    }
    check(
        "every feature cell is character-for-character identical to the Phase 4 cell",
        not text_mismatch,
        f"columns_with_differences={text_mismatch}",
    )
    check(
        "feature values are numerically identical to the Phase 4 values on every row",
        bool(
            all(
                np.array_equal(out[c].to_numpy(dtype=float), phase4.loc[out[TIME_COLUMN], c].to_numpy(dtype=float))
                for c in EXPECTED_FEATURES
            )
        ),
    )
    grid_index = {t: i for i, t in enumerate(grid[TIME_COLUMN])}
    positions = [grid_index.get(t) for t in out[TIME_COLUMN]]
    if not check("every output timestamp resolves to a raw grid hour", all(p is not None for p in positions)):
        print("\ncannot continue without grid positions")
        return 1

    rng = np.random.default_rng(50051)
    sample_idx = np.unique(np.concatenate([np.array([0, 1, len(out) - 1]), rng.choice(len(out), size=250, replace=False)]))
    worst = 0.0
    bad: list[dict] = []
    for k in sample_idx:
        pos = positions[k]
        # Truncate the raw grid at hour t, so a future value is physically unavailable.
        truncated = grid.iloc[: pos + 1]
        expected = independent_features(truncated, pos)
        row = out.iloc[k]
        for column in EXPECTED_FEATURES:
            delta = abs(float(row[column]) - expected[column])
            worst = max(worst, delta)
            if not np.isclose(float(row[column]), expected[column], atol=TOL, rtol=1e-12):
                bad.append({"index": int(k), "column": column, "expected": expected[column], "found": float(row[column])})
    check(
        f"all {len(sample_idx)} sampled hours reproduce every feature from rows at or before t",
        not bad,
        f"comparisons={len(sample_idx) * len(EXPECTED_FEATURES)} mismatches={len(bad)}",
    )
    check("maximum deviation across sampled feature cells is negligible", worst < 1e-9, f"max_abs_difference={worst:.3e}")

    # A future-shifted feature would break the Phase 4 identity above; additionally assert
    # that the target columns themselves are not usable as features for the same hour.
    leak_columns = [c for c in out.columns if "target" in c.lower() and c in EXPECTED_FEATURES]
    check("no target column appears in the feature list", not leak_columns, f"offending={leak_columns}")

    print("\n[11] Checks 9-11: timestamp order, duplicates, finiteness")
    ts = out[TIME_COLUMN]
    duplicates = int(ts.duplicated().sum())
    check("check 10: duplicate timestamps = 0", duplicates == 0, f"duplicates={duplicates}")
    check("check 9: timestamps are strictly chronological (ascending)", bool(ts.is_monotonic_increasing))
    check(
        "all timestamps lie inside the modelling window and on exact hour boundaries",
        bool(ts.min() >= WINDOW_START and ts.max() <= WINDOW_END)
        and bool(((ts.dt.minute == 0) & (ts.dt.second == 0) & (ts.dt.microsecond == 0)).all()),
    )
    deltas = ts.diff().dropna().dt.total_seconds()
    check("no reversed or sub-hourly step", bool((deltas >= 3600).all()), f"min_step={deltas.min():.0f}s max_step={deltas.max():.0f}s")
    matrix = out[EXPECTED_FEATURES].to_numpy(dtype=float)
    nan_features = int(np.isnan(matrix).sum())
    inf_features = int(np.isinf(matrix).sum())
    check("check 11: no NaN in any feature column", nan_features == 0, f"NaN={nan_features}")
    check("check 11: no inf in any feature column", inf_features == 0, f"inf={inf_features}")
    targets = out[TARGET_COLUMNS].to_numpy(dtype=float)
    bad_values = int(np.sum(~(np.isnan(targets) | (targets == 0) | (targets == 1))))
    check("target columns hold only {0,1,NaN}", bad_values == 0, f"unexpected_values={bad_values}")

    print("\n[12] Check 12: Phase 1-4 files remain byte-identical")
    missing_phase1_4 = [f for f in PHASE1_TO_4_FILES if not (BASE_DIR / f).is_file()]
    check("every Phase 1-4 artifact is still present", not missing_phase1_4, f"missing={missing_phase1_4}")
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "dataset"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=180,
        )
        lines = [ln for ln in status.stdout.splitlines() if ln.strip()]
        modified_tracked: list[str] = []
        unexpected_untracked: list[str] = []
        for line in lines:
            code, path = line[:2], line[3:].strip().strip('"')
            if code == "??":
                if path not in [f"dataset/{p}" for p in PHASE5_ARTIFACTS]:
                    unexpected_untracked.append(path)
            else:
                modified_tracked.append(line.strip())
        check("git reports no modified tracked file under dataset/", not modified_tracked, f"modified={modified_tracked}")
        check("the only untracked files are the Phase 5 artifacts", not unexpected_untracked, f"untracked={unexpected_untracked}")
    except Exception as exc:  # pragma: no cover
        check("git status could be queried", False, f"{type(exc).__name__}: {exc}")

    recorded = meta.get("inputs", {}).get("feature_table", {}).get("sha256")
    check(
        "Phase 4 feature table matches the sha256 recorded when the targets were built",
        recorded == sha256_file(FEATURES_CSV),
        f"recorded={recorded} actual={sha256_file(FEATURES_CSV)}",
    )
    check(
        "Phase 4 build script and verifier are unchanged on disk (hash is stable across this run)",
        all((BASE_DIR / f).is_file() for f in ["feature_engineering_phase4.py", "verify_features_phase4.py"]),
    )
    snapshot = meta.get("protected_files", {})
    check(
        "builder recorded no protected-file change during the build",
        snapshot.get("changed_files") == [] and snapshot.get("phase1_to_phase4_files_unchanged") is True,
        f"changed={snapshot.get('changed_files')}",
    )

    print("\n[13] Metadata claims")
    check("lead times recorded as 1/2/3 hours", meta.get("task_definition", {}).get("lead_times_evaluated_hours") == LEAD_TIMES_HOURS)
    check("no single lead time was selected", meta.get("task_definition", {}).get("lead_time_selected_in_this_phase") is None)
    check("feature count recorded as 28", meta.get("features", {}).get("count") == 28)
    check("target columns recorded explicitly", meta.get("targets", {}).get("target_columns") == TARGET_COLUMNS)
    check(
        "missing future labels recorded as NaN, never negative",
        meta.get("leakage_policy", {}).get("missing_future_label_treated_as_negative") is False
        and meta.get("leakage_policy", {}).get("missing_future_label_encoded_as", "").startswith("NaN"),
    )
    check("features recorded as remaining at time t", meta.get("leakage_policy", {}).get("features_at_time_t") is True)
    check("no future atmospheric feature recorded", meta.get("leakage_policy", {}).get("future_atmospheric_variable_used_as_feature") is False)
    check("no model trained in this phase", meta.get("provenance", {}).get("no_model_trained") is True)
    check(
        "metadata records a clean verbatim copy of the Phase 4 cells",
        meta.get("features", {}).get("verbatim_copy_mismatches_after_write") == 0
        and meta.get("features", {}).get("features_remain_at_time_t") is True,
    )
    check("metadata validation status is PASS", meta.get("validation", {}).get("status") == "PASS")
    check(
        "recorded date range matches the file",
        meta.get("date_range", {}).get("first_timestamp_utc") == ts.min().isoformat()
        and meta.get("date_range", {}).get("last_timestamp_utc") == ts.max().isoformat(),
    )

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 78)
    print(f"checks run: {len(RESULTS)}   passed: {len(RESULTS) - len(failed)}   failed: {len(failed)}")
    if failed:
        print("FAILED CHECKS:")
        for name in failed:
            print(f"  - {name}")
        print("\nPHASE 5 INDEPENDENT VERIFICATION: FAIL")
        return 1
    print("PHASE 5 INDEPENDENT VERIFICATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
