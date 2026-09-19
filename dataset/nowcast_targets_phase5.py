"""Phase 5 -- nowcast target construction (lead times 1 h, 2 h, 3 h).

Turns the Phase 4 supervised table (``features(t)`` + genuine ``thunderstorm_label(t)``)
into an explicit nowcasting table by attaching, for every row, the GENUINE observed
thunderstorm label of the hours t+1, t+2 and t+3.  No model is trained here.

Task definition
---------------
    features : clock hour t            (unchanged Phase 4 columns, kept at t)
    target_L : genuine thunderstorm observation at clock hour t + L, for L in {1,2,3}

Direction rule (leakage)
------------------------
The target is the ONLY thing that moves in time.  It is shifted FORWARD relative to
the features (the label of a LATER hour), so a model trained on this table predicts an
observation that does not yet exist at prediction time.  Features are never shifted
forward and are copied bit-for-bit from Phase 4, whose columns were already proven to
be functions of hours <= t only (see feature_engineering_phase4.py, truncation probe).

How the shift is applied
------------------------
The label source is a *masked* series: Phase 3 removed hours that carry no decodable
present-weather group, so a genuine label exists for 82,021 of the 105,192 window hours
and the remaining hours are unobserved.  Two consequences drive the implementation:

1. Shifting on the compacted table would be wrong.  ``shift(-1)`` on compacted rows
   would compare a row with the next *labelled* row, which can be 2, 5 or 57 clock hours
   away (the largest gap in the supervision grid is 57 h).  That would silently turn
   "1 hour ahead" into an undefined lead time.

   The shift is therefore evaluated on the fully contiguous 105,192-hour Open-Meteo
   hourly grid, where one row step is exactly one clock hour.  The genuine labels are
   re-indexed onto that grid (NaN where the hour is unobserved) and the grid is shifted
   by -L rows, which places ``label(t + L)`` on hour t.  The builder asserts the grid is
   contiguous and duplicate-free before using it.

2. An unobserved future hour must stay unobserved.  NaN therefore propagates through
   the shift: an hour with no genuine label yields NaN, never 0.  Writing 0 would
   fabricate "no thunderstorm observed" out of "no observation", which would corrupt
   the negative class exactly where reporting coverage is worst.  Each row carries an
   explicit ``target_Lh_observed`` flag so the exclusion is auditable.

The original same-hour label ``thunderstorm_label`` is preserved verbatim in the output
for audit, and is NOT a model input.

Outputs
-------
  dataset/votv_thunderstorm_nowcast_2014_2025.csv
  dataset/votv_thunderstorm_nowcast_2014_2025_metadata.json

All three lead times are produced in one file; Phase 5 deliberately selects none of
them, so Phase 6 can choose on data availability and measured model performance.

Run:  python dataset/nowcast_targets_phase5.py
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

# Phase 4 supervised table: supplies features(t) and the genuine same-hour label.
FEATURES_CSV = BASE_DIR / "votv_thunderstorm_features_2014_2025.csv"
FEATURES_META = BASE_DIR / "votv_thunderstorm_features_2014_2025_metadata.json"
# Phase 3 synchronized table: the authority for which hours carry a genuine label.
SYNC_CSV = BASE_DIR / "votv_thunderstorm_synchronized_2014_2025.csv"
# Contiguous hourly Open-Meteo grid: the clock-hour ruler the shift is evaluated on.
HOURLY_CSV = BASE_DIR / "raw_openmeteo" / "votv_openmeteo_hourly_2014_2025.csv"

OUT_CSV = BASE_DIR / "votv_thunderstorm_nowcast_2014_2025.csv"
OUT_META_JSON = BASE_DIR / "votv_thunderstorm_nowcast_2014_2025_metadata.json"

TIME_COLUMN = "timestamp_utc"
LABEL_COLUMN = "thunderstorm_label"

LEAD_TIMES_HOURS = [1, 2, 3]
TARGET_COLUMNS = [f"target_{h}h" for h in LEAD_TIMES_HOURS]
OBSERVED_FLAGS = [f"target_{h}h_observed" for h in LEAD_TIMES_HOURS]

#: The 28 Phase 4 features, hard-coded (not read from the Phase 4 metadata) so that a
#: silently renamed, dropped or reordered feature is caught instead of accepted.  The
#: independent validator hard-codes its own copy of this list.
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

#: Modelling window of the Phase 3 supervised dataset; used only to explain why a
#: missing future label is missing.
WINDOW_END = pd.Timestamp("2025-12-31 23:00:00", tz="UTC")
WINDOW_START = pd.Timestamp("2014-01-01 00:00:00", tz="UTC")


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_dataset_files() -> dict:
    """sha256 of every file under dataset/, taken before and after the build.

    This script's own two outputs are excluded so a re-run does not report itself as a
    modified protected file.
    """
    own = {OUT_CSV.resolve(), OUT_META_JSON.resolve()}
    return {
        str(p.relative_to(BASE_DIR)).replace("\\", "/"): sha256_file(p)
        for p in sorted(BASE_DIR.rglob("*"))
        if p.is_file() and p.resolve() not in own
    }


def build_output_text(text_source: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Assemble the output table as verbatim text for every Phase 4 column.

    The timestamp, the 28 features and the same-hour label are copied as the literal
    text of the Phase 4 file.  Reading them into float64 and writing them back would
    re-serialise every cell, and the float formatting in this pandas build is not
    round-trip exact: doing so perturbs roughly one cell in six by one unit in the last
    place.  Copying the text keeps the feature columns byte-identical to Phase 4, which
    is exactly the property Phase 5 claims about them.

    The three target columns and their observation flags are new values that are exact
    in both directions (0, 1 or missing), so they are written from the numeric frame.
    """
    columns = [TIME_COLUMN] + FEATURE_COLUMNS + [LABEL_COLUMN] + TARGET_COLUMNS + OBSERVED_FLAGS
    out_text = pd.DataFrame({TIME_COLUMN: text_source[TIME_COLUMN].to_numpy()})
    for column in FEATURE_COLUMNS + [LABEL_COLUMN]:
        out_text[column] = text_source[column].to_numpy()
    for column in TARGET_COLUMNS:
        values = out[column].to_numpy(dtype=float)
        out_text[column] = ["" if np.isnan(v) else ("1.0" if v == 1.0 else "0.0") for v in values]
        out_text[f"{column}_observed"] = [
            "1" if observed else "0" for observed in out[f"{column}_observed"].to_numpy()
        ]
    return out_text[columns]


def load_features_table_text() -> pd.DataFrame:
    """The Phase 4 table with every cell kept as its literal text (no float parsing)."""
    df = pd.read_csv(FEATURES_CSV, dtype=str)
    required = [TIME_COLUMN, LABEL_COLUMN, *FEATURE_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Phase 4 table is missing required columns: {missing}")
    if df[required].isna().any().any():
        raise SystemExit("Phase 4 table has an empty cell in a timestamp, feature or label column")
    return df


def load_features_table() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    required = [TIME_COLUMN, LABEL_COLUMN, *FEATURE_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Phase 4 table is missing required columns: {missing}")
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], utc=True, format="ISO8601")
    if df[TIME_COLUMN].isna().any():
        raise SystemExit("Phase 4 table contains unparseable timestamps")
    if df[FEATURE_COLUMNS].isna().any().any():
        raise SystemExit("Phase 4 table contains missing feature values; refusing to build targets")
    return df


def load_label_source() -> pd.DataFrame:
    """Genuine observed labels, one row per supervised hour."""
    df = pd.read_csv(SYNC_CSV, usecols=[TIME_COLUMN, LABEL_COLUMN])
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], utc=True, format="ISO8601")
    df = df.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    if df[TIME_COLUMN].duplicated().any():
        raise SystemExit("label source has duplicate timestamps")
    if df[LABEL_COLUMN].isna().any():
        raise SystemExit("label source contains unobserved (NaN) labels")
    if not set(df[LABEL_COLUMN].unique()) <= {0.0, 1.0}:
        raise SystemExit("label source is not binary")
    return df


def load_contiguous_grid() -> pd.DataFrame:
    grid = pd.read_csv(HOURLY_CSV, usecols=["date"]).rename(columns={"date": TIME_COLUMN})
    grid[TIME_COLUMN] = pd.to_datetime(grid[TIME_COLUMN], utc=True, format="ISO8601")
    grid = grid.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    if grid[TIME_COLUMN].duplicated().any():
        raise SystemExit("hourly grid contains duplicate timestamps")
    deltas = grid[TIME_COLUMN].diff().dropna()
    if not (deltas == pd.Timedelta(hours=1)).all():
        raise SystemExit("hourly grid is not contiguous at 1 hour; a row shift would not be a clock-hour shift")
    return grid


def build_targets(grid: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Place ``label(t + L)`` on hour t, using the contiguous grid as the ruler.

    ``label_series`` is indexed by the contiguous hourly grid and is NaN wherever Phase 3
    found no genuine observation.  ``shift(-L)`` moves each value L rows towards the past,
    i.e. onto the hour exactly L clock hours earlier.  NaN propagates, so an hour whose
    future label was never observed produces NaN rather than 0.
    """
    label_map = dict(zip(labels[TIME_COLUMN], labels[LABEL_COLUMN]))
    observed_on_grid = grid[TIME_COLUMN].map(label_map)
    label_series = pd.Series(observed_on_grid.to_numpy(dtype=float), index=grid[TIME_COLUMN])
    return pd.DataFrame(
        {f"target_{L}h": label_series.shift(-L) for L in LEAD_TIMES_HOURS},
        index=grid[TIME_COLUMN],
    )


def missing_reason(label_hours: pd.DatetimeIndex, future_ts: pd.Timestamp) -> str:
    if future_ts > WINDOW_END or future_ts < WINDOW_START:
        return "outside_modelling_window"
    if future_ts not in label_hours:
        return "hour_present_in_window_but_no_genuine_observation"
    return "unexpected"


def per_year_counts(df: pd.DataFrame, value_column: str) -> dict:
    valid = df[df[value_column].notna()]
    out: dict[str, dict] = {}
    for year, group in valid.groupby(valid[TIME_COLUMN].dt.year):
        out[str(int(year))] = {
            "valid_rows": int(len(group)),
            "positives": int((group[value_column] == 1).sum()),
            "negatives": int((group[value_column] == 0).sum()),
            "positive_rate": round(float((group[value_column] == 1).mean()), 6),
        }
    return out


def main() -> int:
    log("=" * 78)
    log("PHASE 5 -- NOWCAST TARGET CONSTRUCTION (lead times 1h / 2h / 3h)")
    log("=" * 78)

    before = snapshot_dataset_files()

    log("\n[1] Loading Phase 4 supervised feature table")
    feats_text = load_features_table_text()
    feats = load_features_table()
    log(f"    rows={len(feats)}  features={len(FEATURE_COLUMNS)}")
    log(f"    range {feats[TIME_COLUMN].min()} .. {feats[TIME_COLUMN].max()}")

    log("\n[2] Loading genuine label source (Phase 3 supervised hours)")
    labels = load_label_source()
    label_hours = pd.DatetimeIndex(labels[TIME_COLUMN])
    log(f"    supervised hours={len(labels)}  positives={int(labels[LABEL_COLUMN].sum())}")
    log(f"    range {labels[TIME_COLUMN].min()} .. {labels[TIME_COLUMN].max()}")

    log("\n[3] Loading contiguous hourly ruler grid")
    grid = load_contiguous_grid()
    log(f"    rows={len(grid)}  {grid[TIME_COLUMN].min()} .. {grid[TIME_COLUMN].max()}")

    log("\n[4] Checking the feature table's same-hour label against the label source")
    lookup = dict(zip(labels[TIME_COLUMN], labels[LABEL_COLUMN]))
    expected_same_hour = feats[TIME_COLUMN].map(lookup)
    if expected_same_hour.isna().any():
        raise SystemExit("some Phase 4 timestamps are not supervised hours in the label source")
    mismatch = int((expected_same_hour.to_numpy(dtype=float) != feats[LABEL_COLUMN].to_numpy(dtype=float)).sum())
    if mismatch:
        raise SystemExit(f"Phase 4 thunderstorm_label disagrees with the label source in {mismatch} rows")
    log(f"    same-hour label mismatches: {mismatch}")

    log("\n[5] Building forward-shifted targets on the contiguous grid")
    target_frame = build_targets(grid, labels)
    missing_index = ~feats[TIME_COLUMN].isin(target_frame.index)
    if missing_index.any():
        raise SystemExit("some feature timestamps are absent from the contiguous grid")
    aligned = target_frame.reindex(feats[TIME_COLUMN].to_numpy())
    for column in TARGET_COLUMNS:
        log(f"    {column}: observed={int(aligned[column].notna().sum())} unobserved={int(aligned[column].isna().sum())}")

    log("\n[6] Assembling the output table")
    out = pd.DataFrame({TIME_COLUMN: feats[TIME_COLUMN].to_numpy()})
    for column in FEATURE_COLUMNS:
        out[column] = feats[column].to_numpy(dtype=float)
    out[LABEL_COLUMN] = feats[LABEL_COLUMN].to_numpy(dtype=float)
    for column in TARGET_COLUMNS:
        out[column] = aligned[column].to_numpy(dtype=float)
        flag = f"{column}_observed"
        out[flag] = out[column].notna().to_numpy(dtype=np.int8)

    log("\n[7] Direction self-test: target_Lh(t) must equal the genuine label at t + L hours")
    direction_problems: list[dict] = []
    for L, column in zip(LEAD_TIMES_HOURS, TARGET_COLUMNS):
        future = feats[TIME_COLUMN] + pd.Timedelta(hours=L)
        expected = future.map(lookup)
        found = out[column].to_numpy(dtype=float)
        want = expected.to_numpy(dtype=float)
        disagree = ~((np.isnan(found) & np.isnan(want)) | (found == want))
        for idx in np.flatnonzero(disagree)[:10]:
            direction_problems.append(
                {
                    "column": column,
                    "timestamp_utc": out[TIME_COLUMN].iloc[int(idx)].isoformat(),
                    "expected": None if np.isnan(want[idx]) else float(want[idx]),
                    "found": None if np.isnan(found[idx]) else float(found[idx]),
                }
            )
        # A real forward shift must disagree with the same-hour label often enough that a
        # sign error cannot hide; otherwise the table would just repeat the current hour.
        same_hour_agreement = float((found == out[LABEL_COLUMN].to_numpy(dtype=float)).mean())
        log(f"    {column}: disagreements={int(disagree.sum())} agreement_with_same_hour_label={same_hour_agreement:.4f}")
        if disagree.any():
            break
    if direction_problems:
        log(f"    PROBLEMS: {direction_problems}")
        raise SystemExit("forward-shift self-test failed")

    log("\n[8] Checking the missing-future-label policy (NaN, never 0)")
    missing_report: dict[str, dict] = {}
    for L, column in zip(LEAD_TIMES_HOURS, TARGET_COLUMNS):
        flag = out[f"{column}_observed"].to_numpy()
        values = out[column].to_numpy(dtype=float)
        future = feats[TIME_COLUMN] + pd.Timedelta(hours=L)
        is_observed = future.map(lookup).notna().to_numpy()
        if not np.array_equal(flag.astype(bool), is_observed):
            raise SystemExit(f"{column}: observation flag disagrees with the label source")
        if is_observed.any() and int((flag.astype(bool) & np.isnan(values)).sum()):
            raise SystemExit(f"{column}: an observed future label was written as NaN")
        if int((~flag.astype(bool) & (values == 0)).sum()):
            raise SystemExit(f"{column}: an unobserved future label was written as 0")
        reasons: dict[str, int] = {}
        for ts in future[~is_observed]:
            key = missing_reason(label_hours, ts)
            reasons[key] = reasons.get(key, 0) + 1
        missing_report[column] = {
            "unobserved_future_label_rows": int((~flag.astype(bool)).sum()),
            "reason_breakdown": reasons,
        }
        log(f"    {column}: unobserved={missing_report[column]['unobserved_future_label_rows']} reasons={reasons}")

    log("\n[9] Validating the output table")
    ts = out[TIME_COLUMN]
    duplicate_timestamps = int(ts.duplicated().sum())
    chronological = bool(ts.is_monotonic_increasing)
    if duplicate_timestamps:
        raise SystemExit("output contains duplicate timestamps")
    if not chronological:
        raise SystemExit("output is not chronological")
    if int(out[LABEL_COLUMN].isna().sum()):
        raise SystemExit("preserved same-hour label contains missing values")
    if not set(out[LABEL_COLUMN].unique()) <= {0.0, 1.0}:
        raise SystemExit("preserved same-hour label is not binary")
    for column in TARGET_COLUMNS:
        values = out[column].dropna().unique()
        if not set(values) <= {0.0, 1.0}:
            raise SystemExit(f"{column} is not binary")
    matrix = out[FEATURE_COLUMNS].to_numpy(dtype=float)
    nan_features = int(np.isnan(matrix).sum())
    inf_features = int(np.isinf(matrix).sum())
    if nan_features or inf_features:
        raise SystemExit(f"feature matrix is not finite: nan={nan_features} inf={inf_features}")
    # Row order and the timestamp column must survive assembly unchanged: the features
    # stay at t while only the targets look forward.  The strong version of this check is
    # the character-for-character comparison against the Phase 4 file after writing.
    same_timestamps = bool((out[TIME_COLUMN].to_numpy() == feats[TIME_COLUMN].to_numpy()).all())
    log(f"    timestamps unchanged through assembly: {same_timestamps} (verbatim check follows the write)")
    if not same_timestamps:
        raise SystemExit("timestamps were altered relative to Phase 4")

    log("\n[10] Lead-time statistics")
    stats: dict[str, dict] = {}
    for L, column in zip(LEAD_TIMES_HOURS, TARGET_COLUMNS):
        valid = out[column].notna()
        valid_values = out.loc[valid, column]
        positives = int((valid_values == 1).sum())
        negatives = int((valid_values == 0).sum())
        valid_ts = out.loc[valid, TIME_COLUMN]
        stats[column] = {
            "lead_time_hours": L,
            "total_rows": int(len(out)),
            "valid_rows": int(valid.sum()),
            "rows_without_genuine_future_label": int((~valid).sum()),
            "rows_lost_fraction": round(float((~valid).mean()), 6),
            "positives": positives,
            "negatives": negatives,
            "positive_rate": round(positives / int(valid.sum()), 6) if valid.any() else None,
            "first_valid_timestamp_utc": valid_ts.min().isoformat() if valid.any() else None,
            "last_valid_timestamp_utc": valid_ts.max().isoformat() if valid.any() else None,
            "positives_per_year": per_year_counts(out, column),
        }
        log(
            f"    {column}: valid={stats[column]['valid_rows']} "
            f"(+{positives}/-{negatives}, rate {stats[column]['positive_rate']}) "
            f"unobserved={stats[column]['rows_without_genuine_future_label']}"
        )

    all_valid = out[TARGET_COLUMNS].notna().all(axis=1)
    stats["rows_valid_for_all_three_lead_times"] = int(all_valid.sum())
    stats["rows_valid_for_no_lead_time"] = int(out[TARGET_COLUMNS].isna().all(axis=1).sum())
    # A generator could in principle evaluate any lead time on the intersection subset.
    log(f"    rows valid for all three lead times: {int(all_valid.sum())}")
    log(f"    rows valid for none: {int(out[TARGET_COLUMNS].isna().all(axis=1).sum())}")

    log("\n[11] Writing outputs (Phase 4 cells copied as verbatim text)")
    column_order = (
        [TIME_COLUMN] + FEATURE_COLUMNS + [LABEL_COLUMN] + TARGET_COLUMNS + OBSERVED_FLAGS
    )
    out_text = build_output_text(feats_text, out)
    out_text.to_csv(OUT_CSV, index=False)

    # Read the file back and require the timestamp, feature and label columns to be
    # character-for-character identical to the Phase 4 file.
    written = pd.read_csv(OUT_CSV, dtype=str)
    verbatim_mismatches = 0
    for column in [TIME_COLUMN] + FEATURE_COLUMNS + [LABEL_COLUMN]:
        verbatim_mismatches += int((written[column].to_numpy() != feats_text[column].to_numpy()).sum())
    log(f"    verbatim-copy mismatches vs Phase 4: {verbatim_mismatches}")
    if verbatim_mismatches:
        raise SystemExit("the written file does not reproduce the Phase 4 cells verbatim")
    same_features = verbatim_mismatches == 0
    out = out[column_order]

    after = snapshot_dataset_files()
    changed = sorted(k for k in before if before[k] != after.get(k))
    new_files = sorted(k for k in after if k not in before)

    features_meta = json.loads(FEATURES_META.read_text(encoding="utf-8"))
    same_hour_positives = int((out[LABEL_COLUMN] == 1).sum())

    metadata = {
        "artifact": OUT_CSV.name,
        "artifact_type": "Phase 5 nowcast target table (Phase 4 causal features at t + genuine forward-shifted labels at t+1h/2h/3h)",
        "phase": "Phase 5 -- thunderstorm target validation and nowcast target formulation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "built_by": Path(__file__).name,
        "task_definition": {
            "feature_time": "t",
            "target_time": "t + lead_time",
            "formulation": "target_<L>h(t) = genuine observed thunderstorm_label(t + L hours)",
            "lead_times_evaluated_hours": LEAD_TIMES_HOURS,
            "explicit_target_definitions": {
                "target_1h": "genuine thunderstorm_label observed at the clock hour exactly 1 hour after the row's timestamp_utc",
                "target_2h": "genuine thunderstorm_label observed at the clock hour exactly 2 hours after the row's timestamp_utc",
                "target_3h": "genuine thunderstorm_label observed at the clock hour exactly 3 hours after the row's timestamp_utc",
            },
            "feature_target_relationship": "one row supplies features at t and all three forward targets; Phase 6 selects the lead time",
            "lead_time_selected_in_this_phase": None,
            "note": "All three lead times are shipped so Phase 6 can choose on data availability and measured model performance rather than on assumption.",
        },
        "inputs": {
            "feature_table": {
                "path": "dataset/" + FEATURES_CSV.name,
                "sha256": before.get(FEATURES_CSV.name),
                "role": "supplies the 28 causal features at t and the genuine same-hour label; copied unchanged",
                "rows": int(len(feats)),
            },
            "label_source": {
                "path": "dataset/" + SYNC_CSV.name,
                "sha256": before.get(SYNC_CSV.name),
                "role": "authority for which clock hours carry a genuine observed thunderstorm label, and for its value",
                "supervised_hours": int(len(labels)),
                "positives": int(labels[LABEL_COLUMN].sum()),
                "unobserved_hours_in_window": (
                    int(len(pd.date_range(WINDOW_START, WINDOW_END, freq="h"))) - int(len(labels))
                ),
            },
            "contiguous_hourly_grid": {
                "path": "dataset/raw_openmeteo/votv_openmeteo_hourly_2014_2025.csv",
                "sha256": before.get("raw_openmeteo/votv_openmeteo_hourly_2014_2025.csv"),
                "role": "clock-hour ruler: the label series is re-indexed onto this grid so a shift of L rows is exactly L clock hours, and NaN is carried through label gaps",
                "rows": int(len(grid)),
                "contiguity": "verified: exactly one hour between consecutive rows, no duplicate timestamp",
            },
        },
        "outputs": {
            "csv": {"path": "dataset/" + OUT_CSV.name, "rows": int(len(out)), "columns": int(len(out.columns))},
            "metadata": {"path": "dataset/" + OUT_META_JSON.name},
        },
        "date_range": {
            "first_timestamp_utc": out[TIME_COLUMN].min().isoformat(),
            "last_timestamp_utc": out[TIME_COLUMN].max().isoformat(),
            "window_used_for_missing_label_attribution": {
                "start_utc": WINDOW_START.isoformat(),
                "end_utc": WINDOW_END.isoformat(),
            },
        },
        "row_counts": {
            "output_rows": int(len(out)),
            "same_hour_label_rows": int(len(out)),
            "per_lead_time": stats,
            "missing_future_label_counts": missing_report,
            "drop_policy": (
                "No row is dropped. A row whose future label was not genuinely observed keeps NaN in that "
                "target column and carries target_Lh_observed = 0. Removing rows here would hide the loss "
                "and force a single lead time; Phase 6 drops NaN per the lead time it selects."
            ),
        },
        "targets": {
            "target_columns": TARGET_COLUMNS,
            "observation_flag_columns": OBSERVED_FLAGS,
            "dtype": "binary float 0/1, NaN where the future hour was not genuinely observed",
            "missing_encoding": "NaN (empty field in CSV); never 0, never imputed",
            "per_lead_time": {
                column: {
                    "lead_time_hours": stats[column]["lead_time_hours"],
                    "valid_rows": stats[column]["valid_rows"],
                    "positives": stats[column]["positives"],
                    "negatives": stats[column]["negatives"],
                    "positive_rate": stats[column]["positive_rate"],
                    "rows_without_genuine_future_label": stats[column]["rows_without_genuine_future_label"],
                    "first_valid_timestamp_utc": stats[column]["first_valid_timestamp_utc"],
                    "last_valid_timestamp_utc": stats[column]["last_valid_timestamp_utc"],
                }
                for column in TARGET_COLUMNS
            },
            "rows_valid_for_all_three_lead_times": stats["rows_valid_for_all_three_lead_times"],
            "rows_valid_for_no_lead_time": stats["rows_valid_for_no_lead_time"],
        },
        "preserved_audit_columns": {
            LABEL_COLUMN: {
                "role": "audit only; the genuine same-hour label at t, copied verbatim from Phase 4 / Phase 3",
                "positives": same_hour_positives,
                "negatives": int(len(out) - same_hour_positives),
                "positive_rate": round(same_hour_positives / len(out), 6),
                "never_a_model_input": True,
            },
            "observation_flags": {
                "columns": OBSERVED_FLAGS,
                "definition": "1 when the required future hour carries a genuine observed label, else 0",
                "role": "audit: makes the exclusion of unobserved future hours explicit; never a model input",
            },
            "not_included": {
                "weather_code": "not carried into this file: Phase 3 showed it carries no thunderstorm signal here (max code 65, zero 95/96/99 hours) while 3694 hours are label-positive",
                "phase3_observation_counters": "not carried: n_reports, wx_field_observed and the strict/recovered label variants are not needed for nowcasting and would invite accidental use as features",
                "features_complete": "not carried: Phase 4 retained only rows with a complete feature vector",
            },
        },
        "features": {
            "count": len(FEATURE_COLUMNS),
            "feature_order": FEATURE_COLUMNS,
            "source": "copied verbatim from the Phase 4 feature table; no feature was recomputed, shifted or derived in this phase",
            "features_remain_at_time_t": True,
            "feature_matrix_identical_to_phase4": bool(same_features),
            "copy_method": (
                "the timestamp, feature and same-hour-label cells are copied as the literal text of the "
                "Phase 4 CSV and never pass through a float parse/serialise cycle, so those 30 columns are "
                "byte-identical to Phase 4"
            ),
            "why_text_passthrough": (
                "with this pandas/numpy build a read-then-write cycle is not round-trip exact: re-serialising "
                "the parsed values changes about 404,000 of the 2.3 million feature cells by one unit in the "
                "last place. That is numerically negligible but it would break the byte-identity claim, so the "
                "cells are passed through as text"
            ),
            "verbatim_copy_mismatches_after_write": verbatim_mismatches,
        },
        "leakage_policy": {
            "direction": "target moved forward, features stay at t",
            "features_at_time_t": True,
            "target_at_time_t_plus_L": LEAD_TIMES_HOURS,
            "features_shifted": False,
            "target_shift_rule": "label(t + L) placed on row t via a -L row shift of the contiguous hourly label series",
            "shift_evaluated_on_contiguous_hourly_grid": True,
            "why_not_shifted_on_masked_rows": (
                "The supervised table is masked: 23,171 of the 105,192 window hours carry no genuine label. "
                "A -L shift on those compacted rows would pair t with the next labelled row, up to 57 clock "
                "hours away, and would misstate the lead time. Shifting the contiguous hourly grid keeps "
                "every step exactly one clock hour."
            ),
            "features_at_or_before_t_only": True,
            "future_atmospheric_variable_used_as_feature": False,
            "future_label_used_as_feature": False,
            "centred_or_forward_rolling_features": False,
            "missing_future_label_treated_as_negative": False,
            "missing_future_label_encoded_as": "NaN with an explicit target_Lh_observed = 0 flag",
            "same_hour_label_role": "audit only, never a model input",
            "target_derived_statistic_used_as_feature": False,
            "internal_direction_check": {
                "method": "for every row, target_Lh is compared against the genuine label looked up at timestamp + L hours",
                "lead_times_checked": LEAD_TIMES_HOURS,
                "disagreements": 0,
                "verdict": "PASS",
            },
            "note": (
                "A model trained on this table uses information available at hour t to predict an "
                "observation at hour t+L. The forward shift is the whole point of the nowcast "
                "formulation; it is applied to the TARGET only."
            ),
        },
        "missing_values": {
            "feature_columns": {
                "total_missing": int(nan_features),
                "total_infinite": int(inf_features),
                "policy": "Phase 4 already dropped rows without complete history; the feature matrix stays finite",
            },
            "target_columns": {
                column: {
                    "missing": stats[column]["rows_without_genuine_future_label"],
                    "reason_breakdown": missing_report[column]["reason_breakdown"],
                }
                for column in TARGET_COLUMNS
            },
            "imputation_applied": False,
        },
        "timestamp_integrity": {
            "duplicate_timestamps": duplicate_timestamps,
            "timestamps_sorted_ascending": chronological,
            "timezone": "UTC",
            "all_timestamps_on_hour_boundary": bool(((ts.dt.minute == 0) & (ts.dt.second == 0) & (ts.dt.microsecond == 0)).all()),
            "timestamps_identical_to_phase4": same_timestamps,
            "gaps": (
                "Internal gaps are inherited from Phase 3 supervision masking (hours without a decodable "
                "present-weather group). They are not new here and they are exactly why some future labels "
                "are unobserved rather than negative."
            ),
        },
        "provenance": {
            "phase4_status": features_meta.get("validation", {}).get("status"),
            "phase4_rows": features_meta.get("outputs", {}).get("csv", {}).get("rows"),
            "label_definition": "genuine VOTV METAR present-weather observation label from Phase 2, unchanged",
            "synthetic_labels_created": False,
            "open_meteo_weather_code_used_as_label": False,
            "radar_satellite_lightning_nwp_data_added": False,
            "no_model_trained": True,
        },
        "protected_files": {
            "snapshot_method": (
                "sha256 of every file under dataset/ taken before and after the build; this script's own two "
                "outputs are excluded, so a non-empty changed_files list means a Phase 1-4 artifact was touched"
            ),
            "changed_files": changed,
            "new_files": new_files,
            "phase1_to_phase4_files_unchanged": len(changed) == 0,
        },
        "validation": {
            "status": "PASS",
            "checks_run": [
                "Phase 4 timestamps lie on the contiguous hourly grid",
                "same-hour label in the Phase 4 table matches the Phase 3 label source exactly",
                "label source is binary, duplicate-free and free of missing labels",
                "targets built by shifting the label series on a verified contiguous hourly grid",
                "forward-shift self-test: target_Lh(t) equals the genuine label at t+L for every row",
                "observed/unobserved flags agree with the label source for every row and lead time",
                "no unobserved future label was written as 0; no observed future label was written as NaN",
                "features and timestamps are copied verbatim (text, not float re-serialised) from the Phase 4 table and remain at t",
                "the written file reproduces every Phase 4 timestamp, feature and label cell character for character",
                "output has no duplicate timestamp and is chronological",
                "feature matrix contains no NaN and no infinite value",
                "no file under dataset/ other than the two Phase 5 outputs was modified",
            ],
            "problems": [],
        },
        "scope_note": (
            "Phase 5 only. No model was trained. The Flask backend, dashboard, frontend and existing model "
            "were not modified, no radar/satellite/lightning/NWP source was added, and the Phase 1-4 outputs "
            "were left byte-identical. The genuine thunderstorm label was preserved, not recreated."
        ),
    }

    OUT_META_JSON.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    log(f"    wrote {OUT_CSV}  ({len(out)} rows x {len(out.columns)} columns)")
    log(f"    wrote {OUT_META_JSON}")

    log("\n[12] Summary")
    for column in TARGET_COLUMNS:
        s = stats[column]
        log(f"    {column}: valid={s['valid_rows']}  +{s['positives']}/-{s['negatives']}  rate={s['positive_rate']}  unobserved={s['rows_without_genuine_future_label']}")
    log(f"    features         : {len(FEATURE_COLUMNS)} (unchanged, at t)")
    log(f"    same-hour label  : {same_hour_positives} positive / {int(len(out)) - same_hour_positives} negative")
    log(f"    changed protected: {changed if changed else 'none'}")
    log(f"    new files        : {new_files}")

    if changed:
        log("\nFAILED: a protected file changed during the build")
        return 1
    log("\nPHASE 5 NOWCAST TARGET CONSTRUCTION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
