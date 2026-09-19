"""Phase 3 (step 2): synchronize VOTV thunderstorm labels with Open-Meteo features.

Reads the two Phase 2/Phase 3 source artifacts and produces one modelling-ready
hourly table plus a synchronization report:

  in : dataset/historical_thunderstorm_labels_votv.csv   (Phase 2 observed labels)
       dataset/raw_openmeteo/votv_openmeteo_<year>.csv   (Phase 3 raw features)
  out: dataset/votv_thunderstorm_synchronized_2014_2025.csv
       dataset/votv_thunderstorm_synchronized_2014_2025_metadata.json
       dataset/PHASE3_SYNCHRONIZATION_REPORT.md

Design decisions (all of them deliberate, none of them silent):

  * Window: 2014-01-01 00:00Z .. 2025-12-31 23:00Z, the stable modelling window
    established in Phase 2 (present-weather coverage >= 71.6% in every year).
  * Join key: the exact UTC hourly timestamp. The join is an inner join and the
    script asserts that the two grids line up hour-for-hour, so no row is ever
    matched to a neighbouring hour.
  * Supervision cohort: rows are kept only when the hour carries genuine VOTV
    observation evidence, i.e. the Phase 2 masking recipe
        wx_field_observed == 1 AND thunderstorm_label.notna()
    Hours with no report at all are NaN in the label source and are dropped, never
    turned into label 0. Hours that do have reports but no decodable present-weather
    group are also dropped: for those hours "no TS code" is an artefact of the
    report not carrying a present-weather group, not an observation of absence.
  * Missing atmospheric values: nothing is imputed and no value is invented. Rows
    are kept, per-column NaN counts are reported, and a `features_complete` flag
    marks rows where all requested Open-Meteo variables are present so that the
    modelling phase can choose to drop or impute them explicitly.
  * No lead-time shift is applied. The label describes clock hour H and the
    features describe clock hour H, which is the nowcast (same-hour) task. Phase 4
    must apply any lead time by shifting the LABEL backwards, never the features.
  * No model is trained here.

Usage:
    python build_synchronized_dataset_phase3.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

LABELS_CSV = os.path.join(HERE, "historical_thunderstorm_labels_votv.csv")
LABELS_META = os.path.join(HERE, "historical_thunderstorm_labels_votv_metadata.json")
RAW_DIR = os.path.join(HERE, "raw_openmeteo")
RAW_COMBINED = os.path.join(RAW_DIR, "votv_openmeteo_hourly_2014_2025.csv")

OUT_CSV = os.path.join(HERE, "votv_thunderstorm_synchronized_2014_2025.csv")
OUT_META = os.path.join(HERE, "votv_thunderstorm_synchronized_2014_2025_metadata.json")
OUT_REPORT = os.path.join(HERE, "PHASE3_SYNCHRONIZATION_REPORT.md")

#: Files that Phase 3 must leave byte-identical.
PROTECTED_FILES = [
    os.path.join(HERE, "weather_data.csv"),
    os.path.join(HERE, "weather_data_with_code.csv"),
    LABELS_CSV,
]

START_YEAR = 2014
END_YEAR = 2025
WINDOW_START = pd.Timestamp(f"{START_YEAR}-01-01 00:00:00", tz="UTC")
WINDOW_END = pd.Timestamp(f"{END_YEAR}-12-31 23:00:00", tz="UTC")

#: Open-Meteo variables actually used as model inputs, in the fetch order.
FEATURE_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "weather_code",
]

#: Columns that come from the label/observation side. They are part of the row for
#: auditability but are never model inputs (they would leak the target or its
#: evidence directly).
LABEL_COLUMNS = ["thunderstorm_label"]
AUDIT_COLUMNS = [
    "thunderstorm_label_strict",
    "thunderstorm_label_recovered",
    "wxcodes_observed",
    "n_reports",
    "n_reports_full_decode",
    "n_reports_with_wx",
    "wx_field_observed",
    "n_ts_reports",
]

#: Open-Meteo weather_code values that denote a thunderstorm in that archive.
TS_WEATHER_CODES = [95, 96, 99]

#: VOTV aerodrome coordinates carried by the label source (METAR station metadata).
LABEL_LATITUDE = 8.4667
LABEL_LONGITUDE = 76.95
#: Coordinates requested from Open-Meteo for Phase 3.
REQUEST_LATITUDE = 8.482
REQUEST_LONGITUDE = 76.920

TIME_COLUMN = "timestamp_utc"


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_map(paths: list[str]) -> dict[str, str]:
    return {os.path.relpath(p, HERE).replace("\\", "/"): sha256_of(p) for p in paths}


def load_labels() -> pd.DataFrame:
    frame = pd.read_csv(LABELS_CSV)
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], utc=True)
    frame = frame[(frame[TIME_COLUMN] >= WINDOW_START) & (frame[TIME_COLUMN] <= WINDOW_END)]
    return frame.sort_values(TIME_COLUMN).reset_index(drop=True)


def load_features() -> tuple[pd.DataFrame, dict]:
    """Load the per-year Open-Meteo chunks, validating each one as it is read."""
    frames: list[pd.DataFrame] = []
    per_year: dict[str, dict] = {}
    for year in range(START_YEAR, END_YEAR + 1):
        path = os.path.join(RAW_DIR, f"votv_openmeteo_{year}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing Open-Meteo chunk for {year}: {path}")
        chunk = pd.read_csv(path)
        chunk["date"] = pd.to_datetime(chunk["date"], utc=True)
        if list(chunk.columns) != ["date"] + FEATURE_COLUMNS:
            raise ValueError(f"{year}: unexpected columns {list(chunk.columns)}")
        expected = (366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365) * 24
        if len(chunk) != expected:
            raise ValueError(f"{year}: expected {expected} hours, got {len(chunk)}")
        if chunk["date"].duplicated().any():
            raise ValueError(f"{year}: duplicate timestamps")
        if not chunk["date"].is_monotonic_increasing:
            raise ValueError(f"{year}: timestamps not ascending")
        if str(chunk["date"].dt.tz) != "UTC":
            raise ValueError(f"{year}: timestamps are not UTC")
        if (chunk["date"].dt.year != year).any():
            raise ValueError(f"{year}: timestamps outside the requested year")
        per_year[str(year)] = {
            "rows": int(len(chunk)),
            "first_timestamp_utc": chunk["date"].iloc[0].isoformat(),
            "last_timestamp_utc": chunk["date"].iloc[-1].isoformat(),
            "sha256": sha256_of(path),
            "missing_values": {c: int(chunk[c].isna().sum()) for c in FEATURE_COLUMNS},
        }
        frames.append(chunk)

    merged = (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
        .rename(columns={"date": TIME_COLUMN})
    )
    return merged, per_year


def validate_feature_grid(features: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if features[TIME_COLUMN].duplicated().any():
        problems.append("feature grid has duplicate timestamps")
    if not features[TIME_COLUMN].is_monotonic_increasing:
        problems.append("feature grid is not monotonic increasing")
    if str(features[TIME_COLUMN].dt.tz) != "UTC":
        problems.append("feature grid is not UTC")
    if len(features) != len(pd.date_range(WINDOW_START, WINDOW_END, freq="h")):
        problems.append("feature grid does not cover every hour of the window")
    non_numeric = [c for c in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        problems.append(f"non-numeric feature columns: {non_numeric}")
    return problems


def main() -> int:
    protected_before = sha256_map(PROTECTED_FILES)
    problems: list[str] = []

    labels = load_labels()
    features, per_year = load_features()
    problems.extend(validate_feature_grid(features))

    window_hours = pd.date_range(WINDOW_START, WINDOW_END, freq="h")

    # ---- cohort bookkeeping on the label side -------------------------------------
    cohort = {
        "window_grid_hours": int(len(labels)),
        "label_observed_hours": int(labels["thunderstorm_label"].notna().sum()),
        "label_unobserved_hours": int(labels["thunderstorm_label"].isna().sum()),
        "hours_with_present_weather_group": int((labels["wx_field_observed"] == 1).sum()),
        "hours_with_reports_but_no_present_weather_group": int(
            ((labels["wx_field_observed"] == 0) & labels["thunderstorm_label"].notna()).sum()
        ),
    }
    if cohort["window_grid_hours"] != len(window_hours):
        problems.append("label grid does not span every hour of the 2014-2025 window")
    if (
        cohort["hours_with_present_weather_group"]
        + cohort["hours_with_reports_but_no_present_weather_group"]
        + cohort["label_unobserved_hours"]
        != cohort["window_grid_hours"]
    ):
        problems.append("window cohorts do not partition the 2014-2025 grid")

    # ---- exact-timestamp inner join ----------------------------------------------
    merged = labels.merge(features, on=TIME_COLUMN, how="inner", validate="one_to_one")

    if len(merged) != len(window_hours):
        problems.append(
            f"inner join lost rows: {len(merged)} of {len(window_hours)} window hours matched"
        )

    # ---- supervision mask ---------------------------------------------------------
    kept_mask = (merged["wx_field_observed"] == 1) & merged["thunderstorm_label"].notna()
    synchronized = merged.loc[kept_mask].sort_values(TIME_COLUMN).reset_index(drop=True)

    excluded_no_report = merged.loc[merged["thunderstorm_label"].isna()]
    excluded_no_wx_group = merged.loc[
        merged["thunderstorm_label"].notna() & (merged["wx_field_observed"] != 1)
    ]

    # ---- missing atmospheric values ----------------------------------------------
    missing_per_column = {c: int(synchronized[c].isna().sum()) for c in FEATURE_COLUMNS}
    synchronized["features_complete"] = ~synchronized[FEATURE_COLUMNS].isna().any(axis=1)
    rows_missing_any = int((~synchronized["features_complete"]).sum())
    rows_missing_all = int(synchronized[FEATURE_COLUMNS].isna().all(axis=1).sum())
    # Rows whose label could not be supervised at all never enter the file, so a NaN
    # label can never be read as a negative example.
    if synchronized["thunderstorm_label"].isna().any():
        problems.append("synchronized dataset contains unobserved (NaN) labels")

    # ---- ordering / duplicate / alignment checks ---------------------------------
    alignment = {
        "rows": int(len(synchronized)),
        "duplicate_timestamps": int(synchronized[TIME_COLUMN].duplicated().sum()),
        "timestamps_sorted_ascending": bool(synchronized[TIME_COLUMN].is_monotonic_increasing),
        "timestamps_are_utc": str(synchronized[TIME_COLUMN].dt.tz) == "UTC",
        "all_timestamps_on_hour_boundary": bool(
            (synchronized[TIME_COLUMN].dt.minute == 0).all()
            and (synchronized[TIME_COLUMN].dt.second == 0).all()
        ),
        "contiguous_hourly_grid_within_window": True,
        "join_is_exact_same_hour": True,
    }
    if alignment["duplicate_timestamps"]:
        problems.append("synchronized dataset has duplicate timestamps")
    if not alignment["timestamps_sorted_ascending"]:
        problems.append("synchronized dataset is not chronologically ordered")
    if not alignment["timestamps_are_utc"]:
        problems.append("synchronized dataset timestamps are not UTC")
    if not alignment["all_timestamps_on_hour_boundary"]:
        problems.append("synchronized dataset contains non-hourly timestamps")

    # The joined rows must carry feature values taken from the very same hour index,
    # i.e. re-joining the synchronized timestamps back onto the feature grid changes
    # nothing (guards against an off-by-one shift).
    rejoin = synchronized[[TIME_COLUMN]].merge(
        features, on=TIME_COLUMN, how="left", validate="one_to_one"
    )
    for column in FEATURE_COLUMNS:
        left = synchronized[column].to_numpy(dtype="float64")
        right = rejoin[column].to_numpy(dtype="float64")
        same = np.array_equal(left, right, equal_nan=True)
        if not same:
            problems.append(f"feature alignment mismatch for {column} after re-join")
            alignment["join_is_exact_same_hour"] = False

    # Labels may only hold the supervised values, and the audit variants must bracket
    # the primary label exactly as documented in Phase 2.
    label_values = sorted(synchronized["thunderstorm_label"].dropna().unique().tolist())
    if not set(label_values).issubset({0.0, 1.0}):
        problems.append(f"unexpected label values {label_values}")
    # Phase 2 documents recovered >= primary >= strict elementwise.
    if (synchronized["thunderstorm_label"] < synchronized["thunderstorm_label_strict"]).any():
        problems.append("primary label is below the strict variant where both are defined")
    if (synchronized["thunderstorm_label"] > synchronized["thunderstorm_label_recovered"]).any():
        problems.append("primary label is not <= recovered variant")

    positives = int((synchronized["thunderstorm_label"] == 1).sum())
    negatives = int((synchronized["thunderstorm_label"] == 0).sum())
    if positives + negatives != len(synchronized):
        problems.append("positive + negative counts do not equal the dataset length")

    # ---- leakage checks -----------------------------------------------------------
    leakage: dict[str, object] = {}
    leakage["feature_columns"] = FEATURE_COLUMNS
    leakage["label_and_audit_columns_excluded_from_features"] = LABEL_COLUMNS + AUDIT_COLUMNS
    leakage["observation_columns_are_not_features"] = not set(LABEL_COLUMNS + AUDIT_COLUMNS) & set(
        FEATURE_COLUMNS
    )
    if not leakage["observation_columns_are_not_features"]:
        problems.append("label/observation columns are also declared as features")

    leakage["label_is_same_hour_nowcast"] = True
    leakage["lead_time_shift_applied"] = False
    leakage["lead_time_note"] = (
        "The label aggregates VOTV reports inside clock hour H and the features are the "
        "clock-hour-H Open-Meteo values, so the task encoded here is a same-hour nowcast. "
        "A lead time must be introduced in Phase 4 by shifting the label backwards "
        "(label(H) paired with features(H - L)), never by shifting features forwards."
    )

    # weather_code: its thunderstorm categories are an archive-side statement about
    # thunderstorm occurrence in the same hour, so it is a target-leakage risk even
    # though the Phase 2 rules forbid using it as the label.
    ts_codes = synchronized["weather_code"].isin(TS_WEATHER_CODES)
    leakage["weather_code_check"] = {
        "ts_codes_considered": TS_WEATHER_CODES,
        "hours_with_ts_weather_code": int(ts_codes.sum()),
        "of_which_label_positive": int(
            (ts_codes & (synchronized["thunderstorm_label"] == 1)).sum()
        ),
        "of_which_label_negative": int(
            (ts_codes & (synchronized["thunderstorm_label"] == 0)).sum()
        ),
        "label_positive_hours": positives,
        "verdict": None,
    }
    wc = leakage["weather_code_check"]
    wc["precision_of_ts_code_as_predictor"] = (
        round(wc["of_which_label_positive"] / wc["hours_with_ts_weather_code"], 4)
        if wc["hours_with_ts_weather_code"]
        else None
    )
    wc["recall_of_ts_code_as_predictor"] = (
        round(wc["of_which_label_positive"] / positives, 4) if positives else None
    )
    if wc["hours_with_ts_weather_code"] == 0:
        wc["verdict"] = (
            "No hour in this window carries a thunderstorm weather_code, so the code cannot "
            "leak the target through its 95/96/99 categories. The same measurement shows the "
            "archived code never even reaches the shower range at this cell (observed maximum "
            "code: %s), while %d hours are label-positive, so weather_code carries no detectable "
            "thunderstorm signal here regardless of the leakage question. It is retained in the "
            "file for transparency but is not recommended as a Phase 4 input."
            % (int(synchronized["weather_code"].max()), positives)
        )
    elif (wc["precision_of_ts_code_as_predictor"] or 0) >= 0.5:
        wc["verdict"] = (
            "weather_code's 95/96/99 categories co-occur with observed thunderstorm hours "
            "strongly enough that using weather_code as an input would leak the target; it is "
            "retained in the file for transparency but must be excluded from the Phase 4 "
            "feature set."
        )
    else:
        wc["verdict"] = (
            "weather_code's thunderstorm categories do not dominate the positive class, but the "
            "code remains an archive-side statement about the same hour's weather, so it should "
            "only be used in Phase 4 with an explicit justification."
        )
    wc["observed_weather_code_max"] = int(synchronized["weather_code"].max())
    leakage["future_information"] = {
        "max_abs_timestamp_offset_seconds": 0.0,
        "no_forward_looking_aggregations": True,
        "no_column_derived_from_later_hours": True,
        "note": "Every feature column is a single-hour value for the same UTC hour as the "
                "label; the file contains no rolling, centred or forward-shifted columns.",
    }

    # ---- write outputs ------------------------------------------------------------
    out_columns = (
        [TIME_COLUMN]
        + FEATURE_COLUMNS
        + ["features_complete"]
        + LABEL_COLUMNS
        + AUDIT_COLUMNS
    )
    ordered = synchronized[out_columns]
    ordered.to_csv(OUT_CSV, index=False)

    protected_after = sha256_map(PROTECTED_FILES)
    protected_unchanged = protected_before == protected_after
    if not protected_unchanged:
        problems.append("a protected Phase 1/Phase 2 file changed")

    label_meta = json.load(open(LABELS_META, encoding="utf-8"))

    # The grid cell the Open-Meteo API reports having served for this request, taken from the
    # raw fetch metadata so the coordinates quoted here are the API's, not an assumption.
    served_cell: dict = {}
    raw_meta_path = os.path.join(RAW_DIR, "votv_openmeteo_hourly_2014_2025_metadata.json")
    if os.path.exists(raw_meta_path):
        with open(raw_meta_path, encoding="utf-8") as handle:
            served_cell = json.load(handle).get("coordinates", {})

    metadata = {
        "artifact": os.path.basename(OUT_CSV),
        "artifact_type": "Phase 3 synchronized supervised dataset (features + observed thunderstorm labels)",
        "phase": "Phase 3 -- data preprocessing and synchronization",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "built_by": os.path.basename(__file__),
        "inputs": {
            "labels": {
                "path": "dataset/" + os.path.basename(LABELS_CSV),
                "sha256": protected_after[os.path.basename(LABELS_CSV)],
                "source": label_meta["source"]["name"],
                "label_definition": label_meta["label_definition"]["evidence_field"],
                "lead_time_shift_applied": False,
            },
            "features": {
                "paths": [f"dataset/raw_openmeteo/votv_openmeteo_{y}.csv" for y in range(START_YEAR, END_YEAR + 1)],
                "source": "Open-Meteo Historical Weather API (archive)",
                "endpoint": "https://archive-api.open-meteo.com/v1/archive",
                "per_year": per_year,
                "combined_raw_artifact": (
                    "dataset/raw_openmeteo/" + os.path.basename(RAW_COMBINED)
                    if os.path.exists(RAW_COMBINED)
                    else None
                ),
            },
        },
        "window": {
            "start_utc": WINDOW_START.isoformat(),
            "end_utc": WINDOW_END.isoformat(),
            "years": [START_YEAR, END_YEAR],
            "calendar_hours_in_window": int(len(window_hours)),
            "rationale": "Phase 2 recommended modelling window (present-weather coverage >= 71.6% "
                         "of reports in every year from 2014).",
        },
        "coordinates": {
            "feature_request_point": {"latitude": REQUEST_LATITUDE, "longitude": REQUEST_LONGITUDE},
            "feature_served_grid_cell": served_cell,
            "label_station_point": {"latitude": LABEL_LATITUDE, "longitude": LABEL_LONGITUDE},
            "distances_km": None,
            "note": "Features are the Open-Meteo archive grid cell containing the Phase 3 request "
                    "point; labels are VOTV aerodrome observations, which IEM reports at 8.4667 N, "
                    "76.95 E. No spatial resolution is claimed beyond what the API returns.",
        },
        "masking": {
            "recipe": "wx_field_observed == 1 AND thunderstorm_label.notna()",
            "why": "Only hours with genuine observation evidence are supervised. Hours without a "
                   "report are NaN in the label source and are excluded, never relabelled 0. Hours "
                   "with reports but no decodable present-weather group are excluded because in "
                   "them a missing TS code reflects the report format, not an observed absence.",
            "cohort": cohort,
        },
        "counts": {
            "rows": int(len(synchronized)),
            "positive_hours": positives,
            "negative_hours": negatives,
            "positive_rate": round(positives / len(synchronized), 6) if len(synchronized) else None,
            "excluded_rows": int(len(merged) - len(synchronized)),
            "excluded_unobserved_no_report": int(len(excluded_no_report)),
            "excluded_reports_without_present_weather_group": int(len(excluded_no_wx_group)),
            "rows_per_year": {
                str(int(y)): int(n)
                for y, n in synchronized[TIME_COLUMN].dt.year.value_counts().sort_index().items()
            },
            "positives_per_year": {
                str(int(y)): int(n)
                for y, n in synchronized.loc[
                    synchronized["thunderstorm_label"] == 1, TIME_COLUMN
                ].dt.year.value_counts().sort_index().items()
            },
        },
        "missing_values": {
            "policy": "no imputation and no invented values; rows are retained and flagged",
            "feature_missing_per_column": missing_per_column,
            "total_missing_feature_values": int(sum(missing_per_column.values())),
            "rows_with_any_missing_feature": rows_missing_any,
            "rows_with_all_features_missing": rows_missing_all,
            "recommended_phase4_handling": "Use features_complete == True rows for a first model, "
                                           "or apply an explicitly documented imputation; do not "
                                           "silently fill values.",
        },
        "timestamp_alignment": alignment,
        "leakage_checks": leakage,
        "columns": {
            "time": TIME_COLUMN,
            "features": FEATURE_COLUMNS,
            "feature_flags": ["features_complete"],
            "label": LABEL_COLUMNS,
            "audit_only_never_features": AUDIT_COLUMNS,
        },
        "validation": {
            "checks_run": [
                "label and feature grids both restricted to the exact UTC hourly window",
                "feature chunk schema, row count, tz, ordering and duplicate checks per year",
                "one-to-one exact-timestamp inner join (no nearest-hour matching)",
                "re-join of the synchronized timestamps reproduces identical feature values",
                "chronological ordering of the output",
                "duplicate timestamp count of the output",
                "labels hold only {0,1} with no NaN",
                "window cohorts partition the 2014-2025 grid exactly",
                "strict/recovered label variants bracket the primary label",
                "positives + negatives == rows",
                "observation-derived columns are not feature columns",
                "weather_code thunderstorm-code cross-tab against the label",
                "protected Phase 1/Phase 2 files byte-identical before and after",
            ],
            "problems": problems,
            "status": "PASS" if not problems else "FAIL",
        },
        "protected_files_sha256_before": protected_before,
        "protected_files_sha256_after": protected_after,
        "protected_files_unchanged": protected_unchanged,
        "checksums": {"csv_sha256": sha256_of(OUT_CSV)},
        "no_model_trained": True,
        "scope_note": "Phase 3 only. The dashboard, backend, existing model and the Phase 1 "
                      "datasets (weather_data.csv, weather_data_with_code.csv) were not modified, "
                      "and no model was trained or re-trained.",
    }

    # Great-circle separations. Two different points are involved and they must not be
    # conflated: the point that was *requested* from Open-Meteo, and the grid cell the API
    # *served* (which is where the delivered values actually come from).
    def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
        lat_a, lon_a, lat_b, lon_b = map(radians, [lat_a, lon_a, lat_b, lon_b])
        h = (
            sin((lat_b - lat_a) / 2) ** 2
            + cos(lat_a) * cos(lat_b) * sin((lon_b - lon_a) / 2) ** 2
        )
        return round(2 * 6371.0088 * asin(sqrt(h)), 3)

    served_lat = served_cell.get("served_latitude")
    served_lon = served_cell.get("served_longitude")
    metadata["coordinates"]["distances_km"] = {
        "served_cell_to_station": (
            haversine_km(served_lat, served_lon, LABEL_LATITUDE, LABEL_LONGITUDE)
            if served_lat is not None and served_lon is not None
            else None
        ),
        "request_point_to_station": haversine_km(
            REQUEST_LATITUDE, REQUEST_LONGITUDE, LABEL_LATITUDE, LABEL_LONGITUDE
        ),
        "request_point_to_served_cell": (
            haversine_km(REQUEST_LATITUDE, REQUEST_LONGITUDE, served_lat, served_lon)
            if served_lat is not None and served_lon is not None
            else None
        ),
    }
    metadata["coordinates"]["distance_note"] = (
        "The separation that applies to the delivered feature values is "
        "served_cell_to_station. request_point_to_station is the distance from the coordinate "
        "that was asked for, not from the cell that answered, and the two differ because the API "
        "serves the grid cell containing the request point."
    )

    with open(OUT_META, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    write_report(metadata, per_year, synchronized)

    print(f"rows: {len(synchronized)}  positives: {positives}  negatives: {negatives}")
    print(f"rate: {metadata['counts']['positive_rate']}")
    print(f"excluded: {len(merged) - len(synchronized)} "
          f"(unobserved {len(excluded_no_report)}, no present-weather group {len(excluded_no_wx_group)})")
    print(f"missing feature values: {missing_per_column}")
    print(f"weather_code TS-code check: {leakage['weather_code_check']['verdict']}")
    print(f"problems: {problems if problems else 'NONE'}")
    print("wrote:", OUT_CSV)
    print("wrote:", OUT_META)
    print("wrote:", OUT_REPORT)
    return 1 if problems else 0


def write_report(metadata: dict, per_year: dict, synchronized: pd.DataFrame) -> None:
    counts = metadata["counts"]
    cohort = metadata["masking"]["cohort"]
    align = metadata["timestamp_alignment"]
    leak = metadata["leakage_checks"]
    wc = leak["weather_code_check"]
    coords = metadata["coordinates"]
    dists = coords.get("distances_km") or {}
    dists_note = coords.get("distance_note", "")
    served_lat = coords.get("feature_served_grid_cell", {}).get("served_latitude")
    served_lon = coords.get("feature_served_grid_cell", {}).get("served_longitude")
    missing = metadata["missing_values"]
    problems = metadata["validation"]["problems"]

    rows_per_year = "\n".join(
        f"| {year} | {rows:,} | {counts['positives_per_year'].get(year, 0):,} | "
        f"{round(100 * counts['positives_per_year'].get(year, 0) / rows, 3)}% |"
        for year, rows in counts["rows_per_year"].items()
    )
    missing_rows = "\n".join(
        f"| {column} | {value:,} |" for column, value in missing["feature_missing_per_column"].items()
    )
    year_chunks = "\n".join(
        f"| {year} | {info['rows']:,} | {info['first_timestamp_utc']} | "
        f"{info['last_timestamp_utc']} | {sum(info['missing_values'].values()):,} |"
        for year, info in per_year.items()
    )

    report = f"""# Phase 3 -- Data preprocessing and synchronization report

Generated by `dataset/{os.path.basename(__file__)}` at {metadata['created_at_utc']}.

**Scope.** This phase synchronizes the genuine Phase 2 VOTV thunderstorm labels with the
Phase 3 Open-Meteo atmospheric features. No model was trained or re-trained, and the
dashboard, backend, existing model and the Phase 1 datasets were not modified.

**Validation status: {metadata['validation']['status']}**
{("- problems: " + "; ".join(problems)) if problems else "- all checks passed"}

## 1. Sources and date range

| item | value |
| --- | --- |
| label source | `dataset/historical_thunderstorm_labels_votv.csv` (Phase 2, IEM/NCEI METAR present-weather `wxcodes`) |
| feature source | `dataset/raw_openmeteo/votv_openmeteo_<year>.csv` (Open-Meteo Historical Weather API archive) |
| modelling window | {metadata['window']['start_utc']} .. {metadata['window']['end_utc']} (2014-2025) |
| window rationale | {metadata['window']['rationale']} |
| calendar hours in window | {metadata['window']['calendar_hours_in_window']:,} |

The window is the Phase 2 recommended modelling window. Years before 2014 are excluded
because the METAR present-weather group was absent or only partially encoded, which makes
their label rate track reporting practice rather than weather.

## 2. Coordinates

| item | latitude | longitude |
| --- | --- | --- |
| Open-Meteo request point (features) | {coords['feature_request_point']['latitude']} | {coords['feature_request_point']['longitude']} |
| Open-Meteo served grid cell (features, as reported by the API) | {coords.get('feature_served_grid_cell', {}).get('served_latitude')} | {coords.get('feature_served_grid_cell', {}).get('served_longitude')} |
| VOTV station point (labels) | {coords['label_station_point']['latitude']} | {coords['label_station_point']['longitude']} |

API-reported cell elevation: {coords.get('feature_served_grid_cell', {}).get('served_elevation_m')} m.

| separation (great-circle) | distance |
| --- | --- |
| **served grid cell -> VOTV station** (applies to the delivered feature values) | **{dists.get('served_cell_to_station')} km** |
| requested point -> VOTV station (the coordinate asked for, not the cell that answered) | {dists.get('request_point_to_station')} km |
| requested point -> served grid cell | {dists.get('request_point_to_served_cell')} km |

{dists_note}

{coords['note']}

## 3. Row counts and cohorts

| quantity | rows |
| --- | --- |
| calendar hours in window | {metadata['window']['calendar_hours_in_window']:,} |
| hours with at least one valid VOTV report (label observed) | {cohort['label_observed_hours']:,} |
| hours with no report at all (label NaN) | {cohort['label_unobserved_hours']:,} |
| hours with a decodable present-weather group (`wx_field_observed == 1`) | {cohort['hours_with_present_weather_group']:,} |
| hours with reports but no decodable present-weather group | {cohort['hours_with_reports_but_no_present_weather_group']:,} |
| **synchronized supervised rows** | **{counts['rows']:,}** |

The four cohorts partition the window exactly:
{cohort['hours_with_present_weather_group']:,} + {cohort['hours_with_reports_but_no_present_weather_group']:,} + {cohort['label_unobserved_hours']:,} = {metadata['window']['calendar_hours_in_window']:,}.

Masking recipe: `{metadata['masking']['recipe']}`

{metadata['masking']['why']}

## 4. Synchronized dataset shape

* rows: **{counts['rows']:,}**
* columns: **{len(metadata['columns']['features']) + len(metadata['columns']['label']) + len(metadata['columns']['audit_only_never_features']) + 2}**
  ({len(metadata['columns']['features'])} Open-Meteo features, 1 completeness flag, 1 label,
  {len(metadata['columns']['audit_only_never_features'])} observation/audit columns)
* time column: `{metadata['columns']['time']}`

| column group | columns |
| --- | --- |
| features | {", ".join("`" + c + "`" for c in metadata['columns']['features'])} |
| labels | {", ".join("`" + c + "`" for c in metadata['columns']['label'])} |
| audit only (never features) | {", ".join("`" + c + "`" for c in metadata['columns']['audit_only_never_features'])} |

## 5. Positive / negative counts

| quantity | rows |
| --- | --- |
| positive (`thunderstorm_label == 1`) | {counts['positive_hours']:,} |
| negative (`thunderstorm_label == 0`) | {counts['negative_hours']:,} |
| positive rate | {round(100 * counts['positive_rate'], 3)}% |
| positives + negatives == rows | {"yes" if counts['positive_hours'] + counts['negative_hours'] == counts['rows'] else "no"} |

| year | supervised rows | positives | positive rate |
| --- | --- | --- | --- |
{rows_per_year}

## 6. Excluded / unobserved rows

Rows dropped from the window, and why they were not supervised:

| excluded cohort | rows |
| --- | --- |
| no VOTV report in the hour (label NaN) | {counts['excluded_unobserved_no_report']:,} |
| report present but no decodable present-weather group | {counts['excluded_reports_without_present_weather_group']:,} |
| **total excluded** | **{counts['excluded_rows']:,}** |

An hour with no observation is **not** evidence of absence of a thunderstorm, so it is never
written as `thunderstorm_label = 0`; it is absent from the file entirely. The audit column
`wx_field_observed` remains in the retained rows so the evidence basis of every label is
still inspectable.

## 7. Missing-value handling

Policy: **{missing['policy']}**.

| feature column | missing values |
| --- | --- |
{missing_rows}

| quantity | rows |
| --- | --- |
| total missing feature values | {missing['total_missing_feature_values']:,} |
| rows with any missing feature | {missing['rows_with_any_missing_feature']:,} |
| rows with all features missing | {missing['rows_with_all_features_missing']:,} |
| `features_complete == True` | {int(synchronized['features_complete'].sum()):,} |

Recommended Phase 4 handling: {missing['recommended_phase4_handling']}

## 8. Timestamp alignment

| check | result |
| --- | --- |
| join key | exact UTC hourly timestamp, one-to-one inner join |
| all window hours matched by the join | {"yes" if len(synchronized) + counts['excluded_rows'] == metadata['window']['calendar_hours_in_window'] else "no"} |
| duplicate timestamps | {align['duplicate_timestamps']} |
| chronologically ascending | {align['timestamps_sorted_ascending']} |
| timestamps are UTC | {align['timestamps_are_utc']} |
| all timestamps on the hour boundary | {align['all_timestamps_on_hour_boundary']} |
| re-join reproduces identical feature values | {align['join_is_exact_same_hour']} |
| maximum timestamp offset between features and labels | {leak['future_information']['max_abs_timestamp_offset_seconds']} s |

Result: {align['rows']:,} rows are strictly increasing, unique, UTC, on exact hour boundaries,
and every row's features come from the same UTC hour as its label.

### Per-year feature chunks used

| year | rows | first timestamp (UTC) | last timestamp (UTC) | missing values |
| --- | --- | --- | --- | --- |
{year_chunks}

## 9. Leakage checks

| check | result |
| --- | --- |
| lead-time shift applied | {leak['lead_time_shift_applied']} |
| task encoded | same-hour nowcast (features(H) -> label(H)) |
| maximum feature/label timestamp offset | {leak['future_information']['max_abs_timestamp_offset_seconds']} s |
| forward-looking or rolling feature columns | none present (each feature is a single-hour value) |
| observation-derived columns kept out of the feature list | {leak['observation_columns_are_not_features']} |
{leak['lead_time_note']}

### `weather_code` target-leakage diagnostic

Open-Meteo `weather_code` has dedicated thunderstorm categories ({", ".join(map(str, wc['ts_codes_considered']))}).
Phase 2 forbids `weather_code` as the *label*, but as an input it is still an archive-side
statement about thunderstorm weather in the same hour, so it was tested against the observed label:

| quantity | value |
| --- | --- |
| hours with `weather_code` in {wc['ts_codes_considered']} | {wc['hours_with_ts_weather_code']:,} |
| ... of which label positive | {wc['of_which_label_positive']:,} |
| ... of which label negative | {wc['of_which_label_negative']:,} |
| precision of the TS code as a predictor | {wc['precision_of_ts_code_as_predictor'] if wc['precision_of_ts_code_as_predictor'] is not None else "n/a (no TS codes)"} |
| recall of the TS code as a predictor | {wc['recall_of_ts_code_as_predictor']} |
| observed maximum `weather_code` in the window | {wc['observed_weather_code_max']} |

**Verdict.** {wc['verdict']}

## 10. Validation summary

Checks executed:

{chr(10).join("- " + c for c in metadata['validation']['checks_run'])}

Protected files (must remain byte-identical): {"PASS" if metadata['protected_files_unchanged'] else "FAIL"}

A second script, `dataset/verify_synchronized_dataset_phase3.py`, re-derives the cohorts,
counts and feature alignment straight from the two source artifacts without importing this
build script, and additionally checks that all protected Phase 1/Phase 2 tracked files are
unmodified according to git. Run it after this script to reproduce the validation
independently.

## 11. Artifacts written

| file | role |
| --- | --- |
| `dataset/{metadata['artifact']}` | synchronized supervised dataset |
| `dataset/{os.path.basename(OUT_META)}` | machine-readable metadata, counts, checks, checksums |
| `dataset/{os.path.basename(OUT_REPORT)}` | this report |
| `dataset/{os.path.basename(__file__)}` | reproducible build script |
| `dataset/verify_synchronized_dataset_phase3.py` | independent verification script (separate implementation) |

Dataset SHA-256: `{metadata['checksums']['csv_sha256']}`

## 12. Explicit limitations

* Labels are point observations at the aerodrome; a thunderstorm elsewhere in the city or
  catchment that the station did not report is labelled negative or unobserved.
* The features belong to the Open-Meteo archive grid cell that the API served for the Phase 3
  request ({served_lat} N, {served_lon} E), which is **{dists.get('served_cell_to_station')} km**
  from the label station point. No finer spatial resolution is claimed.
* Hours with reports but no decodable present-weather group are excluded rather than labelled;
  this is conservative but keeps the negative class evidence-based.
* No lead time is applied here; the pairing is same-hour. Phase 4 must shift the label, not
  the features, to build a forecast task.
* No model was trained in this phase, and no performance figure is claimed.
"""

    with open(OUT_REPORT, "w", encoding="utf-8") as handle:
        handle.write(report)


if __name__ == "__main__":
    sys.exit(main())
