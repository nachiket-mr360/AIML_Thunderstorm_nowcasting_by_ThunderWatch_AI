"""Phase 3 independent verification of the synchronized dataset.

This script deliberately re-derives everything from the primary sources instead of
importing dataset/build_synchronized_dataset_phase3.py, so a mistake in the build
script cannot hide itself behind the same mistake in the check.

Usage:
    python verify_synchronized_dataset_phase3.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

LABELS_CSV = os.path.join(HERE, "historical_thunderstorm_labels_votv.csv")
RAW_DIR = os.path.join(HERE, "raw_openmeteo")
SYNC_CSV = os.path.join(HERE, "votv_thunderstorm_synchronized_2014_2025.csv")

START = pd.Timestamp("2014-01-01 00:00:00", tz="UTC")
END = pd.Timestamp("2025-12-31 23:00:00", tz="UTC")

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "weather_code",
]
AUDIT = [
    "thunderstorm_label_strict",
    "thunderstorm_label_recovered",
    "wxcodes_observed",
    "n_reports",
    "n_reports_full_decode",
    "n_reports_with_wx",
    "wx_field_observed",
    "n_ts_reports",
]

result: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    result.append((name, bool(ok), detail))


def main() -> int:
    # ---- independent rebuild of both sides ---------------------------------------
    labels = pd.read_csv(LABELS_CSV, parse_dates=["timestamp_utc"])
    labels["timestamp_utc"] = labels["timestamp_utc"].dt.tz_convert("UTC") if (
        labels["timestamp_utc"].dt.tz is not None
    ) else labels["timestamp_utc"].dt.tz_localize("UTC")
    labels = labels[(labels["timestamp_utc"] >= START) & (labels["timestamp_utc"] <= END)]

    frames = []
    for year in range(2014, 2026):
        part = pd.read_csv(os.path.join(RAW_DIR, f"votv_openmeteo_{year}.csv"))
        part["timestamp_utc"] = pd.to_datetime(part["date"], utc=True)
        frames.append(part.drop(columns=["date"]))
    features = pd.concat(frames, ignore_index=True)

    window_hours = pd.date_range(START, END, freq="h")

    # ---- what the dataset should contain, computed from scratch -------------------
    expected_keys = set(
        labels.loc[
            (labels["wx_field_observed"] == 1) & labels["thunderstorm_label"].notna(),
            "timestamp_utc",
        ]
    )
    unobserved_keys = set(labels.loc[labels["thunderstorm_label"].isna(), "timestamp_utc"])
    no_wx_keys = set(
        labels.loc[
            labels["thunderstorm_label"].notna() & (labels["wx_field_observed"] != 1),
            "timestamp_utc",
        ]
    )

    sync = pd.read_csv(SYNC_CSV)
    sync["timestamp_utc"] = pd.to_datetime(sync["timestamp_utc"], utc=True)
    keys = set(sync["timestamp_utc"])

    check("row count matches independently rebuilt cohort", len(sync) == len(expected_keys),
          f"dataset={len(sync)} expected={len(expected_keys)}")
    check("timestamp set is exactly the observed+present-weather cohort", keys == expected_keys,
          f"missing={len(expected_keys - keys)} extra={len(keys - expected_keys)}")
    check("no unobserved (NaN-label) hour is present", not (keys & unobserved_keys),
          f"{len(keys & unobserved_keys)} overlapping hours")
    check("no report-without-present-weather hour is present", not (keys & no_wx_keys),
          f"{len(keys & no_wx_keys)} overlapping hours")

    counts = {
        "window_hours": len(window_hours),
        "observed": int(labels["thunderstorm_label"].notna().sum()),
        "unobserved": int(labels["thunderstorm_label"].isna().sum()),
        "no_wx_group": len(no_wx_keys),
        "kept": len(expected_keys),
        "positive": int((sync["thunderstorm_label"] == 1).sum()),
        "negative": int((sync["thunderstorm_label"] == 0).sum()),
    }
    check("cohort accounting closes over the window",
          counts["observed"] + counts["unobserved"] == counts["window_hours"]
          and counts["kept"] + counts["no_wx_group"] + counts["unobserved"] == counts["window_hours"],
          str(counts))
    check("positive + negative == rows",
          counts["positive"] + counts["negative"] == len(sync), str(counts))
    check("independent positive count matches Phase 2 record (3694)",
          counts["positive"] == 3694, str(counts["positive"]))

    # ---- structural checks --------------------------------------------------------
    check("labels are only 0/1 with no NaN",
          sync["thunderstorm_label"].notna().all()
          and set(sync["thunderstorm_label"].unique()) <= {0.0, 1.0})
    check("no duplicate timestamps", not sync["timestamp_utc"].duplicated().any())
    check("chronologically ascending", sync["timestamp_utc"].is_monotonic_increasing)
    check("timestamps are UTC", str(sync["timestamp_utc"].dt.tz) == "UTC")
    check("all timestamps on the hour boundary",
          bool((sync["timestamp_utc"].dt.minute == 0).all()
               and (sync["timestamp_utc"].dt.second == 0).all()))
    check("all rows carry actual observation evidence (wx_field_observed == 1)",
          bool((sync["wx_field_observed"] == 1).all()))
    check("strict/recovered variants still bracket the primary label",
          bool((sync["thunderstorm_label"] >= sync["thunderstorm_label_strict"]).all()
               and (sync["thunderstorm_label"] <= sync["thunderstorm_label_recovered"]).all()))

    missing = {c: int(sync[c].isna().sum()) for c in FEATURES}
    check("no missing atmospheric values in the delivered rows",
          sum(missing.values()) == 0, str(missing))
    check("features_complete agrees with the feature matrix",
          bool((sync["features_complete"] == ~sync[FEATURES].isna().any(axis=1)).all()))

    # ---- feature/label alignment, end to end -------------------------------------
    joined = sync.merge(
        features, on="timestamp_utc", how="left", validate="one_to_one", suffixes=("", "_raw")
    )
    check("every synchronized hour exists in the raw feature grid", len(joined) == len(sync))
    mismatched = [
        c for c in FEATURES
        if not sync[c].reset_index(drop=True).equals(
            joined[c + "_raw"].reset_index(drop=True)
        )
    ]
    check("every feature value equals the raw chunk value for the same hour",
          not mismatched, f"columns differing: {mismatched}")
    check("no feature timestamp is off the label hour",
          not set(sync["timestamp_utc"]) - set(features["timestamp_utc"]))

    # ---- leakiest column: weather_code -------------------------------------------
    wc = sync["weather_code"]
    check("weather_code thunderstorm categories counted from source",
          int(wc.isin([95, 96, 99]).sum()) == 0,
          f"hours with 95/96/99 = {int(wc.isin([95, 96, 99]).sum())}, max code = {int(wc.max())}")

    # ---- nothing outside the Phase 3 outputs changed ------------------------------
    tracked = [
        "dataset/weather_data.csv",
        "dataset/weather_data_with_code.csv",
        "dataset/historical_thunderstorm_labels_votv.csv",
        "dataset/historical_thunderstorm_labels_votv_metadata.json",
        "dataset/historical_thunderstorm_labels_votv_stats.json",
        "dataset/HISTORICAL_THUNDERSTORM_LABELS_README.md",
        "backend/main.py",
    ]
    diff = subprocess.run(
        ["git", "status", "--porcelain", "--"] + tracked,
        cwd=REPO, capture_output=True, text=True,
    ).stdout.strip()
    check("protected Phase 1/2 tracked files are unmodified", diff == "",
          diff.replace("\n", " | "))

    outside = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip().splitlines()
    outside = [
        line for line in outside
        if "dataset/" not in line.replace("\\", "/")
    ]
    check("no changes outside dataset/", not outside, " | ".join(outside))

    # ---- report -------------------------------------------------------------------
    print("Phase 3 independent verification")
    print("=" * 78)
    for name, ok, detail in result:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    failed = [n for n, ok, _ in result if not ok]
    print("=" * 78)
    print(f"{len(result) - len(failed)}/{len(result)} checks passed")
    print("RESULT:", "PASS" if not failed else f"FAIL ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
