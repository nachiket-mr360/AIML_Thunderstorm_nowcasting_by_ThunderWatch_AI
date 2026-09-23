#!/usr/bin/env python3
"""Phase 10I.3 — causal hourly alignment of scan-level CMK features.

Does not modify atmospheric/NWP tables. No training. No forward/zero fill.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "dataset/satellite/insat3dr_cmk_features/cmk_scan_level_features.csv"
HOURS = ROOT / "dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025.csv"
INV = ROOT / "outputs/satellite/phase10h/phase10h_file_inventory.csv"
OUT_CSV = ROOT / "dataset/satellite/insat3dr_cmk_features/cmk_hourly_aligned_features.csv"
OUT_JSON = ROOT / "outputs/satellite/phase10i3/phase10i3_alignment_summary.json"
OUT_DOC = ROOT / "docs/PHASE10I3_INSAT3DR_CAUSAL_ALIGNMENT_REPORT.md"
SAMPLE = ROOT / "outputs/satellite/phase10i3/phase10i3_match_provenance_sample.csv"

MAX_AGE = pd.Timedelta(minutes=90)
SAT_VALUE_COLS = [
    "sat_observation_time_utc",
    "sat_acquisition_start_utc",
    "sat_acquisition_end_utc",
    "sat_cloud_mask_at_station",
    "sat_valid_pixel_count_25km",
    "sat_cloudy_fraction_25km",
    "sat_clear_fraction_25km",
    "sat_nearby_cloud_fraction_25km",
    "source_filename",
    "gId",
]


def main() -> None:
    hours = pd.read_csv(HOURS, usecols=["station_id", "timestamp_utc"])
    hours["timestamp_utc"] = pd.to_datetime(hours["timestamp_utc"], utc=True, format="ISO8601")
    hours = hours.rename(columns={"station_id": "station"})
    hours = hours.drop_duplicates(["station", "timestamp_utc"]).sort_values(
        ["station", "timestamp_utc"]
    )

    scans = pd.read_csv(SCAN)
    scans["sat_acquisition_end_utc"] = pd.to_datetime(scans["sat_acquisition_end_utc"], utc=True)
    scans["sat_acquisition_start_utc"] = pd.to_datetime(scans["sat_acquisition_start_utc"], utc=True)
    scans["sat_observation_time_utc"] = pd.to_datetime(scans["sat_observation_time_utc"], utc=True)
    scans["gId"] = scans["gId"].astype(str)
    scans = scans.rename(columns={"filename": "source_filename"})
    scans = scans.sort_values(["station", "sat_acquisition_end_utc", "gId"])

    inv_ok = pd.read_csv(INV)
    ok_files = set(inv_ok.loc[inv_ok["status"] == "OK", "filename"].astype(str))

    parts = []
    for station, h in hours.groupby("station", sort=False):
        s = scans[scans["station"] == station].copy()
        if s.empty:
            h = h.copy()
            h["sat_observation_available"] = 0
            parts.append(h)
            continue
        merged = pd.merge_asof(
            h,
            s,
            left_on="timestamp_utc",
            right_on="sat_acquisition_end_utc",
            by="station",
            direction="backward",
            tolerance=MAX_AGE,
            suffixes=("", "_scan"),
        )
        parts.append(merged)
    aligned = pd.concat(parts, ignore_index=True)

    end = aligned["sat_acquisition_end_utc"]
    t = aligned["timestamp_utc"]
    age = (t - end) / pd.Timedelta(minutes=1)
    eligible = end.notna() & (end <= t) & (age >= 0) & (age <= 90)
    aligned["sat_observation_available"] = eligible.astype(int)
    aligned["sat_age_minutes"] = np.where(eligible, age, np.nan)

    for col in SAT_VALUE_COLS:
        if col not in aligned.columns:
            aligned[col] = pd.NA
        aligned.loc[~eligible, col] = pd.NA

    out_cols = [
        "station",
        "timestamp_utc",
        "sat_observation_available",
        "sat_observation_time_utc",
        "sat_acquisition_start_utc",
        "sat_acquisition_end_utc",
        "sat_cloud_mask_at_station",
        "sat_valid_pixel_count_25km",
        "sat_cloudy_fraction_25km",
        "sat_clear_fraction_25km",
        "sat_nearby_cloud_fraction_25km",
        "sat_age_minutes",
        "source_filename",
        "gId",
    ]
    aligned = aligned[out_cols].sort_values(["station", "timestamp_utc"])

    n = len(aligned)
    n_match = int(aligned["sat_observation_available"].sum())
    n_un = n - n_match
    dups = int(aligned.duplicated(["station", "timestamp_utc"]).sum())
    matches = aligned[aligned["sat_observation_available"] == 1]
    age_s = matches["sat_age_minutes"]
    n_age_gt90 = int((age_s > 90 + 1e-9).sum()) if len(matches) else 0
    n_end_after_t = 0
    if len(matches):
        n_end_after_t = int(
            (matches["sat_acquisition_end_utc"] > matches["timestamp_utc"]).sum()
        )
    unmatched = aligned[aligned["sat_observation_available"] == 0]
    null_ok = True
    if len(unmatched):
        for c in SAT_VALUE_COLS + ["sat_age_minutes"]:
            if unmatched[c].notna().any():
                null_ok = False
    src_ok = True
    if len(matches):
        src_ok = set(matches["source_filename"].astype(str)).issubset(ok_files)

    per_st = {}
    for st, g in aligned.groupby("station"):
        m = int(g["sat_observation_available"].sum())
        per_st[st] = {
            "n_hours": int(len(g)),
            "n_matched": m,
            "n_unmatched": int(len(g) - m),
            "match_pct": round(100.0 * m / len(g), 4) if len(g) else 0.0,
        }

    sample = matches.sample(n=min(25, len(matches)), random_state=10) if len(matches) else matches
    sample_out = sample[
        [
            "station",
            "timestamp_utc",
            "sat_acquisition_end_utc",
            "sat_age_minutes",
            "source_filename",
            "gId",
        ]
    ].rename(columns={"timestamp_utc": "target_T"})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(OUT_CSV, index=False)
    sample_out.to_csv(SAMPLE, index=False)

    summary = {
        "phase": "10I.3",
        "timestamp_universe": str(HOURS.as_posix()),
        "join_key": ["station", "timestamp_utc"],
        "causal_rule": {
            "availability": "sat_acquisition_end_utc <= T",
            "max_age_minutes": 90,
            "multiple": "latest sat_acquisition_end_utc, tie smaller gId",
            "no_match": "sat_observation_available=0 and satellite fields null",
            "not_used_as_availability": "sat_observation_time_utc",
        },
        "n_hourly_station_rows": n,
        "n_matched": n_match,
        "n_unmatched": n_un,
        "match_pct_overall": round(100.0 * n_match / n, 4) if n else 0.0,
        "per_station": per_st,
        "sat_age_minutes_matches": {
            "min": float(age_s.min()) if len(matches) else None,
            "median": float(age_s.median()) if len(matches) else None,
            "max": float(age_s.max()) if len(matches) else None,
        },
        "n_matches_age_gt_90": n_age_gt90,
        "n_matches_end_after_T": n_end_after_t,
        "n_duplicate_station_timestamp": dups,
        "cmk_class_counts_matches": (
            matches["sat_cloud_mask_at_station"].value_counts(dropna=False).to_dict()
            if len(matches)
            else {}
        ),
        "cloudy_fraction_matches": {
            "min": float(matches["sat_cloudy_fraction_25km"].min()) if len(matches) else None,
            "median": float(matches["sat_cloudy_fraction_25km"].median()) if len(matches) else None,
            "max": float(matches["sat_cloudy_fraction_25km"].max()) if len(matches) else None,
        },
        "clear_fraction_matches": {
            "min": float(matches["sat_clear_fraction_25km"].min()) if len(matches) else None,
            "median": float(matches["sat_clear_fraction_25km"].median()) if len(matches) else None,
            "max": float(matches["sat_clear_fraction_25km"].max()) if len(matches) else None,
        },
        "no_forward_fill": True,
        "unmatched_fields_null": null_ok,
        "all_source_files_in_phase10h_ok": src_ok,
        "unique_source_files_used": int(matches["source_filename"].nunique()) if len(matches) else 0,
        "nwp_or_atmosphere_tables_modified": False,
        "sanity": {
            "age_gt_90_is_0": n_age_gt90 == 0,
            "end_after_T_is_0": n_end_after_t == 0,
            "dups_is_0": dups == 0,
        },
        "output_csv": str(OUT_CSV.as_posix()),
        "provenance_sample_csv": str(SAMPLE.as_posix()),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Phase 10I.3 — Causal INSAT-3DR satellite-to-hourly alignment",
        "",
        "**Causal join only.** No NWP/atmosphere overwrite, no training, no forward/zero fill, no download.",
        "",
        "Timestamp universe: Phase 8B NWP overlap keys",
        f"`{HOURS.as_posix()}` (`station_id` + `timestamp_utc`, 2021-04-01 … 2025-12-31).",
        "",
        "Availability clock: **`sat_acquisition_end_utc`**. `/time` is provenance only.",
        "",
        f"- Hourly station rows: **{n}**",
        f"- Matched: **{n_match}** ({summary['match_pct_overall']}%)",
        f"- Unmatched: **{n_un}**",
        f"- Duplicate keys: **{dups}**",
        f"- Matches with age > 90 min: **{n_age_gt90}**",
        f"- Matches with acquisition_end > T: **{n_end_after_t}**",
        f"- Age min/median/max (min): {summary['sat_age_minutes_matches']}",
        f"- Per station: `{per_st}`",
        f"- CMK classes (matches): `{summary['cmk_class_counts_matches']}`",
        f"- Unmatched sat fields null: **{null_ok}**",
        f"- Sources ⊆ Phase 10H OK files: **{src_ok}**",
        f"- Unique source files used: **{summary['unique_source_files_used']}**",
        "",
        "Low match rate is expected: MVP has ~6 thinned scans per selected window, not a dense hourly archive.",
        "",
        f"Aligned table: `{OUT_CSV.as_posix()}`",
        f"Provenance sample: `{SAMPLE.as_posix()}`",
        "",
        "PHASE 10I.3 COMPLETE — CAUSAL ALIGNMENT ONLY",
        "",
    ]
    OUT_DOC.write_text("\n".join(lines), encoding="utf-8")
    print(
        "rows",
        n,
        "match",
        n_match,
        "un",
        n_un,
        "pct",
        summary["match_pct_overall"],
        "age>90",
        n_age_gt90,
        "end>T",
        n_end_after_t,
        "dups",
        dups,
    )


if __name__ == "__main__":
    main()
