"""Phase 8A: lightweight NWP join audit (no training, no joined feature CSV)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
FEAT_CSV = BASE / "features" / "multilocation_features_2014_2025.csv"
NWP_CSV = BASE / "nwp_full" / "nwp_gfs_pooled.csv"
REPORT = BASE.parent.parent / "docs" / "PHASE8A_NWP_JOIN_AUDIT.md"
AUDIT_JSON = BASE / "nwp_full" / "phase8a_join_audit.json"

STATIONS = ["VOTV", "VECC", "VIDP", "VOCI", "VABB"]
NWP_VARS = [
    "cape",
    "convective_inhibition",
    "lifted_index",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]
PREC_START = pd.Timestamp("2022-11-29 20:00:00+00:00")
PREC_END = pd.Timestamp("2022-12-14 04:00:00+00:00")
CIN_START = pd.Timestamp("2023-12-01 01:00:00+00:00")
CIN_END = pd.Timestamp("2023-12-15 11:00:00+00:00")


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    log(f"Loading feature keys from {FEAT_CSV}")
    feats = pd.read_csv(FEAT_CSV, usecols=["station_id", "timestamp_utc"])
    n_feat = int(len(feats))
    feat_tz_sample = str(feats["timestamp_utc"].iloc[0])
    feats["timestamp_utc"] = pd.to_datetime(feats["timestamp_utc"], utc=True, format="ISO8601")
    feat_tz_after = str(feats["timestamp_utc"].dtype)

    feat_dups = int(feats.duplicated(["station_id", "timestamp_utc"]).sum())
    feat_unique = int(feats.drop_duplicates(["station_id", "timestamp_utc"]).shape[0])
    feat_stations = sorted(feats["station_id"].astype(str).unique().tolist())
    feat_per_station = feats.groupby("station_id").size().to_dict()
    feat_tmin = feats["timestamp_utc"].min().isoformat()
    feat_tmax = feats["timestamp_utc"].max().isoformat()

    log(f"Loading NWP from {NWP_CSV}")
    nwp = pd.read_csv(NWP_CSV, usecols=["station_id", "valid_time_utc"] + NWP_VARS)
    n_nwp = int(len(nwp))
    nwp_tz_sample = str(nwp["valid_time_utc"].iloc[0])
    nwp["timestamp_utc"] = pd.to_datetime(nwp["valid_time_utc"], utc=True)
    nwp_tz_after = str(nwp["timestamp_utc"].dtype)

    nwp_dups = int(nwp.duplicated(["station_id", "timestamp_utc"]).sum())
    nwp_unique = int(nwp.drop_duplicates(["station_id", "timestamp_utc"]).shape[0])
    nwp_stations = sorted(nwp["station_id"].astype(str).unique().tolist())
    nwp_per_station = nwp.groupby("station_id").size().to_dict()
    nwp_tmin = nwp["timestamp_utc"].min().isoformat()
    nwp_tmax = nwp["timestamp_utc"].max().isoformat()

    feat_keys = feats[["station_id", "timestamp_utc"]].drop_duplicates()
    nwp_keys = nwp[["station_id", "timestamp_utc"]].drop_duplicates()

    inner = feat_keys.merge(nwp_keys, on=["station_id", "timestamp_utc"], how="inner")
    left = feat_keys.merge(nwp_keys, on=["station_id", "timestamp_utc"], how="left", indicator=True)
    right = nwp_keys.merge(feat_keys, on=["station_id", "timestamp_utc"], how="left", indicator=True)

    n_inner = int(len(inner))
    n_feat_unmatched = int((left["_merge"] == "left_only").sum())
    n_nwp_unmatched = int((right["_merge"] == "left_only").sum())
    left_retention = float(n_inner / n_feat) if n_feat else 0.0

    unmatched_feat = left.loc[left["_merge"] == "left_only"]
    unmatched_nwp = right.loc[right["_merge"] == "left_only"]
    unmatched_feat_by_station = unmatched_feat.groupby("station_id").size().to_dict()
    unmatched_nwp_by_station = unmatched_nwp.groupby("station_id").size().to_dict()
    inner_by_station = inner.groupby("station_id").size().to_dict()

    # NWP missingness on full NWP table
    miss_full = {}
    for col in NWP_VARS:
        n_miss = int(nwp[col].isna().sum())
        miss_full[col] = {
            "missing_count": n_miss,
            "missing_pct": round(100.0 * n_miss / n_nwp, 4) if n_nwp else None,
        }

    # NWP missingness on inner-join keys only
    nwp_matched = nwp.merge(inner, on=["station_id", "timestamp_utc"], how="inner")
    miss_matched = {}
    for col in NWP_VARS:
        n_miss = int(nwp_matched[col].isna().sum())
        miss_matched[col] = {
            "missing_count": n_miss,
            "missing_pct": round(100.0 * n_miss / len(nwp_matched), 4) if len(nwp_matched) else None,
        }

    any_nwp_miss = int(nwp_matched[NWP_VARS].isna().any(axis=1).sum())
    all_nwp_present = int(nwp_matched[NWP_VARS].notna().all(axis=1).sum())

    miss_by_station = {}
    for st in STATIONS:
        sub = nwp_matched.loc[nwp_matched["station_id"] == st]
        miss_by_station[st] = {
            "rows": int(len(sub)),
            **{c: int(sub[c].isna().sum()) for c in NWP_VARS},
        }

    # Known null windows (NWP clock, all stations)
    prec_mask = (nwp["timestamp_utc"] >= PREC_START) & (nwp["timestamp_utc"] <= PREC_END)
    cin_mask = (nwp["timestamp_utc"] >= CIN_START) & (nwp["timestamp_utc"] <= CIN_END)
    prec_nwp_rows = int(prec_mask.sum())
    cin_nwp_rows = int(cin_mask.sum())
    prec_nulls = int(nwp.loc[prec_mask, "precipitation"].isna().sum())
    cin_nulls = int(nwp.loc[cin_mask, "convective_inhibition"].isna().sum())

    feat_prec_window = feats.loc[
        (feats["timestamp_utc"] >= PREC_START) & (feats["timestamp_utc"] <= PREC_END)
    ]
    feat_cin_window = feats.loc[
        (feats["timestamp_utc"] >= CIN_START) & (feats["timestamp_utc"] <= CIN_END)
    ]

    inner_prec = inner.loc[
        (inner["timestamp_utc"] >= PREC_START) & (inner["timestamp_utc"] <= PREC_END)
    ]
    inner_cin = inner.loc[
        (inner["timestamp_utc"] >= CIN_START) & (inner["timestamp_utc"] <= CIN_END)
    ]

    # unmatched feature timestamps: mostly pre-NWP start
    unmatched_feat_tmin = (
        unmatched_feat["timestamp_utc"].min().isoformat() if len(unmatched_feat) else None
    )
    unmatched_feat_tmax = (
        unmatched_feat["timestamp_utc"].max().isoformat() if len(unmatched_feat) else None
    )
    nwp_start = nwp["timestamp_utc"].min()
    unmatched_before_nwp = int((unmatched_feat["timestamp_utc"] < nwp_start).sum())
    unmatched_after_nwp_start = int((unmatched_feat["timestamp_utc"] >= nwp_start).sum())

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "join_key": ["station_id", "timestamp_utc"],
        "stations_expected": STATIONS,
        "stations_features": feat_stations,
        "stations_nwp": nwp_stations,
        "all_five_stations_in_join": set(inner["station_id"].unique()) == set(STATIONS),
        "feature_rows": n_feat,
        "feature_unique_keys": feat_unique,
        "feature_duplicate_keys": feat_dups,
        "feature_per_station": {str(k): int(v) for k, v in feat_per_station.items()},
        "feature_time_min": feat_tmin,
        "feature_time_max": feat_tmax,
        "feature_timestamp_sample_raw": feat_tz_sample,
        "feature_timestamp_dtype_after_parse": feat_tz_after,
        "nwp_rows": n_nwp,
        "nwp_unique_keys": nwp_unique,
        "nwp_duplicate_keys": nwp_dups,
        "nwp_per_station": {str(k): int(v) for k, v in nwp_per_station.items()},
        "nwp_time_min": nwp_tmin,
        "nwp_time_max": nwp_tmax,
        "nwp_timestamp_sample_raw": nwp_tz_sample,
        "nwp_timestamp_dtype_after_parse": nwp_tz_after,
        "inner_join_rows": n_inner,
        "inner_join_by_station": {str(k): int(v) for k, v in inner_by_station.items()},
        "left_join_retention": left_retention,
        "feature_rows_unmatched": n_feat_unmatched,
        "nwp_rows_unmatched": n_nwp_unmatched,
        "unmatched_features_by_station": {str(k): int(v) for k, v in unmatched_feat_by_station.items()},
        "unmatched_nwp_by_station": {str(k): int(v) for k, v in unmatched_nwp_by_station.items()},
        "unmatched_feature_time_min": unmatched_feat_tmin,
        "unmatched_feature_time_max": unmatched_feat_tmax,
        "unmatched_features_before_nwp_start": unmatched_before_nwp,
        "unmatched_features_on_or_after_nwp_start": unmatched_after_nwp_start,
        "nwp_missingness_full_table": miss_full,
        "nwp_missingness_inner_join": miss_matched,
        "inner_join_any_nwp_missing": any_nwp_miss,
        "inner_join_all_nwp_present": all_nwp_present,
        "nwp_missing_by_station_inner": miss_by_station,
        "precip_null_window": {
            "start": PREC_START.isoformat(),
            "end": PREC_END.isoformat(),
            "nwp_rows_in_window": prec_nwp_rows,
            "precipitation_nulls_in_window": prec_nulls,
            "feature_rows_in_window": int(len(feat_prec_window)),
            "inner_join_rows_in_window": int(len(inner_prec)),
        },
        "cin_null_window": {
            "start": CIN_START.isoformat(),
            "end": CIN_END.isoformat(),
            "nwp_rows_in_window": cin_nwp_rows,
            "cin_nulls_in_window": cin_nulls,
            "feature_rows_in_window": int(len(feat_cin_window)),
            "inner_join_rows_in_window": int(len(inner_cin)),
        },
        "protected": {
            "no_training": True,
            "no_joined_feature_csv": True,
            "phase3_4_5_6_untouched": True,
            "v1_untouched": True,
        },
    }

    AUDIT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def pct(x, n):
        return f"{100.0 * x / n:.2f}%" if n else "NA"

    lines = [
        "# Phase 8A — NWP join audit (V2)",
        "",
        f"**Generated:** {payload['generated_at_utc']}",
        "**Script:** `dataset/multilocation/audit_nwp_join_phase8a.py`",
        "**Scope:** join audit only. No training, no enhanced ML CSV, no models.",
        "",
        "Phase 3 feature files, Phase 4/5/6 artifacts, and V1 were not modified.",
        "",
        "## Objective",
        "",
        "Measure coverage of joining existing V2 Phase 3 feature rows to Phase 7C GFS Historical Forecast NWP using `station_id + UTC timestamp` only.",
        "",
        "## Data sources",
        "",
        "| source | path |",
        "| --- | --- |",
        "| Features | `dataset/multilocation/features/multilocation_features_2014_2025.csv` |",
        "| NWP | `dataset/multilocation/nwp_full/nwp_gfs_pooled.csv` |",
        "",
        "NWP product: Open-Meteo Historical Forecast, `gfs_global`. This is archived forecast-model output, not observations and not ERA5 reanalysis.",
        "",
        "## Join methodology",
        "",
        "- Key: `station_id` + parsed UTC timestamp.",
        "- Feature clock column: `timestamp_utc`.",
        "- NWP clock column: `valid_time_utc` (renamed to `timestamp_utc` after UTC parse).",
        "- No timestamp-only join.",
        "- No NWP time shift.",
        "- No imputation.",
        "- Inner join used only in memory for counts; no joined feature table was written.",
        "",
        "## Timestamp timezone",
        "",
        "| side | raw sample | dtype after `utc=True` parse |",
        "| --- | --- | --- |",
        f"| features | `{feat_tz_sample}` | `{feat_tz_after}` |",
        f"| NWP | `{nwp_tz_sample}` | `{nwp_tz_after}` |",
        "",
        "Both sides parse to timezone-aware UTC.",
        "",
        "## Station consistency",
        "",
        f"- Expected: {', '.join(STATIONS)}",
        f"- Features: {', '.join(feat_stations)}",
        f"- NWP: {', '.join(nwp_stations)}",
        f"- Inner join covers all five stations: **{payload['all_five_stations_in_join']}**",
        "",
        "## Coverage summary",
        "",
        "| metric | count |",
        "| --- | --- |",
        f"| feature rows | {n_feat} |",
        f"| unique feature station/timestamp keys | {feat_unique} |",
        f"| duplicate feature keys | {feat_dups} |",
        f"| NWP rows | {n_nwp} |",
        f"| unique NWP station/timestamp keys | {nwp_unique} |",
        f"| duplicate NWP keys | {nwp_dups} |",
        f"| exact inner-join rows | {n_inner} |",
        f"| left-join retention (inner / feature rows) | {left_retention:.4f} ({pct(n_inner, n_feat)}) |",
        f"| unmatched feature rows | {n_feat_unmatched} |",
        f"| unmatched NWP rows | {n_nwp_unmatched} |",
        "",
        f"Feature time range: `{feat_tmin}` → `{feat_tmax}`  ",
        f"NWP time range: `{nwp_tmin}` → `{nwp_tmax}`",
        "",
        "### Per station",
        "",
        "| station | feature rows | NWP rows | inner join | unmatched features | unmatched NWP |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for st in STATIONS:
        lines.append(
            f"| {st} | {feat_per_station.get(st, 0)} | {nwp_per_station.get(st, 0)} | "
            f"{inner_by_station.get(st, 0)} | {unmatched_feat_by_station.get(st, 0)} | "
            f"{unmatched_nwp_by_station.get(st, 0)} |"
        )
    lines += [
        "",
        "### Unmatched feature rows",
        "",
        f"- Time span of unmatched features: `{unmatched_feat_tmin}` → `{unmatched_feat_tmax}`",
        f"- Unmatched features before NWP start ({nwp_tmin}): **{unmatched_before_nwp}**",
        f"- Unmatched features on/after NWP start: **{unmatched_after_nwp_start}**",
        "",
        "Unmatched NWP rows are GFS hours with no Phase 3 feature row at the same station/timestamp (feature table starts after lag warmup and may drop hours).",
        "",
        "## NWP missingness (full NWP table)",
        "",
        "| variable | missing count | missing % |",
        "| --- | --- | --- |",
    ]
    for col in NWP_VARS:
        m = miss_full[col]
        lines.append(f"| {col} | {m['missing_count']} | {m['missing_pct']}% |")
    lines += [
        "",
        "## NWP missingness (inner-join keys only)",
        "",
        f"Inner-join rows: {n_inner}. Rows with any selected NWP field missing: **{any_nwp_miss}**. Rows with all selected NWP fields present: **{all_nwp_present}**.",
        "",
        "| variable | missing count | missing % of inner join |",
        "| --- | --- | --- |",
    ]
    for col in NWP_VARS:
        m = miss_matched[col]
        lines.append(f"| {col} | {m['missing_count']} | {m['missing_pct']}% |")
    lines += [
        "",
        "### Missing counts by station (inner join)",
        "",
        "| station | rows | cape | CIN | LI | T2m | RH | Psurf | wspd | wdir | precip | cloud |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for st in STATIONS:
        s = miss_by_station[st]
        lines.append(
            f"| {st} | {s['rows']} | {s['cape']} | {s['convective_inhibition']} | {s['lifted_index']} | "
            f"{s['temperature_2m']} | {s['relative_humidity_2m']} | {s['surface_pressure']} | "
            f"{s['wind_speed_10m']} | {s['wind_direction_10m']} | {s['precipitation']} | {s['cloud_cover']} |"
        )
    pw = payload["precip_null_window"]
    cw = payload["cin_null_window"]
    lines += [
        "",
        "## Known Phase 7C null windows",
        "",
        "| window | field | UTC start | UTC end | NWP rows in window | nulls in window | feature rows in window | inner-join rows in window |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| precipitation | precipitation | {pw['start']} | {pw['end']} | {pw['nwp_rows_in_window']} | {pw['precipitation_nulls_in_window']} | {pw['feature_rows_in_window']} | {pw['inner_join_rows_in_window']} |",
        f"| CIN | convective_inhibition | {cw['start']} | {cw['end']} | {cw['nwp_rows_in_window']} | {cw['cin_nulls_in_window']} | {cw['feature_rows_in_window']} | {cw['inner_join_rows_in_window']} |",
        "",
        "These are vendor nulls inside otherwise complete hourly NWP rows. They were not filled.",
        "",
        "## Integrity notes",
        "",
        "- Join key is station + UTC timestamp, not timestamp alone.",
        f"- Duplicate station/timestamp keys: features **{feat_dups}**, NWP **{nwp_dups}**.",
        "- No training, imputation, or feature-importance computation in this phase.",
        "- No huge joined CSV written.",
        "- Phase 3/4/5/6 and V1 not modified.",
        "- NWP is Historical Forecast GFS, not labeled as observation.",
        "",
        "## Artifacts",
        "",
        "| path | role |",
        "| --- | --- |",
        "| `dataset/multilocation/audit_nwp_join_phase8a.py` | audit script |",
        "| `dataset/multilocation/nwp_full/phase8a_join_audit.json` | machine-readable counts |",
        "| `docs/PHASE8A_NWP_JOIN_AUDIT.md` | this report |",
        "",
        "## PHASE STATUS",
        "",
        "**READY** — join audit only. Do not proceed to Phase 8B in this task.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {REPORT}")
    log(json.dumps({k: payload[k] for k in [
        "feature_rows", "nwp_rows", "inner_join_rows",
        "feature_rows_unmatched", "nwp_rows_unmatched",
        "feature_duplicate_keys", "nwp_duplicate_keys",
        "all_five_stations_in_join",
    ]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
