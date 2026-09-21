"""Phase 8B: overlap dataset features(T) + NWP(T) + genuine METAR leads.

Inner join on station_id + UTC timestamp. No training, no imputation, no
overwrite of Phase 3 features.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
FEAT_CSV = BASE / "features" / "multilocation_features_2014_2025.csv"
TGT_CSV = BASE / "targets" / "multilocation_nowcast_targets_2014_2025.csv"
NWP_CSV = BASE / "nwp_full" / "nwp_gfs_pooled.csv"
OUT_DIR = BASE / "features_nwp"
OUT_CSV = OUT_DIR / "multilocation_features_nwp_overlap_2021_2025.csv"
OUT_META = OUT_DIR / "multilocation_features_nwp_overlap_2021_2025_metadata.json"
REPORT = ROOT / "docs" / "PHASE8B_NWP_OVERLAP_REPORT.md"

STATIONS = ["VOTV", "VECC", "VIDP", "VOCI", "VABB"]
TIME = "timestamp_utc"
STATION = "station_id"
HORIZONS = ("target_1h", "target_2h", "target_3h")

PHASE3_FEATURES = [
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
    "wind_u_10m",
    "wind_v_10m",
    "temperature_lag_1h",
    "temperature_lag_3h",
    "temperature_lag_6h",
    "temperature_lag_12h",
    "temperature_lag_24h",
    "humidity_lag_1h",
    "humidity_lag_3h",
    "humidity_lag_6h",
    "humidity_lag_12h",
    "humidity_lag_24h",
    "pressure_lag_1h",
    "pressure_lag_3h",
    "pressure_lag_6h",
    "pressure_lag_12h",
    "pressure_lag_24h",
    "wind_speed_lag_1h",
    "wind_speed_lag_3h",
    "wind_speed_lag_6h",
    "wind_speed_lag_12h",
    "wind_speed_lag_24h",
    "precipitation_lag_1h",
    "precipitation_lag_3h",
    "precipitation_lag_6h",
    "precipitation_lag_12h",
    "precipitation_lag_24h",
    "cloud_cover_lag_1h",
    "cloud_cover_lag_3h",
    "cloud_cover_lag_6h",
    "cloud_cover_lag_12h",
    "cloud_cover_lag_24h",
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
    "cloud_cover_change_1h",
    "cloud_cover_change_3h",
    "precipitation_roll_sum_3h",
    "precipitation_roll_sum_6h",
    "precipitation_roll_sum_12h",
    "precipitation_roll_sum_24h",
    "humidity_roll_mean_3h",
    "humidity_roll_mean_6h",
    "pressure_roll_mean_3h",
    "pressure_roll_mean_6h",
    "temperature_roll_mean_3h",
    "temperature_roll_mean_6h",
    "wind_speed_roll_mean_3h",
    "wind_speed_roll_mean_6h",
    "cloud_cover_roll_mean_3h",
    "cloud_cover_roll_mean_6h",
    "latitude",
    "longitude",
    "elevation_m",
]

NWP_RAW = [
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
NWP_PREFIXED = [f"nwp_{c}" for c in NWP_RAW]

FORBIDDEN_AS_FEATURES = ("target", "label", "weather_code", "thunderstorm")


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(PHASE3_FEATURES) != 73:
        raise SystemExit("expected 73 Phase 3 features")

    log("Loading Phase 3 features (keys + 73 columns)")
    feats = pd.read_csv(FEAT_CSV, usecols=[STATION, TIME] + PHASE3_FEATURES)
    feats[TIME] = pd.to_datetime(feats[TIME], utc=True, format="ISO8601")

    log("Loading Phase 5 targets")
    tgt_cols = [STATION, TIME] + list(HORIZONS) + [f"{h}_observed" for h in HORIZONS]
    tgts = pd.read_csv(TGT_CSV, usecols=tgt_cols)
    tgts[TIME] = pd.to_datetime(tgts[TIME], utc=True, format="ISO8601")

    log("Loading NWP")
    nwp = pd.read_csv(NWP_CSV, usecols=["station_id", "valid_time_utc"] + NWP_RAW)
    nwp[TIME] = pd.to_datetime(nwp["valid_time_utc"], utc=True)
    nwp = nwp.drop(columns=["valid_time_utc"])
    nwp = nwp.rename(columns={c: f"nwp_{c}" for c in NWP_RAW})

    nwp_min, nwp_max = nwp[TIME].min(), nwp[TIME].max()
    log(f"Restricting features/targets to NWP window {nwp_min} -> {nwp_max}")
    feats = feats.loc[(feats[TIME] >= nwp_min) & (feats[TIME] <= nwp_max)].copy()
    tgts = tgts.loc[(tgts[TIME] >= nwp_min) & (tgts[TIME] <= nwp_max)].copy()

    ft = feats.merge(tgts, on=[STATION, TIME], how="inner", validate="one_to_one")
    if len(ft) != len(feats):
        raise SystemExit(f"feature/target overlap mismatch {len(feats)} vs {len(ft)}")

    joined = ft.merge(nwp, on=[STATION, TIME], how="inner", validate="one_to_one")
    log(f"Inner join rows: {len(joined)}")

    if set(joined[STATION].unique()) != set(STATIONS):
        raise SystemExit(f"station mismatch: {sorted(joined[STATION].unique())}")
    dups = int(joined.duplicated([STATION, TIME]).sum())
    if dups:
        raise SystemExit(f"duplicate keys: {dups}")

    leaked = [
        c
        for c in PHASE3_FEATURES + NWP_PREFIXED
        if any(s in c.lower() for s in FORBIDDEN_AS_FEATURES)
    ]
    if leaked:
        raise SystemExit(f"forbidden predictor names: {leaked}")

    out_cols = [STATION, TIME] + PHASE3_FEATURES + NWP_PREFIXED + list(HORIZONS) + [
        f"{h}_observed" for h in HORIZONS
    ]
    joined = joined[out_cols].sort_values([STATION, TIME]).reset_index(drop=True)

    n_rows = int(len(joined))
    per_station = {st: int((joined[STATION] == st).sum()) for st in STATIONS}
    tmin = joined[TIME].min().isoformat()
    tmax = joined[TIME].max().isoformat()

    nwp_miss = {}
    for c in NWP_PREFIXED:
        n_miss = int(joined[c].isna().sum())
        nwp_miss[c] = {"missing_count": n_miss, "missing_pct": round(100.0 * n_miss / n_rows, 4)}

    any_nwp_miss = int(joined[NWP_PREFIXED].isna().any(axis=1).sum())
    all_nwp_ok = int(joined[NWP_PREFIXED].notna().all(axis=1).sum())

    tgt_miss = {}
    for h in HORIZONS:
        n_miss = int(joined[h].isna().sum())
        tgt_miss[h] = {"missing_count": n_miss, "missing_pct": round(100.0 * n_miss / n_rows, 4)}

    log(f"Writing {OUT_CSV}")
    joined.to_csv(OUT_CSV, index=False)

    meta = {
        "phase": "8B",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "join_key": [STATION, TIME],
        "nwp_time_shift": False,
        "nwp_source": "Open-Meteo Historical Forecast gfs_global (not observation, not ERA5)",
        "overlap_start_utc": tmin,
        "overlap_end_utc": tmax,
        "n_rows": n_rows,
        "rows_per_station": per_station,
        "duplicate_station_timestamp": dups,
        "phase3_feature_count": len(PHASE3_FEATURES),
        "phase3_features": PHASE3_FEATURES,
        "nwp_columns": NWP_PREFIXED,
        "targets": list(HORIZONS),
        "targets_missing_filled_with_zero": False,
        "nwp_missingness": nwp_miss,
        "target_missingness": tgt_miss,
        "rows_all_nwp_present": all_nwp_ok,
        "rows_any_nwp_missing": any_nwp_miss,
        "stations": STATIONS,
        "csv": str(OUT_CSV.relative_to(ROOT)).replace("\\", "/"),
        "phase3_features_dir_overwritten": False,
        "training": False,
        "imputation": False,
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8B — NWP overlap dataset (V2)",
        "",
        f"**Generated:** {meta['generated_at_utc']}",
        "**Script:** `dataset/multilocation/build_nwp_overlap_phase8b.py`",
        "**Scope:** overlap table only. No training, no imputation, no models.",
        "",
        "Phase 3 feature CSVs, Phase 4/5/6 artifacts, and V1 were not modified.",
        "",
        "## Objective",
        "",
        "Build a new table for `2021-04-01` → `2025-12-31` where Phase 3 features at T, Phase 5 genuine METAR leads, and Phase 7C GFS NWP at the **same** T can be aligned.",
        "",
        "## Join",
        "",
        "| item | value |",
        "| --- | --- |",
        "| key | `station_id` + UTC timestamp |",
        "| NWP clock | `valid_time_utc` parsed UTC, no shift |",
        "| NWP product | Historical Forecast `gfs_global` (forecast, not observation) |",
        "| Phase 3 columns | all 73, unchanged names/values |",
        "| NWP columns | 10 fields, `nwp_` prefix |",
        "| targets | `target_1h`, `target_2h`, `target_3h` (NA left as NA) |",
        "| not included as predictors | weather_code, BLH, same-hour thunderstorm label |",
        "",
        "## Coverage",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| total rows | {n_rows} |",
        f"| start UTC | {tmin} |",
        f"| end UTC | {tmax} |",
        f"| duplicate station/timestamp | {dups} |",
        f"| five stations present | {set(joined[STATION].unique()) == set(STATIONS)} |",
        f"| all NWP fields present | {all_nwp_ok} |",
        f"| any NWP missing | {any_nwp_miss} |",
        "",
        "| station | rows |",
        "| --- | --- |",
    ]
    for st in STATIONS:
        lines.append(f"| {st} | {per_station[st]} |")
    lines += [
        "",
        "## NWP missingness (no fill)",
        "",
        "| column | missing count | missing % |",
        "| --- | --- | --- |",
    ]
    for c in NWP_PREFIXED:
        m = nwp_miss[c]
        lines.append(f"| {c} | {m['missing_count']} | {m['missing_pct']}% |")
    lines += [
        "",
        "## Target missingness (not filled with zero)",
        "",
        "| target | missing count | missing % |",
        "| --- | --- | --- |",
    ]
    for h in HORIZONS:
        m = tgt_miss[h]
        lines.append(f"| {h} | {m['missing_count']} | {m['missing_pct']}% |")
    lines += [
        "",
        "## Artifacts",
        "",
        "| path | role |",
        "| --- | --- |",
        f"| `{meta['csv']}` | pooled overlap CSV |",
        "| `dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025_metadata.json` | metadata |",
        "| `docs/PHASE8B_NWP_OVERLAP_REPORT.md` | this report |",
        "",
        "Existing `dataset/multilocation/features/` was not overwritten.",
        "",
        "## PHASE STATUS",
        "",
        "**READY** — overlap dataset only. Do not proceed to Phase 8C in this task.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {REPORT}")
    log(f"rows={n_rows} all_nwp={all_nwp_ok} any_nwp_miss={any_nwp_miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
