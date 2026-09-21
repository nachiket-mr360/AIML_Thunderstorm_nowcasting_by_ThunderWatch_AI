"""Phase 3 (V2): causal multi-location feature engineering.

Reads Phase 2B synchronized hourly tables (contiguous predictor hours, genuine
METAR targets with NA preserved). Writes only under dataset/multilocation/features/
and docs/PHASE3_FEATURE_ENGINEERING_REPORT.md.

Every feature at hour T is a function of atmospheric values at or before T on the
contiguous UTC hourly grid. The thunderstorm target is never an input.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent
REPO = DATASET.parent
SYNC_DIR = HERE / "synchronized"
OUT_DIR = HERE / "features"
IEM_META = HERE / "iem_station_metadata.json"
OUT_COMBINED = OUT_DIR / "multilocation_features_2014_2025.csv"
OUT_META = OUT_DIR / "multilocation_features_2014_2025_metadata.json"
OUT_REPORT = REPO / "docs" / "PHASE3_FEATURE_ENGINEERING_REPORT.md"

STATIONS = ["VOTV", "VECC", "VIDP", "VOCI", "VABB"]
TIME = "timestamp_utc"
TARGET = "thunderstorm_target"

PREDICTORS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]

# V1 current atmospheric (raw wind direction is encoded as sin/cos, not used raw).
CURRENT_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
]

LAG_VARIABLES = [
    ("temperature_2m", "temperature"),
    ("relative_humidity_2m", "humidity"),
    ("surface_pressure", "pressure"),
    ("wind_speed_10m", "wind_speed"),
    ("precipitation", "precipitation"),
    ("cloud_cover", "cloud_cover"),
]
LAG_HOURS = (1, 3, 6, 12, 24)

CHANGE_VARIABLES = [
    ("temperature_2m", "temperature"),
    ("relative_humidity_2m", "humidity"),
    ("surface_pressure", "pressure"),
    ("wind_speed_10m", "wind_speed"),
    ("precipitation", "precipitation"),
    ("cloud_cover", "cloud_cover"),
]
CHANGE_HOURS = (1, 3)  # V1 used 1h/3h; 6/12/24 captured by explicit lags

V1_CHANGE_VARIABLES = [
    ("temperature_2m", "temperature"),
    ("relative_humidity_2m", "humidity"),
    ("surface_pressure", "pressure"),
    ("wind_speed_10m", "wind_speed"),
    ("precipitation", "precipitation"),
]

CYCLIC_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
WIND_DIRECTION_FEATURES = ["wind_direction_sin", "wind_direction_cos"]
WIND_UV_FEATURES = ["wind_u_10m", "wind_v_10m"]
LOCATION_FEATURES = ["latitude", "longitude", "elevation_m"]

LAG_FEATURES = [f"{short}_lag_{h}h" for _, short in LAG_VARIABLES for h in LAG_HOURS]
CHANGE_FEATURES = [f"{short}_change_{h}h" for _, short in CHANGE_VARIABLES for h in CHANGE_HOURS]
ROLLING_FEATURES = [
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
]

FEATURE_COLUMNS = (
    CURRENT_FEATURES
    + CYCLIC_FEATURES
    + WIND_DIRECTION_FEATURES
    + WIND_UV_FEATURES
    + LAG_FEATURES
    + CHANGE_FEATURES
    + ROLLING_FEATURES
    + LOCATION_FEATURES
)

V1_FEATURE_COLUMNS = (
    CURRENT_FEATURES
    + CYCLIC_FEATURES
    + WIND_DIRECTION_FEATURES
    + [f"{name}_change_{h}h" for _, name in V1_CHANGE_VARIABLES for h in (1, 3)]
    + [
        "precipitation_roll_sum_3h",
        "precipitation_roll_sum_6h",
        "humidity_roll_mean_3h",
        "pressure_roll_mean_3h",
        "temperature_roll_mean_3h",
        "wind_speed_roll_mean_3h",
    ]
)

MAX_HISTORY_HOURS = 24

V1_PROTECTED = [
    DATASET / "feature_engineering_phase4.py",
    DATASET / "votv_thunderstorm_features_2014_2025.csv",
    DATASET / "votv_thunderstorm_features_2014_2025_metadata.json",
    DATASET / "votv_thunderstorm_synchronized_2014_2025.csv",
    DATASET / "historical_thunderstorm_labels_votv.csv",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_catalog() -> dict:
    catalog = {}
    for col in CURRENT_FEATURES:
        catalog[col] = {
            "group": "current_atmospheric",
            "formula": f"{col}(T) from Open-Meteo archive hour T",
            "lookback_hours": 0,
            "source": "Open-Meteo",
        }
    catalog["hour_sin"] = {
        "group": "cyclic_time",
        "formula": "sin(2π · hour_utc(T) / 24)",
        "lookback_hours": 0,
        "source": "timestamp_utc",
    }
    catalog["hour_cos"] = {
        "group": "cyclic_time",
        "formula": "cos(2π · hour_utc(T) / 24)",
        "lookback_hours": 0,
        "source": "timestamp_utc",
    }
    catalog["month_sin"] = {
        "group": "cyclic_time",
        "formula": "sin(2π · (month(T)−1) / 12)",
        "lookback_hours": 0,
        "source": "timestamp_utc",
    }
    catalog["month_cos"] = {
        "group": "cyclic_time",
        "formula": "cos(2π · (month(T)−1) / 12)",
        "lookback_hours": 0,
        "source": "timestamp_utc",
    }
    catalog["wind_direction_sin"] = {
        "group": "wind_vector",
        "formula": "sin(radians(wind_direction_10m(T))); meteorological FROM direction",
        "lookback_hours": 0,
        "source": "Open-Meteo wind_direction_10m",
    }
    catalog["wind_direction_cos"] = {
        "group": "wind_vector",
        "formula": "cos(radians(wind_direction_10m(T)))",
        "lookback_hours": 0,
        "source": "Open-Meteo wind_direction_10m",
    }
    catalog["wind_u_10m"] = {
        "group": "wind_vector",
        "formula": "−wind_speed_10m(T) · sin(radians(wind_direction_10m(T)))  (eastward, met. convention)",
        "lookback_hours": 0,
        "source": "Open-Meteo",
    }
    catalog["wind_v_10m"] = {
        "group": "wind_vector",
        "formula": "−wind_speed_10m(T) · cos(radians(wind_direction_10m(T)))  (northward, met. convention)",
        "lookback_hours": 0,
        "source": "Open-Meteo",
    }
    for col, short in LAG_VARIABLES:
        for h in LAG_HOURS:
            catalog[f"{short}_lag_{h}h"] = {
                "group": "lag",
                "formula": f"{col}(T−{h}h) on the contiguous UTC hourly grid",
                "lookback_hours": h,
                "source": "Open-Meteo",
            }
    for col, short in CHANGE_VARIABLES:
        for h in CHANGE_HOURS:
            catalog[f"{short}_change_{h}h"] = {
                "group": "tendency",
                "formula": f"{col}(T) − {col}(T−{h}h)",
                "lookback_hours": h,
                "source": "Open-Meteo",
            }
    for h in (3, 6, 12, 24):
        catalog[f"precipitation_roll_sum_{h}h"] = {
            "group": "rolling",
            "formula": f"sum(precipitation(T−{h}+1 … T)) trailing, current hour included",
            "lookback_hours": h,
            "source": "Open-Meteo",
        }
    for col, short in [
        ("relative_humidity_2m", "humidity"),
        ("surface_pressure", "pressure"),
        ("temperature_2m", "temperature"),
        ("wind_speed_10m", "wind_speed"),
        ("cloud_cover", "cloud_cover"),
    ]:
        for h in (3, 6):
            catalog[f"{short}_roll_mean_{h}h"] = {
                "group": "rolling",
                "formula": f"mean({col}(T−{h}+1 … T)) trailing, current hour included",
                "lookback_hours": h,
                "source": "Open-Meteo",
            }
    catalog["latitude"] = {
        "group": "location",
        "formula": "IEM IN__ASOS station latitude (constant per station)",
        "lookback_hours": 0,
        "source": "iem_station_metadata.json",
    }
    catalog["longitude"] = {
        "group": "location",
        "formula": "IEM IN__ASOS station longitude (constant per station)",
        "lookback_hours": 0,
        "source": "iem_station_metadata.json",
    }
    catalog["elevation_m"] = {
        "group": "location",
        "formula": "IEM IN__ASOS station elevation_m (constant per station)",
        "lookback_hours": 0,
        "source": "iem_station_metadata.json",
    }
    return catalog


def add_features(grid: pd.DataFrame, elevation_m: float) -> pd.DataFrame:
    """Causal features on one station's contiguous ascending hourly frame."""
    out = grid.copy()
    ts = out[TIME]
    hour = ts.dt.hour.to_numpy(dtype=float)
    month = ts.dt.month.to_numpy(dtype=float)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)

    wd = np.deg2rad(out["wind_direction_10m"].to_numpy(dtype=float))
    ws = out["wind_speed_10m"].to_numpy(dtype=float)
    out["wind_direction_sin"] = np.sin(wd)
    out["wind_direction_cos"] = np.cos(wd)
    # Meteorological wind FROM direction → earth-relative u (east), v (north).
    out["wind_u_10m"] = -ws * np.sin(wd)
    out["wind_v_10m"] = -ws * np.cos(wd)

    for column, short in LAG_VARIABLES:
        series = out[column]
        for h in LAG_HOURS:
            out[f"{short}_lag_{h}h"] = series.shift(h)

    for column, short in CHANGE_VARIABLES:
        series = out[column]
        for h in CHANGE_HOURS:
            out[f"{short}_change_{h}h"] = series - series.shift(h)

    precip = out["precipitation"]
    out["precipitation_roll_sum_3h"] = precip.rolling(window=3).sum()
    out["precipitation_roll_sum_6h"] = precip.rolling(window=6).sum()
    out["precipitation_roll_sum_12h"] = precip.rolling(window=12).sum()
    out["precipitation_roll_sum_24h"] = precip.rolling(window=24).sum()
    for column, short in [
        ("relative_humidity_2m", "humidity"),
        ("surface_pressure", "pressure"),
        ("temperature_2m", "temperature"),
        ("wind_speed_10m", "wind_speed"),
        ("cloud_cover", "cloud_cover"),
    ]:
        series = out[column]
        out[f"{short}_roll_mean_3h"] = series.rolling(window=3).mean()
        out[f"{short}_roll_mean_6h"] = series.rolling(window=6).mean()

    out["elevation_m"] = float(elevation_m)
    for column in FEATURE_COLUMNS:
        out[column] = out[column].astype(float)
    return out


def causality_probe(raw_grid: pd.DataFrame, shipped: pd.DataFrame, elevation_m: float, n_probes: int = 12, seed: int = 20260919) -> dict:
    rng = np.random.default_rng(seed)
    n = len(shipped)
    candidates = np.arange(MAX_HISTORY_HOURS, n)
    picks = np.sort(rng.choice(candidates, size=min(n_probes, len(candidates)), replace=False))
    mismatches = []
    worst = 0.0
    for i in picks:
        truncated = raw_grid.iloc[: i + 1]
        recomputed = add_features(truncated, elevation_m).iloc[-1]
        stored = shipped.iloc[i]
        for column in FEATURE_COLUMNS:
            if column in LOCATION_FEATURES:
                continue
            a = float(recomputed[column])
            b = float(stored[column])
            delta = abs(a - b)
            worst = max(worst, delta)
            if not np.isclose(a, b, atol=1e-10, rtol=0.0, equal_nan=True):
                mismatches.append({"index": int(i), "column": column, "abs_difference": delta})
    return {
        "method": "recompute features from truncated station grid (rows <= T) vs shipped row T",
        "rows_probed": int(len(picks)),
        "max_abs_difference": worst,
        "mismatches": len(mismatches),
        "mismatch_examples": mismatches[:8],
        "verdict": "PASS" if not mismatches else "FAIL",
    }


def process_station(station: str, elev: float) -> tuple[pd.DataFrame, dict]:
    path = SYNC_DIR / f"{station.lower()}_synchronized_2014_2025.csv"
    df = pd.read_csv(path)
    df[TIME] = pd.to_datetime(df[TIME], utc=True)
    df = df.sort_values(TIME).reset_index(drop=True)
    input_rows = int(len(df))
    if df[TIME].duplicated().any():
        raise SystemExit(f"{station}: duplicate timestamps")
    deltas = df[TIME].diff().dropna()
    contiguous = bool((deltas == pd.Timedelta(hours=1)).all())
    if not contiguous:
        raise SystemExit(f"{station}: predictor grid is not contiguous hourly; lags would be unsafe")

    if "weather_code" in df.columns:
        raise SystemExit(f"{station}: weather_code present; refused")

    grid_f = add_features(df, elev)
    history_ok = grid_f[LAG_FEATURES + CHANGE_FEATURES + ROLLING_FEATURES].notna().all(axis=1)
    n_lost = int((~history_ok).sum())
    first_valid = grid_f.loc[history_ok, TIME].iloc[0] if history_ok.any() else None
    expected_first = df[TIME].iloc[0] + pd.Timedelta(hours=MAX_HISTORY_HOURS - 1)
    # rolling(24) and lag(24) become defined at index 24 (0-based), i.e. T0+24h.
    # shift(24) is NA for first 24 rows (indices 0..23); rolling(24) NA for first 23.
    # Combined: first 24 rows lost. first valid timestamp = start + 24h.
    expected_first = df[TIME].iloc[0] + pd.Timedelta(hours=MAX_HISTORY_HOURS)

    kept = grid_f.loc[history_ok].copy().reset_index(drop=True)
    if len(kept) and kept[TIME].iloc[0] != expected_first:
        lookback_note = (
            f"first valid {kept[TIME].iloc[0].isoformat()} vs expected {expected_first.isoformat()}"
        )
    else:
        lookback_note = "first valid timestamp = station start + 24h (max lag/roll window)"

    probe_src = df[[TIME, *PREDICTORS, "latitude", "longitude"]].copy()
    probe_shipped = add_features(probe_src, elev)
    leakage = causality_probe(probe_src, probe_shipped, elev)

    feat_na = {c: int(kept[c].isna().sum()) for c in FEATURE_COLUMNS}
    identity = ["station_id", "city", TIME, TARGET, "target_observed", "label_status"]
    out_cols = identity + FEATURE_COLUMNS
    extra = [c for c in out_cols if c not in kept.columns]
    if extra:
        raise SystemExit(f"{station}: missing columns {extra}")
    out = kept[out_cols].copy()

    per_path = OUT_DIR / f"{station.lower()}_features_2014_2025.csv"
    out.to_csv(per_path, index=False)

    n_pos = int((out[TARGET] == 1).sum())
    n_neg = int((out[TARGET] == 0).sum())
    n_miss = int(out[TARGET].isna().sum())
    stats = {
        "station_id": station,
        "input_rows": input_rows,
        "output_rows": int(len(out)),
        "feature_count": len(FEATURE_COLUMNS),
        "rows_lost_lookback": n_lost,
        "first_input_timestamp_utc": df[TIME].iloc[0].isoformat(),
        "first_valid_feature_timestamp_utc": out[TIME].iloc[0].isoformat() if len(out) else None,
        "last_timestamp_utc": out[TIME].iloc[-1].isoformat() if len(out) else None,
        "lookback_check": lookback_note,
        "expected_first_valid_utc": expected_first.isoformat(),
        "missing_feature_values_total": int(sum(feat_na.values())),
        "missing_features_nonzero": {k: v for k, v in feat_na.items() if v},
        "duplicate_rows": int(out.duplicated().sum()),
        "duplicate_timestamps": int(out[TIME].duplicated().sum()),
        "timestamps_sorted": bool(out[TIME].is_monotonic_increasing),
        "contiguous_input": contiguous,
        "target_positives": n_pos,
        "target_negatives": n_neg,
        "target_unavailable": n_miss,
        "labels_filled_with_zero": False,
        "causality_probe": leakage,
        "csv": str(per_path.relative_to(REPO)).replace("\\", "/"),
    }
    return out, stats


def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def main() -> int:
    before = {str(p): sha256_file(p) for p in V1_PROTECTED if p.exists()}
    with IEM_META.open(encoding="utf-8") as fh:
        iem = json.load(fh)["stations"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    frames = []
    station_stats = {}

    for station in STATIONS:
        print(f"=== {station} ===", flush=True)
        elev = float(iem[station]["elevation_m"])
        out, stats = process_station(station, elev)
        station_stats[station] = stats
        frames.append(out)
        if stats["causality_probe"]["verdict"] != "PASS":
            problems.append(f"{station}: causality probe FAIL")
        if stats["missing_feature_values_total"]:
            problems.append(f"{station}: residual NaN features")
        if stats["duplicate_timestamps"]:
            problems.append(f"{station}: duplicate timestamps")
        if out[TIME].iloc[0] != pd.Timestamp(stats["expected_first_valid_utc"]):
            problems.append(f"{station}: first valid timestamp mismatch ({stats['lookback_check']})")
        print(
            f"  in={stats['input_rows']} out={stats['output_rows']} lost={stats['rows_lost_lookback']} "
            f"pos={stats['target_positives']} miss={stats['target_unavailable']} "
            f"probe={stats['causality_probe']['verdict']}",
            flush=True,
        )

    combined = pd.concat(frames, ignore_index=True)
    # Station separation: no mixed station_id within a timestamp group required,
    # but (station_id, timestamp) must be unique.
    if combined.duplicated(["station_id", TIME]).any():
        problems.append("pooled table has duplicate station+timestamp")
    station_mix = combined.groupby(TIME)["station_id"].nunique()
    combined.to_csv(OUT_COMBINED, index=False)

    # V1 definition comparison (formulas), optional value check if V1 feature file exists.
    v1_compare = {
        "v1_feature_count": 28,
        "v2_feature_count": len(FEATURE_COLUMNS),
        "v1_columns_reproduced": list(V1_FEATURE_COLUMNS),
        "v2_additions": [c for c in FEATURE_COLUMNS if c not in V1_FEATURE_COLUMNS],
        "v1_formulas_matched": {
            "cyclic_time": "identical sin/cos hour and (month-1)/12",
            "wind_direction": "identical sin/cos of degrees",
            "change_1h_3h": "identical x(T)-x(T-h) for T, RH, P, wind_speed, precipitation",
            "rolling": "identical trailing mean/sum with current hour included for V1 3h/6h set",
            "cloud_cover_change": "V2 addition (not in V1 28)",
            "lags": "V2 addition; V1 encoded 1h/3h via change features only",
            "wind_uv": "V2 addition; meteorological u/v",
            "location": "V2 addition for pooled training",
        },
        "note": "V1 Phase 4 dropped unsupervised hours then dropped first 6h of window; V2 keeps unavailable targets and drops first 24h per station for the longer lookback.",
    }
    v1_feat = DATASET / "votv_thunderstorm_features_2014_2025.csv"
    if v1_feat.exists():
        v1 = pd.read_csv(v1_feat, usecols=lambda c: c in {"timestamp_utc", *V1_FEATURE_COLUMNS})
        v1[TIME] = pd.to_datetime(v1["timestamp_utc"], utc=True)
        votv = frames[0]
        overlap = votv.merge(v1, on=TIME, how="inner", suffixes=("_v2", "_v1"))
        mismatches = {}
        for col in V1_FEATURE_COLUMNS:
            a = overlap[f"{col}_v2"].to_numpy(dtype=float)
            b = overlap[f"{col}_v1"].to_numpy(dtype=float)
            n_bad = int(np.sum(~np.isclose(a, b, atol=1e-6, rtol=0.0, equal_nan=True)))
            if n_bad:
                mismatches[col] = n_bad
        v1_compare["value_overlap_rows"] = int(len(overlap))
        v1_compare["value_mismatches_atol_1e-6"] = mismatches
        v1_compare["value_compare_verdict"] = "MATCH" if not mismatches else "DIFFER (Phase 2A Open-Meteo refetch may differ from V1 archive file)"

    after = {str(p): sha256_file(p) for p in V1_PROTECTED if p.exists()}
    if before != after:
        problems.append("V1 protected file hash changed")

    catalog = feature_catalog()
    metadata = {
        "phase": "3",
        "version": "SIH26072-V2-phase3-features-2014-2025",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "causal": True,
        "max_history_hours": MAX_HISTORY_HOURS,
        "target": {
            "name": TARGET,
            "missing_filled_with_zero": False,
            "label_status_values": [
                "observed_thunderstorm",
                "observed_non_thunderstorm",
                "unavailable",
            ],
        },
        "forbidden": [
            "future predictors",
            "future targets",
            "target-derived features",
            "Open-Meteo weather_code",
            "test-set statistics",
        ],
        "feature_count": len(FEATURE_COLUMNS),
        "feature_order": FEATURE_COLUMNS,
        "feature_catalog": catalog,
        "station_coordinates": {
            s: {
                "latitude": iem[s]["latitude"],
                "longitude": iem[s]["longitude"],
                "elevation_m": iem[s]["elevation_m"],
            }
            for s in STATIONS
        },
        "row_counts": {
            "combined_rows": int(len(combined)),
            "stations": station_stats,
        },
        "v1_comparison": v1_compare,
        "problems": problems,
        "v1_protected_unchanged": before == after,
        "combined_csv": str(OUT_COMBINED.relative_to(REPO)).replace("\\", "/"),
        "combined_sha256": sha256_file(OUT_COMBINED),
        "station_separation": {
            "unique_station_timestamp": bool(not combined.duplicated(["station_id", TIME]).any()),
            "features_computed_within_station_only": True,
        },
    }
    with OUT_META.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    status = "READY" if not problems else "NEEDS REVIEW"
    if any("hash changed" in p for p in problems) or any("causality" in p for p in problems):
        status = "BLOCKED" if any("hash changed" in p or "causality" in p for p in problems) else status

    rows = []
    for s in STATIONS:
        st = station_stats[s]
        rows.append(
            [
                s,
                st["input_rows"],
                st["output_rows"],
                st["feature_count"],
                st["rows_lost_lookback"],
                st["missing_feature_values_total"],
                st["duplicate_timestamps"],
                st["timestamps_sorted"],
                st["target_positives"],
                st["target_negatives"],
                st["target_unavailable"],
                st["causality_probe"]["verdict"],
                st["first_valid_feature_timestamp_utc"],
            ]
        )

    report = f"""# Phase 3 — Multi-location causal feature engineering

**Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d")}  
**Repo:** V2 `{REPO}`  
**V1 files:** not modified (hash check: {before == after})  
**Flask / frontend / models:** not modified  
**Git:** no commit, no push  
**Training:** not run

---

## PHASE STATUS: **{status}**

{"Problems: " + "; ".join(problems) if problems else "Causal probes passed. Missing METAR targets remain NA. No weather_code features."}

---

## Design

Features at hour **T** use only Open-Meteo values at **T and earlier** on each station’s contiguous UTC hourly grid. Lags and rolling windows are **not** evaluated on a table compacted to observed labels (that would jump across missing-METAR hours).

| Rule | Applied |
|------|---------|
| No future predictors | `shift(h)` and trailing `rolling(h)` only |
| No future / current target as feature | `thunderstorm_target` copied, never used in formulas |
| No weather_code | refused if present |
| Missing labels | kept as NA; not filled with 0 |
| Lookback | first **24** hours per station dropped (max lag/roll), not imputed |
| Station isolation | features computed independently per ICAO |

Numeric feature count: **{len(FEATURE_COLUMNS)}** (V1 had 28). Identity columns (`station_id`, `city`, timestamps, target, `label_status`) are extra.

---

## Feature groups

| Group | Columns | Formula sketch |
|-------|---------|----------------|
| Current state | 6 | Open-Meteo hour T (raw wind direction not a model column) |
| Cyclic time | 4 | hour/month sin/cos (V1 identical) |
| Wind vector | 4 | dir sin/cos; u,v = −speed·sin/cos(dir) (met. FROM) |
| Lags | 30 | 6 variables × {{1,3,6,12,24}} h |
| Tendencies | 12 | Δ at 1h and 3h including cloud cover |
| Rolling | 14 | precip sum 3/6/12/24h; means 3h and 6h |
| Location | 3 | IEM lat, lon, elevation_m |

Explicit lag_1h/lag_3h coexist with change_1h/change_3h (change = current − lag). Both are kept so V1’s 28-feature set is a strict subset of formulas.

Full catalog: `dataset/multilocation/features/multilocation_features_2014_2025_metadata.json`.

---

## Per-station validation

{md_table(
    [
        "Station", "Input rows", "Output rows", "Features", "Lost (lookback)",
        "Feature NaNs", "Dup ts", "Sorted", "TS+", "Obs −", "Unavailable",
        "Causal probe", "First valid T",
    ],
    rows,
)}

First valid feature timestamp is **start + 24h** (`2014-01-01 00:00Z` → `2014-01-02 00:00Z`) because `lag_24h` and `precipitation_roll_sum_24h` need 24 hours of history. That is determined only by lookback, not by labels.

Pooled rows: **{len(combined)}**. Unique `(station_id, timestamp_utc)`: **{not combined.duplicated(["station_id", TIME]).any()}**.

---

## V1 28-feature comparison (definitions)

V1 Phase 4 columns reproduced with the same formulas: current 6, cyclic 4, wind sin/cos, 1h/3h changes for T/RH/P/wind/precip, precip roll sum 3h/6h, 3h means for humidity/pressure/temperature/wind_speed.

V2 additions: lags 1–24h, cloud-cover change and roll, 6h/12h/24h rolling precip, 6h means, meteorological u/v, IEM location.

V1 dropped unsupervised hours and only needed 6h history (3 supervised rows dropped at the start of 2014). V2 **keeps unavailable targets** and drops **24** hours of each station grid.

Value overlap vs `dataset/votv_thunderstorm_features_2014_2025.csv`: {v1_compare.get("value_compare_verdict", "not run")} (overlap rows={v1_compare.get("value_overlap_rows", "n/a")}; mismatches={v1_compare.get("value_mismatches_atol_1e-6", {{}})}).

---

## Artifacts

| Path | Role |
|------|------|
| `dataset/multilocation/feature_engineering_phase3.py` | Reusable builder |
| `dataset/multilocation/features/<icao>_features_2014_2025.csv` | Per station |
| `dataset/multilocation/features/multilocation_features_2014_2025.csv` | Pooled |
| `dataset/multilocation/features/multilocation_features_2014_2025_metadata.json` | Feature catalog + counts |

SHA256 pooled: `{metadata["combined_sha256"]}`

---

## Not done

- No training / metrics
- No Flask/frontend
- No radar/satellite/lightning/NWP
- No V1 edits
- No git commit/push
- No t+1/t+2/t+3 targets
"""
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"\nSTATUS {status}")
    print(f"combined rows={len(combined)} features={len(FEATURE_COLUMNS)}")
    print(OUT_REPORT)
    return 1 if status == "BLOCKED" else 0


if __name__ == "__main__":
    sys.exit(main())
