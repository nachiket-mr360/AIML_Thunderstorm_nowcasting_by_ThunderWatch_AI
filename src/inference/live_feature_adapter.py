"""Convert live Open-Meteo Forecast/GFS JSON into the frozen 83-feature vector.

Reuses Phase 14A V2FeatureBuilder (Phase 3 formulas). Fail-closed: no fills.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.features.v2_feature_builder import (
    ATMOSPHERIC_SOURCE,
    MIN_HISTORY_ROWS,
    NWP_SOURCE,
    V2FeatureBuilder,
)
from src.inference.live_openmeteo_client import LiveDataError

BASE_DIR = Path(__file__).resolve().parents[2]
IEM_META = BASE_DIR / "dataset" / "multilocation" / "iem_station_metadata.json"

# V1/V2 Open-Meteo request point for VOTV (not IEM lat/lon).
REQUEST_COORDS = {
    "VOTV": (8.482, 76.920),
}

MAX_DATA_AGE_MINUTES = 180
NWP_API_TO_MODEL = {
    "cape": "nwp_cape",
    "convective_inhibition": "nwp_convective_inhibition",
    "lifted_index": "nwp_lifted_index",
    "temperature_2m": "nwp_temperature_2m",
    "relative_humidity_2m": "nwp_relative_humidity_2m",
    "surface_pressure": "nwp_surface_pressure",
    "wind_speed_10m": "nwp_wind_speed_10m",
    "wind_direction_10m": "nwp_wind_direction_10m",
    "precipitation": "nwp_precipitation",
    "cloud_cover": "nwp_cloud_cover",
}


def load_station_catalog(path: Path | None = None) -> dict[str, dict[str, Any]]:
    payload = json.loads((path or IEM_META).read_text(encoding="utf-8"))
    stations = payload["stations"]
    out: dict[str, dict[str, Any]] = {}
    for sid, meta in stations.items():
        if sid in REQUEST_COORDS:
            lat, lon = REQUEST_COORDS[sid]
        else:
            lat, lon = float(meta["latitude"]), float(meta["longitude"])
        out[sid] = {
            "station_id": sid,
            "request_latitude": lat,
            "request_longitude": lon,
            "latitude": float(meta["latitude"]),
            "longitude": float(meta["longitude"]),
            "elevation_m": float(meta["elevation_m"]),
            "name": meta.get("iem_name"),
        }
    return out


def _hourly_frame(payload: Mapping[str, Any], required: list[str]) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise LiveDataError("LIVE_DATA_UNAVAILABLE", "malformed hourly block")
    times = hourly["time"]
    if not isinstance(times, list) or not times:
        raise LiveDataError("LIVE_DATA_UNAVAILABLE", "empty hourly times")
    missing = [k for k in required if k not in hourly]
    if missing:
        raise LiveDataError(
            "LIVE_DATA_UNAVAILABLE",
            f"missing required API variable: {missing}",
        )
    df = pd.DataFrame({"timestamp_utc": pd.to_datetime(times, utc=True)})
    for k in required:
        df[k] = hourly[k]
    return df


def _finite_row(row: pd.Series, cols: list[str]) -> bool:
    for c in cols:
        try:
            x = float(row[c])
        except (TypeError, ValueError):
            return False
        if not np.isfinite(x):
            return False
    return True


def build_live_history(
    station_id: str,
    forecast_payload: Mapping[str, Any],
    gfs_payload: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    catalog: dict[str, dict[str, Any]] | None = None,
    max_age_minutes: int = MAX_DATA_AGE_MINUTES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return causal history ending at observation T plus provenance."""
    catalog = catalog if catalog is not None else load_station_catalog()
    if station_id not in catalog:
        raise LiveDataError("LIVE_DATA_UNAVAILABLE", f"unknown station {station_id}")
    meta = catalog[station_id]
    now = now_utc if now_utc is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise LiveDataError("LIVE_FEATURE_CONSTRUCTION_FAILED", "now_utc must be timezone-aware")
    now = now.astimezone(timezone.utc)

    atmo = _hourly_frame(forecast_payload, ATMOSPHERIC_SOURCE)
    nwp_req = list(NWP_API_TO_MODEL.keys())
    nwp = _hourly_frame(gfs_payload, nwp_req)
    nwp = nwp.rename(columns=NWP_API_TO_MODEL)

    merged = atmo.merge(nwp, on="timestamp_utc", how="inner")
    merged = merged.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
    merged = merged.loc[merged["timestamp_utc"] <= pd.Timestamp(now)].copy()
    if merged.empty:
        raise LiveDataError("INSUFFICIENT_HISTORY", "no hours at or before now")

    usable_idx = []
    need = ATMOSPHERIC_SOURCE + list(NWP_SOURCE)
    for i, row in merged.iterrows():
        if _finite_row(row, need):
            usable_idx.append(i)
    if not usable_idx:
        raise LiveDataError("LIVE_FEATURE_CONSTRUCTION_FAILED", "no finite complete hours")

    t_row = merged.loc[usable_idx[-1]]
    t_obs = pd.Timestamp(t_row["timestamp_utc"]).to_pydatetime()
    if t_obs.tzinfo is None:
        t_obs = t_obs.replace(tzinfo=timezone.utc)
    age_min = (now - t_obs.astimezone(timezone.utc)).total_seconds() / 60.0
    if age_min > max_age_minutes:
        raise LiveDataError(
            "LIVE_DATA_UNAVAILABLE",
            f"stale observation age_minutes={age_min:.1f} exceeds {max_age_minutes}",
        )

    # Forecast valid times for lead scores (not used as predictors).
    forecast_valid = {
        "1h": (t_obs + timedelta(hours=1)).astimezone(timezone.utc).isoformat(),
        "2h": (t_obs + timedelta(hours=2)).astimezone(timezone.utc).isoformat(),
        "3h": (t_obs + timedelta(hours=3)).astimezone(timezone.utc).isoformat(),
    }
    nwp_times = set(pd.to_datetime(nwp["timestamp_utc"], utc=True))
    for lead, iso in forecast_valid.items():
        ts = pd.Timestamp(iso)
        if ts not in nwp_times:
            raise LiveDataError(
                "LIVE_DATA_UNAVAILABLE",
                f"forecast horizon {lead} unavailable at {iso}",
            )

    causal = merged.loc[merged["timestamp_utc"] <= t_row["timestamp_utc"]].copy()
    if len(causal) < MIN_HISTORY_ROWS:
        raise LiveDataError(
            "INSUFFICIENT_HISTORY",
            f"need {MIN_HISTORY_ROWS} hourly rows, got {len(causal)}",
        )
    span_h = (causal["timestamp_utc"].iloc[-1] - causal["timestamp_utc"].iloc[0]) / pd.Timedelta(
        hours=1
    )
    if span_h < 24:
        raise LiveDataError("INSUFFICIENT_HISTORY", f"history span {span_h}h < 24h")

    causal["station_id"] = station_id
    causal["latitude"] = meta["latitude"]
    causal["longitude"] = meta["longitude"]
    causal["elevation_m"] = meta["elevation_m"]

    builder = V2FeatureBuilder()
    built = builder.build_features(
        causal,
        timestamp_utc=t_row["timestamp_utc"],
        station_id=station_id,
    )
    if not built.ok or built.features is None:
        reason = built.reason or "feature construction failed"
        if "insufficient_history" in reason:
            code = "INSUFFICIENT_HISTORY"
        elif "order" in reason or "nonfinite" in reason or "mismatch" in reason:
            code = "LIVE_MODEL_INPUT_INVALID"
        else:
            code = "LIVE_FEATURE_CONSTRUCTION_FAILED"
        raise LiveDataError(code, reason)

    names = builder.feature_names
    if list(built.features.keys()) != names or len(names) != 83:
        raise LiveDataError("LIVE_MODEL_INPUT_INVALID", "feature count/order invalid")
    vec = [float(built.features[n]) for n in names]
    if not all(np.isfinite(vec)):
        raise LiveDataError("LIVE_MODEL_INPUT_INVALID", "non-finite feature")

    provenance = {
        "station_id": station_id,
        "prediction_time_utc": now.isoformat(),
        "data_observation_time_utc": t_obs.astimezone(timezone.utc).isoformat(),
        "forecast_valid_times_utc": forecast_valid,
        "data_age_minutes": round(age_min, 3),
        "data_source": "open-meteo-forecast+gfs_global",
        "feature_names": names,
        "features": built.features,
        "feature_validation": {
            "feature_count": 83,
            "feature_order_valid": True,
            "all_finite": True,
        },
    }
    return causal, provenance
