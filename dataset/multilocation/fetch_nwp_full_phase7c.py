"""Phase 7C: full Open-Meteo Historical Forecast GFS collection.

Frozen config matches Phase 7B except BLH is NOT requested.
Period: 2021-04-01 .. 2025-12-31 hourly UTC for five V2 stations.

Does not merge into feature datasets, does not train, does not touch V1.
Supports rerunning individual stations/months without corrupting successful chunks.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "nwp_full")
CHUNK_DIR = os.path.join(OUT_DIR, "chunks")
IEM_META_PATH = os.path.join(HERE, "iem_station_metadata.json")

START_DATE = "2021-04-01"
END_DATE = "2025-12-31"
EXPECTED_ROWS_APPROX = 41640
EXPECTED_ROWS_EXACT = (
    pd.date_range(f"{START_DATE} 00:00", f"{END_DATE} 23:00", freq="h", tz="UTC")
)

MODEL = "gfs_global"
ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"
USER_AGENT = "SIH26072-V2-phase7c-nwp-full"

# Phase 7B minus boundary_layer_height (systematically null; must not be collected).
HOURLY_VARIABLES = [
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

SURFACE_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]

LOCATION_NAMES = {
    "VOTV": "Thiruvananthapuram",
    "VECC": "Kolkata",
    "VIDP": "Delhi",
    "VOCI": "Kochi",
    "VABB": "Mumbai",
}
INITIAL_STATIONS = ["VOTV", "VECC", "VIDP", "VOCI", "VABB"]
VOTV_OPENMETEO_REQUEST = {"latitude": 8.482, "longitude": 76.920}

TIMEOUT = 180
MAX_RETRIES = 6
SLEEP_BETWEEN_CHUNKS = 0.8


def load_catalog() -> dict:
    with open(IEM_META_PATH, encoding="utf-8") as handle:
        payload = json.load(handle)
    stations = payload["stations"]
    catalog = {}
    for sid in INITIAL_STATIONS:
        iem = stations[sid]
        if sid == "VOTV":
            lat, lon = VOTV_OPENMETEO_REQUEST["latitude"], VOTV_OPENMETEO_REQUEST["longitude"]
            src = "V1 Open-Meteo convention"
        else:
            lat, lon = iem["latitude"], iem["longitude"]
            src = "IEM IN__ASOS"
        catalog[sid] = {
            "station_id": sid,
            "name": LOCATION_NAMES[sid],
            "request_latitude": lat,
            "request_longitude": lon,
            "request_coordinate_source": src,
        }
    return catalog


def month_windows(start: str, end: str) -> list[tuple[str, str]]:
    months = pd.period_range(start=start[:7], end=end[:7], freq="M")
    windows = []
    for period in months:
        s = max(period.start_time.strftime("%Y-%m-%d"), start)
        e = min(period.end_time.strftime("%Y-%m-%d"), end)
        windows.append((s, e))
    return windows


def chunk_path(station_id: str, start: str, end: str) -> str:
    return os.path.join(CHUNK_DIR, station_id.lower(), f"{start}_{end}.csv")


def request_url(lat: float, lon: float, start: str, end: str) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "GMT",
        "models": MODEL,
        "wind_speed_unit": "kmh",
    }
    return ENDPOINT + "?" + urllib.parse.urlencode(params)


def fetch_json(url: str) -> dict:
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            wait = min(60.0, 2.0 ** attempt)
            print(f"  retry {attempt}/{MAX_RETRIES} after {exc}; sleep {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}") from last_err


def hourly_to_frame(payload: dict, station_id: str, request_lat: float, request_lon: float) -> pd.DataFrame:
    hourly = payload["hourly"]
    times = hourly["time"]
    df = pd.DataFrame({"valid_time_utc": pd.to_datetime(times, utc=True)})
    for var in HOURLY_VARIABLES:
        df[var] = hourly.get(var)
    df.insert(0, "station_id", station_id)
    df.insert(1, "request_latitude", request_lat)
    df.insert(2, "request_longitude", request_lon)
    df.insert(3, "served_latitude", payload.get("latitude"))
    df.insert(4, "served_longitude", payload.get("longitude"))
    df.insert(5, "elevation_m", payload.get("elevation"))
    df.insert(6, "model", MODEL)
    df.insert(7, "source", "open-meteo-historical-forecast")
    return df


def missing_pct(series: pd.Series) -> float:
    if len(series) == 0:
        return 100.0
    return float(series.isna().mean() * 100.0)


def inspect_station(df: pd.DataFrame) -> dict:
    stats = {}
    for var in HOURLY_VARIABLES:
        s = pd.to_numeric(df[var], errors="coerce")
        stats[var] = {
            "missing_pct": round(missing_pct(s), 4),
            "n_missing": int(s.isna().sum()),
            "n": int(len(s)),
            "min": None if s.dropna().empty else float(s.min()),
            "max": None if s.dropna().empty else float(s.max()),
            "mean": None if s.dropna().empty else float(s.mean()),
        }
    times = df["valid_time_utc"]
    expected = EXPECTED_ROWS_EXACT
    present = pd.DatetimeIndex(times).tz_convert("UTC")
    missing_hours = expected.difference(present)
    extra_hours = present.difference(expected)
    diffs = times.diff().dropna().dt.total_seconds() if len(times) > 1 else pd.Series(dtype=float)
    gap_hours = int(((diffs > 3600).sum())) if len(diffs) else 0
    return {
        "rows": int(len(df)),
        "expected_rows_exact": int(len(expected)),
        "expected_rows_approx_spec": EXPECTED_ROWS_APPROX,
        "duplicate_timestamps": int(times.duplicated().sum()),
        "duplicate_station_timestamp": int(df.duplicated(["station_id", "valid_time_utc"]).sum()),
        "min_time": None if times.empty else str(times.min()),
        "max_time": None if times.empty else str(times.max()),
        "monotonic_utc": bool(times.is_monotonic_increasing) if len(times) else True,
        "hourly_step_ok": bool(diffs.eq(3600).all()) if len(diffs) else True,
        "n_gaps_gt_1h": gap_hours,
        "n_missing_hours_vs_requested": int(len(missing_hours)),
        "n_extra_hours_vs_requested": int(len(extra_hours)),
        "missing_hour_examples": [str(t) for t in missing_hours[:20]],
        "has_blh_column": "boundary_layer_height" in df.columns,
        "has_weather_code": "weather_code" in df.columns,
        "variables": stats,
        "cape_missing_pct": stats["cape"]["missing_pct"],
        "cin_missing_pct": stats["convective_inhibition"]["missing_pct"],
        "li_missing_pct": stats["lifted_index"]["missing_pct"],
        "surface_missing_pct": {
            v: stats[v]["missing_pct"] for v in SURFACE_VARS
        },
    }


def fetch_chunk(
    sid: str,
    meta: dict,
    start: str,
    end: str,
    force: bool,
    request_log: list,
) -> pd.DataFrame | None:
    path = chunk_path(sid, start, end)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not force and os.path.getsize(path) > 50:
        try:
            df = pd.read_csv(path, parse_dates=["valid_time_utc"])
        except (pd.errors.EmptyDataError, ValueError):
            df = pd.DataFrame()
        if not df.empty and "valid_time_utc" in df.columns:
            if df["valid_time_utc"].dt.tz is None:
                df["valid_time_utc"] = df["valid_time_utc"].dt.tz_localize("UTC")
            else:
                df["valid_time_utc"] = df["valid_time_utc"].dt.tz_convert("UTC")
            request_log.append(
                {
                    "station_id": sid,
                    "start": start,
                    "end": end,
                    "status": "cache_hit",
                    "rows": int(len(df)),
                    "path": path,
                }
            )
            return df

    url = request_url(meta["request_latitude"], meta["request_longitude"], start, end)
    print(f"GET {sid} {start}..{end}")
    try:
        payload = fetch_json(url)
    except Exception as exc:
        request_log.append(
            {
                "station_id": sid,
                "start": start,
                "end": end,
                "status": "failed",
                "error": str(exc),
                "url": url,
            }
        )
        print(f"  FAILED {sid} {start}..{end}: {exc}")
        return None

    df = hourly_to_frame(payload, sid, meta["request_latitude"], meta["request_longitude"])
    df.to_csv(path, index=False)
    request_log.append(
        {
            "station_id": sid,
            "start": start,
            "end": end,
            "status": "ok",
            "rows": int(len(df)),
            "url": url,
            "generationtime_ms": payload.get("generationtime_ms"),
            "served_latitude": payload.get("latitude"),
            "served_longitude": payload.get("longitude"),
            "elevation_m": payload.get("elevation"),
            "utc_offset_seconds": payload.get("utc_offset_seconds"),
            "hourly_units": payload.get("hourly_units"),
        }
    )
    time.sleep(SLEEP_BETWEEN_CHUNKS)
    return df


def assemble_station(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and len(f)]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)
    df = df.sort_values("valid_time_utc")
    df = df.drop_duplicates(["station_id", "valid_time_utc"], keep="last")
    return df.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7C full GFS historical forecast collector")
    p.add_argument("--stations", nargs="*", default=INITIAL_STATIONS)
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=END_DATE)
    p.add_argument("--force", action="store_true", help="Refetch chunks even if cached")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CHUNK_DIR, exist_ok=True)
    catalog = load_catalog()
    stations = [s.upper() for s in args.stations]
    windows = month_windows(args.start, args.end)

    request_log: list = []
    per_station = {}
    combined = []
    failures = []

    for sid in stations:
        meta = catalog[sid]
        frames = []
        for start, end in windows:
            df = fetch_chunk(sid, meta, start, end, args.force, request_log)
            if df is None:
                failures.append({"station_id": sid, "start": start, "end": end})
            else:
                frames.append(df)
        assembled = assemble_station(frames)
        csv_path = os.path.join(OUT_DIR, f"{sid.lower()}_gfs.csv")
        if not assembled.empty:
            assembled.to_csv(csv_path, index=False)
            combined.append(assembled)
        insp = inspect_station(assembled) if not assembled.empty else {
            "rows": 0,
            "error": "no successful chunks",
        }
        insp["csv"] = os.path.relpath(csv_path, os.path.dirname(HERE)).replace("\\", "/")
        insp["name"] = meta["name"]
        insp["request_latitude"] = meta["request_latitude"]
        insp["request_longitude"] = meta["request_longitude"]
        if not assembled.empty:
            insp["served_latitude"] = float(assembled["served_latitude"].iloc[0])
            insp["served_longitude"] = float(assembled["served_longitude"].iloc[0])
            insp["elevation_m"] = float(assembled["elevation_m"].iloc[0])
        per_station[sid] = insp
        print(f"{sid} rows={insp.get('rows')} missing_hours={insp.get('n_missing_hours_vs_requested')}")

    pooled_path = os.path.join(OUT_DIR, "nwp_gfs_pooled.csv")
    if combined:
        pooled = pd.concat(combined, ignore_index=True)
        pooled = pooled.sort_values(["station_id", "valid_time_utc"]).reset_index(drop=True)
        pooled.to_csv(pooled_path, index=False)
    else:
        pooled = pd.DataFrame()

    forbidden_ok = all(
        not per_station.get(s, {}).get("has_blh_column")
        and not per_station.get(s, {}).get("has_weather_code")
        for s in stations
        if per_station.get(s, {}).get("rows", 0)
    )

    actual_min = None
    actual_max = None
    if not pooled.empty:
        actual_min = str(pooled["valid_time_utc"].min())
        actual_max = str(pooled["valid_time_utc"].max())

    summary = {
        "phase": "7C",
        "pilot_only": False,
        "source": "Open-Meteo Historical Forecast (archived NWP forecast valid-time series)",
        "endpoint": ENDPOINT,
        "model": MODEL,
        "timezone": "GMT (stored as UTC)",
        "wind_speed_unit": "kmh",
        "requested_period": {
            "start": START_DATE + " 00:00 UTC",
            "end": END_DATE + " 23:00 UTC",
            "inclusive_end_date": True,
        },
        "actual_returned_period": {"min": actual_min, "max": actual_max},
        "hourly_variables": HOURLY_VARIABLES,
        "not_collected": [
            "boundary_layer_height",
            "weather_code",
            "radar",
            "lightning",
            "satellite",
        ],
        "coordinates": {
            s: {
                "name": catalog[s]["name"],
                "request_latitude": catalog[s]["request_latitude"],
                "request_longitude": catalog[s]["request_longitude"],
                "served_latitude": per_station.get(s, {}).get("served_latitude"),
                "served_longitude": per_station.get(s, {}).get("served_longitude"),
                "elevation_m": per_station.get(s, {}).get("elevation_m"),
            }
            for s in stations
        },
        "expected_rows_per_station_exact": int(len(EXPECTED_ROWS_EXACT)),
        "expected_rows_per_station_approx_spec": EXPECTED_ROWS_APPROX,
        "per_station": per_station,
        "pooled_rows": int(len(pooled)),
        "api_failures": failures,
        "request_log": request_log,
        "collection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "exact_configuration": {
            "endpoint": ENDPOINT,
            "models": MODEL,
            "timezone": "GMT",
            "wind_speed_unit": "kmh",
            "hourly": HOURLY_VARIABLES,
            "chunking": "calendar month",
            "user_agent": USER_AGENT,
            "max_retries": MAX_RETRIES,
            "fill_missing": False,
            "mix_reanalysis": False,
            "silent_model_fallback": False,
        },
        "validation": {
            "five_stations": stations == INITIAL_STATIONS and all(
                per_station.get(s, {}).get("rows", 0) > 0 for s in INITIAL_STATIONS
            ),
            "no_blh": forbidden_ok,
            "no_weather_code_label": forbidden_ok,
            "no_duplicate_station_timestamp": all(
                per_station.get(s, {}).get("duplicate_station_timestamp", 0) == 0
                for s in stations
                if per_station.get(s, {}).get("rows", 0)
            ),
        },
        "phase7b_match": {
            "endpoint": ENDPOINT,
            "model": MODEL,
            "timezone": "GMT",
            "wind_speed_unit": "kmh",
            "variables_minus_blh": True,
        },
        "do_not": [
            "merge into ML feature dataset",
            "train models",
            "modify dashboard",
            "modify Phase 4/5/6 artifacts",
        ],
    }
    meta_path = os.path.join(OUT_DIR, "nwp_full_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("wrote", OUT_DIR)
    print("failures", len(failures))


if __name__ == "__main__":
    main()
