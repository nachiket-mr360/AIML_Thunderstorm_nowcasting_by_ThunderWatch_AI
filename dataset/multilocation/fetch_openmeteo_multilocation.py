"""Phase 2A: reusable Open-Meteo historical predictor collection for V2 locations.

Writes only under dataset/multilocation/. Does not modify V1 files, Flask, or labels.

Predictor variables only. Open-Meteo weather_code is NOT requested and is not a target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_ROOT = os.path.join(HERE, "raw_openmeteo")
IEM_META_PATH = os.path.join(HERE, "iem_station_metadata.json")

START_YEAR = 2014
END_YEAR = 2025
START_DATE = f"{START_YEAR}-01-01"
END_DATE = f"{END_YEAR}-12-31"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT_SECONDS = 120
MAX_ATTEMPTS_PER_YEAR = 4
SLEEP_BETWEEN_CHUNKS = 1.5

# Display names used in Phase 1B. Coordinates come from IEM metadata, except
# VOTV Open-Meteo request which keeps the V1 modelling-window convention.
LOCATION_NAMES = {
    "VOTV": "Thiruvananthapuram",
    "VECC": "Kolkata",
    "VIDP": "Delhi",
    "VOCI": "Kochi",
    "VABB": "Mumbai",
}

INITIAL_STATIONS = ["VOTV", "VECC", "VIDP", "VOCI", "VABB"]

# V1 Open-Meteo request point (dataset/fetch_votv_openmeteo_phase3.py).
VOTV_OPENMETEO_REQUEST = {"latitude": 8.482, "longitude": 76.920}


def expected_hours(year: int) -> int:
    leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    return (366 if leap else 365) * 24


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_station_catalog() -> dict:
    with open(IEM_META_PATH, encoding="utf-8") as handle:
        payload = json.load(handle)
    stations = payload["stations"]
    catalog = {}
    for sid in INITIAL_STATIONS:
        iem = stations[sid]
        if sid == "VOTV":
            req_lat = VOTV_OPENMETEO_REQUEST["latitude"]
            req_lon = VOTV_OPENMETEO_REQUEST["longitude"]
            request_source = "V1 Open-Meteo convention (fetch_votv_openmeteo_phase3.py)"
        else:
            req_lat = iem["latitude"]
            req_lon = iem["longitude"]
            request_source = "IEM IN__ASOS station table"
        catalog[sid] = {
            "station_id": sid,
            "name": LOCATION_NAMES[sid],
            "iem": iem,
            "request_latitude": req_lat,
            "request_longitude": req_lon,
            "request_coordinate_source": request_source,
        }
    return catalog


def station_dirs(station_id: str) -> dict[str, str]:
    folder = os.path.join(RAW_ROOT, station_id.lower())
    return {
        "dir": folder,
        "combined": os.path.join(
            folder, f"{station_id.lower()}_openmeteo_hourly_{START_YEAR}_{END_YEAR}.csv"
        ),
        "meta": os.path.join(
            folder,
            f"{station_id.lower()}_openmeteo_hourly_{START_YEAR}_{END_YEAR}_metadata.json",
        ),
    }


def chunk_path(station_id: str, year: int) -> str:
    return os.path.join(
        station_dirs(station_id)["dir"],
        f"{station_id.lower()}_openmeteo_{year}.csv",
    )


def request_params(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "GMT",
    }


def attach_identity(frame: pd.DataFrame, station: dict) -> pd.DataFrame:
    out = frame.copy()
    out.insert(1, "station_id", station["station_id"])
    out.insert(2, "latitude", station["request_latitude"])
    out.insert(3, "longitude", station["request_longitude"])
    out.insert(4, "source", "open-meteo-archive")
    return out


EXPECTED_COLUMNS = [
    "date",
    "station_id",
    "latitude",
    "longitude",
    "source",
] + HOURLY_VARIABLES


def fetch_year_via_sdk(station: dict, year: int) -> tuple[pd.DataFrame, dict]:
    import openmeteo_requests
    import requests
    from retry_requests import retry

    session = retry(requests.Session(), retries=3, backoff_factor=0.3)
    client = openmeteo_requests.Client(session=session)
    params = request_params(
        station["request_latitude"],
        station["request_longitude"],
        f"{year}-01-01",
        f"{year}-12-31",
    )
    response = client.weather_api(ARCHIVE_URL, params=params)[0]
    hourly = response.Hourly()
    frame = pd.DataFrame(
        {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }
    )
    for index, name in enumerate(HOURLY_VARIABLES):
        frame[name] = hourly.Variables(index).ValuesAsNumpy()
    frame = attach_identity(frame, station)
    return frame, {
        "returned_latitude": response.Latitude(),
        "returned_longitude": response.Longitude(),
        "returned_elevation_m": response.Elevation(),
        "returned_timezone": str(response.Timezone()),
        "returned_utc_offset_seconds": response.UtcOffsetSeconds(),
        "access_method": "openmeteo_requests SDK",
    }


def fetch_year_via_rest(station: dict, year: int) -> tuple[pd.DataFrame, dict]:
    params = request_params(
        station["request_latitude"],
        station["request_longitude"],
        f"{year}-01-01",
        f"{year}-12-31",
    )
    url = f"{ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "SIH26072-V2-phase2a"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as handle:
        payload = json.load(handle)
    hourly = payload["hourly"]
    frame = pd.DataFrame({name: hourly[name] for name in HOURLY_VARIABLES})
    frame.insert(0, "date", pd.to_datetime(hourly["time"], utc=True))
    frame = attach_identity(frame, station)
    return frame, {
        "returned_latitude": payload.get("latitude"),
        "returned_longitude": payload.get("longitude"),
        "returned_elevation_m": payload.get("elevation"),
        "returned_timezone": payload.get("timezone"),
        "returned_utc_offset_seconds": payload.get("utc_offset_seconds"),
        "access_method": "direct REST",
        "http_status": 200,
    }


def validate_chunk(frame: pd.DataFrame, station: dict, year: int) -> None:
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected columns {list(frame.columns)}")
    want = expected_hours(year)
    if len(frame) != want:
        raise ValueError(f"expected {want} hours for {year}, got {len(frame)}")
    if frame["date"].duplicated().any():
        raise ValueError("duplicate timestamps")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("timestamps not ascending")
    if (frame["date"].dt.year != year).any():
        raise ValueError("timestamps outside the requested year")
    if str(frame["date"].dt.tz) != "UTC":
        raise ValueError("timestamps are not UTC")
    if (frame["station_id"] != station["station_id"]).any():
        raise ValueError("station_id mismatch")
    if (frame["latitude"] != station["request_latitude"]).any():
        raise ValueError("latitude mismatch")
    if (frame["longitude"] != station["request_longitude"]).any():
        raise ValueError("longitude mismatch")
    if "weather_code" in frame.columns:
        raise ValueError("weather_code must not be present in predictor files")
    non_numeric = [c for c in HOURLY_VARIABLES if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError(f"non-numeric columns {non_numeric}")


def load_valid_chunk(station: dict, year: int) -> pd.DataFrame | None:
    path = chunk_path(station["station_id"], year)
    if not os.path.exists(path):
        return None
    try:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        validate_chunk(frame, station, year)
        return frame
    except Exception:
        return None


def quality_report(combined: pd.DataFrame, station: dict) -> dict:
    total_expected = sum(expected_hours(y) for y in range(START_YEAR, END_YEAR + 1))
    missing = {c: int(combined[c].isna().sum()) for c in HOURLY_VARIABLES}
    deltas = combined["date"].diff().dropna()
    hourly_ok = bool((deltas == pd.Timedelta(hours=1)).all())
    problems: list[str] = []
    if list(combined.columns) != EXPECTED_COLUMNS:
        problems.append(f"unexpected columns: {list(combined.columns)}")
    if len(combined) != total_expected:
        problems.append(f"expected {total_expected} hours, got {len(combined)}")
    if combined["date"].duplicated().any():
        problems.append("duplicate timestamps")
    if not combined["date"].is_monotonic_increasing:
        problems.append("not ascending")
    if not hourly_ok:
        problems.append("gaps or non-hourly steps")
    if str(combined["date"].dt.tz) != "UTC":
        problems.append("not UTC")
    if combined["station_id"].nunique() != 1:
        problems.append("mixed station_id")
    if combined["latitude"].nunique() != 1 or combined["longitude"].nunique() != 1:
        problems.append("mixed coordinates")
    return {
        "expected_hours": total_expected,
        "actual_hours": int(len(combined)),
        "first_timestamp_utc": combined["date"].iloc[0].isoformat(),
        "last_timestamp_utc": combined["date"].iloc[-1].isoformat(),
        "duplicate_timestamps": int(combined["date"].duplicated().sum()),
        "hourly_contiguous": hourly_ok,
        "timestamps_sorted_ascending": bool(combined["date"].is_monotonic_increasing),
        "missing_values_per_column": missing,
        "total_missing_values": int(sum(missing.values())),
        "variables_present": HOURLY_VARIABLES,
        "weather_code_present": False,
        "problems": problems,
    }


def fetch_station(station: dict, force: bool) -> dict:
    sid = station["station_id"]
    paths = station_dirs(sid)
    os.makedirs(paths["dir"], exist_ok=True)
    frames: list[pd.DataFrame] = []
    provenance: dict[str, dict] = {}
    served: dict = {}
    errors: list[str] = []

    for year in range(START_YEAR, END_YEAR + 1):
        frame = None if force else load_valid_chunk(station, year)
        if frame is not None:
            print(f"[{sid} {year}] reusing {os.path.basename(chunk_path(sid, year))} rows={len(frame)}")
        else:
            for attempt in range(1, MAX_ATTEMPTS_PER_YEAR + 1):
                for fetcher in (fetch_year_via_sdk, fetch_year_via_rest):
                    try:
                        frame, meta = fetcher(station, year)
                        validate_chunk(frame, station, year)
                        served = meta or served
                        break
                    except Exception as exc:  # noqa: BLE001
                        errors.append(
                            f"{sid} {year} attempt {attempt} {fetcher.__name__}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        frame = None
                if frame is not None:
                    break
                time.sleep(2.0 * attempt)
            if frame is None:
                print(f"[{sid} {year}] FAILED after {MAX_ATTEMPTS_PER_YEAR} attempts")
                for line in errors[-6:]:
                    print("      ", line)
                raise RuntimeError(f"Open-Meteo request failed for {sid} {year}")
            frame.to_csv(chunk_path(sid, year), index=False)
            print(f"[{sid} {year}] downloaded rows={len(frame)}")

        provenance[str(year)] = {
            "rows": int(len(frame)),
            "first_timestamp_utc": frame["date"].iloc[0].isoformat(),
            "last_timestamp_utc": frame["date"].iloc[-1].isoformat(),
            "chunk_sha256": sha256_of(chunk_path(sid, year)),
        }
        frames.append(frame)
        time.sleep(SLEEP_BETWEEN_CHUNKS)

    combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    quality = quality_report(combined, station)
    combined.to_csv(paths["combined"], index=False)

    metadata = {
        "artifact": os.path.basename(paths["combined"]),
        "phase": "Phase 2A — multi-location atmospheric predictor collection",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "station_id": sid,
        "location_name": station["name"],
        "source": {
            "name": "Open-Meteo Historical Weather API (archive)",
            "endpoint": ARCHIVE_URL,
            "access_method": "openmeteo_requests SDK, falling back to a direct REST request",
            "note": "Predictor collection only. weather_code is not requested and is not a thunderstorm target.",
        },
        "request": {
            "latitude": station["request_latitude"],
            "longitude": station["request_longitude"],
            "coordinate_source": station["request_coordinate_source"],
            "iem_station_point": {
                "latitude": station["iem"]["latitude"],
                "longitude": station["iem"]["longitude"],
                "elevation_m": station["iem"]["elevation_m"],
                "iem_name": station["iem"]["iem_name"],
            },
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly_variables": HOURLY_VARIABLES,
            "timezone": "GMT",
            "example_request_url": f"{ARCHIVE_URL}?"
            + urllib.parse.urlencode(
                request_params(
                    station["request_latitude"],
                    station["request_longitude"],
                    START_DATE,
                    f"{START_YEAR}-12-31",
                )
            ),
        },
        "response": served,
        "coordinates": {
            "requested_latitude": station["request_latitude"],
            "requested_longitude": station["request_longitude"],
            "served_latitude": served.get("returned_latitude"),
            "served_longitude": served.get("returned_longitude"),
            "served_elevation_m": served.get("returned_elevation_m"),
            "note": "Open-Meteo serves the archive grid cell containing the requested point.",
        },
        "quality": quality,
        "per_year": provenance,
        "fetch_attempt_errors": errors[-40:],
        "checksums": {"csv_sha256": sha256_of(paths["combined"])},
    }
    with open(paths["meta"], "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    if quality["problems"]:
        raise RuntimeError(f"{sid} quality problems: {quality['problems']}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo history for V2 Phase 2A stations.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stations", nargs="*", default=INITIAL_STATIONS)
    args = parser.parse_args()

    if not os.path.exists(IEM_META_PATH):
        print("missing", IEM_META_PATH)
        return 1

    catalog = load_station_catalog()
    summaries = []
    for sid in args.stations:
        sid = sid.upper()
        if sid not in catalog:
            print("unknown station", sid)
            return 1
        print(f"=== {sid} {catalog[sid]['name']} ===")
        meta = fetch_station(catalog[sid], force=args.force)
        summaries.append(
            {
                "station_id": sid,
                "rows": meta["quality"]["actual_hours"],
                "expected": meta["quality"]["expected_hours"],
                "missing": meta["quality"]["total_missing_values"],
                "served_lat": meta["coordinates"]["served_latitude"],
                "served_lon": meta["coordinates"]["served_longitude"],
            }
        )

    summary_path = os.path.join(HERE, "phase2a_collection_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "phase": "2A",
                "stations": summaries,
                "hourly_variables": HOURLY_VARIABLES,
                "period": {"start": START_DATE, "end": END_DATE},
            },
            handle,
            indent=2,
        )
    print("wrote", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
