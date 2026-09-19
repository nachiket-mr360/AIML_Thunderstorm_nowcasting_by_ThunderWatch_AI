"""Phase 3 (step 1): fetch Open-Meteo historical hourly data for the VOTV site.

This performs the same acquisition the Phase 1 notebook (dataset/data_collection.ipynb)
performed, against the same Open-Meteo Historical Weather API with the same eight
hourly variables -- changing only the coordinates (VOTV aerodrome reference point
8.482 N, 76.920 E) and the period (2014-01-01 .. 2025-12-31, the stable modelling
window identified in Phase 2).

The request is split into one chunk per calendar year. Open-Meteo is a deterministic
archive, so chunking does not change the values returned; it makes the download
resumable and lets a single slow year be retried on its own (a single 12-year request
was observed to stall against this endpoint).

Design constraints honoured here:
  * Writes NEW files only, under dataset/raw_openmeteo/. It never touches
    dataset/weather_data.csv or dataset/weather_data_with_code.csv.
  * Uses no persistent HTTP cache, so the existing dataset/.cache.sqlite is not
    written to.
  * No data is invented: the only values written are those returned by the API.

Usage:
    python fetch_votv_openmeteo_phase3.py            # resume-friendly
    python fetch_votv_openmeteo_phase3.py --force    # re-download every year
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

STATION_ID = "VOTV"
#: VOTV aerodrome reference point supplied for Phase 3 (Thiruvananthapuram).
LATITUDE = 8.482
LONGITUDE = 76.920

START_YEAR = 2014
END_YEAR = 2025
START_DATE = f"{START_YEAR}-01-01"
END_DATE = f"{END_YEAR}-12-31"

#: Same variables, same order, as the Phase 1 acquisition.
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "weather_code",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT_SECONDS = 120
MAX_ATTEMPTS_PER_YEAR = 4
SLEEP_BETWEEN_CHUNKS = 1.5

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw_openmeteo")
OUT_CSV = os.path.join(RAW_DIR, "votv_openmeteo_hourly_2014_2025.csv")
OUT_META = os.path.join(RAW_DIR, "votv_openmeteo_hourly_2014_2025_metadata.json")
EXPECTED_COLUMNS = ["date"] + HOURLY_VARIABLES


def expected_hours(year: int) -> int:
    leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    return (366 if leap else 365) * 24


def chunk_path(year: int) -> str:
    return os.path.join(RAW_DIR, f"votv_openmeteo_{year}.csv")


def request_params(start_date: str, end_date: str) -> dict:
    return {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "GMT",
    }


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_year_via_sdk(year: int) -> tuple[pd.DataFrame, dict]:
    """Fetch one year using the official openmeteo_requests SDK (as Phase 1 did)."""
    import requests
    import openmeteo_requests
    from retry_requests import retry

    session = retry(requests.Session(), retries=3, backoff_factor=0.3)
    client = openmeteo_requests.Client(session=session)
    responses = client.weather_api(ARCHIVE_URL, params=request_params(f"{year}-01-01", f"{year}-12-31"))
    response = responses[0]
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

    return frame, {
        "returned_latitude": response.Latitude(),
        "returned_longitude": response.Longitude(),
        "returned_elevation_m": response.Elevation(),
        "returned_timezone": str(response.Timezone()),
        "returned_utc_offset_seconds": response.UtcOffsetSeconds(),
    }


def fetch_year_via_rest(year: int) -> tuple[pd.DataFrame, dict]:
    """Fallback: the identical request made directly against the REST endpoint."""
    url = f"{ARCHIVE_URL}?{urllib.parse.urlencode(request_params(f'{year}-01-01', f'{year}-12-31'))}"
    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as handle:
        payload = json.load(handle)

    hourly = payload["hourly"]
    frame = pd.DataFrame({name: hourly[name] for name in HOURLY_VARIABLES})
    frame.insert(0, "date", pd.to_datetime(hourly["time"], utc=True))

    return frame, {
        "returned_latitude": payload.get("latitude"),
        "returned_longitude": payload.get("longitude"),
        "returned_elevation_m": payload.get("elevation"),
        "returned_timezone": payload.get("timezone"),
        "returned_utc_offset_seconds": payload.get("utc_offset_seconds"),
    }


def validate_chunk(frame: pd.DataFrame, year: int) -> None:
    """Raise ValueError if a downloaded year is not exactly what was asked for."""
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
    non_numeric = [c for c in HOURLY_VARIABLES if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError(f"non-numeric columns {non_numeric}")


def load_valid_chunk(year: int) -> pd.DataFrame | None:
    """Return the cached chunk for a year if it is present and self-consistent."""
    path = chunk_path(year)
    if not os.path.exists(path):
        return None
    try:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        validate_chunk(frame, year)
        return frame
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch VOTV Open-Meteo history.")
    parser.add_argument("--force", action="store_true", help="re-download every year")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    frames: list[pd.DataFrame] = []
    provenance: dict[str, dict] = {}
    served: dict = {}
    errors: list[str] = []

    for year in range(START_YEAR, END_YEAR + 1):
        frame = None if args.force else load_valid_chunk(year)
        if frame is not None:
            print(f"[{year}] reusing {os.path.basename(chunk_path(year))} rows={len(frame)}")
        else:
            for attempt in range(1, MAX_ATTEMPTS_PER_YEAR + 1):
                for fetcher in (fetch_year_via_sdk, fetch_year_via_rest):
                    try:
                        frame, meta = fetcher(year)
                        validate_chunk(frame, year)
                        served = meta or served
                        break
                    except Exception as exc:  # noqa: BLE001 - retry, then fall back
                        errors.append(
                            f"{year} attempt {attempt} {fetcher.__name__}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        frame = None
                if frame is not None:
                    break
                time.sleep(2.0 * attempt)
            if frame is None:
                print(f"[{year}] FAILED after {MAX_ATTEMPTS_PER_YEAR} attempts")
                for line in errors[-4:]:
                    print("      ", line)
                return 1
            frame.to_csv(chunk_path(year), index=False)
            print(f"[{year}] downloaded rows={len(frame)}")

        provenance[str(year)] = {
            "rows": int(len(frame)),
            "first_timestamp_utc": frame["date"].iloc[0].isoformat(),
            "last_timestamp_utc": frame["date"].iloc[-1].isoformat(),
            "chunk_sha256": sha256_of(chunk_path(year)),
        }
        frames.append(frame)
        time.sleep(SLEEP_BETWEEN_CHUNKS)

    combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)

    problems: list[str] = []
    if list(combined.columns) != EXPECTED_COLUMNS:
        problems.append(f"unexpected combined columns: {list(combined.columns)}")
    total_expected = sum(expected_hours(y) for y in range(START_YEAR, END_YEAR + 1))
    if len(combined) != total_expected:
        problems.append(f"expected {total_expected} combined hours, got {len(combined)}")
    if combined["date"].duplicated().any():
        problems.append("combined frame has duplicate timestamps")
    if not combined["date"].is_monotonic_increasing:
        problems.append("combined frame is not in ascending order")
    if str(combined["date"].dt.tz) != "UTC":
        problems.append("combined timestamps are not UTC")
    missing = {c: int(combined[c].isna().sum()) for c in HOURLY_VARIABLES}

    combined.to_csv(OUT_CSV, index=False)

    metadata = {
        "artifact": os.path.basename(OUT_CSV),
        "phase": "Phase 3 -- data preprocessing and synchronization (raw feature acquisition)",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "station_id": STATION_ID,
        "source": {
            "name": "Open-Meteo Historical Weather API (archive)",
            "endpoint": ARCHIVE_URL,
            "access_method": "openmeteo_requests SDK, falling back to a direct REST request",
            "note": "Same endpoint, variables and timezone setting as the Phase 1 notebook "
                    "dataset/data_collection.ipynb, which produced "
                    "dataset/weather_data_with_code.csv. The request is split per calendar "
                    "year for resumability; Open-Meteo is a deterministic archive, so "
                    "chunking does not alter the returned values.",
        },
        "request": {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly_variables": HOURLY_VARIABLES,
            "timezone": "GMT",
            "example_request_url": f"{ARCHIVE_URL}?"
            + urllib.parse.urlencode(request_params(START_DATE, f"{START_YEAR}-12-31")),
        },
        "response": served,
        "coordinates": {
            "requested_latitude": LATITUDE,
            "requested_longitude": LONGITUDE,
            "served_latitude": served.get("returned_latitude"),
            "served_longitude": served.get("returned_longitude"),
            "served_elevation_m": served.get("returned_elevation_m"),
            "note": "Open-Meteo serves the archive grid cell containing the requested point. "
                    "The 'served_*' values are the API-reported cell for this request and are "
                    "the coordinates the delivered values belong to. No spatial resolution is "
                    "claimed beyond what the API reports for this request.",
        },
        "coverage": {
            "years": [START_YEAR, END_YEAR],
            "expected_hours": total_expected,
            "actual_hours": int(len(combined)),
            "first_timestamp_utc": combined["date"].iloc[0].isoformat(),
            "last_timestamp_utc": combined["date"].iloc[-1].isoformat(),
        },
        "quality": {
            "duplicate_timestamps": int(combined["date"].duplicated().sum()),
            "timestamps_sorted_ascending": bool(combined["date"].is_monotonic_increasing),
            "missing_values_per_column": missing,
            "total_missing_values": int(sum(missing.values())),
        },
        "per_year": provenance,
        "fetch_attempt_errors": errors[-40:],
        "checksums": {"csv_sha256": sha256_of(OUT_CSV)},
    }
    with open(OUT_META, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print(f"requested point: {LATITUDE}, {LONGITUDE}")
    print(f"served cell: lat={served.get('returned_latitude')} lon={served.get('returned_longitude')} "
          f"elev={served.get('returned_elevation_m')} tz={served.get('returned_timezone')}")
    print(f"combined rows: {len(combined)} (expected {total_expected})")
    print(f"range: {combined['date'].iloc[0]} -> {combined['date'].iloc[-1]}")
    print(f"missing values: {missing} (total {sum(missing.values())})")
    print(f"problems: {problems if problems else 'NONE'}")
    print("wrote:", OUT_CSV)
    print("wrote:", OUT_META)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
