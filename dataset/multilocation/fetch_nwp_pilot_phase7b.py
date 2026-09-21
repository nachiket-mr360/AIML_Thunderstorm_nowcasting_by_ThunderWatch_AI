"""Phase 7B: Open-Meteo Historical Forecast NWP PILOT only.

Downloads 2021-04-01 .. 2021-04-07 hourly fields for five V2 sites.
Does not write into raw_openmeteo/, features/, or synchronized tables.
Does not train, merge, or touch V1.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "nwp_pilot")
IEM_META_PATH = os.path.join(HERE, "iem_station_metadata.json")
PHASE2A_ROOT = os.path.join(HERE, "raw_openmeteo")

START_DATE = "2021-04-01"
END_DATE = "2021-04-07"

# Frozen model (Phase 7 audit): do not use silent best_match.
MODEL = "gfs_global"
ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"

HOURLY_VARIABLES = [
    "cape",
    "convective_inhibition",
    "lifted_index",
    "boundary_layer_height",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]

SURFACE_OVERLAP = [
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

TIMEOUT = 120


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


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "SIH26072-V2-phase7b-pilot"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def request_url(lat: float, lon: float) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "GMT",
        "models": MODEL,
        "wind_speed_unit": "kmh",
    }
    return ENDPOINT + "?" + urllib.parse.urlencode(params)


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
    return {
        "rows": int(len(df)),
        "duplicate_timestamps": int(times.duplicated().sum()),
        "min_time": str(times.min()),
        "max_time": str(times.max()),
        "monotonic_utc": bool(times.is_monotonic_increasing),
        "hourly_step_ok": bool(
            times.diff().dropna().dt.total_seconds().eq(3600).all() if len(times) > 1 else True
        ),
        "variables": stats,
        "cape_missing_pct": stats["cape"]["missing_pct"],
        "cin_missing_pct": stats["convective_inhibition"]["missing_pct"],
        "li_missing_pct": stats["lifted_index"]["missing_pct"],
        "blh_missing_pct": stats["boundary_layer_height"]["missing_pct"],
        "other_missing_pct": round(
            float(
                pd.concat([pd.to_numeric(df[v], errors="coerce") for v in SURFACE_OVERLAP], axis=1)
                .isna()
                .mean()
                .mean()
                * 100.0
            ),
            4,
        ),
    }


def compare_phase2a(nwp: pd.DataFrame, station_id: str) -> dict:
    path = os.path.join(
        PHASE2A_ROOT, station_id.lower(), f"{station_id.lower()}_openmeteo_2021.csv"
    )
    p2 = pd.read_csv(path)
    time_col = "date" if "date" in p2.columns else "time"
    p2[time_col] = pd.to_datetime(p2[time_col], utc=True)
    start = pd.Timestamp(START_DATE, tz="UTC")
    end = pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(hours=23)
    p2 = p2[(p2[time_col] >= start) & (p2[time_col] <= end)].copy()
    merged = pd.merge(
        nwp[["valid_time_utc"] + SURFACE_OVERLAP],
        p2[[time_col] + [c for c in SURFACE_OVERLAP if c in p2.columns]],
        left_on="valid_time_utc",
        right_on=time_col,
        how="inner",
        suffixes=("_nwp", "_p2a"),
    )
    out = {"phase2a_rows_in_window": int(len(p2)), "overlap_rows": int(len(merged))}
    for var in SURFACE_OVERLAP:
        a = pd.to_numeric(merged[f"{var}_nwp"], errors="coerce")
        b = pd.to_numeric(merged[f"{var}_p2a"], errors="coerce")
        diff = (a - b).dropna()
        out[var] = {
            "mean_abs_diff": None if diff.empty else float(diff.abs().mean()),
            "max_abs_diff": None if diff.empty else float(diff.abs().max()),
            "corr": None if len(diff) < 3 else float(a.corr(b)),
        }
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    catalog = load_catalog()
    expected = 7 * 24
    combined = []
    per_station = {}
    comparisons = {}
    request_log = []

    for sid, meta in catalog.items():
        url = request_url(meta["request_latitude"], meta["request_longitude"])
        print(f"GET {sid} {url}")
        payload = fetch_json(url)
        df = hourly_to_frame(
            payload, sid, meta["request_latitude"], meta["request_longitude"]
        )
        csv_path = os.path.join(OUT_DIR, f"{sid.lower()}_nwp_pilot_20210401_20210407.csv")
        df.to_csv(csv_path, index=False)
        insp = inspect_station(df)
        insp["expected_rows"] = expected
        insp["csv"] = os.path.relpath(csv_path, os.path.dirname(HERE))
        insp["served_latitude"] = payload.get("latitude")
        insp["served_longitude"] = payload.get("longitude")
        insp["elevation_m"] = payload.get("elevation")
        insp["utc_offset_seconds"] = payload.get("utc_offset_seconds")
        insp["hourly_units"] = payload.get("hourly_units")
        per_station[sid] = insp
        request_log.append(
            {
                "station_id": sid,
                "url": url,
                "http": 200,
                "generationtime_ms": payload.get("generationtime_ms"),
            }
        )
        comparisons[sid] = compare_phase2a(df, sid)
        combined.append(df)
        time.sleep(1.0)

    all_df = pd.concat(combined, ignore_index=True)
    all_path = os.path.join(OUT_DIR, "nwp_pilot_all_locations_20210401_20210407.csv")
    all_df.to_csv(all_path, index=False)

    empty_vars = []
    for var in HOURLY_VARIABLES:
        if all(
            per_station[s]["variables"][var]["missing_pct"] >= 99.9
            for s in INITIAL_STATIONS
        ):
            empty_vars.append(var)

    summary = {
        "phase": "7B",
        "pilot_only": True,
        "period": {"start": START_DATE, "end": END_DATE, "inclusive_end_date": True},
        "endpoint": ENDPOINT,
        "model": MODEL,
        "timezone": "GMT (stored as UTC)",
        "product": "historical forecast (archived NWP forecast valid-time series, not reanalysis, not observations)",
        "hourly_variables": HOURLY_VARIABLES,
        "expected_rows_per_location": expected,
        "all_five_locations_returned_data": all(per_station[s]["rows"] > 0 for s in INITIAL_STATIONS),
        "systematically_empty_variables": empty_vars,
        "per_station": per_station,
        "phase2a_sanity_compare": comparisons,
        "request_log": request_log,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = os.path.join(OUT_DIR, "nwp_pilot_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({k: summary[k] for k in ("all_five_locations_returned_data", "systematically_empty_variables")}, indent=2))
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
