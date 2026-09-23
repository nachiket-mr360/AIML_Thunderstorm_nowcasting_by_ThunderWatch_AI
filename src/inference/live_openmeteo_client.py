"""Open-Meteo Forecast + GFS HTTP client for Phase 15A live inference.

Does not invent values. Timeouts and HTTP failures raise LiveDataError.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from src.inference.live_failure_diagnostics import classify_live_exception

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GFS_URL = "https://api.open-meteo.com/v1/gfs"
USER_AGENT = "ThunderWatchAI-V2-phase15A-live"
DEFAULT_TIMEOUT_S = 30.0

ATMOSPHERIC_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "weather_code",
]

NWP_HOURLY = [
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


class LiveDataError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        category: str | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.category = category
        self.provider = provider


def http_get_json(url: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except TimeoutError as exc:
        raise LiveDataError(
            "LIVE_DATA_UNAVAILABLE",
            "timeout fetching live atmospheric/NWP data",
            category="TIMEOUT",
        ) from exc
    except urllib.error.HTTPError as exc:
        cat = classify_live_exception(exc)
        raise LiveDataError(
            "LIVE_DATA_UNAVAILABLE",
            "http error fetching live atmospheric/NWP data",
            category=cat,
        ) from exc
    except urllib.error.URLError as exc:
        cat = classify_live_exception(exc)
        raise LiveDataError(
            "LIVE_DATA_UNAVAILABLE",
            "network error fetching live atmospheric/NWP data",
            category=cat,
        ) from exc
    except OSError as exc:
        cat = classify_live_exception(exc)
        raise LiveDataError(
            "LIVE_DATA_UNAVAILABLE",
            "network error fetching live atmospheric/NWP data",
            category=cat,
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LiveDataError(
            "LIVE_DATA_UNAVAILABLE",
            "malformed JSON from Open-Meteo",
            category="INVALID_RESPONSE",
        ) from exc
    if not isinstance(payload, dict):
        raise LiveDataError(
            "LIVE_DATA_UNAVAILABLE",
            "malformed Open-Meteo payload",
            category="INVALID_RESPONSE",
        )
    return payload


def forecast_url(lat: float, lon: float, *, past_days: int = 3, forecast_days: int = 2) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(ATMOSPHERIC_HOURLY),
        "timezone": "GMT",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "wind_speed_unit": "kmh",
    }
    return FORECAST_URL + "?" + urllib.parse.urlencode(params)


def gfs_url(lat: float, lon: float, *, past_days: int = 1, forecast_days: int = 2) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(NWP_HOURLY),
        "timezone": "GMT",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "wind_speed_unit": "kmh",
        "models": "gfs_global",
    }
    return GFS_URL + "?" + urllib.parse.urlencode(params)


def fetch_live_payloads(
    lat: float,
    lon: float,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    get_json=http_get_json,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        atmo = get_json(forecast_url(lat, lon), timeout_s)
    except LiveDataError as exc:
        if exc.provider is None:
            exc.provider = "forecast"
        raise
    try:
        nwp = get_json(gfs_url(lat, lon), timeout_s)
    except LiveDataError as exc:
        if exc.provider is None:
            exc.provider = "gfs"
        raise
    return atmo, nwp
