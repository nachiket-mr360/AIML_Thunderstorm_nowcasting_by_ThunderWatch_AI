"""Phase 14B: historical multi-station spatial replay.

Discrete station markers only. Does not interpolate Model B probabilities,
does not form storm cells, and does not retrain frozen models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
METADATA_PATH = (
    BASE_DIR
    / "dataset"
    / "multilocation"
    / "features"
    / "multilocation_features_2014_2025_metadata.json"
)
REPLAY_PATH = BASE_DIR / "outputs" / "v2_replay" / "phase15b" / "replay_predictions.csv"
SATELLITE_PATH = (
    BASE_DIR
    / "dataset"
    / "satellite"
    / "insat3dr_cmk_features"
    / "cmk_hourly_aligned_features.csv"
)

EXPECTED_STATIONS = ("VOTV", "VECC", "VIDP", "VOCI", "VABB")
THRESHOLD = 0.065
MODE = "HISTORICAL_REPLAY"

TARGET_COLUMNS = (
    "target_1h",
    "target_2h",
    "target_3h",
    "target_1h_observed",
    "target_2h_observed",
    "target_3h_observed",
    "thunderstorm_target",
    "thunderstorm",
)

SAT_FIELDS = (
    "sat_cloud_mask_at_station",
    "sat_cloudy_fraction_25km",
    "sat_clear_fraction_25km",
    "sat_valid_pixel_count_25km",
    "sat_observation_time_utc",
    "sat_age_minutes",
    "source_filename",
)


def parse_timestamp_utc(value: Any) -> pd.Timestamp | None:
    """Parse an explicit UTC timestamp. Naive values are treated as UTC."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _iso_z(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_int(value: Any) -> int | None:
    f = _json_float(value)
    if f is None:
        return None
    return int(f)


def _alert_from_prob(p: float | None) -> bool | None:
    if p is None:
        return None
    return bool(p >= THRESHOLD)


class HistoricalSpatialReplay:
    """Join frozen Phase 15B replay rows to IEM station coordinates."""

    def __init__(
        self,
        *,
        metadata_path: Path | None = None,
        replay_path: Path | None = None,
        satellite_path: Path | None = None,
    ) -> None:
        self.metadata_path = Path(metadata_path) if metadata_path else METADATA_PATH
        self.replay_path = Path(replay_path) if replay_path else REPLAY_PATH
        self.satellite_path = Path(satellite_path) if satellite_path else SATELLITE_PATH
        self._coords: dict[str, dict[str, float]] | None = None
        self._replay: pd.DataFrame | None = None
        self._sat: pd.DataFrame | None = None
        self._all_five: list[str] | None = None

    def load_station_coordinates(self) -> dict[str, dict[str, float]]:
        if self._coords is None:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            raw = payload["station_coordinates"]
            coords: dict[str, dict[str, float]] = {}
            for sid in EXPECTED_STATIONS:
                row = raw[sid]
                coords[sid] = {
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "elevation_m": float(row["elevation_m"]),
                }
            self._coords = coords
        return self._coords

    def _load_replay(self) -> pd.DataFrame:
        if self._replay is None:
            usecols = [
                "station_id",
                "prediction_timestamp_utc",
                "data_status",
                "lead_1h_probability",
                "lead_2h_probability",
                "lead_3h_probability",
                "lead_1h_alert",
                "lead_2h_alert",
                "lead_3h_alert",
            ]
            df = pd.read_csv(self.replay_path, usecols=usecols, low_memory=False)
            leaked = [c for c in TARGET_COLUMNS if c in df.columns]
            if leaked:
                df = df.drop(columns=leaked)
            df["station_id"] = df["station_id"].astype(str).str.strip().str.upper()
            df["prediction_timestamp_utc"] = pd.to_datetime(
                df["prediction_timestamp_utc"], utc=True, errors="coerce"
            )
            self._replay = df
        return self._replay

    def _load_satellite(self) -> pd.DataFrame:
        if self._sat is None:
            if not self.satellite_path.is_file():
                self._sat = pd.DataFrame()
                return self._sat
            df = pd.read_csv(self.satellite_path, low_memory=False)
            station_col = "station_id" if "station_id" in df.columns else "station"
            df = df.rename(columns={station_col: "station_id"})
            df["station_id"] = df["station_id"].astype(str).str.strip().str.upper()
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
            self._sat = df
        return self._sat

    def list_available_spatial_timestamps(self) -> list[str]:
        df = self._load_replay()
        complete = df[df["data_status"].astype(str) == "COMPLETE"]
        counts = complete.groupby("prediction_timestamp_utc")["station_id"].nunique()
        stamps = counts[counts == len(EXPECTED_STATIONS)].index
        stamps = pd.DatetimeIndex(stamps).sort_values()
        out = [_iso_z(pd.Timestamp(t)) for t in stamps]
        self._all_five = out
        return out

    def _unavailable_station(self, station_id: str, coords: dict[str, dict[str, float]]) -> dict[str, Any]:
        c = coords[station_id]
        return {
            "station_id": station_id,
            "latitude": c["latitude"],
            "longitude": c["longitude"],
            "lead_1h_probability": None,
            "lead_2h_probability": None,
            "lead_3h_probability": None,
            "alert_1h": None,
            "alert_2h": None,
            "alert_3h": None,
            "data_status": "UNAVAILABLE",
        }

    def _station_from_row(self, row: pd.Series, coords: dict[str, dict[str, float]]) -> dict[str, Any]:
        sid = str(row["station_id"]).upper()
        status = str(row["data_status"])
        if status != "COMPLETE":
            return self._unavailable_station(sid, coords)
        p1 = _json_float(row["lead_1h_probability"])
        p2 = _json_float(row["lead_2h_probability"])
        p3 = _json_float(row["lead_3h_probability"])
        c = coords[sid]
        return {
            "station_id": sid,
            "latitude": c["latitude"],
            "longitude": c["longitude"],
            "lead_1h_probability": p1,
            "lead_2h_probability": p2,
            "lead_3h_probability": p3,
            "alert_1h": _alert_from_prob(p1),
            "alert_2h": _alert_from_prob(p2),
            "alert_3h": _alert_from_prob(p3),
            "data_status": "COMPLETE",
        }

    def _satellite_context(self, ts: pd.Timestamp) -> dict[str, Any]:
        sat = self._load_satellite()
        if sat.empty:
            return {}
        mask = sat["timestamp_utc"] == ts
        if "sat_observation_available" in sat.columns:
            mask = mask & (pd.to_numeric(sat["sat_observation_available"], errors="coerce") == 1)
        rows = sat.loc[mask]
        if rows.empty:
            return {}
        by_station: dict[str, Any] = {}
        for _, row in rows.iterrows():
            sid = str(row["station_id"]).upper()
            if sid not in EXPECTED_STATIONS:
                continue
            item: dict[str, Any] = {"station_id": sid}
            item["sat_cloud_mask_at_station"] = _json_float(row.get("sat_cloud_mask_at_station"))
            item["sat_cloudy_fraction_25km"] = _json_float(row.get("sat_cloudy_fraction_25km"))
            item["sat_clear_fraction_25km"] = _json_float(row.get("sat_clear_fraction_25km"))
            item["sat_valid_pixel_count_25km"] = _json_int(row.get("sat_valid_pixel_count_25km"))
            obs = row.get("sat_observation_time_utc")
            item["sat_observation_time_utc"] = None if pd.isna(obs) else str(obs)
            item["sat_age_minutes"] = _json_float(row.get("sat_age_minutes"))
            src = row.get("source_filename")
            item["source_filename"] = None if pd.isna(src) else str(src)
            item["note"] = (
                "INSAT-3DR CMK evidence/context only. Not a Model B predictor. "
                "Missing satellite is not treated as clear."
            )
            by_station[sid] = item
        return by_station

    def get_multi_station_snapshot(self, timestamp_utc: Any) -> dict[str, Any]:
        parsed = parse_timestamp_utc(timestamp_utc)
        if parsed is None:
            return {
                "ok": False,
                "error": {
                    "code": "INVALID_TIMESTAMP",
                    "message": "timestamp_utc is required and must be a valid UTC datetime",
                },
                "timestamp_utc": None,
                "mode": MODE,
                "stations": [],
                "satellite_context": {},
                "radar_context": None,
                "lightning_context": None,
            }

        coords = self.load_station_coordinates()
        df = self._load_replay()
        slice_df = df[df["prediction_timestamp_utc"] == parsed]
        by_id = {str(r["station_id"]).upper(): r for _, r in slice_df.iterrows()}

        stations = []
        for sid in EXPECTED_STATIONS:
            if sid in by_id:
                stations.append(self._station_from_row(by_id[sid], coords))
            else:
                stations.append(self._unavailable_station(sid, coords))

        return {
            "ok": True,
            "timestamp_utc": _iso_z(parsed),
            "mode": MODE,
            "stations": stations,
            "satellite_context": self._satellite_context(parsed),
            "radar_context": None,
            "lightning_context": None,
            "disclaimer": (
                "This spatial layer represents discrete AI risk at observed station "
                "locations. It does not interpolate risk between stations and does "
                "not constitute a storm-cell forecast."
            ),
        }


_default: HistoricalSpatialReplay | None = None


def get_default_service() -> HistoricalSpatialReplay:
    global _default
    if _default is None:
        _default = HistoricalSpatialReplay()
    return _default
