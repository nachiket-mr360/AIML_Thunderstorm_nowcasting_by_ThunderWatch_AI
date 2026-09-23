"""Application entry point for frozen Model B multi-lead inference.

Historical replay lookup over the NWP-overlap table, then FinalMultiLeadEngine.
Does not retrain or alter scientific scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.inference.final_multi_lead_engine import FinalMultiLeadEngine

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    BASE_DIR
    / "dataset"
    / "multilocation"
    / "features_nwp"
    / "multilocation_features_nwp_overlap_2021_2025.csv"
)
PHASE12E_REPLAY = BASE_DIR / "outputs" / "multimodal" / "phase12e_case_replay.csv"

TARGET_COLUMNS = (
    "target_1h",
    "target_2h",
    "target_3h",
    "thunderstorm_target",
    "thunderstorm",
)


class MultiLeadInferenceService:
    """station_id + timestamp_utc → frozen 83-vector → FinalMultiLeadEngine."""

    def __init__(
        self,
        *,
        engine: FinalMultiLeadEngine | None = None,
        dataset_path: Path | None = None,
        satellite_replay_path: Path | None = None,
    ) -> None:
        self.engine = engine if engine is not None else FinalMultiLeadEngine()
        self.dataset_path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET
        self.satellite_replay_path = (
            Path(satellite_replay_path) if satellite_replay_path is not None else PHASE12E_REPLAY
        )
        self._features: pd.DataFrame | None = None
        self._sat: pd.DataFrame | None = None

    def _load_features(self) -> pd.DataFrame:
        if self._features is None:
            names = self.engine.feature_names
            usecols = ["station_id", "timestamp_utc", *names]
            df = pd.read_csv(self.dataset_path, usecols=lambda c: c in usecols, low_memory=False)
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
            leaked = [c for c in TARGET_COLUMNS if c in df.columns]
            if leaked:
                df = df.drop(columns=leaked)
            self._features = df
        return self._features

    def _load_satellite(self) -> pd.DataFrame | None:
        if self._sat is None:
            if not self.satellite_replay_path.is_file():
                self._sat = pd.DataFrame()
                return self._sat
            cols = [
                "station_id",
                "timestamp_utc",
                "sat_cloudy_fraction_25km",
                "sat_clear_fraction_25km",
                "sat_observation_available",
                "sat_observation_time_utc",
                "sat_age_minutes",
                "event_id",
                "case_type",
            ]
            df = pd.read_csv(
                self.satellite_replay_path,
                usecols=lambda c: c in cols,
                low_memory=False,
            )
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
            self._sat = df
        return self._sat

    def satellite_context_for(self, station_id: str, ts: pd.Timestamp) -> dict[str, Any] | None:
        sat = self._load_satellite()
        if sat is None or sat.empty:
            return None
        mask = (sat["station_id"].astype(str) == station_id) & (sat["timestamp_utc"] == ts)
        rows = sat.loc[mask]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return {
            "available": True,
            "event_id": None if pd.isna(row.get("event_id")) else str(row.get("event_id")),
            "case_type": None if pd.isna(row.get("case_type")) else str(row.get("case_type")),
            "sat_cloudy_fraction_25km": _maybe_float(row.get("sat_cloudy_fraction_25km")),
            "sat_clear_fraction_25km": _maybe_float(row.get("sat_clear_fraction_25km")),
            "sat_observation_time_utc": None
            if pd.isna(row.get("sat_observation_time_utc"))
            else str(row.get("sat_observation_time_utc")),
            "sat_age_minutes": _maybe_float(row.get("sat_age_minutes")),
            "note": "INSAT-3DR CMK evidence/context only. Not a Model B predictor. Not a multimodal probability.",
        }

    def predict_from_features(
        self,
        station_id: Any,
        timestamp_utc: Any,
        features: Mapping[str, Any] | list[Any],
        *,
        inference_mode: str = "HISTORICAL_REPLAY",
        satellite_context: Any = None,
        radar_context: Any = None,
        lightning_context: Any = None,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.engine.predict(
            station_id,
            timestamp_utc,
            features,
            feature_names=feature_names,
            inference_mode=inference_mode,
            satellite_context=satellite_context,
            radar_context=radar_context,
            lightning_context=lightning_context,
        )

    def replay_historical(
        self,
        station_id: Any,
        timestamp_utc: Any,
        *,
        inference_mode: str = "HISTORICAL_REPLAY",
        evidence_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence_context = evidence_context or {}
        sat_ctx = evidence_context.get("satellite_context")
        radar_ctx = evidence_context.get("radar_context")
        lightning_ctx = evidence_context.get("lightning_context")

        sid = "" if station_id is None else str(station_id).strip().upper()
        if sid not in self.engine.stations:
            return self.engine.predict(
                station_id,
                timestamp_utc,
                {},
                inference_mode=inference_mode,
                satellite_context=sat_ctx,
                radar_context=radar_ctx,
                lightning_context=lightning_ctx,
            )

        parsed = pd.to_datetime(timestamp_utc, utc=True, errors="coerce")
        if pd.isna(parsed):
            return self.engine.predict(
                sid,
                timestamp_utc,
                {},
                inference_mode=inference_mode,
                satellite_context=sat_ctx,
                radar_context=radar_ctx,
                lightning_context=lightning_ctx,
            )
        df = self._load_features()
        mask = (df["station_id"].astype(str) == sid) & (df["timestamp_utc"] == parsed)
        rows = df.loc[mask]
        if rows.empty:
            return self.engine._error(
                station_id=sid or station_id,
                timestamp_utc=parsed.isoformat(),
                code="INVALID_TIMESTAMP",
                message="no frozen NWP-overlap row for this station_id and timestamp_utc",
                inference_mode=inference_mode,
                satellite_context=sat_ctx,
                radar_context=radar_ctx,
                lightning_context=lightning_ctx,
            )

        row = rows.iloc[0]
        mapping = {n: row[n] for n in self.engine.feature_names}
        if sat_ctx is None:
            sat_ctx = self.satellite_context_for(sid, parsed)
        return self.engine.predict(
            sid,
            parsed,
            mapping,
            inference_mode=inference_mode,
            satellite_context=sat_ctx,
            radar_context=radar_ctx,
            lightning_context=lightning_ctx,
        )


def _maybe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def predict_multilead(
    station_id: Any,
    timestamp_utc: Any,
    *,
    features: Mapping[str, Any] | list[Any] | None = None,
    inference_mode: str = "HISTORICAL_REPLAY",
    evidence_context: Mapping[str, Any] | None = None,
    service: MultiLeadInferenceService | None = None,
) -> dict[str, Any]:
    svc = service if service is not None else MultiLeadInferenceService()
    evidence_context = evidence_context or {}
    if features is not None:
        return svc.predict_from_features(
            station_id,
            timestamp_utc,
            features,
            inference_mode=inference_mode,
            satellite_context=evidence_context.get("satellite_context"),
            radar_context=evidence_context.get("radar_context"),
            lightning_context=evidence_context.get("lightning_context"),
        )
    return svc.replay_historical(
        station_id,
        timestamp_utc,
        inference_mode=inference_mode,
        evidence_context=evidence_context,
    )
