"""Phase 14A: replay feature builder for Model B (Phase 13A contract).

Converts a chronological atmospheric + NWP history into the exact 83-feature
vector consumed by ``V2InferenceEngine``. Formulas are Phase 3
(``dataset/multilocation/feature_engineering_phase3.py``). Canonical names
and order come from ``outputs/v2_inference/phase13a/inference_contract.json``.

This is a historical/replay builder. It is not a live data pipeline.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
CONTRACT_PATH = BASE_DIR / "outputs" / "v2_inference" / "phase13a" / "inference_contract.json"
PHASE3_PATH = BASE_DIR / "dataset" / "multilocation" / "feature_engineering_phase3.py"

TIME = "timestamp_utc"
STATION = "station_id"

ATMOSPHERIC_SOURCE = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]
NWP_SOURCE = [
    "nwp_cape",
    "nwp_convective_inhibition",
    "nwp_lifted_index",
    "nwp_temperature_2m",
    "nwp_relative_humidity_2m",
    "nwp_surface_pressure",
    "nwp_wind_speed_10m",
    "nwp_wind_direction_10m",
    "nwp_precipitation",
    "nwp_cloud_cover",
]
LOCATION_SOURCE = ["latitude", "longitude", "elevation_m"]

# lag_24h / roll_24h need 25 hourly samples: T-24 ... T inclusive.
MIN_HISTORY_ROWS = 25
MAX_HISTORY_HOURS = 24


def _load_phase3():
    spec = importlib.util.spec_from_file_location("feature_engineering_phase3", PHASE3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Phase 3 builder from {PHASE3_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PHASE3 = _load_phase3()


def load_contract(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else CONTRACT_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def contract_feature_names(contract: Mapping[str, Any] | None = None) -> list[str]:
    c = contract if contract is not None else load_contract()
    names = [f["column"] for f in c["model_b_features"]]
    if len(names) != int(c["n_model_features"]):
        raise RuntimeError("Phase 13A contract feature count mismatch")
    return names


@dataclass
class FeatureBuildResult:
    """Fail-closed result. ``features`` is set only when status is COMPLETE."""

    status: str
    reason: str | None
    station_id: Any
    timestamp_utc: Any
    features: dict[str, float] | None
    feature_names: list[str] = field(default_factory=list)
    data_completeness: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "COMPLETE"


def _unavailable(
    names: Sequence[str],
    reason: str,
    station_id: Any = None,
    timestamp_utc: Any = None,
    completeness: float = 0.0,
) -> FeatureBuildResult:
    return FeatureBuildResult(
        status="UNAVAILABLE",
        reason=reason,
        station_id=station_id,
        timestamp_utc=timestamp_utc,
        features=None,
        feature_names=list(names),
        data_completeness=float(completeness),
    )


def _to_frame(history: pd.DataFrame | Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(history, pd.DataFrame):
        return history.copy()
    return pd.DataFrame(history)


def _finite_series(s: pd.Series) -> bool:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    return bool(np.isfinite(arr).all())


class V2FeatureBuilder:
    """Build the Phase 13A 83-feature record from causal history ending at T."""

    def __init__(self, contract_path: Path | None = None) -> None:
        self.contract = load_contract(contract_path)
        self.feature_names = contract_feature_names(self.contract)
        self.stations = frozenset(self.contract["stations"])

    def build_features(
        self,
        history: pd.DataFrame | Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
        *,
        timestamp_utc: Any = None,
        station_id: Any = None,
    ) -> FeatureBuildResult:
        names = self.feature_names
        try:
            df = _to_frame(history)
        except Exception as exc:
            return _unavailable(names, f"invalid_history:{exc}")

        if df.empty:
            return _unavailable(names, "empty_history")

        missing_cols = [
            c
            for c in [STATION, TIME, *ATMOSPHERIC_SOURCE, *NWP_SOURCE, *LOCATION_SOURCE]
            if c not in df.columns
        ]
        if missing_cols:
            return _unavailable(names, f"missing_source_columns:{missing_cols}")

        if station_id is None:
            stations = df[STATION].astype(str).unique().tolist()
            if len(stations) != 1:
                return _unavailable(names, "invalid_station_identity:multiple_or_missing")
            station_id = stations[0]
        station_id = str(station_id).strip()
        if not station_id or station_id not in self.stations:
            return _unavailable(names, "invalid_station_identity", station_id=station_id)

        df = df.loc[df[STATION].astype(str) == station_id].copy()
        if df.empty:
            return _unavailable(names, "invalid_station_identity", station_id=station_id)

        ts = pd.to_datetime(df[TIME], utc=True, errors="coerce")
        if ts.isna().any():
            return _unavailable(names, "invalid_timestamp", station_id=station_id)
        df[TIME] = ts

        if df[TIME].duplicated().any():
            return _unavailable(names, "duplicate_timestamp", station_id=station_id)

        if not df[TIME].is_monotonic_increasing:
            return _unavailable(names, "non_monotonic_timestamp", station_id=station_id)

        deltas = df[TIME].diff().dropna()
        if len(deltas) and not bool((deltas == pd.Timedelta(hours=1)).all()):
            return _unavailable(
                names,
                "non_hourly_or_non_monotonic_timestamp",
                station_id=station_id,
            )

        if timestamp_utc is None:
            t_pred = df[TIME].iloc[-1]
        else:
            t_pred = pd.to_datetime(timestamp_utc, utc=True, errors="coerce")
            if pd.isna(t_pred):
                return _unavailable(names, "invalid_timestamp", station_id=station_id)
            if t_pred not in set(df[TIME]):
                return _unavailable(
                    names,
                    "timestamp_not_in_history",
                    station_id=station_id,
                    timestamp_utc=t_pred.isoformat(),
                )

        causal = df.loc[df[TIME] <= t_pred].copy()
        if len(causal) < MIN_HISTORY_ROWS:
            return _unavailable(
                names,
                "insufficient_history",
                station_id=station_id,
                timestamp_utc=t_pred.isoformat(),
            )

        span_h = (causal[TIME].iloc[-1] - causal[TIME].iloc[0]) / pd.Timedelta(hours=1)
        if span_h < MAX_HISTORY_HOURS:
            return _unavailable(
                names,
                "insufficient_history",
                station_id=station_id,
                timestamp_utc=t_pred.isoformat(),
            )

        for col in ATMOSPHERIC_SOURCE + LOCATION_SOURCE:
            if not _finite_series(causal[col]):
                return _unavailable(
                    names,
                    f"missing_or_nonfinite_atmospheric_or_location:{col}",
                    station_id=station_id,
                    timestamp_utc=t_pred.isoformat(),
                )

        row_t = causal.loc[causal[TIME] == t_pred].iloc[-1]
        for col in NWP_SOURCE:
            try:
                x = float(row_t[col])
            except (TypeError, ValueError):
                return _unavailable(
                    names,
                    f"missing_required_nwp:{col}",
                    station_id=station_id,
                    timestamp_utc=t_pred.isoformat(),
                )
            if not np.isfinite(x):
                return _unavailable(
                    names,
                    f"missing_required_nwp:{col}",
                    station_id=station_id,
                    timestamp_utc=t_pred.isoformat(),
                )

        elev = float(row_t["elevation_m"])
        grid = causal[[TIME, *ATMOSPHERIC_SOURCE, "latitude", "longitude"]].copy()
        featured = _PHASE3.add_features(grid, elev)
        last = featured.iloc[-1]

        out: dict[str, float] = {}
        for col in names:
            if col in NWP_SOURCE:
                val = float(row_t[col])
            else:
                val = float(last[col])
            if not np.isfinite(val):
                return _unavailable(
                    names,
                    f"derived_nonfinite:{col}",
                    station_id=station_id,
                    timestamp_utc=t_pred.isoformat(),
                )
            out[col] = val

        if list(out.keys()) != names:
            return _unavailable(
                names,
                "feature_order_mismatch",
                station_id=station_id,
                timestamp_utc=t_pred.isoformat(),
            )

        return FeatureBuildResult(
            status="COMPLETE",
            reason=None,
            station_id=station_id,
            timestamp_utc=pd.Timestamp(t_pred).isoformat(),
            features=out,
            feature_names=list(names),
            data_completeness=1.0,
        )


def build_features(
    history: pd.DataFrame | Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
    *,
    timestamp_utc: Any = None,
    station_id: Any = None,
    contract_path: Path | None = None,
) -> FeatureBuildResult:
    """Build contract-ordered Model B features from chronological history."""
    return V2FeatureBuilder(contract_path=contract_path).build_features(
        history, timestamp_utc=timestamp_utc, station_id=station_id
    )
