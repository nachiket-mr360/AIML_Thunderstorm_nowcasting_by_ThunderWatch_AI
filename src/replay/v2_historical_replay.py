"""Phase 15A: deterministic HISTORICAL REPLAY (not live forecasting).

Loads causal history at or before T from the validated NWP overlap CSV,
builds the Phase 14A 83-feature vector, runs the frozen Phase 13B engine,
THEN looks up historical T+1/T+2/T+3 labels for verification only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.features.v2_feature_builder import (
    ATMOSPHERIC_SOURCE,
    LOCATION_SOURCE,
    NWP_SOURCE,
    STATION,
    TIME,
    V2FeatureBuilder,
)
from src.inference.v2_inference_engine import V2InferenceEngine

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    BASE_DIR
    / "dataset"
    / "multilocation"
    / "features_nwp"
    / "multilocation_features_nwp_overlap_2021_2025.csv"
)

REPLAY_MODE = "HISTORICAL_REPLAY"
TARGET_COLUMNS = ("target_1h", "target_2h", "target_3h")
BUILDER_COLUMNS = [STATION, TIME, *ATMOSPHERIC_SOURCE, *NWP_SOURCE, *LOCATION_SOURCE]

KNOWN_STATIONS = ("VOTV", "VECC", "VIDP", "VOCI", "VABB")

REPLAY_OUTPUT_FIELDS = (
    "replay_mode",
    "prediction_timestamp_utc",
    "station_id",
    "model_version",
    "data_status",
    "data_completeness",
    "lead_1h_probability",
    "lead_1h_alert",
    "lead_2h_probability",
    "lead_2h_alert",
    "lead_3h_probability",
    "lead_3h_alert",
    "historical_target_1h",
    "historical_target_2h",
    "historical_target_3h",
    "lead_1h_alert_hit",
    "lead_2h_alert_hit",
    "lead_3h_alert_hit",
    "threshold_1h",
    "threshold_2h",
    "threshold_3h",
    "unavailable_reason",
)


def _iso(ts: Any) -> Any:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return ts
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(t):
        return ts
    return t.isoformat()


def _as_int_label(val: Any) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    if not pd.notna(x):
        return None
    return int(x)


def _alert_hit(alert: Any, target: Any) -> int | None:
    if alert is None or target is None:
        return None
    return int(int(alert) == int(target))


class V2HistoricalReplay:
    """Fail-closed historical replay over the validated NWP overlap dataset."""

    def __init__(
        self,
        *,
        dataset_path: Path | None = None,
        builder: V2FeatureBuilder | None = None,
        engine: V2InferenceEngine | None = None,
        feature_builder_hook: Any = None,
    ) -> None:
        self.dataset_path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET
        self.builder = builder if builder is not None else V2FeatureBuilder()
        self.engine = engine if engine is not None else V2InferenceEngine()
        self.feature_builder_hook = feature_builder_hook
        self.stations = frozenset(self.engine.stations)
        self._frame: pd.DataFrame | None = None
        self.last_builder_columns: list[str] | None = None
        self.prediction_before_target_lookup = False
        self.targets_looked_up = False

    def _unavailable(
        self,
        station_id: Any,
        timestamp_utc: Any,
        reason: str,
        completeness: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "replay_mode": REPLAY_MODE,
            "prediction_timestamp_utc": _iso(timestamp_utc) if timestamp_utc is not None else timestamp_utc,
            "station_id": station_id,
            "model_version": self.engine.model_version,
            "data_status": "UNAVAILABLE",
            "data_completeness": float(completeness),
            "lead_1h_probability": None,
            "lead_1h_alert": None,
            "lead_2h_probability": None,
            "lead_2h_alert": None,
            "lead_3h_probability": None,
            "lead_3h_alert": None,
            "historical_target_1h": None,
            "historical_target_2h": None,
            "historical_target_3h": None,
            "lead_1h_alert_hit": None,
            "lead_2h_alert_hit": None,
            "lead_3h_alert_hit": None,
            "threshold_1h": self.engine.threshold_1h,
            "threshold_2h": self.engine.threshold_2h,
            "threshold_3h": self.engine.threshold_3h,
            "unavailable_reason": reason,
        }

    def load_dataset(self) -> pd.DataFrame:
        if self._frame is None:
            if not self.dataset_path.is_file():
                raise FileNotFoundError(self.dataset_path)
            self._frame = pd.read_csv(self.dataset_path, low_memory=False)
        return self._frame

    @staticmethod
    def _ensure_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
        """Overlap CSV stores wind_direction_sin/cos, not wind_direction_10m.

        Reconstruct the source angle for Phase 14A only; do not fill other fields.
        """
        if df.empty or "wind_direction_10m" in df.columns:
            return df
        if "wind_direction_sin" in df.columns and "wind_direction_cos" in df.columns:
            out = df.copy()
            s = pd.to_numeric(out["wind_direction_sin"], errors="coerce")
            c = pd.to_numeric(out["wind_direction_cos"], errors="coerce")
            ang = np.degrees(np.arctan2(s.to_numpy(dtype=np.float64), c.to_numpy(dtype=np.float64)))
            ang = np.mod(ang, 360.0)
            out["wind_direction_10m"] = ang
            return out
        return df

    def _station_frame(self, source: pd.DataFrame, station_id: str) -> pd.DataFrame:
        df = source.loc[source[STATION].astype(str) == station_id].copy()
        ts = pd.to_datetime(df[TIME], utc=True, errors="coerce")
        df[TIME] = ts
        df = df.loc[ts.notna()].sort_values(TIME)
        return df.reset_index(drop=True)

    def _lookup_targets(self, station_df: pd.DataFrame, t_pred: pd.Timestamp) -> dict[str, int | None]:
        """Post-inference only. Labels for thunderstorm at T+1 / T+2 / T+3."""
        self.targets_looked_up = True
        out: dict[str, int | None] = {
            "historical_target_1h": None,
            "historical_target_2h": None,
            "historical_target_3h": None,
        }
        timed = station_df.set_index(TIME)
        for lead, key in ((1, "historical_target_1h"), (2, "historical_target_2h"), (3, "historical_target_3h")):
            col = f"target_{lead}h"
            if col in station_df.columns and t_pred in timed.index:
                out[key] = _as_int_label(timed.loc[t_pred, col] if not isinstance(timed.loc[t_pred, col], pd.Series) else timed.loc[t_pred, col].iloc[0])
                continue
            t_fut = t_pred + pd.Timedelta(hours=lead)
            if t_fut in timed.index:
                row = timed.loc[t_fut]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                if "thunderstorm" in timed.columns:
                    out[key] = _as_int_label(row["thunderstorm"])
        return out

    def run(
        self,
        station_id: str,
        timestamp_utc: str,
        *,
        history: pd.DataFrame | Mapping[str, Sequence[Any]] | None = None,
    ) -> dict[str, Any]:
        self.last_builder_columns = None
        self.prediction_before_target_lookup = False
        self.targets_looked_up = False

        if station_id is None or str(station_id).strip() == "" or str(station_id) not in self.stations:
            return self._unavailable(station_id, timestamp_utc, "unknown_station")

        station_id = str(station_id)
        t_pred = pd.to_datetime(timestamp_utc, utc=True, errors="coerce")
        if pd.isna(t_pred):
            return self._unavailable(station_id, timestamp_utc, "invalid_timestamp")

        if history is None:
            try:
                source = self.load_dataset()
            except Exception as exc:
                return self._unavailable(station_id, timestamp_utc, f"dataset_unreadable:{exc}")
        else:
            source = history if isinstance(history, pd.DataFrame) else pd.DataFrame(history)

        if TIME not in source.columns or STATION not in source.columns:
            return self._unavailable(station_id, timestamp_utc, "missing_identity_columns")

        all_ts = pd.to_datetime(source[TIME], utc=True, errors="coerce")
        valid_ts = all_ts.dropna()
        if valid_ts.empty:
            return self._unavailable(station_id, timestamp_utc, "timestamp_outside_nwp_overlap")
        tmin, tmax = valid_ts.min(), valid_ts.max()
        if t_pred < tmin or t_pred > tmax:
            return self._unavailable(station_id, timestamp_utc, "timestamp_outside_nwp_overlap")

        station_df = self._station_frame(source, station_id)
        station_df = self._ensure_wind_direction(station_df)
        if station_df.empty:
            return self._unavailable(station_id, timestamp_utc, "unknown_station")
        if station_df[TIME].duplicated().any():
            return self._unavailable(station_id, timestamp_utc, "duplicate_timestamp")

        if t_pred not in set(station_df[TIME]):
            return self._unavailable(station_id, timestamp_utc, "timestamp_outside_nwp_overlap")

        causal = station_df.loc[station_df[TIME] <= t_pred].copy()
        keep = [c for c in BUILDER_COLUMNS if c in causal.columns]
        builder_input = causal[keep].copy()
        leaked = [c for c in TARGET_COLUMNS if c in builder_input.columns]
        if leaked:
            builder_input = builder_input.drop(columns=leaked)
        self.last_builder_columns = list(builder_input.columns)
        if self.feature_builder_hook is not None:
            self.feature_builder_hook(builder_input)

        built = self.builder.build_features(
            builder_input,
            timestamp_utc=t_pred,
            station_id=station_id,
        )
        if not built.ok or built.features is None:
            return self._unavailable(
                station_id,
                t_pred,
                built.reason or "feature_build_unavailable",
                completeness=built.data_completeness,
            )

        pred = self.engine.predict(
            station_id,
            built.timestamp_utc,
            built.features,
            feature_names=built.feature_names,
        )
        self.prediction_before_target_lookup = True

        hist_targets = {
            "historical_target_1h": None,
            "historical_target_2h": None,
            "historical_target_3h": None,
        }
        if pred.get("data_status") == "COMPLETE":
            hist_targets = self._lookup_targets(station_df, t_pred)

        if pred.get("data_status") != "COMPLETE":
            out = self._unavailable(
                station_id,
                t_pred,
                "inference_unavailable",
                completeness=float(pred.get("data_completeness") or 0.0),
            )
            return out

        result = {
            "replay_mode": REPLAY_MODE,
            "prediction_timestamp_utc": pred["prediction_timestamp_utc"],
            "station_id": pred["station_id"],
            "model_version": pred["model_version"],
            "data_status": pred["data_status"],
            "data_completeness": pred["data_completeness"],
            "lead_1h_probability": pred["lead_1h_probability"],
            "lead_1h_alert": pred["lead_1h_alert"],
            "lead_2h_probability": pred["lead_2h_probability"],
            "lead_2h_alert": pred["lead_2h_alert"],
            "lead_3h_probability": pred["lead_3h_probability"],
            "lead_3h_alert": pred["lead_3h_alert"],
            "historical_target_1h": hist_targets["historical_target_1h"],
            "historical_target_2h": hist_targets["historical_target_2h"],
            "historical_target_3h": hist_targets["historical_target_3h"],
            "lead_1h_alert_hit": _alert_hit(pred["lead_1h_alert"], hist_targets["historical_target_1h"]),
            "lead_2h_alert_hit": _alert_hit(pred["lead_2h_alert"], hist_targets["historical_target_2h"]),
            "lead_3h_alert_hit": _alert_hit(pred["lead_3h_alert"], hist_targets["historical_target_3h"]),
            "threshold_1h": pred["threshold_1h"],
            "threshold_2h": pred["threshold_2h"],
            "threshold_3h": pred["threshold_3h"],
            "unavailable_reason": None,
        }
        return result


def run_historical_replay(
    station_id: str,
    timestamp_utc: str,
    **kwargs: Any,
) -> dict[str, Any]:
    replay = kwargs.pop("replay", None)
    if replay is None:
        replay = V2HistoricalReplay(
            dataset_path=kwargs.pop("dataset_path", None),
            builder=kwargs.pop("builder", None),
            engine=kwargs.pop("engine", None),
            feature_builder_hook=kwargs.pop("feature_builder_hook", None),
        )
    return replay.run(station_id, timestamp_utc, history=kwargs.pop("history", None))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Phase 15A historical replay (research prototype; not live forecasting)."
    )
    p.add_argument("--station", required=True, help="ICAO station id (VOTV, VECC, VIDP, VOCI, VABB)")
    p.add_argument("--timestamp", required=True, help="Prediction time T in UTC (e.g. 2024-04-15T12:00:00Z)")
    p.add_argument("--dataset", default=None, help="Optional override path to NWP overlap CSV")
    args = p.parse_args(argv)
    path = Path(args.dataset) if args.dataset else None
    result = run_historical_replay(args.station, args.timestamp, dataset_path=path)
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0 if result.get("data_status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
