"""Phase 13A: final multi-lead inference engine (preparation only).

Uses frozen Phase 8D Model B joblibs and the existing 13A feature contract.
Does not train, download, or write model/data artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
CONTRACT_PATH = BASE_DIR / "outputs" / "v2_inference" / "phase13a" / "inference_contract.json"

MODEL_VERSION = "V2-MODEL-B-MULTILEAD-PHASE8D-RESEARCH"
FROZEN_THRESHOLD = 0.065
LEADS = ("1h", "2h", "3h")

FORBIDDEN_SUBSTITUTION_MARKERS = (
    "satellite",
    "sat_",
    "radar",
    "lightning",
    "lis_",
    "annual_thunder",
    "cmk",
    "target_",
)

CONTEXT_KEYS = frozenset(
    {
        "satellite_context",
        "radar_context",
        "lightning_context",
        "core_model_output",
    }
)


def _parse_utc(value: Any) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


class FinalMultiLeadEngine:
    def __init__(self, contract_path: Path | None = None, base_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else BASE_DIR
        path = Path(contract_path) if contract_path is not None else CONTRACT_PATH
        self.contract = json.loads(path.read_text(encoding="utf-8"))
        self.model_version = str(self.contract.get("model_version", MODEL_VERSION))
        self.stations = frozenset(self.contract["stations"])
        self.feature_names: list[str] = [f["column"] for f in self.contract["model_b_features"]]
        if len(self.feature_names) != 83:
            raise RuntimeError("frozen Model B contract must list 83 features")
        thr = self.contract["thresholds"]
        self.thresholds = {
            "1h": float(thr["1h"]),
            "2h": float(thr["2h"]),
            "3h": float(thr["3h"]),
        }
        if any(abs(v - FROZEN_THRESHOLD) > 1e-12 for v in self.thresholds.values()):
            raise RuntimeError("frozen thresholds must remain 0.065")
        arts = self.contract["frozen_artifacts"]
        self._model_paths = {
            "1h": self.base_dir / arts["target_1h"]["artifact_path"],
            "2h": self.base_dir / arts["target_2h"]["artifact_path"],
            "3h": self.base_dir / arts["target_3h"]["artifact_path"],
        }
        self._models: dict[str, Any] | None = None
        self.models_called = 0

    def load_models(self) -> dict[str, Any]:
        if self._models is None:
            loaded = {}
            for k, p in self._model_paths.items():
                clf = joblib.load(p)
                n_in = int(getattr(clf, "n_features_in_", -1))
                if n_in != 83:
                    raise RuntimeError(f"{k} joblib n_features_in_={n_in}, expected 83")
                loaded[k] = clf
            self._models = loaded
        return self._models

    def _error(
        self,
        *,
        station_id: Any,
        timestamp_utc: Any,
        code: str,
        message: str,
        warnings: list[str] | None = None,
        inference_mode: str = "HISTORICAL_REPLAY",
        satellite_context: Any = None,
        radar_context: Any = None,
        lightning_context: Any = None,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "station_id": station_id,
            "timestamp_utc": timestamp_utc,
            "model_version": self.model_version,
            "inference_mode": inference_mode,
            "lead_1h": None,
            "lead_2h": None,
            "lead_3h": None,
            "data_status": "UNAVAILABLE",
            "error": {"code": code, "message": message},
            "warnings": list(warnings or []),
            "core_model_output": None,
            "satellite_context": satellite_context,
            "radar_context": radar_context,
            "lightning_context": lightning_context,
        }

    def _lead_block(self, probability: float, lead: str) -> dict[str, Any]:
        thr = self.thresholds[lead]
        return {
            "probability": float(probability),
            "threshold": thr,
            "alert": bool(probability >= thr),
        }

    def predict(
        self,
        station_id: Any,
        timestamp_utc: Any,
        features: Mapping[str, Any] | Sequence[Any],
        feature_names: Sequence[str] | None = None,
        *,
        inference_mode: str = "HISTORICAL_REPLAY",
        satellite_context: Any = None,
        radar_context: Any = None,
        lightning_context: Any = None,
        allow_extra_context_keys: bool = True,
    ) -> dict[str, Any]:
        warnings: list[str] = []

        if station_id is None or str(station_id).strip() == "":
            return self._error(
                station_id=station_id,
                timestamp_utc=timestamp_utc,
                code="INVALID_STATION",
                message="station_id is required",
                inference_mode=inference_mode,
                satellite_context=satellite_context,
                radar_context=radar_context,
                lightning_context=lightning_context,
            )
        station = str(station_id).strip().upper()
        if station not in self.stations:
            return self._error(
                station_id=station_id,
                timestamp_utc=timestamp_utc,
                code="INVALID_STATION",
                message=f"station {station_id!r} is not in frozen station set {sorted(self.stations)}",
                inference_mode=inference_mode,
                satellite_context=satellite_context,
                radar_context=radar_context,
                lightning_context=lightning_context,
            )

        parsed_ts = _parse_utc(timestamp_utc)
        if parsed_ts is None:
            return self._error(
                station_id=station,
                timestamp_utc=timestamp_utc,
                code="INVALID_TIMESTAMP",
                message="timestamp_utc must be a valid timezone-aware UTC datetime",
                inference_mode=inference_mode,
                satellite_context=satellite_context,
                radar_context=radar_context,
                lightning_context=lightning_context,
            )
        ts_iso = parsed_ts.isoformat()

        names = self.feature_names
        extra_forbidden: list[str] = []

        if feature_names is not None:
            provided = [str(n) for n in feature_names]
            extra_forbidden = [
                n for n in provided if any(m in n.lower() for m in FORBIDDEN_SUBSTITUTION_MARKERS)
            ]
            if extra_forbidden:
                return self._error(
                    station_id=station,
                    timestamp_utc=ts_iso,
                    code="FORBIDDEN_FEATURE_SUBSTITUTION",
                    message=(
                        "satellite/radar/lightning/target columns must not enter the Model B vector: "
                        + ", ".join(extra_forbidden)
                    ),
                    inference_mode=inference_mode,
                    satellite_context=satellite_context,
                    radar_context=radar_context,
                    lightning_context=lightning_context,
                )
            if provided != names:
                return self._error(
                    station_id=station,
                    timestamp_utc=ts_iso,
                    code="FEATURE_ORDER_MISMATCH",
                    message="feature_names must match frozen Model B order exactly",
                    inference_mode=inference_mode,
                    satellite_context=satellite_context,
                    radar_context=radar_context,
                    lightning_context=lightning_context,
                )

        if isinstance(features, Mapping):
            keys = [str(k) for k in features.keys()]
            extra = [k for k in keys if k not in names and k not in CONTEXT_KEYS]
            extra_forbidden = [
                k for k in extra if any(m in k.lower() for m in FORBIDDEN_SUBSTITUTION_MARKERS)
            ]
            if extra_forbidden:
                return self._error(
                    station_id=station,
                    timestamp_utc=ts_iso,
                    code="FORBIDDEN_FEATURE_SUBSTITUTION",
                    message=(
                        "refusing to substitute satellite/radar/lightning into core Model B: "
                        + ", ".join(extra_forbidden)
                    ),
                    inference_mode=inference_mode,
                    satellite_context=satellite_context,
                    radar_context=radar_context,
                    lightning_context=lightning_context,
                )
            if extra and not allow_extra_context_keys:
                return self._error(
                    station_id=station,
                    timestamp_utc=ts_iso,
                    code="UNEXPECTED_FEATURES",
                    message="unexpected columns: " + ", ".join(extra),
                    inference_mode=inference_mode,
                    satellite_context=satellite_context,
                    radar_context=radar_context,
                    lightning_context=lightning_context,
                )
            missing = [n for n in names if n not in features]
            if missing:
                nwp_missing = [n for n in missing if n.startswith("nwp_")]
                code = "MISSING_NWP_FIELDS" if nwp_missing and len(nwp_missing) == len(missing) else "MISSING_FEATURES"
                return self._error(
                    station_id=station,
                    timestamp_utc=ts_iso,
                    code=code,
                    message="required Model B columns missing: " + ", ".join(missing[:12]),
                    inference_mode=inference_mode,
                    satellite_context=satellite_context,
                    radar_context=radar_context,
                    lightning_context=lightning_context,
                )
            raw = [features[n] for n in names]
        else:
            raw = list(features)
            if len(raw) != len(names):
                return self._error(
                    station_id=station,
                    timestamp_utc=ts_iso,
                    code="FEATURE_COUNT_MISMATCH",
                    message=f"expected 83 values, got {len(raw)}",
                    inference_mode=inference_mode,
                    satellite_context=satellite_context,
                    radar_context=radar_context,
                    lightning_context=lightning_context,
                )

        vec = np.empty((1, 83), dtype=np.float32)
        for i, v in enumerate(raw):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return self._error(
                    station_id=station,
                    timestamp_utc=ts_iso,
                    code="NON_FINITE_FEATURE",
                    message=f"feature {names[i]} is missing or non-finite; fail-closed (no fill)",
                    inference_mode=inference_mode,
                    satellite_context=satellite_context,
                    radar_context=radar_context,
                    lightning_context=lightning_context,
                )
            try:
                x = float(v)
            except (TypeError, ValueError):
                return self._error(
                    station_id=station,
                    timestamp_utc=ts_iso,
                    code="NON_FINITE_FEATURE",
                    message=f"feature {names[i]} is not numeric; fail-closed (no fill)",
                    inference_mode=inference_mode,
                    satellite_context=satellite_context,
                    radar_context=radar_context,
                    lightning_context=lightning_context,
                )
            if not np.isfinite(x):
                return self._error(
                    station_id=station,
                    timestamp_utc=ts_iso,
                    code="NON_FINITE_FEATURE",
                    message=f"feature {names[i]} is missing or non-finite; fail-closed (no fill)",
                    inference_mode=inference_mode,
                    satellite_context=satellite_context,
                    radar_context=radar_context,
                    lightning_context=lightning_context,
                )
            vec[0, i] = np.float32(x)

        models = self.load_models()
        self.models_called += 1
        p1 = float(models["1h"].predict_proba(vec)[0, 1])
        p2 = float(models["2h"].predict_proba(vec)[0, 1])
        p3 = float(models["3h"].predict_proba(vec)[0, 1])
        lead_1h = self._lead_block(p1, "1h")
        lead_2h = self._lead_block(p2, "2h")
        lead_3h = self._lead_block(p3, "3h")
        if inference_mode == "HISTORICAL_REPLAY":
            warnings.append("Result is historical/replay inference, not a live forecast.")
        return {
            "ok": True,
            "station_id": station,
            "timestamp_utc": ts_iso,
            "model_version": self.model_version,
            "inference_mode": inference_mode,
            "lead_1h": lead_1h,
            "lead_2h": lead_2h,
            "lead_3h": lead_3h,
            "data_status": "COMPLETE",
            "error": None,
            "warnings": warnings,
            "core_model_output": {
                "lead_1h": lead_1h,
                "lead_2h": lead_2h,
                "lead_3h": lead_3h,
            },
            "satellite_context": satellite_context,
            "radar_context": radar_context,
            "lightning_context": lightning_context,
        }
