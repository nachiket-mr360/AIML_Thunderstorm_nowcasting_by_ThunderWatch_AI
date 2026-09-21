"""Phase 13B: V2 Model B inference engine (Phase 13A contract).

Loads frozen Phase 8D joblibs. Does not train, fetch data, or serve HTTP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
CONTRACT_PATH = BASE_DIR / "outputs" / "v2_inference" / "phase13a" / "inference_contract.json"

OUTPUT_FIELDS = (
    "prediction_timestamp_utc",
    "station_id",
    "lead_1h_probability",
    "lead_1h_alert",
    "lead_2h_probability",
    "lead_2h_alert",
    "lead_3h_probability",
    "lead_3h_alert",
    "model_version",
    "threshold_1h",
    "threshold_2h",
    "threshold_3h",
    "data_completeness",
    "data_status",
)


class V2InferenceEngine:
    def __init__(self, contract_path: Path | None = None, base_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else BASE_DIR
        path = Path(contract_path) if contract_path is not None else CONTRACT_PATH
        self.contract = json.loads(path.read_text(encoding="utf-8"))
        self.model_version = str(self.contract["model_version"])
        self.stations = frozenset(self.contract["stations"])
        self.feature_names: list[str] = [f["column"] for f in self.contract["model_b_features"]]
        if len(self.feature_names) != int(self.contract["n_model_features"]):
            raise RuntimeError("Phase 13A contract feature count mismatch")
        thr = self.contract["thresholds"]
        self.threshold_1h = float(thr["1h"])
        self.threshold_2h = float(thr["2h"])
        self.threshold_3h = float(thr["3h"])
        arts = self.contract["frozen_artifacts"]
        self._model_paths = {
            "1h": self.base_dir / arts["target_1h"]["artifact_path"],
            "2h": self.base_dir / arts["target_2h"]["artifact_path"],
            "3h": self.base_dir / arts["target_3h"]["artifact_path"],
        }
        self._models: dict[str, Any] | None = None
        self.models_called = 0

    def _load_models(self) -> dict[str, Any]:
        if self._models is None:
            self._models = {k: joblib.load(p) for k, p in self._model_paths.items()}
        return self._models

    def _unavailable(
        self,
        station_id: Any,
        timestamp_utc: Any,
        completeness: float,
    ) -> dict[str, Any]:
        return {
            "prediction_timestamp_utc": timestamp_utc,
            "station_id": station_id,
            "lead_1h_probability": None,
            "lead_1h_alert": None,
            "lead_2h_probability": None,
            "lead_2h_alert": None,
            "lead_3h_probability": None,
            "lead_3h_alert": None,
            "model_version": self.model_version,
            "threshold_1h": self.threshold_1h,
            "threshold_2h": self.threshold_2h,
            "threshold_3h": self.threshold_3h,
            "data_completeness": float(completeness),
            "data_status": "UNAVAILABLE",
        }

    def _completeness(self, values: Sequence[Any] | None) -> float:
        n = len(self.feature_names)
        if not values:
            return 0.0
        ok = 0
        for v in list(values)[:n]:
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(x):
                ok += 1
        return ok / n

    def _to_vector(
        self,
        features: Mapping[str, Any] | Sequence[Any],
        feature_names: Sequence[str] | None,
    ) -> tuple[np.ndarray | None, float, bool]:
        """Return (vector_or_None, completeness, valid)."""
        names = self.feature_names
        if feature_names is not None:
            if list(feature_names) != names:
                vals: list[Any] | None
                if isinstance(features, Mapping):
                    vals = [features.get(n) for n in names]
                else:
                    vals = list(features)
                return None, self._completeness(vals), False

        if isinstance(features, Mapping):
            if set(features.keys()) != set(names):
                vals = [features.get(n) for n in names]
                return None, self._completeness(vals), False
            raw = [features[n] for n in names]
        else:
            raw = list(features)
            if len(raw) != len(names):
                return None, self._completeness(raw), False

        completeness = self._completeness(raw)
        vec = np.empty((1, len(names)), dtype=np.float32)
        for i, v in enumerate(raw):
            if v is None:
                return None, completeness, False
            try:
                x = float(v)
            except (TypeError, ValueError):
                return None, completeness, False
            if not np.isfinite(x):
                return None, completeness, False
            vec[0, i] = np.float32(x)
        return vec, completeness, True

    @staticmethod
    def alert(probability: float, threshold: float) -> int:
        return 1 if probability >= threshold else 0

    def predict(
        self,
        station_id: Any,
        timestamp_utc: Any,
        features: Mapping[str, Any] | Sequence[Any],
        feature_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        identity_ok = (
            station_id is not None
            and str(station_id).strip() != ""
            and timestamp_utc is not None
            and str(timestamp_utc).strip() != ""
            and str(station_id) in self.stations
        )
        vec, completeness, feat_ok = self._to_vector(features, feature_names)
        if not identity_ok or not feat_ok or vec is None:
            return self._unavailable(station_id, timestamp_utc, completeness)

        models = self._load_models()
        self.models_called += 1
        p1 = float(models["1h"].predict_proba(vec)[0, 1])
        p2 = float(models["2h"].predict_proba(vec)[0, 1])
        p3 = float(models["3h"].predict_proba(vec)[0, 1])
        return {
            "prediction_timestamp_utc": timestamp_utc,
            "station_id": station_id,
            "lead_1h_probability": p1,
            "lead_1h_alert": self.alert(p1, self.threshold_1h),
            "lead_2h_probability": p2,
            "lead_2h_alert": self.alert(p2, self.threshold_2h),
            "lead_3h_probability": p3,
            "lead_3h_alert": self.alert(p3, self.threshold_3h),
            "model_version": self.model_version,
            "threshold_1h": self.threshold_1h,
            "threshold_2h": self.threshold_2h,
            "threshold_3h": self.threshold_3h,
            "data_completeness": float(completeness),
            "data_status": "COMPLETE",
        }
