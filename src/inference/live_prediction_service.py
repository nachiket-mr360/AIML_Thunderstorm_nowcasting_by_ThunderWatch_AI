"""Phase 15A live inference: Open-Meteo → 83 features → FinalMultiLeadEngine.

Does not modify the frozen engine. Does not use historical replay rows as live input.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable

from src.inference.final_multi_lead_engine import FinalMultiLeadEngine
from src.inference.live_feature_adapter import (
    MAX_DATA_AGE_MINUTES,
    build_live_history,
    load_station_catalog,
)
from src.inference.live_failure_diagnostics import (
    classify_live_exception,
    live_diagnostics_enabled,
    sanitize_provider,
    station_diagnostic_entry,
)
from src.inference.live_openmeteo_client import (
    LiveDataError,
    fetch_live_payloads,
    fetch_live_payloads_batch,
)

CACHE_TTL_S = 300.0
FROZEN_THRESHOLD = 0.065
LIVE_STATIONS = ("VABB", "VIDP", "VECC", "VOCI", "VOTV")

SIGNAL_KEYS = (
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
    "nwp_cape",
    "nwp_convective_inhibition",
    "nwp_lifted_index",
)


def _signals_from_features(features: dict[str, float] | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    feats = features or {}
    for key in SIGNAL_KEYS:
        if key not in feats:
            out[key] = None
            continue
        try:
            val = float(feats[key])
        except (TypeError, ValueError):
            out[key] = None
            continue
        out[key] = val if val == val else None  # NaN check
    return out


class LivePredictionService:
    def __init__(
        self,
        *,
        engine: FinalMultiLeadEngine | None = None,
        fetch_fn: Callable[..., tuple[dict, dict]] | None = None,
        batch_fetch_fn: Callable[..., dict[str, tuple[dict, dict]]] | None = None,
        cache_ttl_s: float = CACHE_TTL_S,
        max_age_minutes: int = MAX_DATA_AGE_MINUTES,
    ) -> None:
        self.engine = engine if engine is not None else FinalMultiLeadEngine()
        self._default_fetch = fetch_fn is None
        self.fetch_fn = fetch_fn if fetch_fn is not None else fetch_live_payloads
        self.batch_fetch_fn = (
            batch_fetch_fn
            if batch_fetch_fn is not None
            else (fetch_live_payloads_batch if self._default_fetch else None)
        )
        self.cache_ttl_s = cache_ttl_s
        self.max_age_minutes = max_age_minutes
        self.catalog = load_station_catalog()
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _error(
        self,
        code: str,
        message: str,
        station_id: Any = None,
        *,
        category: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "mode": "LIVE",
            "status": code,
            "station_id": station_id,
            "error": {"code": code, "message": message},
            "predictions": None,
            "_diag_category": category,
            "_diag_provider": provider,
        }

    def predict_live(
        self,
        station_id: Any,
        *,
        now_utc: datetime | None = None,
        use_cache: bool = True,
        payloads: tuple[dict[str, Any], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if station_id is None or str(station_id).strip() == "":
            return self._error("LIVE_DATA_UNAVAILABLE", "station_id is required")
        sid = str(station_id).strip().upper()
        if sid not in self.catalog or sid not in self.engine.stations:
            return self._error("LIVE_DATA_UNAVAILABLE", f"invalid station {station_id!r}", sid)

        now = now_utc if now_utc is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            return self._error("LIVE_FEATURE_CONSTRUCTION_FAILED", "now_utc must be UTC-aware", sid)
        now = now.astimezone(timezone.utc)

        cache_hit = False
        cache_age_s = 0.0
        if use_cache:
            with self._lock:
                hit = self._cache.get(sid)
            if hit is not None:
                stored_at, stored = hit
                cache_age_s = time.time() - stored_at
                obs = stored.get("data_observation_time_utc")
                age_ok = True
                if obs:
                    from pandas import Timestamp

                    t_obs = Timestamp(obs).to_pydatetime()
                    if t_obs.tzinfo is None:
                        t_obs = t_obs.replace(tzinfo=timezone.utc)
                    data_age = (now - t_obs.astimezone(timezone.utc)).total_seconds() / 60.0
                    if data_age > self.max_age_minutes:
                        age_ok = False
                if cache_age_s <= self.cache_ttl_s and age_ok and stored.get("status") == "LIVE":
                    cache_hit = True
                    out = dict(stored)
                    out["cache_hit"] = True
                    out["cache_age_seconds"] = round(cache_age_s, 3)
                    return out

        meta = self.catalog[sid]
        try:
            if payloads is not None:
                forecast_payload, gfs_payload = payloads
            else:
                forecast_payload, gfs_payload = self.fetch_fn(
                    meta["request_latitude"],
                    meta["request_longitude"],
                )
        except LiveDataError as exc:
            return self._error(
                exc.code,
                exc.message,
                sid,
                category=classify_live_exception(exc),
                provider=sanitize_provider(getattr(exc, "provider", None)),
            )
        except Exception as exc:
            return self._error(
                "LIVE_DATA_UNAVAILABLE",
                "external atmospheric/NWP data unavailable",
                sid,
                category=classify_live_exception(exc),
            )

        try:
            _, prov = build_live_history(
                sid,
                forecast_payload,
                gfs_payload,
                now_utc=now,
                catalog=self.catalog,
                max_age_minutes=self.max_age_minutes,
            )
        except LiveDataError as exc:
            return self._error(
                exc.code,
                exc.message,
                sid,
                category="FEATURE_BUILD_ERROR",
                provider=sanitize_provider(getattr(exc, "provider", None)),
            )

        engine_out = self.engine.predict(
            sid,
            prov["data_observation_time_utc"],
            prov["features"],
            feature_names=prov["feature_names"],
            inference_mode="LIVE",
        )
        if not engine_out.get("ok"):
            err = engine_out.get("error") or {}
            return self._error(
                "LIVE_MODEL_INPUT_INVALID",
                str(err.get("message") or "engine rejected live vector"),
                sid,
                category="FEATURE_BUILD_ERROR",
            )

        result = {
            "ok": True,
            "mode": "LIVE",
            "status": "LIVE",
            "station_id": sid,
            "prediction_time_utc": prov["prediction_time_utc"],
            "data_source": prov["data_source"],
            "data_observation_time_utc": prov["data_observation_time_utc"],
            "forecast_valid_times_utc": prov["forecast_valid_times_utc"],
            "data_age_minutes": prov["data_age_minutes"],
            "predictions": {
                "1h": engine_out["lead_1h"],
                "2h": engine_out["lead_2h"],
                "3h": engine_out["lead_3h"],
            },
            "feature_validation": prov["feature_validation"],
            "signals": _signals_from_features(prov.get("features")),
            "cache_hit": cache_hit,
            "cache_age_seconds": 0.0,
        }
        for lead in ("1h", "2h", "3h"):
            block = result["predictions"][lead]
            if abs(float(block["threshold"]) - FROZEN_THRESHOLD) > 1e-12:
                return self._error("LIVE_MODEL_INPUT_INVALID", "threshold drift", sid)

        if use_cache:
            with self._lock:
                self._cache[sid] = (time.time(), dict(result))
        return result

    def _as_all_row(self, result: dict[str, Any]) -> dict[str, Any]:
        sid = result.get("station_id")
        if result.get("ok") and result.get("status") == "LIVE" and result.get("predictions"):
            return {
                "station_id": sid,
                "status": "LIVE",
                "predictions": result["predictions"],
                "observation_time_utc": result.get("data_observation_time_utc"),
                "prediction_time_utc": result.get("prediction_time_utc"),
                "forecast_valid_times_utc": result.get("forecast_valid_times_utc"),
                "data_age_minutes": result.get("data_age_minutes"),
                "source": result.get("data_source"),
                "signals": result.get("signals"),
                "feature_validation": result.get("feature_validation"),
                "cache_hit": result.get("cache_hit"),
                "error_code": None,
                "predictions_ok": True,
            }
        return {
            "station_id": sid,
            "status": "UNAVAILABLE",
            "error_code": (result.get("error") or {}).get("code") or result.get("status") or "LIVE_DATA_UNAVAILABLE",
            "predictions": None,
            "error": result.get("error"),
            "cache_hit": False,
            "predictions_ok": False,
            "_diag_category": result.get("_diag_category"),
            "_diag_provider": result.get("_diag_provider"),
        }

    def predict_live_all(
        self,
        station_ids: list[str] | None = None,
        *,
        now_utc: datetime | None = None,
        use_cache: bool = True,
        include_diagnostics: bool | None = None,
    ) -> dict[str, Any]:
        requested = [str(s).strip().upper() for s in (station_ids or LIVE_STATIONS)]
        if not requested:
            requested = list(LIVE_STATIONS)
        now = now_utc if now_utc is not None else datetime.now(timezone.utc)
        by_id: dict[str, dict[str, Any]] = {}

        def run_one(
            sid: str,
            station_payloads: tuple[dict[str, Any], dict[str, Any]] | None = None,
        ) -> tuple[str, dict[str, Any]]:
            return sid, self.predict_live(
                sid,
                now_utc=now,
                use_cache=use_cache,
                payloads=station_payloads,
            )

        if self.batch_fetch_fn is not None:
            locations: list[tuple[str, float, float]] = []
            for sid in requested:
                meta = self.catalog.get(sid)
                if not meta:
                    by_id[sid] = self._as_all_row(
                        self._error("LIVE_DATA_UNAVAILABLE", f"invalid station {sid!r}", sid)
                    )
                    continue
                locations.append((sid, float(meta["request_latitude"]), float(meta["request_longitude"])))
            batch_map: dict[str, tuple[dict[str, Any], dict[str, Any]]] | None = None
            batch_err: LiveDataError | None = None
            if locations:
                try:
                    batch_map = self.batch_fetch_fn(locations)
                except LiveDataError as exc:
                    batch_err = exc
            for sid in requested:
                if sid in by_id:
                    continue
                if batch_err is not None:
                    by_id[sid] = self._as_all_row(
                        self._error(
                            batch_err.code,
                            batch_err.message,
                            sid,
                            category=classify_live_exception(batch_err),
                            provider=sanitize_provider(getattr(batch_err, "provider", None)),
                        )
                    )
                    continue
                pair = (batch_map or {}).get(sid)
                if pair is None:
                    by_id[sid] = self._as_all_row(
                        self._error("LIVE_DATA_UNAVAILABLE", "missing batch payload", sid)
                    )
                    continue
                by_id[sid] = self._as_all_row(run_one(sid, pair)[1])
        else:
            workers = min(5, max(1, len(requested)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(run_one, sid) for sid in requested]
                for fut in as_completed(futs):
                    sid, raw = fut.result()
                    by_id[sid] = self._as_all_row(raw)

        rows = [by_id.get(sid) or self._as_all_row(self._error("LIVE_DATA_UNAVAILABLE", "missing", sid)) for sid in requested]
        live_n = sum(1 for r in rows if r.get("status") == "LIVE")
        diag_on = live_diagnostics_enabled() if include_diagnostics is None else bool(include_diagnostics)
        stations_diag = [
            station_diagnostic_entry(
                str(row.get("station_id") or ""),
                ok=row.get("status") == "LIVE",
                category=row.get("_diag_category"),
                provider=row.get("_diag_provider"),
            )
            for row in rows
        ]
        public_rows = []
        for row in rows:
            public = dict(row)
            public.pop("_diag_category", None)
            public.pop("_diag_provider", None)
            public_rows.append(public)
        out: dict[str, Any] = {
            "ok": live_n > 0,
            "mode": "LIVE",
            "status": "LIVE" if live_n == len(rows) else ("LIVE_PARTIAL" if live_n else "LIVE_DATA_UNAVAILABLE"),
            "requested_stations": len(requested),
            "completed_stations": len(rows),
            "available_stations": live_n,
            "results": public_rows,
        }
        if diag_on:
            out["diagnostics"] = {"stations": stations_diag}
        return out


def predict_live(station_id: Any, *, service: LivePredictionService | None = None) -> dict[str, Any]:
    svc = service if service is not None else LivePredictionService()
    return svc.predict_live(station_id)
