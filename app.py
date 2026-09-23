"""Phase 16B: ThunderWatch AI product UI over HISTORICAL REPLAY.

Presentation layer only. Predictions come from run_historical_replay.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.inference.live_prediction_service import LivePredictionService
from src.inference.multilead_service import MultiLeadInferenceService, predict_multilead
from src.replay.v2_historical_replay import (
    DEFAULT_DATASET,
    KNOWN_STATIONS,
    run_historical_replay,
)
from src.spatial.historical_spatial_replay import HistoricalSpatialReplay

BASE_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv  # optional local .env
    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass

STATION_LABELS = {
    "VOTV": "Thiruvananthapuram",
    "VECC": "Kolkata",
    "VIDP": "Delhi",
    "VOCI": "Kochi",
    "VABB": "Mumbai",
}

DEMO_CASES = [
    {"id": "1", "label": "Demo Case 1", "timestamp_utc": "2023-06-15T12:00:00Z"},
    {"id": "2", "label": "Demo Case 2", "timestamp_utc": "2023-06-15T06:00:00Z"},
    {"id": "3", "label": "Demo Case 3", "timestamp_utc": "2023-06-15T18:00:00Z"},
]

DEFAULT_TS = "2023-06-15T12:00:00Z"

REASON_MESSAGES = {
    "unknown_station": "Unsupported station",
    "invalid_timestamp": "Invalid timestamp",
    "timestamp_outside_nwp_overlap": "Timestamp outside validated overlap",
    "insufficient_history": "Insufficient causal history",
    "missing_nwp": "Missing required NWP input",
    "nwp_incomplete": "Missing required NWP input",
    "feature_build_unavailable": "Insufficient causal history",
}

ATMO_FIELDS = [
    ("temperature_2m", "Temperature", "°C"),
    ("relative_humidity_2m", "Humidity", "%"),
    ("surface_pressure", "Pressure", "Pa"),
    ("wind_speed_10m", "Wind", "m/s"),
    ("precipitation", "Precipitation", "mm"),
    ("cloud_cover", "Cloud cover", "%"),
]

logger = logging.getLogger(__name__)
ReplayFn = Callable[..., dict[str, Any]]
_ATMO_FRAME: pd.DataFrame | None = None


def _friendly_reason(raw: str | None) -> str:
    if not raw:
        return "Required historical inputs were unavailable for this timestamp."
    key = str(raw).strip()
    if key in REASON_MESSAGES:
        return REASON_MESSAGES[key]
    lower = key.lower()
    if "nwp" in lower:
        return "Missing required NWP input"
    if "history" in lower or "causal" in lower or "insufficient" in lower:
        return "Insufficient causal history"
    if "invalid_timestamp" in lower:
        return "Invalid timestamp"
    if "overlap" in lower or "outside" in lower:
        return "Timestamp outside validated overlap"
    return "Required historical inputs were unavailable for this timestamp."


def _normalize_timestamp(raw: str) -> str | None:
    text = (raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        return text
    if len(text) == 16 and text[10] == "T":
        return text + ":00Z"
    if len(text) == 19 and text[10] == "T":
        return text + "Z"
    return text


def _pct(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value) * 100.0:.1f}"
    except (TypeError, ValueError):
        return None


def _thr_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return "6.5%"


def _alert_label(flag: Any) -> str | None:
    if flag is None:
        return None
    return "ALERT" if int(flag) == 1 else "NO ALERT"


def _hit_label(flag: Any) -> str | None:
    if flag is None:
        return None
    return "MATCH" if int(flag) == 1 else "MISS"


def _target_label(flag: Any) -> str:
    if flag is None:
        return "unavailable"
    return "Thunderstorm observed" if int(flag) == 1 else "No thunderstorm observed"


def _prob_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(x):
        return None
    return x


def _threshold_state(prob: float | None, thr: Any) -> str | None:
    if prob is None:
        return None
    try:
        t = float(thr)
    except (TypeError, ValueError):
        t = 0.065
    return "ABOVE THRESHOLD" if prob >= t else "BELOW THRESHOLD"


def load_atmosphere_frame() -> pd.DataFrame | None:
    global _ATMO_FRAME
    if _ATMO_FRAME is not None:
        return _ATMO_FRAME
    path = DEFAULT_DATASET
    if not path.is_file():
        return None
    cols = ["station_id", "timestamp_utc", *[c for c, _, _ in ATMO_FIELDS]]
    try:
        _ATMO_FRAME = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    except Exception:
        logger.exception("could not load atmospheric snapshot columns")
        return None
    return _ATMO_FRAME


def _na_atmosphere() -> list[dict[str, str]]:
    return [{"label": label, "value": "N/A", "unit": ""} for _, label, _ in ATMO_FIELDS]


def atmosphere_for(station_id: str, timestamp_utc: str) -> list[dict[str, str]]:
    frame = load_atmosphere_frame()
    if frame is None or frame.empty:
        return _na_atmosphere()
    t = pd.to_datetime(timestamp_utc, utc=True, errors="coerce")
    if pd.isna(t):
        return _na_atmosphere()
    ts = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    mask = (frame["station_id"].astype(str) == station_id) & (ts == t)
    rows = frame.loc[mask]
    if rows.empty:
        return _na_atmosphere()
    row = rows.iloc[0]
    out: list[dict[str, str]] = []
    for col, label, unit in ATMO_FIELDS:
        if col not in row.index:
            out.append({"label": label, "value": "N/A", "unit": ""})
            continue
        val = row[col]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            out.append({"label": label, "value": "N/A", "unit": ""})
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            out.append({"label": label, "value": "N/A", "unit": ""})
            continue
        if pd.isna(num):
            out.append({"label": label, "value": "N/A", "unit": ""})
            continue
        disp_unit = unit
        if col == "surface_pressure":
            if num > 2000:
                num = num / 100.0
            disp_unit = "hPa"
        fmt = f"{num:.1f}" if abs(num) < 1000 else f"{num:.0f}"
        out.append({"label": label, "value": fmt, "unit": disp_unit})
    return out


def _serialize_station(result: dict[str, Any], timestamp_utc: str) -> dict[str, Any]:
    sid = str(result.get("station_id") or "")
    unavailable = result.get("data_status") != "COMPLETE"
    p1 = _prob_float(result.get("lead_1h_probability"))
    p2 = _prob_float(result.get("lead_2h_probability"))
    p3 = _prob_float(result.get("lead_3h_probability"))
    thr = result.get("threshold_1h")
    atmo = [] if unavailable else atmosphere_for(sid, timestamp_utc)
    return {
        "station_id": sid,
        "station_name": STATION_LABELS.get(sid, sid),
        "replay_mode": result.get("replay_mode"),
        "model_version": result.get("model_version"),
        "data_status": result.get("data_status"),
        "data_completeness": result.get("data_completeness"),
        "unavailable": unavailable,
        "unavailable_reason": _friendly_reason(result.get("unavailable_reason")) if unavailable else None,
        "lead_1h_probability": None if unavailable else p1,
        "lead_2h_probability": None if unavailable else p2,
        "lead_3h_probability": None if unavailable else p3,
        "lead_1h_pct": None if unavailable else _pct(p1),
        "lead_2h_pct": None if unavailable else _pct(p2),
        "lead_3h_pct": None if unavailable else _pct(p3),
        "lead_1h_alert": None if unavailable else result.get("lead_1h_alert"),
        "lead_2h_alert": None if unavailable else result.get("lead_2h_alert"),
        "lead_3h_alert": None if unavailable else result.get("lead_3h_alert"),
        "alert_1h": None if unavailable else _alert_label(result.get("lead_1h_alert")),
        "threshold": thr,
        "threshold_pct": _thr_pct(thr),
        "threshold_state": None if unavailable else _threshold_state(p1, thr),
        "historical_target_1h": result.get("historical_target_1h"),
        "historical_target_2h": result.get("historical_target_2h"),
        "historical_target_3h": result.get("historical_target_3h"),
        "lead_1h_alert_hit": result.get("lead_1h_alert_hit"),
        "lead_2h_alert_hit": result.get("lead_2h_alert_hit"),
        "lead_3h_alert_hit": result.get("lead_3h_alert_hit"),
        "target_1h_text": _target_label(result.get("historical_target_1h")),
        "target_2h_text": _target_label(result.get("historical_target_2h")),
        "target_3h_text": _target_label(result.get("historical_target_3h")),
        "hit_1h": _hit_label(result.get("lead_1h_alert_hit")),
        "hit_2h": _hit_label(result.get("lead_2h_alert_hit")),
        "hit_3h": _hit_label(result.get("lead_3h_alert_hit")),
        "atmosphere": atmo,
    }


def rank_stations(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by descending 1h model probability; UNAVAILABLE last; ICAO tie-break."""

    order = {sid: i for i, sid in enumerate(KNOWN_STATIONS)}

    def key(item: dict[str, Any]) -> tuple:
        p = item.get("lead_1h_probability")
        sid = str(item.get("station_id") or "")
        if p is None:
            return (0, 0.0, -order.get(sid, 99))
        return (1, float(p), -order.get(sid, 99))

    return sorted(payloads, key=key, reverse=True)


def run_all_stations(runner: ReplayFn, timestamp_utc: str) -> dict[str, Any]:
    raw: list[dict[str, Any]] = []
    attempted = []
    for sid in KNOWN_STATIONS:
        attempted.append(sid)
        try:
            result = runner(sid, timestamp_utc)
        except Exception:
            logger.exception("replay failed for %s", sid)
            result = {
                "station_id": sid,
                "data_status": "UNAVAILABLE",
                "unavailable_reason": "Replay could not be completed.",
                "threshold_1h": 0.065,
                "replay_mode": "HISTORICAL_REPLAY",
            }
        raw.append(_serialize_station(result, timestamp_utc))
    ranked = rank_stations(raw)
    available = [x for x in ranked if not x["unavailable"]]
    focus = available[0] if available else None
    return {
        "timestamp_utc": timestamp_utc,
        "replay_mode": "HISTORICAL_REPLAY",
        "stations_attempted": attempted,
        "stations": ranked,
        "focus_station_id": None if focus is None else focus["station_id"],
        "all_unavailable": focus is None,
        "sort_rule": "HIGHER 1H MODEL PROBABILITY",
    }


def _build_display(result: dict[str, Any]) -> dict[str, Any]:
    status = result.get("data_status")
    unavailable = status != "COMPLETE"
    leads = []
    for lead in (1, 2, 3):
        prob = result.get(f"lead_{lead}h_probability")
        thr = result.get(f"threshold_{lead}h")
        alert = result.get(f"lead_{lead}h_alert")
        target = result.get(f"historical_target_{lead}h")
        hit = result.get(f"lead_{lead}h_alert_hit")
        leads.append(
            {
                "lead": f"{lead}h",
                "title": f"{lead}-HOUR RISK",
                "probability_pct": None if unavailable else _pct(prob),
                "threshold_pct": _thr_pct(thr),
                "status": None if unavailable else _alert_label(alert),
                "alert_text": _alert_label(alert) or "—",
                "target_text": _target_label(target),
                "hit": _hit_label(hit),
            }
        )
    reason = None
    if unavailable:
        reason = _friendly_reason(result.get("unavailable_reason"))
    return {
        "unavailable": unavailable,
        "reason": reason,
        "leads": leads,
        "station_name": STATION_LABELS.get(str(result.get("station_id") or ""), ""),
    }


def map_basemap_config() -> dict[str, Any]:
    """MapTiler style URLs from env. Never hardcode secrets."""
    key = (os.environ.get("MAPTILER_API_KEY") or "").strip()
    if not key:
        return {
            "enabled": False,
            "provider": "maptiler",
            "reason": "MAPTILER_API_KEY not set — using fallback geographic view",
        }
    return {
        "enabled": True,
        "provider": "maptiler",
        "styles": {
            "dark": f"https://api.maptiler.com/maps/dataviz-dark/style.json?key={key}",
            "terrain": f"https://api.maptiler.com/maps/outdoor-v2/style.json?key={key}",
            "hybrid": f"https://api.maptiler.com/maps/hybrid-v4/style.json?key={key}",
        },
        "terrain": f"https://api.maptiler.com/tiles/terrain-rgb-v2/tiles.json?key={key}",
    }


def create_app(*, replay_fn: ReplayFn | None = None, live_service: LivePredictionService | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    runner = replay_fn or run_historical_replay
    multilead_service = MultiLeadInferenceService()
    live_service = live_service if live_service is not None else LivePredictionService()
    spatial_service = HistoricalSpatialReplay()

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/")
    def index():
        return render_template(
            "replay.html",
            stations=KNOWN_STATIONS,
            station_labels=STATION_LABELS,
            demo_cases=DEMO_CASES,
            form_station="VOTV",
            form_timestamp="2023-06-15T12:00",
            default_ts=DEFAULT_TS,
            error=None,
            result=None,
            display=None,
        )

    @app.post("/replay")
    def replay():
        station_id = (request.form.get("station_id") or "").strip().upper()
        ts_raw = (request.form.get("timestamp_utc") or "").strip()
        form_ts = ts_raw.replace("Z", "")[:16] if ts_raw else ""

        if station_id not in KNOWN_STATIONS:
            return render_template(
                "replay.html",
                stations=KNOWN_STATIONS,
                station_labels=STATION_LABELS,
                demo_cases=DEMO_CASES,
                form_station=station_id,
                form_timestamp=form_ts,
                default_ts=DEFAULT_TS,
                error="Please choose one of the five supported stations.",
                result=None,
                display=None,
            ), 400

        timestamp_utc = _normalize_timestamp(ts_raw)
        if timestamp_utc is None:
            return render_template(
                "replay.html",
                stations=KNOWN_STATIONS,
                station_labels=STATION_LABELS,
                demo_cases=DEMO_CASES,
                form_station=station_id,
                form_timestamp=form_ts,
                default_ts=DEFAULT_TS,
                error="Please enter a valid prediction time T (UTC).",
                result=None,
                display=None,
            ), 400

        try:
            result = runner(station_id, timestamp_utc)
        except Exception:
            logger.exception("historical replay failed")
            return render_template(
                "replay.html",
                stations=KNOWN_STATIONS,
                station_labels=STATION_LABELS,
                demo_cases=DEMO_CASES,
                form_station=station_id,
                form_timestamp=form_ts,
                default_ts=DEFAULT_TS,
                error="Replay could not be completed.",
                result=None,
                display=None,
            ), 500

        display = _build_display(result)
        return render_template(
            "replay.html",
            stations=KNOWN_STATIONS,
            station_labels=STATION_LABELS,
            demo_cases=DEMO_CASES,
            form_station=station_id,
            form_timestamp=form_ts,
            default_ts=DEFAULT_TS,
            error=None,
            result=result,
            display=display,
        )

    @app.post("/replay/all")
    def replay_all():
        payload = request.get_json(silent=True) or {}
        ts_raw = (
            request.form.get("timestamp_utc")
            or payload.get("timestamp_utc")
            or ""
        )
        timestamp_utc = _normalize_timestamp(str(ts_raw))
        if timestamp_utc is None:
            return jsonify({"error": "Please enter a valid historical replay time (UTC)."}), 400
        data = run_all_stations(runner, timestamp_utc)
        return jsonify(data)

    @app.get("/api/map-config")
    def map_config():
        return jsonify(map_basemap_config())

    @app.post("/api/inference/multilead")
    def inference_multilead():
        """Historical/replay multi-lead scores from frozen Phase 8D Model B.

        Not a live operational forecast API. Not satellite-enhanced.
        """
        payload = request.get_json(silent=True) or {}
        station_id = payload.get("station_id")
        timestamp_utc = payload.get("timestamp_utc")
        inference_mode = payload.get("inference_mode") or "HISTORICAL_REPLAY"
        evidence = payload.get("evidence_context") or {}
        features = payload.get("features")
        result = predict_multilead(
            station_id,
            timestamp_utc,
            features=features,
            inference_mode=str(inference_mode),
            evidence_context=evidence,
            service=multilead_service,
        )
        status = 200 if result.get("ok") else 400
        return jsonify(result), status

    @app.post("/api/inference/live")
    def inference_live():
        """Live Open-Meteo Forecast/GFS → frozen Model B. Not historical replay."""
        payload = request.get_json(silent=True) or {}
        station_id = payload.get("station_id")
        result = live_service.predict_live(station_id)
        status = 200 if result.get("ok") else 400
        return jsonify(result), status

    @app.post("/api/inference/live/all")
    def inference_live_all():
        """Concurrent five-station live inference. Not historical replay."""
        payload = request.get_json(silent=True) or {}
        stations = payload.get("stations")
        result = live_service.predict_live_all(stations)
        return jsonify(result), 200

    @app.post("/api/spatial/replay")
    def spatial_replay():
        """Historical multi-station risk markers. Not a storm-cell forecast."""
        payload = request.get_json(silent=True) or {}
        timestamp_utc = payload.get("timestamp_utc")
        result = spatial_service.get_multi_station_snapshot(timestamp_utc)
        status = 200 if result.get("ok") else 400
        return jsonify(result), status

    @app.get("/api/spatial/replay/timestamps")
    def spatial_replay_timestamps():
        stamps = spatial_service.list_available_spatial_timestamps()
        return jsonify({"ok": True, "count": len(stamps), "timestamps": stamps})

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=5000, debug=False)
