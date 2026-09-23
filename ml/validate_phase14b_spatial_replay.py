"""Phase 14B validation: historical spatial replay (read-only of frozen artifacts)."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import create_app
from src.spatial.historical_spatial_replay import (
    EXPECTED_STATIONS,
    THRESHOLD,
    HistoricalSpatialReplay,
)

VALIDATION_TS = "2021-04-02T00:00:00Z"
TOL = 1e-6
OUT_JSON = ROOT / "outputs" / "spatial" / "phase14b" / "spatial_replay_validation.json"
REPLAY_CSV = ROOT / "outputs" / "v2_replay" / "phase15b" / "replay_predictions.csv"
METADATA = ROOT / "dataset" / "multilocation" / "features" / "multilocation_features_2014_2025_metadata.json"
SERVICE_SRC = ROOT / "src" / "spatial" / "historical_spatial_replay.py"
AUDIT_14A = ROOT / "docs" / "PHASE14A_SPATIAL_RISK_CAPABILITY_AUDIT.md"
AUDIT_JSON = ROOT / "outputs" / "spatial" / "phase14a" / "spatial_capability_audit.json"

FORBIDDEN_INTERP_IMPL = (
    "scipy.interpolate",
    "inverse_distance",
    "idw_interpolat",
    "kriging",
    "griddata(",
    "gaussian_filter",
    "heatmap",
    "pcolormesh",
    "contourf",
    "rbfinterpolator",
)

PHASE8D = [
    ROOT / "models/v2/nwp_experiment/B_atmospheric_nwp_target_1h.joblib",
    ROOT / "models/v2/nwp_experiment/B_atmospheric_nwp_target_2h.joblib",
    ROOT / "models/v2/nwp_experiment/B_atmospheric_nwp_target_3h.joblib",
]
PHASE8E = [
    ROOT / "models/v2/nwp_experiment/phase8e/B_nwp_imputed_target_1h.joblib",
    ROOT / "models/v2/nwp_experiment/phase8e/B_nwp_imputed_target_2h.joblib",
    ROOT / "models/v2/nwp_experiment/phase8e/B_nwp_imputed_target_3h.joblib",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    protected = [p for p in [*PHASE8D, *PHASE8E, AUDIT_14A, AUDIT_JSON] if p.is_file()]
    hashes_before = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in protected}

    svc = HistoricalSpatialReplay()
    coords = svc.load_station_coordinates()
    meta = json.loads(METADATA.read_text(encoding="utf-8"))["station_coordinates"]

    checks: list[dict] = []

    station_ok = list(coords.keys()) == list(EXPECTED_STATIONS) and len(coords) == 5
    checks.append({"name": "A_station_metadata", "passed": station_ok, "detail": list(coords.keys())})

    coord_ok = True
    coord_detail = {}
    for sid in EXPECTED_STATIONS:
        got = coords[sid]
        exp = meta[sid]
        match = (
            abs(got["latitude"] - float(exp["latitude"])) < 1e-9
            and abs(got["longitude"] - float(exp["longitude"])) < 1e-9
        )
        coord_ok = coord_ok and match
        coord_detail[sid] = got
    checks.append({"name": "B_coordinate_join", "passed": coord_ok, "detail": coord_detail})

    snap = svc.get_multi_station_snapshot(VALIDATION_TS)
    stations = {s["station_id"]: s for s in snap.get("stations", [])}
    five_complete = (
        snap.get("ok") is True
        and snap.get("mode") == "HISTORICAL_REPLAY"
        and len(stations) == 5
        and all(stations[s]["data_status"] == "COMPLETE" for s in EXPECTED_STATIONS)
    )
    checks.append({"name": "C_five_station_snapshot", "passed": five_complete, "detail": snap.get("timestamp_utc")})

    replay = pd.read_csv(
        REPLAY_CSV,
        usecols=[
            "station_id",
            "prediction_timestamp_utc",
            "data_status",
            "lead_1h_probability",
            "lead_2h_probability",
            "lead_3h_probability",
        ],
    )
    replay["prediction_timestamp_utc"] = pd.to_datetime(replay["prediction_timestamp_utc"], utc=True)
    ts = pd.Timestamp(VALIDATION_TS)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ref = replay[
        (replay["prediction_timestamp_utc"] == ts)
        & (replay["data_status"] == "COMPLETE")
    ]
    prob_ok = True
    prob_results = {}
    for _, row in ref.iterrows():
        sid = str(row["station_id"]).upper()
        s = stations[sid]
        for lead, col in (
            ("1h", "lead_1h_probability"),
            ("2h", "lead_2h_probability"),
            ("3h", "lead_3h_probability"),
        ):
            delta = abs(float(s[col]) - float(row[col]))
            ok = delta <= TOL
            prob_ok = prob_ok and ok
            prob_results[f"{sid}_{lead}"] = {"delta": delta, "ok": ok, "got": s[col], "csv": float(row[col])}
    checks.append({"name": "D_probability_consistency", "passed": prob_ok, "detail": prob_results})

    thr_ok = True
    thr_detail = {}
    for sid, s in stations.items():
        for lead in ("1h", "2h", "3h"):
            p = s[f"lead_{lead}_probability"]
            a = s[f"alert_{lead}"]
            expected = bool(p >= THRESHOLD)
            ok = a is expected
            thr_ok = thr_ok and ok
            thr_detail[f"{sid}_{lead}"] = {"probability": p, "alert": a, "expected": expected}
    checks.append({"name": "E_threshold", "passed": thr_ok, "detail": thr_detail})

    src = SERVICE_SRC.read_text(encoding="utf-8").lower()
    hits = [w for w in FORBIDDEN_INTERP_IMPL if w in src]
    interp_ok = not hits
    checks.append({"name": "F_no_interpolation", "passed": interp_ok, "detail": hits})

    usecols_ok = "target_" not in SERVICE_SRC.read_text(encoding="utf-8").split("usecols")[1][:800]
    leakage = all(t not in svc._load_replay().columns for t in ("target_1h", "target_2h", "target_3h"))
    checks.append({"name": "G_no_target_leakage", "passed": leakage and usecols_ok, "detail": {"replay_has_targets": not leakage}})

    empty_sat = HistoricalSpatialReplay(
        satellite_path=ROOT / "outputs" / "spatial" / "phase14b" / "_no_sat.csv"
    )
    empty_sat._coords = coords
    empty_sat._replay = svc._load_replay()
    empty_sat._sat = pd.DataFrame()
    snap_nosat = empty_sat.get_multi_station_snapshot(VALIDATION_TS)
    sep_ok = True
    for sid in EXPECTED_STATIONS:
        a = stations[sid]
        b = {x["station_id"]: x for x in snap_nosat["stations"]}[sid]
        for k in ("lead_1h_probability", "lead_2h_probability", "lead_3h_probability"):
            sep_ok = sep_ok and a[k] == b[k]
    checks.append(
        {
            "name": "H_satellite_separation",
            "passed": sep_ok,
            "detail": {"satellite_does_not_alter_probabilities": sep_ok},
        }
    )

    radar_ok = snap.get("radar_context") is None
    lightning_ok = snap.get("lightning_context") is None
    checks.append({"name": "I_radar_null", "passed": radar_ok, "detail": snap.get("radar_context")})
    checks.append({"name": "J_lightning_null", "passed": lightning_ok, "detail": snap.get("lightning_context")})

    missing = svc.get_multi_station_snapshot("1999-01-01T00:00:00Z")
    missing_ok = (
        missing.get("ok") is True
        and all(s["data_status"] == "UNAVAILABLE" for s in missing["stations"])
        and all(s["lead_1h_probability"] is None for s in missing["stations"])
    )
    invalid = svc.get_multi_station_snapshot("not-a-timestamp")
    invalid_ok = (
        invalid.get("ok") is False
        and (invalid.get("error") or {}).get("code") == "INVALID_TIMESTAMP"
        and invalid.get("stations") == []
    )
    checks.append({"name": "K_unavailable_and_invalid", "passed": missing_ok and invalid_ok, "detail": {"missing": missing_ok, "invalid": invalid_ok}})

    stamps = svc.list_available_spatial_timestamps()
    all_five_count = len(stamps)
    checks.append(
        {
            "name": "all_five_timestamp_count",
            "passed": all_five_count > 0 and VALIDATION_TS in stamps,
            "detail": all_five_count,
        }
    )

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    h = client.get("/health")
    root = client.get("/")
    post_all = client.post("/replay/all", json={"timestamp_utc": "2023-06-15T12:00:00Z"})
    ml = client.post(
        "/api/inference/multilead",
        json={"station_id": "VABB", "timestamp_utc": "2021-04-02T00:00:00Z"},
    )
    spatial = client.post("/api/spatial/replay", json={"timestamp_utc": VALIDATION_TS})
    sj = spatial.get_json() or {}
    spatial_ok = (
        spatial.status_code == 200
        and sj.get("ok") is True
        and len(sj.get("stations") or []) == 5
        and sj.get("radar_context") is None
        and sj.get("lightning_context") is None
    )
    ts_api = client.get("/api/spatial/replay/timestamps")
    tj = ts_api.get_json() or {}
    ts_ok = ts_api.status_code == 200 and tj.get("ok") is True and int(tj.get("count") or 0) == all_five_count
    endpoint_ok = (
        h.status_code == 200
        and root.status_code == 200
        and post_all.status_code == 200
        and ml.status_code in {200, 400}
        and spatial_ok
        and ts_ok
    )
    checks.append(
        {
            "name": "L_api",
            "passed": endpoint_ok,
            "detail": {
                "health": h.status_code,
                "index": root.status_code,
                "replay_all": post_all.status_code,
                "multilead": ml.status_code,
                "spatial": spatial.status_code,
                "timestamps": ts_api.status_code,
                "timestamp_count": tj.get("count"),
            },
        }
    )

    hashes_after = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in protected}
    immutable = hashes_before == hashes_after
    checks.append({"name": "frozen_artifacts_unchanged", "passed": immutable, "detail": immutable})

    passed = all(c["passed"] for c in checks)
    status = "SPATIAL_HISTORICAL_REPLAY_READY" if passed else "SPATIAL_HISTORICAL_REPLAY_REQUIRES_REVISION"
    payload = {
        "phase": "14B",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "station_count": 5,
        "station_ids": list(EXPECTED_STATIONS),
        "coordinate_validation": coord_detail,
        "all_station_timestamp_count": all_five_count,
        "validation_timestamp": VALIDATION_TS,
        "probability_match_results": prob_results,
        "threshold_results": thr_detail,
        "interpolation_check": {"passed": interp_ok, "hits": hits},
        "target_leakage_check": leakage,
        "satellite_separation_check": sep_ok,
        "radar_context": None,
        "lightning_context": None,
        "endpoint_tests": checks[-2]["detail"] if checks[-2]["name"] == "L_api" else {},
        "checks": checks,
        "frozen_artifacts_unchanged": immutable,
        "final_status": status,
        "disclaimer": (
            "This spatial layer represents discrete AI risk at observed station locations. "
            "It does not interpolate risk between stations and does not constitute a storm-cell forecast."
        ),
    }
    # endpoint_tests from L
    for c in checks:
        if c["name"] == "L_api":
            payload["endpoint_tests"] = c["detail"]
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(status)
    for c in checks:
        print(f"{'PASS' if c['passed'] else 'FAIL'} {c['name']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
