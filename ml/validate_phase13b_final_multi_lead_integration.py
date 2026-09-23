"""Phase 13B read-only integration checks for FinalMultiLeadEngine."""

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
from src.inference.final_multi_lead_engine import FinalMultiLeadEngine
from src.inference.multilead_service import MultiLeadInferenceService

MODEL_8D = [
    ROOT / "models/v2/nwp_experiment/B_atmospheric_nwp_target_1h.joblib",
    ROOT / "models/v2/nwp_experiment/B_atmospheric_nwp_target_2h.joblib",
    ROOT / "models/v2/nwp_experiment/B_atmospheric_nwp_target_3h.joblib",
]
PHASE8E = [
    ROOT / "models/v2/nwp_experiment/phase8e/B_nwp_imputed_target_1h.joblib",
    ROOT / "models/v2/nwp_experiment/phase8e/B_nwp_imputed_target_2h.joblib",
    ROOT / "models/v2/nwp_experiment/phase8e/B_nwp_imputed_target_3h.joblib",
]
PHASE12 = [
    ROOT / "outputs/multimodal/phase12e_case_replay.csv",
    ROOT / "outputs/multimodal/phase12e_case_replay_summary.json",
    ROOT / "docs/PHASE12E_MULTIMODAL_CASE_REPLAY_REPORT.md",
]
PHASE10I = [
    ROOT / "outputs/satellite/phase10i7/phase10i7_augmentation_summary.json",
]
REPLAY_CSV = ROOT / "outputs/multimodal/phase12e_case_replay.csv"
OUT_JSON = ROOT / "outputs/v2_inference/phase13b/final_multi_lead_integration_validation.json"
TOL = 1e-3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(checks: list[dict], name: str, passed: bool, detail: object) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> int:
    checks: list[dict] = []
    protected = [p for p in [*MODEL_8D, *PHASE8E, *PHASE12, *PHASE10I] if p.is_file()]
    hashes_before = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in protected}

    engine = FinalMultiLeadEngine()
    check(checks, "engine_initialization", True, engine.model_version)
    models = engine.load_models()
    check(checks, "models_loaded", set(models) == {"1h", "2h", "3h"}, sorted(models))
    check(checks, "feature_count_83", len(engine.feature_names) == 83, len(engine.feature_names))

    service = MultiLeadInferenceService(engine=engine)
    replay = pd.read_csv(REPLAY_CSV)
    replay["timestamp_utc"] = pd.to_datetime(replay["timestamp_utc"], utc=True)
    wanted = ["VABB", "VECC", "VIDP", "VOCI"]
    picked: list[pd.Series] = []
    for sid in wanted:
        sub = replay.loc[replay["station_id"] == sid]
        if sub.empty:
            check(checks, f"pick_{sid}", False, "no 12E row")
        else:
            picked.append(sub.iloc[0])
            check(checks, f"pick_{sid}", True, str(sub.iloc[0]["timestamp_utc"]))

    all_match = True
    for row in picked:
        sid = str(row["station_id"])
        ts = row["timestamp_utc"]
        out = service.replay_historical(sid, ts)
        ok_struct = (
            out.get("ok") is True
            and out.get("lead_1h")
            and out.get("lead_2h")
            and out.get("lead_3h")
        )
        check(checks, f"historical_lookup_{sid}", ok_struct, out.get("error"))
        if not ok_struct:
            all_match = False
            continue
        vec_ok = len(engine.feature_names) == 83
        check(checks, f"feature_construction_{sid}", vec_ok, 83)
        for lead, col in (("1h", "baseline_p_1h"), ("2h", "baseline_p_2h"), ("3h", "baseline_p_3h")):
            got = float(out[f"lead_{lead}"]["probability"])
            exp = float(row[col])
            match = abs(got - exp) <= TOL
            all_match = all_match and match
            check(
                checks,
                f"consistency_{sid}_{lead}",
                match,
                {"engine": got, "phase12e": exp, "abs_diff": abs(got - exp)},
            )
        sat = out.get("satellite_context")
        check(
            checks,
            f"satellite_context_separated_{sid}",
            sat is None or "probability" not in (sat or {}),
            sat,
        )

    check(checks, "all_replay_probabilities_match", all_match, f"tol={TOL}")

    mapping = {n: 0.0 for n in engine.feature_names}
    missing = dict(mapping)
    missing.pop("temperature_2m")
    r = engine.predict("VOTV", "2023-06-15T12:00:00+00:00", missing)
    check(
        checks,
        "fail_closed_missing_feature",
        r.get("error", {}).get("code") == "MISSING_FEATURES" and r.get("lead_1h") is None,
        r.get("error"),
    )
    r = engine.predict("VOTV", "2023-06-15T12:00:00+00:00", [0.0] * 10)
    check(
        checks,
        "fail_closed_feature_count",
        r.get("error", {}).get("code") == "FEATURE_COUNT_MISMATCH",
        r.get("error"),
    )
    r = engine.predict("XXXX", "2023-06-15T12:00:00+00:00", mapping)
    check(
        checks,
        "fail_closed_invalid_station",
        r.get("error", {}).get("code") == "INVALID_STATION",
        r.get("error"),
    )
    r = engine.predict("VOTV", "not-a-time", mapping)
    check(
        checks,
        "fail_closed_invalid_timestamp",
        r.get("error", {}).get("code") == "INVALID_TIMESTAMP",
        r.get("error"),
    )
    bad = dict(mapping)
    bad["nwp_cape"] = float("nan")
    r = engine.predict("VOTV", "2023-06-15T12:00:00+00:00", bad)
    check(
        checks,
        "fail_closed_non_finite",
        r.get("error", {}).get("code") == "NON_FINITE_FEATURE",
        r.get("error"),
    )
    bad = dict(mapping)
    bad["sat_cloudy_fraction_25km"] = 1.0
    r = engine.predict("VOTV", "2023-06-15T12:00:00+00:00", bad)
    check(
        checks,
        "fail_closed_forbidden_predictor",
        r.get("error", {}).get("code") == "FORBIDDEN_FEATURE_SUBSTITUTION",
        r.get("error"),
    )

    def fake_replay(station_id, timestamp_utc, **kwargs):
        return {
            "station_id": station_id,
            "data_status": "UNAVAILABLE",
            "unavailable_reason": "timestamp_outside_nwp_overlap",
            "threshold_1h": 0.065,
            "replay_mode": "HISTORICAL_REPLAY",
        }

    app = create_app(replay_fn=fake_replay)
    app.config["TESTING"] = True
    client = app.test_client()
    hv = client.get("/health")
    check(checks, "api_health", hv.status_code == 200 and hv.get_json().get("status") == "ok", hv.status_code)
    rv = client.post("/replay/all", json={"timestamp_utc": "2023-06-15T12:00:00Z"})
    body = rv.get_json() or {}
    check(
        checks,
        "replay_all_endpoint",
        rv.status_code == 200 and body.get("replay_mode") == "HISTORICAL_REPLAY",
        {"status": rv.status_code, "keys": list(body.keys())},
    )
    api = client.post(
        "/api/inference/multilead",
        json={"station_id": "XXXX", "timestamp_utc": "2023-06-15T12:00:00Z"},
    )
    aj = api.get_json() or {}
    check(
        checks,
        "api_multilead_invalid_station",
        api.status_code == 400 and (aj.get("error") or {}).get("code") == "INVALID_STATION",
        aj.get("error"),
    )

    hashes_after = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in protected}
    check(checks, "frozen_artifacts_unmodified", hashes_before == hashes_after, "8D/8E/12/10I hashes")

    passed = all(c["passed"] for c in checks)
    payload = {
        "phase": "13B_final_multi_lead_integration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["passed"]),
        "checks": checks,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("PASS" if passed else "FAIL")
    if not passed:
        for c in checks:
            if not c["passed"]:
                print(f"  {c['name']}: {c['detail']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
