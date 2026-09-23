"""Read-only Phase 13A final multi-lead engine checks.

Does not train or write frozen model/data artifacts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.final_multi_lead_engine import FROZEN_THRESHOLD, FinalMultiLeadEngine

MODEL_PATHS = [
    ROOT / "models/v2/nwp_experiment/B_atmospheric_nwp_target_1h.joblib",
    ROOT / "models/v2/nwp_experiment/B_atmospheric_nwp_target_2h.joblib",
    ROOT / "models/v2/nwp_experiment/B_atmospheric_nwp_target_3h.joblib",
]
REPLAY_CSV = ROOT / "outputs" / "multimodal" / "phase12e_case_replay.csv"
NWP_FEATURES = (
    ROOT
    / "dataset"
    / "multilocation"
    / "features_nwp"
    / "multilocation_features_nwp_overlap_2021_2025.csv"
)
OUT_JSON = ROOT / "outputs" / "v2_inference" / "phase13a" / "final_multi_lead_engine_validation.json"
TOL = 1e-3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    checks: list[dict] = []
    hashes_before = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in MODEL_PATHS}

    engine = FinalMultiLeadEngine()
    models = engine.load_models()
    checks.append(
        {
            "name": "all_three_models_load",
            "passed": set(models) == {"1h", "2h", "3h"},
            "detail": sorted(models),
        }
    )
    checks.append(
        {
            "name": "feature_count_order",
            "passed": len(engine.feature_names) == 83,
            "detail": len(engine.feature_names),
        }
    )
    checks.append(
        {
            "name": "thresholds_0_065",
            "passed": all(abs(v - FROZEN_THRESHOLD) < 1e-12 for v in engine.thresholds.values()),
            "detail": engine.thresholds,
        }
    )

    replay = pd.read_csv(REPLAY_CSV, nrows=1)
    station = str(replay.iloc[0]["station_id"])
    ts = pd.to_datetime(replay.iloc[0]["timestamp_utc"], utc=True)
    expected_p1 = float(replay.iloc[0]["baseline_p_1h"])
    expected_p2 = float(replay.iloc[0]["baseline_p_2h"])
    expected_p3 = float(replay.iloc[0]["baseline_p_3h"])

    usecols = ["station_id", "timestamp_utc", *engine.feature_names]
    feats = pd.read_csv(NWP_FEATURES, usecols=usecols)
    feats["timestamp_utc"] = pd.to_datetime(feats["timestamp_utc"], utc=True)
    row = feats[(feats["station_id"] == station) & (feats["timestamp_utc"] == ts)]
    if row.empty:
        checks.append(
            {
                "name": "replay_row_found",
                "passed": False,
                "detail": f"no NWP-overlap row for {station} {ts}",
            }
        )
        payload = _finish(checks, hashes_before, False)
        print("FINAL_MULTI_LEAD_ENGINE_PREPARATION_REQUIRES_REVISION")
        return 1
    checks.append({"name": "replay_row_found", "passed": True, "detail": f"{station} {ts.isoformat()}"})

    mapping = {n: float(row.iloc[0][n]) for n in engine.feature_names}
    sat_ctx = {
        "sat_cloudy_fraction_25km": float(replay.iloc[0]["sat_cloudy_fraction_25km"]),
        "note": "evidence/context only; not a Model B predictor",
    }
    out = engine.predict(station, ts, mapping, satellite_context=sat_ctx)
    structured = (
        out.get("ok") is True
        and isinstance(out.get("lead_1h"), dict)
        and isinstance(out.get("lead_2h"), dict)
        and isinstance(out.get("lead_3h"), dict)
        and "probability" in out["lead_1h"]
        and "threshold" in out["lead_1h"]
        and "alert" in out["lead_1h"]
    )
    checks.append({"name": "structured_three_leads", "passed": structured, "detail": list(out.keys())})

    p1 = out["lead_1h"]["probability"] if structured else None
    p2 = out["lead_2h"]["probability"] if structured else None
    p3 = out["lead_3h"]["probability"] if structured else None
    match = (
        structured
        and abs(p1 - expected_p1) <= TOL
        and abs(p2 - expected_p2) <= TOL
        and abs(p3 - expected_p3) <= TOL
    )
    checks.append(
        {
            "name": "replay_probability_match",
            "passed": match,
            "detail": {
                "engine": {"1h": p1, "2h": p2, "3h": p3},
                "phase12e": {"1h": expected_p1, "2h": expected_p2, "3h": expected_p3},
                "tol": TOL,
            },
        }
    )
    checks.append(
        {
            "name": "satellite_not_in_core",
            "passed": out.get("satellite_context") == sat_ctx and "sat_cloudy_fraction_25km" not in mapping,
            "detail": "satellite attached as context only",
        }
    )

    missing = dict(mapping)
    missing.pop("nwp_cape")
    fail = engine.predict(station, ts, missing)
    checks.append(
        {
            "name": "fail_closed_missing_feature",
            "passed": fail.get("ok") is False
            and fail.get("data_status") == "UNAVAILABLE"
            and fail.get("lead_1h") is None
            and fail.get("error", {}).get("code") in {"MISSING_FEATURES", "MISSING_NWP_FIELDS"},
            "detail": fail.get("error"),
        }
    )

    bad_sub = dict(mapping)
    bad_sub["sat_cloudy_fraction_25km"] = 1.0
    sub = engine.predict(station, ts, bad_sub)
    checks.append(
        {
            "name": "fail_closed_satellite_substitution",
            "passed": sub.get("ok") is False
            and sub.get("error", {}).get("code") == "FORBIDDEN_FEATURE_SUBSTITUTION",
            "detail": sub.get("error"),
        }
    )

    hashes_after = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in MODEL_PATHS}
    checks.append(
        {
            "name": "frozen_artifacts_unmodified",
            "passed": hashes_before == hashes_after,
            "detail": "sha256 of three Model B joblibs unchanged",
        }
    )

    passed = all(c["passed"] for c in checks)
    _finish(checks, hashes_before, passed)
    if passed:
        print("PASS")
        return 0
    print("FAIL")
    for c in checks:
        if not c["passed"]:
            print(f"  {c['name']}: {c['detail']}")
    return 1


def _finish(checks: list[dict], hashes_before: dict, passed: bool) -> dict:
    payload = {
        "phase": "13A_final_multi_lead_engine",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["passed"]),
        "artifact_sha256_before": hashes_before,
        "checks": checks,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
