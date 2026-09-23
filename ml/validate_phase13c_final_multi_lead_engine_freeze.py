"""Phase 13C: freeze validation for FinalMultiLeadEngine (read-only)."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import create_app
from src.inference.final_multi_lead_engine import FROZEN_THRESHOLD, FinalMultiLeadEngine
from src.inference.multilead_service import MultiLeadInferenceService

PHASE13A_HASHES = {
    "models/v2/nwp_experiment/B_atmospheric_nwp_target_1h.joblib": (
        "a266c71bf901c4ea9392882b619e1a40905279af89db58123227a07f49733f21"
    ),
    "models/v2/nwp_experiment/B_atmospheric_nwp_target_2h.joblib": (
        "e99ff3e9fba4cef8a3e1d15e6c29cfdbee801200369230bd652a45a71b376c6c"
    ),
    "models/v2/nwp_experiment/B_atmospheric_nwp_target_3h.joblib": (
        "20692e504594ad17ea6fcc43dfe6e8899a5d5bdc2df93015df92327b97dc16ed"
    ),
}
MODEL_PATHS = [ROOT / rel for rel in PHASE13A_HASHES]
CONTRACT = ROOT / "outputs/v2_inference/phase13a/inference_contract.json"
REPLAY_CSV = ROOT / "outputs/multimodal/phase12e_case_replay.csv"
OUT_JSON = ROOT / "outputs/v2_inference/phase13c/final_multi_lead_engine_freeze_validation.json"
TOL = 1e-3
FORBIDDEN = (
    "annual_thunder_hours",
    "satellite",
    "radar",
    "lightning",
    "lis_",
    "thunder_hour",
    "weather_code",
    "target_",
    "label",
)
FUTURE = ("t+1", "t+2", "t+3", "lead_1h", "lead_2h", "lead_3h", "future_")
PROTECTED = [
    *MODEL_PATHS,
    ROOT / "models/v2/nwp_experiment/phase8e/B_nwp_imputed_target_1h.joblib",
    ROOT / "models/v2/nwp_experiment/phase8e/B_nwp_imputed_target_2h.joblib",
    ROOT / "models/v2/nwp_experiment/phase8e/B_nwp_imputed_target_3h.joblib",
    ROOT / "outputs/multimodal/phase12e_case_replay.csv",
    ROOT / "outputs/multimodal/phase12e_case_replay_summary.json",
    ROOT / "docs/PHASE12E_MULTIMODAL_CASE_REPLAY_REPORT.md",
    ROOT / "outputs/satellite/phase10i7/phase10i7_augmentation_summary.json",
    CONTRACT,
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail_ok(out: dict, code: str) -> bool:
    return (
        out.get("ok") is False
        and out.get("data_status") == "UNAVAILABLE"
        and out.get("lead_1h") is None
        and out.get("lead_2h") is None
        and out.get("lead_3h") is None
        and (out.get("error") or {}).get("code") == code
        and out.get("lead_1h") is None
    )


def main() -> int:
    checks: list[dict] = []
    hashes_before = {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in PROTECTED if p.is_file()
    }
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    names = [f["column"] for f in contract["model_b_features"]]
    feature_order_hash = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()

    engine = FinalMultiLeadEngine()
    t0 = time.perf_counter()
    models = engine.load_models()
    init_s = time.perf_counter() - t0
    checks.append({"name": "engine_initializes", "passed": True, "detail": f"{init_s:.3f}s"})

    model_hashes = {}
    for rel, expected in PHASE13A_HASHES.items():
        path = ROOT / rel
        exists = path.is_file()
        digest = sha256_file(path) if exists else None
        model_hashes[rel] = digest
        lead = rel.split("_")[-1].replace(".joblib", "")
        clf = models[{"1h": "1h", "2h": "2h", "3h": "3h"}[lead]]
        n_in = int(getattr(clf, "n_features_in_", -1))
        typ = type(clf).__name__
        ok = exists and digest == expected and n_in == 83 and typ == "RandomForestClassifier"
        checks.append(
            {
                "name": f"frozen_joblib_{lead}",
                "passed": ok,
                "detail": {"exists": exists, "sha256": digest, "expected": expected, "n_features_in_": n_in, "type": typ},
            }
        )

    checks.append({"name": "feature_count_83", "passed": len(names) == 83 and len(set(names)) == 83, "detail": len(names)})
    checks.append(
        {
            "name": "engine_uses_contract_order",
            "passed": engine.feature_names == names,
            "detail": feature_order_hash,
        }
    )
    leaked = [n for n in names if any(s in n.lower() for s in FORBIDDEN)]
    future = [n for n in names if any(m in n.lower() for m in FUTURE)]
    checks.append({"name": "no_forbidden_predictors", "passed": not leaked, "detail": leaked})
    checks.append({"name": "no_future_predictors", "passed": not future, "detail": future})

    thr_ok = (
        abs(engine.thresholds["1h"] - 0.065) < 1e-12
        and abs(float(contract["thresholds"]["1h"]) - 0.065) < 1e-12
        and abs(float(contract["thresholds"]["2h"]) - 0.065) < 1e-12
        and abs(float(contract["thresholds"]["3h"]) - 0.065) < 1e-12
        and abs(FROZEN_THRESHOLD - 0.065) < 1e-12
    )
    checks.append({"name": "thresholds_0_065", "passed": thr_ok, "detail": engine.thresholds})

    service = MultiLeadInferenceService(engine=engine)
    replay = pd.read_csv(REPLAY_CSV)
    replay["timestamp_utc"] = pd.to_datetime(replay["timestamp_utc"], utc=True)
    replay_results = []
    mapping_for_later = None
    station_for_later = None
    ts_for_later = None
    for sid in ("VABB", "VECC", "VIDP", "VOCI"):
        sub = replay.loc[replay["station_id"] == sid]
        if sub.empty:
            checks.append({"name": f"replay_{sid}", "passed": False, "detail": "no 12E row"})
            continue
        row = sub.iloc[0]
        ts = row["timestamp_utc"]
        t1 = time.perf_counter()
        out = service.replay_historical(sid, ts)
        elapsed = time.perf_counter() - t1
        if mapping_for_later is None and out.get("ok"):
            feats = service._load_features()
            mask = (feats["station_id"].astype(str) == sid) & (feats["timestamp_utc"] == ts)
            mapping_for_later = {n: float(feats.loc[mask].iloc[0][n]) for n in names}
            station_for_later = sid
            ts_for_later = ts
        ok = out.get("ok") is True and out.get("data_status") == "COMPLETE"
        diffs = {}
        for lead, col in (("1h", "baseline_p_1h"), ("2h", "baseline_p_2h"), ("3h", "baseline_p_3h")):
            got = float(out[f"lead_{lead}"]["probability"]) if ok else None
            exp = float(row[col])
            thr = float(out[f"lead_{lead}"]["threshold"]) if ok else None
            alert = out[f"lead_{lead}"]["alert"] if ok else None
            match = ok and abs(got - exp) <= TOL and abs(thr - 0.065) < 1e-12 and alert == (got >= 0.065)
            diffs[lead] = {"engine": got, "phase12e": exp, "match": match, "alert": alert}
            ok = ok and match
        replay_results.append({"station": sid, "timestamp_utc": str(ts), "elapsed_s": elapsed, "leads": diffs})
        checks.append({"name": f"replay_consistency_{sid}", "passed": ok, "detail": diffs})

    mapping = mapping_for_later or {n: 0.0 for n in names}
    sid0 = station_for_later or "VABB"
    ts0 = ts_for_later or pd.Timestamp("2025-05-31T14:00:00+00:00")

    a = engine.predict(sid0, ts0, mapping)
    b = engine.predict(sid0, ts0, mapping)
    det = (
        a.get("ok")
        and a["lead_1h"]["probability"] == b["lead_1h"]["probability"]
        and a["lead_2h"]["probability"] == b["lead_2h"]["probability"]
        and a["lead_3h"]["probability"] == b["lead_3h"]["probability"]
        and a["lead_1h"]["alert"] == b["lead_1h"]["alert"]
        and a["lead_2h"]["alert"] == b["lead_2h"]["alert"]
        and a["lead_3h"]["alert"] == b["lead_3h"]["alert"]
    )
    checks.append({"name": "determinism", "passed": det, "detail": {"p1": a.get("lead_1h")}})

    p_core = (
        a["lead_1h"]["probability"],
        a["lead_2h"]["probability"],
        a["lead_3h"]["probability"],
    )
    sep = True
    for label, kwargs in (
        ("none", {}),
        ("sat", {"satellite_context": {"sat_cloudy_fraction_25km": 1.0}}),
        ("radar", {"radar_context": {"dbz": 40}}),
        ("lightning", {"lightning_context": {"strokes": 3}}),
    ):
        o = engine.predict(sid0, ts0, mapping, **kwargs)
        same = (
            o["lead_1h"]["probability"],
            o["lead_2h"]["probability"],
            o["lead_3h"]["probability"],
        ) == p_core
        sep = sep and same
        checks.append({"name": f"evidence_separation_{label}", "passed": same, "detail": same})
    checks.append({"name": "evidence_separation_all", "passed": sep, "detail": "context does not change Model B p"})

    fail_cases = [
        ("A_missing_feature", "MISSING_FEATURES", dict(mapping), lambda d: d.pop("temperature_2m", None), None),
        ("B_wrong_count", "FEATURE_COUNT_MISMATCH", [0.0] * 10, None, None),
        ("C_invalid_station", "INVALID_STATION", mapping, None, ("XXXX", ts0)),
        ("D_invalid_timestamp", "INVALID_TIMESTAMP", mapping, None, (sid0, "not-a-time")),
        ("E_non_finite", "NON_FINITE_FEATURE", dict(mapping), lambda d: d.__setitem__("nwp_cape", float("nan")), None),
        ("F_satellite", "FORBIDDEN_FEATURE_SUBSTITUTION", dict(mapping), lambda d: d.__setitem__("sat_cloudy_fraction_25km", 1.0), None),
        ("G_radar", "FORBIDDEN_FEATURE_SUBSTITUTION", dict(mapping), lambda d: d.__setitem__("radar_reflectivity", 40.0), None),
        ("H_lightning", "FORBIDDEN_FEATURE_SUBSTITUTION", dict(mapping), lambda d: d.__setitem__("lightning_density", 1.0), None),
        ("I_wrong_order", "FEATURE_ORDER_MISMATCH", mapping, None, None),
        ("J_missing_nwp", "MISSING_NWP_FIELDS", dict(mapping), lambda d: d.pop("nwp_cape", None), None),
    ]
    fail_closed = {}
    for name, code, feats, mut, ident in fail_cases:
        feats2 = feats
        if mut is not None and isinstance(feats, dict):
            feats2 = dict(feats)
            mut(feats2)
        st, ts = ident if ident else (sid0, ts0)
        kw = {}
        if name == "I_wrong_order":
            kw["feature_names"] = list(reversed(names))
        t_fail = time.perf_counter()
        out = engine.predict(st, ts, feats2, **kw)
        dt = time.perf_counter() - t_fail
        passed = fail_ok(out, code)
        fail_closed[name] = {"passed": passed, "code": (out.get("error") or {}).get("code"), "elapsed_s": dt}
        checks.append({"name": f"fail_closed_{name}", "passed": passed, "detail": fail_closed[name]})

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
    h = client.get("/health")
    checks.append({"name": "api_health", "passed": h.status_code == 200 and h.get_json().get("status") == "ok", "detail": h.status_code})
    root = client.get("/")
    checks.append({"name": "api_index", "passed": root.status_code == 200, "detail": root.status_code})
    get_replay = client.get("/replay")
    checks.append(
        {
            "name": "api_get_replay_existing_behavior",
            "passed": get_replay.status_code in {200, 405},
            "detail": {"status": get_replay.status_code, "note": "Phase 16 uses POST /replay"},
        }
    )
    post_all = client.post("/replay/all", json={"timestamp_utc": "2023-06-15T12:00:00Z"})
    body = post_all.get_json() or {}
    checks.append(
        {
            "name": "api_replay_all",
            "passed": post_all.status_code == 200 and body.get("replay_mode") == "HISTORICAL_REPLAY",
            "detail": post_all.status_code,
        }
    )

    # Valid historical API: reuse Flask app with real service already constructed in create_app.
    # Avoid double-loading CSV: call service then also hit engine-backed JSON shape via test_client
    # using features payload (no second full-table scan).
    api_valid = client.post(
        "/api/inference/multilead",
        json={
            "station_id": sid0,
            "timestamp_utc": pd.Timestamp(ts0).isoformat(),
            "features": mapping,
            "inference_mode": "HISTORICAL_REPLAY",
        },
    )
    aj = api_valid.get_json() or {}
    leads_ok = all(
        isinstance(aj.get(k), dict) and {"probability", "threshold", "alert"} <= set(aj[k])
        for k in ("lead_1h", "lead_2h", "lead_3h")
    )
    api_ok = (
        api_valid.status_code == 200
        and aj.get("ok") is True
        and aj.get("data_status") == "COMPLETE"
        and leads_ok
        and abs(float(aj["lead_1h"]["threshold"]) - 0.065) < 1e-12
    )
    checks.append({"name": "api_multilead_valid", "passed": api_ok, "detail": {"status": api_valid.status_code, "ok": aj.get("ok")}})

    hashes_after = {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in PROTECTED if p.is_file()
    }
    immutable = hashes_before == hashes_after
    checks.append({"name": "source_immutability", "passed": immutable, "detail": "8D/8E/12/10I/contract hashes"})

    passed = all(c["passed"] for c in checks)
    payload = {
        "phase": "13C",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(names),
        "feature_order_hash": feature_order_hash,
        "thresholds": {"1h": 0.065, "2h": 0.065, "3h": 0.065},
        "model_hashes": model_hashes,
        "phase13a_expected_hashes": PHASE13A_HASHES,
        "replay_consistency": replay_results,
        "fail_closed": fail_closed,
        "api_regression": {
            "health": h.status_code,
            "index": root.status_code,
            "get_replay": get_replay.status_code,
            "replay_all": post_all.status_code,
            "multilead_valid": api_valid.status_code,
        },
        "determinism": det,
        "evidence_separation": sep,
        "source_immutability": immutable,
        "engine_init_seconds": init_s,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["passed"]),
        "passed": passed,
        "final_status": "FINAL_MULTI_LEAD_ENGINE_FROZEN" if passed else "FINAL_MULTI_LEAD_ENGINE_FREEZE_REQUIRES_REVISION",
        "checks": checks,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(payload["final_status"])
    if not passed:
        for c in checks:
            if not c["passed"]:
                print(f"  {c['name']}: {c['detail']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
