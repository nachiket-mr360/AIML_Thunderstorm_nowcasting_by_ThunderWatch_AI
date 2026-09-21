"""Phase 13A: validate V2 Model B inference contract (read-only).

Does not train, does not write model artifacts, does not implement serving.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
CONTRACT_DIR = BASE_DIR / "outputs" / "v2_inference" / "phase13a"
CONTRACT_JSON = CONTRACT_DIR / "inference_contract.json"
VALIDATION_JSON = CONTRACT_DIR / "contract_validation.json"

HORIZONS = ("target_1h", "target_2h", "target_3h")
MODEL_STEM = "B_atmospheric_nwp"
FORBIDDEN_SUBSTRINGS = (
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
FUTURE_PREDICTOR_MARKERS = ("t+1", "t+2", "t+3", "lead_1h", "lead_2h", "lead_3h", "future_")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(checks: list[dict], name: str, detail: str) -> None:
    checks.append({"name": name, "passed": False, "detail": detail})


def ok(checks: list[dict], name: str, detail: str = "") -> None:
    checks.append({"name": name, "passed": True, "detail": detail})


def main() -> int:
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    stop = False

    if not CONTRACT_JSON.exists():
        payload = {
            "passed": False,
            "stop": True,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": [{"name": "contract_json_exists", "passed": False, "detail": str(CONTRACT_JSON)}],
        }
        VALIDATION_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("FAIL: inference_contract.json missing")
        return 1

    contract = json.loads(CONTRACT_JSON.read_text(encoding="utf-8"))
    features = contract["model_b_features"]
    names = [f["column"] for f in features]

    if len(names) == 83 and len(set(names)) == 83:
        ok(checks, "exactly_83_features", "83 unique Model B columns")
    else:
        fail(checks, "exactly_83_features", f"count={len(names)} unique={len(set(names))}")
        stop = True

    hashes_before = {}
    for h in HORIZONS:
        art = contract["frozen_artifacts"][h]
        model_path = BASE_DIR / art["artifact_path"]
        metrics_path = BASE_DIR / art["metrics_path"]
        if model_path.is_file():
            hashes_before[str(model_path)] = sha256_file(model_path)
            ok(checks, f"artifact_exists_{h}", art["artifact_path"])
        else:
            fail(checks, f"artifact_exists_{h}", art["artifact_path"])
            stop = True
        if metrics_path.is_file():
            ok(checks, f"metrics_exists_{h}", art["metrics_path"])
        else:
            fail(checks, f"metrics_exists_{h}", art["metrics_path"])
            stop = True

    if stop:
        _write(checks, False, True)
        print("FAIL: missing artifacts")
        return 1

    ref_list = None
    for h in HORIZONS:
        art = contract["frozen_artifacts"][h]
        metrics = json.loads((BASE_DIR / art["metrics_path"]).read_text(encoding="utf-8"))
        fl = metrics["feature_list"]
        if ref_list is None:
            ref_list = fl
        if fl != names:
            fail(checks, f"feature_order_matches_metrics_{h}", "contract vs metrics_B mismatch")
            stop = True
        else:
            ok(checks, f"feature_order_matches_metrics_{h}", "exact order match")
        if int(metrics["n_features"]) != 83:
            fail(checks, f"metrics_n_features_{h}", str(metrics["n_features"]))
            stop = True
        else:
            ok(checks, f"metrics_n_features_{h}", "83")
        frozen_thr = float(art["threshold"])
        if float(metrics["threshold"]) != frozen_thr:
            fail(checks, f"threshold_{h}", f"metrics={metrics['threshold']} contract={frozen_thr}")
            stop = True
        else:
            ok(checks, f"threshold_{h}", str(frozen_thr))
        if metrics.get("horizon") != h:
            fail(checks, f"horizon_label_{h}", str(metrics.get("horizon")))
            stop = True
        else:
            ok(checks, f"horizon_label_{h}", h)

        clf = joblib.load(BASE_DIR / art["artifact_path"])
        n_in = int(getattr(clf, "n_features_in_", -1))
        if n_in != 83:
            fail(checks, f"joblib_n_features_in_{h}", str(n_in))
            stop = True
        else:
            ok(checks, f"joblib_n_features_in_{h}", "83")
        stored_names = getattr(clf, "feature_names_in_", None)
        if stored_names is not None:
            stored = [str(x) for x in list(stored_names)]
            if stored != names:
                fail(checks, f"joblib_feature_names_{h}", "joblib names differ from contract")
                stop = True
            else:
                ok(checks, f"joblib_feature_names_{h}", "match")
        else:
            ok(
                checks,
                f"joblib_feature_names_{h}",
                "feature_names_in_ absent; order taken from Phase 8D metrics feature_list",
            )

    leaked = [n for n in names if any(s in n.lower() for s in FORBIDDEN_SUBSTRINGS)]
    if leaked:
        fail(checks, "no_target_or_forbidden_in_predictors", str(leaked))
        stop = True
    else:
        ok(checks, "no_target_or_forbidden_in_predictors", "no target/label/sat/radar/lightning/thunder_hours")

    future = [n for n in names if any(m in n.lower() for m in FUTURE_PREDICTOR_MARKERS)]
    if future:
        fail(checks, "no_future_tplus_predictors", str(future))
        stop = True
    else:
        ok(checks, "no_future_tplus_predictors", "no T+1/T+2/T+3 predictor columns")

    if any(n == "annual_thunder_hours" for n in names):
        fail(checks, "no_annual_thunder_hours", "present")
        stop = True
    else:
        ok(checks, "no_annual_thunder_hours", "absent")

    required = contract.get("required_input_identity", [])
    for col in ("station_id", "timestamp_utc"):
        if col not in required:
            fail(checks, f"identity_{col}", "missing from required identity")
            stop = True
        else:
            ok(checks, f"identity_{col}", "required identity, not a model feature")

    hashes_after = {p: sha256_file(Path(p)) for p in hashes_before}
    if hashes_before != hashes_after:
        fail(checks, "model_artifacts_unmodified", "hash changed during validation")
        stop = True
    else:
        ok(checks, "model_artifacts_unmodified", "sha256 unchanged")

    passed = all(c["passed"] for c in checks) and not stop
    _write(checks, passed, not passed)
    print("PASS" if passed else "FAIL")
    if not passed:
        for c in checks:
            if not c["passed"]:
                print(f"  {c['name']}: {c['detail']}")
        return 1
    return 0


def _write(checks: list[dict], passed: bool, stop: bool) -> None:
    payload = {
        "phase": "13A",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "stop": stop,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["passed"]),
        "checks": checks,
    }
    VALIDATION_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
