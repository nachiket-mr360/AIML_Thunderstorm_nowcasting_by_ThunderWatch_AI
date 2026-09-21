"""Phase 13C: end-to-end CSV → Phase 13B engine integration (no training)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.v2_inference_engine import V2InferenceEngine

CSV = ROOT / "dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025.csv"
CONTRACT = ROOT / "outputs/v2_inference/phase13a/inference_contract.json"
ENGINE_SRC = ROOT / "src/inference/v2_inference_engine.py"
OUT_DIR = ROOT / "outputs/v2_inference/phase13c"
SAMPLES_CSV = OUT_DIR / "e2e_prediction_samples.csv"
VALIDATION_JSON = OUT_DIR / "e2e_validation.json"

STATIONS = ["VOTV", "VECC", "VIDP", "VOCI", "VABB"]
TARGETS = ["target_1h", "target_2h", "target_3h"]
MODEL_PATHS = [
    ROOT / "models/v2/nwp_experiment/B_atmospheric_nwp_target_1h.joblib",
    ROOT / "models/v2/nwp_experiment/B_atmospheric_nwp_target_2h.joblib",
    ROOT / "models/v2/nwp_experiment/B_atmospheric_nwp_target_3h.joblib",
]
FORBIDDEN_IN_FEATURES = (
    "annual_thunder_hours",
    "satellite",
    "radar",
    "lightning",
    "target_1h",
    "target_2h",
    "target_3h",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_contract_features() -> list[str]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    return [f["column"] for f in contract["model_b_features"]]


def select_rows(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    finite = np.isfinite(df[names].to_numpy(dtype=np.float64)).all(axis=1)
    complete = df.loc[finite].copy()
    complete = complete.sort_values(["station_id", "timestamp_utc"], kind="mergesort")
    picked = []
    for st in STATIONS:
        sub = complete.loc[complete["station_id"] == st]
        if sub.empty:
            raise SystemExit(f"no complete-case row for {st}")
        picked.append(sub.iloc[0])
    return pd.DataFrame(picked)


def run() -> dict:
    hashes_before = {
        "contract": sha256_file(CONTRACT),
        "engine": sha256_file(ENGINE_SRC),
        **{p.name: sha256_file(p) for p in MODEL_PATHS},
    }
    names = load_contract_features()
    if len(names) != 83:
        raise SystemExit(f"expected 83 features, got {len(names)}")
    leaked = [n for n in names if n in FORBIDDEN_IN_FEATURES or any(s in n.lower() for s in ("target_", "satellite", "radar", "lightning"))]
    if leaked:
        raise SystemExit(f"forbidden names in contract features: {leaked}")

    usecols = ["station_id", "timestamp_utc"] + names + TARGETS
    df = pd.read_csv(CSV, usecols=usecols)
    selected = select_rows(df, names)

    engine = V2InferenceEngine()
    rows = []
    checks = []
    for _, rec in selected.iterrows():
        station = rec["station_id"]
        ts = rec["timestamp_utc"]
        feats = {n: rec[n] for n in names}
        out = engine.predict(station, ts, feats, feature_names=names)
        hist = {f"historical_reference_{t}": (None if pd.isna(rec[t]) else float(rec[t])) for t in TARGETS}
        rows.append(
            {
                "station_id": station,
                "timestamp_utc": str(ts),
                "lead_1h_probability": out["lead_1h_probability"],
                "lead_1h_alert": out["lead_1h_alert"],
                "lead_2h_probability": out["lead_2h_probability"],
                "lead_2h_alert": out["lead_2h_alert"],
                "lead_3h_probability": out["lead_3h_probability"],
                "lead_3h_alert": out["lead_3h_alert"],
                "model_version": out["model_version"],
                "data_completeness": out["data_completeness"],
                "data_status": out["data_status"],
                **hist,
            }
        )
        alerts_ok = True
        if out["data_status"] == "COMPLETE":
            for lead in ("1h", "2h", "3h"):
                p = out[f"lead_{lead}_probability"]
                a = out[f"lead_{lead}_alert"]
                if not np.isfinite(p):
                    alerts_ok = False
                if a != (1 if p >= 0.065 else 0):
                    alerts_ok = False
        checks.append(
            {
                "station_id": station,
                "timestamp_utc": str(ts),
                "n_features": len(feats),
                "names_match": list(feats.keys()) == names,
                "all_finite": bool(np.isfinite([float(feats[n]) for n in names]).all()),
                "data_status": out["data_status"],
                "alerts_follow_threshold": alerts_ok,
                "targets_not_in_features": all(t not in feats for t in TARGETS),
            }
        )

    hashes_after = {
        "contract": sha256_file(CONTRACT),
        "engine": sha256_file(ENGINE_SRC),
        **{p.name: sha256_file(p) for p in MODEL_PATHS},
    }
    integrity_ok = hashes_before == hashes_after
    all_complete = all(r["data_status"] == "COMPLETE" for r in rows)
    passed = integrity_ok and all_complete and all(c["alerts_follow_threshold"] for c in checks)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SAMPLES_CSV, index=False)
    payload = {
        "phase": "13C",
        "passed": passed,
        "not_model_evaluation": True,
        "data_source": str(CSV.relative_to(ROOT)).replace("\\", "/"),
        "selection": "complete-case (all 83 finite), sort station_id+timestamp_utc, first row per station",
        "n_samples": len(rows),
        "stations": [r["station_id"] for r in rows],
        "timestamps": [r["timestamp_utc"] for r in rows],
        "feature_count": 83,
        "historical_targets_not_inputs": True,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "integrity_unchanged": integrity_ok,
        "samples": rows,
        "checks": checks,
    }
    VALIDATION_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit("Phase 13C validation failed")
    return payload


if __name__ == "__main__":
    run()
    print("PASS")
