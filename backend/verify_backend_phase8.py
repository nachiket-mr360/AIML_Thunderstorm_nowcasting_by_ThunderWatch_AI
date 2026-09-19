"""Phase 8 verification -- Flask backend over the Phase 7 prediction engine.

SIH26072 -- "AIML based nowcasting of thunderstorm and lightning".

WHAT THIS SCRIPT PROVES
-----------------------
It starts the real Flask application as a separate process, talks to it over
real HTTP on a real socket, and checks the behaviour the Phase 8 requirements
describe. Nothing is asserted from a mocked payload: the prediction checks are
made against a genuine live call, and the "no fabricated values" check compares
the HTTP response against an **independent direct call to the Phase 7 engine**
made by this script.

Sections
--------
 1. The application starts and answers.
 2. ``GET /`` returns 200 and a real dashboard document.
 3. ``GET /api/health`` returns a valid, truthful health report.
 4. ``GET /api/prediction`` returns a real prediction payload.
 5. Probability is in [0, 1] and the class is the locked threshold's verdict.
 6. Lead time is 1 hour and the timestamps are coherent.
 7. The data source is the real provider, with the served grid cell reported.
 8. The model identifier is the 1-hour thunderstorm nowcast.
 9. Feature completeness is reported and honest.
10. No fabricated or static values: HTTP payload == independent engine result.
11. The retained Phase 2 endpoints still work.
12. A live-data failure is handled safely (simulated at the provider boundary).
13. An error response never contains a prediction.
14. Scientific terminology guardrails are present and correct.
15. CORS and cache headers are sensible for a local prototype.
16. Phase 1-7 protected artifacts are unchanged (hash ancestry + index blobs).

Run:
    python backend/verify_backend_phase8.py

Exit code 0 means every check passed. The machine-readable summary is written to
``outputs/PHASE8_BACKEND_VALIDATION.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VALIDATION_JSON = OUTPUTS_DIR / "PHASE8_BACKEND_VALIDATION.json"

#: The instant Phase 8 work began, before any Phase 8 file was written. Any
#: Phase 1-7 protected artifact whose mtime is at or after this instant was
#: touched during this phase, which is a failure whatever its content says.
PHASE8_WORK_START_UTC = datetime(2026, 9, 15, 4, 44, 0, tzinfo=timezone.utc)

#: Files Phase 8 is *allowed* to modify: the backend entry point and the two
#: frontend files that present the nowcast. Everything else tracked by git
#: belongs to Phases 1-7 and must be byte-identical to what was staged.
PHASE8_OWNED_FILES = {
    "app.py",
    "templates/index.html",
    "static/dashboard.js",
    "static/style.css",
}

#: Phase 6/7 artifacts that exist in the working tree but were never staged, so
#: git has no baseline for them. They are covered by the mtime guard and by the
#: before/after manifest taken around the server run.
PHASE8_UNTRACKED_PROTECTED = (
    "ml/predict_thunderstorm_nowcast.py",
    "ml/verify_prediction_engine_phase7.py",
    "outputs/PHASE7_REALTIME_PREDICTION_ENGINE_REPORT.md",
)

#: The recorded Phase 3 -> 4 -> 5 ancestry chain. Each value was written into
#: the *next* phase's metadata when that phase ran, so recomputing it now is a
#: genuine end-to-end integrity check rather than a self-comparison.
DATASET_ANCESTRY_CHAIN = (
    (
        "dataset/raw_openmeteo/votv_openmeteo_hourly_2014_2025.csv",
        "beca4c721cc193717c3d5d6a11547b288f41611eb7dfb655aeb8a386baf3cdae",
        "Phase 3 contiguous hourly context grid (recorded in Phase 4 and Phase 5 metadata)",
    ),
    (
        "dataset/votv_thunderstorm_synchronized_2014_2025.csv",
        "6e0be102df5ef39ebda1847914acc7ed9554a137f00547998785054e384bbc4c",
        "Phase 3 synchronized label source (recorded in Phase 4 and Phase 5 metadata)",
    ),
    (
        "dataset/votv_thunderstorm_features_2014_2025.csv",
        "d18afa7cd04d91cf42d2802852245d2f92c41a74d6935f10cb5bc8b13f271b7d",
        "Phase 4 feature table (recorded in Phase 5 metadata and in the Phase 6 bundle)",
    ),
    (
        "dataset/votv_thunderstorm_nowcast_2014_2025.csv",
        "b74b86f2f9f1c9218bca77dfe0afbe0339ba34687410b1299be094bbae0e1308",
        "Phase 5 nowcast target table (recorded inside the Phase 6 model artifact)",
    ),
)

LOCKED_THRESHOLD = 0.0775
EXPECTED_LEAD_TIME_HOURS = 1
EXPECTED_MODEL_NAME = "thunderstorm_nowcast_1h"
EXPECTED_MODEL_FILE = "thunderstorm_nowcast_1h.joblib"
EXPECTED_FEATURE_COUNT = 28
EXPECTED_GRID_CELL = {"latitude": 8.471002, "longitude": 76.93298}


# --------------------------------------------------------------------------
# Result bookkeeping
# --------------------------------------------------------------------------

RESULTS: list[dict[str, Any]] = []
_EVIDENCE: dict[str, Any] = {}


def record(passed: bool, name: str, evidence: Any = "") -> bool:
    """Record one check and print it. Returns ``passed`` for inline use."""
    text = evidence if isinstance(evidence, str) else json.dumps(evidence, default=str)
    RESULTS.append({"check": name, "passed": bool(passed), "evidence": text[:1200]})
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
    if text:
        print(f"        {text[:600]}")
    return bool(passed)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any) -> float | None:
    """Parse a JSON number, returning ``None`` for anything unusable."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def http_get(base: str, path: str, *, timeout: float = 90.0) -> dict[str, Any]:
    """GET a path and return status, headers and parsed JSON (if any)."""
    try:
        response = requests.get(f"{base}{path}", timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - reported as a failed request
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "status": None, "json": None}
    try:
        body = response.json()
    except ValueError:
        body = None
    return {
        "ok": True,
        "status": response.status_code,
        "json": body,
        "text": response.text,
        "headers": dict(response.headers),
    }


# --------------------------------------------------------------------------
# Server lifecycle
# --------------------------------------------------------------------------


def start_server(port: int, log_path: Path) -> tuple[subprocess.Popen, Any]:
    env = dict(os.environ)
    env.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "FLASK_DEBUG": "0",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    log = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return process, log


def wait_for_server(base: str, process: subprocess.Popen, timeout: float = 300.0) -> dict[str, Any]:
    """Poll ``/api/health?live=0`` until the app answers or the timeout expires.

    The provider probe is skipped while waiting so that a slow upstream cannot
    be mistaken for a failure to start.
    """
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {"status": None}
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return {"started": False, "reason": f"process exited with code {process.returncode}"}
        last = http_get(base, "/api/health?live=0", timeout=30.0)
        if last.get("status") == 200:
            return {"started": True, "health": last["json"]}
        time.sleep(1.5)
    return {"started": False, "reason": "timeout waiting for /api/health", "last": last.get("status")}


# --------------------------------------------------------------------------
# Section 16: protected artifact integrity
# --------------------------------------------------------------------------


def protected_paths() -> list[str]:
    """Every Phase 1-7 file that must be unchanged, as repo-relative paths."""
    listed = git("ls-files", "-s").stdout.splitlines()
    tracked: list[str] = []
    for line in listed:
        if "\t" not in line:
            continue
        tracked.append(line.split("\t", 1)[1].strip())

    protected = [p for p in tracked if p not in PHASE8_OWNED_FILES]
    for extra in PHASE8_UNTRACKED_PROTECTED:
        if extra not in protected:
            protected.append(extra)
    return sorted(set(protected))


def check_integrity() -> None:
    section("16. PHASE 1-7 PROTECTED ARTIFACT INTEGRITY")

    paths = protected_paths()
    existing = [p for p in paths if (PROJECT_ROOT / p).is_file()]
    record(
        len(existing) == len(paths),
        "every protected artifact is still present",
        {"declared": len(paths), "present": len(existing)},
    )

    # -- 16a. recorded ancestry chain --------------------------------------
    chain_ok = True
    chain_evidence = []
    for relative, recorded, description in DATASET_ANCESTRY_CHAIN:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            chain_ok = False
            chain_evidence.append({"file": relative, "status": "missing"})
            continue
        actual = sha256_file(path)
        matches = actual == recorded
        chain_ok = chain_ok and matches
        chain_evidence.append(
            {
                "file": relative,
                "recorded": recorded,
                "recomputed": actual,
                "matches": matches,
                "meaning": description,
            }
        )
    record(
        chain_ok,
        "recorded Phase 3->4->5 dataset hashes recompute exactly",
        chain_evidence,
    )
    _EVIDENCE["dataset_ancestry_chain"] = chain_evidence

    # -- 16b. the Phase 6 model still points at that exact dataset ---------
    import joblib

    bundle = joblib.load(PROJECT_ROOT / "models" / EXPECTED_MODEL_FILE)
    bundle_hash = bundle.get("training_dataset_sha256")
    nowcast_csv_hash = sha256_file(PROJECT_ROOT / "dataset" / "votv_thunderstorm_nowcast_2014_2025.csv")
    record(
        bundle_hash == nowcast_csv_hash,
        "Phase 6 artifact's recorded training-dataset hash matches the dataset on disk",
        {"artifact": bundle_hash, "on_disk": nowcast_csv_hash},
    )
    record(
        float(bundle.get("decision_threshold")) == LOCKED_THRESHOLD,
        "locked decision threshold inside the artifact is still 0.0775",
        {"artifact_threshold": bundle.get("decision_threshold")},
    )
    record(
        list(bundle.get("feature_names") or []) == list(bundle.get("feature_names") or [])
        and len(bundle.get("feature_names") or []) == EXPECTED_FEATURE_COUNT,
        "artifact still declares 28 features",
        {"n_features": len(bundle.get("feature_names") or [])},
    )

    # -- 16c. every tracked protected file is byte-identical to its index blob
    index_blobs: dict[str, str] = {}
    for line in git("ls-files", "-s").stdout.splitlines():
        if "\t" not in line:
            continue
        meta, name = line.split("\t", 1)
        parts = meta.split()
        if len(parts) >= 2:
            index_blobs[name.strip()] = parts[1]

    to_hash = [p for p in paths if p in index_blobs and (PROJECT_ROOT / p).is_file()]
    mismatches: list[dict[str, str]] = []
    if to_hash:
        hashed = git("hash-object", "--", *to_hash)
        working = hashed.stdout.splitlines()
        if len(working) != len(to_hash):
            mismatches.append({"error": "git hash-object returned an unexpected line count"})
        else:
            for name, digest in zip(to_hash, working):
                if digest.strip() != index_blobs[name]:
                    mismatches.append(
                        {"file": name, "index": index_blobs[name], "working": digest.strip()}
                    )
    record(
        not mismatches,
        "every git-tracked Phase 1-7 file is byte-identical to its staged blob",
        {"files_compared": len(to_hash), "mismatches": mismatches or "none"},
    )

    # -- 16d. mtime guard ---------------------------------------------------
    late: list[dict[str, str]] = []
    for relative in paths:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if mtime >= PHASE8_WORK_START_UTC:
            late.append({"file": relative, "mtime_utc": mtime.isoformat()})
    record(
        not late,
        "no protected artifact was written during Phase 8 (mtime guard)",
        {"guard_from_utc": PHASE8_WORK_START_UTC.isoformat(), "violations": late or "none"},
    )

    # -- 16e. git working-tree status for the protected set -----------------
    status = git("status", "--porcelain", "--", *paths).stdout.strip()
    staged_only = [
        line
        for line in status.splitlines()
        if line.strip() and line[:1] != " " and line[1:2] == " "
    ]
    unstaged = [line for line in status.splitlines() if line.strip() and line[1:2] not in (" ", "?")]
    record(
        not unstaged,
        "no Phase 1-7 file has an unstaged modification in the working tree",
        {"porcelain": status or "(clean)", "unstaged": unstaged or "none", "staged_pre_existing": staged_only},
    )

    return None


# --------------------------------------------------------------------------
# Section 12: live-data failure handling
# --------------------------------------------------------------------------


def check_live_data_failure_handling() -> None:
    section("12-13. LIVE-DATA FAILURE HANDLING AND ERROR BODY SAFETY")

    import unittest.mock as mock

    import app as application

    client = application.app.test_client()

    class _FakeResponse:
        def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self) -> Any:
            if self._payload is None:
                raise ValueError("not json")
            return self._payload

    scenarios = [
        (
            "provider unreachable (connection error)",
            requests.exceptions.ConnectionError("simulated network failure"),
            None,
        ),
        (
            "provider timeout",
            requests.exceptions.Timeout("simulated timeout"),
            None,
        ),
        (
            "provider returns HTTP 503",
            None,
            _FakeResponse(503, None, "service unavailable"),
        ),
        (
            "provider returns a body that is not JSON",
            None,
            _FakeResponse(200, None, "<html>not json</html>"),
        ),
        (
            "provider returns the wrong units for wind speed",
            None,
            _FakeResponse(
                200,
                {
                    "latitude": 8.471002,
                    "longitude": 76.93298,
                    "utc_offset_seconds": 0,
                    "hourly_units": {
                        "time": "iso8601",
                        "temperature_2m": "\u00b0C",
                        "relative_humidity_2m": "%",
                        "surface_pressure": "hPa",
                        "wind_speed_10m": "m/s",
                        "wind_direction_10m": "\u00b0",
                        "precipitation": "mm",
                        "cloud_cover": "%",
                    },
                    "hourly": {
                        "time": ["2026-09-15T00:00"],
                        "temperature_2m": [27.0],
                        "relative_humidity_2m": [80.0],
                        "surface_pressure": [1009.0],
                        "wind_speed_10m": [3.0],
                        "wind_direction_10m": [240.0],
                        "precipitation": [0.0],
                        "cloud_cover": [90.0],
                    },
                },
                "",
            ),
        ),
        (
            "provider omits a required variable",
            None,
            _FakeResponse(
                200,
                {
                    "latitude": 8.471002,
                    "longitude": 76.93298,
                    "utc_offset_seconds": 0,
                    "hourly_units": {
                        "time": "iso8601",
                        "temperature_2m": "\u00b0C",
                        "relative_humidity_2m": "%",
                        "surface_pressure": "hPa",
                        "wind_speed_10m": "km/h",
                        "wind_direction_10m": "\u00b0",
                        "precipitation": "mm",
                        "cloud_cover": "%",
                    },
                    "hourly": {
                        "time": ["2026-09-15T00:00"],
                        "temperature_2m": [27.0],
                        "relative_humidity_2m": [80.0],
                        "surface_pressure": [1009.0],
                        "wind_speed_10m": [3.0],
                        "wind_direction_10m": [240.0],
                        "precipitation": [0.0],
                    },
                },
                "",
            ),
        ),
    ]

    all_safe = True
    detail: list[dict[str, Any]] = []

    for label, side_effect, response in scenarios:
        # The Phase 7 engine does `import requests` inside its fetch function, so
        # the failure is injected on the requests module itself -- exactly where
        # the engine will look. This exercises the real engine code path rather
        # than a stand-in for it.
        patch_target = "requests.get"
        if side_effect is not None:
            patcher = mock.patch(patch_target, side_effect=side_effect)
        else:
            patcher = mock.patch(patch_target, return_value=response)

        with patcher:
            result = client.get("/api/prediction")

        body = result.get_json(silent=True) or {}
        has_no_prediction = (
            body.get("probability") is None
            and body.get("predicted_class") is None
            and body.get("risk_label") is None
            and body.get("threshold") is None
        )
        is_error_json = result.status_code >= 400 and body.get("error") is True
        safe = result.status_code != 200 and has_no_prediction and is_error_json
        all_safe = all_safe and safe
        detail.append(
            {
                "scenario": label,
                "http_status": result.status_code,
                "code": body.get("code"),
                "probability": body.get("probability"),
                "predicted_class": body.get("predicted_class"),
                "no_prediction_fields_are_null": has_no_prediction,
                "handled_safely": safe,
            }
        )

    record(
        all_safe,
        "every simulated live-data failure returns an error and never a prediction",
        detail,
    )
    _EVIDENCE["failure_scenarios"] = detail

    # -- 13. the error body must not resemble a prediction -----------------
    with mock.patch(
        "requests.get",
        side_effect=requests.exceptions.ConnectionError("simulated network failure"),
    ):
        result = client.get("/api/prediction")
    body = result.get_json(silent=True) or {}

    forbidden = [key for key in ("probability", "predicted_class", "risk_label") if body.get(key) is not None]
    record(
        not forbidden and body.get("no_prediction_produced") is True,
        "the failure body explicitly marks that no prediction was produced",
        {
            "no_prediction_produced": body.get("no_prediction_produced"),
            "non_null_prediction_fields": forbidden or "none",
            "code": body.get("code"),
            "message": body.get("message"),
        },
    )
    record(
        400 <= result.status_code < 600,
        "a failed prediction is an HTTP error, never 200",
        {"http_status": result.status_code},
    )

    # The health endpoint must not claim ok while the provider is down.
    with mock.patch(
        "backend.nowcast_service.requests.get",
        side_effect=requests.exceptions.ConnectionError("simulated network failure"),
    ):
        health = client.get("/api/health?live=1")
    health_body = health.get_json(silent=True) or {}
    record(
        health.status_code == 503 and health_body.get("status") == "degraded",
        "health reports degraded (HTTP 503) when the live provider is unreachable",
        {
            "http_status": health.status_code,
            "status": health_body.get("status"),
            "live_dependency": (health_body.get("checks") or {}).get("live_data_dependency", {}).get("status"),
        },
    )

    health_offline = client.get("/api/health?live=0")
    offline_body = health_offline.get_json(silent=True) or {}
    record(
        health_offline.status_code == 200 and offline_body.get("status") == "ok",
        "health can be checked offline with ?live=0 and still reports model+engine",
        {
            "http_status": health_offline.status_code,
            "status": offline_body.get("status"),
            "model_available": (offline_body.get("checks") or {}).get("primary_model", {}).get("available"),
            "engine_available": (offline_body.get("checks") or {}).get("prediction_engine", {}).get("available"),
        },
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    section("PHASE 8 BACKEND VERIFICATION")
    print(f"project root : {PROJECT_ROOT}")
    print(f"python       : {sys.version.split()[0]}")
    print(f"started at   : {datetime.now(timezone.utc).isoformat()}")

    # ---- manifest of protected artifacts, taken before anything runs -----
    before_manifest = {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in protected_paths()
        if (PROJECT_ROOT / relative).is_file()
    }
    _EVIDENCE["protected_files_checked"] = len(before_manifest)

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "phase8_server.log"

        section("1. THE APPLICATION STARTS")
        process, log_handle = start_server(port, log_path)
        try:
            startup = wait_for_server(base, process)
            record(
                startup.get("started") is True,
                "python app.py starts a Flask server that answers /api/health",
                startup.get("reason") or f"listening on {base}",
            )
            if not startup.get("started"):
                log_handle.close()
                print(log_path.read_text(encoding="utf-8", errors="replace")[-4000:])
                return 1

            # ---------------- 2. dashboard --------------------------------
            section("2. GET /")
            root = http_get(base, "/", timeout=30.0)
            html = root.get("text") or ""
            record(
                root.get("status") == 200,
                "GET / returns HTTP 200",
                {"status": root.get("status"), "content_type": (root.get("headers") or {}).get("Content-Type")},
            )
            record(
                "thunderstorm" in html.lower() and len(html) > 2000,
                "GET / returns the dashboard document",
                {"bytes": len(html), "mentions_nowcast_titles": html.lower().count("thunderstorm")},
            )
            _EVIDENCE["dashboard_bytes"] = len(html)

            # ---------------- 3. health -----------------------------------
            section("3. GET /api/health")
            health = http_get(base, "/api/health?live=1", timeout=60.0)
            health_body = health.get("json") or {}
            checks = health_body.get("checks") or {}
            model_check = checks.get("primary_model") or {}
            engine_check = checks.get("prediction_engine") or {}
            live_check = checks.get("live_data_dependency") or {}

            record(
                health.get("status") == 200 and health_body.get("status") == "ok",
                "GET /api/health returns HTTP 200 with status ok",
                {"http_status": health.get("status"), "status": health_body.get("status")},
            )
            record(
                (checks.get("application_running") or {}).get("status") == "ok",
                "health confirms the application is running",
                checks.get("application_running"),
            )
            record(
                model_check.get("available") is True
                and model_check.get("artifact") == EXPECTED_MODEL_FILE
                and model_check.get("n_features") == EXPECTED_FEATURE_COUNT,
                "health confirms the primary Phase 6 model is available",
                {
                    "available": model_check.get("available"),
                    "artifact": model_check.get("artifact"),
                    "estimator": model_check.get("estimator_class"),
                    "n_features": model_check.get("n_features"),
                    "load_seconds": model_check.get("load_seconds"),
                },
            )
            record(
                engine_check.get("available") is True and engine_check.get("feature_count") == EXPECTED_FEATURE_COUNT,
                "health confirms the Phase 7 engine and its Phase 4 feature source are usable",
                {
                    "available": engine_check.get("available"),
                    "feature_module_imported": engine_check.get("feature_module_imported"),
                    "feature_count": engine_check.get("feature_count"),
                },
            )
            record(
                live_check.get("reachable") is True,
                "health confirms the live-data dependency is reachable",
                {
                    "status": live_check.get("status"),
                    "http_status": live_check.get("http_status"),
                    "latency_ms": live_check.get("latency_ms"),
                },
            )
            _EVIDENCE["health"] = {
                "status": health_body.get("status"),
                "model": model_check,
                "engine_available": engine_check.get("available"),
                "live_dependency": live_check.get("status"),
            }

            # ---------------- 4. prediction -------------------------------
            section("4-9. GET /api/prediction")
            prediction = http_get(base, "/api/prediction", timeout=120.0)
            body = prediction.get("json") or {}
            record(
                prediction.get("status") == 200 and body.get("error") is not True,
                "GET /api/prediction returns HTTP 200 with a prediction",
                {"http_status": prediction.get("status"), "code": body.get("code")},
            )

            probability = body.get("probability")
            threshold = body.get("threshold")
            predicted_class = body.get("predicted_class")
            risk_label = body.get("risk_label")

            # -- 5. probability and threshold verdict ----------------------
            record(
                isinstance(probability, (int, float)) and 0.0 <= float(probability) <= 1.0,
                "probability is a real number in [0, 1]",
                {"probability": probability},
            )
            record(
                threshold == LOCKED_THRESHOLD,
                "the threshold served is the locked Phase 6 value 0.0775",
                {"threshold": threshold},
            )
            record(
                predicted_class == int(float(probability) >= LOCKED_THRESHOLD),
                "predicted_class is exactly the 0.0775 threshold comparison",
                {
                    "probability": probability,
                    "threshold": threshold,
                    "predicted_class": predicted_class,
                    "expected": int(float(probability) >= LOCKED_THRESHOLD),
                },
            )
            record(
                risk_label == {0: "No thunderstorm alert", 1: "Thunderstorm alert"}.get(predicted_class),
                "risk_label is the label for that class",
                {"predicted_class": predicted_class, "risk_label": risk_label},
            )

            # -- 6. lead time and timestamps -------------------------------
            lead = body.get("lead_time_hours")
            prediction_ts = body.get("prediction_timestamp")
            feature_ts = body.get("feature_timestamp")
            target_ts = body.get("target_timestamp")
            record(
                lead == EXPECTED_LEAD_TIME_HOURS,
                "lead time is 1 hour",
                {"lead_time_hours": lead},
            )
            timestamps_ok = False
            target_delta_hours = None
            if prediction_ts and feature_ts and target_ts:
                try:
                    feature_dt = datetime.fromisoformat(feature_ts)
                    target_dt = datetime.fromisoformat(target_ts)
                    prediction_dt = datetime.fromisoformat(prediction_ts)
                    target_delta_hours = (target_dt - feature_dt).total_seconds() / 3600.0
                    timestamps_ok = (
                        abs(target_delta_hours - EXPECTED_LEAD_TIME_HOURS) < 1e-9
                        and feature_dt <= prediction_dt
                        and feature_ts.endswith(("+00:00", "Z"))
                    )
                except ValueError:
                    timestamps_ok = False
            record(
                timestamps_ok,
                "prediction timestamp, feature hour and target hour are coherent (target = feature + 1 h)",
                {
                    "prediction_timestamp": prediction_ts,
                    "feature_timestamp": feature_ts,
                    "target_timestamp": target_ts,
                    "target_minus_feature_hours": target_delta_hours,
                },
            )
            age = (body.get("temporal_semantics") or {}).get("data_age_hours")
            record(
                isinstance(age, (int, float)) and float(age) <= 3.0 + 1e-9,
                "the feature hour is within the 3 h freshness policy",
                {"data_age_hours": age, "policy": body.get("freshness_policy")},
            )

            # -- 7. data source and served grid ----------------------------
            provenance = (body.get("source") or {}).get("provenance") or {}
            served = body.get("served_grid_cell") or provenance.get("served_grid_cell") or {}
            record(
                provenance.get("provider") == "Open-Meteo"
                and provenance.get("frame_origin") is None
                and str(provenance.get("endpoint") or "").startswith("https://api.open-meteo.com"),
                "the data source is a real Open-Meteo request, not a replay",
                {
                    "frame_origin": (body.get("source") or {}).get("frame_origin"),
                    "provider": provenance.get("provider"),
                    "endpoint": provenance.get("endpoint"),
                    "hours_returned": provenance.get("hours_returned"),
                    "requested_at_utc": provenance.get("requested_at_utc"),
                },
            )
            grid_ok = (
                isinstance(served, dict)
                and abs(float(served.get("latitude", 0)) - EXPECTED_GRID_CELL["latitude"]) < 1e-6
                and abs(float(served.get("longitude", 0)) - EXPECTED_GRID_CELL["longitude"]) < 1e-6
            )
            record(grid_ok, "the served grid coordinates are reported and are the training cell", served)

            freshness_ok = False
            try:
                requested_at = datetime.fromisoformat(str(provenance.get("requested_at_utc")))
                freshness_ok = (datetime.now(timezone.utc) - requested_at).total_seconds() < 600
            except (TypeError, ValueError):
                freshness_ok = False
            record(
                freshness_ok,
                "the upstream request happened during this verification run (not a cached or static value)",
                {"requested_at_utc": provenance.get("requested_at_utc")},
            )

            # -- 8. model identifier ---------------------------------------
            model_block = body.get("model") or {}
            engine_block = body.get("prediction_engine") or {}
            record(
                body.get("model_identifier") == EXPECTED_MODEL_NAME
                and model_block.get("file") == EXPECTED_MODEL_FILE
                and model_block.get("n_features") == EXPECTED_FEATURE_COUNT,
                "the model identifier is the 1-hour thunderstorm nowcast",
                {
                    "model_identifier": body.get("model_identifier"),
                    "file": model_block.get("file"),
                    "estimator_class": model_block.get("estimator_class"),
                    "n_features": model_block.get("n_features"),
                    "target_column": model_block.get("target_column"),
                    "lead_time_hours": model_block.get("lead_time_hours"),
                },
            )
            record(
                engine_block.get("module") == "ml.predict_thunderstorm_nowcast"
                and engine_block.get("function") == "predict_current_thunderstorm_risk",
                "the payload states it came from the Phase 7 engine",
                engine_block,
            )

            # -- 9. feature completeness -----------------------------------
            completeness = body.get("feature_completeness") or {}
            record(
                completeness.get("n_features_missing") == 0
                and completeness.get("n_features_provided") == EXPECTED_FEATURE_COUNT
                and completeness.get("n_features_expected") == EXPECTED_FEATURE_COUNT
                and completeness.get("all_features_finite") is True
                and completeness.get("values_imputed") == 0
                and completeness.get("history_sufficient") is True,
                "feature completeness is reported: 28/28 present, finite, nothing imputed",
                completeness,
            )
            record(
                completeness.get("feature_order_matches_training_metadata") is True,
                "feature order matches the training metadata",
                {
                    "feature_order_matches_training_metadata": completeness.get(
                        "feature_order_matches_training_metadata"
                    )
                },
            )

            # ---------------- 10. no fake values ---------------------------
            section("10. NO FABRICATED OR STATIC VALUES")

            from ml.predict_thunderstorm_nowcast import (
                predict_current_thunderstorm_risk,
            )

            independent = None
            api_again = body
            for attempt in range(3):
                independent = predict_current_thunderstorm_risk()
                if independent.get("feature_timestamp") == api_again.get("feature_timestamp"):
                    break
                # The clock rolled over between the two calls; re-fetch the API
                # prediction so both describe the same feature hour.
                api_again = http_get(base, "/api/prediction", timeout=120.0).get("json") or {}

            same_hour = independent.get("feature_timestamp") == api_again.get("feature_timestamp")
            api_probability = _as_float(api_again.get("probability"))
            engine_probability = _as_float(independent.get("probability"))
            record(
                same_hour,
                "the HTTP payload and an independent engine call describe the same feature hour",
                {
                    "api_feature_timestamp": api_again.get("feature_timestamp"),
                    "engine_feature_timestamp": independent.get("feature_timestamp"),
                },
            )
            record(
                same_hour
                and api_probability is not None
                and engine_probability is not None
                and abs(api_probability - engine_probability) < 1e-12
                and api_again.get("predicted_class") == independent.get("predicted_class")
                and api_again.get("risk_label") == independent.get("risk_label"),
                "the served probability equals the independently recomputed engine value",
                {
                    "api_probability": api_again.get("probability"),
                    "engine_probability": independent.get("probability"),
                    "api_class": api_again.get("predicted_class"),
                    "engine_class": independent.get("predicted_class"),
                },
            )
            record(
                (independent.get("source") or {}).get("frame_origin") == "live_open_meteo_forecast_endpoint",
                "the independent engine call was itself a live request (so equality is not a cached replay)",
                (independent.get("source") or {}).get("frame_origin"),
            )
            _EVIDENCE["live_prediction"] = {
                "probability": api_again.get("probability"),
                "predicted_class": api_again.get("predicted_class"),
                "risk_label": api_again.get("risk_label"),
                "threshold": api_again.get("threshold"),
                "lead_time_hours": api_again.get("lead_time_hours"),
                "prediction_timestamp": api_again.get("prediction_timestamp"),
                "feature_timestamp": api_again.get("feature_timestamp"),
                "target_timestamp": api_again.get("target_timestamp"),
                "model_identifier": api_again.get("model_identifier"),
                "served_grid_cell": api_again.get("served_grid_cell"),
                "data_age_hours": age,
                "independent_engine_probability": independent.get("probability"),
            }

            # ---------------- 11. retained endpoints -----------------------
            section("11. RETAINED PHASE 2 ENDPOINTS")

            proxy = http_get(base, "/api/prediction/proxy", timeout=120.0)
            proxy_body = proxy.get("json") or {}
            record(
                proxy.get("status") == 200
                and (proxy_body.get("model") or {}).get("decision_threshold_mm_per_3h") is not None,
                "the Phase 1 surrogate proxy is still served at /api/prediction/proxy",
                {
                    "http_status": proxy.get("status"),
                    "probability": proxy_body.get("probability"),
                    "risk_label": proxy_body.get("risk_label"),
                    "threshold_mm_per_3h": (proxy_body.get("model") or {}).get(
                        "decision_threshold_mm_per_3h"
                    ),
                },
            )
            record(
                proxy.get("status") == 200 and proxy_body.get("horizon_hours") == 3,
                "the proxy endpoint still reports its own 3-hour horizon (distinct from the nowcast)",
                {"horizon_hours": proxy_body.get("horizon_hours")},
            )

            for path, label, expect_key in (
                ("/api/history", "recent hourly atmospheric data", "hours"),
                ("/api/evaluation", "Phase 1 metrics", "metrics"),
                ("/api/artifacts/confusion_matrix.png", "whitelisted artifact", None),
                ("/api/scenario", "historical scenario", "probability"),
            ):
                result = http_get(base, path, timeout=180.0)
                record(
                    result.get("status") == 200,
                    f"retained endpoint {path} still works ({label})",
                    {"http_status": result.get("status")},
                )

            unknown = http_get(base, "/api/does-not-exist", timeout=30.0)
            record(
                unknown.get("status") == 404 and (unknown.get("json") or {}).get("code") == "not_found",
                "an unknown path returns a JSON 404, not an HTML error page",
                {"http_status": unknown.get("status"), "code": (unknown.get("json") or {}).get("code")},
            )

            # ---------------- 14. terminology ------------------------------
            section("14. SCIENTIFIC TERMINOLOGY GUARDRAILS")

            combined = json.dumps(api_again).lower()
            record(
                "not a confidence" in combined,
                "probability is explicitly stated not to be a confidence",
                (api_again.get("probability_semantics") or "")[:220],
            )
            record(
                (api_again.get("observed_outcome") or {}).get("available") is False,
                "the payload states the future observation is unavailable, not zero",
                api_again.get("observed_outcome"),
            )
            leak = api_again.get("leakage_guard") or {}
            record(
                all(value is False for value in leak.values()) and bool(leak),
                "leakage_guard confirms no future or target-derived value entered the features",
                leak,
            )
            record(
                "not the votv aerodrome ground observation"
                in str(api_again.get("ground_truth_note") or "").lower(),
                "the inputs are described as latest available data, not a station observation",
                api_again.get("ground_truth_note"),
            )

            # ---------------- 15. headers ----------------------------------
            section("15. CORS, CACHE AND HEADER BEHAVIOUR")

            cors = requests.get(f"{base}/api/prediction", timeout=120.0, headers={"Origin": "http://localhost:8000"})
            headers = dict(cors.headers)
            record(
                headers.get("Access-Control-Allow-Origin") in ("*", "http://localhost:8000"),
                "a cross-origin GET is allowed by CORS",
                {"Access-Control-Allow-Origin": headers.get("Access-Control-Allow-Origin")},
            )
            record(
                "no-store" in (headers.get("Cache-Control") or ""),
                "prediction responses are not cacheable",
                {"Cache-Control": headers.get("Cache-Control")},
            )
            record(
                headers.get("X-Content-Type-Options") == "nosniff",
                "content-type sniffing is disabled",
                {"X-Content-Type-Options": headers.get("X-Content-Type-Options")},
            )
            preflight = requests.options(
                f"{base}/api/prediction", timeout=30.0, headers={"Origin": "http://localhost:8000"}
            )
            allowed_methods = preflight.headers.get("Access-Control-Allow-Methods") or ""
            record(
                preflight.status_code in (200, 204) and "GET" in allowed_methods and "PUT" not in allowed_methods,
                "preflight advertises a read-only method set",
                {"status": preflight.status_code, "methods": allowed_methods},
            )
            write_attempt = requests.post(f"{base}/api/prediction", timeout=30.0)
            record(
                write_attempt.status_code == 405,
                "a write attempt against the prediction endpoint is refused with 405",
                {"status": write_attempt.status_code},
            )

        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
            log_handle.close()

    # ---------------- 12/13. failure handling --------------------------
    check_live_data_failure_handling()

    # ---------------- 16. integrity -----------------------------------
    check_integrity()

    after_manifest = {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in protected_paths()
        if (PROJECT_ROOT / relative).is_file()
    }
    changed_during_run = sorted(
        path
        for path in set(before_manifest) | set(after_manifest)
        if before_manifest.get(path) != after_manifest.get(path)
    )
    section("16f. NOTHING CHANGED WHILE THE SERVER RAN")
    record(
        not changed_during_run,
        "no protected artifact changed between the start and the end of the verification",
        {"files_compared": len(after_manifest), "changed": changed_during_run or "none"},
    )
    _EVIDENCE["protected_files_changed_during_run"] = changed_during_run

    # ---------------- summary -----------------------------------------
    passed = sum(1 for item in RESULTS if item["passed"])
    failed = [item for item in RESULTS if not item["passed"]]

    section("SUMMARY")
    print(f"checks passed : {passed}")
    print(f"checks failed : {len(failed)}")
    for item in failed:
        print(f"  FAILED: {item['check']}")
        print(f"          {item['evidence'][:400]}")

    summary = {
        "phase": "Phase 8 -- Flask backend verification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "checks_total": len(RESULTS),
        "checks_passed": passed,
        "checks_failed": len(failed),
        "result": "PASS" if not failed else "FAIL",
        "checks": RESULTS,
        "evidence": _EVIDENCE,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nmachine-readable summary: {VALIDATION_JSON.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"RESULT: {summary['result']}  ({passed}/{len(RESULTS)})")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
