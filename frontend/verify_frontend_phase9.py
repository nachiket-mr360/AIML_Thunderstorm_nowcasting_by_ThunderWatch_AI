"""Phase 9 verification -- frontend + map + spatial visualization.

SIH26072 -- thunderstorm nowcast decision-support dashboard.

WHAT THIS SCRIPT PROVES
-----------------------
1. Flask starts and serves the upgraded dashboard.
2. The dashboard consumes GET /api/prediction (not the proxy as primary).
3. Real probability, threshold 0.0775, 1-hour lead time, model id, timestamps,
   atmospheric cards, and data source are rendered from the live API.
4. Leaflet map centres on the served Open-Meteo grid cell
   (8.471002 N, 76.93298 E).
5. Spatial mode is point/location-based: no invented 1 km resolution and no
   fabricated surrounding risk cells.
6. Loading and API-failure operational states work; stale values are cleared.
7. Assets load; Playwright reports no page errors.
8. Responsive CSS avoids horizontal overflow at common viewports.
9. Phase 1-8 protected science artifacts are unchanged.

Run:
    python frontend/verify_frontend_phase9.py

Exit 0 == PASS. Machine-readable summary:
    outputs/PHASE9_FRONTEND_VALIDATION.json
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VALIDATION_JSON = OUTPUTS_DIR / "PHASE9_FRONTEND_VALIDATION.json"

PHASE9_WORK_START_UTC = datetime(2026, 9, 15, 5, 15, 0, tzinfo=timezone.utc)

PHASE9_OWNED_FILES = {
    "app.py",  # minimal template context for served grid / disclaimer only
    "templates/index.html",
    "static/dashboard.js",
    "static/style.css",
}

PHASE9_UNTRACKED_PROTECTED = (
    "ml/predict_thunderstorm_nowcast.py",
    "ml/verify_prediction_engine_phase7.py",
    "outputs/PHASE7_REALTIME_PREDICTION_ENGINE_REPORT.md",
    "backend/nowcast_service.py",
    "backend/verify_backend_phase8.py",
    "outputs/PHASE8_FLASK_BACKEND_REPORT.md",
    "outputs/PHASE8_BACKEND_VALIDATION.json",
)

LOCKED_THRESHOLD = 0.0775
EXPECTED_LEAD = 1
EXPECTED_MODEL = "thunderstorm_nowcast_1h"
EXPECTED_GRID = {"latitude": 8.471002, "longitude": 76.93298}

RESULTS: list[dict[str, Any]] = []
_EVIDENCE: dict[str, Any] = {}


def record(passed: bool, name: str, evidence: Any = "") -> bool:
    text = evidence if isinstance(evidence, str) else json.dumps(evidence, default=str)
    RESULTS.append({"check": name, "passed": bool(passed), "evidence": text[:1500]})
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
    if text:
        print(f"        {text[:700]}")
    return bool(passed)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


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


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def http_get(base: str, path: str, *, timeout: float = 90.0) -> dict[str, Any]:
    try:
        response = requests.get(f"{base}{path}", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "status": None, "json": None, "text": ""}
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


def start_server(port: int, log_path: Path) -> tuple[subprocess.Popen, Any]:
    env = dict(os.environ)
    env.update({"HOST": "127.0.0.1", "PORT": str(port), "FLASK_DEBUG": "0", "PYTHONIOENCODING": "utf-8"})
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
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {"status": None}
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return {"started": False, "reason": f"process exited with code {process.returncode}"}
        last = http_get(base, "/api/health?live=0", timeout=30.0)
        if last.get("status") == 200:
            return {"started": True}
        time.sleep(1.2)
    return {"started": False, "reason": "timeout", "last": last.get("status")}


def protected_paths() -> list[str]:
    tracked = []
    for line in git("ls-files", "-s").stdout.splitlines():
        if "\t" not in line:
            continue
        tracked.append(line.split("\t", 1)[1].strip())
    protected = [p for p in tracked if p not in PHASE9_OWNED_FILES]
    for extra in PHASE9_UNTRACKED_PROTECTED:
        if extra not in protected and (PROJECT_ROOT / extra).exists():
            protected.append(extra)
    return sorted(set(protected))


def check_integrity() -> None:
    section("PROTECTED ARTIFACT INTEGRITY")
    paths = protected_paths()
    existing = [p for p in paths if (PROJECT_ROOT / p).is_file()]
    record(len(existing) == len(paths), "every declared protected artifact is present",
           {"declared": len(paths), "present": len(existing)})

    # Threshold inside Phase 6 artifact must remain locked.
    import joblib

    bundle = joblib.load(PROJECT_ROOT / "models" / "thunderstorm_nowcast_1h.joblib")
    record(
        float(bundle.get("decision_threshold")) == LOCKED_THRESHOLD,
        "Phase 6 decision threshold remains 0.0775",
        {"threshold": bundle.get("decision_threshold")},
    )

    # Git-tracked protected files vs index blobs.
    index_blobs: dict[str, str] = {}
    for line in git("ls-files", "-s").stdout.splitlines():
        if "\t" not in line:
            continue
        meta, name = line.split("\t", 1)
        parts = meta.split()
        if len(parts) >= 2:
            index_blobs[name.strip()] = parts[1]

    to_hash = [p for p in paths if p in index_blobs and (PROJECT_ROOT / p).is_file()]
    mismatches = []
    if to_hash:
        hashed = git("hash-object", "--", *to_hash)
        working = hashed.stdout.splitlines()
        if len(working) != len(to_hash):
            mismatches.append({"error": "unexpected hash-object line count"})
        else:
            for name, digest in zip(to_hash, working):
                if digest.strip() != index_blobs[name]:
                    mismatches.append({"file": name, "index": index_blobs[name], "working": digest.strip()})
    record(not mismatches, "git-tracked Phase 1-8 science files match staged blobs",
           {"compared": len(to_hash), "mismatches": mismatches or "none"})

    # Backend service must not have been rewritten for Phase 9 (mtime / content).
    service = PROJECT_ROOT / "backend" / "nowcast_service.py"
    engine = PROJECT_ROOT / "ml" / "predict_thunderstorm_nowcast.py"
    for path, label in ((service, "Phase 8 nowcast_service"), (engine, "Phase 7 engine")):
        if not path.is_file():
            record(False, f"{label} present", "missing")
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        record(
            mtime < PHASE9_WORK_START_UTC,
            f"{label} was not rewritten during Phase 9 (mtime guard)",
            {"mtime_utc": mtime.isoformat(), "guard_from": PHASE9_WORK_START_UTC.isoformat()},
        )


def run_playwright(base: str, live_prediction: dict[str, Any]) -> None:
    section("BROWSER / HEADLESS DASHBOARD CHECKS")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        record(False, "Playwright is available", str(exc))
        return

    console_errors: list[str] = []
    page_errors: list[str] = []
    failed_requests: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1366, "height": 768})
        page = context.new_page()

        page.on("console", lambda msg: console_errors.append(f"{msg.type}: {msg.text}")
                if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.on("requestfailed", lambda req: failed_requests.append(f"{req.url} :: {req.failure}"))

        page.goto(f"{base}/", wait_until="networkidle", timeout=120_000)

        # Wait until live prediction render finishes (status LIVE or error).
        page.wait_for_function(
            """() => {
                const pill = document.getElementById('live-status');
                if (!pill) return false;
                const state = pill.getAttribute('data-state');
                return state && state !== 'pending';
            }""",
            timeout=120_000,
        )
        # Give map a moment to place the marker.
        page.wait_for_timeout(800)

        html = page.content()
        body_text = page.inner_text("body")

        record(
            "Thunderstorm Nowcast" in body_text or "Operational Nowcast" in body_text,
            "dashboard loads with nowcast decision-support chrome",
            {"title": page.title()},
        )

        # Primary API must be /api/prediction, not proxy.
        api_calls = []
        # Re-fetch with request listener by reloading under capture.
        captured: list[str] = []

        def on_request(req):
            if "/api/" in req.url:
                captured.append(req.url)

        page.on("request", on_request)
        page.click("#refresh-button")
        page.wait_for_function(
            """() => {
                const pill = document.getElementById('live-status');
                const state = pill && pill.getAttribute('data-state');
                return state && state !== 'pending';
            }""",
            timeout=120_000,
        )
        page.wait_for_timeout(500)
        api_calls = captured[:]

        record(
            any("/api/prediction" in u and "/proxy" not in u for u in api_calls),
            "dashboard consumes GET /api/prediction as primary",
            api_calls,
        )
        record(
            not any(u.rstrip("/").endswith("/api/prediction/proxy") for u in api_calls),
            "dashboard does not call /api/prediction/proxy as primary refresh",
            [u for u in api_calls if "prediction" in u],
        )

        last = page.evaluate("() => window.__PHASE9_LAST_PREDICTION__")
        record(isinstance(last, dict) and last.get("error") is not True,
               "live prediction object attached after successful render",
               {"keys": sorted(list(last.keys()))[:20] if isinstance(last, dict) else type(last).__name__})

        if isinstance(last, dict) and last.get("probability") is not None:
            prob_text = page.inner_text("#probability")
            thr_text = page.inner_text("#threshold")
            lead_text = page.inner_text("#lead-time")
            model_text = page.inner_text("#model-identifier")
            feature_text = page.inner_text("#feature-timestamp")
            source_text = page.inner_text("#data-source")
            coords_text = page.inner_text("#served-coords")
            spatial_text = page.inner_text("#spatial-capability")
            risk_text = page.inner_text("#risk-label")

            displayed_prob = float(last["probability"])
            record(
                abs(displayed_prob - float(live_prediction.get("probability", -1))) < 1e-9
                or abs(displayed_prob - float(live_prediction.get("probability", -1))) < 1e-6,
                "displayed prediction matches a live /api/prediction payload",
                {
                    "api_probability": live_prediction.get("probability"),
                    "ui_attached_probability": last.get("probability"),
                    "ui_probability_text": prob_text,
                    "risk_label": risk_text,
                },
            )
            _EVIDENCE["displayed_prediction"] = {
                "probability": last.get("probability"),
                "probability_text": prob_text,
                "predicted_class": last.get("predicted_class"),
                "risk_label": risk_text,
                "threshold_text": thr_text,
                "lead_time_text": lead_text,
                "model_identifier": model_text,
                "feature_timestamp_text": feature_text,
                "data_source_text": source_text,
                "served_coords_text": coords_text,
            }

            record("0.0775" in thr_text, "locked threshold 0.0775 is displayed", thr_text)
            record("1 hour" in lead_text.lower() or lead_text.strip().startswith("1"),
                   "1-hour lead time is displayed", lead_text)
            record(EXPECTED_MODEL in model_text, "model identifier thunderstorm_nowcast_1h is displayed", model_text)
            record("confidence" not in page.inner_text("#risk-section").lower()
                   or "not confidence" in page.inner_text("#risk-section").lower(),
                   "probability is not presented as confidence",
                   page.inner_text("#skill-note")[:240])
            record(
                "8.471002" in coords_text and "76.93298" in coords_text,
                "served grid coordinates 8.471002 N, 76.93298 E are displayed",
                coords_text,
            )
            record(
                "point" in spatial_text.lower() and "not a forecast grid" in spatial_text.lower(),
                "map labelled as point/location-based risk, not a forecast grid",
                spatial_text,
            )
            record(
                "1 km" not in body_text.lower() and "1km" not in body_text.lower(),
                "dashboard does not invent a 1 km spatial resolution",
                "no '1 km' / '1km' claims found" if ("1 km" not in body_text.lower()) else "FOUND",
            )

            # Atmosphere cards present.
            condition_count = page.locator("#conditions-grid .condition").count()
            record(condition_count == 7, "seven atmospheric condition cards rendered",
                   {"count": condition_count})

            # Map marker count == 1 (no fabricated neighbours).
            marker_count = page.evaluate(
                """() => {
                    if (typeof L === 'undefined') return -1;
                    // Count custom risk markers in the DOM.
                    return document.querySelectorAll('.risk-marker').length;
                }"""
            )
            record(marker_count == 1, "exactly one risk marker on the map (no fake surrounding cells)",
                   {"risk_markers": marker_count})

            map_has_leaflet = page.evaluate(
                """() => typeof L !== 'undefined' &&
                         !!(document.querySelector('#map.leaflet-container') ||
                            document.querySelector('#map .leaflet-pane'))"""
            )
            record(bool(map_has_leaflet), "Leaflet map container loaded",
                   {"leaflet": map_has_leaflet})

        # Loading state: hang fetch briefly so pending/busy is observable.
        page.evaluate(
            """() => {
              window.__phase9_orig_fetch = window.fetch.bind(window);
              window.fetch = function(url, opts) {
                if (String(url).includes('/api/prediction')) {
                  return new Promise(function() { /* intentionally pending */ });
                }
                return window.__phase9_orig_fetch(url, opts);
              };
            }"""
        )
        page.click("#refresh-button")
        page.wait_for_timeout(350)
        pending_seen = page.evaluate(
            """() => {
                const pill = document.getElementById('live-status');
                const busy = document.getElementById('refresh-button')?.disabled === true;
                const state = pill && pill.getAttribute('data-state');
                return { pending: state === 'pending', busy: !!busy, state: state };
            }"""
        )
        record(
            pending_seen.get("pending") is True and pending_seen.get("busy") is True,
            "refresh triggers loading state (pending status and disabled button)",
            pending_seen,
        )
        # Restore fetch and reload so the hung promise cannot leave the UI stuck.
        page.evaluate(
            """() => {
              if (window.__phase9_orig_fetch) {
                window.fetch = window.__phase9_orig_fetch;
              }
            }"""
        )
        page.reload(wait_until="networkidle", timeout=120_000)
        page.wait_for_function(
            """() => {
                const pill = document.getElementById('live-status');
                const state = pill && pill.getAttribute('data-state');
                return state && state !== 'pending';
            }""",
            timeout=120_000,
        )

        # Failure state: route prediction to fail, ensure values clear + banner.
        page.route("**/api/prediction", lambda route: route.fulfill(
            status=503,
            content_type="application/json",
            body=json.dumps({
                "error": True,
                "code": "upstream_unreachable",
                "message": "simulated upstream failure for Phase 9 validation",
                "probability": None,
                "predicted_class": None,
                "risk_label": None,
                "threshold": None,
                "no_prediction_produced": True,
            }),
        ))
        page.click("#refresh-button")
        page.wait_for_function(
            """() => {
                const pill = document.getElementById('live-status');
                const state = pill && pill.getAttribute('data-state');
                return state === 'prediction_unavailable' || state === 'api_unavailable' || state === 'stale';
            }""",
            timeout=60_000,
        )
        banner_visible = page.evaluate("() => !document.getElementById('error-banner')?.hidden")
        prob_cleared = page.inner_text("#probability").strip() in ("—", "\u2014", "-", "")
        markers_after_fail = page.evaluate("() => document.querySelectorAll('.risk-marker').length")
        record(banner_visible and prob_cleared,
               "API failure shows error state and clears prediction values",
               {"banner": banner_visible, "probability": page.inner_text("#probability"),
                "status": page.get_attribute("#live-status", "data-state")})
        record(markers_after_fail == 0,
               "API failure removes risk markers (no stale spatial values)",
               {"markers": markers_after_fail})
        page.unroute("**/api/prediction")

        # Assets.
        for asset in ("/static/style.css", "/static/dashboard.js"):
            resp = http_get(base, asset, timeout=30.0)
            record(resp.get("status") == 200 and len(resp.get("text") or "") > 100,
                   f"asset {asset} loads",
                   {"status": resp.get("status"), "bytes": len(resp.get("text") or "")})

        # Responsive: no horizontal overflow at mobile / laptop widths.
        overflow_report = {}
        for width, height in ((375, 812), (768, 1024), (1366, 768)):
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(200)
            overflow = page.evaluate(
                """() => {
                    const doc = document.documentElement;
                    return {
                      scrollWidth: doc.scrollWidth,
                      clientWidth: doc.clientWidth,
                      overflow: doc.scrollWidth > doc.clientWidth + 1
                    };
                }"""
            )
            overflow_report[f"{width}x{height}"] = overflow
            record(not overflow["overflow"],
                   f"no horizontal overflow at {width}x{height}",
                   overflow)

        # Console / page errors (filter known benign / intentional test noise).
        serious_console = [
            e for e in console_errors
            if "favicon" not in e.lower()
            and "503" not in e  # intentional failure-state probe
            and "service unavailable" not in e.lower()
        ]
        serious_failed = [
            u for u in failed_requests
            if "favicon" not in u.lower() and "/api/prediction" not in u.lower()
        ]
        record(not page_errors, "no uncaught JavaScript page errors", page_errors or "none")
        record(not serious_console, "no JavaScript console errors", serious_console or "none")
        record(not serious_failed, "no broken network assets during dashboard session",
               serious_failed or "none")

        # Static content honesty checks on HTML source.
        record("Point / location-based risk" in html or "point / location-based" in html.lower(),
               "HTML states point/location-based risk capability",
               "found" if "point" in html.lower() else "missing")
        record("not a replacement for official IMD" in body_text.lower()
               or "not a replacement for official IMD" in html,
               "IMD scientific disclaimer present",
               "found")
        record("/api/prediction/proxy" not in page.inner_text("main").lower(),
               "main dashboard copy does not promote the proxy endpoint as primary",
               "ok")

        browser.close()

    _EVIDENCE["console_errors"] = console_errors
    _EVIDENCE["page_errors"] = page_errors
    _EVIDENCE["failed_requests"] = failed_requests


def main() -> int:
    section("PHASE 9 FRONTEND + MAP VERIFICATION")
    print(f"project root : {PROJECT_ROOT}")
    print(f"python       : {sys.version.split()[0]}")
    print(f"started at   : {datetime.now(timezone.utc).isoformat()}")

    before_manifest = {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in protected_paths()
        if (PROJECT_ROOT / relative).is_file()
    }

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = PROJECT_ROOT / "outputs" / "_phase9_server.log"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    section("1. FLASK STARTS")
    process, log_handle = start_server(port, log_path)
    live_prediction: dict[str, Any] = {}
    try:
        started = wait_for_server(base, process)
        record(started.get("started") is True, "python app.py starts and answers /api/health",
               started if not started.get("started") else {"base": base})

        section("2. DASHBOARD + API")
        home = http_get(base, "/", timeout=60.0)
        record(home.get("status") == 200 and "text/html" in str(home.get("headers", {}).get("Content-Type", "")),
               "GET / returns 200 HTML dashboard",
               {"status": home.get("status"), "bytes": len(home.get("text") or "")})
        html = home.get("text") or ""
        record("served_grid" in html or "8.471002" in html,
               "dashboard HTML embeds the served grid coordinates",
               {"has_8_471002": "8.471002" in html, "has_76_93298": "76.93298" in html})
        record("dashboard.js" in html and "leaflet" in html.lower(),
               "dashboard references Leaflet and dashboard.js",
               {"leaflet": "leaflet" in html.lower(), "dashboard_js": "dashboard.js" in html})

        prediction = http_get(base, "/api/prediction", timeout=120.0)
        body = prediction.get("json") or {}
        record(prediction.get("status") == 200 and body.get("error") is not True,
               "GET /api/prediction returns a live nowcast for the UI",
               {"status": prediction.get("status"), "probability": body.get("probability")})
        live_prediction = body
        _EVIDENCE["live_api_prediction"] = {
            "probability": body.get("probability"),
            "predicted_class": body.get("predicted_class"),
            "risk_label": body.get("risk_label"),
            "threshold": body.get("threshold"),
            "lead_time_hours": body.get("lead_time_hours"),
            "model_identifier": body.get("model_identifier"),
            "feature_timestamp": body.get("feature_timestamp"),
            "served_grid_cell": body.get("served_grid_cell"),
        }
        record(
            isinstance(body.get("probability"), (int, float)) and 0.0 <= float(body["probability"]) <= 1.0,
            "live probability is in [0, 1]",
            body.get("probability"),
        )
        record(body.get("threshold") == LOCKED_THRESHOLD, "API threshold is 0.0775", body.get("threshold"))
        record(body.get("lead_time_hours") == EXPECTED_LEAD, "API lead time is 1 hour", body.get("lead_time_hours"))
        record(body.get("model_identifier") == EXPECTED_MODEL, "API model identifier correct",
               body.get("model_identifier"))
        served = body.get("served_grid_cell") or {}
        record(
            abs(float(served.get("latitude", 0)) - EXPECTED_GRID["latitude"]) < 1e-5
            and abs(float(served.get("longitude", 0)) - EXPECTED_GRID["longitude"]) < 1e-4,
            "API served grid matches training cell",
            served,
        )

        # Static source honesty: dashboard.js must not call the proxy as primary.
        js = (PROJECT_ROOT / "static" / "dashboard.js").read_text(encoding="utf-8")
        record('Api.get("/api/prediction")' in js,
               "dashboard.js primary fetch is /api/prediction",
               "found")
        record("/api/prediction/proxy" not in js,
               "dashboard.js never references /api/prediction/proxy",
               "absent" if "/api/prediction/proxy" not in js else "PRESENT")
        record("renderSpatialRisk" in js,
               "spatial renderer (renderSpatialRisk) exists for point/multi-point API points",
               {"has_renderSpatialRisk": "renderSpatialRisk" in js})
        record("1 km" not in js.lower() and "1km" not in js.lower(),
               "dashboard.js does not hard-code a 1 km resolution claim",
               "ok")

        run_playwright(base, live_prediction)

    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        log_handle.close()

    check_integrity()

    after_manifest = {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in protected_paths()
        if (PROJECT_ROOT / relative).is_file()
    }
    changed = sorted(
        path for path in set(before_manifest) | set(after_manifest)
        if before_manifest.get(path) != after_manifest.get(path)
    )
    section("INTEGRITY DURING RUN")
    record(not changed, "no protected artifact changed during verification",
           {"changed": changed or "none"})

    passed = sum(1 for item in RESULTS if item["passed"])
    failed = [item for item in RESULTS if not item["passed"]]
    section("SUMMARY")
    print(f"checks passed : {passed}")
    print(f"checks failed : {len(failed)}")
    for item in failed:
        print(f"  FAILED: {item['check']}")
        print(f"          {item['evidence'][:400]}")

    summary = {
        "phase": "Phase 9 -- frontend + map + spatial visualization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "checks_total": len(RESULTS),
        "checks_passed": passed,
        "checks_failed": len(failed),
        "result": "PASS" if not failed else "FAIL",
        "checks": RESULTS,
        "evidence": _EVIDENCE,
    }
    VALIDATION_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nmachine-readable summary: {VALIDATION_JSON.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"RESULT: {summary['result']}  ({passed}/{len(RESULTS)})")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
