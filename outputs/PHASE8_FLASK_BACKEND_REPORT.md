# Phase 8 — Flask Backend Report

**SIH26072** — AIML based nowcasting of thunderstorm and lightning  
**Phase:** Phase 8 (Flask backend / API)  
**Status:** COMPLETE  
**Validation:** `PASS` (56/56) — see `outputs/PHASE8_BACKEND_VALIDATION.json`  
**Primary product:** 1-hour genuine thunderstorm nowcast (threshold **0.0775**)

---

## 1. Architecture

Phase 8 does **not** reimplement inference. It exposes the verified Phase 7 engine over HTTP.

```
Browser / frontend
        |
        v
   app.py  (Flask routes, CORS, retained Phase 2 endpoints)
        |
        +-- /api/prediction --------> backend/nowcast_service.py
        |                                      |
        |                                      v
        |                         ml/predict_thunderstorm_nowcast.py
        |                           (Phase 7 engine, 74/74 PASS)
        |                                      |
        |                                      +-- models/thunderstorm_nowcast_1h.joblib
        |                                      +-- dataset/feature_engineering_phase4.py
        |                                      +-- Open-Meteo forecast API (live)
        |
        +-- /api/prediction/proxy --> ml/predict.py  (Phase 1 surrogate, retained)
        +-- /api/history|evaluation|artifacts|scenario  (Phase 2, retained)
```

| Layer | Role |
|-------|------|
| `app.py` | Flask application entry point; routes; CORS/security headers; startup load |
| `backend/nowcast_service.py` | Thin service: one-time model load, engine call, HTTP error translation, health |
| `ml/predict_thunderstorm_nowcast.py` | **Only** source of prediction science (unchanged in Phase 8) |

**Design rules enforced**

- No weather value or probability is invented, imputed, defaulted, or carried over.
- A failed live-data call returns an HTTP error with prediction fields set to `null`.
- Probability ≠ confidence; latest available atmospheric data ≠ station observation; prediction ≠ future observation.
- The Phase 1 high-precipitation proxy remains at `/api/prediction/proxy` and is **not** the primary nowcast.

---

## 2. Existing Flask files found

| Path | Role before Phase 8 |
|------|---------------------|
| `app.py` | Phase 2 Flask dashboard/API serving the Phase 1 high-precipitation risk proxy |
| `templates/index.html` | Dashboard shell |
| `static/dashboard.js` | Client for `/api/prediction`, history, evaluation, scenario |
| `static/style.css` | Dashboard styles |
| `ml/predict.py` | Phase 1 surrogate prediction helpers |

Phase 8 **reused** this application: primary prediction was retargeted to the Phase 7 engine; proxy endpoints were kept.

---

## 3. Files created / modified

### Created

| Path | Purpose |
|------|---------|
| `backend/__init__.py` | Backend package marker |
| `backend/nowcast_service.py` | Phase 8 service layer over Phase 7 |
| `backend/verify_backend_phase8.py` | End-to-end HTTP verification (56 checks) |
| `outputs/PHASE8_BACKEND_VALIDATION.json` | Machine-readable verification summary |
| `outputs/PHASE8_FLASK_BACKEND_REPORT.md` | This report |

### Modified

| Path | Change |
|------|--------|
| `app.py` | Wire `/api/prediction` and `/api/health` to `nowcast_service`; keep proxy/history/evaluation/scenario |
| `templates/index.html` | Risk card labels updated for 1-hour thunderstorm nowcast |
| `static/dashboard.js` | Read nowcast fields (`threshold`, `feature_timestamp`, atmospheric conditions) |

### Not modified (protected)

- Phase 7 engine (`ml/predict_thunderstorm_nowcast.py`)
- Phase 6 models / metadata / evaluation artifacts
- Phase 1–5 datasets
- Decision threshold remains **0.0775**

---

## 4. Endpoints

| Method | Path | Behaviour |
|--------|------|-----------|
| `GET` | `/` | Dashboard HTML (200) |
| `GET` | `/api/health` | App + primary model + Phase 7 engine + optional live-provider probe. `?live=0` skips probe. HTTP 200 only when `status == "ok"`; otherwise 503 |
| `GET` | `/api/prediction` | **Primary:** live 1-hour thunderstorm nowcast via Phase 7 |
| `GET` | `/api/prediction/proxy` | Retained Phase 1 high-precipitation surrogate |
| `GET` | `/api/history` | Recent hourly atmospheric series for charts |
| `GET` | `/api/evaluation` | Phase 1 metrics (read-only) |
| `GET` | `/api/artifacts/<file>` | Whitelisted Phase 1 image |
| `GET` | `/api/scenario` | Historical Phase 1 test-set demonstration |

---

## 5. Request / response behaviour

### `GET /api/prediction` (success)

Returns the Phase 7 payload plus a thin API envelope. Required scientific fields include:

- `probability` — model score in `[0, 1]` (not confidence)
- `predicted_class` — `1` iff `probability >= 0.0775`
- `risk_label` — alert text for that class
- `threshold` — locked **0.0775**
- `lead_time_hours` — **1**
- `prediction_timestamp`, `feature_timestamp`, `target_timestamp`
- `source.provenance` — Open-Meteo request facts + served grid cell
- `model` / `model_identifier` — `thunderstorm_nowcast_1h`
- `feature_completeness` — 28/28 finite features, nothing imputed

Dashboard compatibility aliases (copies of engine values only):

- `observation_timestamp_utc` ← `feature_timestamp`
- `current_weather` ← `input_atmospheric_conditions`

### `GET /api/prediction` (live-data failure)

HTTP **503** (or **500** for local artifact faults) with JSON:

- `error: true`, machine-readable `code`, clear `message`
- `probability`, `predicted_class`, `risk_label`, `threshold` all **`null`**
- `no_prediction_produced: true`

A failure must **not** be interpreted as “no thunderstorm”.

### `GET /api/health`

Checks:

1. Application running (answered the request)
2. Primary model available (`thunderstorm_nowcast_1h.joblib` loaded once)
3. Prediction engine available (callable + Phase 4 feature module imports, 28 features)
4. Live-data dependency reachable (Open-Meteo probe; skippable with `?live=0`)

Statuses: `ok` | `degraded` | `unavailable`. Legacy proxy health is reported separately and does not affect nowcast health.

---

## 6. Model integration

| Item | Value |
|------|-------|
| Engine | `ml.predict_thunderstorm_nowcast.predict_current_thunderstorm_risk` |
| Artifact | `models/thunderstorm_nowcast_1h.joblib` |
| Model id | `thunderstorm_nowcast_1h` |
| Estimator | `RandomForestClassifier` |
| Features | 28 (Phase 4 causal set, same order as training) |
| Threshold | **0.0775** (locked in Phase 6; not changed) |
| Lead time | 1 hour (`target_1h`) |
| Load policy | Once per process (~144 MB); reused for every request |

The Flask layer performs **no** feature engineering, scoring, or thresholding of its own.

---

## 7. Live data integration

| Item | Detail |
|------|--------|
| Provider | Open-Meteo forecast endpoint |
| Request point | Training coordinates (Phase 3): 8.482 N, 76.920 E |
| Served grid cell | 8.471002 N, 76.93298 E (reported in payload) |
| Variables | Same seven Phase 4 inputs / units as training |
| Semantics | Latest available atmospheric data (model-derived grid), **not** VOTV METAR |
| Freshness | Engine refuses stale windows (default max age 3 h) |
| Health probe | Separate bounded reachability check; never used as prediction input |

---

## 8. Error handling & security (local prototype)

- JSON error bodies for 404 / 405 / 500 / prediction failures (no HTML traceback to API clients)
- CORS: configurable allow-list; default `*` for local prototype; methods limited to `GET, HEAD, OPTIONS`
- `Cache-Control: no-store` on `/api/*`
- `X-Content-Type-Options: nosniff`
- Write methods (`POST`, etc.) refused with 405 on prediction routes
- Simulated failures verified: connection error, timeout, upstream 503, non-JSON body, wrong units, missing variable — all return errors with null prediction fields

---

## 9. Validation results

Ran: `python backend/verify_backend_phase8.py`

| Metric | Result |
|--------|--------|
| Checks passed | **56 / 56** |
| Result | **PASS** |
| Generated | `2026-09-15T05:05:21Z` |

Covered:

1. Flask starts and answers  
2. `GET /` → 200 dashboard  
3. `GET /api/health` truthful (app, model, engine, live)  
4–9. Live `/api/prediction` fields, threshold class, lead time, timestamps, grid, model id, feature completeness  
10. HTTP payload equals independent Phase 7 engine call (no static/fake values)  
11. Retained Phase 2 endpoints still work  
12–13. Live-data failure handling + error body safety  
14. Scientific terminology guardrails  
15. CORS / cache / read-only headers  
16. Phase 1–7 protected artifact integrity (hashes, threshold, mtime, git blobs)

### Live API prediction captured during verification

| Field | Value |
|-------|-------|
| probability | `0.0` |
| predicted_class | `0` |
| risk_label | `No thunderstorm alert` |
| threshold | `0.0775` |
| lead_time_hours | `1` |
| feature_timestamp | `2026-09-15T05:00:00+00:00` |
| target_timestamp | `2026-09-15T06:00:00+00:00` |
| prediction_timestamp | `2026-09-15T05:04:34.302328+00:00` |
| model_identifier | `thunderstorm_nowcast_1h` |
| served_grid_cell | `{latitude: 8.471002, longitude: 76.93298}` |
| independent engine probability | `0.0` (exact match) |

---

## 10. Protected-file integrity

| Check | Outcome |
|-------|---------|
| Phase 3→4→5 dataset SHA-256 ancestry | Exact match |
| Phase 6 training-dataset hash vs on-disk nowcast CSV | Exact match |
| Locked threshold in artifact | Still **0.0775** |
| Feature count in artifact | Still **28** |
| Git-tracked Phase 1–7 files vs staged blobs | No mismatches |
| Mtime guard (no Phase 1–7 write during Phase 8) | Clean |
| No protected file changed during verification run | Clean |

Phase 7 engine and Phase 6 models were **not** retrained or edited for Phase 8.

---

## 11. How to run

```bash
python app.py
# then
curl http://127.0.0.1:5000/api/health
curl http://127.0.0.1:5000/api/prediction

python backend/verify_backend_phase8.py
```

Optional environment: `HOST`, `PORT`, `FLASK_DEBUG`, `CORS_ALLOWED_ORIGINS`.

---

## 12. Phase 8 completion

| Criterion | Met |
|-----------|-----|
| Flask starts successfully | Yes |
| `GET /`, `/api/health`, `/api/prediction` | Yes |
| Prediction uses Phase 7 engine | Yes |
| Real live prediction (no fakes) | Yes |
| Live-data failure → HTTP error, null prediction | Yes |
| Dashboard compatibility preserved / updated | Yes |
| Proxy endpoints retained | Yes |
| Phase 1–7 artifacts unchanged | Yes |
| Verification script + report | Yes |
| No commit / no push / no Phase 9 | Honoured |

**Phase 8 is COMPLETE.**
