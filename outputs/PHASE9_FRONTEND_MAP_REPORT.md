# Phase 9 — Frontend + Map + Spatial Visualization Report

**SIH26072** — AIML based nowcasting of thunderstorm and lightning  
**Phase:** Phase 9 (frontend + interactive map + spatial risk visualization)  
**Status:** COMPLETE  
**Validation:** `PASS` (49/49) — see `outputs/PHASE9_FRONTEND_VALIDATION.json`

Primary product (unchanged from Phase 8): **1-hour genuine thunderstorm nowcast**, locked threshold **0.0775**, served via `GET /api/prediction`.

---

## 1. Frontend architecture

The existing Phase 2/8 dashboard was **upgraded**, not replaced.

```
Browser (templates/index.html + static/dashboard.js + static/style.css)
        |
        |  GET /api/prediction   ← primary nowcast (Phase 7/8)
        |  GET /api/history      ← trend charts
        |  GET /api/evaluation   ← Phase 1 proxy reference (demoted)
        |  GET /api/scenario     ← Phase 1 historical demo (demoted)
        v
   Flask app.py  (minimal Phase 9 change: served-grid template context + nowcast disclaimer)
        |
        v
   backend/nowcast_service.py  (Phase 8 — untouched)
        |
        v
   ml/predict_thunderstorm_nowcast.py  (Phase 7 — untouched)
```

| Layer | Role |
|-------|------|
| `templates/index.html` | Decision-support layout: operational nowcast → map + atmosphere → trends → pipeline → demoted Phase 1 reference |
| `static/dashboard.js` | API client, operational states, risk render, Leaflet spatial layer, Chart.js trends |
| `static/style.css` | Operational blue-slate HUD styling; responsive desktop/laptop/mobile |
| `app.py` | Passes served Open-Meteo grid coordinates + IMD disclaimer into the template |

No prediction math runs in the browser. Probability is never labelled “confidence”.

---

## 2. API integration

| Endpoint | Dashboard use |
|----------|----------------|
| `GET /api/prediction` | **Primary.** Risk label, probability, threshold, lead time, timestamps, model id, data source, atmosphere, map marker |
| `GET /api/history` | Temperature/humidity + precipitation evolution charts |
| `GET /api/evaluation` | Phase 1 surrogate metrics (collapsed reference panel only) |
| `GET /api/scenario` | Phase 1 historical proxy demo (reference panel only) |

**Not used as primary:** `/api/prediction/proxy`.

Displayed fields from the live nowcast payload:

- Thunderstorm probability (model score)
- Risk / alert classification
- Locked threshold `0.0775`
- Lead time `1 hour`
- Feature hour + prediction timestamp
- Predicted target hour + data age
- Data source (Open-Meteo provenance)
- Model identifier `thunderstorm_nowcast_1h`
- Feature completeness
- Latest available atmospheric variables (7 cards)
- Measured Phase 6 skill (recall / precision) as context

On failure, live values are cleared before/after the request so stale numbers cannot appear current.

---

## 3. Map implementation

- **Library:** Leaflet 1.9.4 (CDN), with graceful fallback text if unavailable.
- **Centre / marker:** real served Open-Meteo grid cell **8.471002° N, 76.93298° E**.
- **Basemap:** OpenStreetMap tiles.
- **Risk layer:** `L.featureGroup` (`riskLayer`) holding only API-derived points.
- **Today:** exactly **one** coloured risk marker (alert vs no-alert).
- **Popup:** probability, threshold, lead time, coordinates, “single grid cell — not a spatial forecast grid”.

Coordinates are seeded from the template (training/served cell) and refreshed from `served_grid_cell` in the live `/api/prediction` response.

---

## 4. Spatial capability / limitation

| Capability | Status |
|------------|--------|
| Validated point / location-based risk at the trained grid cell | **Implemented** |
| Multi-point rendering when real API points exist | **Ready** (`renderSpatialRisk(points[])`) |
| True multi-cell spatial forecast grid | **Not supported** by the current single-cell trained model |
| Invented surrounding risk values | **Forbidden / not done** |
| Claimed 1 km (or other) resolution | **Forbidden / not claimed** |

Honest UI copy states: *“Single validated point (not a forecast grid)”* and that surrounding cells are not predicted.

---

## 5. UI components

1. **Header** — SIH identity, served-grid readout, LIVE status pill, refresh
2. **Operational nowcast card** — alert, probability bar, threshold, lead time, timestamps, model, meta strip, skill note
3. **Spatial risk view** — Leaflet map + legend + coordinate / capability strip
4. **Latest available atmosphere** — 7 condition cards
5. **Recent atmospheric evolution** — Chart.js temp/humidity + precipitation
6. **Nowcast pipeline + integration status** — active vs not-yet-integrated (radar, lightning, multi-cell grid, …)
7. **Reference panel (collapsed)** — Phase 1 evaluation, importance, confusion matrix, historical proxy scenario
8. **Footer** — IMD disclaimer + scientific wording (probability ≠ confidence; prediction ≠ observation)

### Operational states

| State | Meaning |
|-------|---------|
| Loading | Pending refresh; values cleared |
| LIVE · latest available data | Successful `/api/prediction` |
| API unavailable | Network / unreachable API |
| Prediction unavailable | Backend refused prediction |
| Data stale / unavailable | Stale upstream data codes |

---

## 6. Responsive behaviour

Verified viewports (no horizontal overflow):

- Mobile `375×812`
- Tablet `768×1024`
- Laptop / projector `1366×768`

Layout collapses: map/atmosphere stack, risk stats reflow, pipeline columns reduce, charts stack on narrow screens.

---

## 7. Validation results

Ran: `python frontend/verify_frontend_phase9.py` (HTTP + Playwright Chromium headless)

| Metric | Result |
|--------|--------|
| Checks passed | **49 / 49** |
| Result | **PASS** |

Covered: Flask start, dashboard load, primary `/api/prediction` consumption, real probability/threshold/lead/model/coords, Leaflet load, single risk marker, no 1 km claim, loading + failure states, assets, responsive overflow, no JS page errors, Phase 1–8 integrity.

### Exact displayed prediction (verification run)

| Field | Value |
|-------|-------|
| probability | `0.0` (`0.00%`) |
| predicted_class | `0` |
| risk_label | `No thunderstorm alert` |
| threshold | `0.0775` |
| lead_time | `1 hour` |
| model_identifier | `thunderstorm_nowcast_1h` |
| served grid | `8.471002° N, 76.93298° E` |

---

## 8. Protected-file integrity

| Check | Outcome |
|-------|---------|
| Phase 6 locked threshold | Still **0.0775** |
| Git-tracked Phase 1–8 science files vs staged blobs | No mismatches |
| Phase 8 `backend/nowcast_service.py` mtime | Unchanged during Phase 9 |
| Phase 7 prediction engine mtime | Unchanged during Phase 9 |
| Protected files during verification run | Unchanged |

**Not modified:** datasets, models, Phase 7 engine, Phase 8 service logic (only minimal `app.py` template context for the frontend).

---

## 9. Files created / modified

### Created
- `frontend/verify_frontend_phase9.py`
- `outputs/PHASE9_FRONTEND_VALIDATION.json`
- `outputs/PHASE9_FRONTEND_MAP_REPORT.md` (this file)

### Modified
- `templates/index.html` — decision-support upgrade
- `static/dashboard.js` — nowcast-first client + honest spatial layer
- `static/style.css` — presentation / responsive restyle
- `app.py` — served-grid + nowcast/IMD disclaimer template context only

---

## 10. Phase 9 completion

| Criterion | Met |
|-----------|-----|
| Real Phase 8 `/api/prediction` as primary | Yes |
| Probability / risk / threshold / 1 h / timestamps / source / model shown | Yes |
| Leaflet map at served grid cell | Yes |
| Scientifically honest point spatial viz (no fake neighbours / no 1 km claim) | Yes |
| Atmosphere cards + charts | Yes |
| Operational states | Yes |
| Responsive, SIH presentation quality | Yes |
| IMD disclaimer | Yes |
| Phase 1 reference preserved but demoted | Yes |
| Validation + report | Yes |
| No Phase 10 / no commit / no push | Honoured |

**Phase 9 is COMPLETE.**
