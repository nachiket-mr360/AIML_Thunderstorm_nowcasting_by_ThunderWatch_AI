# Thunderstorm Nowcast Decision Support — VOTV (SIH 2026, Problem SIH26072)

A focused, scientifically honest **research prototype**: an AIML 1-hour thunderstorm nowcast for a
single station, trained on genuine observed thunderstorm reports, served through a Flask API and a
decision-support dashboard, and deployable in a container.

> **Prototype, not an operational forecast.** It is station-based, single-cell, and not an IMD
> product. Radar, satellite, lightning, NWP and multi-cell spatial prediction are **not integrated** —
> see [Planned but not integrated](#planned-but-not-integrated).

---
## Demo link: https://aiml-thunderstorm-nowcasting-by.onrender.com/
## Problem statement .

**Smart India Hackathon 2026 — SIH26072**

> AIML based nowcasting of thunderstorm and lightning using atmospheric observation including multiple
> radars, satellite, lightning and model data.

The full statement envisages a multi-source, spatial nowcasting system. This repository is a
deliberately bounded first milestone of that system: **one station, one lead time, one genuine
observation-based label source, and no radar/satellite/lightning/NWP inputs.**

---

## What this prototype actually does

Every hour the service:

1. fetches the latest available atmospheric hours for the VOTV (Thiruvananthapuram) grid cell from
   Open-Meteo,
2. derives the **28 causal features** at the latest usable hour `t` using the *same*
   `dataset/feature_engineering_phase4.py` module that built the training table,
3. scores a **Random Forest** trained on **genuine VOTV METAR thunderstorm observations**,
4. compares the score against the **locked threshold `0.0775`** and returns a two-level alert
   (`Thunderstorm alert` / `No thunderstorm alert`) for the clock hour `t + 1 h`.

If the live input cannot be obtained — provider unreachable, unexpected units, a missing variable, no
contiguous usable window, or data older than the 3-hour freshness limit — the engine **refuses**:
the API returns a JSON error whose prediction fields are explicitly `null`. No value is ever
imputed, defaulted, carried over from a previous request, or turned into an implicit "no
thunderstorm".

### In scope

| Item | Value |
| --- | --- |
| Station | VOTV, Thiruvananthapuram (8.482° N, 76.920° E request point) |
| Lead time served | 1 hour |
| Label source | Genuine VOTV METAR present-weather thunderstorm observations, 2014–2025 |
| Features | 28 causal atmospheric features at time `t` |
| Model | scikit-learn `RandomForestClassifier`, 400 trees, `class_weight='balanced'`, `random_state=42` |
| Decision threshold | `0.0775`, locked from Phase 6 |
| Live input | Open-Meteo forecast API (no API key) |
| Served grid cell | 8.4710016° N, 76.9329834° E, elevation 4 m (about 2 km from the aerodrome) |

### Explicitly out of scope

Nationwide forecasting · live radar ingest · satellite imagery · lightning detection or prediction ·
NWP model data · multi-cell / spatial thunderstorm tracking · official IMD warnings · calibrated
probabilities · production-grade operational forecasting.

---

## Architecture / pipeline

The work was staged in verifiable phases; each phase's evidence is preserved in `outputs/`.

```
                   dataset/raw_metar/            dataset/raw_openmeteo/
       METAR reports (IEM archive)      Open-Meteo archive / model grid series
                    |                                  |
        Phase 2     v                                  |
        build_thunderstorm_labels.py                   |
                    |                                  |
        Phase 3     v                                  v
        build_synchronized_dataset_phase3.py  ---> contiguous hourly grid
                    |        (genuine labels on a strict clock-hour ruler)
        Phase 4     v
        feature_engineering_phase4.py
                    |        28 causal features at time t (no future information)
        Phase 5     v
        nowcast_targets_phase5.py
                    |        forwards the label to t+1h / t+2h / t+3h  (target moves, features do not)
        Phase 6     v
        train_thunderstorm_nowcast_phase6.py
                    |        Random Forest + F2-based threshold selection on validation only
                    |        --> models/thunderstorm_nowcast_1h.joblib  (threshold locked at 0.0775)
        Phase 7     v
        ml/predict_thunderstorm_nowcast.py
                    |        live fetch -> same 28 features -> probability -> locked threshold
        Phase 8     v
        backend/nowcast_service.py  ->  app.py (Flask API)
                    |
        Phase 9     v
        templates/index.html + static/dashboard.js   (dashboard + map, Esri World Street Map)
                    |
        Phase 10    end-to-end technical audit (PASS)
```

Two models live in `models/` and they are **different products**:

| Artifact | Served at | Meaning |
| --- | --- | --- |
| `thunderstorm_nowcast_1h.joblib` | `/api/prediction` | The real 1-hour thunderstorm nowcast (target `target_1h` = genuine observed thunderstorm at `t+1h`) |
| `storm_risk_model.joblib` | `/api/prediction/proxy` | A Phase 1 **surrogate**: high-accumulated-precipitation proxy, *not* a thunderstorm or lightning label |

`thunderstorm_nowcast_2h.joblib` and `thunderstorm_nowcast_3h.joblib` were also trained and validated in
Phase 6; they are retained as evidence but are **not served** by this API.

---

## Data sources

| Source | Use | Notes |
| --- | --- | --- |
| **Iowa Environmental Mesonet (IEM) METAR archive** | Genuine VOTV thunderstorm labels 2000–2025 (273,125 unique report timestamps; the modelled window is 2014–2025) | Present-weather group decoded from the aerodrome routine reports |
| **Open-Meteo archive / model API** | Training and live atmospheric features at the VOTV grid cell | Model-derived grid-cell data, not a direct aerodrome observation |
| **Esri World Street Map tiles** | Map background in the dashboard | Key-free; loaded by the browser |
| **jsDelivr CDN** | Leaflet 1.9.4 and Chart.js 4.4.1 (front-end libraries) | Loaded by the browser |

No radar, satellite, lightning-detection or NWP dataset is used anywhere in this repository.

### The genuine thunderstorm label

* Labels come from **observed METAR present weather** at VOTV, not from a proxy, not from a
  reanalysis weather code, and never from synthetic or imputed rows.
* `weather_code` (the Open-Meteo code) is **not** used as a label: in this record it contains no
  thunderstorm codes at all, while thousands of hours carry genuine thunderstorm observations.
* Modelled window 2014-01-01T12:00Z → 2025-12-30T23:00Z: **82,018 hourly rows**, of which
  **3,694 hours carry a genuine thunderstorm observation** (about 4.5% of the modelled rows).
* Target `target_1h` is the genuine observation at the clock hour exactly 1 hour after the feature
  hour. Where that future hour was **not genuinely observed**, the target is left as `NaN` with an
  explicit `target_1h_observed = 0` flag — never written as `0` and never imputed. That costs 4,708
  rows, leaving **77,310 labelled rows** for the 1-hour task.
* The label series is shifted on a **verified contiguous hourly grid** so that one row step is always
  exactly one clock hour; a shift on the masked rows would have misstated the lead time.
* Splits are strictly chronological (`70% / 15% / 15%`), never shuffled: train 54,117, validation
  11,596, test 11,597 (2024-04-23 → 2025-12-30). The test split was scored once, after the threshold
  was already locked.

---

## The 28 features, at a high level

Everything is a function of hours `≤ t` only. No future atmospheric variable, no future
precipitation, no target-derived statistic, and no `weather_code` is used.

| Group | Count | Contents |
| --- | --- | --- |
| Current atmospheric state | 6 | `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`, `precipitation`, `cloud_cover` |
| Cyclic time encoding | 4 | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |
| Wind direction (cyclic) | 2 | `wind_direction_sin`, `wind_direction_cos` |
| Short-term change (1 h and 3 h) | 10 | temperature, humidity, pressure, wind speed and precipitation deltas |
| Recent history (rolling) | 6 | 3 h and 6 h precipitation accumulation, 3 h mean humidity, pressure, temperature and wind speed |

Each row therefore needs at least 6 preceding hours of history — the same minimum the training rows
had, which is why the live path also requires 6 preceding contiguous hours before it will score.

Indicative RF importance (mean decrease in impurity, 1-hour model): the two precipitation
accumulation features dominate, followed by wind speed and the seasonal encodings. These are
**model-attention diagnostics, not physical causation claims**.

---

## Model and verified metrics

**Model:** `RandomForestClassifier` — 400 trees, `criterion='gini'`, `max_features='sqrt'`,
`class_weight='balanced'`, `random_state=42`. No SMOTE, no oversampling, no synthetic rows; the test
set keeps its natural class distribution.

**Threshold:** `0.0775`, chosen on the **validation split only** by maximising F2 (beta = 2, recall
weighted twice as heavily as precision) subject to a predicted-positive rate cap of 0.2, then
**locked**. It was not re-tuned afterwards, and the test split was not used for model selection.

### Test split — 11,597 rows, 399 genuine thunderstorm hours (base rate 3.44%)

| Metric | Value |
| --- | --- |
| Decision threshold | 0.0775 |
| Recall (probability of detection) | **0.6040** |
| Precision | **0.1332** |
| F1 | 0.2183 |
| ROC-AUC | **0.8453** |
| PR-AUC (average precision) | 0.2060 |
| Accuracy | 0.8512 |
| Predicted alert rate | 15.60% |
| Confusion matrix (TP / FP / FN / TN) | 241 / 1,568 / 158 / 9,630 |
| False alarms per hit | 6.51 |

### Validation split — 11,596 rows (threshold selected here)

| Metric | Value |
| --- | --- |
| Recall | 0.6346 |
| Precision | 0.2024 |
| ROC-AUC | 0.8720 |
| PR-AUC | 0.2560 |
| False alarms per hit | 3.94 |

**How to read these numbers.** Accuracy (0.8512) sits below the trivial "always say no" baseline
(0.9656) and is therefore *not* evidence of skill. The meaningful figures are recall, precision,
PR-AUC and ROC-AUC relative to the 3.44% base rate. With the threshold set for recall, roughly
**1 in 7 alerts is a genuine thunderstorm hour and about 6.5 false alarms accompany each hit** — an
alert is a screening signal, not a warning. The full per-split evidence, confusion matrix, feature
importances and provenance are in `outputs/`.

---

## Live prediction behaviour

* **Provider:** Open-Meteo forecast API, requested at the same point used in training (8.482° N,
  76.920° E) so live features come from the same grid cell the model learned on.
* **Variables:** `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`,
  `wind_direction_10m`, `precipitation`, `cloud_cover` — the seven inputs the 28 features need.
* **Unit guard:** a unit mismatch (for example wind speed in m/s instead of km/h) is a hard failure,
  not something silently rescaled.
* **Freshness:** if the latest usable hour is more than 3 hours old, the engine refuses.
* **History:** at least 6 preceding contiguous hourly rows are required.
* **No fabrication:** on any failure the response carries `probability: null`,
  `predicted_class: null`, `risk_label: null` and an explicit error code. An error must never be
  read as "no thunderstorm".
* **Probability semantics:** the score is a model score for the positive class. It is **not** a
  confidence, **not** a calibrated thunderstorm frequency (the forest was trained with balanced class
  weights, which deliberately shifts scores toward the rare class), and **not** a physical
  probability. Only the score ordering and the locked threshold have been validated.
* **Ground truth caveat:** the observation for hour `t + 1 h` does not exist at prediction time; the
  payload says so explicitly rather than implying verification.

---

## Dashboard and map

`GET /` serves a decision-support dashboard (Phase 9) that:

* shows the current alert state, the score, the locked threshold and the served grid cell,
* exposes the input atmospheric conditions and the freshness/provenance of the live input,
* charts recent **observed** hours only — the trend chart never plots future values as if observed,
* marks the VOTV station on a Leaflet map over **key-free Esri World Street Map** tiles,
* states the limitations and the "prediction is not observation" caveat in the UI itself,
* surfaces API failures as an explicit error banner, never as a silent "no risk".

There is no radar layer, no satellite layer, no lightning layer and no spatial storm-cell overlay:
the map is location context, not a nowcast product.

---

## API endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Dashboard (HTML) |
| `GET /api/prediction` | **Primary product.** 1-hour thunderstorm nowcast from the Phase 7 engine |
| `GET /api/health` | Health of app, primary model, engine and live provider. `?live=0` skips the upstream probe. `200` only when everything is `ok`, otherwise `503` |
| `GET /api/history` | Recent **real** hourly atmospheric data for the trend chart (hours at or before now only) |
| `GET /api/prediction/proxy` | Retained Phase 1 **surrogate** high-precipitation proxy. A different product from `/api/prediction` |
| `GET /api/evaluation` | Read-only Phase 1 evaluation evidence (metrics, feature importance) |
| `GET /api/artifacts/<file>` | Read-only, whitelisted evaluation images/CSVs from `outputs/` |
| `GET /api/scenario` | Historical replay demonstration using the byte-identical Phase 1 chronological split |

The API is read-only and has no authentication, no cookies and no write endpoints.
`CORS_ALLOWED_ORIGINS` defaults to `*` and can be restricted to a comma-separated origin list.

### Example

```bash
curl -s http://localhost:10000/api/prediction
```

```json
{
  "probability": 0.005,
  "predicted_class": 0,
  "risk_label": "No thunderstorm alert",
  "threshold": 0.0775,
  "lead_time_hours": 1
}
```

(The payload also carries provenance, freshness, feature completeness and the scientific disclaimer.)

---

## Limitations

1. **One station, one lead time.** The model describes the VOTV clock-hour thunderstorm observation at
   `t + 1 h`. It is not a spatial or nationwide product.
2. **Modest precision.** Precision 0.1332 / recall 0.6040 on the test split means a high false-alarm
   burden. A single threshold hides the recall-precision trade-off an operator would want.
3. **Uncalibrated scores.** `class_weight='balanced'` shifts scores toward the rare class. The score is
   a ranking device; it is not a calibrated probability of thunderstorm occurrence.
4. **Grid-cell input, station label.** Features come from a model-derived Open-Meteo grid cell about
   2 km from the aerodrome; the label is the station observation. That mismatch is inherent to the
   design and is reported, not hidden.
5. **Provider dependence.** No live atmospheric input means no nowcast. Outages produce explicit
   refusals rather than stale alerts.
6. **Label coverage.** Only hours with a decodable present-weather observation can be supervised;
   unobserved hours are carried through as `NaN` and excluded per lead time instead of being assumed
   negative.
7. **Distribution shift / drift.** The model was trained on 2014–2025 data for one location; there is
   no online re-training, no drift monitoring and no recalibration in this prototype.
8. **Single-site generalisation.** Nothing here demonstrates skill at another airport or in an
   unseen regime.
9. **Not an official warning.** This software does not produce, and must not be presented as, an IMD
   or any other official meteorological warning.
10. **Demonstration capacity.** The deployment target is a small single-instance container with one
    worker; it is not dimensioned or hardened for operational forecasting traffic.

---

## Planned but not integrated

These are roadmap items from the full SIH26072 problem statement. **None of them is implemented, and
none of them contributes to the numbers reported above:**

* **Multiple radars** — no radar volume, reflectivity or dual-pol product is ingested or used.
* **Satellite** — no satellite channel, brightness temperature or cloud product is ingested or used.
* **Lightning** — there is no lightning-detection network input, and the model does **not** predict
  lightning. It predicts the station reporting a thunderstorm in a future clock hour.
* **NWP model data** — no numerical weather prediction fields are used; the only atmospheric source is
  the Open-Meteo model/archive grid series at one cell.
* **Multi-cell / spatial prediction** — there is no storm-cell identification, tracking, advection or
  spatial gridding. One grid cell, one station, one lead time.
* Additionally planned: 2 h and 3 h lead times as served products (both exist as Phase 6 evidence but
  are not exposed), calibrated probabilities, and end-to-end monitoring.

Radar, satellite, lightning and NWP integration would each change the model, the feature set and the
validation evidence, so they cannot be implied by, or bolted onto, the current metrics.

---

## Repository layout

```
app.py                      Flask application (Phase 8 entry point) + Phase 9 template context
backend/                    nowcast_service.py -- transport layer over the Phase 7 engine
ml/                         Phase 1 pipeline, Phase 6 training, Phase 7 live nowcast engine
dataset/                    raw archives, label builder, feature engineering, target builder, tables
models/                     Phase 6 trained artifacts (joblib bundles) + metadata
outputs/                    Phase 6-9 evidence: reports, evaluation JSON, feature importance, PNGs
templates/, static/         Phase 9 dashboard (HTML/CSS/JS)
frontend/                   Phase 9 dashboard copy and its verifier
requirements.txt            Deployment manifest (this image)
requirements_backend.txt    Phase 2 development manifest (historical)
requirements_ml.txt         Phase 1 development manifest, includes matplotlib (training only)
Dockerfile, .dockerignore   Container build for the existing application
```

The Phase 3-10 verifier scripts (`dataset/verify_*.py`, `backend/verify_backend_phase8.py`,
`ml/verify_prediction_engine_phase7.py`, `outputs/verify_phase6_models.py`) re-derive the published
numbers and hash-check protected artifacts. Treat them as audit tooling: they are not part of the
serving path, and some of them rewrite their own validation JSON when run.

---

## Local setup

Requires Python 3.12 (3.12.10 is the interpreter the model was trained and verified with).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

python app.py
# dashboard: http://127.0.0.1:5000/
```

Optional environment variables (all read at startup):

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | Bind address for the built-in development server |
| `PORT` | `5000` | Bind port for the built-in development server |
| `FLASK_DEBUG` | `0` | `1` enables the reloader (development only) |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated origin allow-list, or `*` |

No API keys are required. Live prediction needs outbound HTTPS access to Open-Meteo; the map and
charts need outbound access to the jsDelivr CDN and the Esri tile servers.

## Docker setup

```bash
docker build -t sih26072-nowcast .
docker run --rm -p 10000:10000 -e PORT=10000 sih26072-nowcast
# dashboard: http://localhost:10000/
```

The container:

* uses `python:3.12-slim-bookworm` (same interpreter minor version as training; Debian bookworm
  satisfies the manylinux wheels, and every dependency ships a prebuilt wheel, so no compiler is in
  the image),
* installs `requirements.txt` — `Flask`, `requests`, `numpy`, `pandas`, `scikit-learn`, `joblib`,
  `gunicorn` — and nothing else,
* runs `gunicorn` on `0.0.0.0:${PORT}` with a local default of `10000`,
* preloads the application once and runs a single worker with 4 threads, so the model bundles are read
  once and shared rather than duplicated,
* runs as a non-root user.

Container start command:

```bash
gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 120 \
         --graceful-timeout 30 --preload --forwarded-allow-ips='*' \
         --access-logfile - --error-logfile - app:app
```

`python app.py` (the Flask development server) remains available for local use and is **not** used in
the container.

## Deploying on Render

1. Push the repository to GitHub, then in Render choose **New → Web Service** and connect the repo.
2. Set **Language / Runtime** to **Docker** (Render then uses the root `Dockerfile`). Do not use the
   native Python runtime: it would bypass the container start command.
3. Leave the **Start Command** empty — the image's `CMD` already binds `0.0.0.0:$PORT`, and Render
   supplies `PORT`.
4. Choose an instance with **at least 1 GB of RAM**. Loading the two joblib bundles spikes memory well
   above the steady-state footprint during startup (measured peak about 540 MB on the development
   machine), so a 512 MB instance is likely to be killed while loading.
5. Optional: set the health check path to `/api/health`. It returns `200` only when the model, the
   engine **and** the Open-Meteo probe are all healthy, and `503` otherwise; use `/api/health?live=0`
   for a purely local check.
6. No environment variables and no API keys are required. Optionally set `CORS_ALLOWED_ORIGINS` to a
   comma-separated allow-list to restrict browser access.
7. Expect the first request after an idle period to be slower than the following ones, because the
   live prediction path contacts Open-Meteo.

The image is large (the Phase 6 artifacts and the dataset are committed to the repository, several
hundred megabytes), so the first build takes noticeably longer than a typical Flask image build.

---

## Scientific disclaimer

This is a research and demonstration prototype built for Smart India Hackathon 2026, Problem SIH26072.

It is a **station-based thunderstorm nowcast prototype** for the VOTV aerodrome, produced from a single
Open-Meteo grid cell and a Random Forest trained on genuine VOTV METAR thunderstorm observations. It is
**not** an official India Meteorological Department product, **not** a nationwide forecast, **not** a
live radar, satellite, lightning or NWP system, and **not** an operational warning service. A model
probability is not certainty, and an alert is a screening signal whose measured false-alarm burden is
reported alongside it. High accuracy figures are achievable by predicting the majority class and are
not evidence of skill; the meaningful figures are recall, precision, F1, PR-AUC and ROC-AUC relative to
the reported base rate.

Do not use this software as the sole basis for any safety-critical decision.
