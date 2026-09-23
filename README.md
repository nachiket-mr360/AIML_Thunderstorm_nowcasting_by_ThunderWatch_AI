Absolutely. One correction before you paste it: I checked the **currently deployed dashboard** and the current GitHub repository. The GitHub web view still shows an older README, while the deployed application reflects the newer five-location Model B system. So the README below is written for the **current ThunderWatch AI system**, not the old single-location README. ([Thunderwatch AI][1])

Below is the complete `README.md`. Replace your current README with this:

````markdown
# ⚡ ThunderWatch AI

> **AI-Powered Thunderstorm Nowcasting & Atmospheric Risk Intelligence**

## 🚀 Live Demo

### 👉 https://aiml-thunderstorm-nowcasting-by.onrender.com/

**Smart India Hackathon 2026 · Problem Statement SIH26072 · FANTASTIC6**

---

## 🌩️ What is ThunderWatch AI?

**ThunderWatch AI** is a multi-location AI research prototype designed to explore **short-term thunderstorm nowcasting using atmospheric observations, numerical weather prediction (NWP), and additional multimodal weather evidence**.

The project was developed for **Smart India Hackathon 2026 — Problem Statement SIH26072**:

> **“AIML based Nowcasting of thunderstorm and lightning using atmospheric observation including multiple radars, satellite, lightning and model data.”**

ThunderWatch AI focuses on a scientifically controlled question:

> **Given the atmospheric and NWP information available at time T, can an AI model estimate the likelihood of a thunderstorm being observed at a reference location at T+1 hour, T+2 hours, and T+3 hours?**

The current system combines:

- Genuine historical thunderstorm observations
- Multi-location atmospheric data
- Causal feature engineering
- GFS numerical weather prediction data
- Random Forest machine learning
- Multi-lead +1h / +2h / +3h prediction
- Historical replay
- Live research-mode prediction
- Five-location risk visualization
- INSAT-3DR satellite case evidence
- Radar data availability research
- Lightning data availability research
- Fail-closed inference
- Reproducible validation
- Docker-based deployment
- Responsive web dashboard

ThunderWatch AI is intentionally presented as a **research prototype**, not as an official meteorological warning system.

---

# 🎯 1. Why did we build ThunderWatch AI?

Thunderstorms are highly dynamic atmospheric phenomena.

They can produce:

- Lightning
- Heavy rainfall
- Strong winds
- Hail
- Dangerous outdoor conditions
- Aviation hazards
- Transportation disruption
- Agricultural impacts
- Infrastructure risks

A useful thunderstorm intelligence system therefore needs more than a simple weather display.

The objective of ThunderWatch AI is to explore how **machine learning can transform atmospheric and numerical weather prediction information into short-term, location-based thunderstorm risk signals**.

The project was designed around several principles:

1. Use genuine observations whenever possible.
2. Avoid fabricated radar or lightning data.
3. Prevent future-information leakage.
4. Use genuine future observations as forecasting targets.
5. Separate historical replay from live inference.
6. Make model limitations visible.
7. Fail closed when required data are unavailable.
8. Preserve reproducibility.
9. Keep scientific evidence separate from unsupported claims.
10. Build toward the multimodal architecture described by SIH26072 without pretending unavailable data already exist.

---

# 🧩 2. Problem Statement

## Smart India Hackathon 2026

### Problem Statement: SIH26072

**AIML based Nowcasting of thunderstorm and lightning using atmospheric observation including multiple radars, satellite, lightning and model data.**

The problem statement calls for a multimodal AI system capable of using different atmospheric observation sources.

These can include:

- Atmospheric observations
- Radar
- Satellite
- Lightning
- Numerical weather prediction/model data

ThunderWatch AI implements a scientifically validated subset of this architecture and explicitly documents the status of the remaining data layers.

---

# 🧠 3. What problem does the system solve?

ThunderWatch AI addresses the following research problem:

```text
Atmospheric conditions at time T
              +
NWP information available at T
              ↓
        AI MODEL
              ↓
     Thunderstorm risk
              ↓
      +1h / +2h / +3h
````

The system predicts whether a thunderstorm observation is likely at a supported reference station at future hourly lead times.

The current model is:

* station-based
* multi-location
* multi-lead
* research-oriented
* based on historical observations and NWP predictors

---

# 📍 4. Supported Reference Locations

ThunderWatch AI currently supports five reference locations:

| Station | Location           | Approx. Latitude | Approx. Longitude |
| ------- | ------------------ | ---------------: | ----------------: |
| VOTV    | Thiruvananthapuram |           8.4667 |           76.9500 |
| VECC    | Kolkata            |          22.6547 |           88.4467 |
| VIDP    | Delhi              |          28.5667 |           77.1167 |
| VOCI    | Kochi              |          10.1500 |           76.4000 |
| VABB    | Mumbai             |          19.1005 |           72.8585 |

The system is therefore a **multi-location prototype**, not a single-city application.

---

# 🏗️ 5. High-Level Architecture

```text
                    ┌───────────────────────────────┐
                    │         DATA SOURCES           │
                    ├───────────────────────────────┤
                    │ METAR observations             │
                    │ Open-Meteo atmospheric data    │
                    │ GFS NWP                        │
                    │ INSAT-3DR satellite evidence   │
                    │ Radar research/availability    │
                    │ Lightning research/availability│
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ DATA SYNCHRONIZATION           │
                    │ Station + UTC timestamp         │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ CAUSAL FEATURE ENGINEERING     │
                    │ Atmospheric + NWP predictors   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ FUTURE TARGET GENERATION       │
                    │ T+1 / T+2 / T+3                │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ RANDOM FOREST MODEL B           │
                    │ 83 atmospheric + NWP features │
                    └───────────────┬───────────────┘
                                    │
                     ┌──────────────┴───────────────┐
                     │                              │
                     ▼                              ▼
          ┌──────────────────────┐       ┌──────────────────────┐
          │ HISTORICAL REPLAY    │       │ LIVE RESEARCH        │
          │ Validated historical │       │ Open-Meteo + GFS     │
          │ feature dataset      │       │ Current/near-current │
          └──────────┬───────────┘       └──────────┬───────────┘
                     │                              │
                     └──────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │       THUNDERWATCH AI         │
                    │           DASHBOARD            │
                    ├───────────────────────────────┤
                    │ Maps                          │
                    │ Risk                          │
                    │ +1h / +2h / +3h               │
                    │ Atmospheric signals            │
                    │ NWP context                   │
                    │ Satellite evidence             │
                    └───────────────────────────────┘
```

---

# 📊 6. Data Strategy

A major part of ThunderWatch AI is the distinction between:

* Training-ready data
* Validation data
* Experimental data
* Historical case evidence
* Reference-only data
* Conditional institutional data
* Unavailable data

Not every weather data source has a reproducible historical archive.

Instead of pretending that unavailable radar or lightning data exist, the project explicitly records their status.

This makes the system more scientifically defensible.

---

# 🌦️ 7. Historical Atmospheric Data

The primary historical atmospheric source was **Open-Meteo historical weather data**.

The historical collection covers:

```text
2014-01-01 → 2025-12-31
```

at hourly UTC resolution.

The five locations were collected using a reusable multi-location acquisition pipeline.

### Atmospheric variables

The historical atmospheric dataset contains:

* `temperature_2m`
* `relative_humidity_2m`
* `surface_pressure`
* `wind_speed_10m`
* `wind_direction_10m`
* `precipitation`
* `cloud_cover`

The multi-location atmospheric collection contains:

```text
525,960 station-hour rows
```

The data were audited for:

* missing predictors
* duplicate timestamps
* timestamp continuity
* station/timestamp synchronization

---

# ⛈️ 8. Genuine Thunderstorm Labels

A critical design decision was to avoid using precipitation as a replacement for thunderstorm observations.

The primary historical target is based on **genuine METAR present-weather observations**.

The project audited METAR observations for:

* VOTV
* VECC
* VIDP
* VOCI
* VABB

For the VOTV reference dataset:

```text
82,021 observed rows
3,694 thunderstorm-positive hours
78,327 non-thunderstorm hours
```

The multi-location synchronization process created a common station/time framework for the five locations.

The final pooled labeled modeling universe contained approximately:

```text
480,645 usable labeled rows
```

after the required feature and target alignment.

---

# 🔬 9. Why Genuine Observations Matter

Thunderstorm labels must represent actual thunderstorm observations rather than an unrelated proxy.

Therefore ThunderWatch AI does **not** claim:

```text
precipitation = thunderstorm
```

and does not treat generic weather information as equivalent to lightning.

The project distinguishes:

```text
METAR thunderstorm observation
        ≠
Lightning observation
        ≠
Satellite cloud observation
        ≠
Radar observation
```

Each represents a different information layer.

---

# 🧮 10. Causal Feature Engineering

The feature engineering pipeline follows a strict causal rule:

> At prediction time T, only information available at or before T may be used.

The system creates features from:

* current atmospheric conditions
* time cycles
* wind direction
* short-term changes
* rolling atmospheric history
* precipitation accumulations
* location information
* validated NWP information

The atmospheric feature builder produces:

```text
73 atmospheric/location features
```

before NWP augmentation.

---

# 🛡️ 11. Leakage Prevention

Leakage prevention was explicitly tested.

A leakage-truncation probe removed future rows from the available history and recomputed the features.

The resulting feature values were compared with the original calculations.

Result:

```text
Maximum absolute difference = 0
```

This confirms that the feature builder respects the intended causal history under the tested truncation condition.

---

# ⏩ 12. Future Targets

For every prediction timestamp `T`:

```text
target_1h = thunderstorm at T + 1 hour

target_2h = thunderstorm at T + 2 hours

target_3h = thunderstorm at T + 3 hours
```

The future observation is used only as the target.

It is never included as a predictor.

If the future observation does not exist, the target remains unavailable rather than being silently converted into a negative example.

---

# 🤖 13. Machine Learning

ThunderWatch AI uses **Random Forest** models.

The main frozen deployed model is:

```text
ThunderWatch AI Model B
```

Model B uses:

```text
83 atmospheric + NWP features
```

and produces:

```text
+1 hour
+2 hours
+3 hours
```

predictions.

---

# 🧠 14. Model B

Model B combines:

### Atmospheric features

The validated 73-feature atmospheric/location representation.

### NWP features

Ten validated GFS-related predictors.

The NWP layer includes variables such as:

* CAPE
* CIN
* Lifted Index
* temperature
* relative humidity
* surface pressure
* wind speed
* wind direction
* precipitation
* cloud cover

Boundary Layer Height was investigated during the NWP pilot but was not retained because the available historical source produced unusable missingness for that variable.

---

# 🌍 15. NWP Data Collection

Historical NWP information was collected through the Open-Meteo Historical Forecast / Previous Runs workflow using:

```text
GFS Global
```

The full NWP collection covers:

```text
2021-04-01 → 2025-12-31
```

for all five reference locations.

Total pooled NWP rows:

```text
208,320
```

This corresponds to:

```text
41,664 rows × 5 locations
```

Missing values were preserved where they occurred.

No fabricated NWP values were inserted.

---

# 📈 16. NWP Experiment

Two model configurations were compared:

### Model A

Atmospheric features only.

### Model B

Atmospheric + NWP features.

The complete-case NWP experiment produced the following test results:

| Lead | PR-AUC | ROC-AUC | Precision | Recall |     F1 |
| ---- | -----: | ------: | --------: | -----: | -----: |
| +1h  | 0.1445 |  0.8510 |    0.1100 | 0.6340 | 0.1875 |
| +2h  | 0.1368 |  0.8443 |    0.1074 | 0.6229 | 0.1833 |
| +3h  | 0.1317 |  0.8372 |    0.1029 | 0.6077 | 0.1761 |

These are historical research metrics.

They are **not operational forecast guarantees**.

The NWP experiment showed improved ranking metrics and recall relative to the corresponding atmospheric-only experiment, while F1 did not improve at every lead.

A separate robustness experiment using train-only median imputation for the affected NWP variables produced the same test metrics.

---

# 🎚️ 17. Model Decision Threshold

The deployed Model B uses:

```text
6.5%
```

as its decision threshold.

The threshold was selected during validation according to the project's predefined threshold-selection procedure and alert-rate constraint.

Important:

> **6.5% is an ML decision threshold, not a meteorological warning threshold.**

The model outputs are:

```text
UNCALIBRATED RESEARCH PROBABILITIES
```

They must not be interpreted as calibrated real-world probabilities.

---

# 🔐 18. Frozen Model Artifacts

The production Model B artifacts are:

```text
models/v2/nwp_experiment/B_atmospheric_nwp_target_1h.joblib
models/v2/nwp_experiment/B_atmospheric_nwp_target_2h.joblib
models/v2/nwp_experiment/B_atmospheric_nwp_target_3h.joblib
```

SHA-256 hashes:

```text
1h:
a266c71bf901c4ea9392882b619e1a40905279af89db58123227a07f49733f21

2h:
e99ff3e9fba4cef8a3e1d15e6c29cfdbee801200369230bd652a45a71b376c6c

3h:
20692e504594ad17ea6fcc43dfe6e8899a5d5bdc2df93015df92327b97dc16ed
```

The model artifacts are treated as frozen production research artifacts.

---

# 🧱 19. Frozen 83-Feature Inference Contract

The final inference engine expects exactly the validated feature representation.

It validates:

* feature names
* feature count
* feature order
* finite values
* station ID
* timestamp
* NWP availability
* model availability

Invalid input causes the system to fail safely.

The inference engine does not silently fabricate missing values.

---

# 🧪 20. Fail-Closed Design

ThunderWatch AI intentionally uses a fail-closed strategy.

Examples:

```text
Missing required feature
        ↓
Prediction refused

Missing required NWP value
        ↓
Prediction refused

Invalid station
        ↓
Prediction refused

Invalid timestamp
        ↓
Prediction refused

Unavailable live data
        ↓
Truthful unavailable state
```

The system does not turn:

```text
data unavailable
```

into:

```text
no thunderstorm
```

This is especially important for weather-risk applications.

---

# 🔄 21. Historical Replay

Historical Replay provides a reproducible way to inspect the model on previously validated historical conditions.

The evaluator can:

```text
Select historical timestamp
        ↓
Load validated historical features
        ↓
Run frozen Model B
        ↓
Generate +1h/+2h/+3h
        ↓
Display station-based risk
        ↓
Reveal historical outcomes
```

Historical outcomes are shown after the prediction and are not used as model inputs.

---

# 🛰️ 22. INSAT-3DR Satellite Integration

Satellite was investigated as an important multimodal layer.

The selected product is:

```text
INSAT-3DR IMAGER L2B Cloud Mask
3RIMG_L2B_CMK
```

The pilot verified:

* HDF5 structure
* cloud-mask flags
* latitude/longitude arrays
* observation timing
* station-nearest pixels
* 25 km neighborhood statistics

Candidate features include:

```text
sat_cloud_mask_at_station
sat_nearby_cloud_fraction_25km
sat_valid_pixel_count_25km
sat_clear_fraction_25km
sat_cloudy_fraction_25km
sat_observation_available
sat_observation_time_utc
```

---

# 🛰️ 23. Satellite Historical MVP

The complete historical INSAT-3DR CMK archive is very large.

Downloading the complete archive was not practical within the project scope.

Therefore a controlled case-based satellite MVP was created.

The MVP contains:

```text
120 selected granules
119 successfully validated local files
1 known missing granule
20 case windows
Thunderstorm + control cases
Multiple years and seasons
```

The missing granule was preserved as missing.

No satellite data were fabricated.

---

# ⏱️ 24. Satellite Causal Alignment

Satellite acquisition timing was explicitly audited.

The `/time` field was found to represent the scan/catalog start timing rather than the end of acquisition.

Therefore the causal rule is:

```text
Acquisition end time <= prediction time T
```

and:

```text
Satellite age <= 90 minutes
```

If these conditions are not satisfied:

```text
Satellite physical features = unavailable
Satellite availability = 0
```

No future satellite information is used.

---

# 🔎 25. Satellite Role in the Current System

Satellite evidence is currently:

```text
CASE REPLAY / HISTORICAL CONTEXT
```

It is **not a frozen Model B input**.

The deployed Model B remains:

```text
Atmospheric + NWP
```

Satellite is deliberately kept as a separate evidence layer until a sufficiently large training-ready historical satellite archive is available.

---

# 📊 26. Satellite Case Findings

The controlled case analysis showed higher cloud fractions in selected thunderstorm cases than controls.

Approximate scan-level results:

```text
Thunderstorm cases:
Mean cloudy fraction ≈ 0.987

Control cases:
Mean cloudy fraction ≈ 0.538
```

For causal hourly matches:

```text
Thunderstorm:
Mean cloudy fraction ≈ 0.978

Control:
Mean cloudy fraction ≈ 0.288
```

These are **descriptive case findings**.

They are not presented as independent machine-learning skill metrics.

The project therefore does not claim that satellite integration has already improved Model B accuracy.

---

# ⚡ 27. Lightning Data Investigation

Lightning is a central component of the SIH26072 problem statement.

Therefore the project explicitly investigated genuine lightning data sources.

The investigation included sources such as:

* NASA TRMM/LIS
* ISS-LIS
* LIS climatology
* IITM-related lightning resources
* IMD-related resources
* other research-access possibilities

The key finding was:

> A reproducible public hourly lightning-observation archive covering the complete 2014–2025 period and all five reference locations was not available for direct integration into the current hourly training pipeline.

Therefore:

```text
NO SYNTHETIC LIGHTNING DATA
```

were introduced.

---

# 🌩️ 28. NASA LIS Pilot

A genuine NASA LIS/TRMM pilot was inspected.

The files contained real lightning variables such as:

* lightning events
* lightning groups
* lightning flashes

Station-proximity checks were performed for:

```text
25 km
50 km
100 km
```

around all five locations.

The selected pilot did not contain station-proximal lightning observations.

This does not mean lightning never occurred at those locations.

It means that the selected satellite swath/time coverage did not provide the required station-proximal observations.

The dataset therefore remains:

```text
REFERENCE ONLY
```

---

# 🌍 29. Lightning Climatology

A combined LIS thunder-hour climatology was also investigated.

This provides long-term climatological information rather than hourly lightning observations.

Therefore it can be treated as:

```text
LOCATION CLIMATOLOGY
```

but not:

```text
LIVE HOURLY LIGHTNING OBSERVATION
```

A controlled experiment tested the addition of annual thunder-hour climatology to the model.

It was classified as:

```text
REFERENCE ONLY
```

and was not adopted as the primary multimodal input.

---

# 📡 30. Radar Investigation

Radar is another important layer in SIH26072.

The project performed radar availability and coverage audits for the five reference locations.

For radar to become training-ready, the project requires data such as:

* quantitative reflectivity
* Z / dBZ or equivalent
* radar/station identifier
* UTC scan timestamp
* historical archive
* machine-readable format
* documented variables and units
* joinability with the model timeline

The publicly reproducible resources investigated did not provide the required complete five-location 2014–2025 quantitative radar archive.

Therefore the current radar status is:

```text
CONDITIONAL INSTITUTIONAL ACCESS
```

---

# 🚫 31. Why Radar and Lightning Are Not Faked

ThunderWatch AI deliberately does not display unsupported claims such as:

```text
RADAR ACTIVE
LIVE RADAR
RADAR FORECAST
LIGHTNING ACTIVE
LIVE LIGHTNING
```

Instead the dashboard communicates the actual status:

```text
RADAR — CONDITIONAL ACCESS
GROUND LIGHTNING — DATA ACCESS PENDING
```

This is a deliberate scientific-integrity decision.

---

# 🗺️ 32. Spatial Risk Visualization

ThunderWatch AI provides station-based spatial visualization.

The map displays the five supported reference locations and their model outputs.

The system intentionally does **not** interpolate station predictions into a synthetic continuous surface.

Therefore it does not claim to provide:

* storm-cell boundaries
* radar reflectivity maps
* continuous storm probability fields
* exact storm-cell locations

The current spatial system is:

```text
STATION-BASED AI RISK
```

---

# 🌐 33. Live Research Mode

ThunderWatch AI includes a live research mode.

The live system obtains current/near-current atmospheric information through external data sources rather than replaying historical rows.

### Atmospheric source

Open-Meteo.

### NWP source

Open-Meteo GFS global data.

The live service:

1. Requests current/recent atmospheric data.
2. Requests GFS/NWP information.
3. Finds the latest completed usable hourly observation.
4. Builds the validated feature representation.
5. Validates the 83-feature contract.
6. Runs frozen Model B.
7. Produces +1h/+2h/+3h outputs.

The current model works at hourly resolution.

It does not claim minute-level nowcasting.

---

# 🔄 34. Live Prediction Flow

```text
User opens Live Prediction
          ↓
Current atmospheric data requested
          ↓
GFS/NWP data requested
          ↓
Latest completed hourly observation selected
          ↓
Causal features constructed
          ↓
83-feature contract validated
          ↓
Frozen Model B inference
          ↓
+1h / +2h / +3h
          ↓
Dashboard visualization
```

---

# ⚡ 35. Five-Location Live Prediction

The system supports concurrent live inference for:

```text
VABB
VIDP
VECC
VOCI
VOTV
```

The browser sends a single five-location request.

The backend performs the station processing concurrently.

This avoids five independent browser prediction requests.

---

# ⏱️ 36. Measured Performance

Final local validation measured:

| Operation                       | Wall-clock time |
| ------------------------------- | --------------: |
| Single VABB live prediction     |        10.896 s |
| Five-location cold live request |         4.809 s |
| Five-location historical replay |        42.066 s |

The five-location live request is concurrent.

Live results also use a short cache to avoid unnecessary repeated external requests.

Cached performance should not be interpreted as cold external-fetch performance.

---

# 🖥️ 37. Dashboard

The dashboard has two primary modes.

## Live Prediction

Includes:

* Satellite Hybrid map
* five reference station markers
* Highest Model Risk
* +1h / +2h / +3h
* atmospheric signals
* NWP / multimodal context
* satellite evidence/context
* model/system information
* data age
* observation time
* prediction validity time

## Historical Replay

Includes:

* five-location historical replay
* station-based AI risk
* Location Risk
* +1h / +2h / +3h
* atmospheric context
* NWP context
* satellite case evidence
* historical outcomes
* replay processing feedback

The deployed application currently presents Live Research and Historical Replay as separate modes. ([Thunderwatch AI][1])

---

# 📱 38. Responsive Design

The final dashboard was tested at:

```text
1920 × 1080
1366 × 768
390 × 844
```

The final evaluator journey passed without horizontal overflow.

The mobile interface includes responsive navigation.

---

# 🧪 39. Testing

The final project validation completed with:

```text
113 tests passed
0 failed
```

The complete evaluator journey was tested:

```text
LIVE
  ↓
RUN 5-LOCATION LIVE DATA
  ↓
MAP
  ↓
HIGHEST MODEL RISK
  ↓
+1H / +2H / +3H
  ↓
HISTORICAL REPLAY
  ↓
RUN 5-LOCATION REPLAY
  ↓
HISTORICAL MAP
  ↓
LOCATION RISK
  ↓
SATELLITE EVIDENCE
  ↓
RETURN TO LIVE
```

---

# 🧯 40. Failure-Path Testing

The system was explicitly tested against failure scenarios.

### Mixed valid + invalid live locations

Example:

```text
VABB + XXXX + VOTV
```

Result:

```text
LIVE_PARTIAL
```

Valid locations remain available.

The invalid station does not receive fabricated probabilities.

### Unknown station

Result:

```text
HTTP 400
```

### Invalid historical replay

Result:

```text
all_unavailable
```

No fake probabilities.

### Invalid spatial timestamp

Result:

```text
HTTP 400
```

### Duplicate clicks

Live and replay controls are protected while processing.

---

# 🐳 41. Deployment Architecture

ThunderWatch AI uses:

```text
Docker
   ↓
Gunicorn
   ↓
Flask
   ↓
Render
```

The production configuration uses:

* Docker
* Gunicorn
* 1 worker
* 4 threads
* 180-second timeout
* Render `PORT`
* `0.0.0.0` binding

The production Docker image was successfully built and smoke-tested locally.

---

# ☁️ 42. Production Validation

The production-style container was validated for:

```text
/health
/
POST /api/inference/live
POST /api/inference/live/all
POST /replay/all
POST /api/spatial/replay
```

The container successfully produced:

```text
5/5 LIVE locations
5/5 historical replay locations
5/5 spatial replay locations
```

The application is deployed publicly through Render.

---

# 🔐 43. Secrets and Environment Variables

Private credentials are not stored in source code.

Environment-specific values should be supplied through environment variables.

For example:

```text
MAPTILER_API_KEY
```

may be configured through the deployment environment for the Satellite Hybrid basemap.

Do not commit:

```text
.env
```

or any credential/token/password.

The repository contains an `.env.example` file for configuration guidance.

---

# 📦 44. Runtime Data vs Research Data

The project generated many datasets during the research process.

Not every dataset is required by the deployed Flask application.

### Runtime-required data includes:

```text
dataset/multilocation/iem_station_metadata.json

dataset/multilocation/features/
    multilocation_features_2014_2025_metadata.json

dataset/multilocation/features_nwp/
    multilocation_features_nwp_overlap_2021_2025.csv

dataset/satellite/insat3dr_cmk_features/
    cmk_hourly_aligned_features.csv

dataset/multilocation/
    feature_engineering_phase3.py
```

Additional replay/runtime artifacts are stored under the corresponding `outputs/` paths.

Large research/acquisition datasets that are not required by Flask runtime are kept outside the deployment runtime where appropriate.

---

# 📁 45. Repository Structure

A simplified structure is:

```text
ThunderWatch AI/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
├── .env.example
├── .gitattributes
├── render.yaml
├── README.md
│
├── dataset/
│   └── multilocation/
│       ├── iem_station_metadata.json
│       ├── feature_engineering_phase3.py
│       │
│       ├── features/
│       │   └── multilocation_features_2014_2025_metadata.json
│       │
│       └── features_nwp/
│           └── multilocation_features_nwp_overlap_2021_2025.csv
│
├── dataset/
│   └── satellite/
│       └── insat3dr_cmk_features/
│           └── cmk_hourly_aligned_features.csv
│
├── models/
│   └── v2/
│       └── nwp_experiment/
│           ├── B_atmospheric_nwp_target_1h.joblib
│           ├── B_atmospheric_nwp_target_2h.joblib
│           └── B_atmospheric_nwp_target_3h.joblib
│
├── src/
│   ├── inference/
│   │   ├── final_multi_lead_engine.py
│   │   ├── multilead_service.py
│   │   ├── live_openmeteo_client.py
│   │   ├── live_feature_adapter.py
│   │   └── live_prediction_service.py
│   │
│   └── spatial/
│       └── historical_spatial_replay.py
│
├── templates/
│   └── replay.html
│
├── static/
│   ├── tw.js
│   ├── tw.css
│   ├── geo-map.js
│   └── india-map.js
│
├── ml/
│   └── training and validation scripts
│
├── docs/
│   └── scientific phase reports
│
└── outputs/
    └── validation / replay / evidence artifacts
```

---

# 🔌 46. API Endpoints

## Health

```http
GET /health
```

Checks whether the service is running.

---

## Root

```http
GET /
```

Loads the ThunderWatch AI dashboard.

---

## Historical Replay

```http
POST /replay/all
```

Runs the five-location historical replay.

---

## Live Single Location

```http
POST /api/inference/live
```

Example request:

```json
{
  "station_id": "VABB"
}
```

---

## Live Five Locations

```http
POST /api/inference/live/all
```

Example:

```json
{
  "stations": [
    "VABB",
    "VIDP",
    "VECC",
    "VOCI",
    "VOTV"
  ]
}
```

---

## Historical Spatial Replay

```http
POST /api/spatial/replay
```

Runs station-based historical spatial replay.

---

## Historical Spatial Timestamps

```http
GET /api/spatial/replay/timestamps
```

Returns validated replay timestamps.

---

# 💻 47. Running ThunderWatch AI Locally

## Requirements

Recommended:

* Python 3.11+
* Git
* Git LFS
* Docker Desktop (optional)
* Internet connection for live Open-Meteo/GFS access
* Sufficient disk space for runtime model/data files

For the complete research environment, substantially more storage may be required because the development process generated large datasets.

---

# 📥 48. Clone the Repository

```bash
git clone https://github.com/nachiket-mr360/AIML_Thunderstorm_nowcasting_by_ThunderWatch_AI.git
cd AIML_Thunderstorm_nowcasting_by_ThunderWatch_AI
```

---

# 📦 49. Install Git LFS

The repository uses Git LFS for large runtime artifacts.

Install Git LFS and initialize it:

```bash
git lfs install
```

Then download LFS-managed files:

```bash
git lfs pull
```

---

# 🐍 50. Create a Python Virtual Environment

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell execution policy prevents activation, Command Prompt can be used:

```cmd
.venv\Scripts\activate
```

## Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# 📚 51. Install Python Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the application dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔐 52. Configure Environment Variables

Create a local `.env` from the example file.

## Windows PowerShell

```powershell
Copy-Item .env.example .env
```

## Linux / macOS

```bash
cp .env.example .env
```

Do not commit `.env`.

Do not put private credentials into GitHub.

If MapTiler is configured, place the appropriate key in the environment variable expected by the application.

The current Open-Meteo live implementation does not require an Open-Meteo API key.

---

# ▶️ 53. Run Flask Locally

Start the application:

```bash
python app.py
```

The development server normally runs at:

```text
http://127.0.0.1:5000
```

Open that address in your browser.

---

# 🦄 54. Run with Gunicorn

The production architecture uses Gunicorn.

### Windows PowerShell

```powershell
$env:PORT="10000"
gunicorn --bind 0.0.0.0:$env:PORT --workers 1 --threads 4 --timeout 180 "app:create_app()"
```

### Linux / macOS

```bash
PORT=10000 gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180 "app:create_app()"
```

Then open:

```text
http://127.0.0.1:10000
```

---

# 🐳 55. Run with Docker

Build the production image:

```bash
docker build -t thunderwatch-ai .
```

Run:

## Windows PowerShell

```powershell
docker run --rm -p 10000:10000 -e PORT=10000 thunderwatch-ai
```

Open:

```text
http://127.0.0.1:10000
```

Health check:

```text
http://127.0.0.1:10000/health
```

---

# 🧪 56. Testing the Application

Run the full test suite:

```bash
python -m pytest tests
```

The final validated state contained:

```text
113 passed
0 failed
```

---

# 🌐 57. Live Mode

After starting the application:

```text
LIVE PREDICTION
        ↓
RUN 5-LOCATION LIVE DATA
        ↓
Current/near-current atmospheric + NWP data
        ↓
Latest completed hourly observation
        ↓
Model B
        ↓
+1h / +2h / +3h
```

The live system supports:

```text
VABB
VIDP
VECC
VOCI
VOTV
```

---

# 🔁 58. Historical Replay Mode

Use:

```text
HISTORICAL REPLAY
```

Then:

```text
Select validated timestamp
        ↓
RUN 5-LOCATION REPLAY
        ↓
Historical Model B inference
        ↓
Station-based AI risk
        ↓
Satellite evidence where available
        ↓
Historical outcome comparison
```

---

# ⚠️ 59. Scientific Limitations

ThunderWatch AI is a research prototype.

It is **not** an official meteorological warning system.

The following limitations apply.

### 1. Uncalibrated probabilities

Model scores are not calibrated probabilities.

### 2. Station-based predictions

Predictions are produced for supported reference locations.

### 3. No continuous storm-cell map

The spatial visualization is station-based.

### 4. No operational radar ingest

Historical quantitative radar access is currently conditional.

### 5. No hourly historical lightning model input

A complete reproducible public hourly lightning archive was not available for the target locations and period.

### 6. Satellite is not a frozen Model B input

INSAT-3DR CMK is historical case evidence.

### 7. Historical replay is not operational validation

Replay demonstrates reproducibility but does not establish operational forecasting skill.

### 8. Live source differs from historical training source

The live pipeline obtains current/near-current data from Open-Meteo/GFS, while the historical model was developed from validated historical datasets.

### 9. No warning guarantee

A model risk signal is not an official warning.

---

# 🚫 60. What ThunderWatch AI Does NOT Claim

ThunderWatch AI does **not** claim:

* superiority over IMD
* official IMD status
* official meteorological warnings
* guaranteed thunderstorm prediction
* calibrated probabilities
* exact storm-cell boundaries
* continuous radar-based storm maps
* live radar forecasting
* live lightning forecasting
* synthetic lightning observations
* satellite-powered Model B predictions
* operational national-scale forecasting

---

# 🏛️ 61. Relationship to Official Meteorological Systems

ThunderWatch AI is designed as a **research and decision-support prototype**.

It is not intended to replace official meteorological agencies.

The project explores a lightweight, reproducible AI architecture for:

* station-level thunderstorm nowcasting
* multi-lead prediction
* NWP augmentation
* satellite evidence
* multimodal research
* historical replay
* deployable AI inference

The system deliberately avoids claiming superiority over established operational meteorological systems.

---

# 🧪 62. Research Validation Philosophy

The project follows several rules:

```text
NO FABRICATED DATA

NO SYNTHETIC LIGHTNING

NO FAKE RADAR

NO FUTURE INFORMATION LEAKAGE

NO SILENT IMPUTATION OF UNAVAILABLE REQUIRED DATA

NO FAKE PREDICTIONS

NO UNDISCLOSED DATA SUBSTITUTION

NO CLAIM OF OPERATIONAL WARNING CAPABILITY
```

When a data source is unavailable, the system reports its actual status.

---

# 📚 63. Development Work Completed

The project development covered:

## Data

* Multi-location atmospheric collection
* METAR audit
* Thunderstorm target construction
* Data synchronization
* Missingness checks
* Duplicate checks

## Feature Engineering

* Causal feature construction
* 73 atmospheric/location features
* Leakage validation
* Reproducibility checks

## Machine Learning

* Pooled baseline
* Multi-lead targets
* +1h/+2h/+3h models
* NWP experiment
* NWP robustness experiment
* Model B freeze

## Satellite

* INSAT-3DR product audit
* Archive availability audit
* Controlled case selection
* CMK downloads
* HDF5 validation
* Timing audit
* Causal alignment
* Case replay

## Lightning

* Source availability research
* NASA LIS pilot
* Lightning climatology investigation
* Station-proximity audit
* Capability classification

## Radar

* Availability audit
* Coverage audit
* Institutional-access classification
* Training-readiness definition

## Inference

* Final multi-lead inference engine
* 83-feature contract
* Fail-closed validation
* Historical inference
* Live inference
* Five-location concurrent inference

## Spatial

* Station-based historical risk
* Five-location map
* Historical spatial replay
* Satellite context

## Dashboard

* Live Prediction
* Historical Replay
* Satellite Hybrid map
* Five-location visualization
* Processing states
* Responsive UI
* Mobile navigation
* Failure states

## Deployment

* Docker
* Gunicorn
* Render
* Git LFS
* Production smoke testing
* Runtime-data separation
* Secret protection

---

# 📈 64. Current System Status

```text
PROJECT
ThunderWatch AI

STATUS
Research Prototype Online

PROBLEM STATEMENT
SIH26072

LOCATIONS
5 reference locations

MODEL
ThunderWatch AI Model B

MODEL TYPE
Random Forest

FEATURES
83 atmospheric + NWP

LEADS
1h / 2h / 3h

THRESHOLD
6.5%

PROBABILITIES
Uncalibrated research probabilities

ATMOSPHERIC DATA
Connected

GFS NWP
Connected where validated

SATELLITE
Case Replay Ready

RADAR
Conditional Access

GROUND LIGHTNING
Data Access Pending

MODES
Live Research / Historical Replay

DEPLOYMENT
Docker + Gunicorn + Render
```

The live dashboard itself currently identifies the system as a research prototype and distinguishes satellite case replay, conditional radar access, and pending ground-lightning access. ([Thunderwatch AI][1])

---

# 🔮 65. Future Work

Future development can expand the system when scientifically valid data become available.

Potential directions include:

1. Authorized historical quantitative radar acquisition.
2. Authorized historical lightning observation acquisition.
3. Larger INSAT satellite archive.
4. Training-ready satellite integration.
5. Radar reflectivity feature extraction.
6. Lightning feature extraction.
7. Larger multimodal training experiments.
8. Independent temporal validation.
9. Independent spatial validation.
10. Probability calibration.
11. Uncertainty estimation.
12. Higher-frequency observations.
13. Spatial storm-cell modeling.
14. Operational data-quality monitoring.
15. More reference locations.
16. Additional NWP sources.
17. Advanced multimodal fusion architectures.

These are future research directions.

They are not current production claims.

---

# 💡 66. Why the Project is Useful

ThunderWatch AI demonstrates a complete workflow from:

```text
OBSERVATIONS
      ↓
DATA COLLECTION
      ↓
DATA SYNCHRONIZATION
      ↓
CAUSAL FEATURE ENGINEERING
      ↓
FUTURE TARGET CREATION
      ↓
MACHINE LEARNING
      ↓
NWP AUGMENTATION
      ↓
MULTIMODAL EVIDENCE
      ↓
VALIDATION
      ↓
LIVE INFERENCE
      ↓
HISTORICAL REPLAY
      ↓
SPATIAL VISUALIZATION
      ↓
DEPLOYED WEB APPLICATION
```

The project therefore combines:

* Data engineering
* Weather observation processing
* Machine learning
* Time-series feature engineering
* NWP integration
* Satellite data analysis
* Radar data research
* Lightning data research
* Model validation
* Backend development
* Frontend development
* Visualization
* Deployment
* Scientific reproducibility

into one end-to-end system.

---

# 👥 67. Team

## FANTASTIC6

**ThunderWatch AI**

**Smart India Hackathon 2026**

**Problem Statement: SIH26072**

> **From Data to a Safer Tomorrow.**

---

# 🌩️ 68. Final Summary

ThunderWatch AI is a research prototype exploring AI-based thunderstorm nowcasting using:

```text
Genuine Thunderstorm Observations
            +
Atmospheric Conditions
            +
GFS NWP
            +
Satellite Case Evidence
            +
Machine Learning
            ↓
Thunderstorm Risk
            ↓
+1h / +2h / +3h
```

The project intentionally separates:

```text
WHAT WE HAVE VALIDATED
```

from:

```text
WHAT STILL REQUIRES DATA ACCESS
```

Current core:

```text
Atmospheric observations
        +
GFS NWP
        ↓
ThunderWatch AI Model B
        ↓
+1h / +2h / +3h
```

Additional evidence:

```text
INSAT-3DR CMK
        ↓
Historical Case Evidence
```

Future/conditional layers:

```text
Radar
        ↓
Conditional Institutional Access

Ground Lightning
        ↓
Data Access Pending
```

ThunderWatch AI therefore provides a reproducible, deployable foundation for continued multimodal thunderstorm-nowcasting research while remaining explicit about the boundaries of the current evidence.

---

## 🚀 Try ThunderWatch AI

**Live Demo:**

[https://aiml-thunderstorm-nowcasting-by.onrender.com/](https://aiml-thunderstorm-nowcasting-by.onrender.com/)

---

**ThunderWatch AI · FANTASTIC6 · SIH26072**

**From Data to a Safer Tomorrow.**

```

### One important thing, Nachiket

This README is intentionally **much more comprehensive than the current GitHub README**. Your live deployment currently exposes the newer five-location Model B architecture, +1/+2/+3 leads, 83 features, satellite case replay, and separate Live Research / Historical Replay modes. :contentReference[oaicite:3]{index=3}

Your GitHub page's currently indexed README still describes the **older VOTV-only prototype**, so replacing it with the README above is important to prevent the public repository from contradicting the actual deployed system. :contentReference[oaicite:4]{index=4}

Also, I would **not add raw passwords, API keys, private MOSDAC credentials, or the large local-only research datasets** to the README.
```

[1]: https://aiml-thunderstorm-nowcasting-by.onrender.com/ 
"THUNDERWATCH AI — FANTASTIC6 · SIH26072"
