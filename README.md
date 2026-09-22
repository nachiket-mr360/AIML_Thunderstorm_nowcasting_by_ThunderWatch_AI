# ThunderWatch AI

**AI-Based Multimodal Thunderstorm & Lightning Nowcasting Decision-Support System**

Smart India Hackathon **2026** · Problem Statement **SIH26072**  
Domain: Disaster Management / Weather Forecasting / Artificial Intelligence & Machine Learning  
Target: Short-lead thunderstorm and lightning **risk nowcasting** and **decision support**

ThunderWatch AI is a research-oriented multimodal AI decision-support platform being developed for SIH26072. It estimates short-lead thunderstorm risk from causal atmospheric time-series information, integrates numerical weather prediction (NWP) / model data where historically available, and is designed to add satellite, radar and lightning observations **only after** legitimate data access, temporal alignment, leakage checks and experimental evaluation.

This repository is the **main ThunderWatch AI project** for SIH26072. The work is an evolving research/prototype system: completed atmospheric + NWP modelling, frozen inference, historical replay and a deployed decision-support dashboard already exist; satellite, radar and genuine hourly lightning fusion remain **in progress** and are not claimed as finished.

---

## Live Demo

**ThunderWatch AI — deployed research prototype**

**https://aiml-thunderstorm-nowcasting-by.onrender.com/**

The deployed application currently demonstrates the **historical-replay decision-support workflow** and the **validated atmospheric + NWP model pipeline**.

It is **not** an operational government warning system, not an IMD product, and not a live nationwide thunderstorm forecasting service.

---

## 1. The problem

Thunderstorms can develop rapidly and produce lightning, heavy rainfall, strong winds and hazardous conditions. Those hazards affect aviation, transport, local infrastructure and public safety. Short-lead **nowcasting** (hours, not days) is therefore a high-value decision-support problem.

**Official problem statement (SIH26072):**

> AIML based Nowcasting of thunderstorm and lightning using atmospheric observation including multiple radars, satellite, lightning and model data

The statement asks for AI/ML nowcasting that can draw on **multiple atmospheric information sources**, specifically including:

- atmospheric observations  
- model / NWP data  
- satellite  
- multiple radars  
- lightning  

### Why this is technically difficult

1. **Thunderstorms are highly localized.** Station-scale risk is not the same as a national weather map.  
2. **They evolve rapidly.** Useful nowcast windows are short (here: 1–3 hours).  
3. **Sensors disagree in space and time.** Radar, satellite, lightning networks and surface observations have different resolutions and latencies.  
4. **Data sources are heterogeneous.** Formats, units, timestamps and missingness patterns differ.  
5. **Lightning is hard to obtain as a reproducible historical training source.** Research archives, climatologies and operational networks are not interchangeable.  
6. **Radar data access may require institutional authorization.** Access cannot be faked.  
7. **Satellite products** differ in format, resolution and timestamp conventions.  
8. **Labels must be genuine observed thunderstorms**, not precipitation proxies or synthetic events.  
9. **Future information must never leak into model inputs.** Features at time *T* may only use information available at or before *T*.  
10. **A useful system must know when its input is incomplete.** Missing critical data should yield *unavailable*, not a fabricated score.

ThunderWatch AI is designed as a **complementary AI research and decision-support layer**. Emphasis is on reproducibility, multimodal fusion *when data exist*, transparent provenance, multi-location modelling and historical replay — not on replacing established meteorological agencies.

---

## 2. What is ThunderWatch AI?

ThunderWatch AI is a research-oriented multimodal AI decision-support platform being developed for SIH26072 to estimate short-lead thunderstorm risk from atmospheric time-series information and progressively integrate NWP/model, satellite, radar and lightning observations.

High-level pipeline:

```
Data sources
    ↓
Data quality / synchronization
    ↓
Causal feature engineering
    ↓
Genuine thunderstorm labels
    ↓
Multi-location / multi-lead ML
    ↓
Multimodal fusion
    ↓
Historical replay / evaluation
    ↓
Decision-support dashboard
    ↓
Future near-real-time integration
```

Every new modality is added only after:

- data availability is verified  
- temporal alignment is verified  
- leakage checks are performed  
- missing-data behaviour is defined  
- model impact is experimentally evaluated  

The system is designed around multiple geographic locations, causal time-series features, multi-lead prediction, genuine observed thunderstorm labels, multimodal fusion, transparent model outputs, historical replay, fail-closed data handling, explainable data provenance, and future near-real-time / operational integration **as data-access constraints are solved**.

---

## 3. What we have already built

### A. Genuine historical thunderstorm labels

METAR observations were audited. Training and evaluation use **genuine observed thunderstorm reports**, not synthetic labels or precipitation proxies.

| ICAO | Location |
| --- | --- |
| VOTV | Thiruvananthapuram |
| VECC | Kolkata |
| VIDP | Delhi |
| VOCI | Kochi |
| VABB | Mumbai |

- Historical modelling period: **2014–2025**  
- Labels come from genuine METAR **present-weather thunderstorm** observations.  
- **Unobserved future target hours are not automatically converted to negative labels.** Missing future observations remain unavailable. This is a scientific safeguard against false-negative labelling.

### B. Multi-location atmospheric dataset

A synchronized **hourly** atmospheric dataset covers the five locations. Core variables include temperature, relative humidity, surface pressure, wind speed, wind direction, precipitation and cloud cover. Synchronization is by **station + UTC timestamp**.

### C. Causal feature engineering

The system currently builds **73 atmospheric / location / temporal features**, including:

- current atmospheric state  
- cyclic time features  
- wind-vector representation  
- historical lags  
- short-term tendencies  
- rolling precipitation accumulation  
- rolling atmospheric statistics  
- location features  
- elevation  

All model inputs are **causal**: features at time *T* use only information available at or before *T*. Leakage tests were performed, and feature engineering was validated against the original single-location feature definitions.

### D. Future multi-lead targets

Genuine future thunderstorm targets:

- **T + 1 hour**  
- **T + 2 hours**  
- **T + 3 hours**  

The feature vector remains anchored at *T*. Only the **target** moves forward.

### E. NWP / model-data integration

GFS-based historical forecast / model data were integrated through **Open-Meteo Historical Forecast**. Current NWP variables include CAPE, convective inhibition, lifted index, temperature, relative humidity, surface pressure, wind speed, wind direction, precipitation and cloud cover.

Validated NWP overlap currently covers **2021–2025**.

NWP fields are **model / forecast data**, not direct atmospheric observations.

### F. Multimodal model experiment (Model B)

Current validated research model: **Model B**.

| Item | Value |
| --- | --- |
| Atmospheric / location / temporal features | 73 |
| NWP / model features | 10 |
| **Total features** | **83** |
| Algorithm | Random Forest Classifier |
| Trees | 200 |
| Criterion | Gini |
| `max_features` | `sqrt` |
| `class_weight` | `"balanced"` |
| `random_state` | 42 |
| Leads | 1 h, 2 h, 3 h |
| Frozen research thresholds | **0.065** for each lead |

The output is a **model risk score**. It is **not** a calibrated probability and **not** an official warning probability.

### G. Lightning-related research

Multiple genuine lightning sources were investigated, including NASA LIS / TRMM-LIS and an annual LIS thunder-hour climatology. The climatology was experimentally evaluated as a **reference** feature.

- It is **not** treated as an hourly lightning observation.  
- It is **not** used to fake live lightning.  
- It is **not** represented as a real-time lightning feed.  

The project does **not** claim lightning integration where a training-ready, reproducible real-time historical lightning dataset is unavailable. Future genuine lightning-network integration remains planned, subject to legitimate data access.

### H. Satellite research

INSAT-3DR cloud products via **MOSDAC** were identified as a potential future satellite feature source. **Satellite integration is not complete.** Next work will verify account / data access, product availability, spatial coverage, temporal resolution, timestamp alignment, feature extraction, leakage and model contribution.

### I. Radar research

Radar is part of the official SIH26072 requirement. The IMD radar data-access route has been investigated. **Radar integration is pending legitimate institutional / student access.**

- No fake radar layer  
- No synthetic radar reflectivity  
- No screenshot or map tile presented as live radar  

### J. Inference engine

A frozen inference contract:

- exact feature order  
- validation of all required inputs  
- station identity checks  
- numerical validity  
- rejection of missing / non-finite features  
- frozen model artifacts and frozen thresholds  
- structured 1 h / 2 h / 3 h outputs  
- **fail-closed** when required information is unavailable  

This prevents accidental feature-order mismatch and silent model misuse.

### K. Historical replay

For a historical timestamp *T*:

1. only data available at or before *T* is loaded  
2. causal features are reconstructed  
3. the frozen model generates the prediction  
4. **only after prediction** are future historical labels read for evaluation  

This simulates how a prediction would have looked **without leaking future observations** into the prediction.

### L. Historical replay evaluation

Replay evaluates thousands of historical timestamps across all five locations. This is a **historical research evaluation**, **not** an independent held-out test. That distinction is intentional.

### M. Error analysis

The project also performs false-positive and false-negative analysis, near-threshold analysis, station-level and monthly / seasonal descriptive analysis, and multi-lead alert-pattern analysis. A forecasting research system should not report a single metric in isolation.

### N. Decision-support dashboard

The deployed interface includes a five-location overview, highest-risk location, 1 h / 2 h / 3 h risk timeline, atmospheric signals, model / system transparency, data status, replay controls, location selection, interactive India visualization, and risk markers based on **actual model outputs**.

The interface does **not** fabricate radar cells, satellite imagery, lightning strikes or spatial storm cells.

---

## 4. Current system architecture

```
                    THUNDERWATCH AI
                          │
          ┌───────────────┼────────────────┐
          │               │                │
 Atmospheric Data      NWP/Model       Future Modalities
          │               │          ┌─────┼─────┐
          │               │          │     │     │
          │               │       Satellite Radar Lightning
          └───────────────┴──────────┴─────┴─────┘
                          │
                  Data Synchronization
                          │
                  Quality / Validation
                          │
                 Causal Feature Engine
                          │
                 Multi-location Dataset
                          │
                 Multi-lead ML Engine
                     1h / 2h / 3h
                          │
                  Historical Replay
                          │
              Evaluation / Error Analysis
                          │
                 Decision Support UI
                          │
             Future Near-Real-Time System
```

Satellite, radar and lightning are **modular future inputs**. They are **not** currently completed production integrations.

---

## 5. Current locations

| ICAO | Location | Role |
| --- | --- | --- |
| VOTV | Thiruvananthapuram | Primary southern reference location |
| VECC | Kolkata | Multi-location training / evaluation |
| VIDP | Delhi | Multi-location training / evaluation |
| VOCI | Kochi | Multi-location training / evaluation |
| VABB | Mumbai | Multi-location training / evaluation |

Locations are **not ranked**.

---

## 6. Current model validation

**Frozen held-out test evaluation of the atmospheric + NWP Model B experiment**

| Lead | PR-AUC | ROC-AUC | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 hour | 0.1445 | 0.8510 | 0.6340 | 0.1100 | 0.1875 |
| 2 hours | 0.1368 | 0.8443 | 0.6229 | 0.1074 | 0.1833 |
| 3 hours | 0.1317 | 0.8372 | 0.6077 | 0.1029 | 0.1761 |

These metrics show **measurable ranking skill** on the defined historical test split. Precision remains limited because thunderstorms are **rare events**. **PR-AUC** is particularly informative for this class imbalance.

These numbers are **not** production accuracy, **not** operational accuracy, and **not** a claim of superiority over IMD or any established forecasting system.

Historical replay metrics are **separate** from this held-out test and must not be treated as an independent held-out evaluation.

---

## 7. Why ThunderWatch AI takes a different approach

1. **Genuine observation-based labels.** Thunderstorm reports where available, not generic weather or precipitation as the target.  
2. **Causal feature construction.** Inputs are restricted to information available at prediction time.  
3. **Multi-location architecture.** The current research dataset covers five locations, not a single permanently fixed station.  
4. **Multi-lead forecasting.** 1 h, 2 h and 3 h targets.  
5. **Multimodal design.** NWP has been experimentally integrated; satellite / radar / lightning are added only after data-access validation.  
6. **Historical replay.** Predictions can be reconstructed at historical timestamps without reading future targets before prediction.  
7. **Fail-closed behaviour.** Missing critical data produces **UNAVAILABLE**, not a fabricated risk score.  
8. **Data provenance.** Observations, model data, climatology and future sources are distinguished.  
9. **Reproducibility.** Frozen model artifacts, feature contracts, thresholds and validation reports are retained.  
10. **Decision support rather than unexplained classification.** The dashboard exposes lead times, model scores, threshold state, data completeness and system status.

These design choices are intended to make the prototype more transparent, reproducible and extensible; they are **not** evidence that ThunderWatch AI is operationally superior to established meteorological systems.

---

## 8. Relation to existing operational systems

Operational meteorological agencies such as **IMD** already operate sophisticated forecasting and decision-support infrastructure involving multiple observation and model sources.

ThunderWatch AI is **not** intended to replace an official warning system.

Its research contribution is to investigate a reproducible AI layer that combines heterogeneous inputs, evaluates genuine observed labels, supports multi-location experiments and multi-lead prediction, performs historical replay, exposes model / data provenance, and provides a modular path toward additional modalities.

No unsupported comparison with IMD is made.

---

## 9. Current data status

| Data layer | Current status | Role |
| --- | --- | --- |
| Atmospheric observations / model grid | **Integrated** | Core historical features |
| Genuine METAR thunderstorm labels | **Integrated** | Supervision / evaluation |
| NWP / model data | **Integrated experimentally** | Additional predictor layer |
| Lightning observations | Research / audit completed; live historical feed **pending** | Future multimodal input |
| Lightning climatology (LIS thunder-hour) | **Reference-only** (not hourly lightning) | Experimental reference |
| Satellite | **Access / integration pending** | Future multimodal input |
| Radar | **Institutional access pending** | Future multimodal input |
| Live operational pipeline | **Not claimed** | Future work |

---

## 10. What we are going to build next

### Phase A — Satellite integration

Verify MOSDAC access; obtain INSAT-3DR cloud product; inspect actual files; verify spatial coverage; extract satellite features; align to station / time; leakage audit; compare models with / without satellite; retain features only if evidence supports them.

### Phase B — Radar integration

Obtain legitimate IMD radar access; inspect products; derive station-centric / radar features; temporal and spatial alignment; missing-data handling; leakage audit; radar fusion experiment.

### Phase C — Lightning integration

Pursue legitimate lightning-network / research data access; align observations with station / time windows; derive physically meaningful lightning features; distinguish observations from climatology; test incremental contribution.

### Phase D — Multimodal fusion

Atmosphere + NWP + satellite + radar + lightning. Compare incremental contributions scientifically. **Do not assume every modality improves the model.**

### Phase E — Spatial risk

If sufficient radar / satellite / lightning coverage is obtained: move beyond station-only prediction; construct spatial features / grid cells; investigate storm structure and advection; produce spatial risk fields. Spatial storm-cell forecasting is **not claimed** until data and validation exist.

### Phase F — Near-real-time pipeline

After the historical pipeline is scientifically validated: real-time atmospheric inputs, current NWP runs, satellite / radar / lightning latency, data freshness, automatic quality checks, inference and monitoring.

### Phase G — Final SIH prototype

Target architecture: multimodal + multi-location + multi-lead + explainable + data-aware + decision-support.

---

## 11. Scientific safety and integrity

ThunderWatch AI deliberately follows these principles:

- No synthetic lightning observations presented as real  
- No fake radar imagery  
- No fake satellite imagery  
- No fabricated spatial storm cells  
- No future observations in prediction features  
- No automatic conversion of missing labels to negative labels  
- No hidden imputation of critical live inputs  
- No claim of calibrated probabilities when scores are not calibrated  
- No claim of official IMD warning status  
- No claim of operational superiority  
- No claim that an unfinished modality is integrated  
- No use of evaluation targets as model inputs  

---

## 12. Current deployment

Production-style container:

- Docker  
- Gunicorn  
- Flask  
- Render  

**Demo:** https://aiml-thunderstorm-nowcasting-by.onrender.com/

The current deployed system is a **historical-replay research / decision-support prototype**. It is **not** a fully operational real-time forecasting system.

---

## 13. Technology stack

### Backend

- Python  
- Flask  
- Gunicorn  

### Machine learning

- scikit-learn  
- Random Forest  
- pandas  
- NumPy  
- joblib  

### Data

- METAR observations  
- Open-Meteo atmospheric / model data  
- GFS-based historical forecast / model data  
- NASA LIS research / reference data  
- future MOSDAC / INSAT  
- future IMD radar  
- future lightning-network observations  

### Frontend

- HTML, CSS, JavaScript  
- interactive visualization  
- WebGL / Three.js where currently used  
- map / geospatial visualization  

### Deployment

- Docker  
- Render  
- GitHub  

---

## 14. Repository structure

```
app.py                 Flask application entry
backend/              Nowcast service and backend verification
src/
  inference/           Frozen inference contract / engine
  features/            Causal feature builder
  replay/              Historical replay and analysis
ml/                    Training, prediction, experiment scripts
dataset/
  multilocation/       Multi-station atmospheric, NWP and label pipelines
  lightning/           Lightning research / audit artefacts (not a live feed)
docs/                  Phase reports and scientific notes
models/                Frozen model artefacts and metadata
outputs/               Evaluation reports, metrics, replay analyses
templates/             Dashboard HTML
static/                CSS, JavaScript, map assets
tests/                 Automated checks for inference, replay and dashboard
Dockerfile
render.yaml
requirements.txt
README.md
```

Research datasets, large model files and phase reports live in the repository for **reproducibility**. They are **not** all part of the deployed container image.

---

## 15. How to run locally

Python environment:

```text
python -m venv .venv
```

Windows:

```text
.venv\Scripts\activate
```

Linux / macOS:

```text
source .venv/bin/activate
```

Install and run:

```text
pip install -r requirements.txt
python app.py
```

Docker:

```text
docker build -t thunderwatch-ai .
docker run --rm -p 10000:10000 -e PORT=10000 thunderwatch-ai
```

Dashboard: **http://localhost:10000/**

---

## 16. Project status

| Capability | Status |
| --- | --- |
| Historical atmospheric dataset | COMPLETE |
| Genuine thunderstorm labels | COMPLETE |
| Multi-location dataset | COMPLETE |
| Causal feature engineering | COMPLETE |
| 1 h / 2 h / 3 h targets | COMPLETE |
| NWP integration | COMPLETE / EXPERIMENTALLY VALIDATED |
| Frozen inference engine | COMPLETE |
| Historical replay | COMPLETE |
| Replay evaluation | COMPLETE |
| Error analysis | COMPLETE |
| Decision-support dashboard | COMPLETE |
| Docker deployment | COMPLETE |
| Render deployment | COMPLETE |
| Lightning historical research audit | COMPLETE |
| Satellite integration | NEXT / PENDING ACCESS |
| Radar integration | PENDING INSTITUTIONAL ACCESS |
| Real-time lightning feed | PENDING LEGITIMATE DATA ACCESS |
| Full multimodal fusion | PLANNED |
| Spatial storm-cell prediction | PLANNED |
| Operational real-time forecasting | FUTURE / NOT CLAIMED |

---

## 17. Important current limitation

The deployed application currently demonstrates a **historical-replay decision-support workflow** based on the validated **atmospheric + NWP** model pipeline.

It should **not** be interpreted as:

- an official weather warning  
- an operational IMD system  
- a live nationwide thunderstorm forecasting service  
- a complete implementation of every SIH26072 data modality  

Remaining SIH work focuses on **legitimate integration and validation** of satellite, radar and lightning data, and on determining whether those modalities **actually improve** predictive performance.

---

## 18. References

- [Smart India Hackathon](https://www.sih.gov.in/)  
- [India Meteorological Department (IMD)](https://mausam.imd.gov.in/)  
- [Open-Meteo](https://open-meteo.com/)  
- [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/)  
- [MOSDAC](https://www.mosdac.gov.in/)  
- [NASA Earthdata](https://www.earthdata.nasa.gov/) / [LIS](https://ghrc.nsstc.nasa.gov/home/about-ghrc/lightning-imaging-sensor-lis)  

---

## 19. Project statement

ThunderWatch AI is being developed as a reproducible AI research and decision-support platform for SIH26072. The project starts from genuine observed thunderstorm labels and causal atmospheric information, adds NWP/model information, and is progressively extending toward satellite, radar and lightning fusion. Every new modality is treated as an experimentally verifiable data layer rather than a visual placeholder. The long-term objective is a transparent, multi-location, multi-lead and multimodal thunderstorm nowcasting system that can support faster and more informed decision-making.

**Demo:** https://aiml-thunderstorm-nowcasting-by.onrender.com/
