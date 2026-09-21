# Phase 13A — V2 Prediction / Inference Contract

**Generated:** 2026-09-21  
**Script:** `ml/validate_v2_inference_contract_phase13a.py`  
**Scope:** specification + validation only. No training. No Flask. No dashboard. No satellite/radar/lightning implementation.

**Model version:** `V2-MODEL-B-MULTILEAD-PHASE8D-RESEARCH`  
**Status:** validated **research / prototype** contract. **Not** a production version.

---

## 1. Objective

Define a deterministic, implementation-ready contract for how a future V2 backend transforms an observation at time **T** into:

- 1-hour thunderstorm risk (`target_1h` = METAR thunderstorm at T+1h)
- 2-hour thunderstorm risk (`target_2h` = METAR thunderstorm at T+2h)
- 3-hour thunderstorm risk (`target_3h` = METAR thunderstorm at T+3h)

using the validated Phase 8D **Model B** architecture (atmospheric + location/temporal + NWP = **83** features).

This phase does **not** serve predictions. It freezes names, order, thresholds, missing-data policy, and temporal rules.

---

## 2. Frozen model artifacts

Phase 8D Model B RandomForest classifiers. Artifacts were **not** overwritten.

| lead | artifact | type | n_features | random_state | threshold | metrics JSON |
| --- | --- | --- | --- | --- | --- | --- |
| `target_1h` | `models/v2/nwp_experiment/B_atmospheric_nwp_target_1h.joblib` | `RandomForestClassifier` | 83 | 42 | **0.065** | `outputs/v2_nwp_experiment/metrics_B_atmospheric_nwp_target_1h.json` |
| `target_2h` | `models/v2/nwp_experiment/B_atmospheric_nwp_target_2h.joblib` | `RandomForestClassifier` | 83 | 42 | **0.065** | `outputs/v2_nwp_experiment/metrics_B_atmospheric_nwp_target_2h.json` |
| `target_3h` | `models/v2/nwp_experiment/B_atmospheric_nwp_target_3h.joblib` | `RandomForestClassifier` | 83 | 42 | **0.065** | `outputs/v2_nwp_experiment/metrics_B_atmospheric_nwp_target_3h.json` |

**Training configuration (all three leads):**

| item | value |
| --- | --- |
| `n_estimators` | 200 |
| `criterion` | gini |
| `max_features` | sqrt |
| `class_weight` | balanced |
| `random_state` | 42 |
| `n_jobs` | -1 |
| sklearn (train) | 1.6.1 |
| complete-case NWP | yes (no imputation) |
| script | `ml/train_v2_nwp_experiment_phase8d.py` |

**Feature order reconstruction:** sklearn joblib stores `n_features_in_ = 83`. `feature_names_in_` is **absent** on these dumps. Canonical order is the Phase 8D `feature_list` in each `metrics_B_atmospheric_nwp_target_*h.json` (identical across leads). The 13A contract copies that list exactly. Validation confirmed order match.

---

## 3. Exact 83-feature contract

Identity inputs (not model columns): `station_id`, `timestamp_utc`.

Model B columns in **saved training order** (Phase 3 = 73, NWP = 10):

1–14 current/cyclic/wind: `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`, `precipitation`, `cloud_cover`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `wind_direction_sin`, `wind_direction_cos`, `wind_u_10m`, `wind_v_10m`

15–44 lags (T−1/3/6/12/24h): temperature, humidity, pressure, wind_speed, precipitation, cloud_cover

45–56 tendencies: 1h and 3h change of temperature, humidity, pressure, wind_speed, precipitation, cloud_cover

57–70 rolling: precip sums 3/6/12/24h; 3h/6h means of humidity, pressure, temperature, wind_speed, cloud_cover

71–73 location: `latitude`, `longitude`, `elevation_m`

74–83 NWP: `nwp_cape`, `nwp_convective_inhibition`, `nwp_lifted_index`, `nwp_temperature_2m`, `nwp_relative_humidity_2m`, `nwp_surface_pressure`, `nwp_wind_speed_10m`, `nwp_wind_direction_10m`, `nwp_precipitation`, `nwp_cloud_cover`

Full per-column table: `outputs/v2_inference/phase13a/inference_contract.json` → `model_b_features`.

**Not in the 83:** `annual_thunder_hours`, satellite, radar, lightning observations, `weather_code`, `target_*`.

---

## 4. Feature groups

| group | count | origin |
| --- | --- | --- |
| current_state | 6 | observed (Open-Meteo hour T) |
| cyclic_time | 4 | derived from T |
| wind_vector | 4 | derived at T |
| lag | 30 | derived from T−k, k∈{1,3,6,12,24} |
| tendency | 12 | derived (T minus T−k) |
| rolling | 14 | derived, trailing window ending at T |
| location | 3 | observed station static |
| nwp | 10 | NWP valid at T |

Future groups **SATELLITE**, **RADAR**, **REAL_TIME_LIGHTNING** are architecture extension points only. They are **not** columns in this 83-feature model.

---

## 5. Input units / types

All 83 model features are `float` and **required**.

Typical units (Open-Meteo / Phase 3 / GFS Historical Forecast):

- temperature: °C  
- relative humidity / cloud cover: %  
- surface_pressure (Phase 3 atmospheric): Pa  
- `nwp_surface_pressure`: hPa (as stored in NWP join)  
- wind speed / u / v: m/s  
- precipitation: mm  
- CAPE / CIN: J/kg  
- lifted index: K  
- wind direction NWP: degrees  
- lat/lon: degrees; elevation: m  
- cyclic / dir sin/cos: dimensionless  

A future backend must feed **the same physical units as training**, in the frozen column order, as `float32` (Phase 8D `to_numpy(dtype=np.float32)`).

---

## 6. Temporal availability rule

**Prediction time = T** (`timestamp_utc`).

All predictors must be available **at or before T**.

The system must **never** use T+1, T+2, or T+3 observations as predictors.

| quantity | clock |
| --- | --- |
| features (atmospheric, lags, rolls, NWP) | ≤ T |
| `target_1h` | METAR at T+1h — **label only** |
| `target_2h` | METAR at T+2h — **label only** |
| `target_3h` | METAR at T+3h — **label only** |

NWP join: same UTC hour as T (`nwp_time_shift = false`). No future NWP valid-times as predictors.

Lags/rolls: Phase 3 causal construction (`shift` / trailing `rolling` only). First 24 hours of station history cannot form a complete vector.

---

## 7. Prediction output schema

Every prediction record:

| field | meaning |
| --- | --- |
| `prediction_timestamp_utc` | T |
| `station_id` | ICAO |
| `lead_1h_probability` | RF `predict_proba[:,1]` for 1h |
| `lead_1h_alert` | 1 iff probability ≥ `threshold_1h` |
| `lead_2h_probability` | RF score 2h |
| `lead_2h_alert` | 1 iff ≥ `threshold_2h` |
| `lead_3h_probability` | RF score 3h |
| `lead_3h_alert` | 1 iff ≥ `threshold_3h` |
| `model_version` | `V2-MODEL-B-MULTILEAD-PHASE8D-RESEARCH` |
| `threshold_1h` / `_2h` / `_3h` | frozen 0.065 |
| `data_completeness` | fraction of 83 features present and finite in [0, 1] |
| `data_status` | `COMPLETE` or `UNAVAILABLE` |

Do **not** invent confidence intervals. The RF output is a **probability / risk score**, not statistical confidence.

If `data_status = UNAVAILABLE`: set all six lead probability/alert fields to **null**. Do not call the models.

---

## 8. Frozen thresholds

From Phase 8D Model B validation (max F2, predicted positive rate ≤ 0.20; test unused for selection). **Not re-optimized in 13A.**

| lead | threshold |
| --- | --- |
| 1h | **0.065** |
| 2h | **0.065** |
| 3h | **0.065** |

**Alert rule (exact):** `alert = 1` if and only if `probability >= threshold`; else `alert = 0`.

---

## 9. Missing-data policy (fail-closed)

Phase 8D Model B was **complete-case**. There is **no** documented optional predictor.

| class | when | action |
| --- | --- | --- |
| **complete prediction** | all 83 finite; identity valid; station in {VOTV, VECC, VIDP, VOCI, VABB}; 24h lag history present; NWP all 10 present at T | `data_status=COMPLETE`; run all three RFs |
| **prediction with allowed documented missingness** | **none** for this model | not used |
| **unavailable prediction** | any required feature missing/non-finite; unknown station; missing identity; insufficient history; T outside NWP overlap; CIN or NWP precip missing | `data_status=UNAVAILABLE`; no RF call |

**Forbidden:** replace missing predictors with zero; invent observations; carry future observations backward; fabricate satellite/radar/lightning.

`data_completeness` may be &lt; 1 when unavailable; it must not be used to justify a silent fill.

---

## 10. NWP requirements

| item | value |
| --- | --- |
| product | Open-Meteo Historical Forecast `gfs_global` (forecast, not observation, not ERA5) |
| variables | 10 `nwp_*` listed in §3 |
| valid time | same T as the observation hour |
| coverage in repo | 2021-04-01T00:00Z → 2025-12-31T23:00Z |
| complete NWP rows | 204860 / 208320 overlap |
| known missing | `nwp_convective_inhibition` 1735 (0.8329%); `nwp_precipitation` 1725 (0.8281%) |
| leakage | none; no T+k NWP |

Phase 13A does **not** fetch new NWP. Historical Forecast **valid-time** GFS is **not** claimed to be available at live inference without qualification.

---

## 11. Future satellite / radar / lightning extension points

Documented groups only: `SATELLITE`, `RADAR`, `REAL_TIME_LIGHTNING`.

They must **not** be injected as placeholder zeros into the current 83-vector. Model B continues to run without them. A future multimodal model would be a **new** version string and feature list.

---

## 12. Model version

```
V2-MODEL-B-MULTILEAD-PHASE8D-RESEARCH
```

Label: **validated research / prototype**. Not a production release.

---

## 13. Validation results

Command: `python ml/validate_v2_inference_contract_phase13a.py`

| result | value |
| --- | --- |
| passed | **true** |
| stop | false |
| checks | 31 |
| failed | 0 |
| log | `outputs/v2_inference/phase13a/contract_validation.json` |

Verified:

- exactly 83 Model B features, unique names  
- feature order matches all three Phase 8D `feature_list` arrays  
- 1h/2h/3h joblib + metrics exist  
- `n_features_in_ == 83` on each RF  
- thresholds 0.065 frozen  
- no target / label / weather_code / annual_thunder_hours / satellite / radar / lightning in predictors  
- no T+1/T+2/T+3 predictor names  
- joblib SHA-256 unchanged after validation  

---

## 14. Limitations

- Not production; complete-case experiment on 2021–2025 NWP overlap only.  
- Valid-time Historical Forecast NWP is not an operational latency contract.  
- Uncalibrated RF scores; do not report as confidence.  
- Five ICAO stations only.  
- No allowed missingness; fail-closed if any of 83 is missing.  
- sklearn `feature_names_in_` not stored; order is metrics JSON, not the pickle.  
- Phase 8E imputed models are **out of contract**.  
- Phase 12B/12C lightning climatology / LOSO are **out of contract**.

---

## Artifacts

| path | role |
| --- | --- |
| `docs/PHASE13A_INFERENCE_CONTRACT.md` | this report |
| `outputs/v2_inference/phase13a/inference_contract.json` | machine-readable contract |
| `outputs/v2_inference/phase13a/contract_validation.json` | validation log |
| `ml/validate_v2_inference_contract_phase13a.py` | read-only validator |

Existing Phase 3/6/8/12 model files were not modified.

---

## PHASE STATUS

**READY** — Phase 13A contract and validation complete.

**STOP.** Do not implement Flask. Do not implement the dashboard. Do not start Phase 13B.
