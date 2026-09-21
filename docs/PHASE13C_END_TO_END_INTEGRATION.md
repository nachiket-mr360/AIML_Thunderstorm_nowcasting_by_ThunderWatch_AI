# Phase 13C — V2 End-to-End Feature-to-Inference Pipeline

**Scope:** integration validation only. No training, no Flask, no new data fetch.

**This is NOT model evaluation.** Five samples do **not** demonstrate forecasting accuracy. Historical METAR targets are recorded only as `historical_reference_target_*` and were **never** passed into `V2InferenceEngine`.

---

## 1. Objective

Prove a real historical overlap row can travel:

prepared atmospheric + NWP CSV → existing Phase 3/NWP columns → Phase 13A 83-feature contract → Phase 13B `V2InferenceEngine` → 1h/2h/3h output with `data_status=COMPLETE`.

---

## 2. Data source

`dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025.csv`

Only rows where all **83** Model B features are present and finite. No formula recreation; columns are used as stored.

---

## 3. Sample-selection method

Deterministic, not random:

1. Keep complete-case rows (all 83 finite).
2. Sort by `station_id`, `timestamp_utc` (stable).
3. Take the **first** row for each of VOTV, VECC, VIDP, VOCI, VABB.

Script: `ml/run_v2_e2e_phase13c.py`

---

## 4. Five selected observations

| station_id | timestamp_utc |
| --- | --- |
| VOTV | 2021-04-01 00:00:00+00:00 |
| VECC | 2021-04-01 00:00:00+00:00 |
| VIDP | 2021-04-01 00:00:00+00:00 |
| VOCI | 2021-04-01 00:00:00+00:00 |
| VABB | 2021-04-01 00:00:00+00:00 |

---

## 5. Feature-contract verification

- Names and order loaded from `outputs/v2_inference/phase13a/inference_contract.json` → `model_b_features`.
- Count = 83.
- Identity: `station_id`, `timestamp_utc` only as metadata.
- No `target_*`, no T+1/T+2/T+3 predictors, no `annual_thunder_hours`, no satellite/radar/lightning columns in the inference vector.

---

## 6. Inference results

Engine: `src/inference/v2_inference_engine.py` (unchanged). Version `V2-MODEL-B-MULTILEAD-PHASE8D-RESEARCH`. All five `COMPLETE`, completeness 1.0.

| station | 1h p / alert | 2h p / alert | 3h p / alert | hist. targets 1/2/3 (not inputs) |
| --- | --- | --- | --- | --- |
| VOTV | 0.015 / 0 | 0.005 / 0 | 0.000 / 0 | 0 / 0 / 0 |
| VECC | 0.010 / 0 | 0.000 / 0 | 0.005 / 0 | 0 / 0 / 0 |
| VIDP | 0.005 / 0 | 0.010 / 0 | 0.000 / 0 | 0 / 0 / 0 |
| VOCI | 0.000 / 0 | 0.000 / 0 | 0.010 / 0 | 0 / 0 / 0 |
| VABB | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 | 0 / 0 / 0 |

Alert rule: `probability >= 0.065`. These five first-of-overlap hours all scored below threshold; that is **not** an accuracy claim.

CSV: `outputs/v2_inference/phase13c/e2e_prediction_samples.csv`

---

## 7. Integration checks

All passed for every sample: valid station/timestamp; 83 features; names/order match contract; all finite; no targets/future/climatology/sat/radar/lightning in X; engine `COMPLETE`; three finite probabilities; alerts match 0.065 rule.

Tests: `python -m pytest tests/test_v2_e2e_phase13c.py -q` → **9 passed**.

---

## 8. Model integrity checks

SHA-256 before = after for:

- three Phase 8D Model B joblibs
- Phase 13A `inference_contract.json`
- Phase 13B `v2_inference_engine.py`

No retraining. Thresholds still 0.065.

---

## 9. Not model evaluation

These rows only prove the **plumbing**. They are the first complete overlap hour (2021-04-01 00:00Z) at each station. Do not interpret probabilities vs historical labels as skill, calibration, or operational performance. Skill remains documented in Phase 8D test metrics.

---

## 10. Limitations

- Five rows, one timestamp, all historically negative METAR hours.
- Relies on pre-joined overlap CSV, not a live feature builder.
- Historical Forecast NWP at valid time T is not an operational latency contract.
- Research prototype version string only.

---

## Artifacts

| path | role |
| --- | --- |
| `ml/run_v2_e2e_phase13c.py` | selection + inference |
| `tests/test_v2_e2e_phase13c.py` | pytest |
| `outputs/v2_inference/phase13c/e2e_prediction_samples.csv` | sample outputs |
| `outputs/v2_inference/phase13c/e2e_validation.json` | validation log |
| `docs/PHASE13C_END_TO_END_INTEGRATION.md` | this report |

---

## PHASE STATUS

**READY** — Phase 13C complete.

**STOP.** Do not start Phase 13D. Do not implement Flask or the dashboard.
