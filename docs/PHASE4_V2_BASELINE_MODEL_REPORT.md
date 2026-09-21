# Phase 4 — V2 pooled multi-location baseline model

**Generated:** 2026-09-19T17:28:37.648805+00:00
**Script:** `ml/train_v2_pooled_baseline_phase4.py`
**PHASE STATUS:** **READY**

V1 was not modified. Flask was not modified. The synchronized and feature CSVs were not rewritten.
This establishes a V2 pooled same-hour baseline. It is **not** a claim that pooled multi-location
training is superior to V1 because more locations exist. Tasks differ (V1: 1h lead, VOTV-only).

## Task

| item | value |
| --- | --- |
| target | `thunderstorm_target` genuine METAR (1 / 0); NA excluded |
| NA handling | never converted to negative examples |
| features at | hour T (causal lags/rolls at or before T) |
| label at | hour T (same-hour nowcast, not t+1) |
| locations | VOTV, VECC, VIDP, VOCI, VABB |
| labeled rows used | 480645 of 525840 feature rows |
| model | `RandomForestClassifier` |
| hyperparameters | n_estimators=100, criterion=gini, max_features=sqrt, class_weight=balanced, random_state=42, n_jobs=-1. V1 used 400 trees on ~54k VOTV rows; V2 uses 100 trees on ~334k pooled train rows (same RF family). |
| split | chronological unique UTC hours 70/15/15; no shuffle |
| threshold | max F2 on validation with predicted alert rate ≤ 0.20; frozen before test |

## Split boundaries (shared across stations)

| partition | start UTC | end UTC | labeled rows | unique hours |
| --- | --- | --- | --- | --- |
| train | 2014-01-02T00:00:00+00:00 | 2022-05-27T01:00:00+00:00 | 334388 | 73409 |
| validation | 2022-05-27T02:00:00+00:00 | 2024-03-14T11:00:00+00:00 | 72558 | 15730 |
| test | 2024-03-14T12:00:00+00:00 | 2025-12-30T23:00:00+00:00 | 73699 | 15732 |

Train max timestamp is strictly before validation min; validation max is strictly before test min.

## Threshold rule

Do **not** use 0.5. Class-weighted forests typically score well below 0.5.
Candidates: distinct validation probabilities. Guardrail: predicted positive rate ≤ 20%.
Among survivors, maximise F2 (β=2). Tie-break: lower threshold. Then freeze and score test once.

| experiment | locked threshold | selected on test? |
| --- | --- | --- |
| A atmospheric-only | 0.070000 | no |
| B atmospheric+location | 0.070000 | no |

## Experiment A — atmospheric only (70 features)

### Pooled A

| split | n | pos | pos rate | alert rate | acc | P | R | F1 | ROC-AUC | PR-AUC | FP | FN | TN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 334388 | 9204 | 0.0275 | 0.0477 | 0.9798 | 0.5773 | 1.0000 | 0.7320 | 1.0000 | 1.0000 | 6740 | 0 | 318444 | 9204 |
| validation | 72558 | 1965 | 0.0271 | 0.0992 | 0.9022 | 0.1437 | 0.5267 | 0.2259 | 0.8520 | 0.1628 | 6165 | 930 | 64428 | 1035 |
| test | 73699 | 2172 | 0.0295 | 0.1234 | 0.8786 | 0.1275 | 0.5341 | 0.2059 | 0.8371 | 0.1437 | 7937 | 1012 | 63590 | 1160 |

### Test by location A

| split | n | pos | pos rate | alert rate | acc | P | R | F1 | ROC-AUC | PR-AUC | FP | FN | TN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | 12900 | 509 | 0.0395 | 0.1323 | 0.8753 | 0.1781 | 0.5972 | 0.2744 | 0.8517 | 0.2179 | 1403 | 205 | 10988 | 304 |
| VECC | 15628 | 637 | 0.0408 | 0.1936 | 0.8139 | 0.1246 | 0.5918 | 0.2059 | 0.8222 | 0.1472 | 2648 | 260 | 12343 | 377 |
| VIDP | 15270 | 364 | 0.0238 | 0.0680 | 0.9241 | 0.1174 | 0.3352 | 0.1739 | 0.8178 | 0.0920 | 917 | 242 | 13989 | 122 |
| VOCI | 14536 | 361 | 0.0248 | 0.1083 | 0.8916 | 0.1144 | 0.4986 | 0.1860 | 0.8065 | 0.1199 | 1394 | 181 | 12781 | 180 |
| VABB | 15365 | 301 | 0.0196 | 0.1140 | 0.8894 | 0.1010 | 0.5880 | 0.1724 | 0.8722 | 0.1284 | 1575 | 124 | 13489 | 177 |
| pooled | 73699 | 2172 | 0.0295 | 0.1234 | 0.8786 | 0.1275 | 0.5341 | 0.2059 | 0.8371 | 0.1437 | 7937 | 1012 | 63590 | 1160 |

## Experiment B — atmospheric + location (73 features)

### Pooled B

| split | n | pos | pos rate | alert rate | acc | P | R | F1 | ROC-AUC | PR-AUC | FP | FN | TN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 334388 | 9204 | 0.0275 | 0.0475 | 0.9800 | 0.5795 | 1.0000 | 0.7337 | 1.0000 | 1.0000 | 6680 | 0 | 318504 | 9204 |
| validation | 72558 | 1965 | 0.0271 | 0.1001 | 0.9022 | 0.1471 | 0.5440 | 0.2316 | 0.8584 | 0.1741 | 6197 | 896 | 64396 | 1069 |
| test | 73699 | 2172 | 0.0295 | 0.1236 | 0.8788 | 0.1292 | 0.5419 | 0.2086 | 0.8388 | 0.1518 | 7935 | 995 | 63592 | 1177 |

### Test by location B

| split | n | pos | pos rate | alert rate | acc | P | R | F1 | ROC-AUC | PR-AUC | FP | FN | TN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | 12900 | 509 | 0.0395 | 0.1619 | 0.8503 | 0.1595 | 0.6542 | 0.2564 | 0.8520 | 0.2132 | 1755 | 176 | 10636 | 333 |
| VECC | 15628 | 637 | 0.0408 | 0.2015 | 0.8084 | 0.1258 | 0.6217 | 0.2092 | 0.8237 | 0.1492 | 2753 | 241 | 12238 | 396 |
| VIDP | 15270 | 364 | 0.0238 | 0.0735 | 0.9187 | 0.1095 | 0.3379 | 0.1654 | 0.8192 | 0.0871 | 1000 | 241 | 13906 | 123 |
| VOCI | 14536 | 361 | 0.0248 | 0.0966 | 0.9024 | 0.1232 | 0.4792 | 0.1960 | 0.8064 | 0.1363 | 1231 | 188 | 12944 | 173 |
| VABB | 15365 | 301 | 0.0196 | 0.0877 | 0.9125 | 0.1128 | 0.5050 | 0.1844 | 0.8655 | 0.1583 | 1196 | 149 | 13868 | 152 |
| pooled | 73699 | 2172 | 0.0295 | 0.1236 | 0.8788 | 0.1292 | 0.5419 | 0.2086 | 0.8388 | 0.1518 | 7935 | 995 | 63592 | 1177 |

## Does location metadata add value?

| metric (validation, frozen threshold) | A | B |
| --- | --- | --- |
| pr_auc | 0.1628 | 0.1741 |
| roc_auc | 0.8520 | 0.8584 |
| f1 | 0.2259 | 0.2316 |
| recall | 0.5267 | 0.5440 |
| precision | 0.1437 | 0.1471 |

| metric (test, frozen threshold) | A | B |
| --- | --- | --- |
| pr_auc | 0.1437 | 0.1518 |
| roc_auc | 0.8371 | 0.8388 |
| f1 | 0.2059 | 0.2086 |
| recall | 0.5341 | 0.5419 |
| precision | 0.1275 | 0.1292 |

Preferred experiment by **validation PR-AUC** (chosen without using test): `B_atmospheric_plus_location`.

## V1 1-hour metrics (historical reference only)

Source: `outputs/PHASE6_MODEL_TRAINING_REPORT.md`. VOTV-only, 28 features, **t+1h** genuine METAR.
Not comparable as a superiority claim.

| | V1 1h test | V2-A same-hour pooled test | V2-B same-hour pooled test |
| --- | --- | --- | --- |
| recall | 0.6040 | 0.5341 | 0.5419 |
| precision | 0.1332 | 0.1275 | 0.1292 |
| F1 | 0.2183 | 0.2059 | 0.2086 |
| PR-AUC | 0.2060 | 0.1437 | 0.1518 |
| ROC-AUC | 0.8453 | 0.8371 | 0.8388 |
| threshold | 0.077500 | 0.070000 | 0.070000 |

## Validation checklist

- `no_future_leakage_time_cutoffs`: **True**
- `chronological_split`: **True**
- `no_na_targets_in_training`: **True**
- `threshold_frozen_before_test`: **True**
- `all_five_locations_represented`: **True**
- `models_reload`: **True**
- `weather_code_not_used`: **True**
- `v1_untouched`: **True**
- `random_split_not_used`: **True**
- `na_not_converted_to_negatives`: **True**

## Artifacts

| path | role |
| --- | --- |
| `models/v2/pooled_rf_A_atmospheric_only.joblib` | experiment A model |
| `models/v2/pooled_rf_B_atmospheric_plus_location.joblib` | experiment B model |
| `models/v2/metadata_*.json` | features, threshold, split, hyperparameters |
| `outputs/v2_baseline/` | metrics JSON, importance, split, checks |
| `docs/PHASE4_V2_BASELINE_MODEL_REPORT.md` | this report |

## Not done

- No NWP, lightning, satellite, radar
- No Flask / V1 changes
- No git commit/push
- No t+1/t+2/t+3 V2 models

