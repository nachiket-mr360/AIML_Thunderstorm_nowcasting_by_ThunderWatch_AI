# Phase 6 — Multi-lead nowcast models (V2)

**Generated:** 2026-09-19T18:10:42.575773+00:00
**Script:** `ml/train_v2_multilead_phase6.py`
**PHASE STATUS:** **READY**

V1, Flask, and frontend were not modified. Feature and target CSVs were not rewritten.
These are independent pooled Random Forests for T+1h / T+2h / T+3h. They are **not**
copied from V1 2h/3h stub files. No winner is declared from a single metric.

## Task

| item | value |
| --- | --- |
| features | Phase 3 causal 73 columns at hour **T** |
| targets | Phase 5 genuine METAR `target_1h` / `target_2h` / `target_3h` |
| NA labels | excluded from train/val/test; never converted to 0 |
| future atmosphere | not used |
| locations | VOTV, VECC, VIDP, VOCI, VABB (pooled train) |
| model | `RandomForestClassifier` (three independent fits) |
| hyperparameters | n_estimators=200, criterion=gini, max_features=sqrt, class_weight=balanced, random_state=42, n_jobs=-1 |
| split | **same** Phase 4 chronological UTC boundaries; no shuffle |
| threshold | max F2 on validation, alert rate ≤ 0.20; frozen before test; **per horizon** |

## Split boundaries (shared across horizons)

| partition | start UTC | end UTC |
| --- | --- | --- |
| train | 2014-01-02T00:00:00+00:00 | 2022-05-27T01:00:00+00:00 |
| validation | 2022-05-27T02:00:00+00:00 | 2024-03-14T11:00:00+00:00 |
| test | 2024-03-14T12:00:00+00:00 | 2025-12-30T23:00:00+00:00 |

Train max timestamp is strictly before validation min; validation max is strictly before test min.
Row counts differ slightly by horizon because a missing METAR at T+L drops that row.

## Thresholds (validation only)

| horizon | locked threshold | selected on test? |
| --- | --- | --- |
| target_1h | 0.060000 | no |
| target_2h | 0.065000 | no |
| target_3h | 0.065000 | no |

## target_1h

Train n=334389 pos=9204; val n=72558 pos=1965; test n=73694 pos=2172.

### Pooled target_1h

| split | n | pos | pos rate | alert rate | acc | P | R | F1 | ROC-AUC | PR-AUC | FP | FN | TN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 334389 | 9204 | 0.0275 | 0.0510 | 0.9766 | 0.5400 | 1.0000 | 0.7013 | 1.0000 | 1.0000 | 7841 | 0 | 317344 | 9204 |
| validation | 72558 | 1965 | 0.0271 | 0.1089 | 0.8959 | 0.1465 | 0.5893 | 0.2347 | 0.8729 | 0.1799 | 6746 | 807 | 63847 | 1158 |
| test | 73694 | 2172 | 0.0295 | 0.1375 | 0.8672 | 0.1244 | 0.5801 | 0.2048 | 0.8462 | 0.1511 | 8871 | 912 | 62651 | 1260 |

### Test by location target_1h

| split | n | pos | pos rate | alert rate | acc | P | R | F1 | ROC-AUC | PR-AUC | FP | FN | TN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | 12899 | 509 | 0.0395 | 0.1760 | 0.8374 | 0.1502 | 0.6699 | 0.2454 | 0.8571 | 0.2070 | 1929 | 168 | 10461 | 341 |
| VECC | 15627 | 637 | 0.0408 | 0.2228 | 0.7915 | 0.1235 | 0.6750 | 0.2088 | 0.8294 | 0.1486 | 3052 | 207 | 11938 | 430 |
| VIDP | 15269 | 364 | 0.0238 | 0.0836 | 0.9088 | 0.0972 | 0.3407 | 0.1512 | 0.8178 | 0.0813 | 1152 | 240 | 13753 | 124 |
| VOCI | 14535 | 361 | 0.0248 | 0.1058 | 0.8949 | 0.1209 | 0.5152 | 0.1959 | 0.8097 | 0.1480 | 1352 | 175 | 12822 | 186 |
| VABB | 15364 | 301 | 0.0196 | 0.1019 | 0.9018 | 0.1144 | 0.5947 | 0.1919 | 0.8925 | 0.1539 | 1386 | 122 | 13677 | 179 |
| pooled | 73694 | 2172 | 0.0295 | 0.1375 | 0.8672 | 0.1244 | 0.5801 | 0.2048 | 0.8462 | 0.1511 | 8871 | 912 | 62651 | 1260 |

Top impurity importance (not a causal ranking):

| rank | feature | importance |
| --- | --- | --- |
| 1 | precipitation_roll_sum_12h | 0.058383 |
| 2 | precipitation_roll_sum_6h | 0.057249 |
| 3 | precipitation_roll_sum_3h | 0.053140 |
| 4 | precipitation_roll_sum_24h | 0.042611 |
| 5 | temperature_lag_3h | 0.021130 |
| 6 | temperature_roll_mean_6h | 0.020199 |
| 7 | cloud_cover_lag_3h | 0.020152 |
| 8 | cloud_cover_roll_mean_3h | 0.019853 |
| 9 | precipitation_lag_1h | 0.019717 |
| 10 | temperature_lag_6h | 0.019546 |

## target_2h

Train n=334388 pos=9204; val n=72559 pos=1965; test n=73689 pos=2172.

### Pooled target_2h

| split | n | pos | pos rate | alert rate | acc | P | R | F1 | ROC-AUC | PR-AUC | FP | FN | TN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 334388 | 9204 | 0.0275 | 0.0467 | 0.9808 | 0.5895 | 1.0000 | 0.7417 | 1.0000 | 1.0000 | 6409 | 0 | 318775 | 9204 |
| validation | 72559 | 1965 | 0.0271 | 0.0969 | 0.9044 | 0.1466 | 0.5247 | 0.2291 | 0.8620 | 0.1690 | 6003 | 934 | 64591 | 1031 |
| test | 73689 | 2172 | 0.0295 | 0.1246 | 0.8770 | 0.1247 | 0.5272 | 0.2016 | 0.8423 | 0.1435 | 8040 | 1027 | 63477 | 1145 |

### Test by location target_2h

| split | n | pos | pos rate | alert rate | acc | P | R | F1 | ROC-AUC | PR-AUC | FP | FN | TN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | 12898 | 509 | 0.0395 | 0.1645 | 0.8480 | 0.1579 | 0.6582 | 0.2547 | 0.8587 | 0.2031 | 1787 | 174 | 10602 | 335 |
| VECC | 15626 | 637 | 0.0408 | 0.2065 | 0.8020 | 0.1193 | 0.6044 | 0.1993 | 0.8226 | 0.1441 | 2842 | 252 | 12147 | 385 |
| VIDP | 15268 | 364 | 0.0238 | 0.0722 | 0.9167 | 0.0880 | 0.2665 | 0.1323 | 0.8117 | 0.0709 | 1005 | 267 | 13899 | 97 |
| VOCI | 14534 | 361 | 0.0248 | 0.0932 | 0.9050 | 0.1233 | 0.4626 | 0.1948 | 0.8047 | 0.1326 | 1187 | 194 | 12986 | 167 |
| VABB | 15363 | 301 | 0.0196 | 0.0898 | 0.9115 | 0.1167 | 0.5349 | 0.1916 | 0.8882 | 0.1429 | 1219 | 140 | 13843 | 161 |
| pooled | 73689 | 2172 | 0.0295 | 0.1246 | 0.8770 | 0.1247 | 0.5272 | 0.2016 | 0.8423 | 0.1435 | 8040 | 1027 | 63477 | 1145 |

Top impurity importance (not a causal ranking):

| rank | feature | importance |
| --- | --- | --- |
| 1 | precipitation_roll_sum_6h | 0.054712 |
| 2 | precipitation_roll_sum_3h | 0.053461 |
| 3 | precipitation_roll_sum_12h | 0.052895 |
| 4 | precipitation_roll_sum_24h | 0.042506 |
| 5 | precipitation | 0.023406 |
| 6 | cloud_cover | 0.021665 |
| 7 | temperature_lag_3h | 0.020737 |
| 8 | temperature_roll_mean_6h | 0.019633 |
| 9 | cloud_cover_lag_3h | 0.018600 |
| 10 | cloud_cover_roll_mean_6h | 0.017991 |

## target_3h

Train n=334387 pos=9204; val n=72560 pos=1965; test n=73684 pos=2172.

### Pooled target_3h

| split | n | pos | pos rate | alert rate | acc | P | R | F1 | ROC-AUC | PR-AUC | FP | FN | TN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 334387 | 9204 | 0.0275 | 0.0459 | 0.9816 | 0.6000 | 1.0000 | 0.7500 | 1.0000 | 1.0000 | 6137 | 0 | 319046 | 9204 |
| validation | 72560 | 1965 | 0.0271 | 0.0966 | 0.9034 | 0.1402 | 0.5003 | 0.2191 | 0.8504 | 0.1570 | 6027 | 982 | 64568 | 983 |
| test | 73684 | 2172 | 0.0295 | 0.1242 | 0.8758 | 0.1186 | 0.4995 | 0.1917 | 0.8346 | 0.1367 | 8064 | 1087 | 63448 | 1085 |

### Test by location target_3h

| split | n | pos | pos rate | alert rate | acc | P | R | F1 | ROC-AUC | PR-AUC | FP | FN | TN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | 12897 | 509 | 0.0395 | 0.1641 | 0.8454 | 0.1493 | 0.6208 | 0.2407 | 0.8464 | 0.2014 | 1801 | 193 | 10587 | 316 |
| VECC | 15625 | 637 | 0.0408 | 0.2056 | 0.8027 | 0.1192 | 0.6013 | 0.1990 | 0.8181 | 0.1398 | 2829 | 254 | 12159 | 383 |
| VIDP | 15267 | 364 | 0.0238 | 0.0733 | 0.9128 | 0.0679 | 0.2088 | 0.1025 | 0.7948 | 0.0617 | 1043 | 288 | 13860 | 76 |
| VOCI | 14533 | 361 | 0.0248 | 0.0897 | 0.9068 | 0.1189 | 0.4294 | 0.1862 | 0.8031 | 0.1222 | 1149 | 206 | 13023 | 155 |
| VABB | 15362 | 301 | 0.0196 | 0.0909 | 0.9096 | 0.1110 | 0.5150 | 0.1826 | 0.8862 | 0.1252 | 1242 | 146 | 13819 | 155 |
| pooled | 73684 | 2172 | 0.0295 | 0.1242 | 0.8758 | 0.1186 | 0.4995 | 0.1917 | 0.8346 | 0.1367 | 8064 | 1087 | 63448 | 1085 |

Top impurity importance (not a causal ranking):

| rank | feature | importance |
| --- | --- | --- |
| 1 | precipitation_roll_sum_3h | 0.048443 |
| 2 | precipitation_roll_sum_24h | 0.047696 |
| 3 | precipitation_roll_sum_6h | 0.046252 |
| 4 | precipitation_roll_sum_12h | 0.046138 |
| 5 | precipitation | 0.022756 |
| 6 | temperature_lag_24h | 0.020894 |
| 7 | temperature_roll_mean_3h | 0.020248 |
| 8 | cloud_cover | 0.020079 |
| 9 | temperature_roll_mean_6h | 0.019938 |
| 10 | cloud_cover_roll_mean_3h | 0.018872 |

## Comparison — pooled test (no ranking claim)

Phase 4 is a **different task** (label at T, 100 trees). Leads use 200 trees and T+L labels.
Read the curve as skill vs lead time, not as a contest.

### Pooled test across tasks

| split | n | pos | pos rate | alert rate | acc | P | R | F1 | ROC-AUC | PR-AUC | FP | FN | TN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 4 same-hour (B, 73 feat, 100 trees) | 73699 | 2172 | 0.0295 | 0.1236 | 0.8788 | 0.1292 | 0.5419 | 0.2086 | 0.8388 | 0.1518 | 7935 | 995 | 63592 | 1177 |
| 1h nowcast | 73694 | 2172 | 0.0295 | 0.1375 | 0.8672 | 0.1244 | 0.5801 | 0.2048 | 0.8462 | 0.1511 | 8871 | 912 | 62651 | 1260 |
| 2h nowcast | 73689 | 2172 | 0.0295 | 0.1246 | 0.8770 | 0.1247 | 0.5272 | 0.2016 | 0.8423 | 0.1435 | 8040 | 1027 | 63477 | 1145 |
| 3h nowcast | 73684 | 2172 | 0.0295 | 0.1242 | 0.8758 | 0.1186 | 0.4995 | 0.1917 | 0.8346 | 0.1367 | 8064 | 1087 | 63448 | 1085 |

## Validation checks

- `models_reload`: **True**
- `models_not_stubs`: **True**
- `chronological_split`: **True**
- `same_utc_boundaries_all_horizons`: **True**
- `no_future_feature_leakage`: **True**
- `no_na_targets_in_training`: **True**
- `thresholds_frozen_before_test`: **True**
- `station_boundaries_respected`: **True**
- `three_horizons_trained`: **True**
- `v1_untouched`: **True**
- `weather_code_not_used`: **True**

## Artifacts

| path | role |
| --- | --- |
| `models/v2/multilead/pooled_rf_target_1h.joblib` | target_1h RF |
| `models/v2/multilead/metadata_target_1h.json` | metadata |
| `outputs/v2_multilead/metrics_target_1h.json` | full metrics |
| `models/v2/multilead/pooled_rf_target_2h.joblib` | target_2h RF |
| `models/v2/multilead/metadata_target_2h.json` | metadata |
| `outputs/v2_multilead/metrics_target_2h.json` | full metrics |
| `models/v2/multilead/pooled_rf_target_3h.joblib` | target_3h RF |
| `models/v2/multilead/metadata_target_3h.json` | metadata |
| `outputs/v2_multilead/metrics_target_3h.json` | full metrics |
| `docs/PHASE6_MULTILEAD_MODEL_REPORT.md` | this report |

## Not done

- No Flask/frontend/V1 edits
- No NWP / lightning / satellite / radar
- No spatial prediction
- No git commit or push
