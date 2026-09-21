# Phase 8D — First NWP model experiment (complete-case)

**Generated:** 2026-09-20T16:21:35.019638+00:00
**Script:** `ml/train_v2_nwp_experiment_phase8d.py`
**Scope:** experiment only. Not a production model. No imputation.

Phase 3/4/5/6 artifacts and V1 were not modified.

## Setup

| item | value |
| --- | --- |
| rows | overlap 2021-04-01 to 2025-12-31 with **all 10 NWP fields present** |
| Model A | Phase 3 73 atmospheric+location features |
| Model B | A + 10 `nwp_*` GFS Historical Forecast fields |
| A vs B rows | identical station+timestamp and identical targets per lead |
| RF | n_estimators=200, gini, sqrt, class_weight=balanced, random_state=42 |
| split | frozen Phase 6/4 UTC cutoffs; no shuffle; no new dates |
| threshold | max F2 on validation, alert rate <= 0.20; frozen before test |
| NWP | Open-Meteo Historical Forecast `gfs_global` (forecast, not observation) |

## Row counts (complete-case + observed target)

| lead | train n (pos) | val n (pos) | test n (pos) |
| --- | --- | --- | --- |
| target_1h | 46501 (1721) | 69273 (1902) | 73694 (2172) |
| target_2h | 46500 (1721) | 69275 (1902) | 73689 (2172) |
| target_3h | 46499 (1721) | 69277 (1902) | 73684 (2172) |

## Compact test comparison

| Lead | Atmospheric PR-AUC | Atmospheric+NWP PR-AUC | Atmospheric ROC-AUC | Atmospheric+NWP ROC-AUC | Atmospheric Recall | Atmospheric+NWP Recall | Atmospheric P | Atmospheric+NWP P | Atmospheric F1 | Atmospheric+NWP F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target_1h | 0.1319 | 0.1445 | 0.8360 | 0.8510 | 0.5866 | 0.6340 | 0.1136 | 0.1100 | 0.1904 | 0.1875 |
| target_2h | 0.1260 | 0.1368 | 0.8262 | 0.8443 | 0.5138 | 0.6229 | 0.1133 | 0.1074 | 0.1857 | 0.1833 |
| target_3h | 0.1242 | 0.1317 | 0.8180 | 0.8372 | 0.5879 | 0.6077 | 0.0980 | 0.1029 | 0.1681 | 0.1761 |

## Per-model test details

### target_1h

| model | threshold | PR-AUC | ROC-AUC | P | R | F1 | alert rate | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A atmospheric | 0.070000 | 0.1319 | 0.8360 | 0.1136 | 0.5866 | 0.1904 | 0.1522 | 61583 | 9939 | 898 | 1274 |
| B atmospheric+NWP | 0.065000 | 0.1445 | 0.8510 | 0.1100 | 0.6340 | 0.1875 | 0.1698 | 60386 | 11136 | 795 | 1377 |

Test by station (B):

| station | n | pos | PR-AUC | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| VOTV | 12899 | 509 | 0.2328 | 0.1485 | 0.7269 | 0.2467 |
| VECC | 15627 | 637 | 0.1274 | 0.1102 | 0.7206 | 0.1912 |
| VIDP | 15269 | 364 | 0.0732 | 0.0872 | 0.3269 | 0.1377 |
| VOCI | 14535 | 361 | 0.1479 | 0.0966 | 0.5900 | 0.1660 |
| VABB | 15364 | 301 | 0.1188 | 0.0944 | 0.7176 | 0.1669 |

### target_2h

| model | threshold | PR-AUC | ROC-AUC | P | R | F1 | alert rate | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A atmospheric | 0.080000 | 0.1260 | 0.8262 | 0.1133 | 0.5138 | 0.1857 | 0.1337 | 62784 | 8733 | 1056 | 1116 |
| B atmospheric+NWP | 0.065000 | 0.1368 | 0.8443 | 0.1074 | 0.6229 | 0.1833 | 0.1709 | 60277 | 11240 | 819 | 1353 |

Test by station (B):

| station | n | pos | PR-AUC | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| VOTV | 12898 | 509 | 0.2242 | 0.1479 | 0.7348 | 0.2463 |
| VECC | 15626 | 637 | 0.1229 | 0.1066 | 0.7143 | 0.1856 |
| VIDP | 15268 | 364 | 0.0669 | 0.0725 | 0.2857 | 0.1156 |
| VOCI | 14534 | 361 | 0.1256 | 0.0970 | 0.5568 | 0.1652 |
| VABB | 15363 | 301 | 0.1077 | 0.0956 | 0.7276 | 0.1690 |

### target_3h

| model | threshold | PR-AUC | ROC-AUC | P | R | F1 | alert rate | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A atmospheric | 0.065000 | 0.1242 | 0.8180 | 0.0980 | 0.5879 | 0.1681 | 0.1768 | 59764 | 11748 | 895 | 1277 |
| B atmospheric+NWP | 0.065000 | 0.1317 | 0.8372 | 0.1029 | 0.6077 | 0.1761 | 0.1740 | 60009 | 11503 | 852 | 1320 |

Test by station (B):

| station | n | pos | PR-AUC | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| VOTV | 12897 | 509 | 0.2160 | 0.1465 | 0.7191 | 0.2434 |
| VECC | 15625 | 637 | 0.1207 | 0.1002 | 0.7206 | 0.1759 |
| VIDP | 15267 | 364 | 0.0615 | 0.0647 | 0.2527 | 0.1030 |
| VOCI | 14533 | 361 | 0.1169 | 0.0946 | 0.5568 | 0.1617 |
| VABB | 15362 | 301 | 0.1029 | 0.0920 | 0.6711 | 0.1619 |

## Interpretation constraint

This is a complete-case experiment on the NWP overlap window only. It does **not** automatically mean NWP improves operational nowcasting. Possible outcomes include improvement, no change, degradation, or mixed results by lead. Do not treat this as a production model. Historical Forecast valid-time GFS is not claimed to be available at inference without qualification.

## Sanity / integrity

- Identical A/B keys and targets per lead: **True**
- NWP missing in used rows: **0**
- Duplicate station/timestamp: **0**
- Protected Phase 3/6/V1 hashes unchanged: **True**
- Thresholds selected on validation only.
- No imputation.

## Artifacts

| path | role |
| --- | --- |
| `ml/train_v2_nwp_experiment_phase8d.py` | this experiment |
| `outputs/v2_nwp_experiment/` | metrics JSON |
| `models/v2/nwp_experiment/` | experiment RFs (not production) |
| `docs/PHASE8D_NWP_EXPERIMENT_REPORT.md` | this report |

## PHASE STATUS

**READY FOR REVIEW** — stop here. Do not proceed to Phase 8E in this task.
