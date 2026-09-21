# Phase 8E — NWP missing-value robustness (train-only median)

**Generated:** 2026-09-20T16:30:06.753670+00:00
**Script:** `ml/train_v2_nwp_experiment_phase8e.py`
**Scope:** robustness check only. Phase 8D complete-case remains the reference. Not a production model.

Phase 3/4/5/6/8D artifacts and V1 were not modified. Complete-case and atmospheric-only models were not retrained.

## Method

- Same overlap CSV, stations, leads, Phase 6 UTC cutoffs, RF config, and F2/alert-rate validation threshold as Phase 8D.
- Keep rows with missing `nwp_convective_inhibition` or `nwp_precipitation`.
- Median of each of those two columns fit on **train only**, then applied to train/val/test.
- No forward/back fill, no time interpolation, targets not filled.
- Other eight NWP fields had zero missingness and were not imputed.

## Imputation (train medians)

| lead | nwp_convective_inhibition median | nwp_precipitation median | train imputed CIN | train imputed precip | val imputed CIN | val imputed precip | test imputed CIN | test imputed precip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target_1h | -7.000000 | 0.000000 | 0 | 0 | 1625 | 1660 | 0 | 0 |
| target_2h | -7.000000 | 0.000000 | 0 | 0 | 1625 | 1659 | 0 | 0 |
| target_3h | -6.000000 | 0.000000 | 0 | 0 | 1625 | 1658 | 0 | 0 |

## Row counts vs Phase 8D complete-case

| lead | 8E train (pos) | 8D train | 8E val (pos) | 8D val | 8E test (pos) | 8D test |
| --- | --- | --- | --- | --- | --- | --- |
| target_1h | 46501 (1721) | 46501 | 72558 (1965) | 69273 | 73694 (2172) | 73694 |
| target_2h | 46500 (1721) | 46500 | 72559 (1965) | 69275 | 73689 (2172) | 73689 |
| target_3h | 46499 (1721) | 46499 | 72560 (1965) | 69277 | 73684 (2172) | 73684 |

## Compact test comparison vs Phase 8D NWP complete-case

| Lead | Phase 8D Complete-case PR-AUC | Phase 8E Imputed PR-AUC | Phase 8D Recall | Phase 8E Recall | Phase 8D F1 | Phase 8E F1 |
| --- | --- | --- | --- | --- | --- | --- |
| target_1h | 0.1445 | 0.1445 | 0.6340 | 0.6340 | 0.1875 | 0.1875 |
| target_2h | 0.1368 | 0.1368 | 0.6229 | 0.6229 | 0.1833 | 0.1833 |
| target_3h | 0.1317 | 0.1317 | 0.6077 | 0.6077 | 0.1761 | 0.1761 |

## Phase 8E test details

| lead | threshold | PR-AUC | ROC-AUC | P | R | F1 | alert rate | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target_1h | 0.065000 | 0.1445 | 0.8510 | 0.1100 | 0.6340 | 0.1875 | 0.1698 | 60386 | 11136 | 795 | 1377 |
| target_2h | 0.065000 | 0.1368 | 0.8443 | 0.1074 | 0.6229 | 0.1833 | 0.1709 | 60277 | 11240 | 819 | 1353 |
| target_3h | 0.065000 | 0.1317 | 0.8372 | 0.1029 | 0.6077 | 0.1761 | 0.1740 | 60009 | 11503 | 852 | 1320 |

## Interpretation

Imputation is not claimed to improve the model. The question is whether Phase 8D's complete-case NWP ranking-skill pattern is reasonably stable when the ~3,460 NWP-null rows are retained via train-only medians. Row counts differ (especially validation, which contained both documented CIN and precip null windows). Differences may reflect extra rows, extra positives, and/or the filled values—not a production recommendation.

## Integrity

- Medians fit on train only.
- Thresholds selected on validation only.
- Phase 8D files hash-unchanged.
- No FF/BF/time interpolation; targets not filled.

## PHASE STATUS

**READY FOR REVIEW** — stop. Do not proceed to Phase 9 in this task.
