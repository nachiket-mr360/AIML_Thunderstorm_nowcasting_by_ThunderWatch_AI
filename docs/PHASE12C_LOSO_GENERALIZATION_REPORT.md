# Phase 12C — Leave-one-station-out generalization

**Generated:** 2026-09-21T05:10:21.885147+00:00
**Script:** `ml/train_v2_loso_phase12c.py`
**Label:** LEAVE-ONE-STATION-OUT GENERALIZATION
**Scope:** experiment only. Not a replacement for the frozen Phase 8D pooled chronological benchmark.

Phase 3 / 6 / 8A–8E / 12B artifacts were not modified. Frozen models were not overwritten.

## 1. Objective

Evaluate whether validated V2 **Model B** (atmospheric + location/temporal + NWP) transfers to a station that never appears in training or threshold selection.

## 2. Dataset

| item | value |
| --- | --- |
| CSV | `dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025.csv` |
| complete-case NWP rows | 204860 / 208320 |
| missing targets | excluded, not filled with 0 |
| lightning / satellite / radar | not used |

## 3. Feature groups

Model B: **83** columns = Phase 3 atmospheric + location/temporal (73) + 10 `nwp_*` (Phase 8D).

## 4. LOSO methodology

Five independent experiments. For each held-out ICAO:

- Fit RF on the other four stations, **train** timestamps only (`timestamp <= train_end`).
- Select threshold on the other four stations, **validation** timestamps only.
- Evaluate on the held-out station, **test** timestamps only (`timestamp > validation_end`).

This keeps chronological validation **entirely before** held-out test evaluation, and keeps the held-out station out of fitting and thresholding.

This is **spatial transfer**, not the Phase 8D pooled test. Do not treat LOSO metrics as beating or replacing 8D.

## 5. Station split table

| held-out test | train stations |
| --- | --- |
| VOTV | VECC, VIDP, VOCI, VABB |
| VECC | VOTV, VIDP, VOCI, VABB |
| VIDP | VOTV, VECC, VOCI, VABB |
| VOCI | VOTV, VECC, VIDP, VABB |
| VABB | VOTV, VECC, VIDP, VOCI |

## 6. Threshold methodology

- Same Phase 8D rule: maximize validation **F2**, predicted alert rate **≤ 0.20**, tie-break lower threshold.
- Validation pool = training stations only.
- Frozen threshold applied to the unseen station’s test window.
- Test station unused for threshold selection.

- Train end UTC: `2022-05-27T01:00:00+00:00`
- Validation end UTC: `2024-03-14T11:00:00+00:00`

## 7. 1h results

| held-out | n_test | pos | threshold | PR-AUC | ROC-AUC | P | R | F1 | F2 | alert rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | 12899 | 509 | 0.065000 | 0.2199 | 0.8301 | 0.2237 | 0.4086 | 0.2891 | 0.3506 | 0.0721 |
| VECC | 15627 | 637 | 0.070000 | 0.1059 | 0.8007 | 0.1111 | 0.5416 | 0.1844 | 0.3051 | 0.1987 |
| VIDP | 15269 | 364 | 0.080000 | 0.0549 | 0.7344 | 0.0625 | 0.3681 | 0.1069 | 0.1861 | 0.1404 |
| VOCI | 14535 | 361 | 0.075000 | 0.0665 | 0.7827 | 0.0649 | 0.6510 | 0.1180 | 0.2319 | 0.2492 |
| VABB | 15364 | 301 | 0.065000 | 0.0728 | 0.8463 | 0.0579 | 0.8206 | 0.1081 | 0.2257 | 0.2777 |

## 8. 2h results

| held-out | n_test | pos | threshold | PR-AUC | ROC-AUC | P | R | F1 | F2 | alert rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | 12898 | 509 | 0.060000 | 0.2057 | 0.8288 | 0.2118 | 0.4165 | 0.2808 | 0.3490 | 0.0776 |
| VECC | 15626 | 637 | 0.060000 | 0.0980 | 0.7878 | 0.1025 | 0.5699 | 0.1738 | 0.2981 | 0.2266 |
| VIDP | 15268 | 364 | 0.085000 | 0.0323 | 0.5729 | 0.0341 | 0.4780 | 0.0637 | 0.1327 | 0.3339 |
| VOCI | 14534 | 361 | 0.065000 | 0.0633 | 0.7754 | 0.0584 | 0.6953 | 0.1078 | 0.2187 | 0.2955 |
| VABB | 15363 | 301 | 0.065000 | 0.0590 | 0.8227 | 0.0533 | 0.8106 | 0.1000 | 0.2109 | 0.2982 |

## 9. 3h results

| held-out | n_test | pos | threshold | PR-AUC | ROC-AUC | P | R | F1 | F2 | alert rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | 12897 | 509 | 0.060000 | 0.1958 | 0.8287 | 0.2062 | 0.4342 | 0.2796 | 0.3555 | 0.0831 |
| VECC | 15625 | 637 | 0.070000 | 0.1009 | 0.7838 | 0.1020 | 0.5447 | 0.1718 | 0.2915 | 0.2178 |
| VIDP | 15267 | 364 | 0.075000 | 0.0383 | 0.5876 | 0.0315 | 0.5659 | 0.0597 | 0.1288 | 0.4284 |
| VOCI | 14533 | 361 | 0.075000 | 0.0627 | 0.7656 | 0.0613 | 0.5956 | 0.1112 | 0.2171 | 0.2413 |
| VABB | 15362 | 301 | 0.060000 | 0.0556 | 0.8118 | 0.0493 | 0.8140 | 0.0929 | 0.1983 | 0.3237 |

## 10. Per-station results

Pooled station × lead table (same numbers as above; not a ranking).

| station | lead | n_test | pos | threshold | PR-AUC | ROC-AUC | P | R | F1 | F2 | alert rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | target_1h | 12899 | 509 | 0.065000 | 0.2199 | 0.8301 | 0.2237 | 0.4086 | 0.2891 | 0.3506 | 0.0721 |
| VOTV | target_2h | 12898 | 509 | 0.060000 | 0.2057 | 0.8288 | 0.2118 | 0.4165 | 0.2808 | 0.3490 | 0.0776 |
| VOTV | target_3h | 12897 | 509 | 0.060000 | 0.1958 | 0.8287 | 0.2062 | 0.4342 | 0.2796 | 0.3555 | 0.0831 |
| VECC | target_1h | 15627 | 637 | 0.070000 | 0.1059 | 0.8007 | 0.1111 | 0.5416 | 0.1844 | 0.3051 | 0.1987 |
| VECC | target_2h | 15626 | 637 | 0.060000 | 0.0980 | 0.7878 | 0.1025 | 0.5699 | 0.1738 | 0.2981 | 0.2266 |
| VECC | target_3h | 15625 | 637 | 0.070000 | 0.1009 | 0.7838 | 0.1020 | 0.5447 | 0.1718 | 0.2915 | 0.2178 |
| VIDP | target_1h | 15269 | 364 | 0.080000 | 0.0549 | 0.7344 | 0.0625 | 0.3681 | 0.1069 | 0.1861 | 0.1404 |
| VIDP | target_2h | 15268 | 364 | 0.085000 | 0.0323 | 0.5729 | 0.0341 | 0.4780 | 0.0637 | 0.1327 | 0.3339 |
| VIDP | target_3h | 15267 | 364 | 0.075000 | 0.0383 | 0.5876 | 0.0315 | 0.5659 | 0.0597 | 0.1288 | 0.4284 |
| VOCI | target_1h | 14535 | 361 | 0.075000 | 0.0665 | 0.7827 | 0.0649 | 0.6510 | 0.1180 | 0.2319 | 0.2492 |
| VOCI | target_2h | 14534 | 361 | 0.065000 | 0.0633 | 0.7754 | 0.0584 | 0.6953 | 0.1078 | 0.2187 | 0.2955 |
| VOCI | target_3h | 14533 | 361 | 0.075000 | 0.0627 | 0.7656 | 0.0613 | 0.5956 | 0.1112 | 0.2171 | 0.2413 |
| VABB | target_1h | 15364 | 301 | 0.065000 | 0.0728 | 0.8463 | 0.0579 | 0.8206 | 0.1081 | 0.2257 | 0.2777 |
| VABB | target_2h | 15363 | 301 | 0.065000 | 0.0590 | 0.8227 | 0.0533 | 0.8106 | 0.1000 | 0.2109 | 0.2982 |
| VABB | target_3h | 15362 | 301 | 0.060000 | 0.0556 | 0.8118 | 0.0493 | 0.8140 | 0.0929 | 0.1983 | 0.3237 |

Stations are **not** ranked. No best/worst location is declared.

## 11. Limitations

- Five aerodromes only; climate regimes differ (coastal vs inland monsoon).
- Location features (lat/lon/elev) of the held-out site were never seen in training; the RF may not interpolate geography.
- NWP overlap from 2021-04-01; complete-case drops CIN/precip holes.
- Held-out evaluation uses the chronological **test** window only, so sample size is smaller than all-years LOSO.
- Alert-rate cap is enforced on **training-station validation**, so held-out alert rate may exceed 20%.
- Not comparable 1:1 to Phase 8D pooled test (that test still saw the station in training).

## 12. Interpretation

LOSO measures **cross-station transfer** of Model B. High pooled 8D skill can coexist with weak transfer if the forest relies on station-specific climate encoded in location/NWP climatology.

These results do **not** claim operational superiority, production readiness, or that a new site can be added without local labels.

## Integrity

- Five stations; held-out absent from train/val: **True**
- Feature count 83; no targets/lightning/satellite/radar: **True**
- Complete-case NWP as Phase 8D: **True**
- Validation max timestamp < held-out test min: **True**
- Protected hashes unchanged: **True**

## Artifacts

| path | role |
| --- | --- |
| `outputs/v2_generalization/phase12c/loso_metrics.csv` | full metrics |
| `outputs/v2_generalization/phase12c/loso_station_summary.csv` | compact station × lead |
| `outputs/v2_generalization/phase12c/loso_config.json` | config |
| `docs/PHASE12C_LOSO_GENERALIZATION_REPORT.md` | this report |

## PHASE STATUS

**READY** — LOSO generalization experiment complete. Stop; do not start Phase 12D.
