# Phase 12B — Lightning climatology fusion experiment

**Generated:** 2026-09-21T04:58:12.060303+00:00
**Script:** `ml/train_v2_lightning_climatology_phase12b.py`
**Scope:** experiment only. Not production. No Phase 3/6/8 mutation. No frozen model overwrite.

## Objective

Compare Model B (atmospheric + location/temporal + NWP) to Model C (B + `annual_thunder_hours`) on identical complete-case NWP-overlap rows, frozen Phase 8D RF/split/threshold protocol.

## Data sources

| source | path |
| --- | --- |
| NWP overlap | `dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025.csv` |
| Lightning | `outputs/lightning/combined_lis_thunder_hour_locations.csv` (`station`, `annual_thunder_hours` only) |

Complete-case: all 10 NWP fields present. Missing METAR targets excluded, not zero-filled.

## Lightning join validation

- Join: `station_id` = `station` (not timestamp).
- Row count after left join: unchanged vs overlap table.
- `annual_thunder_hours` missing after join: **0**.
- Unique values per station: **1** (static climatology).
- Not hourly lightning; not nowcast observations.

## Identical-row verification

For each lead, Model B and Model C used the same `station_id` + `timestamp_utc` rows, same targets, same train/validation/test membership.

| lead | train n (pos) | val n (pos) | test n (pos) | B vs C rows identical |
| --- | --- | --- | --- | --- |
| target_1h | 46501 (1721) | 69273 (1902) | 73694 (2172) | True |
| target_2h | 46500 (1721) | 69275 (1902) | 73689 (2172) | True |
| target_3h | 46499 (1721) | 69277 (1902) | 73684 (2172) | True |

## Model B configuration

- Features: 83 (Phase 3 73 + 10 `nwp_*`).
- RF: n_estimators=200, gini, max_features=sqrt, class_weight=balanced, random_state=42.
- Threshold: max validation F2, alert rate ≤ 0.20; frozen for test.

## Model C configuration

- Features: Model B + `annual_thunder_hours` only (84 columns).
- Same RF, split, complete-case, threshold protocol.

## Split verification

- Train if timestamp ≤ `2022-05-27T01:00:00+00:00`.
- Validation if ≤ `2024-03-14T11:00:00+00:00`.
- Else test. No shuffle. Train max < val min; val max < test min.

## target_1h results (test)

| model | threshold | PR-AUC | ROC-AUC | P | R | F1 | F2 | alert rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B atmo+NWP | 0.065000 | 0.1445 | 0.8510 | 0.1100 | 0.6340 | 0.1875 | 0.3247 | 0.1698 |
| C + thunder hours | 0.065000 | 0.1471 | 0.8475 | 0.1124 | 0.6381 | 0.1912 | 0.3297 | 0.1673 |

## target_2h results (test)

| model | threshold | PR-AUC | ROC-AUC | P | R | F1 | F2 | alert rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B atmo+NWP | 0.065000 | 0.1368 | 0.8443 | 0.1074 | 0.6229 | 0.1833 | 0.3179 | 0.1709 |
| C + thunder hours | 0.065000 | 0.1387 | 0.8431 | 0.1070 | 0.6220 | 0.1826 | 0.3169 | 0.1714 |

## target_3h results (test)

| model | threshold | PR-AUC | ROC-AUC | P | R | F1 | F2 | alert rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B atmo+NWP | 0.065000 | 0.1317 | 0.8372 | 0.1029 | 0.6077 | 0.1761 | 0.3068 | 0.1740 |
| C + thunder hours | 0.070000 | 0.1326 | 0.8380 | 0.1083 | 0.5856 | 0.1828 | 0.3112 | 0.1594 |

## Station-level Model C results (test, same threshold as pooled C)

| lead | station | n | pos | PR-AUC | ROC-AUC | P | R | F1 | F2 | alert rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target_1h | VOTV | 12899 | 509 | 0.2370 | 0.8661 | 0.1493 | 0.7485 | 0.2489 | 0.4152 | 0.1978 |
| target_1h | VECC | 15627 | 637 | 0.1267 | 0.8238 | 0.1111 | 0.7221 | 0.1925 | 0.3438 | 0.2650 |
| target_1h | VIDP | 15269 | 364 | 0.0721 | 0.8126 | 0.0884 | 0.3324 | 0.1396 | 0.2142 | 0.0897 |
| target_1h | VOCI | 14535 | 361 | 0.1500 | 0.8162 | 0.0991 | 0.5651 | 0.1687 | 0.2913 | 0.1416 |
| target_1h | VABB | 15364 | 301 | 0.1226 | 0.8994 | 0.0996 | 0.7309 | 0.1753 | 0.3223 | 0.1438 |
| target_2h | VOTV | 12898 | 509 | 0.2316 | 0.8708 | 0.1457 | 0.7466 | 0.2438 | 0.4091 | 0.2022 |
| target_2h | VECC | 15626 | 637 | 0.1210 | 0.8127 | 0.1057 | 0.7221 | 0.1844 | 0.3334 | 0.2784 |
| target_2h | VIDP | 15268 | 364 | 0.0666 | 0.8026 | 0.0701 | 0.2802 | 0.1121 | 0.1751 | 0.0954 |
| target_2h | VOCI | 14534 | 361 | 0.1278 | 0.8185 | 0.1018 | 0.5596 | 0.1722 | 0.2945 | 0.1366 |
| target_2h | VABB | 15363 | 301 | 0.1083 | 0.8933 | 0.0929 | 0.6877 | 0.1636 | 0.3015 | 0.1451 |
| target_3h | VOTV | 12897 | 509 | 0.2076 | 0.8599 | 0.1463 | 0.6680 | 0.2400 | 0.3899 | 0.1802 |
| target_3h | VECC | 15625 | 637 | 0.1255 | 0.8134 | 0.1068 | 0.7049 | 0.1855 | 0.3324 | 0.2691 |
| target_3h | VIDP | 15267 | 364 | 0.0625 | 0.7913 | 0.0700 | 0.2418 | 0.1085 | 0.1621 | 0.0824 |
| target_3h | VOCI | 14533 | 361 | 0.1183 | 0.8162 | 0.1011 | 0.5402 | 0.1704 | 0.2891 | 0.1327 |
| target_3h | VABB | 15362 | 301 | 0.1086 | 0.8911 | 0.0984 | 0.6645 | 0.1715 | 0.3090 | 0.1323 |

Stations are not ranked; no “best location” is declared.

## Metric deltas C − B (test)

| lead | Δ PR-AUC | Δ ROC-AUC | Δ P | Δ R | Δ F1 | Δ F2 | Δ alert rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| target_1h | +0.0026 | -0.0035 | +0.0024 | +0.0041 | +0.0036 | +0.0050 | -0.0025 |
| target_2h | +0.0020 | -0.0011 | -0.0005 | -0.0009 | -0.0007 | -0.0010 | +0.0005 |
| target_3h | +0.0009 | +0.0008 | +0.0053 | -0.0221 | +0.0067 | +0.0044 | -0.0146 |

## Proxy / interpretation caveat

`annual_thunder_hours` is **constant within each station**. Any Model C change vs B can only come from a **station-level climate/location offset**, not from time-varying lightning. Location is already in Model B (`latitude`, `longitude`, `elevation_m`). LIS thunder hours are **not** observed lightning at prediction time and must not be described as nowcast lightning.

## Scientific limitations

- Combined LIS climatology mixes TRMM (1998–2013) and ISS LIS (2017–2023); not 2014–2025 hourly flashes.
- Complete-case NWP overlap only (from 2021-04-01); CIN/precip holes dropped as in Phase 8D.
- RF can use a constant-per-station column as a surrogate station ID.
- Thresholds chosen on validation only; test unused for tuning.

## Decision for future multimodal fusion

**Classification: `REFERENCE_ONLY`**

Deltas vs B are negligible or mixed relative to NWP already encoding location climate. Do not treat as operational lightning. Keep as a documented reference feature, not a default fusion layer.

Mean test Δ PR-AUC (C−B) = +0.0018; mean Δ ROC-AUC = -0.0013; mean Δ F2 = +0.0028.

Satellite and radar remain future groups (Phases 10–11). This experiment does not add them.

## Artifacts

| path | role |
| --- | --- |
| `outputs/v2_multimodal/phase12b/model_b_vs_c_metrics.csv` | pooled B vs C + deltas |
| `outputs/v2_multimodal/phase12b/model_c_station_metrics.csv` | Model C by station |
| `outputs/v2_multimodal/phase12b/experiment_config.json` | config + sanity |
| `docs/PHASE12B_LIGHTNING_CLIMATOLOGY_EXPERIMENT.md` | this report |

## PHASE STATUS

**READY** — classification `REFERENCE_ONLY`. Stop; do not proceed to Phase 12C in this task.
