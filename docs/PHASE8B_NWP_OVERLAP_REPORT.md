# Phase 8B — NWP overlap dataset (V2)

**Generated:** 2026-09-20T16:09:03.885548+00:00
**Script:** `dataset/multilocation/build_nwp_overlap_phase8b.py`
**Scope:** overlap table only. No training, no imputation, no models.

Phase 3 feature CSVs, Phase 4/5/6 artifacts, and V1 were not modified.

## Objective

Build a new table for `2021-04-01` → `2025-12-31` where Phase 3 features at T, Phase 5 genuine METAR leads, and Phase 7C GFS NWP at the **same** T can be aligned.

## Join

| item | value |
| --- | --- |
| key | `station_id` + UTC timestamp |
| NWP clock | `valid_time_utc` parsed UTC, no shift |
| NWP product | Historical Forecast `gfs_global` (forecast, not observation) |
| Phase 3 columns | all 73, unchanged names/values |
| NWP columns | 10 fields, `nwp_` prefix |
| targets | `target_1h`, `target_2h`, `target_3h` (NA left as NA) |
| not included as predictors | weather_code, BLH, same-hour thunderstorm label |

## Coverage

| metric | value |
| --- | --- |
| total rows | 208320 |
| start UTC | 2021-04-01T00:00:00+00:00 |
| end UTC | 2025-12-31T23:00:00+00:00 |
| duplicate station/timestamp | 0 |
| five stations present | True |
| all NWP fields present | 204860 |
| any NWP missing | 3460 |

| station | rows |
| --- | --- |
| VOTV | 41664 |
| VECC | 41664 |
| VIDP | 41664 |
| VOCI | 41664 |
| VABB | 41664 |

## NWP missingness (no fill)

| column | missing count | missing % |
| --- | --- | --- |
| nwp_cape | 0 | 0.0% |
| nwp_convective_inhibition | 1735 | 0.8329% |
| nwp_lifted_index | 0 | 0.0% |
| nwp_temperature_2m | 0 | 0.0% |
| nwp_relative_humidity_2m | 0 | 0.0% |
| nwp_surface_pressure | 0 | 0.0% |
| nwp_wind_speed_10m | 0 | 0.0% |
| nwp_wind_direction_10m | 0 | 0.0% |
| nwp_precipitation | 1725 | 0.8281% |
| nwp_cloud_cover | 0 | 0.0% |

## Target missingness (not filled with zero)

| target | missing count | missing % |
| --- | --- | --- |
| target_1h | 15567 | 7.4726% |
| target_2h | 15572 | 7.475% |
| target_3h | 15577 | 7.4774% |

## Artifacts

| path | role |
| --- | --- |
| `dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025.csv` | pooled overlap CSV |
| `dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025_metadata.json` | metadata |
| `docs/PHASE8B_NWP_OVERLAP_REPORT.md` | this report |

Existing `dataset/multilocation/features/` was not overwritten.

## PHASE STATUS

**READY** — overlap dataset only. Do not proceed to Phase 8C in this task.
