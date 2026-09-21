# Phase 8A — NWP join audit (V2)

**Generated:** 2026-09-20T16:04:24.669223+00:00
**Script:** `dataset/multilocation/audit_nwp_join_phase8a.py`
**Scope:** join audit only. No training, no enhanced ML CSV, no models.

Phase 3 feature files, Phase 4/5/6 artifacts, and V1 were not modified.

## Objective

Measure coverage of joining existing V2 Phase 3 feature rows to Phase 7C GFS Historical Forecast NWP using `station_id + UTC timestamp` only.

## Data sources

| source | path |
| --- | --- |
| Features | `dataset/multilocation/features/multilocation_features_2014_2025.csv` |
| NWP | `dataset/multilocation/nwp_full/nwp_gfs_pooled.csv` |

NWP product: Open-Meteo Historical Forecast, `gfs_global`. This is archived forecast-model output, not observations and not ERA5 reanalysis.

## Join methodology

- Key: `station_id` + parsed UTC timestamp.
- Feature clock column: `timestamp_utc`.
- NWP clock column: `valid_time_utc` (renamed to `timestamp_utc` after UTC parse).
- No timestamp-only join.
- No NWP time shift.
- No imputation.
- Inner join used only in memory for counts; no joined feature table was written.

## Timestamp timezone

| side | raw sample | dtype after `utc=True` parse |
| --- | --- | --- |
| features | `2014-01-02 00:00:00+00:00` | `datetime64[us, UTC]` |
| NWP | `2021-04-01 00:00:00+00:00` | `datetime64[us, UTC]` |

Both sides parse to timezone-aware UTC (`datetime64[us, UTC]`).

## Station consistency

- Expected: VOTV, VECC, VIDP, VOCI, VABB
- Features: VABB, VECC, VIDP, VOCI, VOTV
- NWP: VABB, VECC, VIDP, VOCI, VOTV
- Inner join covers all five stations: **True**

## Coverage summary

| metric | count |
| --- | --- |
| feature rows | 525840 |
| unique feature station/timestamp keys | 525840 |
| duplicate feature keys | 0 |
| NWP rows | 208320 |
| unique NWP station/timestamp keys | 208320 |
| duplicate NWP keys | 0 |
| exact inner-join rows | 208320 |
| left-join retention (inner / feature rows) | 0.3962 (39.62%) |
| unmatched feature rows | 317520 |
| unmatched NWP rows | 0 |

Feature time range: `2014-01-02T00:00:00+00:00` → `2025-12-31T23:00:00+00:00`  
NWP time range: `2021-04-01T00:00:00+00:00` → `2025-12-31T23:00:00+00:00`

### Per station

| station | feature rows | NWP rows | inner join | unmatched features | unmatched NWP |
| --- | --- | --- | --- | --- | --- |
| VOTV | 105168 | 41664 | 41664 | 63504 | 0 |
| VECC | 105168 | 41664 | 41664 | 63504 | 0 |
| VIDP | 105168 | 41664 | 41664 | 63504 | 0 |
| VOCI | 105168 | 41664 | 41664 | 63504 | 0 |
| VABB | 105168 | 41664 | 41664 | 63504 | 0 |

### Unmatched feature rows

- Time span of unmatched features: `2014-01-02T00:00:00+00:00` → `2021-03-31T23:00:00+00:00`
- Unmatched features before NWP start (2021-04-01T00:00:00+00:00): **317520**
- Unmatched features on/after NWP start: **0**

Unmatched NWP rows are GFS hours with no Phase 3 feature row at the same station/timestamp (feature table starts after lag warmup and may drop hours).

## NWP missingness (full NWP table)

| variable | missing count | missing % |
| --- | --- | --- |
| cape | 0 | 0.0% |
| convective_inhibition | 1735 | 0.8329% |
| lifted_index | 0 | 0.0% |
| temperature_2m | 0 | 0.0% |
| relative_humidity_2m | 0 | 0.0% |
| surface_pressure | 0 | 0.0% |
| wind_speed_10m | 0 | 0.0% |
| wind_direction_10m | 0 | 0.0% |
| precipitation | 1725 | 0.8281% |
| cloud_cover | 0 | 0.0% |

## NWP missingness (inner-join keys only)

Inner-join rows: 208320. Rows with any selected NWP field missing: **3460**. Rows with all selected NWP fields present: **204860**.

| variable | missing count | missing % of inner join |
| --- | --- | --- |
| cape | 0 | 0.0% |
| convective_inhibition | 1735 | 0.8329% |
| lifted_index | 0 | 0.0% |
| temperature_2m | 0 | 0.0% |
| relative_humidity_2m | 0 | 0.0% |
| surface_pressure | 0 | 0.0% |
| wind_speed_10m | 0 | 0.0% |
| wind_direction_10m | 0 | 0.0% |
| precipitation | 1725 | 0.8281% |
| cloud_cover | 0 | 0.0% |

### Missing counts by station (inner join)

| station | rows | cape | CIN | LI | T2m | RH | Psurf | wspd | wdir | precip | cloud |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | 41664 | 0 | 347 | 0 | 0 | 0 | 0 | 0 | 0 | 345 | 0 |
| VECC | 41664 | 0 | 347 | 0 | 0 | 0 | 0 | 0 | 0 | 345 | 0 |
| VIDP | 41664 | 0 | 347 | 0 | 0 | 0 | 0 | 0 | 0 | 345 | 0 |
| VOCI | 41664 | 0 | 347 | 0 | 0 | 0 | 0 | 0 | 0 | 345 | 0 |
| VABB | 41664 | 0 | 347 | 0 | 0 | 0 | 0 | 0 | 0 | 345 | 0 |

## Known Phase 7C null windows

| window | field | UTC start | UTC end | NWP rows in window | nulls in window | feature rows in window | inner-join rows in window |
| --- | --- | --- | --- | --- | --- | --- | --- |
| precipitation | precipitation | 2022-11-29T20:00:00+00:00 | 2022-12-14T04:00:00+00:00 | 1725 | 1725 | 1725 | 1725 |
| CIN | convective_inhibition | 2023-12-01T01:00:00+00:00 | 2023-12-15T11:00:00+00:00 | 1735 | 1735 | 1735 | 1735 |

These are vendor nulls inside otherwise complete hourly NWP rows. They were not filled.

## Integrity notes

- Join key is station + UTC timestamp, not timestamp alone.
- Duplicate station/timestamp keys: features **0**, NWP **0**.
- No training, imputation, or feature-importance computation in this phase.
- No huge joined CSV written.
- Phase 3/4/5/6 and V1 not modified.
- NWP is Historical Forecast GFS, not labeled as observation.

## Artifacts

| path | role |
| --- | --- |
| `dataset/multilocation/audit_nwp_join_phase8a.py` | audit script |
| `dataset/multilocation/nwp_full/phase8a_join_audit.json` | machine-readable counts |
| `docs/PHASE8A_NWP_JOIN_AUDIT.md` | this report |

## PHASE STATUS

**READY** — join audit only. Do not proceed to Phase 8B in this task.
