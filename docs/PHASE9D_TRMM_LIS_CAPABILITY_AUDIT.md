# Phase 9D — TRMM-LIS lightning capability audit

**Date:** 2026-09-22  
**Audit only.** No training, no Model B edits, no inference-contract changes, no dashboard/Flask edits, no synthetic lightning, no hourly 0-fill, no commit.

## 1. Objective

Determine whether **existing** TRMM-LIS pilot granules contain event/group/flash observations that could support a **legitimate historical lightning feature experiment**. This does **not** claim lightning is usable in the final ThunderWatch model.

## 2. Data inspected

- `dataset/lightning/trmm_lis_pilot/` — **101** NetCDF granules opened
- `dataset/lightning/mendeley_western_ghats/` — Fig_*.mat research files
- `dataset/lightning/Combined_LIS_th.nc` — annual thunder-hour climatology

Stations (decimal degrees):

| station | lat | lon |
|---------|-----|-----|
| VOTV | 8.4667 | 76.95 |
| VECC | 22.6547 | 88.4467 |
| VIDP | 28.5667 | 77.1167 |
| VOCI | 10.15 | 76.4 |
| VABB | 19.1005 | 72.8585 |

## 3. Variable inventory

Discovered from the first granule (names not assumed a priori). Lightning objects:

| Role | Variable |
|------|----------|
| events lat/lon/time | `lightning_event_lat`, `lightning_event_lon`, `lightning_event_TAI93_time` |
| groups lat/lon/time | `lightning_group_lat`, `lightning_group_lon`, `lightning_group_TAI93_time` |
| flashes lat/lon/time | `lightning_flash_lat`, `lightning_flash_lon`, `lightning_flash_TAI93_time` |
| orbit start/end | `orbit_summary_TAI93_start`, `orbit_summary_TAI93_end`, `orbit_summary_UTC_start` |

LIS hierarchy: **event** = illuminated pixel; **group** = adjacent events in one frame; **flash** = groups over successive frames.

**Time conversion:** lightning_*_TAI93_time units: seconds since 1993-01-01 00:00:00 TAI. UTC = TAI93_epoch + seconds - 35.0 s (TAI-UTC leap-second offset applicable 2012-07 through 2015-06). Compared to orbit_summary_UTC_start.

Median residual of converted TAI93 start vs `orbit_summary_UTC_start`: `27.0` seconds (near 0 validates the leap-second offset).

Machine-readable first-granule catalog is in `outputs/lightning/phase9d/phase9d_summary.json` (`variable_inventory_first_granule`).

## 4. Valid lightning record counts

Counts use finite lat/lon in geographic range; fill/missing excluded. Not raw dimension sizes.

- Granules: **101**
- Valid events: **1021914**
- Valid groups: **247144**
- Valid flashes: **24849**

Per-granule CSV: `outputs/lightning/phase9d/trmm_lis_capability_inventory.csv`.

## 5. Geographic coverage

A granule counts as station-proximal **only** if valid lightning coordinates fall inside the radius. The satellite swath is **not** treated as lightning observed.

| station | type | radius_km | granules_checked | granules_with_lightning | total_lightning_records |
|---------|------|-----------|------------------|-------------------------|-------------------------|
| VOTV | event | 25.0 | 101 | 0 | 0 |
| VOTV | group | 25.0 | 101 | 0 | 0 |
| VOTV | flash | 25.0 | 101 | 0 | 0 |
| VOTV | event | 50.0 | 101 | 0 | 0 |
| VOTV | group | 50.0 | 101 | 0 | 0 |
| VOTV | flash | 50.0 | 101 | 0 | 0 |
| VOTV | event | 100.0 | 101 | 0 | 0 |
| VOTV | group | 100.0 | 101 | 0 | 0 |
| VOTV | flash | 100.0 | 101 | 0 | 0 |
| VECC | event | 25.0 | 101 | 0 | 0 |
| VECC | group | 25.0 | 101 | 0 | 0 |
| VECC | flash | 25.0 | 101 | 0 | 0 |
| VECC | event | 50.0 | 101 | 0 | 0 |
| VECC | group | 50.0 | 101 | 0 | 0 |
| VECC | flash | 50.0 | 101 | 0 | 0 |
| VECC | event | 100.0 | 101 | 0 | 0 |
| VECC | group | 100.0 | 101 | 0 | 0 |
| VECC | flash | 100.0 | 101 | 0 | 0 |
| VIDP | event | 25.0 | 101 | 0 | 0 |
| VIDP | group | 25.0 | 101 | 0 | 0 |
| VIDP | flash | 25.0 | 101 | 0 | 0 |
| VIDP | event | 50.0 | 101 | 0 | 0 |
| VIDP | group | 50.0 | 101 | 0 | 0 |
| VIDP | flash | 50.0 | 101 | 0 | 0 |
| VIDP | event | 100.0 | 101 | 0 | 0 |
| VIDP | group | 100.0 | 101 | 0 | 0 |
| VIDP | flash | 100.0 | 101 | 0 | 0 |
| VOCI | event | 25.0 | 101 | 0 | 0 |
| VOCI | group | 25.0 | 101 | 0 | 0 |
| VOCI | flash | 25.0 | 101 | 0 | 0 |
| VOCI | event | 50.0 | 101 | 0 | 0 |
| VOCI | group | 50.0 | 101 | 0 | 0 |
| VOCI | flash | 50.0 | 101 | 0 | 0 |
| VOCI | event | 100.0 | 101 | 0 | 0 |
| VOCI | group | 100.0 | 101 | 0 | 0 |
| VOCI | flash | 100.0 | 101 | 0 | 0 |
| VABB | event | 25.0 | 101 | 0 | 0 |
| VABB | group | 25.0 | 101 | 0 | 0 |
| VABB | flash | 25.0 | 101 | 0 | 0 |
| VABB | event | 50.0 | 101 | 0 | 0 |
| VABB | group | 50.0 | 101 | 0 | 0 |
| VABB | flash | 50.0 | 101 | 0 | 0 |
| VABB | event | 100.0 | 101 | 0 | 0 |
| VABB | group | 100.0 | 101 | 0 | 0 |
| VABB | flash | 100.0 | 101 | 0 | 0 |

## 6. Temporal coverage

- Usable lightning timestamp range (orbit start min … orbit end max): **2013-12-31T22:51:57.800000+00:00** → **2014-01-08T00:13:57+00:00**
- Sum of granule observation durations: **154.76 hours** (not calendar span; orbits are short)
- Coverage is **not continuous**. TRMM LIS is LEO; ~16 orbits/day globally, minutes of view per overpass at a point.

## 7. Station overlap

- Any station-proximal lightning (≤100 km): **False**
- Any ≤25 km: **False**
- Any ≤50 km: **False**
- Any ≤100 km: **False**
- Sample rows written: **0** (`station_lightning_samples.csv`)

**No** valid event/group/flash coordinates fell within 100 km of any ThunderWatch station in this pilot. That is **no lightning detected during available satellite observation** near the stations — **not** a claim that no lightning occurred in those hours.

## 8. Coverage gaps

- Number of granules: 101
- Total observation duration: 154.76 h
- Station-overlap granules: see table in §5 (`granules_with_lightning`)
- Temporal gaps: large; files span ~8 calendar days with discrete orbits, not hourly cadence
- Continuous coverage: **false**
- Hourly aggregation of this pilot would create **severe sampling bias** (overpass hours vs unobserved hours).

Explicit distinction:

- **NO_SATELLITE_COVERAGE / NOT_OBSERVED** — LIS not overhead; must not be stored as 0 lightning.
- **NO_LIGHTNING_DETECTED** — overpass (or lightning arrays) existed, but no valid lightning coordinate in the station radius.

## 9. Hourly aggregation feasibility

Hourly feature extraction is NOT scientifically feasible as a continuous station time series. TRMM-LIS is LEO orbital sampling (~minutes per overpass). Hours without an overpass are NO_SATELLITE_COVERAGE / NOT_OBSERVED, not zero lightning. Broadcasting overpass counts into unobserved hours is forbidden. The pilot span is only ~8 days (late 2013–early 2014), far too short for training.

**Scientifically feasible as continuous hourly features:** `False`

If proximal observations existed, counts such as `lightning_flash_count_1h` would be valid **only** for hours containing an actual overpass. Unobserved hours must remain `NOT_OBSERVED`.

## 10. Mendeley assessment

- Files: **39** in `dataset/lightning/mendeley_western_ghats`
- Classification: **RESEARCH_FIGURE_DATA**
- Event-level lightning: `False`
- Gridded lightning archive: `False`
- Station-aligned observations: `False`
- Timestamps: `False`
- Note: All files are Fig_*.mat research-figure arrays, not an event-level lightning archive. Do not reverse-engineer into fake event records.

Do **not** use these `.mat` files for model training. They are research/figure arrays.

## 11. Combined LIS assessment

- Path: `dataset/lightning/Combined_LIS_th.nc`
- Dimensions: `{'LatitudeDim': 2190, 'LongitudeDim': 7200}`
- Variables: `['lat', 'lon', 'thunder_hours']`
- Time dimension: `False`
- Title: Annual-mean Combined Lightning Imaging Sensor (LIS) 
 Thunder Hours
- Classification: **REFERENCE_ONLY / CLIMATOLOGY_FEATURE**
- Annual-mean thunder hours climatology; no time dimension; must not be broadcast into hourly lightning rows.

## 12. Scientific limitations

- TRMM-LIS is not continuous hourly lightning.
- Absence of station-proximal flashes is not proof that no lightning occurred.
- Do not 0-fill unobserved hours.
- Do not broadcast a granule across hours.
- Do not convert Combined_LIS_th.nc annual climatology into hourly lightning.
- Pilot covers ~2013-12-31 through ~2014-01-07 only.
- TRMM-LIS mission ended 2015; cannot cover 2014–2025 training window.

Atmospheric / METAR window is 2014-01-01 through 2025-12-31. The pilot **does** overlap the **start** of that window (early January 2014) but **does not** overlap the rest of the training period. Overlap of a few days is **not** sufficient for historical model training.

## 13. Final classification

**`REFERENCE_ONLY`**

TRMM-LIS existing pilot sufficient for:

| Use | Sufficient? |
|-----|-------------|
| 1. Hourly feature extraction | `False` |
| 2. Historical model training | `False` |
| 3. Case-study visualization (global pilot flashes) | `True` |
| 3b. Case-study visualization (station-proximal) | `False` |
| 4. Climatological reference | `False` (use Combined_LIS_th.nc instead; still not hourly) |

## 14. Recommendation for ThunderWatch AI

Keep Model B, inference contract, thresholds, feature order, replay, and dashboard unchanged. Do not ingest TRMM-LIS as hourly predictors. Combined LIS thunder hours remain a **static climatology candidate only** (Phase 9C). Operational/real-time lightning still requires institutional sources (e.g. IITM LLN / Damini), not this pilot.

PHASE 9D COMPLETE — TRMM-LIS CAPABILITY AUDIT READY
