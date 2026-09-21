# Phase 12A — Multimodal fusion readiness audit (V2)

**Date:** 2026-09-21  
**Scope:** inventory, join-key audit, missingness, modular feature-group design.  
**Training:** not run.  
**Phase 3 / 6 / 8 datasets and models:** not modified.  
**Satellite / radar:** not merged. No placeholder columns in training tables.

---

## Objective

Prepare an extensible V2 feature-fusion structure for:

Atmospheric observations + location/temporal + NWP + lightning climatology

with **optional future layers** (satellite, radar) that can be attached later without rebuilding atmospheric or NWP definitions.

Phase 10 (MOSDAC satellite) and Phase 11 (radar / college letter) remain pending. This phase only audits and designs.

---

## Existing feature sources

| Layer | Phase | Artifact | Role |
|-------|-------|----------|------|
| Atmospheric + location/temporal | 3 | `dataset/multilocation/features/multilocation_features_2014_2025.csv` | Causal 73 features at hour T; 525840 rows |
| Feature catalog | 3 | `dataset/multilocation/features/multilocation_features_2014_2025_metadata.json` | Exact names, groups, formulas |
| Multi-lead targets | 5/6 | joined in Phase 6 training; overlap copy in Phase 8B CSV | `target_1h/2h/3h` from METAR at T+L, not predictors |
| NWP overlap | 8B | `dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025.csv` | Inner join of Phase 3 + GFS Historical Forecast |
| NWP metadata | 8B | `.../multilocation_features_nwp_overlap_2021_2025_metadata.json` | 10 `nwp_*` columns + missingness |
| NWP join audit | 8A | `docs/PHASE8A_NWP_JOIN_AUDIT.md` | `station_id` + UTC timestamp, no time shift |
| Lightning climatology | 9C | `outputs/lightning/combined_lis_thunder_hour_locations.csv` | Static `annual_thunder_hours` per station |

NWP product: Open-Meteo Historical Forecast `gfs_global` (archived forecast valid time, **not** observation, **not** ERA5).  
Lightning: NASA Combined ISS LIS + TRMM LIS **annual thunder hours** (climatology, **not** hourly flashes).

---

## Exact feature inventory

### Keys / identity (not model features)

- `station_id` (ICAO)
- `timestamp_utc` (timezone-aware UTC hourly grid)
- Phase 3 also carries `thunderstorm_target` / `label_status` (labels, **not** features)
- Phase 8 overlap also carries `target_1h`, `target_2h`, `target_3h` (**not** features)

### Atmospheric group (66 columns)

Current state (6): `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`, `precipitation`, `cloud_cover`

Wind vector (4): `wind_direction_sin`, `wind_direction_cos`, `wind_u_10m`, `wind_v_10m`

Lags (30): six variables × {1, 3, 6, 12, 24} h

Tendencies (12): Δ 1h and 3h for T, RH, P, wind, precip, cloud cover

Rolling (14): precip sum 3/6/12/24 h; means 3h and 6h for humidity, pressure, temperature, wind, cloud cover

Source: Open-Meteo archive at **T and earlier** only (Phase 3 causal formulas unchanged).

### Location / temporal group (7 columns)

Cyclic time from **current** `timestamp_utc` (4): `hour_sin`, `hour_cos`, `month_sin`, `month_cos`

Static IEM location (3): `latitude`, `longitude`, `elevation_m`

**Phase 3 numeric feature count: 73** = 66 atmospheric + 7 location/temporal.

### NWP group (10 columns, Phase 8 validated)

Instability: `nwp_cape`, `nwp_convective_inhibition` (CIN), `nwp_lifted_index` (LI)

Surface / cloud: `nwp_temperature_2m`, `nwp_relative_humidity_2m`, `nwp_surface_pressure`, `nwp_wind_speed_10m`, `nwp_wind_direction_10m`, `nwp_precipitation`, `nwp_cloud_cover`

Aligned on `station_id` + `timestamp_utc` = NWP `valid_time_utc`. **`nwp_time_shift`: false** (Phase 8B). No invented NWP variables.

### Lightning climatology group (1 model feature)

From Combined LIS thunder-hour extraction (5 stations, 0 missing):

| station | city | annual_thunder_hours |
|---------|------|----------------------|
| VOTV | Thiruvananthapuram | 332.0780944824219 |
| VECC | Kolkata | 370.5028991699219 |
| VIDP | Delhi | 227.69419860839844 |
| VOCI | Kochi | 385.9404296875 |
| VABB | Mumbai | 129.1292266845703 |

Also present in the CSV (join/audit metadata, not hourly predictors): `station_lat`, `station_lon`, `grid_lat`, `grid_lon`, `distance_km`.

**Join key: `station` only.** Broadcast as a **constant per station**. Not hourly lightning.

### Future groups (not present in data)

- `satellite` — Phase 10 pending MOSDAC
- `radar` — Phase 11 pending access

Do **not** add dummy satellite/radar columns to existing CSVs.

---

## Common-key audit

### Atmospheric ↔ NWP

| Item | Result |
|------|--------|
| Join keys | `station_id` + `timestamp_utc` |
| Phase 3 rows | 525840 |
| Duplicate Phase 3 keys | **0** |
| NWP overlap rows | 208320 |
| Duplicate NWP keys | **0** |
| Inner-join rows | **208320** |
| Feature-only (no NWP) | **317520** |
| NWP-only | **0** |
| Timestamp format | ISO-like UTC; parse `utc=True` → `datetime64[us, UTC]` |
| Feature sample | `2014-01-02 00:00:00+00:00` |
| NWP sample | `2021-04-01 00:00:00+00:00` |
| Feature range | 2014-01-02T00:00:00+00:00 → 2025-12-31T23:00:00+00:00 |
| NWP range | 2021-04-01T00:00:00+00:00 → 2025-12-31T23:00:00+00:00 |
| Stations (all layers) | VABB, VECC, VIDP, VOCI, VOTV |

Unmatched Phase 3 rows are **entirely before NWP start** (2014-01-02 … 2021-03-31). After 2021-04-01 every Phase 3 key has NWP.

### Lightning climatology

Joins by **`station` / `station_id`**, not timestamp. All five ICAOs match Phase 3/8 coverage. One scalar per station.

---

## Missingness audit

No imputation in this phase.

### Atmospheric (Phase 3, n=525840)

Every atmospheric feature: **0 missing (0%)**. After 24 h lookback drop, Open-Meteo fields are complete on the hourly grid.

### Location / temporal (Phase 3)

`hour_sin/cos`, `month_sin/cos`, `latitude`, `longitude`, `elevation_m`: **0 missing**.

### NWP (overlap table, n=208320)

| Column | Missing count | Missing % | Class |
|--------|---------------|-----------|--------|
| `nwp_cape` | 0 | 0 | present |
| `nwp_convective_inhibition` | 1735 | 0.8329 | **genuinely missing** (GFS field holes) |
| `nwp_lifted_index` | 0 | 0 | present |
| `nwp_temperature_2m` | 0 | 0 | present |
| `nwp_relative_humidity_2m` | 0 | 0 | present |
| `nwp_surface_pressure` | 0 | 0 | present |
| `nwp_wind_speed_10m` | 0 | 0 | present |
| `nwp_wind_direction_10m` | 0 | 0 | present |
| `nwp_precipitation` | 1725 | 0.8281 | **genuinely missing** |
| `nwp_cloud_cover` | 0 | 0 | present |

Rows with any NWP missing: 3460 (Phase 8B). Rows with all 10 NWP present: 204860.

**Structurally unavailable:** 317520 Phase 3 hours **before 2021-04-01** have no GFS Historical Forecast in this collection (not “NaN in overlap”; the layer does not exist).

### Lightning climatology (n=5 stations)

`annual_thunder_hours`: **0 missing**.

### Targets (not features)

| Field | n | Missing | % | Class |
|-------|---|---------|---|--------|
| `thunderstorm_target` (Phase 3) | 525840 | 45195 | 8.5948 | unavailable METAR hours; **not filled with 0** |
| `target_1h` (overlap) | 208320 | 15567 | 7.4726 | same |
| `target_2h` | 208320 | 15572 | 7.4750 | same |
| `target_3h` | 208320 | 15577 | 7.4774 | same |

### Satellite / radar

**Not applicable** — no ingested products. Absence is structural (access pending), not row-wise missingness.

---

## Feature-group architecture

```
Feature Groups
├── atmospheric              # Phase 3; frozen formulas
├── location_temporal        # Phase 3 cyclic time + IEM lat/lon/elev
├── nwp                      # Phase 8 `nwp_*` at valid time T
├── lightning_climatology    # Phase 9C station scalar
├── satellite                # [future] Phase 10
└── radar                    # [future] Phase 11
```

**Contract**

1. Each group is a named column list + join spec.
2. Adding satellite/radar **appends groups**; it must not rename or recompute atmospheric/NWP columns.
3. Fusion at train time is **left-join of groups onto the observation key**, not a rewrite of Phase 3 CSVs.
4. Groups may be independently absent (NWP pre-2021; satellite until MOSDAC). Downstream experiments document complete-case vs group-missing policy **without** changing source tables.

---

## Candidate experiment definitions

**Do not train in Phase 12A.**

| Model | Groups | Intended row universe | Notes |
|-------|--------|----------------------|-------|
| **A** | atmospheric + location_temporal | Phase 3 2014–2025 (or NWP-overlap subset for fair A/B/C) | Same 73 columns as Phase 6 |
| **B** | A + nwp | Phase 8 overlap 2021-04-01–2025-12-31 | Same as Phase 8D Model B feature set |
| **C** | B + lightning_climatology | Same overlap as B | `annual_thunder_hours` mapped by station only |
| **D (future)** | C + satellite | TBD after Phase 10 | Same `station_id` + `timestamp_utc` if scientifically valid |
| **E (future)** | D + radar | TBD after Phase 11 | Same |

Fair A vs B vs C comparison should use **identical station+timestamp rows and identical targets**, as Phase 8D did for A vs B. Model A on full 2014–2025 remains the Phase 6 production-style baseline and is a **different sample**, not a nested ablation.

Leads: keep Phase 5/6 `target_1h`, `target_2h`, `target_3h`. Frozen Phase 6 UTC cutoffs. Missing labels excluded, never zero-filled.

---

## Leakage checks

| Risk | Status |
|------|--------|
| Future target used as feature | **Blocked.** Targets are identity/label columns only. |
| Future NWP (T+1… valid times) as features at T | **Blocked.** Phase 8 join uses NWP valid time = observation `timestamp_utc`; `nwp_time_shift=false`. |
| Lightning as hourly nowcast | **Blocked.** Climatology scalar; no time dimension. |
| Future clock features | **Blocked.** Cyclic hour/month from current T only. |
| Phase 3 lag/roll looking forward | **Blocked.** `shift` / trailing rolling only; 24 h lookback drop. |
| `weather_code` / target-derived features | **Forbidden** in Phase 3 catalog. |
| Altering Phase 3 formulas | **Not done.** |

Causal definitions remain those in `multilocation_features_2014_2025_metadata.json`.

---

## Satellite compatibility design (Phase 10, not implemented)

**Required interface when data exist:**

```
station_id, timestamp_utc, <satellite_features...>
```

- Same ICAO set and UTC hourly (or scientifically documented resample to that grid).
- Features at T must use satellite **valid at or before T** (no future scan as a nowcast predictor).
- Join: `station_id` + `timestamp_utc` into group `satellite`.
- Until MOSDAC approval: **no columns, no NaN placeholders in training CSVs.**
- Coverage holes are a **group-missing** policy, not a reason to rewrite atmospheric/NWP.

---

## Radar compatibility design (Phase 11, not implemented)

**Required interface when data exist:**

```
station_id, timestamp_utc, <radar_features...>
```

- Station-centric extract (e.g. reflectivity / echo-top statistics at aerodrome) aligned to UTC hour T or earlier.
- Join: `station_id` + `timestamp_utc` into group `radar`.
- Do not fabricate radar fields. Pending college/access letter = group absent.

---

## Recommended next experiment

**Phase 12B (not this task):** train **Model C** vs frozen **Model B** (and optionally Model A) on the **NWP overlap complete-case (or documented 8E policy)**, adding only `annual_thunder_hours` by `station_id`.

- Same RF / split / F2 threshold protocol as Phase 6/8D.
- Do not wait for satellite/radar.
- Do not rewrite Phase 6/8 files; build an **experiment view** only.

---

## Limitations

- NWP unavailable before 2021-04-01 (~60% of Phase 3 hours).
- NWP is GFS forecast, not in-situ soundings.
- LIS thunder hours mix TRMM (1998–2013) and ISS LIS (2017–2023); not 2014–2025 hourly lightning; no diurnal/seasonal cycle.
- Station climatology can act as a **location proxy** (collinear with lat/lon/city climate); interpret C vs B accordingly.
- Satellite and radar science, resolution, and latency are unknown until Phases 10–11.
- CIN and NWP precip have small genuine missingness (~0.83%).

---

## Artifacts

| Path | Role |
|------|------|
| `docs/PHASE12A_MULTIMODAL_FUSION_READINESS.md` | this report |
| `outputs/v2_multimodal/phase12a_feature_inventory.json` | machine-readable inventory |

---

## PHASE STATUS

**READY** — fusion architecture audited and designed. No models trained. No Phase 3/6/8 mutation. No satellite/radar merge.
