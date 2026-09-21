# Phase 7C — Full NWP / Historical Forecast collection (V2)

**Date:** 2026-09-19  
**Repo:** V2 working copy `C:\College\SIH2_v2`  
**V1:** not modified  
**V2 atmospheric / feature / label / synchronized CSVs:** not modified  
**Phase 4 / 5 / 6 models and outputs:** not modified  
**Flask / frontend:** not modified  
**Training:** not run  
**Feature join:** not performed  

Phase 7B (7-day GFS pilot) was **not** repeated. This phase extends the frozen Historical Forecast `gfs_global` configuration to the full useful window.

---

## Objective

Collect hourly Open-Meteo **Historical Forecast** GFS fields for the five approved V2 stations over `2021-04-01 00:00 UTC` → `2025-12-31 23:00 UTC`, without merging into the ML feature table.

---

## Source / configuration (frozen from Phase 7B)

| Item | Value |
|------|--------|
| Product | Open-Meteo Historical Forecast (archived NWP **forecast** valid-time series) |
| Not | ERA5 / Archive API reanalysis; observations; live forecast; Previous Runs |
| Endpoint | `https://historical-forecast-api.open-meteo.com/v1/forecast` |
| Model | `models=gfs_global` (no `best_match`, no silent model swap) |
| Timezone | `timezone=GMT` → stored UTC (`utc_offset_seconds = 0`) |
| Wind unit | `wind_speed_unit=kmh` |
| Chunking | Calendar month; resume-safe chunk CSVs |
| User-Agent | `SIH26072-V2-phase7c-nwp-full` |
| Collector | `dataset/multilocation/fetch_nwp_full_phase7c.py` |
| Output dir | `dataset/multilocation/nwp_full/` |
| Collection UTC | `2026-09-19T18:51:23.994700+00:00` |
| Fill missing | **No** |
| Mix reanalysis | **No** |

---

## Locations

| ICAO | Name | Request lat, lon | GFS served cell | Elev (m) |
|------|------|------------------|-----------------|----------|
| VOTV | Thiruvananthapuram | 8.482, 76.920 | 8.493332, 76.99219 | 4 |
| VECC | Kolkata | 22.6547, 88.4467 | 22.668404, 88.47656 | 5 |
| VIDP | Delhi | 28.5667, 77.1167 | 28.525871, 77.109375 | 230 |
| VOCI | Kochi | 10.15, 76.4 | 10.133423, 76.40625 | 8 |
| VABB | Mumbai | 19.1005, 72.8585 | 19.153923, 72.890625 | 16 |

Served cells match Phase 7B.

---

## Period

| | |
|--|--|
| Requested | 2021-04-01 00:00 UTC → 2025-12-31 23:00 UTC (API `end_date` inclusive) |
| Actual returned | 2021-04-01 00:00:00+00:00 → 2025-12-31 23:00:00+00:00 |
| Exact hourly count | **41,664** per station (spec “≈41,640” is the same window; 1,736 days × 24 h = 41,664) |
| Timestamp gaps vs requested hours | **0** |
| Duplicate station/timestamp | **0** |
| Hourly UTC order | **OK** (monotonic, 3600 s step) |

---

## Variables collected

1. CAPE (`cape`)  
2. Convective inhibition / CIN (`convective_inhibition`)  
3. Lifted index (`lifted_index`)  
4. Temperature 2 m  
5. Relative humidity 2 m  
6. Surface pressure  
7. Wind speed 10 m  
8. Wind direction 10 m  
9. Precipitation  
10. Cloud cover  

**Not collected:** `boundary_layer_height`, `weather_code`, radar, lightning, satellite.

---

## Per-station row counts and missingness

Missing values were **left as null**. No interpolation.

| Station | Rows | Dup | CAPE miss % | CIN miss % (n) | LI miss % | Precip miss % (n) | Other surface miss % |
|---------|------|-----|-------------|----------------|-----------|-------------------|----------------------|
| VOTV | 41664 | 0 | 0.0 | 0.8329 (347) | 0.0 | 0.8281 (345) | 0.0 |
| VECC | 41664 | 0 | 0.0 | 0.8329 (347) | 0.0 | 0.8281 (345) | 0.0 |
| VIDP | 41664 | 0 | 0.0 | 0.8329 (347) | 0.0 | 0.8281 (345) | 0.0 |
| VOCI | 41664 | 0 | 0.0 | 0.8329 (347) | 0.0 | 0.8281 (345) | 0.0 |
| VABB | 41664 | 0 | 0.0 | 0.8329 (347) | 0.0 | 0.8281 (345) | 0.0 |

“Other surface” = T2m, RH2m, surface pressure, wind speed, wind direction, cloud cover.

Pooled file: **208,320** rows (`5 × 41664`).

---

## Continuity and documented gaps

**Timestamp continuity:** complete hourly UTC series at every station. No missing hours, no extras.

**Field-level API nulls (not filled):**

| Field | Window (UTC) | Hours | Notes |
|-------|--------------|-------|--------|
| `precipitation` | 2022-11-29 20:00 → 2022-12-14 04:00 | 345 | Same hours at all five sites |
| `convective_inhibition` | 2023-12-01 01:00 → 2023-12-15 11:00 | 347 | Same hours at all five sites |

These are **vendor nulls inside otherwise complete hourly rows**, not dropped timestamps. CAPE and LI remain populated through both windows.

**API failures / retries:** `api_failures: []` on the successful assemble pass. Chunk cache is month-level so a rerun does not overwrite good months.

---

## Comparison with Phase 7B pilot

| Item | 7B | 7C |
|------|----|----|
| Endpoint | Historical Forecast | same |
| Model | `gfs_global` | same |
| Timezone / wind unit | GMT / kmh | same |
| Coordinates / served cells | five sites | identical |
| Period | 2021-04-01 … 2021-04-07 (168 h) | 2021-04-01 … 2025-12-31 (41664 h) |
| BLH | requested; 100% null | **not requested** |
| weather_code | not requested | not requested |

Pilot week remains a subset of the full files.

---

## Limitations

- Product is **forecast**, not Phase 2A reanalysis. Do not replace atmospheric predictors with these surface fields without an explicit later experiment.
- BLH is still unavailable on this frozen GFS Historical Forecast product.
- Cloud cover occasionally reports −1 / 101 (GFS encoding); values were **not** clipped.
- Two multi-day vendor null windows for CIN and precipitation (table above).
- Not joined to METAR / V2 features.

---

## Validation checklist

1. Five stations present — **pass**  
2. Hourly row count 41,664 vs ≈41,640 spec — **pass** (exact calendar count)  
3. No duplicate station/timestamp — **pass**  
4. UTC hourly ordered — **pass**  
5. CAPE / CIN / LI missingness measured separately — **pass**  
6. Surface missing rates measured — **pass**  
7. No BLH column — **pass**  
8. No weather-code thunderstorm label — **pass**  
9. Phase 4/5/6 artifacts untouched — **pass**  
10. Pilot config matches Phase 7B (minus BLH) — **pass**  
11. Gaps documented, not filled — **pass**  

---

## READY for feature integration?

**YES — READY FOR REVIEW / READY for a later join phase.**

Do **not** integrate in this phase.

---

## Artifacts

**Created / written**

- `dataset/multilocation/fetch_nwp_full_phase7c.py`
- `dataset/multilocation/nwp_full/votv_gfs.csv`
- `dataset/multilocation/nwp_full/vecc_gfs.csv`
- `dataset/multilocation/nwp_full/vidp_gfs.csv`
- `dataset/multilocation/nwp_full/voci_gfs.csv`
- `dataset/multilocation/nwp_full/vabb_gfs.csv`
- `dataset/multilocation/nwp_full/nwp_gfs_pooled.csv`
- `dataset/multilocation/nwp_full/nwp_full_metadata.json`
- `dataset/multilocation/nwp_full/chunks/<icao>/<start>_<end>.csv` (month cache)
- `docs/PHASE7C_FULL_NWP_DATA_REPORT.md`

**Not modified:** V1 data, Phase 2A archive, Phase 4/5/6 models, dashboard, feature CSVs.

---

## PHASE STATUS

**READY**
