# Phase 2A — Multi-location atmospheric predictor collection

**Date:** 2026-09-19  
**Repo:** V2 working copy `C:\College\SIH2_v2`  
**V1 files:** not modified  
**Flask / models / synchronized datasets / thunderstorm labels:** not created or changed  
**Git:** no commit, no push

This phase collects **Open-Meteo historical hourly predictors** for the five Phase 1B training-ready locations. It is **not** a label build and **not** a synchronized training table.

---

## Status

**READY**

All five locations returned a complete 2014-01-01 00:00 UTC → 2025-12-31 23:00 UTC hourly series. No API failure, no source substitution.

---

## What was inspected (V1)

| Item | Role |
|------|------|
| `dataset/fetch_votv_openmeteo_phase3.py` | Year-chunked archive fetch, SDK then REST fallback, UTC hourly validation |
| `dataset/raw_openmeteo/votv_openmeteo_hourly_2014_2025_metadata.json` | V1 request point 8.482° N, 76.920° E; served cell 8.4710° N, 76.9330° E |
| `dataset/build_thunderstorm_labels.py` | IEM station point for VOTV: 8.4667° N, 76.9500° E (labels, not this fetch) |

V2 collection **reuses the same archive endpoint, timezone (GMT/UTC), year chunking, and predictor variables**. It does **not** copy the V1 script five times. One collector parameterizes station identity and coordinates.

---

## Design

Reusable collector: `dataset/multilocation/fetch_openmeteo_multilocation.py`

- One station catalog, one fetch/validate path.
- Raw files stored **per location** under `dataset/multilocation/raw_openmeteo/<icao>/`.
- Year CSV chunks plus a combined hourly CSV and JSON metadata per station.
- Resume-friendly (valid year files are reused).
- If any year fails after retries, the process **stops** (no alternate weather source).

**Not requested:** Open-Meteo `weather_code` (must not be used as a thunderstorm target).

---

## Coordinates (not invented)

Station points from Iowa Environmental Mesonet **IN__ASOS** table:

`https://mesonet.agron.iastate.edu/sites/networks.php?network=IN__ASOS&format=csv&nohtml=on`

Saved as `dataset/multilocation/iem_station_metadata.json`.

| ICAO | Name | IEM lat | IEM lon | Open-Meteo **request** | Request source |
|------|------|---------|---------|------------------------|----------------|
| VOTV | Thiruvananthapuram | 8.4667 | 76.95 | **8.482, 76.920** | V1 Open-Meteo convention |
| VECC | Kolkata | 22.6547 | 88.4467 | 22.6547, 88.4467 | IEM |
| VIDP | Delhi (IGI) | 28.5667 | 77.1167 | 28.5667, 77.1167 | IEM |
| VOCI | Kochi | 10.15 | 76.4 | 10.15, 76.4 | IEM |
| VABB | Mumbai | 19.1005 | 72.8585 | 19.1005, 72.8585 | IEM |

Open-Meteo returns the **archive grid cell** containing the request point. Delivered values belong to the served cell, not the aerodrome marker.

VOTV served cell **8.471001625061035, 76.9329833984375, elev 4 m** matches V1 Phase 3 metadata.

---

## Request

| Item | Value |
|------|--------|
| Endpoint | `https://archive-api.open-meteo.com/v1/archive` |
| Period | 2014-01-01 … 2025-12-31 |
| Frequency | Hourly |
| Timezone | GMT (timestamps stored UTC) |
| Hourly variables | `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`, `wind_direction_10m`, `precipitation`, `cloud_cover` |
| Identity columns | `date`, `station_id`, `latitude`, `longitude`, `source` |
| Chunking | One calendar year per request |

Expected hours per station: **105,192**  
(9 × 8760 + 3 leap years 2016/2020/2024 × 8784)

---

## Row counts (combined files)

| Station | Combined CSV | Rows | Expected | Duplicates | Hourly contiguous | Missing cells | `weather_code` |
|---------|--------------|------|----------|------------|-------------------|---------------|----------------|
| VOTV | `dataset/multilocation/raw_openmeteo/votv/votv_openmeteo_hourly_2014_2025.csv` | **105192** | 105192 | 0 | yes | **0** | absent |
| VECC | `dataset/multilocation/raw_openmeteo/vecc/vecc_openmeteo_hourly_2014_2025.csv` | **105192** | 105192 | 0 | yes | **0** | absent |
| VIDP | `dataset/multilocation/raw_openmeteo/vidp/vidp_openmeteo_hourly_2014_2025.csv` | **105192** | 105192 | 0 | yes | **0** | absent |
| VOCI | `dataset/multilocation/raw_openmeteo/voci/voci_openmeteo_hourly_2014_2025.csv` | **105192** | 105192 | 0 | yes | **0** | absent |
| VABB | `dataset/multilocation/raw_openmeteo/vabb/vabb_openmeteo_hourly_2014_2025.csv` | **105192** | 105192 | 0 | yes | **0** | absent |
| **Total** | | **525960** | 525960 | 0 | | **0** | |

First timestamp (all): `2014-01-01T00:00:00+00:00`  
Last timestamp (all): `2025-12-31T23:00:00+00:00`

Per-year files: 8760 rows (non-leap), 8784 (leap). Twelve year files per station.

---

## Served grid cells

| Station | Served lat | Served lon | Served elevation (m) |
|---------|------------|------------|----------------------|
| VOTV | 8.471001625061035 | 76.9329833984375 | 4.0 |
| VECC | 22.67135238647461 | 88.40956115722656 | 5.0 |
| VIDP | 28.576448440551758 | 77.08428192138672 | 230.0 |
| VOCI | 10.158172607421875 | 76.42105102539062 | 8.0 |
| VABB | 19.08611488342285 | 72.85291290283203 | 16.0 |

Request coordinates are constant in every row of each station file.

---

## Quality checks performed

For each year chunk and each combined file:

- timestamps UTC, sorted, unique
- exact expected hour count
- 1-hour contiguous steps (combined)
- all seven predictor variables present and numeric
- `weather_code` not present
- `station_id` / request lat/lon consistent
- missingness counted (all zeros)

Fetch retry errors recorded in metadata: **empty** for VOTV (and collection completed for all five without abort).

---

## Reproducibility artifacts

| Path | Contents |
|------|----------|
| `dataset/multilocation/iem_station_metadata.json` | IEM IN__ASOS coordinates |
| `dataset/multilocation/fetch_openmeteo_multilocation.py` | Collector |
| `dataset/multilocation/phase2a_collection_summary.json` | Short row-count summary |
| `dataset/multilocation/raw_openmeteo/<icao>/*_metadata.json` | Endpoint, request, served cell, per-year SHA-256, quality |

V1 `dataset/raw_openmeteo/` was **not** overwritten.

---

## Explicitly out of scope

- Thunderstorm labels / METAR synchronization
- Training
- Radar, satellite, lightning, or fake fields
- Secondary stations (VOBG, VOBL, VOHY, VOMM, VOCB, VOML)
- Flask application changes

---

## Next (not this phase)

Join these predictors to genuine METAR thunderstorm targets per station, then consider a pooled table.

**PHASE STATUS: READY**
