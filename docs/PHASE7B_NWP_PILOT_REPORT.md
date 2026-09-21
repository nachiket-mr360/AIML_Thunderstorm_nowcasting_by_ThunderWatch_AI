# Phase 7B — NWP Historical Forecast PILOT (V2)

**Date:** 2026-09-19  
**Repo:** V2 working copy `C:\College\SIH2_v2`  
**V1:** not modified  
**V2 feature / label / synchronized CSVs:** not modified  
**Flask / frontend:** not modified  
**Git:** no commit, no push  
**CDS:** not used; no account, no credentials

This is a **seven-day point extract only**. It is **not** a full 2021–2025 download, **not** a feature join, and **not** training.

---

## PHASE STATUS

**READY**

All five locations returned a complete hourly series for 2021-04-01 00:00 UTC → 2021-04-07 23:00 UTC. CAPE, CIN, lifted index, and all requested surface fields are populated. **`boundary_layer_height` is systematically null** on frozen `gfs_global` for this window and **must not be used** from this product/model. Overlapping surface fields were compared to Phase 2A archive data for sanity only; Phase 2A files were not replaced.

---

## Exact API / model configuration

| Item | Value |
|------|--------|
| Product | Open-Meteo **Historical Forecast** (archived NWP **forecast** valid-time series) |
| Not | Reanalysis (ERA5 / Archive API). Not observations. Not live forecast. |
| Endpoint | `https://historical-forecast-api.open-meteo.com/v1/forecast` |
| Access | Unauthenticated HTTPS GET; `User-Agent: SIH26072-V2-phase7b-pilot` |
| Frozen model | `models=gfs_global` (Phase 7: do not use silent `best_match`) |
| Timezone | `timezone=GMT` → timestamps stored **UTC** (`utc_offset_seconds = 0`) |
| Wind unit | `wind_speed_unit=kmh` (aligned with Phase 2A Open-Meteo km/h) |
| Period | **2021-04-01 … 2021-04-07** (API `end_date` inclusive) |
| Frequency | Hourly |
| Expected rows / site | 7 × 24 = **168** |
| Collector | `dataset/multilocation/fetch_nwp_pilot_phase7b.py` |
| Output dir | `dataset/multilocation/nwp_pilot/` |

Previous Runs API was **not** pulled in this pilot (same vendor family; Historical Forecast is the “valid at T” series). CDS / ERA5 was **not** used.

Example request (VOTV):

```
https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=8.482&longitude=76.92&start_date=2021-04-01&end_date=2021-04-07&hourly=cape,convective_inhibition,lifted_index,boundary_layer_height,temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover&timezone=GMT&models=gfs_global&wind_speed_unit=kmh
```

### Requested hourly variables

| Request | API name | Units returned |
|---------|----------|----------------|
| CAPE | `cape` | J/kg |
| Convective inhibition / CIN | `convective_inhibition` | J/kg |
| Lifted index | `lifted_index` | (unit string empty; values present) |
| Boundary layer height | `boundary_layer_height` | m (field present, **values all null**) |
| 2 m temperature | `temperature_2m` | °C |
| 2 m RH | `relative_humidity_2m` | % |
| Surface pressure | `surface_pressure` | hPa |
| 10 m wind speed | `wind_speed_10m` | km/h |
| 10 m wind direction | `wind_direction_10m` | ° |
| Precipitation | `precipitation` | mm |
| Cloud cover | `cloud_cover` | % |

Open-Meteo `weather_code` was **not** requested.

### Coordinates (same as Phase 2A)

| ICAO | Request lat, lon | GFS served cell | Elev (m) |
|------|------------------|-----------------|----------|
| VOTV | 8.482, 76.920 | 8.493332, 76.99219 | 4 |
| VECC | 22.6547, 88.4467 | 22.668404, 88.47656 | 5 |
| VIDP | 28.5667, 77.1167 | 28.525871, 77.109375 | 230 |
| VOCI | 10.15, 76.4 | 10.133423, 76.40625 | 8 |
| VABB | 19.1005, 72.8585 | 19.153923, 72.890625 | 16 |

Served cells differ from Phase 2A **archive** cells (ERA5-based grid). That is expected.

---

## Location summary table

Location | Rows | CAPE missing % | CIN missing % | LI missing % | BLH missing % | Other-variable missing %
---------|------|----------------|---------------|--------------|---------------|--------------------------
VOTV | 168 | 0.0 | 0.0 | 0.0 | **100.0** | 0.0
VECC | 168 | 0.0 | 0.0 | 0.0 | **100.0** | 0.0
VIDP | 168 | 0.0 | 0.0 | 0.0 | **100.0** | 0.0
VOCI | 168 | 0.0 | 0.0 | 0.0 | **100.0** | 0.0
VABB | 168 | 0.0 | 0.0 | 0.0 | **100.0** | 0.0

“Other-variable missing %” = mean missingness of `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`, `wind_direction_10m`, `precipitation`, `cloud_cover`.

All five locations returned data. Duplicate timestamps: **0**. Hourly UTC step: **OK**. Range: 2021-04-01 00:00 UTC … 2021-04-07 23:00 UTC.

---

## Min / max / mean (pilot week)

### VOTV

| Variable | min | max | mean |
|----------|-----|-----|------|
| cape | 50 | 3000 | 851.3 |
| convective_inhibition | −321 | 0 | −82.9 |
| lifted_index | −6.7 | 1.8 | −2.14 |
| boundary_layer_height | — | — | — |
| temperature_2m | 23.5 | 35.0 | 28.27 |
| relative_humidity_2m | 38 | 96 | 71.39 |
| surface_pressure | 1004.9 | 1012.4 | 1008.39 |
| wind_speed_10m | 0.7 | 23.8 | 12.01 |
| wind_direction_10m | 6 | 349 | 266.1 |
| precipitation | 0.0 | 0.7 | 0.039 |
| cloud_cover | 0 | 100 | 47.39 |

### VECC

| Variable | min | max | mean |
|----------|-----|-----|------|
| cape | 0 | 2510 | 744.6 |
| convective_inhibition | −384 | 0 | −112.3 |
| lifted_index | −11.3 | 6.9 | −2.76 |
| temperature_2m | 23.6 | 41.4 | 30.58 |
| relative_humidity_2m | 6 | 98 | 50.48 |
| surface_pressure | 997.3 | 1011.7 | 1006.24 |
| wind_speed_10m | 5.7 | 31.4 | 16.17 |
| precipitation | 0.0 | 0.6 | 0.004 |
| cloud_cover | 0 | 100 | 9.55 |

### VIDP

| Variable | min | max | mean |
|----------|-----|-----|------|
| cape | 0 | 210 | 4.35 |
| convective_inhibition | −350 | 0 | −7.09 |
| lifted_index | −0.7 | 18.2 | 7.40 |
| temperature_2m | 22.1 | 39.4 | 30.95 |
| relative_humidity_2m | 3 | 28 | 8.15 |
| surface_pressure | 973.5 | 984.9 | 980.16 |
| wind_speed_10m | 1.4 | 29.2 | 10.53 |
| precipitation | 0.0 | 0.0 | 0.0 |
| cloud_cover | 0 | 100 | 27.07 |

VIDP in this dry pre-monsoon week is **stable / low CAPE / high LI / very dry** in GFS — physically plausible for Delhi early April, not a fetch error.

### VOCI

| Variable | min | max | mean |
|----------|-----|-----|------|
| cape | 10 | 3000 | 867.3 |
| convective_inhibition | −289 | 0 | −75.9 |
| lifted_index | −6.2 | 2.3 | −1.63 |
| temperature_2m | 22.9 | 37.0 | 28.44 |
| relative_humidity_2m | 37 | 99 | 71.02 |
| surface_pressure | 1003.5 | 1011.8 | 1008.01 |
| wind_speed_10m | 0.5 | 21.3 | 7.20 |
| precipitation | 0.0 | 1.5 | 0.072 |
| cloud_cover | 0 | 100 | 55.08 |

### VABB

| Variable | min | max | mean |
|----------|-----|-----|------|
| cape | 0 | 1230 | 201.1 |
| convective_inhibition | −452 | 0 | −220.8 |
| lifted_index | −4.2 | 5.4 | −0.24 |
| temperature_2m | 26.4 | 35.7 | 29.80 |
| relative_humidity_2m | 24 | 74 | 51.89 |
| surface_pressure | 1003.4 | 1010.7 | 1007.15 |
| wind_speed_10m | 1.5 | 29.1 | 12.51 |
| precipitation | 0.0 | 0.0 | 0.0 |
| cloud_cover | 0 | 100 | 14.83 |

---

## Variables that cannot be reliably used (this config)

| Variable | Verdict |
|----------|---------|
| `boundary_layer_height` | **Do not use.** 100% missing at all five sites on `gfs_global` Historical Forecast for this week. Units advertised as `m`; values null. Phase 2A **archive** BLH is a different product (reanalysis) and was **not** substituted. |
| `lifted_index` | **Usable numerically** (0% missing). Unit string empty; treat as dimensionless K (standard LI). |
| CAPE / CIN | **Usable** in this window (0% missing). Phase 7 already noted convective fields fill from ~2021-04; this week is inside that window. |
| Surface seven | **Usable** (0% missing). They are **forecast**, not Phase 2A reanalysis. |

---

## Phase 2A sanity compare (do not replace Phase 2A)

Join: inner merge on UTC hour against `raw_openmeteo/<icao>/<icao>_openmeteo_2021.csv` for the same 168 hours. **168/168 overlap** at every site.

Mean absolute difference (GFS historical forecast vs Phase 2A archive):

| Site | T2m MAD °C | RH MAD % | Psurf MAD hPa | Wind spd MAD km/h | Precip MAD mm | Cloud MAD % |
|------|------------|----------|---------------|-------------------|---------------|-------------|
| VOTV | 1.51 | 8.07 | 0.34 | 3.07 | 0.11 | 37.6 |
| VECC | 1.65 | 16.74 | 0.62 | 6.80 | 0.012 | 19.4 |
| VIDP | 4.31 | 17.31 | 0.96 | 3.31 | 0.00 | 13.5 |
| VOCI | 1.40 | 11.41 | 0.56 | 2.35 | 0.12 | 38.3 |
| VABB | 2.28 | 22.46 | 0.25 | 2.92 | 0.001 | 17.3 |

Temperature and surface pressure **correlate strongly** (T corr ~0.86–0.97; pressure corr ~0.93–0.99). Cloud and precipitation **do not** match closely (different model, different grid). Wind direction MAD is inflated by 0/360 wrap. This is **expected disagreement**, not a reason to overwrite Phase 2A.

VIDP / VABB precipitation corr is undefined because both series are (near) all zeros that week.

---

## Artifacts written (new directory only)

- `dataset/multilocation/nwp_pilot/votv_nwp_pilot_20210401_20210407.csv`
- `dataset/multilocation/nwp_pilot/vecc_nwp_pilot_20210401_20210407.csv`
- `dataset/multilocation/nwp_pilot/vidp_nwp_pilot_20210401_20210407.csv`
- `dataset/multilocation/nwp_pilot/voci_nwp_pilot_20210401_20210407.csv`
- `dataset/multilocation/nwp_pilot/vabb_nwp_pilot_20210401_20210407.csv`
- `dataset/multilocation/nwp_pilot/nwp_pilot_all_locations_20210401_20210407.csv`
- `dataset/multilocation/nwp_pilot/nwp_pilot_metadata.json`
- `dataset/multilocation/fetch_nwp_pilot_phase7b.py`

---

## What this phase did not do

- No full 2021–2025 download  
- No merge into V2 features / targets  
- No training  
- No V1, Flask, or existing CSV edits  
- No CDS account / credentials  
- No git commit / push  

---

## Implication for a later NWP experiment

A frozen **`gfs_global` Historical Forecast** extract can supply **CAPE, CIN, LI, and surface fields** at the five V2 points from at least this 2021-04 week, without an account. **BLH must come from another frozen model or from the existing Archive API (reanalysis)**, and that choice must be documented — not silently mixed. Do not treat GFS surface fields as replacements for Phase 2A.

---

## PHASE STATUS (repeat)

**READY**
