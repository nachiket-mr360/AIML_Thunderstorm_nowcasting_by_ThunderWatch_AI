# Phase 9C — Combined ISS + TRMM LIS annual thunder-hour audit

**Date:** 2026-09-21  
**File:** `dataset/lightning/Combined_LIS_th.nc`  
**No training, no model edits, no Phase 6/8 dataset edits, no hourly conversion, no merge into V2 features**

Station coordinates: `dataset/multilocation/iem_station_metadata.json` (IEM IN__ASOS).

---

## Dataset identity

NASA **Combined ISS LIS + TRMM LIS annual thunder-hour** climatology. This is a **historical annual** lightning/thunderstorm-frequency grid, **not** hourly flashes, **not** real-time lightning, and **not** a nowcast observation stream.

Global attributes (truncated):

```json
{
  "Title": "Annual-mean Combined Lightning Imaging Sensor (LIS) \n Thunder Hours",
  "Description": "Two LIS optical imagers detected lightning by monitoring \n emissions at 777.4 nm.  Because they operated in low-Earth \n orbit, thunder hours are calculated for each sensor by \n scaling the fraction of overpasses during which lightning \n was observed within 15 km of a grid point.  The TRMM and ISS \n thunder hour datasets are then combined by scaling \n according to the number of days each sensor operated.",
  "Institution": "NASA Marshall Space Flight Center",
  "Source": "Optical imager",
  "History": "Created: 2025-08-08 16:36:37",
  "Conventions": "CF-1.13",
  "References": "[link to readme document and (eventually) paper]",
  "Comment": "Thunder hours are based on the full TRMM LIS operational \n period of 1 January 1998 to 8 April 2013 and the full ISS \n LIS operational period of 1 March 2017 to 16 November \n 2023 and were produced by Katrina Virts (katrina.virts@uah.edu)."
}
```

---

## File inspected

| Item | Value |
|------|--------|
| Path | `dataset/lightning/Combined_LIS_th.nc` |
| Size | 63132967 bytes |
| Opened with | netCDF4 |

---

## Dataset structure

**Dimensions:** `{'LatitudeDim': 2190, 'LongitudeDim': 7200}`

**Latitude range:** -54.724998474121094 … 54.724998474121094 (lat)  
**Longitude range:** -179.97500610351562 … 179.97500610351562 (lon)  
**Grid resolution (median spacing):** Δlat ≈ 0.049999237060546875°, Δlon ≈ 0.05000114440917969°

**Variables:**

| Name | Dims | Shape | Dtype | long_name | units |
|------|------|-------|-------|-----------|-------|
| `lat` | ('LatitudeDim',) | (2190,) | `float32` |  | degree_north |
| `lon` | ('LongitudeDim',) | (7200,) | `float32` |  | degree_east |
| `thunder_hours` | ('LongitudeDim', 'LatitudeDim') | (7200, 2190) | `float32` |  | hour |

**Coordinate attributes**

Latitude `lat`:
- `Long_name`: Grid point latitude
- `Units`: degree_north
- `Description`: Latitude of grid point
- `Standard_name`: latitude

Longitude `lon`:
- `Long_name`: Grid point longitude
- `Units`: degree_east
- `Description`: Longitude of grid point
- `Standard_name`: longitude

Data `thunder_hours`:
- `Long_name`: Annual-mean thunder hours
- `Units`: hour
- `Description`: Average number of hours per year that thunder 
 would have been heard at a grid point

---

## Variable used

Extracted field: **`thunder_hours`** (squeezed shape `(7200, 2190)`).

This is the 2-D annual thunder-hour field. Values are **climatological hours per year** (see units in the table), not lightning counts in a clock hour.

Fill / missing: `_FillValue` / `missing_value` = `None`; finite cells = **15768000** / 15768000.

---

## Grid resolution

Approximate cell size **0.049999237060546875° × 0.05000114440917969°** from unique 1-D coordinate spacing. Nearest **valid** (finite) cell was used so land/ocean missing masks are respected.

---

## Temporal coverage

This product is an **annual climatology** (one 2-D map; no time dimension). File `Comment`: thunder hours from **TRMM LIS 1998-01-01 → 2013-04-08** and **ISS LIS 2017-03-01 → 2023-11-16**, combined by days of operation. That is **not** 2014–2025 hourly data, and it includes a **2013–2017 gap** between sensors.

No time dimension was expanded into years or hours in this audit.

---

## Five-location extraction table

Nearest **valid** grid cell; haversine distance; value copied from `thunder_hours[i,j]` only.

| station | city | station_lat | station_lon | grid_lat | grid_lon | distance_km | annual_thunder_hours | missing? |
|---------|------|-------------|-------------|----------|----------|-------------|----------------------|----------|
| VOTV | Thiruvananthapuram | 8.4667 | 76.95 | 8.475000381469727 | 76.92500305175781 | 2.9 | 332.0780944824219 | False |
| VECC | Kolkata | 22.6547 | 88.4467 | 22.674999237060547 | 88.42500305175781 | 3.1704 | 370.5028991699219 | False |
| VIDP | Delhi | 28.5667 | 77.1167 | 28.575000762939453 | 77.125 | 1.2284 | 227.69419860839844 | False |
| VOCI | Kochi | 10.15 | 76.4 | 10.175000190734863 | 76.375 | 3.9006 | 385.9404296875 | False |
| VABB | Mumbai | 19.1005 | 72.8585 | 19.125 | 72.875 | 3.2291 | 129.1292266845703 | False |

CSV: `outputs/lightning/combined_lis_thunder_hour_locations.csv`

---

## Distance to nearest grid cell

All five stations fall inside the Combined LIS domain (tropics / LIS FOV). Distances are listed above. They are **cell-centre to station**, not lightning-event distances.

---

## Missing-value checks

- Values labelled **NA** were missing/fill in the NetCDF (not replaced by 0).
- Finite values were read from the file at the chosen index; they were **not** interpolated in time or fabricated.
- Valid stations with data: **5 / 5**.

---

## Scientific interpretation

`annual_thunder_hours` is a **static climatological frequency**: roughly how many hours per year that cell is estimated to have thunder/lightning in the Combined LIS climatology.

It **does not** say whether lightning is occurring now, in the next 1–3 hours, or in a given year of the V2 METAR window. It **must not** be broadcast into hourly rows as if it were an observation.

A legitimate later use (not done here) would be a **single constant per station** (or a spatial prior), clearly labelled climatology.

---

## Limitations

- Not hourly; not orbital event lists; not Damini/IITM LLN.
- LIS sampling and detection efficiency are baked into the climatology.
- One number per site cannot encode monsoon vs winter, diurnal cycle, or trends 2014–2025.
- Grid is coarse relative to an aerodrome.
- Combining ISS-LIS and TRMM-LIS eras does not create 2024–2025 hourly coverage.

---

## Recommendation for V2

Do **not** merge into Phase 6/8 training tables in this phase. If a later experiment wants a **station climatology covariate**, this file can supply five scalar thunder-hour values. It cannot supply lightning nowcast features.

**Classification: `CLIMATOLOGY_FEATURE_CANDIDATE`**

Rationale: 5 of 5 stations have a finite nearest-cell annual thunder-hour value from the official grid. That is enough to consider a **static climatology feature** in a future experiment, and **not** enough to treat LIS as hourly lightning observations.

---

PHASE 9C COMPLETE — COMBINED LIS THUNDER-HOUR AUDIT READY
