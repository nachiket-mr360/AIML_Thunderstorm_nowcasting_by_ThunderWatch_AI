# Phase 10A — INSAT-3DR 3RIMG L2B CMK pilot audit

**Date:** 2026-09-22  
**File:** `dataset/satellite/insat3dr_cmk_pilot/3RIMG_22SEP2026_0015_L2B_CMK_V01R00.h5`  
**Audit only.** HDF not modified. No training, no feature implementation, no extra download, no commit.

## 1. Objective

Determine how cloud-mask data in this single INSAT-3DR L2B CMK HDF5 granule can be extracted and whether it can be synchronized with ThunderWatch station timestamps. This is not historical collection and not model integration.

## 2. HDF structure

- Opened with **h5py 3.16.0** (read-only).
- Root groups/datasets: `/CMK`, `/GeoX`, `/GeoY`, `/Latitude`, `/Longitude`, `/time`.
- No nested groups beyond the file root.

| Dataset | Shape | Dtype | Notes |
|---------|-------|-------|-------|
| `/CMK` | [1, 2816, 2805] | `int8` | {'_FillValue': 9, 'coordinates': 'time Latitude Longitude', 'flag_meanings': 'Clear Cloudy Probably_Clear Probably_Cloudy', 'flag_values': [0, 1, 2, 3], 'long_name': 'Cloud Mask'} |
| `/GeoX` | [2805] | `int32` | {} |
| `/GeoY` | [2816] | `int32` | {} |
| `/Latitude` | [2816, 2805] | `int16` | {'_FillValue': 32767, 'add_offset': 0.0, 'long_name': 'latitude', 'scale_factor': 0.009999999776482582, 'standard_name': 'latitude', 'units': 'degrees_north'} |
| `/Longitude` | [2816, 2805] | `int16` | {'_FillValue': 32767, 'add_offset': 0.0, 'long_name': 'longitude', 'scale_factor': 0.009999999776482582, 'standard_name': 'longitude', 'units': 'degrees_east'} |
| `/time` | [1] | `float64` | {'units': 'minutes since 2000-01-01 00:00:00'} |

Selected root attributes:

- `Satellite_Name`: `INSAT-3DR`
- `Sensor_Name`: `IMAGER`
- `Processing_Level`: `L2B`
- `Product_Type`: `GEOPHY`
- `HDF_Product_File_Name`: `3RIMG_22SEP2026_0015_L2B_CMK_V01R00.h5`
- `Acquisition_Date`: `22SEP2026`
- `Acquisition_Start_Time`: `22-SEP-2026T00:15:19`
- `Acquisition_End_Time`: `22-SEP-2026T00:42:13`
- `Acquisition_Time_in_GMT`: `0015`
- `Nominal_Central_Point_Coordinates(degrees)_Latitude_Longitude`: `[0.0, 74.0]`
- `left_longitude`: `-7.156703472137451`
- `right_longitude`: `155.15670776367188`
- `lower_latitude`: `-81.0415267944336`
- `upper_latitude`: `81.0415267944336`
- `title`: `3RIMG_22SEP2026_0015_L2B`
- `conventions`: `CF-1.6`

## 3. CMK variable and official metadata

- Dataset: `/CMK` shape `[1, 2816, 2805]` dtype `int8`
- `long_name`: Cloud Mask
- `coordinates`: time Latitude Longitude
- `_FillValue`: **9**
- Official classes from **file attributes** `flag_values` + `flag_meanings` (not invented):

| flag_value | flag_meaning |
|------------|--------------|
| 0 | Clear |
| 1 | Cloudy |
| 2 | Probably_Clear |
| 3 | Probably_Cloudy |

Observed value counts in this granule (including fill):

| raw value | count | meaning |
|-----------|-------|---------|
| 0 | 2061537 | Clear |
| 1 | 2471319 | Cloudy |
| 2 | 476241 | Probably_Clear |
| 3 | 749279 | Probably_Cloudy |
| 9 | 2140504 | fill |

## 4. Geolocation and time

- Latitude/Longitude are **2-D** (`int16` on disk), same spatial size as CMK spatial dims.
- Decode: `physical = raw * 0.01 + 0.0`; fill `32767`.
- Units: degrees (`degrees_north` / `degrees_east`).

- `/time` value: `14055855.0`
- `/time` units: `minutes since 2000-01-01 00:00:00`
- Decoded UTC from units: `2026-09-22T00:15:00+00:00`
- Acquisition start (attr): `22-SEP-2026T00:15:19`
- Acquisition end (attr): `22-SEP-2026T00:42:13`
- Acquisition GMT slot (attr): `0015` on `22SEP2026`

- Decoded valid lat range: -81.03999818861485 … 81.03999818861485
- Decoded valid lon range: -7.149999840185046 … 155.14999653212726
- Geo-valid pixels: 5761417; CMK in {0,1,2,3}: 5758376; both: 5758376

## 5. Five-station extraction

Nearest pixel among locations with valid lat/lon **and** CMK in official `flag_values`. Distance is haversine (km).

| station | st_lat | st_lon | pixel_lat | pixel_lon | distance_km | row | col | CMK raw | CMK class (from file) |
|---------|--------|--------|-----------|-----------|-------------|-----|-----|---------|----------------------|
| VOTV | 8.4667 | 76.95 | 8.459999810904264 | 76.94999828003347 | 0.7450270590207373 | 1176 | 1483 | 1 | Cloudy |
| VECC | 22.6547 | 88.4467 | 22.669999493286014 | 88.45999802276492 | 2.1808365550077426 | 811 | 1762 | 1 | Cloudy |
| VIDP | 28.5667 | 77.1167 | 28.57999936118722 | 77.13999827578664 | 2.71350302609852 | 669 | 1477 | 0 | Clear |
| VOCI | 10.15 | 76.4 | 10.159999772906303 | 76.41999829187989 | 2.4551047449266585 | 1130 | 1468 | 1 | Cloudy |
| VABB | 19.1005 | 72.8585 | 19.099999573081732 | 72.83999837189913 | 1.9448245604149514 | 896 | 1372 | 3 | Probably_Cloudy |

25 km neighbourhood (feasibility only; not a training feature):

| station | n_pixels | n_valid | cloud_fraction_among_valid | status |
|---------|----------|---------|----------------------------|--------|
| VOTV | 121 | 121 | 1.0 | OK |
| VECC | 98 | 98 | 1.0 | OK |
| VIDP | 98 | 98 | 0.0 | OK |
| VOCI | 116 | 116 | 1.0 | OK |
| VABB | 108 | 108 | 0.9814814814814815 | OK |

## 6. Station-level feature feasibility

- **cloud_mask_at_station**: feasible_for_this_file=`True`. Nearest valid CMK pixel can be sampled when lat/lon/CMK are valid. One file = one scan time, not a time series.
- **nearby_cloud_fraction**: feasible_for_this_file=`True`. Fraction of valid pixels within a radius using official flags Cloudy/Probably_Cloudy vs Clear/Probably_Clear. Unobserved pixels must not be treated as clear or zero.
- **clear_cloudy_indicator**: feasible_for_this_file=`True`. Can map flag 0/2 vs 1/3 only as labelled in flag_meanings. Probably_* must not be collapsed without documenting uncertainty.

These features are **not implemented**. Missing pixels must remain missing (`NOT_OBSERVED`), never 0.

## 7. Historical-data feasibility

- Single pilot file sufficient for historical model training: **False**
- This audit opened exactly one L2B CMK granule at one acquisition (22-SEP-2026T00:15:19–22-SEP-2026T00:42:13). ThunderWatch training uses hourly 2014–2025 station rows. One scan cannot be broadcast across historical hours and cannot supply a training time series. Institutional MOSDAC/ISRO archive access would be required for historical CMK.

Synchronization with ThunderWatch hourly timestamps is possible **in principle** by matching `Acquisition_Start_Time` / decoded `/time` to the nearest hour **only when a granule exists**. This file covers one ~27-minute scan on 22 Sep 2026 00:15–00:42 GMT. It must not be copied into other hours.

## 8. Classification

**`PILOT_STRUCTURE_VALID / HISTORICAL_ARCHIVE_REQUIRED`**

## 9. Recommended next step

If satellite CMK is still desired, Phase 10B should define an institutional MOSDAC download/access plan for a multi-year, station-timed L2B CMK archive and a join that leaves unobserved hours as NOT_OBSERVED. Do not implement features or retrain Model B until that archive exists.

## 10. Scientific non-negotiables

- Do not 0-fill missing satellite pixels.
- Do not broadcast this scan across historical hours.
- Do not invent CMK classes beyond flag_meanings.
- Do not train on this pilot.

PHASE 10A COMPLETE — INSAT-3DR CMK PILOT AUDIT READY
