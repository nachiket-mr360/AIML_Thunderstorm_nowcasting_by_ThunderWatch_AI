# Phase 9B — TRMM-LIS Science Data V5 pilot audit

**Date:** 2026-09-21  
**Product:** NASA TRMM LIS Science Data V5 (`TRMM_LIS_SC.05.0_*`) already on disk  
**No new download, no training, no V1/V2 model or feature-table edits**

Coordinates from `dataset/multilocation/iem_station_metadata.json` (IEM IN__ASOS station table). VOTV uses the **IEM station point** (8.4667°N, 76.95°E), not the Open-Meteo request point.

---

## Objective

Audit existing TRMM-LIS orbital NetCDF granules in `dataset/lightning/trmm_lis_pilot/`: discover structure, count lightning objects, and test whether any **detected lightning** falls within 50 km of the five V2 stations. This is **not** a feature build and **not** a training experiment.

---

## Input files

- Directory: `dataset/lightning/trmm_lis_pilot/`
- Granule count: **101** `.nc` files
- Opened successfully: **101**
- Open failures: **0**
- Filename pattern: `TRMM_LIS_SC.05.0_YYYY.DDD.ORBIT.nc` (year, day-of-year, orbit id)
- Naming span in this folder: `TRMM_LIS_SC.05.0_2013.365.91865.nc` … `TRMM_LIS_SC.05.0_2014.007.91974.nc`

NASA describes this product as **orbital total-lightning observations**, typically **one file per orbit** and **~16 orbits/day**. That is **not** hourly continuous coverage at a station.

---

## NetCDF structure

Discovered dynamically from `TRMM_LIS_SC.05.0_2013.365.91865.nc` using `netCDF4`.

- Opened with: `netCDF4`
- Root attributes (truncated): {"history": "This TRMM LIS product in netCDF4/HDF5 format is converted from original LIS science data product in hdf4 format", "Conventions": "CF-1.6"}
- Dimensions: {'image_xsize': 720, 'image_ysize': 500, 'clr_table_dim1': 3, 'clr_table_dim2': 256, 'background_summary_dim': 115, 'latlon_dim': 2, 'corners_dim': 8, 'area_dim': 17, 'flash_dim': 25, 'group_dim': 390, 'event_dim': 1987, 'viewtime_dim': 10938, 'one_second_dim': 4081, 'threshold_dim': 16, 'vector_dim': 3, 'transform_matrix_dim': 9, 'processing_stage_dim': 6}
- Groups: ['/']
- Variable count: 147

### Variables (first file)

| Path | Shape | Dtype | long_name / units |
|------|-------|-------|-------------------|
| `/raster_image` | (500, 720) | `uint8` | raster summary image 1 |
| `/raster_image_color_table` | (256, 3) | `uint8` | color table for raster summary image 1 |
| `/orbit_summary_id_number` | () | `int32` | orbit ID 1 |
| `/orbit_summary_TAI93_start` | () | `float64` | TAI93 start time seconds since 1993-01-01 00:00:00.000 |
| `/orbit_summary_UTC_start` | () | `<class 'str'>` | UTC start time  |
| `/orbit_summary_GPS_start` | () | `float64` | GPS start time seconds |
| `/orbit_summary_TAI93_end` | () | `float64` | TAI93 end time seconds since 1993-01-01 00:00:00.000 |
| `/orbit_summary_start_longitude` | () | `float32` | start longitude degrees_east |
| `/orbit_summary_end_longitude` | () | `float32` | end longitude degrees_east |
| `/orbit_summary_point_data_count` | () | `int16` | number of point data records count |
| `/orbit_summary_point_data_address` | () | `int16` | point data child record number 1 |
| `/orbit_summary_one_second_count` | () | `int32` | number of one second records count |
| `/orbit_summary_one_second_address` | () | `int32` | one-second child record number 1 |
| `/orbit_summary_summary_image_count` | () | `int16` | number of summary GIF images count |
| `/orbit_summary_summary_image_address` | () | `int16` | summary GIF image record number 1 |
| `/orbit_summary_inspection_code` | () | `uint16` | inspection code 1 |
| `/orbit_summary_configuration_code` | () | `uint16` | configuration code 1 |
| `/point_summary_parent_address` | () | `int32` | parent record number 1 |
| `/point_summary_event_count` | () | `int32` | number of events count |
| `/point_summary_event_address` | () | `int32` | event record number 1 |
| `/point_summary_group_count` | () | `int32` | number of groups count |
| `/point_summary_group_address` | () | `int32` | group record number 1 |
| `/point_summary_flash_count` | () | `int32` | number of flashes count |
| `/point_summary_flash_address` | () | `int32` | flash record number 1 |
| `/point_summary_area_count` | () | `int32` | number of areas count |
| `/point_summary_area_address` | () | `int32` | area record number 1 |
| `/point_summary_bg_count` | () | `int32` | number of backgrounds count |
| `/point_summary_bg_address` | () | `int32` | background image summary record number 1 |
| `/point_summary_vt_count` | () | `int32` | number of viewtime granules count |
| `/point_summary_vt_address` | () | `int32` | viewtime granule record number 1 |
| `/bg_summary_TAI93_time` | (115,) | `float64` | background image start time seconds since 1993-01-01 00:00:00.000 |
| `/bg_summary_address` | (115,) | `int32` | background image number 1 |
| `/bg_summary_boresight` | (115, 2) | `float32` | background image boresight position degrees |
| `/bg_summary_lat` | (115,) | `float32` | background image boresight latitude degrees_north |
| `/bg_summary_lon` | (115,) | `float32` | background image boresight longitude degrees_east |
| `/bg_summary_corners` | (115, 8) | `float32` | background image corner positions degree |
| `/lightning_area_TAI93_time` | (17,) | `float64` | area time seconds since 1993-01-01 00:00:00.000 |
| `/lightning_area_delta_time` | (17,) | `float32` | area point data time span seconds |
| `/lightning_area_observe_time` | (17,) | `int16` | area duration of observation seconds |
| `/lightning_area_location` | (17, 2) | `float32` | area geolocated position degree |
| `/lightning_area_lat` | (17,) | `float32` | area latitude degrees_north |
| `/lightning_area_lon` | (17,) | `float32` | area longitude degrees_east |
| `/lightning_area_net_radiance` | (17,) | `float32` | area calibrated radiance uJ/sr/m2/um |
| `/lightning_area_footprint` | (17,) | `float32` | area footprint size km2 |
| `/lightning_area_address` | (17,) | `int32` | area record number 1 |
| `/lightning_area_parent_address` | (17,) | `int32` | area parent record number 1 |
| `/lightning_area_child_address` | (17,) | `int32` | area child record number 1 |
| `/lightning_area_child_count` | (17,) | `int32` | area child record count count |
| `/lightning_area_grandchild_count` | (17,) | `int32` | area grandchild record count count |
| `/lightning_area_greatgrandchild_count` | (17,) | `int32` | area greatgrandchild record count count |
| `/lightning_area_approx_threshold` | (17,) | `int8` | area 8-bit threshold 1 |
| `/lightning_area_alert_flag` | (17,) | `uint8` | area alert flag 1 |
| `/lightning_area_cluster_index` | (17,) | `int8` | area clustering probability percent |
| `/lightning_area_density_index` | (17,) | `int8` | area lightning activity 1 |
| `/lightning_area_noise_index` | (17,) | `int8` | area noise index 100percent |
| `/lightning_area_oblong_index` | (17,) | `float32` | area eccentricity 1 |
| `/lightning_area_grouping_sequence` | (17,) | `int32` | area time order 1 |
| `/lightning_area_grouping_status` | (17,) | `int8` | area grouping_status 1 |
| `/lightning_flash_TAI93_time` | (25,) | `float64` | flash time seconds since 1993-01-01 00:00:00.000 |
| `/lightning_flash_delta_time` | (25,) | `float32` | flash point data time span seconds |
| `/lightning_flash_observe_time` | (25,) | `int16` | flash duration of observation seconds |
| `/lightning_flash_location` | (25, 2) | `float32` | flash geolocated position degrees |
| `/lightning_flash_lat` | (25,) | `float32` | flash latitude degrees_north |
| `/lightning_flash_lon` | (25,) | `float32` | flash longitude degrees_east |
| `/lightning_flash_radiance` | (25,) | `float32` | flash calibrated radiance uJ/sr/m2/um |
| `/lightning_flash_footprint` | (25,) | `float32` | flash footprint size km2 |
| `/lightning_flash_address` | (25,) | `int32` | flash record number 1 |
| `/lightning_flash_parent_address` | (25,) | `int32` | flash parent record number 1 |
| `/lightning_flash_child_address` | (25,) | `int32` | flash child record number 1 |
| `/lightning_flash_child_count` | (25,) | `int32` | flash child record count count |
| `/lightning_flash_grandchild_count` | (25,) | `int32` | flash grandchild record count count |
| `/lightning_flash_approx_threshold` | (25,) | `int8` | flash 8-bit threshold 1 |
| `/lightning_flash_alert_flag` | (25,) | `uint8` | flash alert flag 1 |
| `/lightning_flash_cluster_index` | (25,) | `int8` | flash clustering probability percent |
| `/lightning_flash_density_index` | (25,) | `int8` | flash lightning activity 1 |
| `/lightning_flash_noise_index` | (25,) | `int8` | flash noise index 100percent |
| `/lightning_flash_oblong_index` | (25,) | `float32` | flash eccentricity 1 |
| `/lightning_flash_grouping_sequence` | (25,) | `int32` | flash time order 1 |
| `/lightning_flash_grouping_status` | (25,) | `int8` | flash grouping_status 1 |
| `/lightning_flash_glint_index` | (25,) | `float32` | flash solar glint cosine 1 |
| `/lightning_group_TAI93_time` | (390,) | `float64` | group time seconds since 1993-01-01 00:00:00.000 |
| `/lightning_group_observe_time` | (390,) | `int16` | length of observation seconds |
| `/lightning_group_location` | (390, 2) | `float32` | group geolocated position degrees |
| `/lightning_group_lat` | (390,) | `float32` | group latitude degrees_north |
| `/lightning_group_lon` | (390,) | `float32` | group longitude degrees_east |
| `/lightning_group_radiance` | (390,) | `float32` | group calibrated radiance uJ/sr/m2/um |
| `/lightning_group_footprint` | (390,) | `float32` | group footprint size km2 |
| `/lightning_group_address` | (390,) | `int32` | group record number 1 |
| `/lightning_group_parent_address` | (390,) | `int32` | group parent record number 1 |
| `/lightning_group_child_address` | (390,) | `int32` | group child record number 1 |
| `/lightning_group_child_count` | (390,) | `int32` | group child record count count |
| `/lightning_group_approx_threshold` | (390,) | `int8` | group 8-bit threshold 1 |
| `/lightning_group_alert_flag` | (390,) | `uint8` | group alert flag 1 |
| `/lightning_group_cluster_index` | (390,) | `int8` | group clustering probability percent |
| `/lightning_group_density_index` | (390,) | `int8` | group lightning activity 1 |
| `/lightning_group_noise_index` | (390,) | `int8` | group noise index 100percent |
| `/lightning_group_oblong_index` | (390,) | `float32` | group eccentricity 1 |
| `/lightning_group_grouping_sequence` | (390,) | `int32` | group time order 1 |
| `/lightning_group_grouping_status` | (390,) | `int8` | group grouping_status 1 |
| `/lightning_group_glint_index` | (390,) | `float32` | group solar glint cosine 1 |
| `/lightning_event_TAI93_time` | (1987,) | `float64` | event time seconds since 1993-01-01 00:00:00.000 |
| `/lightning_event_observe_time` | (1987,) | `int16` | event duration of observation seconds |
| `/lightning_event_location` | (1987, 2) | `float32` | event geolocated position degree |
| `/lightning_event_lat` | (1987,) | `float32` | event latitude degrees_north |
| `/lightning_event_lon` | (1987,) | `float32` | event longitude degrees_east |
| `/lightning_event_radiance` | (1987,) | `float32` | event calibrated radiance uJ/sr/m2/um |
| `/lightning_event_footprint` | (1987,) | `float32` | event footprint size km2 |
| `/lightning_event_address` | (1987,) | `int32` | event record number 1 |
| `/lightning_event_parent_address` | (1987,) | `int32` | event parent record number 1 |
| `/lightning_event_x_pixel` | (1987,) | `int8` | event CCD pixel column 1 |
| `/lightning_event_y_pixel` | (1987,) | `int8` | event CCD pixel row 1 |
| `/lightning_event_bg_value` | (1987,) | `int16` | event background illumination 1 |
| `/lightning_event_bg_radiance` | (1987,) | `int16` | event background radiance W/sr/m2/um |
| `/lightning_event_approx_threshold` | (1987,) | `int8` | event 8-bit threshold 1 |
| `/lightning_event_alert_flag` | (1987,) | `uint8` | event alert flag 1 |
| `/lightning_event_cluster_index` | (1987,) | `int8` | event clustering probability percent |
| `/lightning_event_density_index` | (1987,) | `int8` | event lightning activity 1 |
| `/lightning_event_noise_index` | (1987,) | `int8` | event noise index 100percent |
| `/lightning_event_grouping_sequence` | (1987,) | `int32` | event time order 1 |
| `/lightning_event_amplitude` | (1987,) | `int8` | event 7-bit amplitude 1 |
| `/lightning_event_sza_index` | (1987,) | `uint8` | event solar zenith angle degree |
| `/lightning_event_glint_index` | (1987,) | `uint8` | event solar glint cosine degrees |
| `/lightning_event_bg_value_flag` | (1987,) | `int8` | event background illumination flag 1 |
| `/viewtime_location` | (10938, 2) | `float32` | viewtime grid cell position degree |
| `/viewtime_lat` | (10938,) | `float32` | viewtime latitude degrees_north |
| `/viewtime_lon` | (10938,) | `float32` | viewtime longitude degrees_east |
| `/viewtime_TAI93_start` | (10938,) | `int32` | viewtime start time seconds since 1993-01-01 00:00:00.000 |
| `/viewtime_TAI93_end` | (10938,) | `int32` | viewtime end time seconds since 1993-01-01 00:00:00.000 |
| `/viewtime_effective_obs` | (10938,) | `float32` | effective viewtime seconds |
| `/viewtime_alert_flag` | (10938,) | `uint8` | viewtime alert flag 1 |
| `/viewtime_approx_threshold` | (10938,) | `int8` | viewtime 8-bit threshold 1 |
| `/one_second_TAI93_time` | (4081,) | `float64` | one second granule start time seconds since 1993-01-01 00:00:00.000 |
| `/one_second_alert_summary` | (4081,) | `uint8` | one second granule alert summary flag 1 |
| `/one_second_instrument_alert` | (4081,) | `uint8` | one second granule instrument alert flag 1 |
| `/one_second_platform_alert` | (4081,) | `uint8` | one second granule platform alert flag 1 |
| `/one_second_external_alert` | (4081,) | `uint8` | one second granule external alert flag 1 |
| `/one_second_processing_alert` | (4081,) | `uint8` | one second granule processing alert flag 1 |
| `/one_second_position_vector` | (4081, 3) | `float32` | one second granule platform coordinates m |
| `/one_second_velocity_vector` | (4081, 3) | `float32` | one second granule platform velocity m/s |
| `/one_second_transform_matrix` | (4081, 9) | `float32` | one second granule transform matrix 1 |
| `/one_second_solar_vector` | (4081, 3) | `float32` | one second granule solar vector 1 |
| `/one_second_ephemeris_quality_flag` | (4081,) | `int32` | one second granule ephemeris quality flag 1 |
| `/one_second_attitude_quality_flag` | (4081,) | `int32` | one second granule attituide quality flag 1 |
| `/one_second_boresight_threshold` | (4081,) | `int8` | one second granule threshold estimate 1 |
| `/one_second_thresholds` | (4081, 16) | `int8` | one second granule 8-bit threshold values 1 |
| `/one_second_noise_index` | (4081,) | `int8` | one second granule noise index 100percent |
| `/one_second_event_count` | (4081, 6) | `int16` | one second granule event count count |

Full catalog is also in `outputs/lightning/trmm_lis_pilot_summary.json` (`variable_catalog`).

---

## Identified lightning variables

Names below are **as found in the files**, not assumed a priori.

| Role | Variable path(s) |
|------|------------------|
| Lightning events (lat) | `/lightning_event_lat` |
| Lightning events (lon) | `/lightning_event_lon` |
| Lightning groups (lat) | `/lightning_group_lat` |
| Lightning groups (lon) | `/lightning_group_lon` |
| Lightning flashes (lat) | `/lightning_flash_lat` |
| Lightning flashes (lon) | `/lightning_flash_lon` |
| Event time | `/lightning_event_TAI93_time` |
| Group time | `/lightning_group_TAI93_time` |
| Flash time | `/lightning_flash_TAI93_time` |
| Orbit/other time | `/orbit_summary_TAI93_start` |

All lat-like paths: ['/bg_summary_lat', '/lightning_area_lat', '/lightning_flash_lat', '/lightning_group_lat', '/lightning_event_lat', '/viewtime_lat', '/one_second_platform_alert']

All lon-like paths: ['/orbit_summary_start_longitude', '/orbit_summary_end_longitude', '/bg_summary_lon', '/lightning_area_lon', '/lightning_area_oblong_index', '/lightning_flash_lon', '/lightning_flash_oblong_index', '/lightning_group_lon', '/lightning_group_oblong_index', '/lightning_event_lon', '/viewtime_lon']

All time-like paths: ['/orbit_summary_TAI93_start', '/orbit_summary_TAI93_end', '/bg_summary_TAI93_time', '/lightning_area_TAI93_time', '/lightning_area_delta_time', '/lightning_area_observe_time', '/lightning_flash_TAI93_time', '/lightning_flash_delta_time', '/lightning_flash_observe_time', '/lightning_group_TAI93_time', '/lightning_group_observe_time', '/lightning_event_TAI93_time', '/lightning_event_observe_time', '/viewtime_location', '/viewtime_lat', '/viewtime_lon', '/viewtime_TAI93_start', '/viewtime_TAI93_end', '/viewtime_effective_obs', '/viewtime_alert_flag', '/viewtime_approx_threshold', '/one_second_TAI93_time']

**LIS hierarchy (from NASA product meaning, checked against names present):** an **event** is a single illuminated pixel; a **group** is adjacent events in one frame; a **flash** is grouped groups over successive frames. Counts of events ≠ flashes.

---

## Per-file inventory summary

CSV: `outputs/lightning/trmm_lis_pilot_inventory.csv` (101 rows).

Numeric flash counts available for 100 files; event counts for 100 files.

If a field could not be determined it is **NA** (not imputed as 0).

First rows:

| filename | n_events | n_groups | n_flashes | lat min/max | lon min/max | start | end | n_geoloc |
|----------|----------|----------|-----------|-------------|-------------|-------|-----|----------|
| `TRMM_LIS_SC.05.0_2013.365.91865.nc` | 1987 | 390 | 25 | -31.0469913482666/35.46873092651367 | 25.76117515563965/149.95359802246094 | 2013-12-31T22:59:17.194568+00:00 | 2013-12-31T23:36:09.463184+00:00 | 25 |
| `TRMM_LIS_SC.05.0_2014.001.91866.nc` | 2425 | 529 | 44 | -35.558815002441406/37.04716873168945 | -60.865875244140625/150.5687713623047 | 2014-01-01T00:25:54.697986+00:00 | 2014-01-01T01:55:38.908908+00:00 | 44 |
| `TRMM_LIS_SC.05.0_2014.001.91867.nc` | 1595 | 376 | 52 | -0.9491630792617798/33.038665771484375 | 32.494873046875/146.37820434570312 | 2014-01-01T02:20:04.129947+00:00 | 2014-01-01T02:51:04.827869+00:00 | 52 |
| `TRMM_LIS_SC.05.0_2014.001.91868.nc` | 3723 | 863 | 50 | -31.318233489990234/29.887046813964844 | -126.79517364501953/52.81311798095703 | 2014-01-01T03:37:25.703601+00:00 | 2014-01-01T04:56:11.719531+00:00 | 50 |
| `TRMM_LIS_SC.05.0_2014.001.91869.nc` | 2426 | 600 | 74 | -25.569759368896484/31.206253051757812 | -169.2617950439453/156.98326110839844 | 2014-01-01T05:11:28.569186+00:00 | 2014-01-01T06:23:02.506168+00:00 | 74 |
| `TRMM_LIS_SC.05.0_2014.001.91870.nc` | 15808 | 2220 | 178 | -20.402856826782227/7.549598217010498 | -66.56358337402344/138.1962890625 | 2014-01-01T06:48:34.553470+00:00 | 2014-01-01T07:47:25.550006+00:00 | 178 |
| `TRMM_LIS_SC.05.0_2014.001.91871.nc` | 12780 | 3099 | 238 | -22.29457664489746/36.570152282714844 | -76.28314208984375/131.9052734375 | 2014-01-01T08:25:05.904666+00:00 | 2014-01-01T09:25:38.894504+00:00 | 238 |
| `TRMM_LIS_SC.05.0_2014.001.91872.nc` | 1431 | 278 | 13 | -30.363086700439453/6.181515216827393 | -80.45941162109375/129.79295349121094 | 2014-01-01T10:04:27.559131+00:00 | 2014-01-01T11:04:08.353214+00:00 | 13 |
| `TRMM_LIS_SC.05.0_2014.001.91873.nc` | 301 | 64 | 5 | -35.34311294555664/-20.00443458557129 | -144.8514862060547/131.9207763671875 | 2014-01-01T11:25:00.565295+00:00 | 2014-01-01T12:42:41.575961+00:00 | 5 |
| `TRMM_LIS_SC.05.0_2014.001.91874.nc` | 6373 | 1473 | 143 | -27.595182418823242/17.635889053344727 | -173.0841827392578/50.5419807434082 | 2014-01-01T12:54:36.881183+00:00 | 2014-01-01T13:59:50.891620+00:00 | 143 |
| `TRMM_LIS_SC.05.0_2014.001.91875.nc` | 15648 | 3383 | 300 | -35.36710739135742/6.914726257324219 | 0.6270714402198792/169.46078491210938 | 2014-01-01T14:19:39.428821+00:00 | 2014-01-01T15:37:45.623996+00:00 | 300 |
| `TRMM_LIS_SC.05.0_2014.001.91877.nc` | 11151 | 3381 | 303 | -32.87221145629883/4.308852195739746 | -42.424381256103516/148.8790740966797 | 2014-01-01T17:35:33.528156+00:00 | 2014-01-01T18:50:13.990543+00:00 | 303 |
| `TRMM_LIS_SC.05.0_2014.001.91878.nc` | 2641 | 643 | 74 | -12.040209770202637/3.0218493938446045 | -65.26750946044922/125.16692352294922 | 2014-01-01T19:11:50.967899+00:00 | 2014-01-01T20:08:07.616687+00:00 | 74 |
| `TRMM_LIS_SC.05.0_2014.001.91879.nc` | 16767 | 3235 | 285 | -27.238143920898438/7.4336042404174805 | -78.35066986083984/117.65006256103516 | 2014-01-01T20:40:32.457721+00:00 | 2014-01-01T21:48:08.478900+00:00 | 285 |
| `TRMM_LIS_SC.05.0_2014.001.91880.nc` | 7589 | 2296 | 201 | -30.945837020874023/37.83904266357422 | -68.0576400756836/163.42523193359375 | 2014-01-01T22:04:05.172473+00:00 | 2014-01-01T23:24:30.845538+00:00 | 201 |
| … | 86 more rows in CSV | | | | | | | |

---

## Five-location coverage table

50 km radius, haversine, IEM station coordinates.

**Swath** = at least one geolocated coverage/view point (or analogous lat/lon array used as coverage) within 50 km.  
**Lightning** = at least one **flash/group/event coordinate** (whichever lightning lat/lon pair was used: prefer flash) within 50 km.

These are **not** the same.

| Station | Lat | Lon | Files with swath in 50 km | Lightning obs within 50 km | Files with lightning in 50 km | Earliest | Latest | Distinct orbits (lightning) | Distinct orbits (swath) |
|---------|-----|-----|---------------------------|----------------------------|-------------------------------|----------|--------|-----------------------------|-------------------------|
| VOTV (Thiruvananthapuram) | 8.4667 | 76.95 | 8 | 0 | 0 | NA | NA | 0 | 8 |
| VECC (Kolkata) | 22.6547 | 88.4467 | 10 | 0 | 0 | NA | NA | 0 | 10 |
| VIDP (Delhi) | 28.5667 | 77.1167 | 15 | 0 | 0 | NA | NA | 0 | 15 |
| VOCI (Kochi) | 10.15 | 76.4 | 7 | 0 | 0 | NA | NA | 0 | 7 |
| VABB (Mumbai) | 19.1005 | 72.8585 | 8 | 0 | 0 | NA | NA | 0 | 8 |

---

## Actual lightning-event proximity results

- Any of five stations with **detected lightning** within 50 km in this pilot: **NO**
- Any of five stations with **swath/coverage** within 50 km: **YES**

Absence of lightning in a granule is **not** encoded as a meteorological zero-flash hour. Unobserved time (no overpass) is **missing**, not zero.

---

## Missing / unknown information

- If a variable path is `None` above, that object class was **not found by name** in the first file.
- Times decoded from TAI-93 seconds when numeric magnitudes were large; if decoding is wrong, CSV may show filename-derived date only for start and **NA** for end.
- Detection efficiency, view-time, and optical-pixel footprint quality flags were **not** turned into features.
- Pilot does **not** cover the full V2 window 2014-01-01 → 2025-12-31 (TRMM-LIS ends 2015; this folder is a short orbital sample around 2013-365 / 2014-001…).

---

## Scientific limitations

1. **Orbital sampling:** ~16 orbits/day globally does **not** yield continuous hourly lightning at VOTV/VECC/VIDP/VOCI/VABB.
2. **Swath ≠ lightning:** an overpass can view a station with **zero** flashes.
3. **No 0-fill:** hours without LIS view must remain missing if this product were ever joined.
4. **Not a thunderstorm label:** LIS flashes are not METAR `TS` and must not replace V2 labels.
5. **Short pilot:** even if a few nearby flashes exist, sample size is far below a V2 training layer.
6. **TRMM-LIS overlap with V2** is at most through April 2015 (mission), not 2014–2025.

---

## Final conclusion

This audit confirms the on-disk files are **NASA TRMM-LIS Science orbital granules** with discoverable lightning object arrays (see catalog). It does **not** show a continuous, five-station, hourly observation series suitable for V2 feature training.

**Classification: `REFERENCE_ONLY`**

Rationale: files opened and lightning coordinates exist, but the pilot is orbital/short and does not support V2 hourly lightning predictors; nearby-flash counts must not be treated as a training-ready layer.

---

## Critical scientific rules

- Do not convert missing lightning observations into 0.
- Do not claim that TRMM-LIS provides continuous hourly lightning observations.
- Do not create synthetic lightning values.
- Do not create a thunderstorm label from this dataset.
- Do not train anything on this pilot.
- Do not calculate model performance.
- Do not claim the lightning layer is ready for V2 training.
- Distinguish satellite **observation coverage** from **actual detected lightning**.

---

PHASE 9B COMPLETE — TRMM-LIS PILOT AUDIT READY
