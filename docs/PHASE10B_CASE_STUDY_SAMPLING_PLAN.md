# Phase 10B-B — INSAT-3DR CMK case-study sampling plan

**Date:** 2026-09-22  
**Search + planning only.** No granule download. No MOSDAC credentials. No training. No Model B edits. No commit.

## 1. Objective

Design a **manageable, METAR-grounded** satellite case-study sample for five ThunderWatch stations using catalog `3RIMG_L2B_CMK` (available from 2016-10-11). Full archive (~1.365 TiB / 153,845 files) is **not** in scope.

## 2. Label source

Genuine nowcast tables: `dataset/multilocation/targets/{station}_nowcast_targets_2014_2025.csv`.
Positive hour = `thunderstorm_target==1` **and** `target_observed==1`. Window: 2016-10-11 through 2025-12-31.

Positive hours by station: `{'VOTV': 2904, 'VECC': 2867, 'VIDP': 1561, 'VOCI': 1935, 'VABB': 1315}`
All clustered episodes (gap ≤ 2 h): `{'VOTV': 907, 'VECC': 974, 'VIDP': 595, 'VOCI': 797, 'VABB': 382}`

## 3. Thunderstorm episodes selected

**25** episodes.
Distribution: `{'VABB': 5, 'VECC': 5, 'VIDP': 5, 'VOCI': 5, 'VOTV': 5}`

Selection: greedy per station, prefer longer observed events, cap 24 h duration, diversify year×season (winter / pre-monsoon / monsoon / post-monsoon). Not invented.

## 4. Controls

**22** control periods (same station; offsets among [-14, 14, -21, 21, -7, 7, -28, 28] days; ±6 h pad with all hours observed and no `thunderstorm_target` or observed `target_1h==1`). Controls omitted if no such span exists.

## 5. Satellite window

Each case uses **T_start−2 h through T_end+2 h** (UTC). Catalog queried **by calendar day**, granules kept if **filename UTC** falls in the window. Expected cadence for gap counting: **:15 and :45** slots (from Phase 10B-A names). Missing expected slots are `MISSING_EXPECTED_SCAN`, not zeros.

## 6. Catalog search results

- Unique CMK files that would later be downloaded: **1357**
- Expected :15/:45 slots across windows (with overlap counted per window): **1514**
- Missing expected scan rows: **345**
- Estimated storage: **12330.1 MB** (~12.04 GB) — Phase 10B-A mean file size totalSizeMB/totalResults; labeled estimate
- Practical to download later: **False**

No granules were downloaded.

## 7. Future extraction contract (not computed)

- `cloud_mask_at_station` — nearest valid CMK pixel (Phase 10A flags 0–3)
- `nearby_cloud_fraction` — Cloudy+Probably_Cloudy among valid pixels in a radius
- `clear_cloudy_indicator` — 0/2 vs 1/3 with `Probably_*` documented
- Unavailable scan → **`NOT_OBSERVED`**, never 0-fill, never broadcast one scan across hours

## 8. Temporal alignment rule

**Status:** PROPOSED / PENDING PILOT VALIDATION

Let T be the atmospheric/model timestamp (hourly, UTC). A CMK granule may be associated with T only if its filename UTC (DDMMMYYYY_HHMM) falls in [T, T+1h). Do not use acquisition end as T. If later HDF attributes give Acquisition_Start/End, require the acquisition interval to overlap [T, T+1h) and still record start, end, and filename time separately. Never copy one granule into other hours. Hours with no overlapping granule are NOT_OBSERVED.

Phase 10A pilot 0015 file acquired 00:15:19–00:42:13 UTC; filename slot is scan start, not an instant. Alignment needs HDF-level validation on a downloaded case before freezing the contract.

## 9. Unresolved before extraction

- No historical HDF files downloaded.
- Acquisition interval vs filename slot not validated on case-study files.
- MOSDAC login required to download.
- Duplicate near-slots (e.g. 2344 vs 2345) need a keep-one rule after inspection.

## 10. Classification

**`CASE_STUDY_SAMPLE_DESIGNED / DOWNLOAD_NOT_PERFORMED`**

This sample is **not** model validation and must not be used to claim satellite skill or to retrain Model B.

PHASE 10B-B COMPLETE — CASE-STUDY SAMPLE DESIGNED
