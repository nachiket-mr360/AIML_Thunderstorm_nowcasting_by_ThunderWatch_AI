# Phase 10B-A — MOSDAC INSAT-3DR CMK historical availability audit

**Date:** 2026-09-22  
**Search only.** No granule download. No MOSDAC username/password. No training. No feature integration. No commit.

## 1. Objective

Determine whether historical `3RIMG_L2B_CMK` is **practically obtainable** for ThunderWatch AI over **2016-10-11 through 2025-12-31**, using the official MOSDAC Data Download API **search** (unauthenticated).

## 2. Official search implementation

Inspected:

- Manual: https://mosdac.gov.in/downloadapi-manual
- Package: https://www.mosdac.gov.in/software/mdapi.zip (`mdapi.py`, `config.json`)

From `mdapi.py` (not modified, not executed for download):

| Item | Value |
|------|-------|
| Search URL | `https://mosdac.gov.in/apios/datasets.json` |
| Method | HTTP GET, query parameters |
| Auth for search | **Not required** (manual §4) |
| Auth for download | Required (`gettoken` / Bearer) — **not used** |
| Download URL | `https://mosdac.gov.in/download_api/download` — **not called** |

Search parameters (official): `datasetId` (required), `startTime`, `endTime` (`YYYY-MM-DD`), `count` (max 100), `boundingBox` (`minLon,minLat,maxLon,maxLat`), `gId`, plus `startIndex` used internally for pagination of **downloads**.

Search JSON fields used by mdapi: `totalResults`, `totalSizeMB`, `itemsPerPage`, `entries[]` with `identifier`, `id`, `updated`.

mdapi prints **`totalResults` and `totalSizeMB` from the first search response** even though the `entries` list is a page. This audit therefore treats those two fields as the API’s reported archive totals **if present**, and does **not** paginate thousands of records.

Daily download quota stated in the manual: **5000 files/user/day**. Search does not consume that quota.

## 3. Search performed

- `datasetId=3RIMG_L2B_CMK`
- `startTime=2016-10-11` `endTime=2025-12-31`
- No boundingBox on the primary search
- No credentials
- No `startIndex` loop

- HTTP status: **200**
- Request: `https://mosdac.gov.in/apios/datasets.json?datasetId=3RIMG_L2B_CMK&startTime=2016-10-11&endTime=2025-12-31`
- Top-level keys: `['author', 'entries', 'itemsPerPage', 'links', 'message', 'query', 'startIndex', 'title', 'totalResults', 'totalSizeMB', 'updated']`

## 4. Availability numbers (API-reported)

- **A. File/granule count (`totalResults`):** `153845`
- **B. Total size (`totalSizeMB`):** `1397877`
- Page `itemsPerPage`: `100`
- Entries returned on this call: `100`
- First-page listing is newest-first. Latest listed: **2025-12-31T23:45:00Z** (`3RIMG_31DEC2025_2345_L2B_CMK_V01R00.h5`, gId `16905652`).
- Search-only day `2016-10-11`: `totalResults=35`. Earliest listed slot that day: **2016-10-11T00:45:00Z**. Not a walk of 153845 records.
- Per-entry `size` is null; volume is only collection `totalSizeMB`.

The 100-record page limit applies to **listed entries**, not necessarily to `totalResults`/`totalSizeMB`. mdapi uses those totals without summing pages. This audit **did not** iterate pages to verify by counting filenames.

### Sample granules (first page / samples)

| id | identifier | updated | size (if any) |
|----|------------|---------|---------------|
| 16905652 | 3RIMG_31DEC2025_2345_L2B_CMK_V01R00.h5 | 2025-12-31T23:45:00Z | None |
| 16905714 | 3RIMG_31DEC2025_2344_L2B_CMK_V01R00.h5 | 2025-12-31T23:44:00Z | None |
| 16905476 | 3RIMG_31DEC2025_2315_L2B_CMK_V01R00.h5 | 2025-12-31T23:15:00Z | None |
| 16905586 | 3RIMG_31DEC2025_2314_L2B_CMK_V01R00.h5 | 2025-12-31T23:14:00Z | None |
| 16905384 | 3RIMG_31DEC2025_2245_L2B_CMK_V01R00.h5 | 2025-12-31T22:45:00Z | None |
| 16905436 | 3RIMG_31DEC2025_2244_L2B_CMK_V01R00.h5 | 2025-12-31T22:44:00Z | None |
| 16905286 | 3RIMG_31DEC2025_2215_L2B_CMK_V01R00.h5 | 2025-12-31T22:15:00Z | None |
| 16905166 | 3RIMG_31DEC2025_2145_L2B_CMK_V01R00.h5 | 2025-12-31T21:45:00Z | None |

### One-day search (cadence check, still search-only)

- Window `2024-01-01`–`2024-01-01` status `200` `totalResults=44` `itemsPerPage=100`

If `totalResults` for a single calendar day is **44**, that is **near** the 48 half-hourly IMAGER slots, not exact. The product is **approximately half-hourly with gaps**, not a guaranteed 48 files every day. Extra near-duplicate slots (e.g. `2344` and `2345` on 31 Dec 2025) also appear. This is an API count for that day, not files on disk here.

### Bounding-box test (search-only, count=5)

- `boundingBox=68.0,6.0,98.0,37.0` (covers the five stations loosely)
- status `200` `totalResults=153845` `itemsPerPage=5`
- INSAT-3DR CMK boundbox on samples is full-disk (`west -7.15, south -81.04, east 155.15, north 81.04`). Unbounded and India bbox both report **`totalResults=153845`**. Bounding box is **not** a useful volume filter for this product.

## 5. Answers

**A.** CMK files 2016-10-11 → 2025-12-31 according to API `totalResults`: **153845**

**B.** Total volume according to API `totalSizeMB`: **1397877 MB** (API field totalSizeMB on unpaginated search (mdapi uses this as archive total, not page sum)). Not estimated.

**C.** Complete-archive download practically reasonable?

```json
{
  "total_files_api": 153845,
  "total_size_mb_api": 1397877,
  "total_size_gb_api": 1365.1142578125,
  "mosdac_daily_download_quota_files": 5000,
  "calendar_days_at_quota_if_all_files_wanted": 30.769,
  "complete_archive_practically_reasonable": false,
  "note": "Quota 5000 files/day is from the official mdapi manual. Complete-archive reasonableness uses API totals, not estimates."
}
```

**D.** Historical satellite-derived features for five stations? Not yet. Totals show the product exists in the catalog for the window, but building five-station hourly features still requires authenticated download, timestamp join, and NOT_OBSERVED for missing scans. Search ≠ local archive.

**E.** Next: Do not pull the complete archive in one shot. Next: selected periods / case-study sampling after MOSDAC login, or stop satellite training until institutional download capacity exists. Daily quota is 5000 files.

## 6. Classification

**`CATALOG_SEARCH_OK / DOWNLOAD_NOT_PERFORMED`**

## 7. Scientific rules observed

- Availability taken from the API response, not assumed.
- No volume invented; estimates, if any, are labeled.
- The Phase 10A single HDF is not treated as historical coverage.
- No scan broadcast across hours.
- Satellite acquisition time ≠ hourly atmospheric timestamp.

PHASE 10B-A COMPLETE — MOSDAC CMK AVAILABILITY SEARCH READY
