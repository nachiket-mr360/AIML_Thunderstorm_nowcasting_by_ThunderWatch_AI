#!/usr/bin/env python3
"""Phase 10B-A — MOSDAC 3RIMG_L2B_CMK availability SEARCH ONLY.

Does not download granules, does not authenticate, does not train models.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "satellite" / "phase10b"
DOC = ROOT / "docs" / "PHASE10B_MOSDAC_CMK_AVAILABILITY_AUDIT.md"
SEARCH_URL = "https://mosdac.gov.in/apios/datasets.json"
DATASET_ID = "3RIMG_L2B_CMK"
START = "2016-10-11"
END = "2025-12-31"
# India-ish box covering five ThunderWatch stations (search test only)
INDIA_BBOX = "68.0,6.0,98.0,37.0"
HEADERS = {"User-Agent": "ThunderWatch-SIH26072-audit/10B (search-only; no download)"}


def search(**params):
    r = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=120)
    out = {
        "request_url": r.url,
        "status_code": r.status_code,
        "ok": r.status_code == 200,
    }
    try:
        body = r.json()
    except Exception:
        out["body_text_preview"] = r.text[:2000]
        return out
    out["response_type"] = type(body).__name__
    if isinstance(body, dict):
        out["top_level_keys"] = sorted(body.keys())
        for k in ("totalResults", "totalSizeMB", "itemsPerPage", "startIndex"):
            if k in body:
                out[k] = body[k]
        entries = body.get("entries") or []
        out["n_entries_returned"] = len(entries)
        sample = []
        for item in entries[:8]:
            sample.append(
                {
                    "keys": sorted(item.keys()) if isinstance(item, dict) else None,
                    "id": item.get("id") if isinstance(item, dict) else None,
                    "identifier": item.get("identifier") if isinstance(item, dict) else None,
                    "updated": item.get("updated") if isinstance(item, dict) else None,
                    "size": item.get("size") if isinstance(item, dict) else None,
                    "preview": {k: item.get(k) for k in list(item)[:20]} if isinstance(item, dict) else item,
                }
            )
        out["sample_entries"] = sample
        out["raw_truncated"] = {k: body[k] for k in body if k != "entries"}
        if entries:
            out["raw_truncated"]["first_entry"] = entries[0]
            out["raw_truncated"]["last_entry_in_page"] = entries[-1]
    else:
        out["body"] = body
    return out


def cadence_hint(entries):
    names = [e.get("identifier") for e in entries if isinstance(e, dict)]
    times = []
    for ident in names:
        if not ident:
            continue
        # 3RIMG_22SEP2026_0015_L2B_CMK_V01R00.h5 → slot HHMM
        parts = ident.split("_")
        if len(parts) >= 3 and len(parts[2]) == 4 and parts[2].isdigit():
            times.append(parts[2])
    unique = sorted(set(times))
    return {"hhmm_slots_on_this_page": unique, "n_unique_hhmm": len(unique)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full = search(datasetId=DATASET_ID, startTime=START, endTime=END)
    bbox = search(
        datasetId=DATASET_ID,
        startTime=START,
        endTime=END,
        boundingBox=INDIA_BBOX,
        count="5",
    )
    # one-day probe to inspect cadence without paging the full archive
    day = search(datasetId=DATASET_ID, startTime="2024-01-01", endTime="2024-01-01")

    entries = []
    if full.get("sample_entries"):
        pass
    # re-get first-page identifiers from saved truncated
    first_page_idents = []
    first_page_ids = []
    first_page_sizes = []
    first_page_updated = []
    # We only have samples in `full`. Store cadence from day probe samples.

    total = full.get("totalResults")
    size_mb = full.get("totalSizeMB")
    items = full.get("itemsPerPage")

    practical = None
    if isinstance(total, (int, float)) and isinstance(size_mb, (int, float)):
        gb = size_mb / 1024.0
        daily_quota = 5000
        days_at_quota = total / daily_quota if total else None
        practical = {
            "total_files_api": total,
            "total_size_mb_api": size_mb,
            "total_size_gb_api": gb,
            "mosdac_daily_download_quota_files": daily_quota,
            "calendar_days_at_quota_if_all_files_wanted": days_at_quota,
            "complete_archive_practically_reasonable": bool(total <= 5000 and gb <= 50),
            "note": (
                "Quota 5000 files/day is from the official mdapi manual. "
                "Complete-archive reasonableness uses API totals, not estimates."
            ),
        }

    summary = {
        "phase": "10B-A",
        "audit_only": True,
        "search_only": True,
        "authenticated": False,
        "downloaded_granules": 0,
        "search_endpoint": SEARCH_URL,
        "datasetId": DATASET_ID,
        "period": {"startTime": START, "endTime": END},
        "mdapi_source": {
            "zip": "https://www.mosdac.gov.in/software/mdapi.zip",
            "manual": "https://mosdac.gov.in/downloadapi-manual",
            "search_requires_login": False,
            "download_requires_login": True,
        },
        "search_full_period_no_bbox": full,
        "search_one_day_2024_01_01": day,
        "search_bbox_india_count5": bbox,
        "answers": {
            "A_n_files_2016_10_11_to_2025_12_31": total,
            "B_total_volume": {
                "totalSizeMB": size_mb,
                "source": "API field totalSizeMB on unpaginated search (mdapi uses this as archive total, not page sum)",
            },
            "C_complete_archive_practically_reasonable": practical,
            "D_historical_station_features": None,
            "E_next": None,
        },
        "limitations": [
            "Search result page is limited (mdapi count max 100; itemsPerPage is the page).",
            "This audit did not paginate startIndex beyond the first page.",
            "Did not download any CMK HDF files.",
            "Did not use MOSDAC credentials.",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Fill D/E after we know totals
    if total is None:
        summary["answers"]["D_historical_station_features"] = (
            "UNKNOWN — search did not return totalResults."
        )
        summary["answers"]["E_next"] = "STOP until search totals are available."
        summary["classification"] = "SEARCH_FAILED_OR_INCOMPLETE"
    else:
        half_hourly_expected = 48 * 365.25 * (2025 - 2016 + 1)  # rough, labeled estimate only
        summary["cadence_estimate_not_from_api"] = {
            "label": "ESTIMATE_ONLY not used as availability",
            "if_half_hourly_every_day_2017_2025": "not computed as fact",
        }
        n_day = day.get("totalResults")
        summary["answers"]["D_historical_station_features"] = (
            "Not yet. Totals show the product exists in the catalog for the window, "
            "but building five-station hourly features still requires authenticated download, "
            "timestamp join, and NOT_OBSERVED for missing scans. Search ≠ local archive."
        )
        if practical and practical["complete_archive_practically_reasonable"] is False:
            summary["answers"]["E_next"] = (
                "Do not pull the complete archive in one shot. Next: selected periods / "
                "case-study sampling after MOSDAC login, or stop satellite training until "
                "institutional download capacity exists. Daily quota is 5000 files."
            )
        else:
            summary["answers"]["E_next"] = (
                "If volume is modest, a credentialed selected-period download can be planned later. "
                "Do not train until files are on disk and joined without broadcasting scans."
            )
        summary["one_day_totalResults"] = n_day
        summary["classification"] = "CATALOG_SEARCH_OK / DOWNLOAD_NOT_PERFORMED"

    with (OUT_DIR / "availability_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    write_doc(summary)
    print("totalResults", total, "totalSizeMB", size_mb, "itemsPerPage", items)
    print("day total", day.get("totalResults"), "bbox total", bbox.get("totalResults"))
    print("status", full.get("status_code"), day.get("status_code"), bbox.get("status_code"))


def write_doc(s):
    a = []
    p = a.append
    p("# Phase 10B-A — MOSDAC INSAT-3DR CMK historical availability audit")
    p("")
    p("**Date:** 2026-09-22  ")
    p("**Search only.** No granule download. No MOSDAC username/password. No training. No feature integration. No commit.")
    p("")
    p("## 1. Objective")
    p("")
    p("Determine whether historical `3RIMG_L2B_CMK` is **practically obtainable** for ThunderWatch AI over **2016-10-11 through 2025-12-31**, using the official MOSDAC Data Download API **search** (unauthenticated).")
    p("")
    p("## 2. Official search implementation")
    p("")
    p("Inspected:")
    p("")
    p("- Manual: https://mosdac.gov.in/downloadapi-manual")
    p("- Package: https://www.mosdac.gov.in/software/mdapi.zip (`mdapi.py`, `config.json`)")
    p("")
    p("From `mdapi.py` (not modified, not executed for download):")
    p("")
    p("| Item | Value |")
    p("|------|-------|")
    p("| Search URL | `https://mosdac.gov.in/apios/datasets.json` |")
    p("| Method | HTTP GET, query parameters |")
    p("| Auth for search | **Not required** (manual §4) |")
    p("| Auth for download | Required (`gettoken` / Bearer) — **not used** |")
    p("| Download URL | `https://mosdac.gov.in/download_api/download` — **not called** |")
    p("")
    p("Search parameters (official): `datasetId` (required), `startTime`, `endTime` (`YYYY-MM-DD`), `count` (max 100), `boundingBox` (`minLon,minLat,maxLon,maxLat`), `gId`, plus `startIndex` used internally for pagination of **downloads**.")
    p("")
    p("Search JSON fields used by mdapi: `totalResults`, `totalSizeMB`, `itemsPerPage`, `entries[]` with `identifier`, `id`, `updated`.")
    p("")
    p("mdapi prints **`totalResults` and `totalSizeMB` from the first search response** even though the `entries` list is a page. This audit therefore treats those two fields as the API’s reported archive totals **if present**, and does **not** paginate thousands of records.")
    p("")
    p("Daily download quota stated in the manual: **5000 files/user/day**. Search does not consume that quota.")
    p("")
    p("## 3. Search performed")
    p("")
    p(f"- `datasetId={s['datasetId']}`")
    p(f"- `startTime={s['period']['startTime']}` `endTime={s['period']['endTime']}`")
    p("- No boundingBox on the primary search")
    p("- No credentials")
    p("- No `startIndex` loop")
    p("")
    full = s["search_full_period_no_bbox"]
    p(f"- HTTP status: **{full.get('status_code')}**")
    p(f"- Request: `{full.get('request_url')}`")
    p(f"- Top-level keys: `{full.get('top_level_keys')}`")
    p("")
    p("## 4. Availability numbers (API-reported)")
    p("")
    p(f"- **A. File/granule count (`totalResults`):** `{full.get('totalResults')}`")
    p(f"- **B. Total size (`totalSizeMB`):** `{full.get('totalSizeMB')}`")
    p(f"- Page `itemsPerPage`: `{full.get('itemsPerPage')}`")
    p(f"- Entries returned on this call: `{full.get('n_entries_returned')}`")
    p("")
    p("The 100-record page limit applies to **listed entries**, not necessarily to `totalResults`/`totalSizeMB`. mdapi uses those totals without summing pages. This audit **did not** iterate pages to verify by counting filenames.")
    p("")
    p("### Sample granules (first page / samples)")
    p("")
    samples = full.get("sample_entries") or []
    if not samples:
        p("_No entries in the search response._")
    else:
        p("| id | identifier | updated | size (if any) |")
        p("|----|------------|---------|---------------|")
        for e in samples:
            p(f"| {e.get('id')} | {e.get('identifier')} | {e.get('updated')} | {e.get('size')} |")
    p("")
    day = s.get("search_one_day_2024_01_01") or {}
    p("### One-day search (cadence check, still search-only)")
    p("")
    p(f"- Window `2024-01-01`–`2024-01-01` status `{day.get('status_code')}` `totalResults={day.get('totalResults')}` `itemsPerPage={day.get('itemsPerPage')}`")
    p("")
    if day.get("totalResults") is not None:
        n = day.get("totalResults")
        p(f"If `totalResults` for a single calendar day is **{n}**, half-hourly INSAT IMAGER (48 slots/day) would be **consistent only if n is near 48** (full-disk CMK). This is an observation of the API total for that day, not a claim that every hour has a file on disk here.")
    p("")
    bbox = s.get("search_bbox_india_count5") or {}
    p("### Bounding-box test (search-only, count=5)")
    p("")
    p(f"- `boundingBox=68.0,6.0,98.0,37.0` (covers the five stations loosely)")
    p(f"- status `{bbox.get('status_code')}` `totalResults={bbox.get('totalResults')}` `itemsPerPage={bbox.get('itemsPerPage')}`")
    p("- INSAT-3DR CMK is a **full-disk** product in the Phase 10A pilot; a bbox may or may not reduce `totalResults`. If totals match the unbounded search, bbox is **not** a useful volume filter for this dataset.")
    p("")
    p("## 5. Answers")
    p("")
    ans = s["answers"]
    p(f"**A.** CMK files 2016-10-11 → 2025-12-31 according to API `totalResults`: **{ans['A_n_files_2016_10_11_to_2025_12_31']}**")
    p("")
    p(f"**B.** Total volume according to API `totalSizeMB`: **{ans['B_total_volume']['totalSizeMB']} MB** ({ans['B_total_volume']['source']}). Not estimated.")
    p("")
    p("**C.** Complete-archive download practically reasonable?")
    p("")
    c = ans["C_complete_archive_practically_reasonable"]
    p(f"```json\n{json.dumps(c, indent=2)}\n```")
    p("")
    p(f"**D.** Historical satellite-derived features for five stations? {ans['D_historical_station_features']}")
    p("")
    p(f"**E.** Next: {ans['E_next']}")
    p("")
    p("## 6. Classification")
    p("")
    p(f"**`{s.get('classification')}`**")
    p("")
    p("## 7. Scientific rules observed")
    p("")
    p("- Availability taken from the API response, not assumed.")
    p("- No volume invented; estimates, if any, are labeled.")
    p("- The Phase 10A single HDF is not treated as historical coverage.")
    p("- No scan broadcast across hours.")
    p("- Satellite acquisition time ≠ hourly atmospheric timestamp.")
    p("")
    p("PHASE 10B-A COMPLETE — MOSDAC CMK AVAILABILITY SEARCH READY")
    p("")
    DOC.write_text("\n".join(a), encoding="utf-8")


if __name__ == "__main__":
    main()
