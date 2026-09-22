#!/usr/bin/env python3
"""Phase 10B-B — INSAT-3DR CMK case-study sampling (search only)."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "dataset" / "multilocation" / "targets"
OUT = ROOT / "outputs" / "satellite" / "phase10b"
DOC = ROOT / "docs" / "PHASE10B_CASE_STUDY_SAMPLING_PLAN.md"
SEARCH_URL = "https://mosdac.gov.in/apios/datasets.json"
DATASET = "3RIMG_L2B_CMK"
HEADERS = {"User-Agent": "ThunderWatch-SIH26072-audit/10BB (search-only)"}

STATIONS = {
    "VOTV": (8.4667, 76.9500),
    "VECC": (22.6547, 88.4467),
    "VIDP": (28.5667, 77.1167),
    "VOCI": (10.1500, 76.4000),
    "VABB": (19.1005, 72.8585),
}
CMK_START = datetime(2016, 10, 11, tzinfo=timezone.utc)
CMK_END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
MEAN_MB = 1397877.0 / 153845.0  # Phase 10B-A catalog mean; labeled estimate
GAP_HOURS = 2  # hours with no positive label that split episodes
CONTROL_OFFSETS_DAYS = (-14, 14, -21, 21, -7, 7, -28, 28)


def season_of(dt: datetime) -> str:
    m = dt.month
    if m in (3, 4, 5):
        return "pre_monsoon"
    if m in (6, 7, 8, 9):
        return "monsoon"
    if m in (10, 11, 12):
        return "post_monsoon"
    return "winter"


def load_station(sid: str) -> pd.DataFrame:
    path = TARGETS / f"{sid.lower()}_nowcast_targets_2014_2025.csv"
    df = pd.read_csv(
        path,
        usecols=[
            "station_id",
            "timestamp_utc",
            "thunderstorm_target",
            "target_observed",
            "target_1h",
            "target_1h_observed",
        ],
        parse_dates=["timestamp_utc"],
    )
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df[(df["timestamp_utc"] >= CMK_START) & (df["timestamp_utc"] <= CMK_END)].copy()
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def episodes_from(df: pd.DataFrame, sid: str):
    pos = df[(df["thunderstorm_target"] == 1) & (df["target_observed"] == 1)]
    times = list(pos["timestamp_utc"])
    if not times:
        return []
    eps = []
    start = times[0]
    prev = times[0]
    hours = [times[0]]
    for t in times[1:]:
        if (t - prev) <= timedelta(hours=GAP_HOURS):
            hours.append(t)
            prev = t
        else:
            eps.append(_ep(sid, start, prev, hours))
            start = t
            prev = t
            hours = [t]
    eps.append(_ep(sid, start, prev, hours))
    return eps


def _ep(sid, start, end, hours):
    return {
        "station": sid,
        "event_start_utc": start,
        "event_end_utc": end,
        "n_positive_hours": len(hours),
        "duration_h": (end - start).total_seconds() / 3600.0 + 1.0,
        "year": start.year,
        "season": season_of(start),
        "hours": hours,
    }


def select_episodes(all_eps):
    """Greedy: up to 5 per station, different year-season, prefer longer events."""
    selected = []
    by_st = defaultdict(list)
    for e in all_eps:
        by_st[e["station"]].append(e)
    for sid in STATIONS:
        pool = sorted(
            by_st[sid],
            key=lambda x: (-x["n_positive_hours"], -x["duration_h"], x["event_start_utc"]),
        )
        used_bins = set()
        picked = []
        for e in pool:
            b = (e["year"], e["season"])
            if b in used_bins:
                continue
            # keep events reasonably compact for satellite windows
            if e["duration_h"] > 24:
                continue
            used_bins.add(b)
            picked.append(e)
            if len(picked) >= 5:
                break
        if len(picked) < 5:
            for e in pool:
                if e in picked or e["duration_h"] > 24:
                    continue
                picked.append(e)
                if len(picked) >= 5:
                    break
        selected.extend(picked)
    selected.sort(key=lambda x: (x["station"], x["event_start_utc"]))
    for i, e in enumerate(selected, 1):
        e["event_id"] = f"{e['station']}_TS_{i:02d}_{e['event_start_utc'].strftime('%Y%m%dT%H')}"
    return selected


def window_for(start, end):
    w0 = start - timedelta(hours=2)
    w1 = end + timedelta(hours=2)
    return w0, w1


def is_clear_span(df, t0, t1):
    sl = df[(df["timestamp_utc"] >= t0) & (df["timestamp_utc"] <= t1)]
    if sl.empty:
        return False
    if not (sl["target_observed"] == 1).all():
        return False
    if (sl["thunderstorm_target"] == 1).any():
        return False
    if ((sl["target_1h"] == 1) & (sl["target_1h_observed"] == 1)).any():
        return False
    return True


def find_control(df, ep):
    start, end = ep["event_start_utc"], ep["event_end_utc"]
    dur = end - start
    pad = timedelta(hours=6)
    for d in CONTROL_OFFSETS_DAYS:
        c0 = start + timedelta(days=d)
        c1 = c0 + dur
        if c0 < CMK_START or c1 > CMK_END:
            continue
        if not is_clear_span(df, c0 - pad, c1 + pad):
            continue
        return {
            "event_start_utc": c0,
            "event_end_utc": c1,
            "n_positive_hours": 0,
            "duration_h": ep["duration_h"],
            "year": c0.year,
            "season": season_of(c0),
            "offset_days": d,
        }
    return None


def expected_slots(w0, w1):
    """INSAT CMK filenames typically use :15 and :45 UTC (Phase 10B-A)."""
    slots = []
    # align to previous :15 or :45 at/before w0
    t = w0.replace(minute=0, second=0, microsecond=0)
    while t < w0:
        t += timedelta(minutes=15)
    # step 15 min then keep 15/45
    cur = t.replace(second=0, microsecond=0)
    while cur <= w1:
        if cur.minute in (15, 45):
            slots.append(cur)
        cur += timedelta(minutes=15)
    return slots


def parse_ident_time(ident: str):
    # 3RIMG_31DEC2025_2345_L2B_CMK_V01R00.h5
    parts = ident.split("_")
    if len(parts) < 3:
        return None
    try:
        dt = datetime.strptime(parts[1] + parts[2], "%d%b%Y%H%M").replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def search_day(day: datetime, cache: dict):
    key = day.strftime("%Y-%m-%d")
    if key in cache:
        return cache[key]
    disk = OUT / "_catalog_day_cache"
    disk.mkdir(parents=True, exist_ok=True)
    fp = disk / f"{key}.json"
    if fp.is_file():
        pack = json.loads(fp.read_text(encoding="utf-8"))
        for e in pack.get("entries", []):
            e["filename_dt"] = parse_ident_time(e.get("filename") or "")
        cache[key] = pack
        return pack
    print("catalog search", key, flush=True)
    params = {"datasetId": DATASET, "startTime": key, "endTime": key}
    recs = []
    status = None
    total = None
    err = None
    try:
        r = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=25)
        status = r.status_code
        body = r.json() if r.status_code == 200 else {}
        total = body.get("totalResults")
        for e in body.get("entries") or []:
            ident = e.get("identifier")
            recs.append(
                {
                    "filename": ident,
                    "gId": e.get("id"),
                    "catalog_timestamp": e.get("updated") or e.get("dcDate"),
                    "filename_timestamp_utc": parse_ident_time(ident).isoformat() if parse_ident_time(ident) else None,
                }
            )
    except Exception as ex:
        err = f"{type(ex).__name__}: {ex}"
        status = status or 0
    pack = {"status": status, "totalResults": total, "entries": recs, "error": err}
    fp.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    pack_rt = json.loads(json.dumps(pack))
    for e in pack_rt.get("entries", []):
        e["filename_dt"] = parse_ident_time(e.get("filename") or "")
    cache[key] = pack_rt
    return pack_rt


def granules_in_window(w0, w1, cache):
    days = []
    d = w0.date()
    while d <= w1.date():
        days.append(datetime(d.year, d.month, d.day, tzinfo=timezone.utc))
        d += timedelta(days=1)
    hits = []
    for day in days:
        pack = search_day(day, cache)
        for e in pack["entries"]:
            ft = e["filename_dt"]
            if ft is None:
                continue
            if w0 <= ft <= w1:
                hits.append(e)
    # unique by filename
    seen = {}
    for h in hits:
        seen[h["filename"]] = h
    return list(seen.values())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {sid: load_station(sid) for sid in STATIONS}
    all_eps = []
    n_pos = {}
    n_eps = {}
    for sid, df in frames.items():
        eps = episodes_from(df, sid)
        n_pos[sid] = int(((df["thunderstorm_target"] == 1) & (df["target_observed"] == 1)).sum())
        n_eps[sid] = len(eps)
        all_eps.extend(eps)

    selected = select_episodes(all_eps)
    candidates = []
    catalog_rows = []
    cache = {}

    for ep in selected:
        sid = ep["station"]
        lat, lon = STATIONS[sid]
        w0, w1 = window_for(ep["event_start_utc"], ep["event_end_utc"])
        candidates.append(
            {
                "station": sid,
                "station_lat": lat,
                "station_lon": lon,
                "event_id": ep["event_id"],
                "event_start_utc": ep["event_start_utc"].isoformat(),
                "event_end_utc": ep["event_end_utc"].isoformat(),
                "case_type": "thunderstorm_episode",
                "control_reference_if_any": "",
                "satellite_window_start_utc": w0.isoformat(),
                "satellite_window_end_utc": w1.isoformat(),
                "selection_reason": (
                    f"METAR thunderstorm_target=1 observed; {ep['n_positive_hours']} positive hours; "
                    f"year={ep['year']} season={ep['season']}"
                ),
                "n_positive_hours": ep["n_positive_hours"],
                "year": ep["year"],
                "season": ep["season"],
            }
        )
        ctrl = find_control(frames[sid], ep)
        if ctrl:
            cid = ep["event_id"].replace("_TS_", "_CTL_")
            cw0, cw1 = window_for(ctrl["event_start_utc"], ctrl["event_end_utc"])
            candidates.append(
                {
                    "station": sid,
                    "station_lat": lat,
                    "station_lon": lon,
                    "event_id": cid,
                    "event_start_utc": ctrl["event_start_utc"].isoformat(),
                    "event_end_utc": ctrl["event_end_utc"].isoformat(),
                    "case_type": "control",
                    "control_reference_if_any": ep["event_id"],
                    "satellite_window_start_utc": cw0.isoformat(),
                    "satellite_window_end_utc": cw1.isoformat(),
                    "selection_reason": (
                        f"Same station; offset {ctrl['offset_days']}d; no thunderstorm_target "
                        f"or target_1h in ±6h pad; all hours observed"
                    ),
                    "n_positive_hours": 0,
                    "year": ctrl["year"],
                    "season": ctrl["season"],
                }
            )

    # catalog search per candidate
    unique_files = {}
    missing_total = 0
    expected_total = 0
    for c in candidates:
        w0 = datetime.fromisoformat(c["satellite_window_start_utc"])
        w1 = datetime.fromisoformat(c["satellite_window_end_utc"])
        expected = expected_slots(w0, w1)
        expected_total += len(expected)
        hits = granules_in_window(w0, w1, cache)
        hit_times = {h["filename_dt"] for h in hits if h["filename_dt"] is not None}
        for h in hits:
            unique_files[h["filename"]] = h
            catalog_rows.append(
                {
                    "station": c["station"],
                    "event_id": c["event_id"],
                    "case_type": c["case_type"],
                    "catalog_timestamp": h["catalog_timestamp"],
                    "filename": h["filename"],
                    "gId": h["gId"],
                    "availability_status": "AVAILABLE",
                }
            )
        for sl in expected:
            if sl not in hit_times:
                missing_total += 1
                catalog_rows.append(
                    {
                        "station": c["station"],
                        "event_id": c["event_id"],
                        "case_type": c["case_type"],
                        "catalog_timestamp": sl.isoformat(),
                        "filename": f"EXPECTED_3RIMG_{sl.strftime('%d%b%Y').upper()}_{sl.strftime('%H%M')}_L2B_CMK",
                        "gId": "",
                        "availability_status": "MISSING_EXPECTED_SCAN",
                    }
                )
        if not hits:
            catalog_rows.append(
                {
                    "station": c["station"],
                    "event_id": c["event_id"],
                    "case_type": c["case_type"],
                    "catalog_timestamp": "",
                    "filename": "",
                    "gId": "",
                    "availability_status": "NO_GRANULES_IN_WINDOW",
                }
            )

    n_ts = sum(1 for c in candidates if c["case_type"] == "thunderstorm_episode")
    n_ctl = sum(1 for c in candidates if c["case_type"] == "control")
    n_dl = len(unique_files)
    est_mb = n_dl * MEAN_MB

    cand_path = OUT / "case_study_candidates.csv"
    fields = [
        "station",
        "station_lat",
        "station_lon",
        "event_id",
        "event_start_utc",
        "event_end_utc",
        "case_type",
        "control_reference_if_any",
        "satellite_window_start_utc",
        "satellite_window_end_utc",
        "selection_reason",
    ]
    with cand_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(candidates)

    aud_path = OUT / "case_study_catalog_audit.csv"
    with aud_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "station",
                "event_id",
                "case_type",
                "catalog_timestamp",
                "filename",
                "gId",
                "availability_status",
            ],
        )
        w.writeheader()
        w.writerows(catalog_rows)

    dist = defaultdict(int)
    for c in candidates:
        if c["case_type"] == "thunderstorm_episode":
            dist[c["station"]] += 1

    alignment = {
        "status": "PROPOSED / PENDING PILOT VALIDATION",
        "rule": (
            "Let T be the atmospheric/model timestamp (hourly, UTC). "
            "A CMK granule may be associated with T only if its filename UTC "
            "(DDMMMYYYY_HHMM) falls in [T, T+1h). Do not use acquisition end "
            "as T. If later HDF attributes give Acquisition_Start/End, require "
            "the acquisition interval to overlap [T, T+1h) and still record "
            "start, end, and filename time separately. Never copy one granule "
            "into other hours. Hours with no overlapping granule are NOT_OBSERVED."
        ),
        "rationale": (
            "Phase 10A pilot 0015 file acquired 00:15:19–00:42:13 UTC; filename "
            "slot is scan start, not an instant. Alignment needs HDF-level "
            "validation on a downloaded case before freezing the contract."
        ),
    }

    summary = {
        "phase": "10B-B",
        "search_only": True,
        "downloaded_granules": 0,
        "label_source": "dataset/multilocation/targets/*_nowcast_targets_2014_2025.csv",
        "positive_hours_2016_10_11_2025_12_31": n_pos,
        "episodes_all": n_eps,
        "n_thunderstorm_episodes_selected": n_ts,
        "n_controls_selected": n_ctl,
        "distribution_episodes_by_station": dict(dist),
        "unique_cmk_files_to_download_later": n_dl,
        "expected_15_45_slots_across_windows": expected_total,
        "missing_expected_scans_rows": missing_total,
        "estimated_storage_mb": est_mb,
        "estimated_storage_gb": est_mb / 1024.0,
        "storage_estimate_basis": "Phase 10B-A mean file size totalSizeMB/totalResults; labeled estimate",
        "sample_practical_to_download": n_dl < 500 and est_mb < 5000,
        "temporal_alignment": alignment,
        "extraction_contract_future": {
            "cloud_mask_at_station": "nearest valid CMK pixel (Phase 10A)",
            "nearby_cloud_fraction": "valid pixels within radius; Cloudy+Probably_Cloudy / valid",
            "clear_cloudy_indicator": "map flags 0/2 vs 1/3; do not hide Probably_*",
            "missing": "NOT_OBSERVED never 0",
            "not_computed_this_phase": True,
        },
        "unresolved_before_extraction": [
            "No historical HDF files downloaded.",
            "Acquisition interval vs filename slot not validated on case-study files.",
            "MOSDAC login required to download.",
            "Duplicate near-slots (e.g. 2344 vs 2345) need a keep-one rule after inspection.",
        ],
        "classification": "CASE_STUDY_SAMPLE_DESIGNED / DOWNLOAD_NOT_PERFORMED",
    }
    with (OUT / "case_study_sampling_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    write_doc(summary, candidates)
    print("episodes", n_ts, "controls", n_ctl, "unique files", n_dl)
    print("missing expected", missing_total, "est MB", round(est_mb, 1))
    print("dist", dict(dist))


def write_doc(s, candidates):
    lines = []
    a = lines.append
    a("# Phase 10B-B — INSAT-3DR CMK case-study sampling plan")
    a("")
    a("**Date:** 2026-09-22  ")
    a("**Search + planning only.** No granule download. No MOSDAC credentials. No training. No Model B edits. No commit.")
    a("")
    a("## 1. Objective")
    a("")
    a("Design a **manageable, METAR-grounded** satellite case-study sample for five ThunderWatch stations using catalog `3RIMG_L2B_CMK` (available from 2016-10-11). Full archive (~1.365 TiB / 153,845 files) is **not** in scope.")
    a("")
    a("## 2. Label source")
    a("")
    a("Genuine nowcast tables: `dataset/multilocation/targets/{station}_nowcast_targets_2014_2025.csv`.")
    a("Positive hour = `thunderstorm_target==1` **and** `target_observed==1`. Window: 2016-10-11 through 2025-12-31.")
    a("")
    a(f"Positive hours by station: `{s['positive_hours_2016_10_11_2025_12_31']}`")
    a(f"All clustered episodes (gap ≤ {GAP_HOURS} h): `{s['episodes_all']}`")
    a("")
    a("## 3. Thunderstorm episodes selected")
    a("")
    a(f"**{s['n_thunderstorm_episodes_selected']}** episodes.")
    a(f"Distribution: `{s['distribution_episodes_by_station']}`")
    a("")
    a("Selection: greedy per station, prefer longer observed events, cap 24 h duration, diversify year×season (winter / pre-monsoon / monsoon / post-monsoon). Not invented.")
    a("")
    a("## 4. Controls")
    a("")
    a(f"**{s['n_controls_selected']}** control periods (same station; offsets among {list(CONTROL_OFFSETS_DAYS)} days; ±6 h pad with all hours observed and no `thunderstorm_target` or observed `target_1h==1`). Controls omitted if no such span exists.")
    a("")
    a("## 5. Satellite window")
    a("")
    a("Each case uses **T_start−2 h through T_end+2 h** (UTC). Catalog queried **by calendar day**, granules kept if **filename UTC** falls in the window. Expected cadence for gap counting: **:15 and :45** slots (from Phase 10B-A names). Missing expected slots are `MISSING_EXPECTED_SCAN`, not zeros.")
    a("")
    a("## 6. Catalog search results")
    a("")
    a(f"- Unique CMK files that would later be downloaded: **{s['unique_cmk_files_to_download_later']}**")
    a(f"- Expected :15/:45 slots across windows (with overlap counted per window): **{s['expected_15_45_slots_across_windows']}**")
    a(f"- Missing expected scan rows: **{s['missing_expected_scans_rows']}**")
    a(f"- Estimated storage: **{s['estimated_storage_mb']:.1f} MB** (~{s['estimated_storage_gb']:.2f} GB) — {s['storage_estimate_basis']}")
    a(f"- Practical to download later: **{s['sample_practical_to_download']}**")
    a("")
    a("No granules were downloaded.")
    a("")
    a("## 7. Future extraction contract (not computed)")
    a("")
    a("- `cloud_mask_at_station` — nearest valid CMK pixel (Phase 10A flags 0–3)")
    a("- `nearby_cloud_fraction` — Cloudy+Probably_Cloudy among valid pixels in a radius")
    a("- `clear_cloudy_indicator` — 0/2 vs 1/3 with `Probably_*` documented")
    a("- Unavailable scan → **`NOT_OBSERVED`**, never 0-fill, never broadcast one scan across hours")
    a("")
    a("## 8. Temporal alignment rule")
    a("")
    a(f"**Status:** {s['temporal_alignment']['status']}")
    a("")
    a(s["temporal_alignment"]["rule"])
    a("")
    a(s["temporal_alignment"]["rationale"])
    a("")
    a("## 9. Unresolved before extraction")
    a("")
    for u in s["unresolved_before_extraction"]:
        a(f"- {u}")
    a("")
    a("## 10. Classification")
    a("")
    a(f"**`{s['classification']}`**")
    a("")
    a("This sample is **not** model validation and must not be used to claim satellite skill or to retrain Model B.")
    a("")
    a("PHASE 10B-B COMPLETE — CASE-STUDY SAMPLE DESIGNED")
    a("")
    DOC.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
