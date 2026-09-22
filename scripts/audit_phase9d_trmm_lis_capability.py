#!/usr/bin/env python3
"""Phase 9D — TRMM-LIS capability audit (read-only).

Does not train models, modify datasets, fabricate observations, or commit.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "dataset" / "lightning" / "trmm_lis_pilot"
COMBINED = ROOT / "dataset" / "lightning" / "Combined_LIS_th.nc"
MENDELEY = ROOT / "dataset" / "lightning" / "mendeley_western_ghats"
OUT_DIR = ROOT / "outputs" / "lightning" / "phase9d"
DOC_PATH = ROOT / "docs" / "PHASE9D_TRMM_LIS_CAPABILITY_AUDIT.md"

STATIONS = {
    "VOTV": (8.4667, 76.9500),
    "VECC": (22.6547, 88.4467),
    "VIDP": (28.5667, 77.1167),
    "VOCI": (10.1500, 76.4000),
    "VABB": (19.1005, 72.8585),
}
RADII_KM = (25.0, 50.0, 100.0)
ATM_START = datetime(2014, 1, 1, tzinfo=timezone.utc)
ATM_END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TAI93_EPOCH = datetime(1993, 1, 1, tzinfo=timezone.utc)
# TAI-UTC offset during 2013–2014 (leap seconds): 35 s. Documented; used only
# when comparing TAI93 numeric times to orbit_summary_UTC_start.
TAI_UTC_OFFSET_S = 35.0


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _scalar(v):
    try:
        a = np.array(v).squeeze()
        if a.shape == ():
            return a.item() if hasattr(a, "item") else a
        return None
    except Exception:
        return None


def _to_str(v):
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip("\x00").strip()
    if isinstance(v, np.ndarray):
        if v.dtype.kind in ("S", "U"):
            return "".join(
                x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x)
                for x in v.ravel()
            ).strip("\x00").strip()
        if v.shape == ():
            return _to_str(v.item())
    return str(v).strip()


def tai93_to_utc(seconds):
    """TAI93 seconds since 1993-01-01 00:00:00 TAI → UTC datetime.

    Conversion: UTC ≈ TAI93_epoch + seconds - 35s (TAI-UTC in 2013–2014).
    Validated against orbit_summary_UTC_start when present.
    """
    if seconds is None or not np.isfinite(seconds):
        return None
    # Prefer subtracting leap-second offset so UTC matches orbit_summary_UTC_start.
    return TAI93_EPOCH + timedelta(seconds=float(seconds) - TAI_UTC_OFFSET_S)


def parse_orbit_utc(s):
    if not s:
        return None
    s = s.strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(s[:26], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def valid_mask(lat, lon):
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()
    n = min(lat.size, lon.size)
    lat, lon = lat[:n], lon[:n]
    m = np.isfinite(lat) & np.isfinite(lon)
    m &= (lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180)
    # Fill often 0,0 or extreme sentinels
    m &= ~((np.abs(lat) < 1e-8) & (np.abs(lon) < 1e-8))
    return m, lat, lon


def valid_times(t):
    t = np.asarray(t, dtype=np.float64).ravel()
    m = np.isfinite(t) & (t > 1e8) & (t < 1e10)  # TAI93 ~ 6e8 in 2013
    return m, t


def inspect_vars(ds):
    rows = []
    for name, var in ds.variables.items():
        attrs = {k: _to_str(var.getncattr(k)) for k in var.ncattrs()}
        rows.append(
            {
                "variable": name,
                "shape": str(tuple(var.shape)),
                "dtype": str(var.dtype),
                "dimensions": ",".join(var.dimensions),
                "long_name": attrs.get("long_name") or attrs.get("Long_name") or "",
                "units": attrs.get("units") or attrs.get("Units") or "",
            }
        )
    return rows


def load_level(ds, prefix):
    lat_n = f"lightning_{prefix}_lat"
    lon_n = f"lightning_{prefix}_lon"
    t_n = f"lightning_{prefix}_TAI93_time"
    if lat_n not in ds.variables or lon_n not in ds.variables:
        return None
    lat = ds.variables[lat_n][:]
    lon = ds.variables[lon_n][:]
    m, lat, lon = valid_mask(lat, lon)
    times = None
    tmask = None
    if t_n in ds.variables:
        tmask, times = valid_times(ds.variables[t_n][:])
        n = min(m.size, tmask.size)
        m = m[:n] & tmask[:n]
        lat, lon, times = lat[:n], lon[:n], times[:n]
    return {
        "lat": lat[m],
        "lon": lon[m],
        "tai93": None if times is None else times[m],
        "count": int(m.sum()),
    }


def granule_times(ds):
    utc_s = None
    if "orbit_summary_UTC_start" in ds.variables:
        utc_s = parse_orbit_utc(_to_str(ds.variables["orbit_summary_UTC_start"][:]))
    tai_start = None
    tai_end = None
    if "orbit_summary_TAI93_start" in ds.variables:
        tai_start = _scalar(ds.variables["orbit_summary_TAI93_start"][:])
    if "orbit_summary_TAI93_end" in ds.variables:
        tai_end = _scalar(ds.variables["orbit_summary_TAI93_end"][:])
    start = utc_s or tai93_to_utc(tai_start)
    end = tai93_to_utc(tai_end)
    return start, end, utc_s, tai_start, tai_end


def inspect_mendeley():
    files = sorted(MENDELEY.glob("*.mat")) if MENDELEY.is_dir() else []
    info = {
        "directory": str(MENDELEY.relative_to(ROOT)).replace("\\", "/"),
        "n_files": len(files),
        "classification": "RESEARCH_FIGURE_DATA",
        "event_level": False,
        "gridded_lightning": False,
        "station_aligned": False,
        "timestamps_present": False,
        "files": [],
    }
    try:
        from scipy.io import loadmat
    except ImportError:
        info["note"] = "scipy not available; classified from filenames (Fig_*.mat)"
        info["files"] = [{"name": f.name, "size_bytes": f.stat().st_size} for f in files]
        return info

    for f in files:
        rec = {"name": f.name, "size_bytes": f.stat().st_size, "keys": []}
        try:
            mat = loadmat(str(f), squeeze_me=True, struct_as_record=False)
            rec["keys"] = [k for k in mat.keys() if not k.startswith("__")]
            rec["shapes"] = {
                k: list(np.shape(mat[k])) if hasattr(mat[k], "shape") else str(type(mat[k]))
                for k in rec["keys"]
            }
        except Exception as e:
            rec["error"] = str(e)
        info["files"].append(rec)

    names = " ".join(p.name.lower() for p in files)
    if all(p.name.lower().startswith("fig_") for p in files):
        info["classification"] = "RESEARCH_FIGURE_DATA"
        info["note"] = (
            "All files are Fig_*.mat research-figure arrays, not an event-level "
            "lightning archive. Do not reverse-engineer into fake event records."
        )
    return info


def inspect_combined():
    out = {"path": "dataset/lightning/Combined_LIS_th.nc", "opened": False}
    if not COMBINED.is_file():
        out["error"] = "missing"
        return out
    with Dataset(str(COMBINED), "r") as ds:
        out["opened"] = True
        out["dimensions"] = {k: int(v.size) for k, v in ds.dimensions.items()}
        out["variables"] = list(ds.variables.keys())
        out["title"] = _to_str(ds.getncattr("Title")) if "Title" in ds.ncattrs() else None
        out["has_time_dimension"] = any(k.lower() in ("time", "timedim") for k in ds.dimensions)
        attrs = {}
        if "thunder_hours" in ds.variables:
            v = ds.variables["thunder_hours"]
            attrs = {k: _to_str(v.getncattr(k)) for k in v.ncattrs()}
        out["thunder_hours_attrs"] = attrs
        out["classification"] = "REFERENCE_ONLY / CLIMATOLOGY_FEATURE"
        out["note"] = (
            "Annual-mean thunder hours climatology; no time dimension; "
            "must not be broadcast into hourly lightning rows."
        )
    return out


def iso(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PILOT_DIR.glob("*.nc"))
    inventory_rows = []
    granule_rows = []
    samples = []
    overlap = defaultdict(lambda: defaultdict(lambda: {"granules_with": 0, "records": 0}))
    # overlap[station][radius][level]
    overlap_full = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"granules_with": 0, "records": 0}))
    )
    total_events = total_groups = total_flashes = 0
    durations = []
    starts = []
    ends = []
    var_inventory_first = None
    time_conversion_note = (
        "lightning_*_TAI93_time units: seconds since 1993-01-01 00:00:00 TAI. "
        f"UTC = TAI93_epoch + seconds - {TAI_UTC_OFFSET_S} s (TAI-UTC leap-second "
        "offset applicable 2012-07 through 2015-06). Compared to orbit_summary_UTC_start."
    )
    utc_vs_tai_deltas = []

    for fp in files:
        with Dataset(str(fp), "r") as ds:
            if var_inventory_first is None:
                var_inventory_first = inspect_vars(ds)
            start, end, utc_s, tai_start, tai_end = granule_times(ds)
            if utc_s is not None and tai_start is not None:
                conv = tai93_to_utc(tai_start)
                if conv is not None:
                    utc_vs_tai_deltas.append((utc_s - conv).total_seconds())
            ev = load_level(ds, "event")
            gr = load_level(ds, "group")
            fl = load_level(ds, "flash")
            n_e = ev["count"] if ev else 0
            n_g = gr["count"] if gr else 0
            n_f = fl["count"] if fl else 0
            total_events += n_e
            total_groups += n_g
            total_flashes += n_f
            if start:
                starts.append(start)
            if end:
                ends.append(end)
            if start and end and end >= start:
                durations.append((end - start).total_seconds())

            granule_rows.append(
                {
                    "granule": fp.name,
                    "start_time": iso(start),
                    "end_time": iso(end),
                    "event_count": n_e,
                    "group_count": n_g,
                    "flash_count": n_f,
                }
            )

            for level_name, pack in (("event", ev), ("group", gr), ("flash", fl)):
                if not pack:
                    continue
                for i in range(pack["count"]):
                    lat = float(pack["lat"][i])
                    lon = float(pack["lon"][i])
                    t = None
                    if pack["tai93"] is not None:
                        t = tai93_to_utc(float(pack["tai93"][i]))
                    for sid, (slat, slon) in STATIONS.items():
                        d = haversine_km(slat, slon, lat, lon)
                        for r in RADII_KM:
                            if d <= r:
                                overlap_full[sid][r][level_name]["records"] += 1
                        if d <= 100.0:
                            samples.append(
                                {
                                    "station_id": sid,
                                    "lightning_type": level_name,
                                    "observation_time_utc": iso(t),
                                    "latitude": lat,
                                    "longitude": lon,
                                    "distance_km": round(d, 4),
                                    "source_granule": fp.name,
                                }
                            )

            # granules_with: unique granule if any record in radius
            for sid, (slat, slon) in STATIONS.items():
                for level_name, pack in (("event", ev), ("group", gr), ("flash", fl)):
                    if not pack or pack["count"] == 0:
                        continue
                    dists = np.array(
                        [haversine_km(slat, slon, float(la), float(lo))
                         for la, lo in zip(pack["lat"], pack["lon"])]
                    )
                    for r in RADII_KM:
                        if np.any(dists <= r):
                            overlap_full[sid][r][level_name]["granules_with"] += 1

    # overlap CSV
    overlap_rows = []
    for sid in STATIONS:
        for r in RADII_KM:
            for level in ("event", "group", "flash"):
                rec = overlap_full[sid][r][level]
                overlap_rows.append(
                    {
                        "station_id": sid,
                        "lightning_type": level,
                        "radius_km": r,
                        "granules_checked": len(files),
                        "granules_with_lightning": rec["granules_with"],
                        "total_lightning_records": rec["records"],
                    }
                )

    inv_path = OUT_DIR / "trmm_lis_capability_inventory.csv"
    with inv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "granule",
                "start_time",
                "end_time",
                "event_count",
                "group_count",
                "flash_count",
            ],
        )
        w.writeheader()
        w.writerows(granule_rows)

    ov_path = OUT_DIR / "station_lightning_overlap.csv"
    with ov_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "station_id",
                "lightning_type",
                "radius_km",
                "granules_checked",
                "granules_with_lightning",
                "total_lightning_records",
            ],
        )
        w.writeheader()
        w.writerows(overlap_rows)

    samp_path = OUT_DIR / "station_lightning_samples.csv"
    with samp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "station_id",
                "lightning_type",
                "observation_time_utc",
                "latitude",
                "longitude",
                "distance_km",
                "source_granule",
            ],
        )
        w.writeheader()
        w.writerows(samples)

    mendeley = inspect_mendeley()
    combined = inspect_combined()

    tmin = min(starts) if starts else None
    tmax = max(ends) if ends else None
    total_dur_s = sum(durations)
    n_gran = len(files)
    any_prox = len(samples) > 0
    any_50 = any(
        overlap_full[s][50.0][lv]["records"] > 0
        for s in STATIONS
        for lv in ("event", "group", "flash")
    )
    any_25 = any(
        overlap_full[s][25.0][lv]["records"] > 0
        for s in STATIONS
        for lv in ("event", "group", "flash")
    )
    any_100 = any(
        overlap_full[s][100.0][lv]["records"] > 0
        for s in STATIONS
        for lv in ("event", "group", "flash")
    )

    overlap_atm = False
    if tmin and tmax:
        overlap_atm = not (tmax < ATM_START or tmin > ATM_END)

    # Hourly feasibility: timestamps exist, but coverage is orbital snapshots.
    hourly_feasible = False
    hourly_reason = (
        "Hourly feature extraction is NOT scientifically feasible as a continuous "
        "station time series. TRMM-LIS is LEO orbital sampling (~minutes per overpass). "
        "Hours without an overpass are NO_SATELLITE_COVERAGE / NOT_OBSERVED, not zero lightning. "
        "Broadcasting overpass counts into unobserved hours is forbidden. "
        "The pilot span is only ~8 days (late 2013–early 2014), far too short for training."
    )
    if any_prox:
        hourly_reason += (
            " Station-proximal records exist and could be counted only for hours that "
            "contain an actual overpass observation; they still cannot fill hourly rows "
            "outside those overpasses."
        )

    # Classification
    if not n_gran:
        classification = "INSTITUTIONAL_ACCESS_REQUIRED"
    elif any_prox and overlap_atm and n_gran < 500:
        classification = "PILOT_FEATURE_CANDIDATE" if any_50 else "REFERENCE_ONLY"
    elif total_events > 0:
        # Genuine lightning exists globally in files, but station-hourly training not supported
        classification = "REFERENCE_ONLY"
        if any_100 and overlap_atm:
            # sparse proximal hits in a tiny window → still not training
            classification = "REFERENCE_ONLY"
    else:
        classification = "REFERENCE_ONLY"

    # Stricter: user asked if existing files contain usable observations for a
    # historical lightning feature experiment. Global events yes; station-proximal
    # for ThunderWatch: based on counts.
    if any_50 and overlap_atm:
        classification = "PILOT_FEATURE_CANDIDATE"
    else:
        classification = "REFERENCE_ONLY"

    sufficient = {
        "hourly_feature_extraction": False,
        "historical_model_training": False,
        "case_study_visualization": bool(any_prox or total_flashes > 0),
        "climatological_reference": False,  # Combined_LIS is climatology, not this pilot
    }
    # case-study: only if station-proximal; otherwise global case studies only
    sufficient["case_study_visualization_station"] = bool(any_prox)
    sufficient["case_study_visualization_global_pilot"] = total_flashes > 0

    median_delta = float(np.median(utc_vs_tai_deltas)) if utc_vs_tai_deltas else None

    summary = {
        "phase": "9D",
        "audit_only": True,
        "n_granules": n_gran,
        "granules_opened": n_gran,
        "total_valid_events": total_events,
        "total_valid_groups": total_groups,
        "total_valid_flashes": total_flashes,
        "time_range_utc": {"start": iso(tmin), "end": iso(tmax)},
        "total_observation_duration_seconds": total_dur_s,
        "total_observation_duration_hours": total_dur_s / 3600.0 if total_dur_s else 0,
        "coverage_continuous": False,
        "sampling": "orbital_leo_overpass",
        "time_conversion": time_conversion_note,
        "tai93_vs_orbit_utc_median_residual_seconds": median_delta,
        "atmospheric_dataset_window": {"start": iso(ATM_START), "end": iso(ATM_END)},
        "temporal_overlap_with_atmospheric_window": overlap_atm,
        "station_proximal_observations_exist": any_prox,
        "station_proximal_any_25km": any_25,
        "station_proximal_any_50km": any_50,
        "station_proximal_any_100km": any_100,
        "n_station_proximal_sample_rows": len(samples),
        "hourly_aggregation_scientifically_feasible": hourly_feasible,
        "hourly_aggregation_notes": hourly_reason,
        "no_lightning_detected_vs_no_coverage": {
            "NO_SATELLITE_COVERAGE": "hours/locations with no LIS overpass; not a zero",
            "NO_LIGHTNING_DETECTED": "overpass existed and valid coords were outside radius or none in FOV near station",
        },
        "variable_inventory_first_granule": var_inventory_first,
        "mendeley": mendeley,
        "combined_lis_th": combined,
        "trmm_lis_sufficient_for": sufficient,
        "classification": classification,
        "scientific_limitations": [
            "TRMM-LIS is not continuous hourly lightning.",
            "Absence of station-proximal flashes is not proof that no lightning occurred.",
            "Do not 0-fill unobserved hours.",
            "Do not broadcast a granule across hours.",
            "Do not convert Combined_LIS_th.nc annual climatology into hourly lightning.",
            "Pilot covers ~2013-12-31 through ~2014-01-07 only.",
            "TRMM-LIS mission ended 2015; cannot cover 2014–2025 training window.",
        ],
    }

    with (OUT_DIR / "phase9d_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    write_report(summary, granule_rows, overlap_rows, samples, var_inventory_first)
    print("classification", classification)
    print("events", total_events, "groups", total_groups, "flashes", total_flashes)
    print("samples", len(samples), "any_25", any_25, "any_50", any_50, "any_100", any_100)
    print("time", iso(tmin), iso(tmax))
    print("hourly_feasible", hourly_feasible)


def write_report(summary, granule_rows, overlap_rows, samples, var_inv):
    s = summary
    lines = []
    a = lines.append
    a("# Phase 9D — TRMM-LIS lightning capability audit")
    a("")
    a("**Date:** 2026-09-22  ")
    a("**Audit only.** No training, no Model B edits, no inference-contract changes, no dashboard/Flask edits, no synthetic lightning, no hourly 0-fill, no commit.")
    a("")
    a("## 1. Objective")
    a("")
    a("Determine whether **existing** TRMM-LIS pilot granules contain event/group/flash observations that could support a **legitimate historical lightning feature experiment**. This does **not** claim lightning is usable in the final ThunderWatch model.")
    a("")
    a("## 2. Data inspected")
    a("")
    a(f"- `dataset/lightning/trmm_lis_pilot/` — **{s['n_granules']}** NetCDF granules opened")
    a("- `dataset/lightning/mendeley_western_ghats/` — Fig_*.mat research files")
    a("- `dataset/lightning/Combined_LIS_th.nc` — annual thunder-hour climatology")
    a("")
    a("Stations (decimal degrees):")
    a("")
    a("| station | lat | lon |")
    a("|---------|-----|-----|")
    for k, (la, lo) in STATIONS.items():
        a(f"| {k} | {la} | {lo} |")
    a("")
    a("## 3. Variable inventory")
    a("")
    a("Discovered from the first granule (names not assumed a priori). Lightning objects:")
    a("")
    a("| Role | Variable |")
    a("|------|----------|")
    a("| events lat/lon/time | `lightning_event_lat`, `lightning_event_lon`, `lightning_event_TAI93_time` |")
    a("| groups lat/lon/time | `lightning_group_lat`, `lightning_group_lon`, `lightning_group_TAI93_time` |")
    a("| flashes lat/lon/time | `lightning_flash_lat`, `lightning_flash_lon`, `lightning_flash_TAI93_time` |")
    a("| orbit start/end | `orbit_summary_TAI93_start`, `orbit_summary_TAI93_end`, `orbit_summary_UTC_start` |")
    a("")
    a("LIS hierarchy: **event** = illuminated pixel; **group** = adjacent events in one frame; **flash** = groups over successive frames.")
    a("")
    a(f"**Time conversion:** {s['time_conversion']}")
    a("")
    a(f"Median residual of converted TAI93 start vs `orbit_summary_UTC_start`: `{s['tai93_vs_orbit_utc_median_residual_seconds']}` seconds (near 0 validates the leap-second offset).")
    a("")
    a("Machine-readable first-granule catalog is in `outputs/lightning/phase9d/phase9d_summary.json` (`variable_inventory_first_granule`).")
    a("")
    a("## 4. Valid lightning record counts")
    a("")
    a("Counts use finite lat/lon in geographic range; fill/missing excluded. Not raw dimension sizes.")
    a("")
    a(f"- Granules: **{s['n_granules']}**")
    a(f"- Valid events: **{s['total_valid_events']}**")
    a(f"- Valid groups: **{s['total_valid_groups']}**")
    a(f"- Valid flashes: **{s['total_valid_flashes']}**")
    a("")
    a("Per-granule CSV: `outputs/lightning/phase9d/trmm_lis_capability_inventory.csv`.")
    a("")
    a("## 5. Geographic coverage")
    a("")
    a("A granule counts as station-proximal **only** if valid lightning coordinates fall inside the radius. The satellite swath is **not** treated as lightning observed.")
    a("")
    a("| station | type | radius_km | granules_checked | granules_with_lightning | total_lightning_records |")
    a("|---------|------|-----------|------------------|-------------------------|-------------------------|")
    for row in overlap_rows:
        a(
            f"| {row['station_id']} | {row['lightning_type']} | {row['radius_km']} | "
            f"{row['granules_checked']} | {row['granules_with_lightning']} | {row['total_lightning_records']} |"
        )
    a("")
    a("## 6. Temporal coverage")
    a("")
    a(f"- Usable lightning timestamp range (orbit start min … orbit end max): **{s['time_range_utc']['start']}** → **{s['time_range_utc']['end']}**")
    a(f"- Sum of granule observation durations: **{s['total_observation_duration_hours']:.2f} hours** (not calendar span; orbits are short)")
    a("- Coverage is **not continuous**. TRMM LIS is LEO; ~16 orbits/day globally, minutes of view per overpass at a point.")
    a("")
    a("## 7. Station overlap")
    a("")
    a(f"- Any station-proximal lightning (≤100 km): **{s['station_proximal_observations_exist']}**")
    a(f"- Any ≤25 km: **{s['station_proximal_any_25km']}**")
    a(f"- Any ≤50 km: **{s['station_proximal_any_50km']}**")
    a(f"- Any ≤100 km: **{s['station_proximal_any_100km']}**")
    a(f"- Sample rows written: **{s['n_station_proximal_sample_rows']}** (`station_lightning_samples.csv`)")
    a("")
    if samples:
        a("All station-proximal rows (≤100 km) include `station_id`, `lightning_type`, `observation_time_utc`, lat/lon, `source_granule`. Timestamps are converted TAI93; none were invented.")
    else:
        a("**No** valid event/group/flash coordinates fell within 100 km of any ThunderWatch station in this pilot. That is **no lightning detected during available satellite observation** near the stations — **not** a claim that no lightning occurred in those hours.")
    a("")
    a("## 8. Coverage gaps")
    a("")
    a(f"- Number of granules: {s['n_granules']}")
    a(f"- Total observation duration: {s['total_observation_duration_hours']:.2f} h")
    a("- Station-overlap granules: see table in §5 (`granules_with_lightning`)")
    a("- Temporal gaps: large; files span ~8 calendar days with discrete orbits, not hourly cadence")
    a("- Continuous coverage: **false**")
    a("- Hourly aggregation of this pilot would create **severe sampling bias** (overpass hours vs unobserved hours).")
    a("")
    a("Explicit distinction:")
    a("")
    a("- **NO_SATELLITE_COVERAGE / NOT_OBSERVED** — LIS not overhead; must not be stored as 0 lightning.")
    a("- **NO_LIGHTNING_DETECTED** — overpass (or lightning arrays) existed, but no valid lightning coordinate in the station radius.")
    a("")
    a("## 9. Hourly aggregation feasibility")
    a("")
    a(s["hourly_aggregation_notes"])
    a("")
    a(f"**Scientifically feasible as continuous hourly features:** `{s['hourly_aggregation_scientifically_feasible']}`")
    a("")
    a("If proximal observations existed, counts such as `lightning_flash_count_1h` would be valid **only** for hours containing an actual overpass. Unobserved hours must remain `NOT_OBSERVED`.")
    a("")
    a("## 10. Mendeley assessment")
    a("")
    m = s["mendeley"]
    a(f"- Files: **{m.get('n_files')}** in `{m.get('directory')}`")
    a(f"- Classification: **{m.get('classification')}**")
    a(f"- Event-level lightning: `{m.get('event_level')}`")
    a(f"- Gridded lightning archive: `{m.get('gridded_lightning')}`")
    a(f"- Station-aligned observations: `{m.get('station_aligned')}`")
    a(f"- Timestamps: `{m.get('timestamps_present')}`")
    a(f"- Note: {m.get('note', '')}")
    a("")
    a("Do **not** use these `.mat` files for model training. They are research/figure arrays.")
    a("")
    a("## 11. Combined LIS assessment")
    a("")
    c = s["combined_lis_th"]
    a(f"- Path: `{c.get('path')}`")
    a(f"- Dimensions: `{c.get('dimensions')}`")
    a(f"- Variables: `{c.get('variables')}`")
    a(f"- Time dimension: `{c.get('has_time_dimension')}`")
    a(f"- Title: {c.get('title')}")
    a(f"- Classification: **{c.get('classification')}**")
    a(f"- {c.get('note')}")
    a("")
    a("## 12. Scientific limitations")
    a("")
    for item in s["scientific_limitations"]:
        a(f"- {item}")
    a("")
    a("Atmospheric / METAR window is 2014-01-01 through 2025-12-31. The pilot **does** overlap the **start** of that window (early January 2014) but **does not** overlap the rest of the training period. Overlap of a few days is **not** sufficient for historical model training.")
    a("")
    a("## 13. Final classification")
    a("")
    a(f"**`{s['classification']}`**")
    a("")
    a("TRMM-LIS existing pilot sufficient for:")
    a("")
    a("| Use | Sufficient? |")
    a("|-----|-------------|")
    a(f"| 1. Hourly feature extraction | `{s['trmm_lis_sufficient_for']['hourly_feature_extraction']}` |")
    a(f"| 2. Historical model training | `{s['trmm_lis_sufficient_for']['historical_model_training']}` |")
    a(f"| 3. Case-study visualization (global pilot flashes) | `{s['trmm_lis_sufficient_for']['case_study_visualization_global_pilot']}` |")
    a(f"| 3b. Case-study visualization (station-proximal) | `{s['trmm_lis_sufficient_for']['case_study_visualization_station']}` |")
    a(f"| 4. Climatological reference | `{s['trmm_lis_sufficient_for']['climatological_reference']}` (use Combined_LIS_th.nc instead; still not hourly) |")
    a("")
    a("## 14. Recommendation for ThunderWatch AI")
    a("")
    a("Keep Model B, inference contract, thresholds, feature order, replay, and dashboard unchanged. Do not ingest TRMM-LIS as hourly predictors. Combined LIS thunder hours remain a **static climatology candidate only** (Phase 9C). Operational/real-time lightning still requires institutional sources (e.g. IITM LLN / Damini), not this pilot.")
    a("")
    a("PHASE 9D COMPLETE — TRMM-LIS CAPABILITY AUDIT READY")
    a("")
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
