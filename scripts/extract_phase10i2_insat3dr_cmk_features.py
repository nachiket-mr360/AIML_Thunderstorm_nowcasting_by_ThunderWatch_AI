#!/usr/bin/env python3
"""Phase 10I.2 — scan-level INSAT-3DR CMK features (119 files × 5 stations).

No download, no hourly join, no NWP edits, no training.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INV = ROOT / "outputs/satellite/phase10h/phase10h_file_inventory.csv"
MANIFEST = ROOT / "outputs/satellite/phase10b/cmk_mvp_download_manifest.csv"
DATA = ROOT / "dataset/satellite/insat3dr_cmk"
OUT_DIR = ROOT / "dataset/satellite/insat3dr_cmk_features"
OUT_CSV = OUT_DIR / "cmk_scan_level_features.csv"
OUT_JSON = ROOT / "outputs/satellite/phase10i2/phase10i2_extraction_summary.json"
OUT_DOC = ROOT / "docs/PHASE10I2_INSAT3DR_FEATURE_EXTRACTION_REPORT.md"
SKIP_GID = "7680747"

STATIONS = {
    "VOTV": (8.4667, 76.9500),
    "VECC": (22.6547, 88.4467),
    "VIDP": (28.5667, 77.1167),
    "VOCI": (10.1500, 76.4000),
    "VABB": (19.1005, 72.8585),
}
RADIUS_KM = 25.0
VALID_CMK = np.array([0, 1, 2, 3], dtype=np.int16)
CLOUDY = np.array([1, 3], dtype=np.int16)
CLEAR = np.array([0, 2], dtype=np.int16)
CLASS_NAME = {0: "Clear", 1: "Cloudy", 2: "Probably_Clear", 3: "Probably_Cloudy"}
ACQ_FMT = "%d-%b-%YT%H:%M:%S"


def decode_attr(v):
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip("\x00").strip()
    if isinstance(v, np.bytes_):
        return bytes(v).decode("utf-8", errors="replace").strip("\x00").strip()
    if isinstance(v, np.ndarray) and v.size == 1:
        return decode_attr(v.reshape(-1)[0])
    if hasattr(v, "item") and np.ndim(v) == 0:
        return decode_attr(v.item())
    s = str(v).strip()
    return s or None


def parse_acq(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), ACQ_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def iso_z(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def decode_time_ds(ds) -> datetime | None:
    raw = float(np.array(ds[:]).reshape(-1)[0])
    units = decode_attr(ds.attrs.get("units")) or ""
    if units.startswith("minutes since "):
        origin = units[len("minutes since ") :].strip()
        origin_dt = datetime.strptime(origin, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return origin_dt + timedelta(minutes=raw)
    return None


def decode_geo(ds) -> np.ndarray:
    raw = np.asarray(ds[:], dtype=np.float64)
    fill = ds.attrs.get("_FillValue")
    if fill is not None:
        fv = float(np.array(fill).reshape(-1)[0])
        raw = np.where(raw == fv, np.nan, raw)
    scale = float(np.array(ds.attrs.get("scale_factor", 1.0)).reshape(-1)[0])
    offset = float(np.array(ds.attrs.get("add_offset", 0.0)).reshape(-1)[0])
    return raw * scale + offset


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def extract_file(path: Path, gid: str) -> list[dict]:
    rows = []
    with h5py.File(path, "r") as h5:
        cmk = np.asarray(h5["CMK"][:]).squeeze()
        lat = decode_geo(h5["Latitude"])
        lon = decode_geo(h5["Longitude"])
        t = decode_time_ds(h5["time"]) if "time" in h5 else None
        start = parse_acq(decode_attr(h5.attrs.get("Acquisition_Start_Time")))
        end = parse_acq(decode_attr(h5.attrs.get("Acquisition_End_Time")))
        fill = int(np.array(h5["CMK"].attrs.get("_FillValue", 9)).reshape(-1)[0])

        geo_ok = np.isfinite(lat) & np.isfinite(lon)
        valid = geo_ok & np.isin(cmk, VALID_CMK)
        vr, vc = np.nonzero(valid)
        vlats = lat[vr, vc]
        vlons = lon[vr, vc]
        vcmk = cmk[vr, vc]
        _ = fill

        for sid, (slat, slon) in STATIONS.items():
            rec = {
                "filename": path.name,
                "gId": gid,
                "station": sid,
                "station_lat": slat,
                "station_lon": slon,
                "sat_observation_time_utc": iso_z(t),
                "sat_acquisition_start_utc": iso_z(start),
                "sat_acquisition_end_utc": iso_z(end),
                "sat_cloud_mask_at_station": None,
                "sat_cloud_mask_class": None,
                "sat_nearest_pixel_distance_km": None,
                "sat_nearest_row": None,
                "sat_nearest_col": None,
                "sat_valid_pixel_count_25km": 0,
                "sat_cloudy_fraction_25km": None,
                "sat_clear_fraction_25km": None,
                "sat_nearby_cloud_fraction_25km": None,
                "sat_observation_available": 0,
            }
            if vlats.size == 0:
                rows.append(rec)
                continue
            box = (np.abs(vlats - slat) <= 1.0) & (np.abs(vlons - slon) <= 1.0)
            if not np.any(box):
                box = np.ones(vlats.shape, dtype=bool)
            d = haversine_km(slat, slon, vlats[box], vlons[box])
            j = int(np.argmin(d))
            rec["sat_cloud_mask_at_station"] = int(vcmk[box][j])
            rec["sat_cloud_mask_class"] = CLASS_NAME.get(rec["sat_cloud_mask_at_station"])
            rec["sat_nearest_pixel_distance_km"] = round(float(d[j]), 6)
            rec["sat_nearest_row"] = int(vr[box][j])
            rec["sat_nearest_col"] = int(vc[box][j])
            rec["sat_observation_available"] = 1
            near = d <= RADIUS_KM
            n_valid = int(near.sum())
            rec["sat_valid_pixel_count_25km"] = n_valid
            if n_valid > 0:
                vals = vcmk[box][near]
                n_cloud = int(np.isin(vals, CLOUDY).sum())
                n_clear = int(np.isin(vals, CLEAR).sum())
                rec["sat_cloudy_fraction_25km"] = n_cloud / n_valid
                rec["sat_clear_fraction_25km"] = n_clear / n_valid
                rec["sat_nearby_cloud_fraction_25km"] = rec["sat_cloudy_fraction_25km"]
            rows.append(rec)
    return rows


def main() -> None:
    inv = list(csv.DictReader(INV.open(encoding="utf-8", newline="")))
    ok = [r for r in inv if r.get("status") == "OK" and r.get("gId") != SKIP_GID]
    files = []
    seen = set()
    for r in ok:
        fn = r["filename"]
        if fn in seen:
            continue
        seen.add(fn)
        files.append((r["gId"], fn))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    errors = []
    for i, (gid, fn) in enumerate(files, 1):
        path = DATA / fn
        try:
            all_rows.extend(extract_file(path, gid))
            if i % 10 == 0 or i == len(files):
                print(f"processed {i}/{len(files)}", flush=True)
        except Exception as e:
            errors.append({"gId": gid, "filename": fn, "error": type(e).__name__})
            print(f"FAIL {fn} {type(e).__name__}", flush=True)

    fields = [
        "filename",
        "gId",
        "station",
        "station_lat",
        "station_lon",
        "sat_observation_time_utc",
        "sat_acquisition_start_utc",
        "sat_acquisition_end_utc",
        "sat_cloud_mask_at_station",
        "sat_cloud_mask_class",
        "sat_nearest_pixel_distance_km",
        "sat_nearest_row",
        "sat_nearest_col",
        "sat_valid_pixel_count_25km",
        "sat_cloudy_fraction_25km",
        "sat_clear_fraction_25km",
        "sat_nearby_cloud_fraction_25km",
        "sat_observation_available",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    n = len(all_rows)
    expected = 119 * 5
    keys = [(r["filename"], r["station"]) for r in all_rows]
    dup = n - len(set(keys))
    masks = [r["sat_cloud_mask_at_station"] for r in all_rows]
    mask_counts = Counter(masks)
    avail = sum(r["sat_observation_available"] == 1 for r in all_rows)
    nearest_ok = sum(r["sat_cloud_mask_at_station"] is not None for r in all_rows)
    neigh0 = sum(r["sat_valid_pixel_count_25km"] == 0 for r in all_rows)
    frac_bad = []
    alias_bad = 0
    fill_as_class = sum(r["sat_cloud_mask_at_station"] == 9 for r in all_rows)
    for r in all_rows:
        for k in (
            "sat_cloudy_fraction_25km",
            "sat_clear_fraction_25km",
            "sat_nearby_cloud_fraction_25km",
        ):
            v = r[k]
            if v is None:
                continue
            if not (0.0 <= v <= 1.0):
                frac_bad.append(r["filename"])
        c = r["sat_cloudy_fraction_25km"]
        a = r["sat_nearby_cloud_fraction_25km"]
        if c != a:
            alias_bad += 1
    counts_25 = [r["sat_valid_pixel_count_25km"] for r in all_rows]
    durs = []
    for r in all_rows:
        if r["sat_acquisition_start_utc"] and r["sat_acquisition_end_utc"]:
            a = datetime.fromisoformat(r["sat_acquisition_start_utc"].replace("Z", "+00:00"))
            b = datetime.fromisoformat(r["sat_acquisition_end_utc"].replace("Z", "+00:00"))
            durs.append((b - a).total_seconds() / 60.0)
    dist = [r["sat_nearest_pixel_distance_km"] for r in all_rows if r["sat_nearest_pixel_distance_km"] is not None]

    sanity = {
        "expected_rows_119x5": expected,
        "produced_rows": n,
        "rows_match_expected": n == expected,
        "duplicate_file_station_keys": dup,
        "cmk_fill_9_as_station_class": fill_as_class,
        "fractions_outside_0_1": len(frac_bad),
        "cloudy_ne_nearby_alias": alias_bad,
        "nwp_tables_modified": False,
    }

    summary = {
        "phase": "10I.2",
        "extracted": True,
        "hourly_join": False,
        "trained": False,
        "source_files_processed": len(files),
        "errors": errors,
        "station_rows_expected": expected,
        "station_rows_produced": n,
        "sat_observation_available_1": avail,
        "nearest_pixel_valid": nearest_ok,
        "neighborhood_zero_valid": neigh0,
        "cmk_station_class_counts": {str(k): v for k, v in sorted(mask_counts.items(), key=lambda x: str(x[0]))},
        "valid_pixel_count_25km": {
            "min": min(counts_25) if counts_25 else None,
            "median": float(np.median(counts_25)) if counts_25 else None,
            "max": max(counts_25) if counts_25 else None,
        },
        "nearest_distance_km": {
            "min": min(dist) if dist else None,
            "median": float(np.median(dist)) if dist else None,
            "max": max(dist) if dist else None,
        },
        "acquisition_duration_minutes": {
            "min": min(durs) if durs else None,
            "median": float(np.median(durs)) if durs else None,
            "max": max(durs) if durs else None,
        },
        "skipped_gid": SKIP_GID,
        "output_csv": OUT_CSV.as_posix(),
        "sanity": sanity,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Phase 10I.2 — INSAT-3DR CMK scan-level feature extraction",
        "",
        "**Scan-level only.** No hourly join, no causal table, no NWP edits, no training, no download, no retry of `7680747`.",
        "",
        f"- Source files processed: **{len(files)}**",
        f"- Expected rows (119 × 5): **{expected}**",
        f"- Produced rows: **{n}**",
        f"- Duplicate (filename, station): **{dup}**",
        f"- Extraction errors: **{len(errors)}**",
        f"- `sat_observation_available=1`: **{avail}**",
        f"- Nearest valid station pixel: **{nearest_ok}**",
        f"- Zero valid 25 km pixels: **{neigh0}**",
        f"- Station CMK class counts: `{dict(mask_counts)}`",
        f"- Fill 9 used as station class: **{fill_as_class}**",
        f"- Fractions outside [0,1]: **{len(frac_bad)}**",
        f"- cloudy ≠ nearby alias: **{alias_bad}**",
        f"- 25 km valid count min/median/max: {summary['valid_pixel_count_25km']}",
        f"- Nearest pixel km min/median/max: {summary['nearest_distance_km']}",
        f"- Acquisition duration min/median/max min: {summary['acquisition_duration_minutes']}",
        "",
        f"CSV: `{OUT_CSV.as_posix()}`",
        "",
        "PHASE 10I.2 COMPLETE — SCAN-LEVEL EXTRACTION ONLY",
        "",
    ]
    OUT_DOC.write_text("\n".join(lines), encoding="utf-8")
    print("files", len(files), "rows", n, "expected", expected, "dup", dup, "err", errors)


if __name__ == "__main__":
    main()
