#!/usr/bin/env python3
"""Phase 10A — INSAT-3DR L2B CMK pilot audit (read-only).

Does not train models, download data, modify the HDF, or commit.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HDF_PATH = (
    ROOT
    / "dataset"
    / "satellite"
    / "insat3dr_cmk_pilot"
    / "3RIMG_22SEP2026_0015_L2B_CMK_V01R00.h5"
)
OUT_DIR = ROOT / "outputs" / "satellite" / "phase10a"
DOC_PATH = ROOT / "docs" / "PHASE10A_INSAT3DR_CMK_PILOT_AUDIT.md"

STATIONS = {
    "VOTV": (8.4667, 76.9500),
    "VECC": (22.6547, 88.4467),
    "VIDP": (28.5667, 77.1167),
    "VOCI": (10.1500, 76.4000),
    "VABB": (19.1005, 72.8585),
}

FLAG_MAP_FROM_FILE = None  # filled from flag_values / flag_meanings


def _decode_attr(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return bytes(v).decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        if v.dtype.kind in ("S", "O"):
            if v.size == 1:
                return _decode_attr(v.reshape(-1)[0])
            return [_decode_attr(x) for x in v.ravel()]
        if v.size == 1:
            return v.reshape(-1)[0].item()
        return v.tolist()
    if hasattr(v, "item") and np.ndim(v) == 0:
        return v.item()
    return v


def attr_dict(obj):
    out = {}
    for k in obj.attrs:
        if k in ("DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME"):
            continue
        out[k] = _decode_attr(obj.attrs[k])
    return out


def walk_h5(f):
    groups = []
    datasets = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            groups.append({"path": "/" + name if not name.startswith("/") else name, "nkeys": len(obj.keys())})
        elif isinstance(obj, h5py.Dataset):
            datasets.append(
                {
                    "path": "/" + name,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "attrs": attr_dict(obj),
                }
            )

    visitor("", f)
    f.visititems(visitor)
    return groups, datasets


def decode_geo(ds):
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


def parse_cmk_flags(cmk_ds):
    values = np.array(cmk_ds.attrs["flag_values"]).astype(int).tolist()
    meanings = _decode_attr(cmk_ds.attrs["flag_meanings"]).split()
    mapping = {}
    for val, mean in zip(values, meanings):
        mapping[int(val)] = mean
    fill = int(np.array(cmk_ds.attrs["_FillValue"]).reshape(-1)[0])
    return mapping, fill


def decode_time(time_ds, root_attrs):
    raw = np.array(time_ds[:]).reshape(-1)[0]
    units = _decode_attr(time_ds.attrs.get("units", b""))
    dt = None
    if units.startswith("minutes since "):
        origin = units[len("minutes since ") :].strip()
        origin_dt = datetime.strptime(origin, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        dt = origin_dt + timedelta(minutes=float(raw))
    start = _decode_attr(root_attrs.get("Acquisition_Start_Time", b""))
    end = _decode_attr(root_attrs.get("Acquisition_End_Time", b""))
    gmt = _decode_attr(root_attrs.get("Acquisition_Time_in_GMT", b""))
    date = _decode_attr(root_attrs.get("Acquisition_Date", b""))
    return {
        "time_dataset_value": float(raw),
        "time_units": units,
        "time_utc_decoded": dt.isoformat() if dt else None,
        "Acquisition_Date": date,
        "Acquisition_Start_Time": start,
        "Acquisition_End_Time": end,
        "Acquisition_Time_in_GMT": gmt,
    }


def nearby_cloud_stats(lat, lon, cmk, slat, slon, radius_km, fill, valid_flags):
    dist = haversine_km(slat, slon, lat, lon)
    near = np.isfinite(dist) & (dist <= radius_km)
    vals = cmk[near]
    n = int(vals.size)
    if n == 0:
        return {"radius_km": radius_km, "n_pixels": 0, "status": "NO_COVERAGE"}
    cloudy_vals = {1, 3}  # Cloudy, Probably_Cloudy from file flag_meanings
    n_cloudy = int(np.isin(vals, list(cloudy_vals)).sum())
    n_clearish = int(np.isin(vals, [0, 2]).sum())  # Clear, Probably_Clear
    n_fill = int((vals == fill).sum())
    n_valid = int(np.isin(vals, valid_flags).sum())
    frac = (n_cloudy / n_valid) if n_valid else None
    return {
        "radius_km": radius_km,
        "n_pixels": n,
        "n_valid_cmk": n_valid,
        "n_fill": n_fill,
        "n_cloudy_or_probably_cloudy": n_cloudy,
        "n_clear_or_probably_clear": n_clearish,
        "cloud_fraction_among_valid": frac,
        "status": "OK" if n_valid else "NO_VALID_CMK",
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not HDF_PATH.is_file():
        raise SystemExit(f"missing {HDF_PATH}")

    with h5py.File(str(HDF_PATH), "r") as f:
        groups, datasets = walk_h5(f)
        root_attrs = attr_dict(f)
        cmk_ds = f["CMK"]
        lat_ds = f["Latitude"]
        lon_ds = f["Longitude"]
        time_ds = f["time"]
        flag_map, fill = parse_cmk_flags(cmk_ds)
        time_info = decode_time(time_ds, f.attrs)

        cmk = np.asarray(cmk_ds[0], dtype=np.int16)
        lat = decode_geo(lat_ds)
        lon = decode_geo(lon_ds)

        geo_ok = np.isfinite(lat) & np.isfinite(lon)
        valid_flags = sorted(flag_map.keys())
        cmk_ok = np.isin(cmk, valid_flags)
        both = geo_ok & cmk_ok

        unique, counts = np.unique(cmk, return_counts=True)
        value_counts = {int(u): int(c) for u, c in zip(unique, counts)}

        lat_valid = lat[geo_ok]
        lon_valid = lon[geo_ok]
        coverage = {
            "lat_min": float(np.nanmin(lat_valid)) if lat_valid.size else None,
            "lat_max": float(np.nanmax(lat_valid)) if lat_valid.size else None,
            "lon_min": float(np.nanmin(lon_valid)) if lon_valid.size else None,
            "lon_max": float(np.nanmax(lon_valid)) if lon_valid.size else None,
            "n_geo_valid_pixels": int(geo_ok.sum()),
            "n_cmk_valid_pixels": int(cmk_ok.sum()),
            "n_both_valid": int(both.sum()),
            "spatial_shape_cmk": list(cmk.shape),
            "latitude_longitude_layout": "2D",
        }

        station_rows = []
        for sid, (slat, slon) in STATIONS.items():
            dist = haversine_km(slat, slon, lat, lon)
            dist = np.where(both, dist, np.nan)
            if not np.isfinite(dist).any():
                station_rows.append(
                    {
                        "station_id": sid,
                        "station_lat": slat,
                        "station_lon": slon,
                        "status": "NO_VALID_PIXEL",
                        "pixel_lat": None,
                        "pixel_lon": None,
                        "distance_km": None,
                        "row": None,
                        "col": None,
                        "cmk_raw": None,
                        "cmk_class": None,
                        "nearby_25km": nearby_cloud_stats(lat, lon, cmk, slat, slon, 25.0, fill, valid_flags),
                    }
                )
                continue
            idx = int(np.nanargmin(dist))
            r, c = np.unravel_index(idx, dist.shape)
            raw = int(cmk[r, c])
            station_rows.append(
                {
                    "station_id": sid,
                    "station_lat": slat,
                    "station_lon": slon,
                    "status": "OK",
                    "pixel_lat": float(lat[r, c]),
                    "pixel_lon": float(lon[r, c]),
                    "distance_km": float(dist[r, c]),
                    "row": int(r),
                    "col": int(c),
                    "cmk_raw": raw,
                    "cmk_class": flag_map.get(raw),
                    "nearby_25km": nearby_cloud_stats(lat, lon, cmk, slat, slon, 25.0, fill, valid_flags),
                }
            )

    n_ok = sum(1 for x in station_rows if x["status"] == "OK")
    single_file = True
    historical_training = False
    feature_feasibility = {
        "cloud_mask_at_station": {
            "feasible_for_this_file": n_ok == 5,
            "note": "Nearest valid CMK pixel can be sampled when lat/lon/CMK are valid. One file = one scan time, not a time series.",
        },
        "nearby_cloud_fraction": {
            "feasible_for_this_file": True,
            "note": "Fraction of valid pixels within a radius using official flags Cloudy/Probably_Cloudy vs Clear/Probably_Clear. Unobserved pixels must not be treated as clear or zero.",
        },
        "clear_cloudy_indicator": {
            "feasible_for_this_file": n_ok == 5,
            "note": "Can map flag 0/2 vs 1/3 only as labelled in flag_meanings. Probably_* must not be collapsed without documenting uncertainty.",
        },
    }

    summary = {
        "phase": "10A",
        "audit_only": True,
        "hdf_path": str(HDF_PATH.relative_to(ROOT)).replace("\\", "/"),
        "h5py_version": h5py.__version__,
        "root_attributes": root_attrs,
        "groups": groups,
        "datasets": datasets,
        "cmk": {
            "dataset": "/CMK",
            "shape": [1, 2816, 2805],
            "dtype": "int8",
            "long_name": "Cloud Mask",
            "coordinates": "time Latitude Longitude",
            "flag_values": valid_flags,
            "flag_meanings": flag_map,
            "_FillValue": fill,
            "value_counts": value_counts,
            "interpretation_source": "HDF dataset attributes flag_values and flag_meanings; no invented classes",
        },
        "geolocation": {
            "latitude_dataset": "/Latitude",
            "longitude_dataset": "/Longitude",
            "layout": "2D",
            "dtype_on_disk": "int16",
            "scale_factor": 0.01,
            "add_offset": 0.0,
            "_FillValue": 32767,
            "decoded_units": "degrees",
        },
        "time": time_info,
        "coverage": coverage,
        "stations": station_rows,
        "feature_feasibility": feature_feasibility,
        "historical_model_training": {
            "single_pilot_file_sufficient": False,
            "reason": (
                "This audit opened exactly one L2B CMK granule at one acquisition "
                f"({time_info.get('Acquisition_Start_Time')}–{time_info.get('Acquisition_End_Time')}). "
                "ThunderWatch training uses hourly 2014–2025 station rows. One scan cannot be "
                "broadcast across historical hours and cannot supply a training time series. "
                "Institutional MOSDAC/ISRO archive access would be required for historical CMK."
            ),
        },
        "scientific_rules": [
            "Do not 0-fill missing satellite pixels.",
            "Do not broadcast this scan across historical hours.",
            "Do not invent CMK classes beyond flag_meanings.",
            "Do not train on this pilot.",
        ],
        "recommended_next_step": (
            "If satellite CMK is still desired, Phase 10B should define an institutional "
            "MOSDAC download/access plan for a multi-year, station-timed L2B CMK archive "
            "and a join that leaves unobserved hours as NOT_OBSERVED. Do not implement "
            "features or retrain Model B until that archive exists."
        ),
        "classification": "PILOT_STRUCTURE_VALID / HISTORICAL_ARCHIVE_REQUIRED",
    }

    with (OUT_DIR / "phase10a_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, default=str)

    write_report(summary)
    print("classification", summary["classification"])
    for row in station_rows:
        print(
            row["station_id"],
            row["status"],
            row.get("cmk_raw"),
            row.get("cmk_class"),
            row.get("distance_km"),
        )


def write_report(s):
    a = []
    p = a.append
    p("# Phase 10A — INSAT-3DR 3RIMG L2B CMK pilot audit")
    p("")
    p("**Date:** 2026-09-22  ")
    p("**File:** `dataset/satellite/insat3dr_cmk_pilot/3RIMG_22SEP2026_0015_L2B_CMK_V01R00.h5`  ")
    p("**Audit only.** HDF not modified. No training, no feature implementation, no extra download, no commit.")
    p("")
    p("## 1. Objective")
    p("")
    p("Determine how cloud-mask data in this single INSAT-3DR L2B CMK HDF5 granule can be extracted and whether it can be synchronized with ThunderWatch station timestamps. This is not historical collection and not model integration.")
    p("")
    p("## 2. HDF structure")
    p("")
    p(f"- Opened with **h5py {s['h5py_version']}** (read-only).")
    p("- Root groups/datasets: `/CMK`, `/GeoX`, `/GeoY`, `/Latitude`, `/Longitude`, `/time`.")
    p("- No nested groups beyond the file root.")
    p("")
    p("| Dataset | Shape | Dtype | Notes |")
    p("|---------|-------|-------|-------|")
    for d in s["datasets"]:
        p(f"| `{d['path']}` | {d['shape']} | `{d['dtype']}` | {d['attrs']} |")
    p("")
    p("Selected root attributes:")
    p("")
    ra = s["root_attributes"]
    for k in (
        "Satellite_Name",
        "Sensor_Name",
        "Processing_Level",
        "Product_Type",
        "HDF_Product_File_Name",
        "Acquisition_Date",
        "Acquisition_Start_Time",
        "Acquisition_End_Time",
        "Acquisition_Time_in_GMT",
        "Nominal_Central_Point_Coordinates(degrees)_Latitude_Longitude",
        "left_longitude",
        "right_longitude",
        "lower_latitude",
        "upper_latitude",
        "title",
        "conventions",
    ):
        if k in ra:
            p(f"- `{k}`: `{ra[k]}`")
    p("")
    p("## 3. CMK variable and official metadata")
    p("")
    c = s["cmk"]
    p(f"- Dataset: `{c['dataset']}` shape `{c['shape']}` dtype `{c['dtype']}`")
    p(f"- `long_name`: {c['long_name']}")
    p(f"- `coordinates`: {c['coordinates']}")
    p(f"- `_FillValue`: **{c['_FillValue']}**")
    p("- Official classes from **file attributes** `flag_values` + `flag_meanings` (not invented):")
    p("")
    p("| flag_value | flag_meaning |")
    p("|------------|--------------|")
    for k, v in c["flag_meanings"].items():
        p(f"| {k} | {v} |")
    p("")
    p("Observed value counts in this granule (including fill):")
    p("")
    p("| raw value | count | meaning |")
    p("|-----------|-------|---------|")
    for val, n in sorted(c["value_counts"].items()):
        meaning = c["flag_meanings"].get(val, "fill" if val == c["_FillValue"] else "unlisted")
        p(f"| {val} | {n} | {meaning} |")
    p("")
    p("## 4. Geolocation and time")
    p("")
    g = s["geolocation"]
    p(f"- Latitude/Longitude are **2-D** (`int16` on disk), same spatial size as CMK spatial dims.")
    p(f"- Decode: `physical = raw * {g['scale_factor']} + {g['add_offset']}`; fill `{g['_FillValue']}`.")
    p(f"- Units: {g['decoded_units']} (`degrees_north` / `degrees_east`).")
    p("")
    t = s["time"]
    p(f"- `/time` value: `{t['time_dataset_value']}`")
    p(f"- `/time` units: `{t['time_units']}`")
    p(f"- Decoded UTC from units: `{t['time_utc_decoded']}`")
    p(f"- Acquisition start (attr): `{t['Acquisition_Start_Time']}`")
    p(f"- Acquisition end (attr): `{t['Acquisition_End_Time']}`")
    p(f"- Acquisition GMT slot (attr): `{t['Acquisition_Time_in_GMT']}` on `{t['Acquisition_Date']}`")
    p("")
    cov = s["coverage"]
    p(f"- Decoded valid lat range: {cov['lat_min']} … {cov['lat_max']}")
    p(f"- Decoded valid lon range: {cov['lon_min']} … {cov['lon_max']}")
    p(f"- Geo-valid pixels: {cov['n_geo_valid_pixels']}; CMK in {{0,1,2,3}}: {cov['n_cmk_valid_pixels']}; both: {cov['n_both_valid']}")
    p("")
    p("## 5. Five-station extraction")
    p("")
    p("Nearest pixel among locations with valid lat/lon **and** CMK in official `flag_values`. Distance is haversine (km).")
    p("")
    p("| station | st_lat | st_lon | pixel_lat | pixel_lon | distance_km | row | col | CMK raw | CMK class (from file) |")
    p("|---------|--------|--------|-----------|-----------|-------------|-----|-----|---------|----------------------|")
    for row in s["stations"]:
        p(
            f"| {row['station_id']} | {row['station_lat']} | {row['station_lon']} | "
            f"{row.get('pixel_lat')} | {row.get('pixel_lon')} | {row.get('distance_km')} | "
            f"{row.get('row')} | {row.get('col')} | {row.get('cmk_raw')} | {row.get('cmk_class')} |"
        )
    p("")
    p("25 km neighbourhood (feasibility only; not a training feature):")
    p("")
    p("| station | n_pixels | n_valid | cloud_fraction_among_valid | status |")
    p("|---------|----------|---------|----------------------------|--------|")
    for row in s["stations"]:
        n = row["nearby_25km"]
        p(
            f"| {row['station_id']} | {n.get('n_pixels')} | {n.get('n_valid_cmk')} | "
            f"{n.get('cloud_fraction_among_valid')} | {n.get('status')} |"
        )
    p("")
    p("## 6. Station-level feature feasibility")
    p("")
    for name, rec in s["feature_feasibility"].items():
        p(f"- **{name}**: feasible_for_this_file=`{rec['feasible_for_this_file']}`. {rec['note']}")
    p("")
    p("These features are **not implemented**. Missing pixels must remain missing (`NOT_OBSERVED`), never 0.")
    p("")
    p("## 7. Historical-data feasibility")
    p("")
    h = s["historical_model_training"]
    p(f"- Single pilot file sufficient for historical model training: **{h['single_pilot_file_sufficient']}**")
    p(f"- {h['reason']}")
    p("")
    p("Synchronization with ThunderWatch hourly timestamps is possible **in principle** by matching `Acquisition_Start_Time` / decoded `/time` to the nearest hour **only when a granule exists**. This file covers one ~27-minute scan on 22 Sep 2026 00:15–00:42 GMT. It must not be copied into other hours.")
    p("")
    p("## 8. Classification")
    p("")
    p(f"**`{s['classification']}`**")
    p("")
    p("## 9. Recommended next step")
    p("")
    p(s["recommended_next_step"])
    p("")
    p("## 10. Scientific non-negotiables")
    p("")
    for item in s["scientific_rules"]:
        p(f"- {item}")
    p("")
    p("PHASE 10A COMPLETE — INSAT-3DR CMK PILOT AUDIT READY")
    p("")
    DOC_PATH.write_text("\n".join(a), encoding="utf-8")


if __name__ == "__main__":
    main()
