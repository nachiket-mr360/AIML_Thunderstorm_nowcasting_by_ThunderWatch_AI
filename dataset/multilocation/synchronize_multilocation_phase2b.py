"""Phase 2B: synchronize Open-Meteo hourly predictors with genuine METAR thunderstorm labels.

Writes only under dataset/multilocation/ and docs/PHASE2B_SYNCHRONIZATION_REPORT.md.
Does not modify V1 files, Flask, models, or git.

Missing METAR hours stay NA. They are never filled with 0.
Open-Meteo weather_code is not used as the target (Phase 2A did not even fetch it).
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.dirname(HERE)
REPO = os.path.dirname(DATASET)

RAW_OM = os.path.join(HERE, "raw_openmeteo")
RAW_METAR = os.path.join(HERE, "raw_metar")
SYNC_DIR = os.path.join(HERE, "synchronized")
IEM_META_PATH = os.path.join(HERE, "iem_station_metadata.json")
OUT_COMBINED = os.path.join(SYNC_DIR, "multilocation_synchronized_2014_2025.csv")
OUT_META = os.path.join(SYNC_DIR, "multilocation_synchronized_2014_2025_metadata.json")
OUT_REPORT = os.path.join(REPO, "docs", "PHASE2B_SYNCHRONIZATION_REPORT.md")

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
USER_AGENT = "SIH26072-v2-phase2b-sync/1.0 (research use)"
MAX_ATTEMPTS = 4

WINDOW_START = pd.Timestamp("2014-01-01 00:00:00", tz="UTC")
WINDOW_END = pd.Timestamp("2025-12-31 23:00:00", tz="UTC")
START_YEAR, END_YEAR = 2014, 2025

PREDICTORS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]

CITIES = {
    "VOTV": "Thiruvananthapuram",
    "VECC": "Kolkata",
    "VIDP": "Delhi",
    "VOCI": "Kochi",
    "VABB": "Mumbai",
}

STATION_ORDER = ["VOTV", "VECC", "VIDP", "VOCI", "VABB"]

V1_PROTECTED = [
    os.path.join(DATASET, "weather_data.csv"),
    os.path.join(DATASET, "weather_data_with_code.csv"),
    os.path.join(DATASET, "historical_thunderstorm_labels_votv.csv"),
    os.path.join(DATASET, "votv_thunderstorm_synchronized_2014_2025.csv"),
    os.path.join(DATASET, "build_thunderstorm_labels.py"),
    os.path.join(DATASET, "build_synchronized_dataset_phase3.py"),
]


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_wx_codes(value: str) -> list[str]:
    if value is None:
        return []
    raw = str(value).strip()
    if raw == "" or raw.upper() == "M":
        return []
    return [tok for tok in re.split(r"[\/\s]+", raw) if tok]


def has_decodable_wx(value: str) -> bool:
    return len(split_wx_codes(value)) > 0


def classify_token(token: str) -> str:
    tok = token.strip().upper()
    if not tok:
        return "other"
    tok = tok.lstrip("+-")
    if tok.startswith("VC"):
        body = tok[2:].lstrip("+-")
        return "vcts" if body.startswith("TS") else "other"
    return "ts" if tok.startswith("TS") else "other"


def code_family(value: str) -> tuple[bool, bool]:
    station = vicinity = False
    for tok in split_wx_codes(value):
        kind = classify_token(tok)
        if kind == "ts":
            station = True
        elif kind == "vcts":
            vicinity = True
    return station, vicinity


def build_iem_url(station: str, year: int) -> str:
    params = [
        ("station", station),
        ("data", "wxcodes"),
        ("year1", str(year)),
        ("month1", "1"),
        ("day1", "1"),
        ("year2", str(year)),
        ("month2", "12"),
        ("day2", "31"),
        ("tz", "UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "no"),
        ("report_type", "3"),
        ("report_type", "4"),
    ]
    return IEM_URL + "?" + urllib.parse.urlencode(params, doseq=True)


def validate_body(body: str) -> None:
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("empty response body")
    header = lines[0].rstrip("\r\n")
    if not header.startswith("station,valid"):
        raise RuntimeError(f"unexpected header: {header[:120]!r}")
    expected = len(next(csv.reader([header])))
    for i, line in enumerate(lines[1:], start=2):
        if len(next(csv.reader([line]))) != expected:
            raise RuntimeError(f"ragged row at line {i}")


def fetch(url: str) -> str:
    last_err: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as resp:
                chunks: list[bytes] = []
                try:
                    while True:
                        block = resp.read(1 << 16)
                        if not block:
                            break
                        chunks.append(block)
                except http.client.IncompleteRead as exc:
                    raise OSError(f"incomplete read ({len(exc.partial)} bytes)") from exc
                body = b"".join(chunks).decode("utf-8", errors="replace")
            validate_body(body)
            return body
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_err = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(3.0 * attempt)
    raise RuntimeError(f"download failed after {MAX_ATTEMPTS} attempts: {last_err}")


def chunk_is_valid(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return False
        if len(header) < 3 or header[0] != "station" or header[1] != "valid":
            return False
        expected = len(header)
        return all(len(row) == expected for row in reader)


def download_metar(station: str) -> dict:
    os.makedirs(os.path.join(RAW_METAR, station.lower()), exist_ok=True)
    log = {"station": station, "chunks": [], "source": IEM_URL}
    paths = []
    header = None
    for year in range(START_YEAR, END_YEAR + 1):
        path = os.path.join(RAW_METAR, station.lower(), f"{station.lower()}_metar_iem_{year}.csv")
        paths.append(path)
        if not chunk_is_valid(path):
            body = fetch(build_iem_url(station, year))
            lines = [ln for ln in body.splitlines() if ln.strip()]
            tmp = path + ".part"
            with open(tmp, "w", encoding="utf-8", newline="") as fh:
                fh.write(body if body.endswith("\n") else body + "\n")
            os.replace(tmp, path)
            print(f"  [{station} {year}] downloaded {len(lines) - 1} rows", flush=True)
            time.sleep(0.4)
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            hdr = next(reader)
            rows = sum(1 for _ in reader)
        if header is None:
            header = ",".join(hdr)
        log["chunks"].append({"year": year, "file": os.path.basename(path), "data_rows": rows})
        print(f"  [{station} {year}] rows={rows}", flush=True)

    combined = os.path.join(RAW_METAR, station.lower(), f"{station.lower()}_metar_iem_raw.csv")
    total = 0
    with open(combined, "w", encoding="utf-8", newline="") as out:
        out.write((header or "station,valid,wxcodes") + "\n")
        for path in paths:
            with open(path, "r", encoding="utf-8", newline="") as fh:
                fh.readline()
                for line in fh:
                    if line.strip():
                        out.write(line)
                        total += 1
    log["combined_file"] = os.path.basename(combined)
    log["combined_rows"] = total
    log["combined_sha256"] = sha256_of(combined)
    with open(os.path.join(RAW_METAR, station.lower(), "download_log.json"), "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    return log


def load_predictors(station: str) -> pd.DataFrame:
    path = os.path.join(RAW_OM, station.lower(), f"{station.lower()}_openmeteo_hourly_2014_2025.csv")
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"{station}: missing date column")
    frame["timestamp_utc"] = pd.to_datetime(frame["date"], utc=True)
    missing_cols = [c for c in PREDICTORS if c not in frame.columns]
    if missing_cols:
        raise ValueError(f"{station}: missing predictors {missing_cols}")
    if "weather_code" in frame.columns:
        raise ValueError(f"{station}: weather_code present; must not be used as target")
    frame = frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="first")
    frame = frame[(frame["timestamp_utc"] >= WINDOW_START) & (frame["timestamp_utc"] <= WINDOW_END)]
    keep = ["timestamp_utc"] + PREDICTORS
    return frame[keep].reset_index(drop=True)


def hourly_labels(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["valid_dt"] = pd.to_datetime(raw["valid"], format="%Y-%m-%d %H:%M", utc=True, errors="coerce")
    raw = raw.dropna(subset=["valid_dt"])
    raw = raw[(raw["valid_dt"] >= WINDOW_START) & (raw["valid_dt"] <= WINDOW_END + pd.Timedelta(hours=0, minutes=59))]
    raw["hour"] = raw["valid_dt"].dt.floor("h")
    raw["wxcodes"] = raw["wxcodes"].fillna("").astype(str)
    raw["wx_decodable"] = raw["wxcodes"].map(has_decodable_wx)
    fam = raw["wxcodes"].map(code_family)
    raw["ts_station"] = [a for a, _ in fam]
    raw["ts_vicinity"] = [b for _, b in fam]
    raw["ts_any"] = raw["ts_station"] | raw["ts_vicinity"]
    raw = raw.sort_values(["valid_dt"]).drop_duplicates(subset=["valid_dt"], keep="first")

    g = raw.groupby("hour")
    agg = pd.DataFrame(
        {
            "n_reports": g.size(),
            "n_reports_with_wx": g["wx_decodable"].sum(),
            "n_ts_reports": g["ts_any"].sum(),
            "ts_any": g["ts_any"].any(),
        }
    )
    grid = pd.DataFrame({"timestamp_utc": pd.date_range(WINDOW_START, WINDOW_END, freq="h")})
    out = grid.merge(agg, left_on="timestamp_utc", right_index=True, how="left")
    out["n_reports"] = out["n_reports"].fillna(0).astype(int)
    out["n_reports_with_wx"] = out["n_reports_with_wx"].fillna(0).astype(int)
    out["n_ts_reports"] = out["n_ts_reports"].fillna(0).astype(int)
    out["ts_any"] = out["ts_any"].fillna(False).astype(bool)

    # Observed label only when a present-weather group exists (Phase 1B).
    # Hours with reports but only M/empty wxcodes are target-unavailable, not 0.
    usable = out["n_reports_with_wx"] > 0
    target = pd.Series(np.nan, index=out.index, dtype="float64")
    target[usable] = 0.0
    target[usable & out["ts_any"]] = 1.0
    out["thunderstorm_target"] = target
    out["target_observed"] = usable.astype(int)
    status = np.full(len(out), "unavailable", dtype=object)
    status[usable & (out["thunderstorm_target"] == 0)] = "observed_non_thunderstorm"
    status[usable & (out["thunderstorm_target"] == 1)] = "observed_thunderstorm"
    out["label_status"] = status
    out["metar_observation_count"] = int(len(raw))
    return out, int(len(raw)), int(raw["ts_any"].sum())


def continuity_stats(ts: pd.Series) -> dict:
    ts = ts.sort_values()
    expected = pd.date_range(ts.min(), ts.max(), freq="h")
    missing = int(len(expected) - ts.nunique())
    diffs = ts.diff().dropna()
    gaps = int((diffs > pd.Timedelta(hours=1)).sum())
    return {
        "unique_hours": int(ts.nunique()),
        "duplicates": int(ts.duplicated().sum()),
        "sorted": bool(ts.is_monotonic_increasing),
        "all_hour_boundary": bool((ts.dt.minute == 0).all() and (ts.dt.second == 0).all()),
        "internal_missing_hours": missing,
        "gaps_gt_1h": gaps,
        "first": ts.iloc[0].isoformat() if len(ts) else None,
        "last": ts.iloc[-1].isoformat() if len(ts) else None,
    }


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def main() -> int:
    protected_before = {p: sha256_of(p) for p in V1_PROTECTED if os.path.exists(p)}
    with open(IEM_META_PATH, encoding="utf-8") as fh:
        iem = json.load(fh)["stations"]

    os.makedirs(SYNC_DIR, exist_ok=True)
    os.makedirs(RAW_METAR, exist_ok=True)

    station_frames = []
    station_stats = {}
    problems: list[str] = []

    for station in STATION_ORDER:
        print(f"=== {station} ===", flush=True)
        metar_log = download_metar(station)
        raw_path = os.path.join(RAW_METAR, station.lower(), f"{station.lower()}_metar_iem_raw.csv")
        raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False, na_values=[])
        labels, n_metar, n_ts_reports = hourly_labels(raw)
        preds = load_predictors(station)

        pred_hours = set(preds["timestamp_utc"])
        window = pd.date_range(WINDOW_START, WINDOW_END, freq="h")
        unmatched_pred = int(len(preds) - preds["timestamp_utc"].isin(labels["timestamp_utc"]).sum())
        # Join on predictor timestamps only.
        merged = preds.merge(labels, on="timestamp_utc", how="left", validate="one_to_one")
        if merged["thunderstorm_target"].isna().all() and n_metar == 0:
            problems.append(f"{station}: no METAR observations")

        meta = iem[station]
        merged["station_id"] = station
        merged["city"] = CITIES[station]
        merged["latitude"] = meta["latitude"]
        merged["longitude"] = meta["longitude"]
        # left join can leave label cols NA for unmatched hours
        merged["target_observed"] = merged["target_observed"].fillna(0).astype(int)
        merged["label_status"] = merged["label_status"].fillna("unavailable")
        merged["n_reports"] = merged["n_reports"].fillna(0).astype(int)
        merged["n_reports_with_wx"] = merged["n_reports_with_wx"].fillna(0).astype(int)
        merged["n_ts_reports"] = merged["n_ts_reports"].fillna(0).astype(int)

        cols = [
            "station_id",
            "city",
            "latitude",
            "longitude",
            "timestamp_utc",
            *PREDICTORS,
            "thunderstorm_target",
            "target_observed",
            "label_status",
            "n_reports",
            "n_reports_with_wx",
            "n_ts_reports",
        ]
        merged = merged[cols].sort_values("timestamp_utc").reset_index(drop=True)

        observed = merged["target_observed"] == 1
        n_pos = int((merged["thunderstorm_target"] == 1).sum())
        n_neg = int((merged["thunderstorm_target"] == 0).sum())
        n_miss = int(merged["thunderstorm_target"].isna().sum())
        n_obs = int(observed.sum())
        pos_rate = round(n_pos / n_obs, 6) if n_obs else None

        missing_pred = {c: int(merged[c].isna().sum()) for c in PREDICTORS}
        cont = continuity_stats(merged["timestamp_utc"])
        if cont["duplicates"]:
            problems.append(f"{station}: duplicate predictor timestamps after join")
        if not cont["sorted"]:
            problems.append(f"{station}: timestamps not sorted")
        if len(merged) != len(preds):
            problems.append(f"{station}: join changed predictor row count")

        unmatched_predictor_hours = int((~merged["timestamp_utc"].isin(labels["timestamp_utc"])).sum())
        # All predictor hours should be in the full hourly label grid.
        if unmatched_predictor_hours:
            problems.append(f"{station}: {unmatched_predictor_hours} predictor hours not on label grid")

        per_path = os.path.join(SYNC_DIR, f"{station.lower()}_synchronized_2014_2025.csv")
        merged.to_csv(per_path, index=False)

        stats = {
            "station_id": station,
            "city": CITIES[station],
            "latitude": meta["latitude"],
            "longitude": meta["longitude"],
            "predictor_rows": int(len(preds)),
            "metar_observations": n_metar,
            "metar_ts_reports": n_ts_reports,
            "matched_rows": int(len(merged)),
            "unmatched_predictor_rows": unmatched_predictor_hours,
            "observed_target_rows": n_obs,
            "thunderstorm_positive_rows": n_pos,
            "observed_negative_rows": n_neg,
            "missing_target_rows": n_miss,
            "positive_rate_among_observed": pos_rate,
            "date_range": {"first": cont["first"], "last": cont["last"]},
            "duplicate_count": cont["duplicates"],
            "timestamp_continuity": cont,
            "missingness_per_predictor": missing_pred,
            "window_hours_expected": int(len(window)),
            "predictor_hours_vs_window": int(len(preds) - len(window)),
            "csv": os.path.relpath(per_path, REPO).replace("\\", "/"),
            "metar_combined_rows_log": metar_log["combined_rows"],
        }
        station_stats[station] = stats
        station_frames.append(merged)
        print(
            f"  preds={len(preds)} metar={n_metar} pos={n_pos} neg={n_neg} miss={n_miss} rate={pos_rate}",
            flush=True,
        )

    combined = pd.concat(station_frames, ignore_index=True)
    combined.to_csv(OUT_COMBINED, index=False)

    # VOTV vs V1 baseline (Phase 1B / V1 modelled window: 3694 TS hours).
    v1_sync_meta = os.path.join(DATASET, "votv_thunderstorm_synchronized_2014_2025_metadata.json")
    v1_compare = {"phase1b_votv_ts_hours": 3694, "note": "Phase 1B / README modelled-window VOTV TS hours"}
    votv = station_stats["VOTV"]
    v1_compare["phase2b_votv_positive_rows"] = votv["thunderstorm_positive_rows"]
    v1_compare["phase2b_votv_observed_rows"] = votv["observed_target_rows"]
    v1_compare["delta_positives_vs_3694"] = votv["thunderstorm_positive_rows"] - 3694
    if os.path.exists(v1_sync_meta):
        with open(v1_sync_meta, encoding="utf-8") as fh:
            v1m = json.load(fh)
        # metadata structure may vary
        v1_compare["v1_sync_metadata_keys"] = sorted(v1m.keys())[:20]

    protected_after = {p: sha256_of(p) for p in V1_PROTECTED if os.path.exists(p)}
    if protected_before != protected_after:
        problems.append("V1 protected file hash changed")

    metadata = {
        "phase": "2B",
        "version": "SIH26072-V2-phase2b-sync-2014-2025",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "predictors": "Open-Meteo Historical Weather API archive (Phase 2A files)",
            "labels": "Iowa Environmental Mesonet ASOS/METAR wxcodes (METAR+SPECI)",
            "iem_service": IEM_URL,
        },
        "retrieval_period": {
            "start_utc": WINDOW_START.isoformat(),
            "end_utc": WINDOW_END.isoformat(),
        },
        "variables": PREDICTORS,
        "station_coordinates": {s: {"latitude": iem[s]["latitude"], "longitude": iem[s]["longitude"], "city": CITIES[s]} for s in STATION_ORDER},
        "target_definition": (
            "Hourly UTC thunderstorm_target = 1 if any METAR/SPECI in the clock hour has "
            "a TS-family IEM wxcodes token (station TS* or VCTS); = 0 if the hour has at "
            "least one decodable present-weather group and none is TS-family; else NA."
        ),
        "timestamp_convention": "UTC clock hour; METAR valid times floored to the hour; join key station_id + timestamp_utc",
        "missing_label_policy": (
            "A missing METAR observation is not a negative thunderstorm observation. "
            "Hours with no report, or reports whose wxcodes are empty/M, remain NA "
            "(label_status=unavailable, target_observed=0). Zeros are never imputed."
        ),
        "lead_time_targets": "not created in Phase 2B (base observed hour only)",
        "weather_code_as_target": False,
        "row_counts": {
            "combined_rows": int(len(combined)),
            "stations": station_stats,
        },
        "v1_votv_comparison": v1_compare,
        "problems": problems,
        "v1_protected_hashes_unchanged": protected_before == protected_after,
        "combined_csv": os.path.relpath(OUT_COMBINED, REPO).replace("\\", "/"),
        "combined_sha256": sha256_of(OUT_COMBINED),
    }
    with open(OUT_META, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    status = "NEEDS REVIEW" if problems else "READY"
    if any("hash changed" in p for p in problems):
        status = "BLOCKED"

    rows = []
    for s in STATION_ORDER:
        st = station_stats[s]
        rows.append(
            [
                s,
                st["city"],
                st["predictor_rows"],
                st["metar_observations"],
                st["matched_rows"],
                st["unmatched_predictor_rows"],
                st["observed_target_rows"],
                st["thunderstorm_positive_rows"],
                st["observed_negative_rows"],
                st["missing_target_rows"],
                st["positive_rate_among_observed"],
                f"{st['date_range']['first']} → {st['date_range']['last']}",
                st["duplicate_count"],
                st["timestamp_continuity"]["internal_missing_hours"],
            ]
        )

    miss_rows = []
    for s in STATION_ORDER:
        st = station_stats[s]
        miss_rows.append([s] + [st["missingness_per_predictor"][c] for c in PREDICTORS])

    report = f"""# Phase 2B — Multi-location dataset synchronization

**Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d")}  
**Repo:** V2 working copy `{REPO}`  
**V1 files:** not modified (hash check: {protected_before == protected_after})  
**Flask / frontend / models:** not modified  
**Git:** no commit, no push  
**Training / model metrics:** not run

Predictors are Open-Meteo atmospheric/reanalysis inputs. The thunderstorm target is **only** genuine IEM METAR `wxcodes`. Open-Meteo `weather_code` is not a target and was not joined as a label.

---

## PHASE STATUS: **{status}**

{"Problems: " + "; ".join(problems) if problems else "No serious synchronization failures reported. Missing labels remain NA."}

---

## Scientific rules applied

| Rule | Implementation |
|------|----------------|
| Target source | IEM `wxcodes` present-weather tokens (METAR + SPECI) |
| Positive | Any clock hour with ≥1 TS-family token (`TS`, `TSRA`, `-TSRA`, `VCTS`, …) |
| Observed negative | Hour has ≥1 **decodable** present-weather group and no TS-family token |
| Missing | No METAR in the hour, **or** reports exist but `wxcodes` is empty/`M` → `thunderstorm_target` NA |
| Never | Fill missing hours with 0; use Open-Meteo weather_code as target; t+1/t+2/t+3 targets |
| Join | `station_id` + UTC hour; keep rows with valid predictor timestamps (left join from predictors) |

`label_status`: `observed_thunderstorm` / `observed_non_thunderstorm` / `unavailable`.  
`target_observed`: 1 only when a 0/1 label is scientifically assignable.

---

## Per-station validation

{md_table(
    [
        "Station", "City", "Predictor rows", "METAR obs", "Matched rows",
        "Unmatched pred", "Observed targets", "TS+", "Obs −", "Missing target",
        "Pos rate (obs)", "Date range", "Duplicates", "Internal missing hours",
    ],
    rows,
)}

Expected predictor hours in 2014–2025 UTC: **105192** per station (leap years included).

### Predictor missingness (NaN counts)

{md_table(["Station"] + PREDICTORS, miss_rows)}

---

## VOTV vs V1 baseline

| Item | Value |
|------|--------|
| Phase 1B / V1 modelled-window VOTV TS hours | 3694 |
| Phase 2B VOTV `thunderstorm_target==1` | {votv["thunderstorm_positive_rows"]} |
| Delta | {votv["thunderstorm_positive_rows"] - 3694} |
| Phase 2B VOTV observed labels | {votv["observed_target_rows"]} |
| Phase 1B observed hours (any report) | 102816 |

A small delta is expected if Phase 2B **does not** treat hours-with-reports-but-no-wx-group as negatives (Phase 1B / this phase: empty `wxcodes` is not a negative). Positive **hours** should still be close to 3694 because positives require a TS token, which implies a decodable wx group.

---

## Artifacts

| Path | Role |
|------|------|
| `dataset/multilocation/raw_metar/<icao>/` | IEM yearly + combined `wxcodes` extracts |
| `dataset/multilocation/synchronized/<icao>_synchronized_2014_2025.csv` | Per-station join |
| `dataset/multilocation/synchronized/multilocation_synchronized_2014_2025.csv` | Pooled five-station table |
| `dataset/multilocation/synchronized/multilocation_synchronized_2014_2025_metadata.json` | Machine-readable metadata |

Combined rows: **{len(combined)}**. Combined SHA256: `{sha256_of(OUT_COMBINED)}`.

---

## What this phase did not do

- No V1 file edits
- No Flask/frontend changes
- No model training or skill metrics
- No radar / satellite / lightning / NWP
- No fabricated METAR
- No git commit or push
- No t+1/t+2/t+3 label columns
"""

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"\nSTATUS {status}")
    print(f"report {OUT_REPORT}")
    print(f"combined {OUT_COMBINED} rows={len(combined)}")
    return 1 if status == "BLOCKED" else 0


if __name__ == "__main__":
    sys.exit(main())
