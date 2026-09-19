"""Phase 2 (step 1): download historical VOTV METAR observations from IEM ASOS.

Source: Iowa Environmental Mesonet (IEM) ASOS/METAR archive request service,
which serves the raw METAR body plus a pre-parsed ``wxcodes`` column holding the
OBSERVED present-weather group (not trend/forecast groups such as TEMPO/BECMG).

This script only WRITES NEW files under dataset/raw_metar/. It never modifies or
deletes any pre-existing project file.

Usage:
    python download_votv_metar_iem.py            # resume-friendly
    python download_votv_metar_iem.py --force    # re-download all chunks
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

STATION = "VOTV"
START_YEAR = 2000
END_YEAR = 2025
BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
USER_AGENT = "SIH26072-phase2-metar-acquisition/1.0 (research use)"
MAX_ATTEMPTS = 4

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw_metar")
LOG_PATH = os.path.join(RAW_DIR, "votv_metar_iem_download_log.json")


def build_url(year: int) -> str:
    params = [
        ("station", STATION),
        ("data", "all"),
        ("year1", str(year)),
        ("month1", "1"),
        ("day1", "1"),
        ("year2", str(year)),
        ("month2", "12"),
        ("day2", "31"),
        ("tz", "UTC"),
        ("format", "onlycomma"),
        ("latlon", "yes"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "no"),
        # 3 = routine METAR, 4 = SPECI (special observation).
        ("report_type", "3"),
        ("report_type", "4"),
    ]
    return BASE_URL + "?" + urllib.parse.urlencode(params, doseq=True)


def fetch(url: str) -> str:
    """Fetch and fully validate a response body.

    IEM occasionally closes a chunked response early; a truncated body would
    silently lose observations, so any partial transfer is treated as a failure
    and retried rather than written to disk.
    """
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
                    # Some bytes did arrive, but the transfer is incomplete.
                    raise OSError(f"incomplete read ({exc.partial.__len__()} partial bytes)") from exc
                body = b"".join(chunks).decode("utf-8", errors="replace")
            validate_body(body)
            return body
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_err = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(3.0 * attempt)
    raise RuntimeError(f"download failed after {MAX_ATTEMPTS} attempts: {last_err}")


def validate_body(body: str) -> None:
    """Reject empty, headerless, or field-count-inconsistent responses."""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("empty response body")
    header = lines[0].rstrip("\r\n")
    if not header.startswith("station,valid,"):
        raise RuntimeError(f"unexpected header: {header[:120]!r}")
    expected = len(next(csv.reader([header])))
    for i, line in enumerate(lines[1:], start=2):
        if len(next(csv.reader([line]))) != expected:
            raise RuntimeError(f"ragged row at line {i}: {line[:120]!r}")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count_data_rows(path: str) -> int:
    """Count data rows using the csv module so embedded commas in METAR text
    cannot inflate/deflate the count spuriously."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def chunk_is_valid(path: str) -> bool:
    """A cached chunk is reusable only if it is non-empty, has the expected
    header, and every row has the header's field count."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return False
        if len(header) < 2 or header[0] != "station" or header[1] != "valid":
            return False
        expected = len(header)
        return all(len(row) == expected for row in reader)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download existing chunks")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    log = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "IEM ASOS/METAR archive request service",
        "service_url": BASE_URL,
        "station": STATION,
        "requested_range_utc": [f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"],
        "report_types": {"3": "routine METAR", "4": "SPECI"},
        "timezone_requested": "UTC",
        "chunks": [],
    }

    header: str | None = None
    chunk_paths: list[str] = []

    for year in range(START_YEAR, END_YEAR + 1):
        path = os.path.join(RAW_DIR, f"votv_metar_iem_{year}.csv")
        chunk_paths.append(path)

        if chunk_is_valid(path) and not args.force:
            exists = True
        else:
            exists = False

        if not exists:
            url = build_url(year)
            body = fetch(url)
            lines = [ln for ln in body.splitlines() if ln.strip()]
            if header is None:
                header = lines[0]
            tmp = path + ".part"
            with open(tmp, "w", encoding="utf-8", newline="") as fh:
                fh.write(body if body.endswith("\n") else body + "\n")
            os.replace(tmp, path)
            print(f"[{year}] downloaded {len(lines) - 1} data rows", flush=True)
            time.sleep(0.5)

        rows = count_data_rows(path)
        with open(path, "r", encoding="utf-8", newline="") as fh:
            chunk_header = fh.readline().rstrip("\r\n")
        if header is None:
            header = chunk_header
        log["chunks"].append(
            {
                "year": year,
                "file": os.path.basename(path),
                "data_rows": rows,
                "bytes": os.path.getsize(path),
                "sha256": sha256_of(path),
                "header_matches_expected": chunk_header == header,
            }
        )
        print(f"[{year}] rows={rows}", flush=True)

    combined = os.path.join(RAW_DIR, "votv_metar_iem_raw.csv")
    total_rows = 0
    with open(combined, "w", encoding="utf-8", newline="") as out:
        out.write((header or "station,valid,lon,lat") + "\n")
        for path in chunk_paths:
            with open(path, "r", encoding="utf-8", newline="") as fh:
                fh.readline()  # drop per-chunk header
                for line in fh:
                    if line.strip():
                        out.write(line)
                        total_rows += 1

    log["combined_file"] = os.path.basename(combined)
    log["combined_rows"] = total_rows
    log["combined_sha256"] = sha256_of(combined)
    log["combined_bytes"] = os.path.getsize(combined)
    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)

    print(f"\ncombined rows: {total_rows}")
    print(f"raw dir: {RAW_DIR}")
    print(f"log: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
