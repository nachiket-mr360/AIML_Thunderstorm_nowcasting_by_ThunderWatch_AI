#!/usr/bin/env python3
"""Download INSAT-3DR 3RIMG_L2B_CMK granules from the frozen Phase 10B-D manifest.

Official MOSDAC mdapi flow (Phase 10B-E):
  POST https://mosdac.gov.in/download_api/gettoken  JSON {username, password}
  GET  https://mosdac.gov.in/download_api/download?id=<gId>  Authorization: Bearer

Credentials: untracked project .env (MOSDAC_USERNAME, MOSDAC_PASSWORD).
Never print password or access/refresh tokens.

This module does not run a download on import. Invoke from CLI with --limit.
Does not modify the manifest or any model.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "satellite" / "phase10b" / "cmk_download_manifest.csv"
OUT_DIR = ROOT / "dataset" / "satellite" / "insat3dr_cmk"
ENV_PATH = ROOT / ".env"
LOG_PATH = ROOT / "outputs" / "satellite" / "phase10b" / "cmk_batch_download_log.json"

TOKEN_URL = "https://mosdac.gov.in/download_api/gettoken"
DOWNLOAD_URL = "https://mosdac.gov.in/download_api/download"
LOGOUT_URL = "https://mosdac.gov.in/download_api/logout"
UA = "ThunderWatch-SIH26072-cmk-batch/1"

REQUIRED_FILENAME_MARKERS = ("3RIMG", "L2B_CMK")
MAX_DOWNLOAD_ATTEMPTS = 3
RETRY_DELAYS_SEC = (5, 15, 30)
RETRYABLE_REQUEST_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


class DownloadError(RuntimeError):
    """Safe per-file download failure. Never includes credentials, tokens, or bodies."""

    def __init__(
        self,
        reason: str,
        status_code: int | None = None,
        content_type: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code
        self.content_type = content_type


def _safe_content_type(value: str | None) -> str | None:
    if not value:
        return None
    return value.split(";")[0].strip().lower() or None


def fail_diagnostics(exc: BaseException) -> dict[str, object]:
    reason = str(exc) if str(exc) else type(exc).__name__
    status = getattr(exc, "status_code", None)
    ctype = getattr(exc, "content_type", None)
    if ctype is not None:
        ctype = _safe_content_type(str(ctype))
    return {
        "error": type(exc).__name__,
        "reason": reason,
        "http_status": status,
        "content_type": ctype,
    }


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        raise FileNotFoundError(f"missing credentials file: {path.name}")
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        gid = (row.get("gId") or "").strip()
        filename = (row.get("filename") or "").strip()
        if not gid or not filename:
            continue
        if not all(m in filename for m in REQUIRED_FILENAME_MARKERS):
            continue
        out.append(row)
    return out


def already_downloaded(dest: Path) -> bool:
    return dest.is_file() and dest.stat().st_size > 0


def get_token(username: str, password: str) -> str:
    response = requests.post(
        TOKEN_URL,
        json={"username": username, "password": password},
        headers={"User-Agent": UA},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"gettoken_http_{response.status_code}")
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("gettoken_no_access_token")
    return token


class TokenSession:
    """One access token, reused until a download 401 triggers a single gettoken renew."""

    def __init__(self, username: str, password: str) -> None:
        self._username = username
        self._password = password
        self.token: str | None = None

    def acquire(self) -> str:
        self.token = get_token(self._username, self._password)
        return self.token

    def renew(self) -> str:
        return self.acquire()


def logout(token: str) -> None:
    try:
        requests.post(
            LOGOUT_URL,
            json={"access_token": token},
            headers={"User-Agent": UA, "Authorization": f"Bearer {token}"},
            timeout=30,
        )
    except requests.RequestException:
        return


def is_http_401(exc: BaseException) -> bool:
    return isinstance(exc, DownloadError) and exc.status_code == 401


def is_retryable_download_error(exc: BaseException) -> bool:
    """Retry network/server faults only. Never retry HTTP 4xx or auth failures."""
    if isinstance(exc, RETRYABLE_REQUEST_ERRORS):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        if msg.startswith("download_http_"):
            try:
                code = int(msg.rsplit("_", 1)[-1])
            except ValueError:
                return False
            return 500 <= code <= 599
        if msg == "empty_download":
            return True
    return False


def download_one(token: str, gid: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    headers = {"Authorization": f"Bearer {token}", "User-Agent": UA}
    response = requests.get(
        DOWNLOAD_URL,
        headers=headers,
        params={"id": gid},
        stream=True,
        timeout=180,
    )
    status = response.status_code
    ctype = _safe_content_type(response.headers.get("Content-Type"))
    if status != 200:
        raise DownloadError(f"download_http_{status}", status_code=status, content_type=ctype)
    if ctype and ("json" in ctype or "html" in ctype):
        raise DownloadError(
            f"unexpected_content_type_{ctype or 'empty'}",
            status_code=status,
            content_type=ctype,
        )

    nbytes = 0
    try:
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
                    nbytes += len(chunk)
        if nbytes <= 0:
            raise DownloadError("empty_download", status_code=status, content_type=ctype)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return nbytes


def download_one_with_retries(session: TokenSession, gid: str, dest: Path) -> int:
    last_exc: BaseException | None = None
    auth_refreshed = False
    attempt = 1
    while attempt <= MAX_DOWNLOAD_ATTEMPTS:
        if not session.token:
            session.acquire()
        try:
            return download_one(session.token, gid, dest)
        except Exception as exc:
            last_exc = exc
            if is_http_401(exc) and not auth_refreshed:
                print(f"AUTH_REFRESH gId={gid} reason=download_http_401", flush=True)
                session.renew()
                auth_refreshed = True
                continue
            if not is_retryable_download_error(exc) or attempt >= MAX_DOWNLOAD_ATTEMPTS:
                if is_retryable_download_error(exc) and attempt >= MAX_DOWNLOAD_ATTEMPTS:
                    time.sleep(RETRY_DELAYS_SEC[attempt - 1])
                raise
            wait = RETRY_DELAYS_SEC[attempt - 1]
            print(
                f"RETRY gId={gid} attempt={attempt}/{MAX_DOWNLOAD_ATTEMPTS} "
                f"wait={wait}s {type(exc).__name__}",
                flush=True,
            )
            time.sleep(wait)
            attempt += 1
    assert last_exc is not None
    raise last_exc


def parse_gids_arg(raw: str | None) -> list[str]:
    if not raw:
        return []
    seen: list[str] = []
    for part in raw.split(","):
        gid = part.strip()
        if gid and gid not in seen:
            seen.append(gid)
    return seen


def filter_manifest_by_gids(
    rows: list[dict[str, str]], requested: list[str]
) -> tuple[list[dict[str, str]], list[str]]:
    by_gid = {(row.get("gId") or "").strip(): row for row in rows}
    matched: list[dict[str, str]] = []
    missing: list[str] = []
    used: set[str] = set()
    for gid in requested:
        row = by_gid.get(gid)
        if row is None or gid in used:
            if row is None:
                missing.append(gid)
            continue
        used.add(gid)
        matched.append(row)
    return matched, missing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download missing CMK granules from a frozen manifest. "
            "Default is the 1094-file Phase 10B-D manifest; pass --manifest to use "
            "the reduced MVP CSV (or any compatible gId/filename CSV)."
        )
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of new files to download this run (skip already-present files).",
    )
    p.add_argument(
        "--gids",
        type=str,
        default=None,
        help="Optional comma-separated gId list; download only those manifest rows (diagnostic).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help=(
            "Manifest CSV with gId and filename columns (not modified). "
            "Default: outputs/satellite/phase10b/cmk_download_manifest.csv (1094). "
            "MVP: outputs/satellite/phase10b/cmk_mvp_download_manifest.csv"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for HDF5 files.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested_gids = parse_gids_arg(args.gids)
    if args.limit is None and not requested_gids:
        print("ERROR --limit is required unless --gids is supplied", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("ERROR --limit must be >= 1", file=sys.stderr)
        return 2

    env = load_env(ENV_PATH)
    username = env.get("MOSDAC_USERNAME") or ""
    password = env.get("MOSDAC_PASSWORD") or ""
    if not username or not password:
        print("ERROR MOSDAC_USERNAME or MOSDAC_PASSWORD missing in .env", file=sys.stderr)
        return 2

    rows = load_manifest(args.manifest)
    if requested_gids:
        rows, missing_gids = filter_manifest_by_gids(rows, requested_gids)
        if missing_gids:
            print(
                f"ERROR gIds not in frozen manifest: {','.join(missing_gids)}",
                file=sys.stderr,
            )
            if not rows:
                return 2
        print(
            f"gids_mode n={len(rows)} requested={len(requested_gids)}",
            flush=True,
        )
        if args.limit is None:
            args.limit = max(len(rows), 1)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    successful: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, object]] = []
    attempted = 0
    session: TokenSession | None = None

    print(
        f"manifest={args.manifest.as_posix()} manifest_rows={len(rows)} "
        f"limit={args.limit} out_dir={out_dir.as_posix()}",
        flush=True,
    )

    try:
        session = TokenSession(username, password)
        session.acquire()
        for row in rows:
            if attempted >= args.limit:
                break
            filename = row["filename"].strip()
            gid = row["gId"].strip()
            dest = out_dir / filename
            if already_downloaded(dest):
                skipped.append(filename)
                print(f"SKIP {filename}", flush=True)
                continue
            attempted += 1
            try:
                n = download_one_with_retries(session, gid, dest)
                successful.append(filename)
                print(f"OK gId={gid} bytes={n} {filename}", flush=True)
            except Exception as exc:
                diag = fail_diagnostics(exc)
                failed.append({"gId": gid, "filename": filename, **diag})
                print(
                    f"FAIL gId={gid} {diag['error']} {filename} "
                    f"reason={diag['reason']} http_status={diag['http_status']} "
                    f"content_type={diag['content_type']}",
                    flush=True,
                )
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        if session and session.token:
            logout(session.token)

    processed = len(successful) + len(skipped) + len(failed)
    summary = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": args.limit,
        "successful": len(successful),
        "skipped": len(skipped),
        "failed": len(failed),
        "total_processed": processed,
        "successful_filenames": successful,
        "skipped_filenames": skipped,
        "failed_rows": failed,
        "out_dir": out_dir.as_posix(),
        "manifest": args.manifest.as_posix(),
        "manifest_not_modified": True,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"successful={len(successful)} skipped={len(skipped)} "
        f"failed={len(failed)} total_processed={processed}",
        flush=True,
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
