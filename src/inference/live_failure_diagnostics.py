"""Sanitized live-fetch failure categories. No secrets, traces, URLs, or paths."""

from __future__ import annotations

import os
import re
import socket
import ssl
import urllib.error
from typing import Any

ALLOWED_CATEGORIES = frozenset(
    {
        "DNS_ERROR",
        "TIMEOUT",
        "HTTP_4XX",
        "HTTP_5XX",
        "SSL_ERROR",
        "CONNECTION_ERROR",
        "INVALID_RESPONSE",
        "FEATURE_BUILD_ERROR",
        "UNKNOWN_EXTERNAL_ERROR",
    }
)

FEATURE_CODES = frozenset(
    {
        "LIVE_FEATURE_CONSTRUCTION_FAILED",
        "INSUFFICIENT_HISTORY",
        "LIVE_MODEL_INPUT_INVALID",
    }
)

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_WINPATH_RE = re.compile(r"[A-Za-z]:\\[^\s]+")
_UNIXPATH_RE = re.compile(r"(?:/[\w.-]+){2,}")
_ENV_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}=[^\s]+")
_SECRETISH_RE = re.compile(r"(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*\S+", re.I)

TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def live_diagnostics_enabled(environ: dict[str, str] | None = None) -> bool:
    env = environ if environ is not None else os.environ
    return str(env.get("LIVE_DIAGNOSTICS", "")).strip().lower() in TRUE_VALUES


def sanitize_failure_category(value: Any) -> str:
    if not isinstance(value, str):
        return "UNKNOWN_EXTERNAL_ERROR"
    token = value.strip().upper()
    if token in ALLOWED_CATEGORIES:
        return token
    return "UNKNOWN_EXTERNAL_ERROR"


def _haystack(exc: BaseException) -> str:
    parts = [type(exc).__name__, str(exc)]
    reason = getattr(exc, "reason", None)
    if reason is not None:
        parts.append(type(reason).__name__)
        parts.append(str(reason))
    return " ".join(parts).lower()


def classify_live_exception(exc: BaseException) -> str:
    preset = getattr(exc, "category", None)
    if isinstance(preset, str) and preset.strip().upper() in ALLOWED_CATEGORIES:
        return preset.strip().upper()

    code = getattr(exc, "code", None)
    if isinstance(code, str) and code in FEATURE_CODES:
        return "FEATURE_BUILD_ERROR"

    if isinstance(exc, TimeoutError):
        return "TIMEOUT"
    if isinstance(exc, ssl.SSLError):
        return "SSL_ERROR"
    if isinstance(exc, socket.gaierror):
        return "DNS_ERROR"
    if isinstance(exc, urllib.error.HTTPError):
        status = int(getattr(exc, "code", 0) or 0)
        if 400 <= status < 500:
            return "HTTP_4XX"
        if status >= 500:
            return "HTTP_5XX"
        return "UNKNOWN_EXTERNAL_ERROR"
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, TimeoutError):
            return "TIMEOUT"
        if isinstance(reason, ssl.SSLError):
            return "SSL_ERROR"
        if isinstance(reason, socket.gaierror):
            return "DNS_ERROR"
        text = _haystack(exc)
        if "timed out" in text or "timeout" in text:
            return "TIMEOUT"
        if "ssl" in text or "certificate" in text:
            return "SSL_ERROR"
        if any(
            m in text
            for m in (
                "getaddrinfo",
                "name or service not known",
                "nodename nor servname",
                "dns",
                "temporary failure in name resolution",
                "nameresolutionerror",
            )
        ):
            return "DNS_ERROR"
        return "CONNECTION_ERROR"
    if isinstance(exc, OSError):
        text = _haystack(exc)
        if "timed out" in text or "timeout" in text:
            return "TIMEOUT"
        return "CONNECTION_ERROR"

    text = _haystack(exc)
    if "json" in text or "malformed" in text:
        return "INVALID_RESPONSE"
    return "UNKNOWN_EXTERNAL_ERROR"


def sanitize_provider(value: Any) -> str | None:
    if value in ("forecast", "gfs"):
        return value
    return None


def station_diagnostic_entry(
    station_id: str,
    *,
    ok: bool,
    category: Any = None,
    provider: Any = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "station_id": str(station_id),
        "ok": bool(ok),
        "provider": None,
        "category": None,
    }
    if not ok:
        entry["provider"] = sanitize_provider(provider)
        entry["category"] = sanitize_failure_category(category)
    return entry


def assert_no_sensitive_payload(payload: Any) -> None:
    """Test helper: diagnostics must not leak urls, paths, env, or traces."""
    blob = str(payload)
    assert not _URL_RE.search(blob)
    assert not _WINPATH_RE.search(blob)
    assert not _ENV_RE.search(blob)
    assert not _SECRETISH_RE.search(blob)
    assert "traceback" not in blob.lower()
    assert "Traceback" not in blob
