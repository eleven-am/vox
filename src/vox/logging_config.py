"""Process-wide logging setup for Vox.

`configure_logging()` installs a single stdout handler on the root logger and
threads the request-ID filter through every record. Safe to call multiple times.

Environment knobs:
  * ``VOX_LOG_LEVEL`` — ``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``
    (default ``INFO``).
  * ``VOX_LOG_FORMAT`` — ``auto`` / ``plain`` / ``color`` / ``json``
    (default ``auto`` = color if stdout is a TTY, else plain).

Noisy third-party loggers (huggingface_hub, transformers, urllib3, httpx …)
are clamped to WARNING so startup isn't drowned in their INFO spam.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import sys
from typing import Any
from urllib.parse import unquote_plus

from vox.logging_context import RequestIdFilter

_CONFIGURED = False

_NOISY_LOGGERS: tuple[str, ...] = (
    "urllib3",
    "httpx",
    "httpcore",
    "huggingface_hub",
    "transformers",
    "filelock",
    "asyncio",
    "multipart",
)

_DEFAULT_FMT = "%(asctime)s %(levelname)-8s %(name)s [rid=%(request_id)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"
_QUERY_PARAMETER_PATTERN = re.compile(r"([?&])([^=&#\s\"']+)=([^&#\s\"']*)")
_QUERY_CREDENTIAL_NAMES = frozenset(
    {
        "api_key",
        "access_token",
        "client_token",
        "media_token",
        "token",
    }
)


def _redact_query_credentials(message: str) -> str:
    def replace(match: re.Match[str]) -> str:
        if unquote_plus(match.group(2)).lower() not in _QUERY_CREDENTIAL_NAMES:
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}=[REDACTED]"

    return _QUERY_PARAMETER_PATTERN.sub(replace, message)


def _resolve_level(raw: str | None) -> int:
    if not raw:
        return logging.INFO
    candidate = raw.strip()
    if candidate.isdigit():
        return int(candidate)
    level = logging.getLevelName(candidate.upper())
    return level if isinstance(level, int) else logging.INFO


def _resolve_format(raw: str | None) -> str:
    val = (raw or "auto").strip().lower()
    if val not in {"auto", "plain", "color", "json"}:
        val = "auto"
    if val == "auto":
        return "color" if sys.stdout.isatty() else "plain"
    return val


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter — one line per record, stable key order."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt=_DEFAULT_DATEFMT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "rid": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _CredentialRedactingFormatter(logging.Formatter):
    def __init__(self, formatter: logging.Formatter) -> None:
        self._formatter = formatter

    def format(self, record: logging.LogRecord) -> str:
        return _redact_query_credentials(self._formatter.format(record))


def _build_formatter(kind: str) -> logging.Formatter:
    formatter: logging.Formatter
    if kind == "color":
        try:
            colorlog = importlib.import_module("colorlog")
            formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s "
                "%(cyan)s%(name)s%(reset)s [rid=%(request_id)s] %(message)s",
                datefmt=_DEFAULT_DATEFMT,
            )
        except ImportError:
            formatter = logging.Formatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT)
    elif kind == "json":
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT)
    return _CredentialRedactingFormatter(formatter)


def _build_handler(kind: str) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_build_formatter(kind))
    handler.addFilter(RequestIdFilter())
    return handler


def configure_logging() -> None:
    """Install the Vox root handler. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = _resolve_level(os.environ.get("VOX_LOG_LEVEL"))
    fmt = _resolve_format(os.environ.get("VOX_LOG_FORMAT"))

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.addHandler(_build_handler(fmt))
    root.setLevel(level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def reset_for_tests() -> None:
    """Drop the configured state so tests can reconfigure cleanly."""
    global _CONFIGURED
    _CONFIGURED = False
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
