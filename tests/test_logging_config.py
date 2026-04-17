"""Tests for vox.logging_config."""

from __future__ import annotations

import io
import logging
import os
from unittest.mock import patch

import pytest

from vox.logging_config import (
    _resolve_format,
    _resolve_level,
    configure_logging,
    reset_for_tests,
)
from vox.logging_context import RequestIdFilter, request_id_var


@pytest.fixture(autouse=True)
def _reset_logging():
    reset_for_tests()
    yield
    reset_for_tests()


class TestResolveLevel:
    def test_default_is_info(self):
        assert _resolve_level(None) == logging.INFO
        assert _resolve_level("") == logging.INFO

    def test_named_levels(self):
        assert _resolve_level("DEBUG") == logging.DEBUG
        assert _resolve_level("info") == logging.INFO
        assert _resolve_level("warning") == logging.WARNING
        assert _resolve_level("ERROR") == logging.ERROR

    def test_numeric_level(self):
        assert _resolve_level("20") == 20

    def test_invalid_falls_back_to_info(self):
        assert _resolve_level("bogus") == logging.INFO


class TestResolveFormat:
    def test_default_auto(self):
        with patch("sys.stdout.isatty", return_value=True):
            assert _resolve_format(None) == "color"
        with patch("sys.stdout.isatty", return_value=False):
            assert _resolve_format(None) == "plain"

    def test_explicit_plain(self):
        assert _resolve_format("plain") == "plain"

    def test_explicit_json(self):
        assert _resolve_format("json") == "json"

    def test_invalid_falls_back_to_auto(self):
        with patch("sys.stdout.isatty", return_value=False):
            assert _resolve_format("xml") == "plain"


class TestConfigureLogging:
    def test_idempotent(self):
        configure_logging()
        handlers_after_first = len(logging.getLogger().handlers)
        configure_logging()
        handlers_after_second = len(logging.getLogger().handlers)
        assert handlers_after_first == handlers_after_second == 1

    def test_respects_vox_log_level_env(self):
        with patch.dict(os.environ, {"VOX_LOG_LEVEL": "DEBUG"}):
            reset_for_tests()
            configure_logging()
            assert logging.getLogger().level == logging.DEBUG

    def test_noisy_loggers_clamped_to_warning(self):
        configure_logging()
        for name in ("urllib3", "httpx", "transformers"):
            assert logging.getLogger(name).level == logging.WARNING

    def test_vox_info_reaches_handler(self, capsys):
        with patch.dict(os.environ, {
            "VOX_LOG_LEVEL": "INFO",
            "VOX_LOG_FORMAT": "plain",
        }):
            reset_for_tests()
            configure_logging()
            logging.getLogger("vox.test").info("hello")
            captured = capsys.readouterr()
            assert "hello" in captured.out

    def test_vox_debug_filtered_at_info_level(self, capsys):
        with patch.dict(os.environ, {
            "VOX_LOG_LEVEL": "INFO",
            "VOX_LOG_FORMAT": "plain",
        }):
            reset_for_tests()
            configure_logging()
            logging.getLogger("vox.test").debug("should not appear")
            captured = capsys.readouterr()
            assert "should not appear" not in captured.out


class TestRequestIdFilter:
    def test_injects_default_dash_when_unset(self):
        record = logging.LogRecord(
            "x", logging.INFO, "f", 1, "msg", None, None,
        )
        RequestIdFilter().filter(record)
        assert record.request_id == "-"

    def test_injects_current_contextvar_value(self):
        token = request_id_var.set("abc123")
        try:
            record = logging.LogRecord(
                "x", logging.INFO, "f", 1, "msg", None, None,
            )
            RequestIdFilter().filter(record)
            assert record.request_id == "abc123"
        finally:
            request_id_var.reset(token)

    def test_emitted_log_line_contains_rid(self):
        with patch.dict(os.environ, {
            "VOX_LOG_LEVEL": "INFO",
            "VOX_LOG_FORMAT": "plain",
        }):
            reset_for_tests()
            configure_logging()

            buffer = io.StringIO()
            handler = logging.StreamHandler(buffer)
            handler.setFormatter(logging.Formatter("%(request_id)s | %(message)s"))
            handler.addFilter(RequestIdFilter())
            logging.getLogger("vox.rid_test").addHandler(handler)
            logging.getLogger("vox.rid_test").setLevel(logging.INFO)

            token = request_id_var.set("xyz789")
            try:
                logging.getLogger("vox.rid_test").info("with rid")
            finally:
                request_id_var.reset(token)

            assert "xyz789 | with rid" in buffer.getvalue()
