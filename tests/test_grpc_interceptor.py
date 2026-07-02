from __future__ import annotations

import logging

import pytest

from vox.grpc.interceptor import _extract_rid, _grpc_request_scope
from vox.logging_context import request_id_var


def test_extract_rid_accepts_string_and_bytes_metadata():
    assert _extract_rid((("x-request-id", "  rid-string  "),)) == "rid-string"
    assert _extract_rid((("X-Request-ID", b"  rid-bytes  "),)) == "rid-bytes"


def test_extract_rid_ignores_missing_and_blank_metadata():
    assert _extract_rid(None) is None
    assert _extract_rid((("other", "rid"),)) is None
    assert _extract_rid((("x-request-id", "   "),)) is None


def test_grpc_request_scope_sets_resets_and_logs_success(caplog):
    caplog.set_level(logging.INFO, logger="vox.grpc.request")
    outer = request_id_var.set("outer")
    try:
        with _grpc_request_scope("/vox.Test/Call", "  inbound-rid  "):
            assert request_id_var.get() == "inbound-rid"

        assert request_id_var.get() == "outer"
    finally:
        request_id_var.reset(outer)

    assert any("/vox.Test/Call OK" in record.message for record in caplog.records)


def test_grpc_request_scope_resets_and_logs_errors(caplog):
    caplog.set_level(logging.INFO, logger="vox.grpc.request")
    outer = request_id_var.set("outer")
    try:
        with pytest.raises(RuntimeError, match="boom"):
            with _grpc_request_scope("/vox.Test/Call", "inbound-rid"):
                assert request_id_var.get() == "inbound-rid"
                raise RuntimeError("boom")

        assert request_id_var.get() == "outer"
    finally:
        request_id_var.reset(outer)

    assert any("/vox.Test/Call ERROR" in record.message for record in caplog.records)
