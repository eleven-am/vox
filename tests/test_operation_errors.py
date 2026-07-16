from __future__ import annotations

import logging

import pytest
from fastapi import HTTPException

from vox.operations.errors import (
    CatalogEntryNotFoundError,
    EmptyAudioError,
    EmptyInputError,
    InvalidConfigError,
    InvalidRtcCandidateError,
    MemoryBudgetExceededError,
    ModelInUseError,
    NoAudioGeneratedError,
    NoDefaultModelError,
    OperationErrorKind,
    RtcControlAlreadyAttachedError,
    RtcControlTransportMismatchError,
    RtcSessionNotFoundError,
    SessionNotConfiguredError,
    StoredModelNotFoundError,
    UnsupportedFormatError,
    VoiceAudioRequiredError,
    VoiceNotFoundOperationError,
    VoiceReferenceInvalidError,
    VoiceReferenceNotFoundError,
    WrongModelTypeError,
    classify_operation_error,
)
from vox.server.operation_errors import map_operation_errors_to_http, map_route_errors_to_http


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (NoDefaultModelError("stt"), OperationErrorKind.INVALID_ARGUMENT),
        (WrongModelTypeError("kokoro", "stt"), OperationErrorKind.INVALID_ARGUMENT),
        (EmptyAudioError(), OperationErrorKind.INVALID_ARGUMENT),
        (EmptyInputError(), OperationErrorKind.INVALID_ARGUMENT),
        (VoiceAudioRequiredError(), OperationErrorKind.INVALID_ARGUMENT),
        (SessionNotConfiguredError(), OperationErrorKind.INVALID_ARGUMENT),
        (UnsupportedFormatError("audio format", "aac", ["wav"]), OperationErrorKind.INVALID_ARGUMENT),
        (InvalidConfigError("bad config"), OperationErrorKind.INVALID_ARGUMENT),
        (InvalidRtcCandidateError(), OperationErrorKind.INVALID_ARGUMENT),
        (VoiceReferenceInvalidError("bad reference"), OperationErrorKind.UNPROCESSABLE_ENTITY),
        (CatalogEntryNotFoundError("missing:model"), OperationErrorKind.NOT_FOUND),
        (StoredModelNotFoundError("missing:model"), OperationErrorKind.NOT_FOUND),
        (VoiceNotFoundOperationError("voice-1"), OperationErrorKind.NOT_FOUND),
        (VoiceReferenceNotFoundError("voice-1"), OperationErrorKind.NOT_FOUND),
        (RtcSessionNotFoundError("rtc_missing"), OperationErrorKind.NOT_FOUND),
        (RtcControlAlreadyAttachedError("rtc_1"), OperationErrorKind.CONFLICT),
        (
            RtcControlTransportMismatchError(
                "rtc_1",
                expected="pondsocket",
                received="grpc",
            ),
            OperationErrorKind.CONFLICT,
        ),
        (ModelInUseError("parakeet"), OperationErrorKind.CONFLICT),
        (MemoryBudgetExceededError("budget exceeded"), OperationErrorKind.RESOURCE_EXHAUSTED),
        (NoAudioGeneratedError(), OperationErrorKind.INTERNAL),
    ],
)
def test_classify_operation_error(error, kind):
    assert classify_operation_error(error) is kind


def test_map_operation_errors_to_http_maps_operation_errors():
    with pytest.raises(HTTPException) as exc_info, map_operation_errors_to_http():
        raise ModelInUseError("parakeet")

    assert exc_info.value.status_code == 409
    assert "parakeet" in exc_info.value.detail


def test_map_operation_errors_to_http_preserves_success_path():
    with map_operation_errors_to_http():
        value = "ok"

    assert value == "ok"


def test_map_operation_errors_to_http_does_not_hide_unexpected_errors():
    with pytest.raises(RuntimeError, match="boom"), map_operation_errors_to_http():
        raise RuntimeError("boom")


def test_map_route_errors_to_http_preserves_operation_http_mapping():
    with (
        pytest.raises(HTTPException) as exc_info,
        map_route_errors_to_http(
            logger=logging.getLogger("tests.operation_errors"),
            unexpected_detail="internal",
            unexpected_log_message="unexpected route failure",
        ),
    ):
        raise ModelInUseError("parakeet")

    assert exc_info.value.status_code == 409
    assert "parakeet" in exc_info.value.detail


def test_map_route_errors_to_http_converts_unexpected_errors(caplog):
    logger = logging.getLogger("tests.operation_errors")

    with (
        pytest.raises(HTTPException) as exc_info,
        map_route_errors_to_http(
            logger=logger,
            unexpected_detail="internal failure",
            unexpected_log_message="unexpected route failure",
        ),
    ):
        raise RuntimeError("boom")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "internal failure"
    assert "unexpected route failure" in caplog.text
