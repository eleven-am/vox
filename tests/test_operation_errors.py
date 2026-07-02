from __future__ import annotations

import pytest

from vox.operations.errors import (
    CatalogEntryNotFoundError,
    EmptyAudioError,
    EmptyInputError,
    InvalidConfigError,
    ModelInUseError,
    NoAudioGeneratedError,
    NoDefaultModelError,
    OperationErrorKind,
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
        (VoiceReferenceInvalidError("bad reference"), OperationErrorKind.UNPROCESSABLE_ENTITY),
        (CatalogEntryNotFoundError("missing:model"), OperationErrorKind.NOT_FOUND),
        (StoredModelNotFoundError("missing:model"), OperationErrorKind.NOT_FOUND),
        (VoiceNotFoundOperationError("voice-1"), OperationErrorKind.NOT_FOUND),
        (VoiceReferenceNotFoundError("voice-1"), OperationErrorKind.NOT_FOUND),
        (ModelInUseError("parakeet"), OperationErrorKind.CONFLICT),
        (NoAudioGeneratedError(), OperationErrorKind.INTERNAL),
    ],
)
def test_classify_operation_error(error, kind):
    assert classify_operation_error(error) is kind
