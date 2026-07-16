from __future__ import annotations

from enum import StrEnum


class OperationErrorKind(StrEnum):
    UNAUTHENTICATED = "unauthenticated"
    INVALID_ARGUMENT = "invalid_argument"
    UNPROCESSABLE_ENTITY = "unprocessable_entity"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    INTERNAL = "internal"


class OperationError(Exception):
    """Base class for transport-agnostic errors raised by operation modules."""


class InternalOperationError(OperationError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class RtcSessionNotFoundError(OperationError):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"RTC session '{session_id}' not found")


class RtcControlAlreadyAttachedError(OperationError):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"RTC session '{session_id}' already has an attached control transport")


class RtcControlTransportMismatchError(OperationError):
    def __init__(self, session_id: str, *, expected: str, received: str) -> None:
        self.session_id = session_id
        self.expected = expected
        self.received = received
        super().__init__(f"RTC session '{session_id}' requires {expected} control; received {received}")


class InvalidRtcCandidateError(OperationError):
    def __init__(self) -> None:
        super().__init__("invalid ICE candidate")


class NoDefaultModelError(OperationError):
    def __init__(self, model_type: str) -> None:
        self.model_type = model_type
        super().__init__(f"no model specified and no default {model_type.upper()} model available")


class WrongModelTypeError(OperationError):
    def __init__(self, model: str, expected: str) -> None:
        self.model = model
        self.expected = expected
        super().__init__(f"Model '{model}' is not {'an' if expected.lower() == 'stt' else 'a'} {expected} model")


class EmptyAudioError(OperationError):
    def __init__(self) -> None:
        super().__init__("No audio data provided")


class EmptyInputError(OperationError):
    def __init__(self) -> None:
        super().__init__("No input text provided")


class NoAudioGeneratedError(OperationError):
    def __init__(self) -> None:
        super().__init__("No audio generated")


class ModelInUseError(OperationError):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Model '{model}' is currently in use")


class MemoryBudgetExceededError(OperationError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class CatalogEntryNotFoundError(OperationError):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Model '{model}' not found in catalog")


class AdapterInstallError(OperationError):
    def __init__(self, package: str) -> None:
        self.package = package
        super().__init__(f"Failed to install adapter package: {package}")


class ModelIncompatibleError(OperationError):
    def __init__(self, model: str, reasons: list[str]) -> None:
        self.model = model
        self.reasons = reasons
        detail = "; ".join(reasons)
        super().__init__(
            f"Model '{model}' cannot run in this environment: {detail}. Set VOX_ALLOW_INCOMPATIBLE=1 to pull anyway."
        )


class StoredModelNotFoundError(OperationError):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Model '{model}' not found")


class VoiceNameRequiredError(OperationError):
    def __init__(self) -> None:
        super().__init__("Voice name is required")


class VoiceAudioRequiredError(OperationError):
    def __init__(self) -> None:
        super().__init__("Audio sample is required")


class VoiceReferenceInvalidError(OperationError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class VoiceIdRequiredError(OperationError):
    def __init__(self) -> None:
        super().__init__("Voice ID is required")


class VoiceNotFoundOperationError(OperationError):
    def __init__(self, voice_id: str) -> None:
        self.voice_id = voice_id
        super().__init__(f"Voice '{voice_id}' not found")


class VoiceReferenceNotFoundError(OperationError):
    def __init__(self, voice_id: str) -> None:
        self.voice_id = voice_id
        super().__init__(f"Reference audio for voice '{voice_id}' not found")


class VoiceCloningUnsupportedOperationError(OperationError):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"model '{model}' does not support cloned voices")


class SessionAlreadyConfiguredError(OperationError):
    def __init__(self) -> None:
        super().__init__("Session already configured")


class SessionNotConfiguredError(OperationError):
    def __init__(self) -> None:
        super().__init__("Session not configured")


class UnknownMessageTypeError(OperationError):
    def __init__(self, msg_type: str) -> None:
        self.msg_type = msg_type
        super().__init__(f"Unknown message type: {msg_type}")


class UnsupportedFormatError(OperationError):
    def __init__(self, kind: str, value: str, supported: list[str]) -> None:
        self.kind = kind
        self.value = value
        self.supported = supported
        super().__init__(f"Unsupported {kind} '{value}'. Supported values: {sorted(supported)}")


class InvalidConfigError(OperationError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def classify_operation_error(exc: OperationError) -> OperationErrorKind:
    if isinstance(
        exc,
        (
            NoDefaultModelError,
            WrongModelTypeError,
            EmptyAudioError,
            EmptyInputError,
            VoiceNameRequiredError,
            VoiceAudioRequiredError,
            VoiceIdRequiredError,
            VoiceCloningUnsupportedOperationError,
            SessionAlreadyConfiguredError,
            SessionNotConfiguredError,
            UnknownMessageTypeError,
            UnsupportedFormatError,
            InvalidConfigError,
            InvalidRtcCandidateError,
        ),
    ):
        return OperationErrorKind.INVALID_ARGUMENT
    if isinstance(exc, VoiceReferenceInvalidError):
        return OperationErrorKind.UNPROCESSABLE_ENTITY
    if isinstance(
        exc,
        (
            CatalogEntryNotFoundError,
            StoredModelNotFoundError,
            VoiceNotFoundOperationError,
            VoiceReferenceNotFoundError,
            RtcSessionNotFoundError,
        ),
    ):
        return OperationErrorKind.NOT_FOUND
    if isinstance(
        exc,
        (
            ModelInUseError,
            ModelIncompatibleError,
            RtcControlAlreadyAttachedError,
            RtcControlTransportMismatchError,
        ),
    ):
        return OperationErrorKind.CONFLICT
    if isinstance(exc, MemoryBudgetExceededError):
        return OperationErrorKind.RESOURCE_EXHAUSTED
    return OperationErrorKind.INTERNAL
