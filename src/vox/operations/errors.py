from __future__ import annotations


class OperationError(Exception):
    """Base class for transport-agnostic errors raised by operation modules."""


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


class CatalogEntryNotFoundError(OperationError):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Model '{model}' not found in catalog")


class AdapterInstallError(OperationError):
    def __init__(self, package: str) -> None:
        self.package = package
        super().__init__(f"Failed to install adapter package: {package}")


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
