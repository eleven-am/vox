from __future__ import annotations

from typing import Any

from vox.core.adapter import TTSAdapter
from vox.core.cloned_voices import resolve_voice_request
from vox.core.errors import VoiceCloningUnsupportedError, VoiceNotFoundError
from vox.operations.errors import (
    VoiceCloningUnsupportedOperationError,
    VoiceNotFoundOperationError,
    VoiceReferenceNotFoundError,
)


def resolve_tts_voice_request(
    adapter: TTSAdapter,
    store: Any,
    voice: str | None,
    language: str | None,
) -> tuple[str | None, str | None, Any, str | None]:
    try:
        return resolve_voice_request(adapter, store, voice, language)
    except VoiceNotFoundError as exc:
        raise VoiceNotFoundOperationError(exc.voice_id) from exc
    except VoiceCloningUnsupportedError as exc:
        raise VoiceCloningUnsupportedOperationError(exc.adapter_name) from exc
    except FileNotFoundError as exc:
        raise VoiceReferenceNotFoundError(str(voice or "")) from exc
