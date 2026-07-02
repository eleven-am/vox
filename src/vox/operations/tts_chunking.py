from __future__ import annotations

from typing import Any

from vox.conversation.text_buffer import split_for_tts


def tts_adapter_text_cap(adapter: Any) -> int:
    return max(0, int(getattr(adapter.info(), "max_input_chars", 0) or 0))


def effective_tts_text_cap(adapter: Any, override_chars: int | None = None) -> int:
    if override_chars is not None:
        return max(0, int(override_chars))
    return tts_adapter_text_cap(adapter)


def split_text_for_tts_adapter(
    text: str,
    adapter: Any,
    *,
    override_chars: int | None = None,
) -> list[str]:
    if not text.strip():
        return []
    cap = effective_tts_text_cap(adapter, override_chars)
    if cap <= 0:
        return [text]
    return split_for_tts(text, max_chars=cap)
