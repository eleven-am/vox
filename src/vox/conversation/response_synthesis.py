"""Response stream synthesis loop for assistant text output."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from vox.conversation.response_stream import ResponseStream
from vox.conversation.text_buffer import StreamingTextBuffer, split_for_tts
from vox.core.adapter import TTSAdapter
from vox.core.synthesis_validation import (
    call_accepts_keyword,
    validate_adapter_synthesis_params,
)

AudioStartedCallback = Callable[[], Awaitable[None]]
AudioChunkCallback = Callable[[bytes, int, int], Awaitable[None]]


async def synthesize_response_stream(
    *,
    adapter: TTSAdapter,
    stream: ResponseStream,
    voice: str | None,
    language: str | None,
    speed: float = 1.0,
    params: dict[str, Any] | None = None,
    reference_audio: Any = None,
    reference_text: str | None = None,
    on_audio_started: AudioStartedCallback,
    on_audio_chunk: AudioChunkCallback,
) -> bool:
    """Drain a committed response stream and synthesize its text incrementally."""

    synthesis_params = dict(params or {})
    validate_adapter_synthesis_params(adapter, synthesis_params)
    validation_kwargs: dict[str, Any] = {
        "voice": voice,
        "language": language,
        "reference_audio": reference_audio,
        "reference_text": reference_text,
    }
    if call_accepts_keyword(adapter.validate_synthesis_request, "params"):
        validation_kwargs["params"] = synthesis_params
    adapter.validate_synthesis_request(**validation_kwargs)

    audio_started = False
    text_buffer = StreamingTextBuffer()
    max_input_chars = int(getattr(adapter.info(), "max_input_chars", 0) or 0)

    while True:
        item_text = await stream.next_text()
        if item_text is None:
            break
        for text in text_buffer.push(item_text):
            audio_started = await _synthesize_text(
                adapter,
                text,
                stream=stream,
                voice=voice,
                language=language,
                speed=speed,
                params=synthesis_params,
                reference_audio=reference_audio,
                reference_text=reference_text,
                audio_started=audio_started,
                max_input_chars=max_input_chars,
                on_audio_started=on_audio_started,
                on_audio_chunk=on_audio_chunk,
            )

    for text in text_buffer.flush():
        audio_started = await _synthesize_text(
            adapter,
            text,
            stream=stream,
            voice=voice,
            language=language,
            speed=speed,
            params=synthesis_params,
            reference_audio=reference_audio,
            reference_text=reference_text,
            audio_started=audio_started,
            max_input_chars=max_input_chars,
            on_audio_started=on_audio_started,
            on_audio_chunk=on_audio_chunk,
        )

    return audio_started


async def _synthesize_text(
    adapter: TTSAdapter,
    text: str,
    *,
    stream: ResponseStream,
    voice: str | None,
    language: str | None,
    speed: float,
    params: dict[str, Any],
    reference_audio: Any,
    reference_text: str | None,
    audio_started: bool,
    max_input_chars: int,
    on_audio_started: AudioStartedCallback,
    on_audio_chunk: AudioChunkCallback,
) -> bool:
    for chunk_text in split_for_tts(text, max_chars=max_input_chars):
        span_id = stream.spoken_history.begin_span(chunk_text)
        chunk_started = False
        synthesis_kwargs: dict[str, Any] = {
            "voice": voice,
            "speed": speed,
            "language": language,
            "reference_audio": reference_audio,
            "reference_text": reference_text,
        }
        if call_accepts_keyword(adapter.synthesize, "params"):
            synthesis_kwargs["params"] = params
        chunks = adapter.synthesize(chunk_text, **synthesis_kwargs)
        async for chunk in adapter.iterate_synthesis(chunks):
            if chunk.is_final and not chunk.audio:
                continue
            if not audio_started:
                audio_started = True
                await on_audio_started()
            chunk_started = True
            await on_audio_chunk(chunk.audio, chunk.sample_rate, span_id)
        if chunk_started:
            stream.spoken_history.finish_span(span_id)
        else:
            stream.spoken_history.discard_span(span_id)

    return audio_started
