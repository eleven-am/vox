"""Response stream synthesis loop for assistant text output."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from vox.conversation.response_stream import ResponseStream
from vox.conversation.text_buffer import StreamingTextBuffer, split_for_tts
from vox.core.adapter import TTSAdapter

AudioStartedCallback = Callable[[], Awaitable[None]]
AudioChunkCallback = Callable[[bytes, int], Awaitable[None]]


async def synthesize_response_stream(
    *,
    adapter: TTSAdapter,
    stream: ResponseStream,
    voice: str | None,
    language: str,
    on_audio_started: AudioStartedCallback,
    on_audio_chunk: AudioChunkCallback,
) -> bool:
    """Drain a committed response stream and synthesize its text incrementally."""

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
    language: str,
    audio_started: bool,
    max_input_chars: int,
    on_audio_started: AudioStartedCallback,
    on_audio_chunk: AudioChunkCallback,
) -> bool:
    for chunk_text in split_for_tts(text, max_chars=max_input_chars):
        chunk_started = False
        async for chunk in adapter.synthesize(
            chunk_text,
            voice=voice,
            language=language,
        ):
            if chunk.is_final and not chunk.audio:
                continue
            if not audio_started:
                audio_started = True
                await on_audio_started()
            chunk_started = True
            await on_audio_chunk(chunk.audio, chunk.sample_rate)
        if chunk_started:
            stream.add_heard_text(chunk_text)

    return audio_started
