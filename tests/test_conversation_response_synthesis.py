from __future__ import annotations

import pytest

from vox.conversation.response_stream import ResponseStream
from vox.conversation.response_synthesis import synthesize_response_stream
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo


class RecordingTTSAdapter(TTSAdapter):
    def __init__(self, *, max_input_chars: int = 0, emit_audio: bool = True) -> None:
        self.max_input_chars = max_input_chars
        self.emit_audio = emit_audio
        self.calls: list[tuple[str, str | None, str]] = []

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="recording",
            type=ModelType.TTS,
            architectures=("test",),
            default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
            max_input_chars=self.max_input_chars,
        )

    def load(self, *_args, **_kwargs) -> None: ...

    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def list_voices(self):
        return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text: str, *, voice=None, language="en", **_kwargs):
        self.calls.append((text, voice, language))
        if self.emit_audio:
            yield SynthesizeChunk(audio=b"\x01\x02", sample_rate=24_000, is_final=False)
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


async def _run_synthesis(adapter: RecordingTTSAdapter, *texts: str):
    stream = ResponseStream.create(response_id="resp_1")
    audio_started = 0
    chunks: list[tuple[bytes, int]] = []

    async def on_audio_started() -> None:
        nonlocal audio_started
        audio_started += 1

    async def on_audio_chunk(audio: bytes, sample_rate: int) -> None:
        chunks.append((audio, sample_rate))

    for text in texts:
        await stream.append_text(text)
    await stream.enqueue_end()

    result = await synthesize_response_stream(
        adapter=adapter,
        stream=stream,
        voice="voice-a",
        language="fr",
        on_audio_started=on_audio_started,
        on_audio_chunk=on_audio_chunk,
    )
    return stream, result, audio_started, chunks


@pytest.mark.asyncio
async def test_streaming_deltas_are_buffered_until_sentence_boundary() -> None:
    adapter = RecordingTTSAdapter()

    stream, result, audio_started, chunks = await _run_synthesis(
        adapter,
        "Hello ",
        "world. Next",
        " sentence.",
    )

    assert result is True
    assert audio_started == 1
    assert adapter.calls == [
        ("Hello world.", "voice-a", "fr"),
        ("Next sentence.", "voice-a", "fr"),
    ]
    assert chunks == [(b"\x01\x02", 24_000), (b"\x01\x02", 24_000)]
    assert stream.assistant_context_text(separator=" ") == "Hello world. Next sentence."


@pytest.mark.asyncio
async def test_adapter_input_cap_splits_long_text_without_losing_heard_context() -> None:
    adapter = RecordingTTSAdapter(max_input_chars=12)

    stream, result, audio_started, chunks = await _run_synthesis(
        adapter,
        "Alpha beta gamma delta.",
    )

    assert result is True
    assert audio_started == 1
    assert [call[0] for call in adapter.calls] == ["Alpha beta", "gamma delta."]
    assert chunks == [(b"\x01\x02", 24_000), (b"\x01\x02", 24_000)]
    assert stream.assistant_context_text(separator=" ") == "Alpha beta gamma delta."


@pytest.mark.asyncio
async def test_final_empty_chunks_do_not_mark_audio_started_or_heard() -> None:
    adapter = RecordingTTSAdapter(emit_audio=False)

    stream, result, audio_started, chunks = await _run_synthesis(adapter, "silent response.")

    assert result is False
    assert audio_started == 0
    assert chunks == []
    assert stream.assistant_context_text(separator=" ") == "silent response."
    assert stream.heard_parts == []
