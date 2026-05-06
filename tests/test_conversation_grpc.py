"""Wire-mapping tests for the ConversationService gRPC servicer.

Behavioural orchestration tests live in test_operations_conversation.py.
This file focuses on proto encoding/decoding and error mapping.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import numpy as np
import pytest

from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.grpc import vox_pb2
from vox.grpc.conversation_servicer import ConversationServicer, _wire_event_to_pb


class ScriptedTTS(TTSAdapter):
    def __init__(self, chunks: int = 2) -> None:
        self._chunks = chunks
        self.texts: list[str] = []

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="scripted", type=ModelType.TTS,
            architectures=("scripted",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self): return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text: str, **_):
        self.texts.append(text)
        for _ in range(self._chunks):
            yield SynthesizeChunk(
                audio=np.full(256, 0.02, dtype=np.float32).tobytes(),
                sample_rate=24_000, is_final=False,
            )
            await asyncio.sleep(0.005)
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


class DummyScheduler:
    def __init__(self, adapter): self._a = adapter

    @asynccontextmanager
    async def acquire(self, _model: str):
        yield self._a


class FakeContext:
    def cancelled(self) -> bool:
        return False


async def _drive_until(servicer, messages, predicate, *, timeout: float = 2.0, max_items: int = 40):
    client_queue: asyncio.Queue = asyncio.Queue()
    for msg in messages:
        await client_queue.put(msg)

    async def client_stream():
        while True:
            item = await client_queue.get()
            if item is None:
                return
            yield item

    out: list[vox_pb2.ConverseServerMessage] = []

    async def run():
        gen = servicer.Converse(client_stream(), FakeContext())
        try:
            async for server_msg in gen:
                out.append(server_msg)
                if predicate(server_msg):
                    break
                if len(out) >= max_items:
                    break
        finally:
            await client_queue.put(None)

    await asyncio.wait_for(run(), timeout=timeout)
    return out


@pytest.mark.asyncio
async def test_session_update_proto_maps_to_session_created_pb():
    servicer = ConversationServicer(
        store=None, registry=None, scheduler=DummyScheduler(ScriptedTTS()),
    )
    out = await _drive_until(
        servicer,
        messages=[vox_pb2.ConverseClientMessage(
            session_update=vox_pb2.ConversationSessionUpdate(
                stt_model="x:1", tts_model="y:1", voice="default", language="en",
            ),
        )],
        predicate=lambda m: m.WhichOneof("msg") == "session_created",
    )
    assert any(m.WhichOneof("msg") == "session_created" for m in out)


@pytest.mark.asyncio
async def test_missing_stt_model_proto_maps_to_error_pb():
    servicer = ConversationServicer(
        store=None, registry=None, scheduler=DummyScheduler(ScriptedTTS()),
    )
    out = await _drive_until(
        servicer,
        messages=[vox_pb2.ConverseClientMessage(
            session_update=vox_pb2.ConversationSessionUpdate(tts_model="y:1"),
        )],
        predicate=lambda m: m.WhichOneof("msg") == "error",
    )
    errors = [m for m in out if m.WhichOneof("msg") == "error"]
    assert errors
    assert "stt_model" in errors[0].error.message


@pytest.mark.asyncio
async def test_audio_delta_event_decodes_to_pcm_bytes_in_proto():
    adapter = ScriptedTTS(chunks=2)
    servicer = ConversationServicer(
        store=None, registry=None, scheduler=DummyScheduler(adapter),
    )
    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.ConverseClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(
                    stt_model="x:1", tts_model="y:1", voice="default", sample_rate=48_000,
                ),
            ),
            vox_pb2.ConverseClientMessage(
                response_delta=vox_pb2.ConversationResponseDelta(delta="hi there"),
            ),
            vox_pb2.ConverseClientMessage(
                response_commit=vox_pb2.ConversationResponseCommit(),
            ),
        ],
        predicate=lambda m: m.WhichOneof("msg") == "response_done",
    )
    delta = next(m for m in out if m.WhichOneof("msg") == "audio_delta")
    assert len(delta.audio_delta.audio) > 0
    assert delta.audio_delta.sample_rate == 48_000


@pytest.mark.asyncio
async def test_audio_before_session_update_maps_to_error_pb():
    servicer = ConversationServicer(
        store=None, registry=None, scheduler=DummyScheduler(ScriptedTTS()),
    )
    out = await _drive_until(
        servicer,
        messages=[vox_pb2.ConverseClientMessage(
            audio_append=vox_pb2.ConversationAudioAppend(pcm16=b"\x00" * 100),
        )],
        predicate=lambda m: m.WhichOneof("msg") == "error",
    )
    assert any(
        m.WhichOneof("msg") == "error" and "session_update" in m.error.message
        for m in out
    )


def test_wire_event_to_pb_coerces_missing_numeric_fields_to_zero():
    msg = _wire_event_to_pb({
        "type": "input_audio_buffer.speech_started",
        "timestamp_ms": None,
    })
    assert msg is not None
    assert msg.speech_started.timestamp_ms == 0

    transcript = _wire_event_to_pb({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "hello",
        "language": "en-us",
        "start_ms": None,
        "end_ms": None,
        "words": [{"word": "hello", "start_ms": None, "end_ms": None}],
    })
    assert transcript is not None
    assert transcript.transcript_done.start_ms == 0
    assert transcript.transcript_done.end_ms == 0
    assert transcript.transcript_done.words[0].start_ms == 0
    assert transcript.transcript_done.words[0].end_ms == 0
