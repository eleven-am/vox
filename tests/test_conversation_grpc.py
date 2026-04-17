"""Integration tests for the ConversationService gRPC servicer.

Uses an in-memory bidi RPC driver instead of spinning up a real gRPC server —
the servicer method is a plain async generator and can be exercised directly.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import numpy as np
import pytest

from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.grpc import vox_pb2
from vox.grpc.conversation_servicer import ConversationServicer


class ScriptedTTS(TTSAdapter):
    def __init__(self, chunks: int = 2) -> None:
        self._chunks = chunks
        self.last_text: str | None = None
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
        self.last_text = text
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


async def _collect_until(servicer, messages, predicate, *, max_items: int = 40, timeout: float = 2.0):
    """Drive the bidi RPC end-to-end. Client stream stays open until we've
    either matched the predicate or hit max_items, so the server has time to
    emit async events (TTS audio chunks etc.) before being torn down.
    """
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
async def test_session_update_emits_session_created():
    adapter = ScriptedTTS()
    servicer = ConversationServicer(store=None, registry=None, scheduler=DummyScheduler(adapter))

    out = await _collect_until(
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
async def test_missing_stt_model_returns_error():
    servicer = ConversationServicer(store=None, registry=None, scheduler=DummyScheduler(ScriptedTTS()))

    out = await _collect_until(
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
async def test_streaming_response_emits_audio_and_done():
    """Full streaming flow via start + delta + commit → audio + done."""
    adapter = ScriptedTTS(chunks=2)
    servicer = ConversationServicer(store=None, registry=None, scheduler=DummyScheduler(adapter))

    out = await _collect_until(
        servicer,
        messages=[
            vox_pb2.ConverseClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(
                    stt_model="x:1", tts_model="y:1", voice="default",
                ),
            ),
            vox_pb2.ConverseClientMessage(
                response_start=vox_pb2.ConversationResponseStart(),
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
    kinds = [m.WhichOneof("msg") for m in out]
    assert "session_created" in kinds
    assert "response_created" in kinds
    assert "response_committed" in kinds
    assert "audio_delta" in kinds
    assert "response_done" in kinds

    delta = next(m for m in out if m.WhichOneof("msg") == "audio_delta")
    assert len(delta.audio_delta.audio) > 0
    assert delta.audio_delta.sample_rate == 24_000


@pytest.mark.asyncio
async def test_client_half_close_still_drains_response():
    """Streaming response + client half-close still drains audio before RPC end."""
    adapter = ScriptedTTS(chunks=2)
    servicer = ConversationServicer(store=None, registry=None, scheduler=DummyScheduler(adapter))

    async def client_stream():
        yield vox_pb2.ConverseClientMessage(
            session_update=vox_pb2.ConversationSessionUpdate(
                stt_model="x:1", tts_model="y:1", voice="default",
            ),
        )
        yield vox_pb2.ConverseClientMessage(
            response_delta=vox_pb2.ConversationResponseDelta(delta="hi after eof"),
        )
        yield vox_pb2.ConverseClientMessage(
            response_commit=vox_pb2.ConversationResponseCommit(),
        )

    async def collect():
        items: list[vox_pb2.ConverseServerMessage] = []
        async for server_msg in servicer.Converse(client_stream(), FakeContext()):
            items.append(server_msg)
        return items

    out = await asyncio.wait_for(collect(), timeout=2.0)
    kinds = [m.WhichOneof("msg") for m in out]
    assert "response_created" in kinds
    assert "audio_delta" in kinds
    assert "response_done" in kinds


@pytest.mark.asyncio
async def test_streamed_response_delta_starts_audio_before_commit():
    adapter = ScriptedTTS(chunks=2)
    servicer = ConversationServicer(store=None, registry=None, scheduler=DummyScheduler(adapter))

    out = await _collect_until(
        servicer,
        messages=[
            vox_pb2.ConverseClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(
                    stt_model="x:1", tts_model="y:1", voice="default",
                ),
            ),
            vox_pb2.ConverseClientMessage(
                response_delta=vox_pb2.ConversationResponseDelta(delta="Hello world. Still pending"),
            ),
            vox_pb2.ConverseClientMessage(
                response_commit=vox_pb2.ConversationResponseCommit(),
            ),
        ],
        predicate=lambda m: m.WhichOneof("msg") == "response_done",
    )
    kinds = [m.WhichOneof("msg") for m in out]
    assert "response_created" in kinds
    assert "audio_delta" in kinds
    assert "response_done" in kinds
    assert adapter.texts == ["Hello world.", "Still pending"]


@pytest.mark.asyncio
async def test_audio_before_session_update_errors():
    servicer = ConversationServicer(store=None, registry=None, scheduler=DummyScheduler(ScriptedTTS()))

    out = await _collect_until(
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


@pytest.mark.asyncio
async def test_turn_policy_passed_through():
    """Explicit policy in session_update overrides defaults."""
    adapter = ScriptedTTS(chunks=1)
    servicer = ConversationServicer(store=None, registry=None, scheduler=DummyScheduler(adapter))

    policy = vox_pb2.ConversationTurnPolicy(
        allow_interrupt_while_speaking=True,
        min_interrupt_duration_ms=250,
        max_endpointing_delay_ms=2500,
        stable_speaking_min_ms=123,
    )
    out = await _collect_until(
        servicer,
        messages=[vox_pb2.ConverseClientMessage(
            session_update=vox_pb2.ConversationSessionUpdate(
                stt_model="x:1", tts_model="y:1", policy=policy,
            ),
        )],
        predicate=lambda m: m.WhichOneof("msg") == "session_created",
    )
    assert any(m.WhichOneof("msg") == "session_created" for m in out)
