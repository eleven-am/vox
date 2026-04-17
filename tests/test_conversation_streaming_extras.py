"""Tests covering the v1.5 streaming-response cleanup:

  * EOU history gets exactly ONE assistant turn per streamed response
    (not per-sentence; not per-chunk).
  * A failed response does NOT pollute EOU history with the agent's text.
  * The response-stream queue is bounded; `append_response_text` blocks when
    full instead of growing memory unbounded.
  * `commit_response_stream` emits a `response.committed` wire event so the
    agent has an acknowledgement heartbeat.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import numpy as np
import pytest

from vox.conversation import TurnPolicy, TurnState
from vox.conversation.session import (
    RESPONSE_STREAM_QUEUE_MAX,
    WIRE_ERROR,
    WIRE_RESPONSE_COMMITTED,
    WIRE_RESPONSE_DONE,
    ConversationConfig,
    ConversationSession,
)
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo


class QuickTTS(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="quick", type=ModelType.TTS,
            architectures=("quick",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )
    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self): return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text, **_):
        yield SynthesizeChunk(
            audio=np.zeros(256, dtype=np.float32).tobytes(),
            sample_rate=24_000, is_final=False,
        )
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


class BrokenTTS(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="broken", type=ModelType.TTS,
            architectures=("broken",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )
    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self): return []

    async def synthesize(self, text, **_):
        raise RuntimeError("adapter exploded")
        yield  # pragma: no cover


class MockScheduler:
    def __init__(self, adapter): self._a = adapter
    @asynccontextmanager
    async def acquire(self, _): yield self._a


class Collector:
    def __init__(self): self.events = []
    async def __call__(self, e): self.events.append(e)
    def by_type(self, t): return [e for e in self.events if e.get("type") == t]


def _build(adapter=None, **cfg_overrides):
    tts = adapter or QuickTTS()
    coll = Collector()
    cfg = ConversationConfig(
        stt_model="x:1", tts_model="y:1", voice="default", language="en",
        policy=TurnPolicy(min_interrupt_duration_ms=50, max_endpointing_delay_ms=200),
        **cfg_overrides,
    )
    session = ConversationSession(scheduler=MockScheduler(tts), config=cfg, on_event=coll)
    return session, coll, tts


class TestSingleAssistantTurn:
    @pytest.mark.asyncio
    async def test_streamed_response_commits_single_turn(self):
        """Multi-delta stream → exactly one assistant turn with the full concatenated text."""
        session, _, _ = _build()
        await session.start()

        await session.start_response_stream()
        await session.append_response_text("Hello world. ")
        await session.append_response_text("This is ")
        await session.append_response_text("the second sentence.")
        await session.commit_response_stream()


        for _ in range(100):
            await asyncio.sleep(0.01)
            if session._tts_task is None or session._tts_task.done():
                break

        history = session._pipeline._conversation_history
        assistant_turns = [t for t in history if t.role == "assistant"]
        assert len(assistant_turns) == 1
        assert "Hello world." in assistant_turns[0].content
        assert "the second sentence." in assistant_turns[0].content

        await session.close()

    @pytest.mark.asyncio
    async def test_one_shot_submit_commits_single_turn(self):
        """submit_response_text (sugar) goes through the same commit path."""
        session, _, _ = _build()
        await session.start()
        await session.submit_response_text("hello from the bot")
        for _ in range(50):
            await asyncio.sleep(0.01)
            if session._tts_task is None or session._tts_task.done():
                break
        assistant = [t for t in session._pipeline._conversation_history if t.role == "assistant"]
        assert len(assistant) == 1
        assert "hello from the bot" in assistant[0].content
        await session.close()


class TestFailedResponseHistory:
    @pytest.mark.asyncio
    async def test_failed_response_does_not_pollute_history(self):
        session, coll, _ = _build(adapter=BrokenTTS())
        await session.start()

        await session.submit_response_text("agent reply that will never synthesize")
        for _ in range(50):
            await asyncio.sleep(0.01)
            if session._tts_task is None or session._tts_task.done():
                break

        errors = coll.by_type(WIRE_ERROR)
        assert errors
        assert "adapter exploded" in errors[0]["message"]

        history = session._pipeline._conversation_history
        assistant = [t for t in history if t.role == "assistant"]
        assert assistant == []

        await session.close()


class _HangingTTS(TTSAdapter):
    """Blocks forever inside synthesize so the worker never drains the queue.

    Used to prove `append_response_text` applies backpressure when the queue fills.
    """

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="hanging", type=ModelType.TTS,
            architectures=("hanging",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )
    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self): return []

    async def synthesize(self, text, **_):
        await asyncio.sleep(3600)
        yield  # pragma: no cover


class TestQueueBackpressure:
    @pytest.mark.asyncio
    async def test_queue_blocks_when_full(self):
        """Append beyond the queue cap blocks instead of growing memory.

        Uses a hanging TTS adapter: once the worker consumes the first item
        and calls synthesize(), it blocks forever. Further appends pile up in
        the queue until we hit the cap, at which point put() blocks.
        """
        session, _, _ = _build(adapter=_HangingTTS())
        await session.start()




        await session.append_response_text("trigger synthesis")

        for _ in range(10):
            await asyncio.sleep(0.01)




        assert session._response_stream is not None
        session._response_stream.queue = asyncio.Queue(maxsize=2)

        await session.append_response_text("first")
        await session.append_response_text("second")



        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(session.append_response_text("third"), timeout=0.1)

        await session.close()

    @pytest.mark.asyncio
    async def test_bounded_queue_primitive_blocks(self):
        """Sanity: asyncio.Queue(maxsize=N) actually blocks on full. Defends the
        invariant our code relies on against accidental changes.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        await q.put(1)
        await q.put(2)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(q.put(3), timeout=0.05)

    def test_default_queue_size_is_defined(self):
        assert isinstance(RESPONSE_STREAM_QUEUE_MAX, int)
        assert RESPONSE_STREAM_QUEUE_MAX >= 64


class TestCommitFeedback:
    @pytest.mark.asyncio
    async def test_commit_emits_response_committed_before_done(self):
        session, coll, _ = _build()
        await session.start()

        await session.start_response_stream()
        await session.append_response_text("hello")
        await session.commit_response_stream()

        for _ in range(100):
            await asyncio.sleep(0.01)
            if coll.by_type(WIRE_RESPONSE_DONE):
                break

        committed = coll.by_type(WIRE_RESPONSE_COMMITTED)
        done = coll.by_type(WIRE_RESPONSE_DONE)
        assert committed, f"no response.committed emitted; events={[e.get('type') for e in coll.events]}"
        assert done, "no response.done after commit"


        types = [e.get("type") for e in coll.events]
        assert types.index(WIRE_RESPONSE_COMMITTED) < types.index(WIRE_RESPONSE_DONE)

        await session.close()

    @pytest.mark.asyncio
    async def test_double_commit_emits_only_once(self):
        session, coll, _ = _build()
        await session.start()
        await session.start_response_stream()
        await session.append_response_text("hi")
        await session.commit_response_stream()
        await session.commit_response_stream()

        for _ in range(50):
            await asyncio.sleep(0.01)
            if coll.by_type(WIRE_RESPONSE_DONE):
                break

        assert len(coll.by_type(WIRE_RESPONSE_COMMITTED)) == 1

        await session.close()
