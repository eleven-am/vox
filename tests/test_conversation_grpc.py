"""Wire-mapping tests for the ConversationService gRPC servicer.

Behavioural orchestration tests live in test_operations_conversation.py.
This file focuses on proto encoding/decoding and error mapping.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from tests.fakes import FakeScheduler
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.grpc import vox_pb2
from vox.grpc.conversation_commands import (
    conversation_session_update_to_command,
    converse_client_message_to_command,
)
from vox.grpc.conversation_events import conversation_event_to_pb
from vox.grpc.conversation_servicer import ConversationServicer
from vox.operations.conversation import (
    ConvAudioClearEvent,
    ConvTranscriptDoneEvent,
    parse_conversation_wire_event,
)


def _engine_wire_event_to_pb(event: dict):
    mapped = parse_conversation_wire_event(event)
    if mapped is None:
        return None
    return conversation_event_to_pb(mapped)


class ScriptedTTS(TTSAdapter):
    def __init__(self, chunks: int = 2) -> None:
        self._chunks = chunks
        self.texts: list[str] = []

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="scripted",
            type=ModelType.TTS,
            architectures=("scripted",),
            default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self):
        return True

    def list_voices(self):
        return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text: str, **_):
        self.texts.append(text)
        for _ in range(self._chunks):
            yield SynthesizeChunk(
                audio=np.full(256, 0.02, dtype=np.float32).tobytes(),
                sample_rate=24_000,
                is_final=False,
            )
            await asyncio.sleep(0.005)
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


DummyScheduler = FakeScheduler


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
        store=None,
        registry=None,
        scheduler=DummyScheduler(ScriptedTTS()),
    )
    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.ConverseClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(
                    stt_model="x:1",
                    tts_model="y:1",
                    voice="default",
                    language="en",
                ),
            )
        ],
        predicate=lambda m: m.WhichOneof("msg") == "session_created",
    )
    created = next(m.session_created for m in out if m.WhichOneof("msg") == "session_created")
    assert created.turn_profile == "default"
    assert created.policy.min_interrupt_duration_ms == 250


def test_session_update_proto_preserves_backend_selection():
    config = conversation_session_update_to_command(
        vox_pb2.ConversationSessionUpdate(
            stt_model="x:1",
            tts_model="y:1",
            vad_backend="ten-vad",
            turn_detector="ten-turn",
        )
    ).config

    assert config.vad_backend == "ten-vad"
    assert config.turn_detector == "ten-turn"


def test_session_update_proto_applies_turn_profile_defaults():
    config = conversation_session_update_to_command(
        vox_pb2.ConversationSessionUpdate(
            stt_model="x:1",
            tts_model="y:1",
            turn_profile="speakerphone",
        )
    ).config

    assert config.turn_profile == "speakerphone"
    assert config.policy is not None
    assert config.policy.aec_warmup_ms == 1100
    assert config.policy.self_echo_min_overlap == pytest.approx(0.82)


def test_session_update_proto_preserves_speaking_interrupt_policy():
    config = conversation_session_update_to_command(
        vox_pb2.ConversationSessionUpdate(
            stt_model="x:1",
            tts_model="y:1",
            policy=vox_pb2.ConversationTurnPolicy(
                speaking_interrupt_min_duration_ms=650,
                speaking_interrupt_min_words=3,
                self_echo_min_words=4,
                self_echo_min_overlap=0.8,
            ),
        )
    ).config

    assert config.policy is not None
    assert config.policy.speaking_interrupt_min_duration_ms == 650
    assert config.policy.speaking_interrupt_min_words == 3
    assert config.policy.self_echo_min_words == 4
    assert config.policy.self_echo_min_overlap == pytest.approx(0.8)


def test_session_update_proto_preserves_profile_defaults_when_overriding_one_field():
    config = conversation_session_update_to_command(
        vox_pb2.ConversationSessionUpdate(
            stt_model="x:1",
            tts_model="y:1",
            turn_profile="headset",
            policy=vox_pb2.ConversationTurnPolicy(
                speaking_interrupt_min_duration_ms=300,
            ),
        )
    ).config

    assert config.turn_profile == "headset"
    assert config.policy is not None
    assert config.policy.speaking_interrupt_min_duration_ms == 300
    assert config.policy.partial_interrupts is True
    assert config.policy.dynamic_endpointing is True
    assert config.policy.aec_warmup_ms == 250


def test_turn_eou_predicted_event_maps_to_proto():
    msg = _engine_wire_event_to_pb(
        {
            "type": "turn.eou.predicted",
            "probability": 0.25,
            "threshold": 0.5,
            "decision": "incomplete",
            "action": "wait",
            "delay_ms": 800,
            "turn_detector": "livekit",
            "start_ms": 20,
            "end_ms": 120,
        }
    )

    assert msg is not None
    assert msg.WhichOneof("msg") == "turn_eou_predicted"
    assert msg.turn_eou_predicted.probability == pytest.approx(0.25)
    assert msg.turn_eou_predicted.action == "wait"
    assert msg.turn_eou_predicted.delay_ms == 800


@pytest.mark.asyncio
async def test_missing_stt_model_proto_maps_to_error_pb():
    servicer = ConversationServicer(
        store=None,
        registry=None,
        scheduler=DummyScheduler(ScriptedTTS()),
    )
    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.ConverseClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(tts_model="y:1"),
            )
        ],
        predicate=lambda m: m.WhichOneof("msg") == "error",
    )
    errors = [m for m in out if m.WhichOneof("msg") == "error"]
    assert errors
    assert "stt_model" in errors[0].error.message
    assert errors[0].error.code == "command_invalid"
    assert errors[0].error.recoverable is True
    assert errors[0].error.generation_id == ""


@pytest.mark.asyncio
async def test_audio_delta_event_decodes_to_pcm_bytes_in_proto():
    adapter = ScriptedTTS(chunks=2)
    servicer = ConversationServicer(
        store=None,
        registry=None,
        scheduler=DummyScheduler(adapter),
    )
    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.ConverseClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(
                    stt_model="x:1",
                    tts_model="y:1",
                    voice="default",
                    sample_rate=48_000,
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
    delta = next(m for m in out if m.WhichOneof("msg") == "audio_delta")
    assert len(delta.audio_delta.audio) > 0
    assert delta.audio_delta.sample_rate == 48_000
    assert delta.audio_delta.response_id
    assert delta.audio_delta.sequence > 0


@pytest.mark.asyncio
async def test_audio_before_session_update_maps_to_error_pb():
    servicer = ConversationServicer(
        store=None,
        registry=None,
        scheduler=DummyScheduler(ScriptedTTS()),
    )
    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.ConverseClientMessage(
                audio_append=vox_pb2.ConversationAudioAppend(pcm16=b"\x00" * 100),
            )
        ],
        predicate=lambda m: m.WhichOneof("msg") == "error",
    )
    assert any(m.WhichOneof("msg") == "error" and "session_update" in m.error.message for m in out)


def test_wire_event_to_pb_coerces_missing_numeric_fields_to_zero():
    msg = _engine_wire_event_to_pb(
        {
            "type": "input_audio_buffer.speech_started",
            "timestamp_ms": None,
        }
    )
    assert msg is not None
    assert msg.speech_started.timestamp_ms == 0

    transcript = _engine_wire_event_to_pb(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "hello",
            "language": "en-us",
            "start_ms": None,
            "end_ms": None,
            "words": [{"word": "hello", "start_ms": None, "end_ms": None}],
        }
    )
    assert transcript is not None
    assert transcript.transcript_done.start_ms == 0
    assert transcript.transcript_done.end_ms == 0
    assert transcript.transcript_done.words[0].start_ms == 0
    assert transcript.transcript_done.words[0].end_ms == 0


def test_transcript_done_event_maps_metadata_with_shared_transcript_converters():
    msg = conversation_event_to_pb(
        ConvTranscriptDoneEvent(
            transcript="Alice visited Paris",
            language="en",
            start_ms=100,
            end_ms=900,
            eou_probability=0.72,
            entities=(
                {
                    "type": "PERSON",
                    "text": "Alice",
                    "start_char": 0,
                    "end_char": 5,
                },
            ),
            topics=("travel",),
            words=(
                {
                    "word": "Alice",
                    "start_ms": 100,
                    "end_ms": 240,
                    "confidence": 0.91,
                },
            ),
            speech_context={
                "schema_version": 2,
                "status": "failed",
                "unavailable": ["speaker", "sounds"],
            },
        )
    )

    assert msg is not None
    assert msg.WhichOneof("msg") == "transcript_done"
    transcript = msg.transcript_done
    assert transcript.eou_probability == pytest.approx(0.72)
    assert transcript.entities[0].text == "Alice"
    assert list(transcript.topics) == ["travel"]
    assert transcript.words[0].word == "Alice"
    assert transcript.words[0].confidence == pytest.approx(0.91)
    assert transcript.speech_context["status"] == "failed"


def test_response_commands_decode_generation_id_with_backward_compat():
    with_generation = converse_client_message_to_command(
        vox_pb2.ConverseClientMessage(
            response_start=vox_pb2.ConversationResponseStart(generation_id="gen-1"),
        )
    )
    assert with_generation.generation_id == "gen-1"

    without_generation = converse_client_message_to_command(
        vox_pb2.ConverseClientMessage(response_start=vox_pb2.ConversationResponseStart()),
    )
    assert without_generation.generation_id is None

    commit = converse_client_message_to_command(
        vox_pb2.ConverseClientMessage(
            response_commit=vox_pb2.ConversationResponseCommit(generation_id="gen-1"),
        )
    )
    assert commit.generation_id == "gen-1"

    cancel = converse_client_message_to_command(
        vox_pb2.ConverseClientMessage(response_cancel=vox_pb2.ConversationResponseCancel()),
    )
    assert cancel.generation_id is None


@pytest.mark.asyncio
async def test_response_lifecycle_pb_round_trips_generation_id():
    servicer = ConversationServicer(
        store=None,
        registry=None,
        scheduler=DummyScheduler(ScriptedTTS(chunks=2)),
    )
    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.ConverseClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(
                    stt_model="x:1",
                    tts_model="y:1",
                    voice="default",
                ),
            ),
            vox_pb2.ConverseClientMessage(
                response_start=vox_pb2.ConversationResponseStart(generation_id="gen-1"),
            ),
            vox_pb2.ConverseClientMessage(
                response_delta=vox_pb2.ConversationResponseDelta(delta="hi there", generation_id="gen-1"),
            ),
            vox_pb2.ConverseClientMessage(
                response_commit=vox_pb2.ConversationResponseCommit(generation_id="gen-1"),
            ),
        ],
        predicate=lambda m: m.WhichOneof("msg") == "response_done",
    )
    created = next(m for m in out if m.WhichOneof("msg") == "response_created")
    committed = next(m for m in out if m.WhichOneof("msg") == "response_committed")
    done = next(m for m in out if m.WhichOneof("msg") == "response_done")
    assert created.response_created.generation_id == "gen-1"
    assert committed.response_committed.generation_id == "gen-1"
    assert done.response_done.generation_id == "gen-1"


@pytest.mark.asyncio
async def test_stale_generation_delta_maps_to_typed_error_pb():
    servicer = ConversationServicer(
        store=None,
        registry=None,
        scheduler=DummyScheduler(ScriptedTTS()),
    )
    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.ConverseClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(
                    stt_model="x:1",
                    tts_model="y:1",
                    voice="default",
                ),
            ),
            vox_pb2.ConverseClientMessage(
                response_delta=vox_pb2.ConversationResponseDelta(delta="late", generation_id="gen-a"),
            ),
        ],
        predicate=lambda m: m.WhichOneof("msg") == "error",
    )
    errors = [m for m in out if m.WhichOneof("msg") == "error"]
    assert errors
    assert errors[0].error.code == "response_stale_generation"
    assert errors[0].error.recoverable is True
    assert errors[0].error.generation_id == "gen-a"


@pytest.mark.asyncio
async def test_response_commands_without_generation_id_still_work_end_to_end():
    servicer = ConversationServicer(
        store=None,
        registry=None,
        scheduler=DummyScheduler(ScriptedTTS(chunks=2)),
    )
    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.ConverseClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(
                    stt_model="x:1",
                    tts_model="y:1",
                    voice="default",
                ),
            ),
            vox_pb2.ConverseClientMessage(response_start=vox_pb2.ConversationResponseStart()),
            vox_pb2.ConverseClientMessage(response_delta=vox_pb2.ConversationResponseDelta(delta="hi there")),
            vox_pb2.ConverseClientMessage(response_commit=vox_pb2.ConversationResponseCommit()),
        ],
        predicate=lambda m: m.WhichOneof("msg") == "response_done",
    )
    created = next(m for m in out if m.WhichOneof("msg") == "response_created")
    done = next(m for m in out if m.WhichOneof("msg") == "response_done")
    assert created.response_created.generation_id == ""
    assert done.response_done.generation_id == ""
    assert done.response_done.response_id


def test_audio_clear_event_maps_to_proto_message():
    msg = _engine_wire_event_to_pb({"type": "response.audio.clear", "response_id": "resp_1"})
    assert msg is not None
    assert msg.WhichOneof("msg") == "audio_clear"
    assert msg.audio_clear.response_id == "resp_1"

    direct = conversation_event_to_pb(ConvAudioClearEvent(response_id="resp_2"))
    assert direct is not None
    assert direct.WhichOneof("msg") == "audio_clear"
    assert direct.audio_clear.response_id == "resp_2"
