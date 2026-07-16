from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import numpy as np
import pytest
from google.protobuf.struct_pb2 import Struct

from tests.fakes import FakeCloneableTTSAdapter
from vox.audio.codecs import encode_wav
from vox.core.adapter import STTAdapter
from vox.core.cloned_voices import create_stored_voice
from vox.core.store import BlobStore
from vox.core.types import (
    AdapterInfo,
    LoadedModelInfo,
    ModelFormat,
    ModelInfo,
    ModelType,
    TranscribeResult,
    TranscriptSegment,
    VoiceInfo,
    WordTimestamp,
)
from vox.grpc import vox_pb2
from vox.operations.errors import InvalidConfigError
from vox.operations.models import (
    ModelLayer,
    PullEvent,
    ShowResult,
    delete_model_payload,
    list_models_payload,
    pull_event_payload,
    show_model_payload,
)
from vox.operations.synthesis import SynthesisRawChunk
from vox.operations.system import (
    HealthStatusResult,
    ListLoadedModelsResult,
    health_status_payload,
    list_loaded_models_payload,
)
from vox.operations.transcription import (
    AnnotateResult,
    Entity,
    TranscriptionResultBundle,
)
from vox.operations.voices import DeleteVoiceResult, ListedVoice, created_voice_payload, list_voices_payload
from vox.streaming.types import StreamTranscript


def _make_store(tmp_path: Path) -> BlobStore:
    return BlobStore(root=tmp_path)


def _make_registry_mock() -> MagicMock:
    registry = MagicMock()
    registry.resolve_model_ref.side_effect = lambda name, tag, explicit_tag=False: (name, tag)
    return registry


class FakeContext:
    def __init__(self):
        self._code = None
        self._details = None

    async def abort(self, code, details):
        self._code = code
        self._details = details
        raise Exception(f"gRPC abort: {code} {details}")


class DummyScheduler:
    def __init__(self, adapter, loaded_models=None):
        self._adapter = adapter
        self._loaded_models = loaded_models or []

    @asynccontextmanager
    async def acquire(self, _model_name: str):
        yield self._adapter

    def list_loaded(self):
        return self._loaded_models


class FakeSTTAdapter(STTAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-stt",
            type=ModelType.STT,
            architectures=("fake",),
            default_sample_rate=16_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *a, **k):
        pass

    def unload(self):
        pass

    @property
    def is_loaded(self):
        return True

    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        return TranscribeResult(
            text="hello grpc",
            language="en",
            duration_ms=1000,
            model="fake-stt:latest",
            segments=(TranscriptSegment(text="hello grpc", start_ms=0, end_ms=1000),),
        )


class TestHealthServicer:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        from vox.grpc.health_servicer import HealthServicer

        servicer = HealthServicer(MagicMock())
        resp = await servicer.Health(vox_pb2.HealthRequest(), FakeContext())
        assert resp.status == "ok"

    @pytest.mark.asyncio
    async def test_list_loaded_returns_models(self):
        from vox.grpc.health_servicer import HealthServicer

        loaded_model = LoadedModelInfo(
            name="whisper",
            tag="large-v3",
            type=ModelType.STT,
            device="cpu",
            vram_bytes=1000,
            loaded_at=1.0,
            last_used=2.0,
            ref_count=1,
        )
        scheduler = MagicMock()
        scheduler.list_loaded.return_value = [loaded_model]
        servicer = HealthServicer(scheduler)
        resp = await servicer.ListLoaded(vox_pb2.ListLoadedRequest(), FakeContext())
        assert len(resp.models) == 1
        assert resp.models[0].name == "whisper"
        assert resp.models[0].type == "stt"


class TestModelServicerMapping:
    @pytest.mark.asyncio
    async def test_list_returns_models(self, tmp_path):
        from vox.grpc.model_servicer import ModelServicer

        store = _make_store(tmp_path)
        registry = _make_registry_mock()
        scheduler = MagicMock()

        model_info = MagicMock()
        model_info.full_name = "whisper:large-v3"
        model_info.type = MagicMock(value="stt")
        model_info.format = MagicMock(value="ct2")
        model_info.architecture = "whisper"
        model_info.size_bytes = 5000
        model_info.description = "test model"
        store.list_models = MagicMock(return_value=[model_info])

        servicer = ModelServicer(store, registry, scheduler)
        resp = await servicer.List(vox_pb2.ListModelsRequest(), FakeContext())
        assert len(resp.models) == 1
        assert resp.models[0].name == "whisper:large-v3"
        assert resp.models[0].type == "stt"

    @pytest.mark.asyncio
    async def test_show_not_found_aborts_with_not_found(self, tmp_path):
        from vox.grpc.model_servicer import ModelServicer

        servicer = ModelServicer(_make_store(tmp_path), _make_registry_mock(), MagicMock())
        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Show(vox_pb2.ShowRequest(name="nonexistent:v1"), FakeContext())

    @pytest.mark.asyncio
    async def test_delete_not_found_aborts_with_not_found(self, tmp_path):
        from vox.grpc.model_servicer import ModelServicer

        scheduler = MagicMock()
        scheduler.unload = AsyncMock(return_value=True)
        servicer = ModelServicer(_make_store(tmp_path), _make_registry_mock(), scheduler)
        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Delete(vox_pb2.DeleteRequest(name="nonexistent:v1"), FakeContext())

    @pytest.mark.asyncio
    async def test_pull_unknown_yields_error_event(self, tmp_path):
        from vox.grpc.model_servicer import ModelServicer

        registry = _make_registry_mock()
        registry.lookup.return_value = None
        servicer = ModelServicer(_make_store(tmp_path), registry, MagicMock())

        messages = []
        async for msg in servicer.Pull(vox_pb2.PullRequest(name="nonexistent:v1"), FakeContext()):
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].status == "error"
        assert "not found" in messages[0].error

    @pytest.mark.asyncio
    async def test_pull_forwards_variant_field(self, tmp_path):
        from vox.grpc.model_servicer import ModelServicer

        async def fake_events(**kwargs):
            request = kwargs["request"]
            assert request.name == "kokoro-tts:v1.0"
            assert request.variant == "onnx"
            yield PullEvent(status="success")

        servicer = ModelServicer(_make_store(tmp_path), _make_registry_mock(), MagicMock())
        with patch("vox.grpc.model_servicer.pull_model", side_effect=fake_events):
            messages = []
            async for msg in servicer.Pull(
                vox_pb2.PullRequest(name="kokoro-tts:v1.0", variant="onnx"),
                FakeContext(),
            ):
                messages.append(msg)

        assert len(messages) == 1
        assert messages[0].status == "success"


class TestGrpcModelMessages:
    def test_list_models_response_matches_operation_payload_fields(self):
        from vox.grpc.model_messages import list_models_response

        model = ModelInfo(
            name="parakeet-stt-onnx",
            tag="tdt-0.6b-v3",
            type=ModelType.STT,
            format=ModelFormat.ONNX,
            architecture="parakeet",
            adapter="parakeet",
            size_bytes=123,
            description="fast stt",
        )

        operation_payload = list_models_payload([model])["models"][0]
        message = list_models_response([model]).models[0]

        for field, value in operation_payload.items():
            assert getattr(message, field) == value

    def test_show_model_response_matches_operation_payload_fields(self):
        from vox.grpc.model_messages import show_model_response

        result = ShowResult(
            name="foo:latest",
            config={"architecture": "fake", "quantized": True, "size": 7},
            layers=(
                ModelLayer(
                    media_type="application/vox.model.bin",
                    digest="sha256-x",
                    size=12,
                    filename="model.bin",
                ),
            ),
        )

        operation_payload = show_model_payload(result)
        message = show_model_response(result)

        assert message.name == operation_payload["name"]
        assert dict(message.config) == {
            "architecture": "fake",
            "quantized": "True",
            "size": "7",
        }
        assert message.layers[0].media_type == operation_payload["layers"][0]["media_type"]
        assert message.layers[0].digest == operation_payload["layers"][0]["digest"]
        assert message.layers[0].size == operation_payload["layers"][0]["size"]
        assert message.layers[0].filename == operation_payload["layers"][0]["filename"]

    def test_pull_progress_message_matches_operation_payload_fields(self):
        from vox.grpc.model_messages import pull_progress_message

        event = PullEvent(status="downloading model.onnx", completed=1, total=3, error="slow")
        operation_payload = pull_event_payload(event)
        message = pull_progress_message(event)

        for field, value in operation_payload.items():
            assert getattr(message, field) == value

    def test_pull_error_message_preserves_stream_error_contract(self):
        from vox.grpc.model_messages import pull_error_message
        from vox.operations.errors import CatalogEntryNotFoundError

        message = pull_error_message(CatalogEntryNotFoundError("missing:v1"))

        assert message.status == "error"
        assert "missing:v1" in message.error
        assert message.completed == 0
        assert message.total == 0

    def test_delete_model_response_matches_operation_payload_fields(self):
        from vox.grpc.model_messages import delete_model_response

        operation_payload = delete_model_payload()
        message = delete_model_response()

        for field, value in operation_payload.items():
            assert getattr(message, field) == value


class TestTranscriptionServicerMapping:
    @pytest.mark.asyncio
    async def test_transcribe_returns_text_and_segments(self, tmp_path):
        from vox.grpc.transcription_servicer import TranscriptionServicer

        store = _make_store(tmp_path)
        registry = MagicMock()
        scheduler = DummyScheduler(FakeSTTAdapter())
        servicer = TranscriptionServicer(store, registry, scheduler)

        response = await servicer.Transcribe(
            vox_pb2.TranscribeRequest(
                model="fake-stt:latest",
                audio=encode_wav(np.zeros(16_000, dtype=np.float32), 16_000),
                format_hint="wav",
            ),
            FakeContext(),
        )
        assert response.text == "hello grpc"
        assert response.model == "fake-stt:latest"

    @pytest.mark.asyncio
    async def test_transcribe_no_audio_aborts_with_invalid_argument(self, tmp_path):
        from vox.grpc.transcription_servicer import TranscriptionServicer

        servicer = TranscriptionServicer(_make_store(tmp_path), MagicMock(), MagicMock())
        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Transcribe(
                vox_pb2.TranscribeRequest(model="whisper:large-v3", audio=b""),
                FakeContext(),
            )

    @pytest.mark.asyncio
    async def test_transcribe_no_default_model_aborts_with_invalid_argument(self, tmp_path):
        from vox.grpc.transcription_servicer import TranscriptionServicer

        registry = MagicMock()
        registry.available_models.return_value = {}
        servicer = TranscriptionServicer(_make_store(tmp_path), registry, MagicMock())
        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Transcribe(
                vox_pb2.TranscribeRequest(audio=b"\x00\x00"),
                FakeContext(),
            )

    @pytest.mark.asyncio
    async def test_transcribe_request_build_errors_are_mapped_to_grpc(self, tmp_path, monkeypatch):
        from vox.grpc import transcription_servicer
        from vox.grpc.transcription_servicer import TranscriptionServicer

        def fail_build(**_kwargs):
            raise InvalidConfigError("bad grpc transcription config")

        monkeypatch.setattr(transcription_servicer, "transcription_request_from_fields", fail_build)

        context = FakeContext()
        servicer = TranscriptionServicer(_make_store(tmp_path), MagicMock(), MagicMock())
        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Transcribe(
                vox_pb2.TranscribeRequest(model="whisper:large-v3", audio=b"abc"),
                context,
            )

        assert context._code is grpc.StatusCode.INVALID_ARGUMENT
        assert context._details == "bad grpc transcription config"

    @pytest.mark.asyncio
    async def test_annotate_request_build_errors_are_mapped_to_grpc(self, tmp_path, monkeypatch):
        from vox.grpc import transcription_servicer
        from vox.grpc.transcription_servicer import TranscriptionServicer

        def fail_build(**_kwargs):
            raise InvalidConfigError("bad grpc annotation config")

        monkeypatch.setattr(transcription_servicer, "annotate_request_from_fields", fail_build)

        context = FakeContext()
        servicer = TranscriptionServicer(_make_store(tmp_path), MagicMock(), MagicMock())
        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Annotate(
                vox_pb2.AnnotateRequest(text="Alice visited Paris", language="en"),
                context,
            )

        assert context._code is grpc.StatusCode.INVALID_ARGUMENT
        assert context._details == "bad grpc annotation config"


class TestGrpcTranscriptMessages:
    def test_transcript_segment_list_helper_reuses_single_segment_contract(self):
        from vox.grpc.transcript_messages import (
            transcript_segment_message,
            transcript_segment_messages,
        )

        segment = TranscriptSegment(
            text="Alice visited Paris",
            start_ms=100,
            end_ms=900,
            words=(WordTimestamp(word="Alice", start_ms=100, end_ms=300),),
        )

        single = transcript_segment_message(segment)
        listed = transcript_segment_messages((segment,))[0]

        assert listed == single

    def test_transcribe_response_encodes_segments_words_entities_and_topics(self):
        from vox.grpc.transcript_messages import transcribe_response

        bundle = TranscriptionResultBundle(
            result=TranscribeResult(
                text="Alice visited Paris",
                language="en",
                duration_ms=1200,
                model="fake-stt:latest",
                segments=(
                    TranscriptSegment(
                        text="Alice visited Paris",
                        start_ms=0,
                        end_ms=1200,
                        words=(
                            WordTimestamp(word="Alice", start_ms=0, end_ms=500, confidence=0.9),
                            WordTimestamp(word="Paris", start_ms=800, end_ms=1200),
                        ),
                    ),
                ),
            ),
            processing_ms=25,
            entities=(Entity(type="PERSON", text="Alice", start_char=0, end_char=5),),
            topics=("travel",),
        )

        message = transcribe_response(bundle)

        assert message.model == "fake-stt:latest"
        assert message.text == "Alice visited Paris"
        assert message.language == "en"
        assert message.duration_ms == 1200
        assert message.processing_ms == 25
        assert message.segments[0].words[0].word == "Alice"
        assert message.segments[0].words[0].confidence == pytest.approx(0.9)
        assert not message.segments[0].words[1].HasField("confidence")
        assert message.entities[0].type == "PERSON"
        assert list(message.topics) == ["travel"]

    def test_stream_transcript_result_encodes_dict_shapes(self):
        from vox.grpc.transcript_messages import stream_transcript_result

        transcript = StreamTranscript(
            text="hello",
            is_partial=True,
            start_ms=10,
            end_ms=210,
            audio_duration_ms=200,
            processing_duration_ms=15,
            model="fake-stt:latest",
            eou_probability=0.7,
            entities=[{"type": "PLACE", "text": "Paris", "start_char": 0, "end_char": 5}],
            topics=["travel"],
            words=[{"word": "Paris", "start_ms": 10, "end_ms": 210, "confidence": 0.8}],
            segments=[
                {
                    "text": "Paris",
                    "start_ms": 10,
                    "end_ms": 210,
                    "words": [{"word": "Paris", "start_ms": 10, "end_ms": 210}],
                }
            ],
        )

        message = stream_transcript_result(transcript)

        assert message.text == "hello"
        assert message.is_partial is True
        assert message.eou_probability == pytest.approx(0.7)
        assert message.entities[0].text == "Paris"
        assert list(message.topics) == ["travel"]
        assert message.words[0].confidence == pytest.approx(0.8)
        assert message.segments[0].words[0].word == "Paris"

    def test_annotate_response_encodes_entities_and_topics(self):
        from vox.grpc.transcript_messages import annotate_response

        message = annotate_response(
            AnnotateResult(
                entities=(Entity(type="PERSON", text="Roy", start_char=0, end_char=3),),
                topics=("voice",),
            )
        )

        assert message.entities[0].text == "Roy"
        assert list(message.topics) == ["voice"]


class TestGrpcStreamingMessages:
    def test_stream_config_request_decodes_defaults_and_explicit_fields(self):
        from vox.grpc import vox_pb2
        from vox.grpc.streaming_messages import stream_config_request

        defaults = stream_config_request(vox_pb2.StreamConfig())
        explicit = stream_config_request(
            vox_pb2.StreamConfig(
                model="parakeet",
                language="fr",
                sample_rate=48_000,
                partials=True,
                partial_window_ms=1200,
                partial_stride_ms=400,
                include_word_timestamps=True,
                temperature=0.25,
            )
        )

        assert defaults.model == ""
        assert defaults.language == "en"
        assert defaults.sample_rate == 16_000
        assert defaults.partials is False
        assert defaults.partial_window_ms == 1500
        assert defaults.partial_stride_ms == 700
        assert defaults.include_word_timestamps is False
        assert defaults.temperature == 0.0

        assert explicit.model == "parakeet"
        assert explicit.language == "fr"
        assert explicit.sample_rate == 48_000
        assert explicit.partials is True
        assert explicit.partial_window_ms == 1200
        assert explicit.partial_stride_ms == 400
        assert explicit.include_word_timestamps is True
        assert explicit.temperature == pytest.approx(0.25)

    def test_stream_output_message_encodes_event_variants(self):
        from vox.grpc.streaming_messages import stream_output_message
        from vox.operations.streaming_transcription import (
            ErrorEvent,
            SessionReadyEvent,
            SpeechStartedEvent,
            SpeechStoppedEvent,
            TranscriptEvent,
        )

        transcript = StreamTranscript(
            text="hello",
            is_partial=False,
            start_ms=100,
            end_ms=300,
            audio_duration_ms=200,
            processing_duration_ms=20,
            model="fake-stt:latest",
        )

        assert stream_output_message(SessionReadyEvent("model", "en", 16000)).WhichOneof("msg") == "ready"
        assert stream_output_message(SpeechStartedEvent(10)).speech_started.timestamp_ms == 10
        assert stream_output_message(SpeechStoppedEvent(20)).speech_stopped.timestamp_ms == 20
        assert stream_output_message(TranscriptEvent(transcript)).transcript.text == "hello"
        assert stream_output_message(ErrorEvent("boom")).error.message == "boom"

    @pytest.mark.asyncio
    async def test_execute_stream_input_message_dispatches_stream_commands(self):
        from vox.grpc import vox_pb2
        from vox.grpc.streaming_messages import execute_stream_input_message

        class RecordingSession:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple, dict]] = []

            async def configure_or_report(self, config):
                self.calls.append(("configure", (config,), {}))

            async def submit_pcm16_or_report(self, *args, **kwargs):
                self.calls.append(("pcm16", args, kwargs))

            async def submit_opus_or_report(self, *args, **kwargs):
                self.calls.append(("opus", args, kwargs))

            async def submit_encoded_or_report(self, *args, **kwargs):
                self.calls.append(("encoded", args, kwargs))

        session = RecordingSession()

        assert (
            await execute_stream_input_message(
                session,
                vox_pb2.StreamInput(config=vox_pb2.StreamConfig(model="parakeet", sample_rate=48_000)),
            )
            is True
        )
        assert (
            await execute_stream_input_message(
                session,
                vox_pb2.StreamInput(audio=vox_pb2.AudioFrame(pcm16=b"pcm", sample_rate=44_100)),
            )
            is True
        )
        assert (
            await execute_stream_input_message(
                session,
                vox_pb2.StreamInput(opus_frame=vox_pb2.OpusFrame(data=b"opus", sample_rate=48_000, channels=2)),
            )
            is True
        )
        assert (
            await execute_stream_input_message(
                session,
                vox_pb2.StreamInput(encoded_audio=vox_pb2.EncodedAudioFrame(data=b"wav", format="wav")),
            )
            is True
        )

        assert session.calls[0][0] == "configure"
        assert session.calls[0][1][0].model == "parakeet"
        assert session.calls[0][1][0].sample_rate == 48_000
        assert session.calls[1] == ("pcm16", (b"pcm",), {"sample_rate": 44_100})
        assert session.calls[2] == (
            "opus",
            (b"opus",),
            {"sample_rate": 48_000, "channels": 2},
        )
        assert session.calls[3] == ("encoded", (b"wav",), {"format_hint": "wav"})

    @pytest.mark.asyncio
    async def test_execute_stream_input_message_reports_end_without_side_effects(self):
        from vox.grpc import vox_pb2
        from vox.grpc.streaming_messages import execute_stream_input_message

        class RecordingSession:
            calls: list[str] = []

        assert (
            await execute_stream_input_message(
                RecordingSession(),
                vox_pb2.StreamInput(end_of_stream=vox_pb2.EndOfStream()),
            )
            is False
        )

        assert RecordingSession.calls == []

    @pytest.mark.asyncio
    async def test_execute_stream_input_message_preserves_empty_message_noop(self):
        from vox.grpc import vox_pb2
        from vox.grpc.streaming_messages import execute_stream_input_message

        class RecordingSession:
            calls: list[str] = []

        assert await execute_stream_input_message(RecordingSession(), vox_pb2.StreamInput()) is True
        assert RecordingSession.calls == []


class TestSynthesisServicerMapping:
    @pytest.mark.asyncio
    async def test_synthesize_no_input_aborts_with_invalid_argument(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        servicer = SynthesisServicer(_make_store(tmp_path), MagicMock(), MagicMock())
        with pytest.raises(Exception, match="gRPC abort"):
            async for _ in servicer.Synthesize(
                vox_pb2.SynthesizeRequest(model="kokoro:v1.0", input=""),
                FakeContext(),
            ):
                pass

    @pytest.mark.asyncio
    async def test_synthesize_request_build_errors_are_mapped_to_grpc(self, tmp_path, monkeypatch):
        from vox.grpc import synthesis_servicer
        from vox.grpc.synthesis_servicer import SynthesisServicer

        def fail_build(**_kwargs):
            raise InvalidConfigError("bad grpc synthesis config")

        monkeypatch.setattr(synthesis_servicer, "synthesis_request_from_fields", fail_build)

        context = FakeContext()
        servicer = SynthesisServicer(_make_store(tmp_path), MagicMock(), MagicMock())
        with pytest.raises(Exception, match="gRPC abort"):
            async for _ in servicer.Synthesize(
                vox_pb2.SynthesizeRequest(model="kokoro:v1.0", input="hello"),
                context,
            ):
                pass

        assert context._code is grpc.StatusCode.INVALID_ARGUMENT
        assert context._details == "bad grpc synthesis config"

    @pytest.mark.asyncio
    async def test_synthesize_passes_json_params_to_operation(self, tmp_path, monkeypatch):
        from vox.grpc import synthesis_servicer
        from vox.grpc.synthesis_servicer import SynthesisServicer

        captured = None

        async def fake_synthesize_raw(**kwargs):
            nonlocal captured
            captured = kwargs["request"]

            async def chunks():
                yield SynthesisRawChunk(audio=b"audio", sample_rate=24_000, is_final=True)

            return chunks()

        monkeypatch.setattr(synthesis_servicer, "synthesize_raw", fake_synthesize_raw)
        params = Struct()
        params.update(
            {
                "temperature": 0.7,
                "seed": 123,
                "stream": True,
                "style": "warm",
            }
        )
        servicer = SynthesisServicer(_make_store(tmp_path), MagicMock(), MagicMock())

        chunks = [
            chunk
            async for chunk in servicer.Synthesize(
                vox_pb2.SynthesizeRequest(model="expressive:latest", input="hello", params=params),
                FakeContext(),
            )
        ]

        assert len(chunks) == 1
        assert captured is not None
        assert captured.params == {
            "temperature": pytest.approx(0.7),
            "seed": 123,
            "stream": True,
            "style": "warm",
        }
        assert isinstance(captured.params["seed"], int)

    @pytest.mark.asyncio
    async def test_list_voices_empty(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        scheduler = MagicMock()
        scheduler.list_loaded.return_value = []
        servicer = SynthesisServicer(_make_store(tmp_path), MagicMock(), scheduler)
        resp = await servicer.ListVoices(vox_pb2.ListVoicesRequest(), FakeContext())
        assert len(resp.voices) == 0

    @pytest.mark.asyncio
    async def test_create_voice_returns_proto_with_id_and_name(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        store = _make_store(tmp_path)
        servicer = SynthesisServicer(store, MagicMock(), MagicMock())
        resp = await servicer.CreateVoice(
            vox_pb2.CreateVoiceRequest(
                name="Roy",
                audio=encode_wav(np.full(16_000, 0.1, dtype=np.float32), 16_000),
                format_hint="wav",
                language="en",
                gender="male",
            ),
            FakeContext(),
        )
        assert resp.voice.name == "Roy"
        assert resp.voice.is_cloned is True

    @pytest.mark.asyncio
    async def test_create_voice_request_build_errors_are_mapped_to_grpc(self, tmp_path, monkeypatch):
        from vox.grpc import synthesis_servicer
        from vox.grpc.synthesis_servicer import SynthesisServicer

        def fail_build(**_kwargs):
            raise InvalidConfigError("bad grpc voice config")

        monkeypatch.setattr(synthesis_servicer, "create_voice_request_from_fields", fail_build)

        context = FakeContext()
        servicer = SynthesisServicer(_make_store(tmp_path), MagicMock(), MagicMock())
        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.CreateVoice(
                vox_pb2.CreateVoiceRequest(name="Roy", audio=b"abc"),
                context,
            )

        assert context._code is grpc.StatusCode.INVALID_ARGUMENT
        assert context._details == "bad grpc voice config"

    @pytest.mark.asyncio
    async def test_create_voice_invalid_reference_aborts_with_invalid_argument(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        store = _make_store(tmp_path)
        context = FakeContext()
        servicer = SynthesisServicer(store, MagicMock(), MagicMock())

        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.CreateVoice(
                vox_pb2.CreateVoiceRequest(
                    name="Roy",
                    audio=encode_wav(np.full(4_000, 0.1, dtype=np.float32), 16_000),
                    format_hint="wav",
                ),
                context,
            )

        assert context._code is grpc.StatusCode.INVALID_ARGUMENT
        assert "too short" in context._details

    @pytest.mark.asyncio
    async def test_create_voice_decode_failure_aborts_with_invalid_argument(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        store = _make_store(tmp_path)
        context = FakeContext()
        servicer = SynthesisServicer(store, MagicMock(), MagicMock())

        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.CreateVoice(
                vox_pb2.CreateVoiceRequest(
                    name="Roy",
                    audio=b"not an audio file",
                    format_hint="wav",
                ),
                context,
            )

        assert context._code is grpc.StatusCode.INVALID_ARGUMENT

    @pytest.mark.asyncio
    async def test_delete_voice_returns_deleted_proto(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        store = _make_store(tmp_path)
        create_stored_voice(
            store,
            voice_id="voice1234",
            name="Roy",
            audio_bytes=encode_wav(np.full(16_000, 0.1, dtype=np.float32), 16_000),
            content_type="audio/wav",
        )
        servicer = SynthesisServicer(store, MagicMock(), MagicMock())
        resp = await servicer.DeleteVoice(
            vox_pb2.DeleteVoiceRequest(id="voice1234"),
            FakeContext(),
        )
        assert resp.id == "voice1234"
        assert resp.deleted is True

    @pytest.mark.asyncio
    async def test_list_voices_includes_loaded_model_field(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        store = _make_store(tmp_path)
        create_stored_voice(
            store,
            voice_id="voice1234",
            name="Roy",
            audio_bytes=encode_wav(np.full(16_000, 0.1, dtype=np.float32), 16_000),
            content_type="audio/wav",
        )
        loaded = LoadedModelInfo(name="test-tts", tag="latest", type=ModelType.TTS, device="cpu")
        scheduler = DummyScheduler(FakeCloneableTTSAdapter(), loaded_models=[loaded])
        servicer = SynthesisServicer(store, MagicMock(), scheduler)
        resp = await servicer.ListVoices(vox_pb2.ListVoicesRequest(), FakeContext())
        assert any(v.id == "voice1234" and v.is_cloned for v in resp.voices)


class TestGrpcVoiceMessages:
    def test_list_voices_response_matches_operation_payload_with_proto_defaults(self):
        from vox.grpc.voice_messages import list_voices_response

        listed = [
            ListedVoice(
                voice=VoiceInfo(
                    id="default",
                    name="Default",
                    language=None,
                    gender=None,
                    description=None,
                    is_cloned=False,
                ),
                model=None,
            )
        ]

        operation_payload = list_voices_payload(listed, include_model=True)["voices"][0]
        message = list_voices_response(listed).voices[0]

        assert message.id == operation_payload["id"]
        assert message.name == operation_payload["name"]
        assert message.language == ""
        assert message.gender == ""
        assert message.description == ""
        assert message.model == ""
        assert message.is_cloned is False

    def test_create_voice_response_uses_created_voice_contract(self):
        from vox.grpc.voice_messages import create_voice_response

        voice = SimpleNamespace(
            id="voice1234",
            name="Roy",
            language=None,
            gender="male",
            created_at=123,
        )

        operation_payload = created_voice_payload(voice)
        message = create_voice_response(voice)

        assert message.voice.id == operation_payload["id"]
        assert message.voice.name == operation_payload["name"]
        assert message.voice.language == ""
        assert message.voice.gender == operation_payload["gender"]
        assert message.voice.is_cloned == operation_payload["is_cloned"]
        assert message.created_at == operation_payload["created_at"]

    def test_delete_voice_response_uses_deleted_voice_contract(self):
        from vox.grpc.voice_messages import delete_voice_response

        message = delete_voice_response(DeleteVoiceResult(voice_id="voice1234"))

        assert message.id == "voice1234"
        assert message.deleted is True


class TestGrpcHealthMessages:
    def test_health_status_response_matches_operation_payload(self):
        from vox.grpc.health_messages import health_status_response

        result = HealthStatusResult()
        operation_payload = health_status_payload(result)
        message = health_status_response(result)

        assert message.status == operation_payload["status"]

    def test_list_loaded_models_response_matches_operation_payload_fields(self):
        from vox.grpc.health_messages import list_loaded_models_response

        result = ListLoadedModelsResult(
            models=[
                LoadedModelInfo(
                    name="parakeet-stt-onnx",
                    tag="tdt-0.6b-v3",
                    type=ModelType.STT,
                    device="cuda",
                    vram_bytes=4096,
                    loaded_at=1.5,
                    last_used=2.5,
                    ref_count=2,
                    is_evictable=True,
                    is_trimmable=True,
                    backend_memory={"workspace_bytes": 128},
                )
            ]
        )

        operation_payload = list_loaded_models_payload(result)["models"][0]
        message = list_loaded_models_response(result).models[0]

        for field, value in operation_payload.items():
            assert getattr(message, field) == value

        assert not hasattr(message, "is_evictable")
        assert not hasattr(message, "backend_memory")


class TestGrpcSynthesisMessages:
    def test_audio_chunk_message_uses_raw_synthesis_chunk_contract(self):
        from vox.grpc.synthesis_messages import audio_chunk_message

        chunk = SynthesisRawChunk(audio=b"abc", sample_rate=24_000, is_final=False)

        message = audio_chunk_message(chunk)

        assert message.audio == chunk.audio
        assert message.sample_rate == chunk.sample_rate
        assert message.is_final == chunk.is_final


class TestProtoMessages:
    def test_pull_progress_fields(self):
        msg = vox_pb2.PullProgress(status="downloading model.onnx", completed=1, total=3)
        assert msg.status == "downloading model.onnx"
        assert msg.completed == 1
        assert msg.total == 3

    def test_audio_chunk_fields(self):
        audio_bytes = np.zeros(100, dtype=np.float32).tobytes()
        msg = vox_pb2.AudioChunk(audio=audio_bytes, sample_rate=24000, is_final=False)
        assert len(msg.audio) == 400
        assert msg.sample_rate == 24000
        assert msg.is_final is False

    def test_transcribe_request_fields(self):
        msg = vox_pb2.TranscribeRequest(
            audio=b"\x00\x00",
            model="whisper:large-v3",
            language="en",
            word_timestamps=True,
            temperature=0.2,
        )
        assert msg.model == "whisper:large-v3"
        assert msg.word_timestamps is True

    def test_synthesize_request_fields(self):
        params = Struct()
        params.update({"temperature": 0.7, "seed": 123})
        msg = vox_pb2.SynthesizeRequest(
            model="kokoro:v1.0",
            input="Hello world",
            voice="af_heart",
            speed=1.5,
            params=params,
        )
        assert msg.input == "Hello world"
        assert msg.speed == pytest.approx(1.5)
        assert dict(msg.params) == {"temperature": pytest.approx(0.7), "seed": 123.0}

    def test_voice_info_fields(self):
        msg = vox_pb2.VoiceInfo(id="af_heart", name="Heart", language="en-us", gender="female")
        assert msg.id == "af_heart"
        assert msg.gender == "female"

    def test_create_voice_request_fields(self):
        msg = vox_pb2.CreateVoiceRequest(name="Roy", audio=b"123", format_hint="wav")
        assert msg.name == "Roy"
        assert msg.audio == b"123"
        assert msg.format_hint == "wav"

    def test_delete_voice_response_fields(self):
        msg = vox_pb2.DeleteVoiceResponse(id="voice1234", deleted=True)
        assert msg.id == "voice1234"
        assert msg.deleted is True
