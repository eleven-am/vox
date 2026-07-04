from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.fakes import FakeScheduler as DummyScheduler
from tests.fakes import FakeSTTAdapter as FakeSTT
from tests.fakes import FakeTTSAdapter as FakeTTS
from vox.audio.codecs import encode_wav
from vox.core.types import AdapterInfo, ModelFormat, ModelType, TranscribeResult, TranscriptSegment, WordTimestamp
from vox.operations.errors import (
    EmptyAudioError,
    NoDefaultModelError,
    StoredModelNotFoundError,
    WrongModelTypeError,
)
from vox.operations.transcription import (
    AnnotateRequest,
    Entity,
    TranscriptionRequest,
    TranscriptionResultBundle,
    _choose_onset_result,
    _transcribe_chunk,
    annotate_request_from_fields,
    annotate_text,
    entity_payload,
    format_hint_from_content_type,
    openai_transcription_payload,
    openai_transcription_response,
    parse_timestamp_granularities,
    transcribe,
    transcript_segment_payload,
    transcription_request_from_fields,
    word_timestamp_payload,
)


def _wav_bytes(dur_s: float = 1.0, sr: int = 16_000) -> bytes:
    audio = np.zeros(int(dur_s * sr), dtype=np.float32)
    return encode_wav(audio, sr)


def _tone_wav_bytes(dur_s: float = 1.0, sr: int = 16_000) -> bytes:
    audio = np.full(int(dur_s * sr), 0.25, dtype=np.float32)
    return encode_wav(audio, sr)


class LeadingContextSensitiveSTT(FakeSTT):
    def __init__(self) -> None:
        super().__init__()
        self.last_audio: np.ndarray | None = None

    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        self.last_audio = audio
        context_samples = 5 * 16_000
        has_leading_context = (
            audio.shape[0] > context_samples
            and np.allclose(audio[:context_samples], 0)
            and np.max(np.abs(audio[context_samples:])) > 0
        )
        if not has_leading_context:
            return TranscribeResult(
                text="kept",
                language="en",
                duration_ms=1000,
                segments=(TranscriptSegment(text="kept", start_ms=0, end_ms=1000),),
            )

        return TranscribeResult(
            text="start kept",
            language="en",
            duration_ms=6000,
            segments=(
                TranscriptSegment(
                    text="start kept",
                    start_ms=5000,
                    end_ms=6000,
                    words=(
                        WordTimestamp(word="start", start_ms=5000, end_ms=5400),
                        WordTimestamp(word="kept", start_ms=5400, end_ms=6000),
                    ),
                ),
            ),
        )


def test_transcription_request_from_fields_normalizes_transport_input():
    request = transcription_request_from_fields(
        audio=b"abc",
        model="",
        format_hint="",
        language="",
        word_timestamps=True,
        temperature=-1.0,
        annotate_text=True,
    )

    assert request == TranscriptionRequest(
        audio=b"abc",
        model="",
        format_hint=None,
        language=None,
        word_timestamps=True,
        temperature=0.0,
        annotate_text=True,
    )


def test_transcription_request_from_fields_preserves_positive_temperature():
    request = transcription_request_from_fields(
        audio=b"abc",
        model="fake-stt:latest",
        temperature=0.25,
    )

    assert request.temperature == 0.25


def test_annotate_request_from_fields_normalizes_transport_input():
    assert annotate_request_from_fields(text=None, language="") == AnnotateRequest(
        text="",
        language="en",
    )
    assert annotate_request_from_fields(text="Alice visited Paris", language="fr") == AnnotateRequest(
        text="Alice visited Paris",
        language="fr",
    )


def test_format_hint_from_content_type_normalizes_audio_mime_types():
    assert format_hint_from_content_type("audio/wav") == "wav"
    assert format_hint_from_content_type("audio/mpeg") == "mp3"
    assert format_hint_from_content_type("audio/x-wav") == "wav"
    assert format_hint_from_content_type("audio/x-flac") == "flac"
    assert format_hint_from_content_type("audio/ogg") == "ogg"
    assert format_hint_from_content_type("audio/webm") == "webm"
    assert format_hint_from_content_type("audio/flac") == "flac"
    assert format_hint_from_content_type("application/octet-stream") is None
    assert format_hint_from_content_type(None) is None
    assert format_hint_from_content_type("") is None


def test_parse_timestamp_granularities_accepts_openai_form_variants():
    assert parse_timestamp_granularities([]) == {"segment"}
    assert parse_timestamp_granularities(["word"]) == {"word"}
    assert parse_timestamp_granularities(['["word","segment"]']) == {"word", "segment"}
    assert parse_timestamp_granularities(["word,segment"]) == {"word", "segment"}
    assert parse_timestamp_granularities([" ", "word"]) == {"word"}


class LeadingContextHurtsSTT(FakeSTT):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        self.calls += 1
        context_samples = 5 * 16_000
        has_leading_context = (
            audio.shape[0] > context_samples
            and np.allclose(audio[:context_samples], 0)
            and np.max(np.abs(audio[context_samples:])) > 0
        )
        text = "middle only" if has_leading_context else "beginning middle only"
        return TranscribeResult(
            text=text,
            language="en",
            duration_ms=int(audio.shape[0] / 16_000 * 1000),
            segments=(TranscriptSegment(text=text, start_ms=0, end_ms=1000),),
        )


@pytest.mark.asyncio
async def test_transcribe_returns_bundle_with_processing_ms():
    adapter = FakeSTT()
    sched = DummyScheduler(adapter)
    registry = MagicMock()
    bundle = await transcribe(
        scheduler=sched, registry=registry, store=None,
        request=TranscriptionRequest(audio=_wav_bytes(), model="fake-stt:latest"),
    )
    assert bundle.result.text == "hello world"
    assert bundle.result.model == "fake-stt:latest"
    assert bundle.processing_ms >= 0


@pytest.mark.asyncio
async def test_transcribe_passes_kwargs_to_adapter():
    adapter = FakeSTT()
    sched = DummyScheduler(adapter)
    registry = MagicMock()
    await transcribe(
        scheduler=sched, registry=registry, store=None,
        request=TranscriptionRequest(
            audio=_wav_bytes(),
            model="fake-stt:latest",
            language="fr",
            word_timestamps=True,
            temperature=0.7,
        ),
    )
    assert adapter.last_kwargs == {"language": "fr", "word_timestamps": True, "temperature": 0.7}


@pytest.mark.asyncio
async def test_transcribe_adds_leading_context_and_strips_timestamps():
    adapter = LeadingContextSensitiveSTT()
    sched = DummyScheduler(adapter)
    registry = MagicMock()

    bundle = await transcribe(
        scheduler=sched, registry=registry, store=None,
        request=TranscriptionRequest(audio=_tone_wav_bytes(), model="fake-stt:latest"),
    )

    assert bundle.result.text == "start kept"
    assert bundle.result.duration_ms == 1000
    assert adapter.last_audio is not None
    assert adapter.last_audio.shape[0] == 6 * 16_000
    assert np.allclose(adapter.last_audio[: 5 * 16_000], 0)
    assert np.max(np.abs(adapter.last_audio[5 * 16_000 :])) > 0

    segment = bundle.result.segments[0]
    assert (segment.start_ms, segment.end_ms) == (0, 1000)
    assert [(word.word, word.start_ms, word.end_ms) for word in segment.words] == [
        ("start", 0, 400),
        ("kept", 400, 1000),
    ]


@pytest.mark.asyncio
async def test_transcribe_onset_guard_keeps_direct_result_when_padding_hurts():
    adapter = LeadingContextHurtsSTT()
    sched = DummyScheduler(adapter)
    registry = MagicMock()

    bundle = await transcribe(
        scheduler=sched, registry=registry, store=None,
        request=TranscriptionRequest(audio=_tone_wav_bytes(), model="fake-stt:latest"),
    )

    assert bundle.result.text == "beginning middle only"
    assert adapter.calls == 2


class OnsetHallucinatingSTT(FakeSTT):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        self.calls += 1
        context_samples = 5 * 16_000
        has_leading_context = (
            audio.shape[0] > context_samples
            and np.allclose(audio[:context_samples], 0)
            and np.max(np.abs(audio[context_samples:])) > 0
        )
        text = (
            "trying my speech to text model again"
            if has_leading_context
            else "trying my speech to text mode that i can"
        )
        return TranscribeResult(
            text=text,
            language="en",
            duration_ms=int(audio.shape[0] / 16_000 * 1000),
            segments=(TranscriptSegment(text=text, start_ms=0, end_ms=1000),),
        )


class ConfidentNemoLikeSTT(FakeSTT):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="parakeet-stt-nemo",
            type=ModelType.STT,
            architectures=("parakeet-nemo",),
            default_sample_rate=16_000,
            supported_formats=(ModelFormat.PYTORCH,),
        )

    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        self.calls += 1
        text = "this direct transcript is already confident"
        return TranscribeResult(
            text=text,
            language="en",
            duration_ms=int(audio.shape[0] / 16_000 * 1000),
            segments=(TranscriptSegment(text=text, start_ms=0, end_ms=1000),),
        )


@pytest.mark.asyncio
async def test_transcribe_onset_guard_prefers_padded_over_longer_hallucination():
    adapter = OnsetHallucinatingSTT()
    sched = DummyScheduler(adapter)
    registry = MagicMock()

    bundle = await transcribe(
        scheduler=sched, registry=registry, store=None,
        request=TranscriptionRequest(audio=_tone_wav_bytes(), model="fake-stt:latest"),
    )

    assert bundle.result.text == "trying my speech to text model again"
    assert adapter.calls == 2


@pytest.mark.asyncio
async def test_transcribe_onset_guard_skips_padded_pass_for_confident_nemo_result():
    adapter = ConfidentNemoLikeSTT()
    sched = DummyScheduler(adapter)
    registry = MagicMock()

    bundle = await transcribe(
        scheduler=sched, registry=registry, store=None,
        request=TranscriptionRequest(audio=_tone_wav_bytes(dur_s=4), model="fake-stt:latest"),
    )

    assert bundle.result.text == "this direct transcript is already confident"
    assert adapter.calls == 1


def test_choose_onset_result_falls_back_to_direct_when_padded_empty():
    direct = TranscribeResult(text="hello world", language="en", duration_ms=1000)
    padded = TranscribeResult(text="", language="en", duration_ms=6000)
    assert _choose_onset_result(direct, padded) is direct


def test_choose_onset_result_keeps_direct_when_padded_degenerate():
    direct = TranscribeResult(text="beginning middle only", language="en", duration_ms=1000)
    padded = TranscribeResult(text="middle only", language="en", duration_ms=6000)
    assert _choose_onset_result(direct, padded) is direct


def test_choose_onset_result_prefers_padded_within_ratio():
    direct = TranscribeResult(text="trying my speech to text mode that i can", language="en", duration_ms=1000)
    padded = TranscribeResult(text="trying my speech to text model again", language="en", duration_ms=6000)
    assert _choose_onset_result(direct, padded) is padded


class CountingScheduler(DummyScheduler):
    def __init__(self, adapter):
        super().__init__(adapter)
        self.acquire_names: list[str] = []

    @asynccontextmanager
    async def acquire(self, name: str):
        self.acquire_names.append(name)
        async with super().acquire(name) as adapter:
            yield adapter


def test_openai_transcription_payload_uses_seconds_and_top_level_words():
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
                        WordTimestamp(word="Paris", start_ms=800, end_ms=1200, confidence=0.8),
                    ),
                ),
            ),
        ),
        processing_ms=25,
        entities=(Entity(type="PERSON", text="Alice", start_char=0, end_char=5),),
        topics=("travel",),
    )

    payload = openai_transcription_payload(
        bundle,
        include_segments=True,
        include_words=True,
    )

    assert payload["duration"] == 1.2
    assert payload["processing_ms"] == 25
    assert payload["entities"] == [
        {"type": "PERSON", "text": "Alice", "start_char": 0, "end_char": 5},
    ]
    assert payload["topics"] == ["travel"]
    assert payload["segments"][0] == {
        "id": 0,
        "seek": 0,
        "start": 0.0,
        "end": 1.2,
        "text": "Alice visited Paris",
        "tokens": [],
        "temperature": 0.0,
        "avg_logprob": 0.0,
        "compression_ratio": 0.0,
        "no_speech_prob": 0.0,
    }
    assert payload["words"] == [
        {"word": "Alice", "start": 0.0, "end": 0.5},
        {"word": "Paris", "start": 0.8, "end": 1.2},
    ]


def test_openai_transcription_payload_respects_granularity_flags():
    bundle = TranscriptionResultBundle(
        result=TranscribeResult(
            text="hello",
            language="en",
            duration_ms=100,
            model="fake-stt:latest",
            segments=(TranscriptSegment(text="hello", start_ms=0, end_ms=100),),
        ),
        processing_ms=1,
    )

    payload = openai_transcription_payload(bundle)

    assert payload["text"] == "hello"
    assert "segments" not in payload
    assert "words" not in payload


def test_openai_transcription_response_owns_response_format_policy():
    bundle = TranscriptionResultBundle(
        result=TranscribeResult(
            text="hello",
            language="en",
            duration_ms=100,
            model="fake-stt:latest",
            segments=(
                TranscriptSegment(
                    text="hello",
                    start_ms=0,
                    end_ms=100,
                    words=(WordTimestamp(word="hello", start_ms=0, end_ms=100),),
                ),
            ),
        ),
        processing_ms=1,
    )

    text_response = openai_transcription_response(
        bundle,
        response_format="text",
        timestamp_granularities=set(),
    )
    assert text_response.is_text
    assert text_response.payload == "hello"

    default_response = openai_transcription_response(
        bundle,
        response_format="json",
        timestamp_granularities=set(),
    )
    assert not default_response.is_text
    assert default_response.payload == {"text": "hello"}

    verbose_response = openai_transcription_response(
        bundle,
        response_format="verbose_json",
        timestamp_granularities={"word"},
    )
    assert not verbose_response.is_text
    assert "words" in verbose_response.payload
    assert "segments" not in verbose_response.payload


def test_transcript_payload_helpers_normalize_dataclass_and_dict_shapes():
    entity = Entity(type="PERSON", text="Alice", start_char=0, end_char=5)
    word = WordTimestamp(word="Alice", start_ms=10, end_ms=500, confidence=0.9)
    segment = TranscriptSegment(text="Alice", start_ms=10, end_ms=500, words=(word,))

    assert entity_payload(entity) == {
        "type": "PERSON",
        "text": "Alice",
        "start_char": 0,
        "end_char": 5,
    }
    assert word_timestamp_payload(word) == {
        "word": "Alice",
        "start_ms": 10,
        "end_ms": 500,
        "confidence": 0.9,
    }
    assert transcript_segment_payload(segment) == {
        "text": "Alice",
        "start_ms": 10,
        "end_ms": 500,
        "words": [{"word": "Alice", "start_ms": 10, "end_ms": 500, "confidence": 0.9}],
    }

    assert transcript_segment_payload({
        "text": "Paris",
        "start_ms": 20,
        "end_ms": 700,
        "words": [{"word": "Paris", "start_ms": 20, "end_ms": 700}],
    }) == {
        "text": "Paris",
        "start_ms": 20,
        "end_ms": 700,
        "words": [{"word": "Paris", "start_ms": 20, "end_ms": 700}],
    }


@pytest.mark.asyncio
async def test_transcribe_chunk_without_onset_guard_runs_single_stt_pass():
    adapter = LeadingContextHurtsSTT()
    audio = np.full(16_000, 0.25, dtype=np.float32)

    result = await _transcribe_chunk(
        adapter,
        audio,
        sample_rate=16_000,
        duration_ms=1000,
        guard_onset=False,
        language=None,
        word_timestamps=False,
        temperature=0.0,
    )

    assert adapter.calls == 1
    assert result.text == "middle only"


@pytest.mark.asyncio
async def test_transcribe_releases_scheduler_between_long_audio_chunks():
    adapter = ConfidentNemoLikeSTT()
    sched = CountingScheduler(adapter)
    registry = MagicMock()

    bundle = await transcribe(
        scheduler=sched,
        registry=registry,
        store=None,
        request=TranscriptionRequest(
            audio=_tone_wav_bytes(dur_s=5 * 60 + 1),
            model="fake-stt:latest",
        ),
    )

    assert bundle.result.text
    assert sched.acquire_names == ["fake-stt:latest", "fake-stt:latest"]


@pytest.mark.asyncio
async def test_transcribe_raises_on_empty_audio():
    sched = DummyScheduler(FakeSTT())
    registry = MagicMock()
    with pytest.raises(EmptyAudioError):
        await transcribe(
            scheduler=sched, registry=registry, store=None,
            request=TranscriptionRequest(audio=b"", model="fake-stt:latest"),
        )


@pytest.mark.asyncio
async def test_transcribe_raises_when_no_default_model():
    sched = DummyScheduler(FakeSTT())
    registry = MagicMock()
    registry.available_models.return_value = {}
    with pytest.raises(NoDefaultModelError):
        await transcribe(
            scheduler=sched, registry=registry, store=None,
            request=TranscriptionRequest(audio=_wav_bytes(), model=""),
        )


@pytest.mark.asyncio
async def test_transcribe_raises_when_adapter_is_tts():
    sched = DummyScheduler(FakeTTS())
    registry = MagicMock()
    with pytest.raises(WrongModelTypeError):
        await transcribe(
            scheduler=sched, registry=registry, store=None,
            request=TranscriptionRequest(audio=_wav_bytes(), model="fake-tts:latest"),
        )


@pytest.mark.asyncio
async def test_transcribe_translates_model_not_found_to_operation_error():
    sched = DummyScheduler()
    registry = MagicMock()

    with pytest.raises(StoredModelNotFoundError):
        await transcribe(
            scheduler=sched,
            registry=registry,
            store=None,
            request=TranscriptionRequest(audio=_wav_bytes(), model="missing:latest"),
        )


@pytest.mark.asyncio
async def test_transcribe_resolves_default_from_registry():
    adapter = FakeSTT()
    sched = DummyScheduler(adapter)
    registry = MagicMock()
    registry.available_models.return_value = {
        "fake-stt": {"latest": {"type": "stt"}},
    }
    bundle = await transcribe(
        scheduler=sched, registry=registry, store=None,
        request=TranscriptionRequest(audio=_wav_bytes()),
    )
    assert bundle.result.model == "fake-stt:latest"


@pytest.mark.asyncio
async def test_transcribe_annotate_text_when_requested():
    adapter = FakeSTT(text="Alice visited Paris", language="en")
    sched = DummyScheduler(adapter)
    registry = MagicMock()
    bundle = await transcribe(
        scheduler=sched, registry=registry, store=None,
        request=TranscriptionRequest(
            audio=_wav_bytes(), model="fake-stt:latest", annotate_text=True,
        ),
    )
    assert isinstance(bundle.entities, tuple)
    assert isinstance(bundle.topics, tuple)


def test_annotate_text_returns_dataclass():
    result = annotate_text(AnnotateRequest(text="Alice visited Paris", language="en"))
    assert hasattr(result, "entities")
    assert hasattr(result, "topics")
