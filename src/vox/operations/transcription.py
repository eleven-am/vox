from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from vox.audio.merger import merge_transcripts
from vox.audio.pipeline import prepare_for_stt_chunks
from vox.audio.stt_runner import run_stt, run_stt_with_leading_context
from vox.core.adapter import STTAdapter
from vox.core.errors import ModelNotFoundError, VoxError
from vox.core.ner import annotate
from vox.core.types import TranscribeResult
from vox.operations.defaults import resolve_default_model
from vox.operations.errors import (
    EmptyAudioError,
    NoDefaultModelError,
    StoredModelNotFoundError,
    WrongModelTypeError,
)

logger = logging.getLogger(__name__)

ONSET_GUARD_COMPARE_MAX_MS = 60_000
ONSET_GUARD_SPARSE_MIN_MS = 1_500
ONSET_GUARD_SPARSE_MAX_WORDS = 1
ONSET_GUARD_PADDED_MIN_SCORE_RATIO = 0.75


@dataclass(frozen=True)
class Entity:
    type: str
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class TranscriptionRequest:
    audio: bytes
    model: str = ""
    format_hint: str | None = None
    language: str | None = None
    word_timestamps: bool = False
    temperature: float = 0.0
    annotate_text: bool = False


@dataclass(frozen=True)
class TranscriptionResultBundle:
    result: TranscribeResult
    processing_ms: int
    entities: tuple[Entity, ...] = ()
    topics: tuple[str, ...] = ()


def milliseconds_to_seconds(value: int | None) -> float:
    if value is None:
        return 0.0
    return value / 1000.0


def openai_transcription_payload(
    bundle: TranscriptionResultBundle,
    *,
    include_segments: bool = False,
    include_words: bool = False,
) -> dict[str, Any]:
    result = bundle.result
    response: dict[str, Any] = {
        "model": result.model,
        "text": result.text,
        "language": result.language,
        "duration": milliseconds_to_seconds(result.duration_ms),
        "processing_ms": bundle.processing_ms,
    }
    if bundle.entities:
        response["entities"] = [asdict(entity) for entity in bundle.entities]
    if bundle.topics:
        response["topics"] = list(bundle.topics)
    if include_segments:
        response["segments"] = openai_segments_payload(result)
    if include_words:
        response["words"] = openai_words_payload(result)
    return response


def openai_segments_payload(result: TranscribeResult) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for idx, segment in enumerate(result.segments):
        segments.append(
            {
                "id": idx,
                "seek": segment.start_ms or 0,
                "start": milliseconds_to_seconds(segment.start_ms),
                "end": milliseconds_to_seconds(segment.end_ms),
                "text": segment.text,
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
            }
        )
    return segments


def openai_words_payload(result: TranscribeResult) -> list[dict[str, Any]]:
    return [
        {
            "word": word.word,
            "start": milliseconds_to_seconds(word.start_ms),
            "end": milliseconds_to_seconds(word.end_ms),
        }
        for segment in result.segments
        for word in segment.words
    ]


async def transcribe(
    *,
    scheduler: Any,
    registry: Any,
    store: Any | None,
    request: TranscriptionRequest,
) -> TranscriptionResultBundle:
    model = request.model or resolve_default_model("stt", registry, store) or ""
    if not model:
        raise NoDefaultModelError("stt")

    if not request.audio:
        raise EmptyAudioError()

    chunks = prepare_for_stt_chunks(request.audio, format_hint=request.format_hint)
    first_signal_ms = _first_signal_ms(chunks[0].data, sample_rate=chunks[0].sample_rate)

    start_time = time.perf_counter()
    try:
        async with scheduler.acquire(model) as adapter:
            if not isinstance(adapter, STTAdapter):
                raise WrongModelTypeError(model, "STT")
            per_chunk: list[tuple] = []
            for chunk in chunks:
                partial = await _transcribe_chunk(
                    adapter,
                    chunk.data,
                    sample_rate=chunk.sample_rate,
                    duration_ms=chunk.duration_ms,
                    guard_onset=chunk.offset_ms == 0,
                    language=request.language,
                    word_timestamps=request.word_timestamps,
                    temperature=request.temperature,
                )
                per_chunk.append((partial, chunk.offset_ms))
            result = merge_transcripts(per_chunk)
    except ModelNotFoundError as exc:
        raise StoredModelNotFoundError(exc.model) from exc
    except (WrongModelTypeError, VoxError):
        raise
    except Exception:
        logger.exception(f"Transcription failed for model {model}")
        raise

    processing_ms = int((time.perf_counter() - start_time) * 1000)
    result = replace(result, model=model)

    entities: tuple[Entity, ...] = ()
    topics: tuple[str, ...] = ()
    if request.annotate_text and result.text:
        lang = request.language or result.language or "en"
        ents, tops = annotate(result.text, lang)
        entities = tuple(
            Entity(type=e.type, text=e.text, start_char=e.start_char, end_char=e.end_char)
            for e in ents
        )
        topics = tuple(tops)

    logger.info(
        "transcribe %s input_bytes=%d format=%s audio_ms=%d first_signal_ms=%s chunks=%d processing_ms=%d chars=%d",
        model,
        len(request.audio),
        request.format_hint or "auto",
        result.duration_ms,
        first_signal_ms if first_signal_ms is not None else "none",
        len(chunks),
        processing_ms,
        len(result.text or ""),
    )
    return TranscriptionResultBundle(
        result=result,
        processing_ms=processing_ms,
        entities=entities,
        topics=topics,
    )


async def _transcribe_chunk(
    adapter: STTAdapter,
    audio,
    *,
    sample_rate: int,
    duration_ms: int,
    guard_onset: bool,
    language: str | None,
    word_timestamps: bool,
    temperature: float,
) -> TranscribeResult:
    if not guard_onset:
        return await _run_padded_stt(
            adapter,
            audio,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            language=language,
            word_timestamps=word_timestamps,
            temperature=temperature,
        )

    direct = await _run_stt(
        adapter,
        audio,
        language=language,
        word_timestamps=word_timestamps,
        temperature=temperature,
    )
    direct = replace(direct, duration_ms=duration_ms)

    if duration_ms > ONSET_GUARD_COMPARE_MAX_MS and not _looks_sparse(direct, duration_ms=duration_ms):
        return direct

    padded = await _run_padded_stt(
        adapter,
        audio,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        language=language,
        word_timestamps=word_timestamps,
        temperature=temperature,
    )
    chosen = _choose_onset_result(direct, padded)
    logger.info(
        "transcribe onset guard selected %s result direct_chars=%d padded_chars=%d duration_ms=%d",
        "direct" if chosen is direct else "padded",
        len(direct.text or ""),
        len(padded.text or ""),
        duration_ms,
    )
    return chosen


async def _run_padded_stt(
    adapter: STTAdapter,
    audio,
    *,
    sample_rate: int,
    duration_ms: int,
    language: str | None,
    word_timestamps: bool,
    temperature: float,
) -> TranscribeResult:
    return await run_stt_with_leading_context(
        adapter,
        audio,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        language=language,
        word_timestamps=word_timestamps,
        temperature=temperature,
    )


async def _run_stt(
    adapter: STTAdapter,
    audio,
    *,
    language: str | None,
    word_timestamps: bool,
    temperature: float,
) -> TranscribeResult:
    return await run_stt(
        adapter,
        audio,
        language=language,
        word_timestamps=word_timestamps,
        temperature=temperature,
    )


def _looks_sparse(result: TranscribeResult, *, duration_ms: int) -> bool:
    if duration_ms < ONSET_GUARD_SPARSE_MIN_MS:
        return False
    return len((result.text or "").split()) <= ONSET_GUARD_SPARSE_MAX_WORDS


def _choose_onset_result(direct: TranscribeResult, padded: TranscribeResult) -> TranscribeResult:
    padded_score = _transcript_score(padded)
    direct_score = _transcript_score(direct)
    if padded_score == 0:
        return direct
    if padded_score >= direct_score * ONSET_GUARD_PADDED_MIN_SCORE_RATIO:
        return padded
    return direct


def _transcript_score(result: TranscribeResult) -> int:
    text = (result.text or "").strip()
    if not text:
        return 0
    words = text.split()
    return len(words) * 8 + len(text)


def _first_signal_ms(audio, *, sample_rate: int) -> int | None:
    if sample_rate <= 0 or audio.size == 0:
        return None
    threshold = max(1e-4, float(np.max(np.abs(audio))) * 0.02)
    indices = np.flatnonzero(np.abs(audio) >= threshold)
    if indices.size == 0:
        return None
    return int(indices[0] / sample_rate * 1000)


@dataclass(frozen=True)
class AnnotateRequest:
    text: str = ""
    language: str = "en"


@dataclass(frozen=True)
class AnnotateResult:
    entities: tuple[Entity, ...] = field(default_factory=tuple)
    topics: tuple[str, ...] = field(default_factory=tuple)


def annotate_text(request: AnnotateRequest) -> AnnotateResult:
    text = request.text or ""
    language = request.language or "en"
    ents, tops = annotate(text, language)
    return AnnotateResult(
        entities=tuple(
            Entity(type=e.type, text=e.text, start_char=e.start_char, end_char=e.end_char)
            for e in ents
        ),
        topics=tuple(tops),
    )
