from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any

from vox.audio.merger import merge_transcripts
from vox.audio.pipeline import prepare_for_stt_chunks
from vox.core.adapter import STTAdapter
from vox.core.errors import ModelNotFoundError, VoxError
from vox.core.ner import annotate
from vox.core.types import TranscribeResult
from vox.operations.defaults import resolve_default_model
from vox.operations.errors import (
    EmptyAudioError,
    NoDefaultModelError,
    WrongModelTypeError,
)

logger = logging.getLogger(__name__)


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

    start_time = time.perf_counter()
    try:
        async with scheduler.acquire(model) as adapter:
            if not isinstance(adapter, STTAdapter):
                raise WrongModelTypeError(model, "STT")
            per_chunk: list[tuple] = []
            for chunk in chunks:
                partial = await asyncio.to_thread(
                    adapter.transcribe,
                    chunk.data,
                    language=request.language,
                    word_timestamps=request.word_timestamps,
                    temperature=request.temperature,
                )
                partial = replace(partial, duration_ms=chunk.duration_ms)
                per_chunk.append((partial, chunk.offset_ms))
            result = merge_transcripts(per_chunk)
    except (WrongModelTypeError, ModelNotFoundError, VoxError):
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
        "transcribe %s audio_ms=%d processing_ms=%d chars=%d",
        model, result.duration_ms, processing_ms, len(result.text or ""),
    )
    return TranscriptionResultBundle(
        result=result,
        processing_ms=processing_ms,
        entities=entities,
        topics=topics,
    )


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
