from __future__ import annotations

import asyncio
from concurrent.futures import Executor
from functools import partial

from numpy.typing import NDArray

from vox.audio.stt_context import add_stt_leading_context, strip_stt_leading_context
from vox.core.adapter import STTAdapter
from vox.core.types import TranscribeResult


async def run_stt(
    adapter: STTAdapter,
    audio: NDArray,
    *,
    language: str | None,
    word_timestamps: bool,
    temperature: float = 0.0,
    executor: Executor | None = None,
) -> TranscribeResult:
    call = partial(
        adapter.transcribe,
        audio,
        language=language,
        word_timestamps=word_timestamps,
        temperature=temperature,
    )
    if executor is None:
        return await asyncio.to_thread(call)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, call)


async def run_stt_with_leading_context(
    adapter: STTAdapter,
    audio: NDArray,
    *,
    sample_rate: int,
    duration_ms: int,
    language: str | None,
    word_timestamps: bool,
    temperature: float = 0.0,
    executor: Executor | None = None,
) -> TranscribeResult:
    audio_with_context, leading_context_ms = add_stt_leading_context(
        audio,
        sample_rate=sample_rate,
    )
    result = await run_stt(
        adapter,
        audio_with_context,
        language=language,
        word_timestamps=word_timestamps,
        temperature=temperature,
        executor=executor,
    )
    return strip_stt_leading_context(
        result,
        context_ms=leading_context_ms,
        duration_ms=duration_ms,
    )
