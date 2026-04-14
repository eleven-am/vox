from __future__ import annotations

import logging

import grpc
import numpy as np

from vox.core.adapter import TTSAdapter
from vox.core.errors import ModelNotFoundError, VoxError
from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2, vox_pb2_grpc

logger = logging.getLogger(__name__)


def _get_default_tts(registry: ModelRegistry, store: BlobStore) -> str:
    for m in store.list_models():
        if m.type.value == "tts":
            return m.full_name
    for name, tags in registry.available_models().items():
        for tag, entry in tags.items():
            if entry.get("type") == "tts":
                return f"{name}:{tag}"
    return ""


class SynthesisServicer(vox_pb2_grpc.SynthesisServiceServicer):

    def __init__(self, store: BlobStore, registry: ModelRegistry, scheduler: Scheduler) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler

    async def Synthesize(self, request, context):
        model = request.model or _get_default_tts(self._registry, self._store)
        if not model:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No model specified and no default TTS model available")

        if not request.input or not request.input.strip():
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No input text provided")

        try:
            async with self._scheduler.acquire(model) as adapter:
                if not isinstance(adapter, TTSAdapter):
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Model '{model}' is not a TTS model")

                async for chunk in adapter.synthesize(
                    request.input,
                    voice=request.voice or None,
                    speed=request.speed if request.speed > 0 else 1.0,
                    language=request.language or None,
                ):
                    yield vox_pb2.AudioChunk(
                        audio=chunk.audio,
                        sample_rate=chunk.sample_rate,
                        is_final=chunk.is_final,
                    )
        except ModelNotFoundError as e:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        except VoxError as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
        except Exception:
            logger.exception(f"Synthesis failed for model {model}")
            await context.abort(grpc.StatusCode.INTERNAL, "Internal synthesis error")

    async def ListVoices(self, request, context):
        scheduler = self._scheduler

        try:
            if not request.model:
                all_voices = []
                for loaded in scheduler.list_loaded():
                    if loaded.type.value == "tts":
                        full_name = f"{loaded.name}:{loaded.tag}"
                        async with scheduler.acquire(full_name) as adapter:
                            if isinstance(adapter, TTSAdapter):
                                for v in adapter.list_voices():
                                    all_voices.append(vox_pb2.VoiceInfo(
                                        id=v.id,
                                        name=v.name,
                                        language=v.language or "",
                                        gender=v.gender or "",
                                        description=v.description or "",
                                        is_cloned=v.is_cloned,
                                        model=full_name,
                                    ))
                return vox_pb2.ListVoicesResponse(voices=all_voices)

            async with scheduler.acquire(request.model) as adapter:
                if not isinstance(adapter, TTSAdapter):
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Model '{request.model}' is not a TTS model")

                voices = [
                    vox_pb2.VoiceInfo(
                        id=v.id,
                        name=v.name,
                        language=v.language or "",
                        gender=v.gender or "",
                        description=v.description or "",
                        is_cloned=v.is_cloned,
                    )
                    for v in adapter.list_voices()
                ]
                return vox_pb2.ListVoicesResponse(voices=voices)
        except ModelNotFoundError as e:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        except VoxError as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
