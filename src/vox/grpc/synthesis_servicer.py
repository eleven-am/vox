from __future__ import annotations

import logging

import grpc

from vox.conversation.text_buffer import split_for_tts
from vox.core.adapter import TTSAdapter
from vox.core.cloned_voices import (
    create_stored_voice,
    delete_stored_voice,
    generate_voice_id,
    list_stored_voices,
    resolve_voice_request,
)
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
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "No model specified and no default TTS model available",
            )

        if not request.input or not request.input.strip():
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No input text provided")

        try:
            async with self._scheduler.acquire(model) as adapter:
                if not isinstance(adapter, TTSAdapter):
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Model '{model}' is not a TTS model")

                voice, language, reference_audio, reference_text = _resolve_voice_request(
                    adapter,
                    self._store,
                    request.voice or None,
                    request.language or None,
                )
                max_chars = int(getattr(adapter.info(), "max_input_chars", 0) or 0)
                if max_chars > 0:
                    text_chunks = split_for_tts(request.input, max_chars=max_chars)
                else:
                    text_chunks = [request.input] if request.input.strip() else []
                for idx, text_chunk in enumerate(text_chunks):
                    is_last_text_chunk = idx == len(text_chunks) - 1
                    async for chunk in adapter.synthesize(
                        text_chunk,
                        voice=voice,
                        speed=request.speed if request.speed > 0 else 1.0,
                        language=language,
                        reference_audio=reference_audio,
                        reference_text=reference_text,
                    ):
                        yield vox_pb2.AudioChunk(
                            audio=chunk.audio,
                            sample_rate=chunk.sample_rate,
                            is_final=chunk.is_final and is_last_text_chunk,
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
                                for v in _voices_for_adapter(adapter, self._store):
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
                    for v in _voices_for_adapter(adapter, self._store)
                ]
                return vox_pb2.ListVoicesResponse(voices=voices)
        except ModelNotFoundError as e:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        except VoxError as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def CreateVoice(self, request, context):
        if not request.name or not request.name.strip():
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Voice name is required")
        if not request.audio:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Audio sample is required")

        try:
            voice = create_stored_voice(
                self._store,
                voice_id=generate_voice_id(self._store),
                name=request.name,
                audio_bytes=request.audio,
                content_type=request.format_hint or None,
                language=request.language or None,
                gender=request.gender or None,
                reference_text=request.reference_text or None,
            )
        except (TypeError, ValueError, RuntimeError) as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        return vox_pb2.CreateVoiceResponse(
            voice=vox_pb2.VoiceInfo(
                id=voice.id,
                name=voice.name,
                language=voice.language or "",
                gender=voice.gender or "",
                description=voice.description or "",
                is_cloned=True,
            ),
            created_at=voice.created_at,
        )

    async def DeleteVoice(self, request, context):
        if not request.id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Voice ID is required")

        deleted = delete_stored_voice(self._store, request.id)
        if not deleted:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Voice '{request.id}' not found")

        return vox_pb2.DeleteVoiceResponse(id=request.id, deleted=True)


def _voices_for_adapter(adapter: TTSAdapter, store: BlobStore):
    voices = list(adapter.list_voices())
    if adapter.info().supports_voice_cloning:
        voices.extend(voice.to_voice_info() for voice in list_stored_voices(store))
    return voices


def _resolve_voice_request(
    adapter: TTSAdapter,
    store: BlobStore,
    voice_id: str | None,
    language: str | None,
):
    return resolve_voice_request(adapter, store, voice_id, language)
