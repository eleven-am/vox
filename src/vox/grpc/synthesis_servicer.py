from __future__ import annotations

import logging

import grpc

from vox.core.errors import (
    ModelNotFoundError,
    VoiceCloningUnsupportedError,
    VoiceNotFoundError,
    VoxError,
)
from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.operations.errors import (
    EmptyInputError,
    NoAudioGeneratedError,
    NoDefaultModelError,
    OperationError,
    VoiceAudioRequiredError,
    VoiceIdRequiredError,
    VoiceNameRequiredError,
    VoiceNotFoundOperationError,
    WrongModelTypeError,
)
from vox.operations.synthesis import SynthesisRequest, synthesize_raw
from vox.operations.voices import (
    CreateVoiceRequest,
    create_voice,
    delete_voice,
    list_voices,
)

logger = logging.getLogger(__name__)


def _operation_error_status(exc: OperationError) -> tuple[grpc.StatusCode, str]:
    if isinstance(exc, (
        NoDefaultModelError,
        EmptyInputError,
        WrongModelTypeError,
        VoiceNameRequiredError,
        VoiceAudioRequiredError,
        VoiceIdRequiredError,
    )):
        return grpc.StatusCode.INVALID_ARGUMENT, str(exc)
    if isinstance(exc, VoiceNotFoundOperationError):
        return grpc.StatusCode.NOT_FOUND, str(exc)
    if isinstance(exc, NoAudioGeneratedError):
        return grpc.StatusCode.INTERNAL, str(exc)
    return grpc.StatusCode.INTERNAL, str(exc)


class SynthesisServicer(vox_pb2_grpc.SynthesisServiceServicer):

    def __init__(self, store: BlobStore, registry: ModelRegistry, scheduler: Scheduler) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler

    async def Synthesize(self, request, context):
        op_req = SynthesisRequest(
            input=request.input,
            model=request.model,
            voice=request.voice or None,
            speed=request.speed if request.speed > 0 else 1.0,
            language=request.language or None,
            response_format="wav",
        )
        try:
            iterator = await synthesize_raw(
                scheduler=self._scheduler,
                registry=self._registry,
                store=self._store,
                request=op_req,
            )
            async for chunk in iterator:
                yield vox_pb2.AudioChunk(
                    audio=chunk.audio,
                    sample_rate=chunk.sample_rate,
                    is_final=chunk.is_final,
                )
        except OperationError as exc:
            code, msg = _operation_error_status(exc)
            await context.abort(code, msg)
            return
        except (VoiceCloningUnsupportedError, VoiceNotFoundError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return
        except ModelNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
            return
        except VoxError as exc:
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return
        except Exception:
            logger.exception("Synthesis failed")
            await context.abort(grpc.StatusCode.INTERNAL, "Internal synthesis error")

    async def ListVoices(self, request, context):
        try:
            listed = await list_voices(
                scheduler=self._scheduler,
                store=self._store,
                model=request.model or None,
            )
        except OperationError as exc:
            code, msg = _operation_error_status(exc)
            await context.abort(code, msg)
            return
        except ModelNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
            return
        except VoxError as exc:
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return

        voices = []
        for entry in listed:
            v = entry.voice
            voices.append(vox_pb2.VoiceInfo(
                id=v.id,
                name=v.name,
                language=v.language or "",
                gender=v.gender or "",
                description=v.description or "",
                is_cloned=v.is_cloned,
                model=entry.model or "",
            ))
        return vox_pb2.ListVoicesResponse(voices=voices)

    async def CreateVoice(self, request, context):
        op_req = CreateVoiceRequest(
            name=request.name,
            audio=request.audio,
            content_type=request.format_hint or None,
            language=request.language or None,
            gender=request.gender or None,
            reference_text=request.reference_text or None,
        )
        try:
            voice = create_voice(store=self._store, request=op_req)
        except OperationError as exc:
            code, msg = _operation_error_status(exc)
            await context.abort(code, msg)
            return
        except (TypeError, ValueError, RuntimeError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return

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
        try:
            delete_voice(store=self._store, voice_id=request.id)
        except OperationError as exc:
            code, msg = _operation_error_status(exc)
            await context.abort(code, msg)
            return
        return vox_pb2.DeleteVoiceResponse(id=request.id, deleted=True)
