from __future__ import annotations

from typing import Any

from vox.grpc import vox_pb2
from vox.operations.voices import (
    DeleteVoiceResult,
    ListedVoice,
    created_voice_payload,
    deleted_voice_payload,
    list_voices_payload,
)


def list_voices_response(listed: list[ListedVoice]) -> vox_pb2.ListVoicesResponse:
    payload = list_voices_payload(listed, include_model=True)
    return vox_pb2.ListVoicesResponse(
        voices=[_voice_info_message(voice) for voice in payload["voices"]]
    )


def create_voice_response(voice: Any) -> vox_pb2.CreateVoiceResponse:
    payload = created_voice_payload(voice)
    return vox_pb2.CreateVoiceResponse(
        voice=_voice_info_message(payload),
        created_at=payload["created_at"],
    )


def delete_voice_response(result: DeleteVoiceResult) -> vox_pb2.DeleteVoiceResponse:
    payload = deleted_voice_payload(result)
    return vox_pb2.DeleteVoiceResponse(**payload)


def _voice_info_message(payload: dict[str, Any]) -> vox_pb2.VoiceInfo:
    return vox_pb2.VoiceInfo(
        id=_string(payload.get("id")),
        name=_string(payload.get("name")),
        language=_string(payload.get("language")),
        gender=_string(payload.get("gender")),
        description=_string(payload.get("description")),
        is_cloned=bool(payload.get("is_cloned")),
        model=_string(payload.get("model")),
    )


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""
