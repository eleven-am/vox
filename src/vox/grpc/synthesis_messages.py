from __future__ import annotations

from typing import Any

from google.protobuf.struct_pb2 import Struct

from vox.grpc import vox_pb2
from vox.operations.synthesis import SynthesisRawChunk


def synthesis_params_from_message(params: Struct) -> dict[str, Any]:
    values = dict(params)
    return {
        name: int(value) if isinstance(value, float) and value.is_integer() else value
        for name, value in values.items()
    }


def audio_chunk_message(chunk: SynthesisRawChunk) -> vox_pb2.AudioChunk:
    return vox_pb2.AudioChunk(
        audio=chunk.audio,
        sample_rate=chunk.sample_rate,
        is_final=chunk.is_final,
    )
