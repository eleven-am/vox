from __future__ import annotations

from vox.grpc import vox_pb2
from vox.operations.synthesis import SynthesisRawChunk


def audio_chunk_message(chunk: SynthesisRawChunk) -> vox_pb2.AudioChunk:
    return vox_pb2.AudioChunk(
        audio=chunk.audio,
        sample_rate=chunk.sample_rate,
        is_final=chunk.is_final,
    )
