from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class ModelType(str, Enum):
    STT = "stt"
    TTS = "tts"

class ModelFormat(str, Enum):
    ONNX = "onnx"
    CT2 = "ct2"
    PYTORCH = "pytorch"
    GGUF = "gguf"

@dataclass
class WordTimestamp:
    word: str
    start_ms: int
    end_ms: int
    confidence: float | None = None

@dataclass
class TranscriptSegment:
    text: str
    start_ms: int
    end_ms: int
    words: list[WordTimestamp] = field(default_factory=list)
    language: str | None = None
    confidence: float | None = None

@dataclass
class TranscribeResult:
    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str | None = None
    duration_ms: int = 0
    model: str = ""

@dataclass
class SynthesizeChunk:
    """A chunk of synthesized audio."""
    audio: bytes  # raw float32 PCM bytes
    sample_rate: int
    is_final: bool = False

@dataclass
class VoiceInfo:
    id: str
    name: str
    language: str | None = None
    gender: str | None = None
    description: str | None = None
    is_cloned: bool = False

@dataclass
class AdapterInfo:
    """Metadata an adapter provides about itself."""
    name: str                                # e.g. "whisper"
    type: ModelType
    architectures: list[str]                 # e.g. ["whisper", "distil-whisper"]
    default_sample_rate: int                 # e.g. 16000 for STT, 24000 for TTS
    supported_formats: list[ModelFormat]
    supports_streaming: bool = False
    supports_word_timestamps: bool = False
    supports_language_detection: bool = False
    supports_voice_cloning: bool = False
    supported_languages: list[str] = field(default_factory=list)

@dataclass
class ModelInfo:
    """Info about a locally stored model."""
    name: str           # e.g. "whisper"
    tag: str            # e.g. "large-v3"
    type: ModelType
    format: ModelFormat
    architecture: str
    adapter: str        # entry point name
    size_bytes: int = 0
    description: str = ""
    license: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.name}:{self.tag}"

@dataclass
class Speechfile:
    """Parsed Speechfile — declarative model spec (like Ollama's Modelfile)."""
    source: str                          # FROM line (HF repo, URL, or local path)
    architecture: str                    # whisper, parakeet, kokoro, piper, etc.
    type: ModelType
    adapter: str                         # entry point name to resolve
    format: ModelFormat
    parameters: dict[str, Any] = field(default_factory=dict)
    voices: list[VoiceInfo] = field(default_factory=list)
    license: str = ""
    description: str = ""
    files: list[str] = field(default_factory=list)  # specific files to download from source

@dataclass
class PullProgress:
    """Progress update during model download."""
    status: str
    digest: str | None = None
    total_bytes: int = 0
    completed_bytes: int = 0

@dataclass
class LoadedModelInfo:
    """Info about a currently loaded model."""
    name: str
    tag: str
    type: ModelType
    device: str
    vram_bytes: int = 0
    loaded_at: float = 0.0
    last_used: float = 0.0
    ref_count: int = 0
