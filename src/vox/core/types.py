from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class ModelRef:
    """A parsed model reference (name:tag)."""
    name: str
    tag: str = "latest"

    @classmethod
    def parse(cls, s: str) -> ModelRef:
        if ":" in s:
            n, t = s.split(":", 1)
            return cls(n, t)
        return cls(s)

    def __str__(self) -> str:
        return f"{self.name}:{self.tag}"


def parse_model_name(name: str) -> tuple[str, str]:
    """Parse 'name:tag' into (name, tag). Default tag is 'latest'."""
    ref = ModelRef.parse(name)
    return ref.name, ref.tag


class ModelType(str, Enum):
    STT = "stt"
    TTS = "tts"

class ModelFormat(str, Enum):
    ONNX = "onnx"
    CT2 = "ct2"
    PYTORCH = "pytorch"
    GGUF = "gguf"

@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start_ms: int
    end_ms: int
    confidence: float | None = None

@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start_ms: int
    end_ms: int
    words: tuple[WordTimestamp, ...] = ()
    language: str | None = None
    confidence: float | None = None

@dataclass(frozen=True)
class TranscribeResult:
    text: str
    segments: tuple[TranscriptSegment, ...] = ()
    language: str | None = None
    duration_ms: int = 0
    model: str = ""

@dataclass(frozen=True)
class SynthesizeChunk:
    """A chunk of synthesized audio."""
    audio: bytes  # raw float32 PCM bytes
    sample_rate: int
    is_final: bool = False

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

@dataclass(frozen=True)
class VoiceInfo:
    id: str
    name: str
    language: str | None = None
    gender: str | None = None
    description: str | None = None
    is_cloned: bool = False

@dataclass(frozen=True)
class AdapterInfo:
    """Metadata an adapter provides about itself."""
    name: str                                # e.g. "whisper"
    type: ModelType
    architectures: tuple[str, ...]           # e.g. ("whisper", "distil-whisper")
    default_sample_rate: int                 # e.g. 16000 for STT, 24000 for TTS
    supported_formats: tuple[ModelFormat, ...]
    supports_streaming: bool = False
    supports_word_timestamps: bool = False
    supports_language_detection: bool = False
    supports_voice_cloning: bool = False
    supported_languages: tuple[str, ...] = ()

@dataclass(frozen=True)
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

    def __post_init__(self):
        if not self.name:
            raise ValueError("ModelInfo.name must be non-empty")
        if not self.tag:
            raise ValueError("ModelInfo.tag must be non-empty")
        if not self.adapter:
            raise ValueError("ModelInfo.adapter must be non-empty")

    @property
    def full_name(self) -> str:
        return f"{self.name}:{self.tag}"

    @classmethod
    def from_manifest_config(cls, name: str, tag: str, config: dict[str, Any], size_bytes: int = 0) -> ModelInfo:
        for key in ("type", "format", "adapter"):
            if key not in config:
                raise ValueError(f"Manifest config missing required key: {key!r}")
        return cls(
            name=name,
            tag=tag,
            type=ModelType(config["type"]),
            format=ModelFormat(config["format"]),
            architecture=config.get("architecture", ""),
            adapter=config["adapter"],
            size_bytes=size_bytes,
            description=config.get("description", ""),
            license=config.get("license", ""),
            parameters=config.get("parameters", {}),
        )

@dataclass(frozen=True)
class Speechfile:
    """Parsed Speechfile — declarative model spec (like Ollama's Modelfile)."""
    source: str                          # FROM line (HF repo, URL, or local path)
    architecture: str                    # whisper, parakeet, kokoro, piper, etc.
    type: ModelType
    adapter: str                         # entry point name to resolve
    format: ModelFormat
    parameters: dict[str, Any] = field(default_factory=dict)
    voices: tuple[VoiceInfo, ...] = ()
    license: str = ""
    description: str = ""
    files: tuple[str, ...] = ()          # specific files to download from source

@dataclass(frozen=True)
class PullProgress:
    """Progress update during model download."""
    status: str
    digest: str | None = None
    total_bytes: int = 0
    completed_bytes: int = 0

@dataclass(frozen=True)
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
