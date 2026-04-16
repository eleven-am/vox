"""Built-in model catalog, adapter discovery, and model registry for Vox."""

from __future__ import annotations

import importlib
import logging
import os
import platform
import subprocess
import sys
from dataclasses import replace
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from vox.core.errors import AdapterNotFoundError, ModelLoadError, ModelNotFoundError
from vox.core.store import BlobStore
from vox.core.types import ModelInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A. Built-in model catalog
# ---------------------------------------------------------------------------

CATALOG: dict[str, dict[str, dict[str, Any]]] = {
    "parakeet-stt-onnx": {
        "tdt-0.6b": {
            "source": "nvidia/parakeet-tdt-0.6b-v2",
            "architecture": "parakeet",
            "type": "stt",
            "adapter": "parakeet-stt-onnx",
            "format": "onnx",
            "description": "NVIDIA Parakeet TDT 0.6B — top Open ASR Leaderboard model",
            "license": "CC-BY-4.0",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-parakeet",
        },
        "tdt-0.6b-v3": {
            "source": "nvidia/parakeet-tdt-0.6b-v3",
            "architecture": "parakeet",
            "type": "stt",
            "adapter": "parakeet-stt-onnx",
            "format": "onnx",
            "description": "NVIDIA Parakeet TDT 0.6B v3 — 25 languages, streaming support",
            "license": "CC-BY-4.0",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-parakeet",
        },
    },
    "parakeet-stt-nemo": {
        "tdt-0.6b-v3": {
            "source": "nvidia/parakeet-tdt-0.6b-v3",
            "architecture": "parakeet-nemo",
            "type": "stt",
            "adapter": "parakeet-stt-nemo",
            "format": "pytorch",
            "description": "NVIDIA Parakeet TDT 0.6B v3 — native NeMo/PyTorch CUDA ASR",
            "license": "CC-BY-4.0",
            "parameters": {"sample_rate": 16000},
            "files": ["parakeet-tdt-0.6b-v3.nemo"],
            "adapter_package": "vox-parakeet",
        },
        "tdt-1.1b": {
            "source": "nvidia/parakeet-tdt-1.1b",
            "architecture": "parakeet-nemo",
            "type": "stt",
            "adapter": "parakeet-stt-nemo",
            "format": "pytorch",
            "description": "NVIDIA Parakeet TDT 1.1B — larger native NeMo/PyTorch CUDA ASR",
            "license": "CC-BY-4.0",
            "parameters": {"sample_rate": 16000},
            "files": ["parakeet-tdt-1.1b.nemo"],
            "adapter_package": "vox-parakeet",
        },
    },
    "kokoro-tts-onnx": {
        "v1.0": {
            "source": "onnx-community/Kokoro-82M-v1.0-ONNX",
            "architecture": "kokoro",
            "type": "tts",
            "adapter": "kokoro-tts-onnx",
            "format": "onnx",
            "description": "Kokoro 82M ONNX — fast, lightweight TTS with preset voices",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 24000, "default_voice": "af_heart"},
            "adapter_package": "vox-kokoro",
        },
    },
    "kokoro-tts-torch": {
        "v1.0": {
            "source": "hexgrad/Kokoro-82M",
            "architecture": "kokoro-torch",
            "type": "tts",
            "adapter": "kokoro-tts-torch",
            "format": "pytorch",
            "description": "Kokoro 82M native runtime — PyTorch backend for Spark/CUDA systems",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 24000, "default_voice": "af_heart"},
            "files": ["kokoro-v1_0.pth"],
            "adapter_package": "vox-kokoro",
        },
    },
    "xtts-tts-torch": {
        "v2": {
            "source": "coqui/XTTS-v2",
            "architecture": "xtts-v2",
            "type": "tts",
            "adapter": "xtts-tts-torch",
            "format": "pytorch",
            "description": "Coqui XTTS-v2 — multilingual voice cloning TTS",
            "parameters": {"sample_rate": 24000},
            "adapter_package": "vox-xtts",
        },
    },
    "whisper-stt-ct2": {
        "large-v3": {
            "source": "Systran/faster-whisper-large-v3",
            "architecture": "whisper",
            "type": "stt",
            "adapter": "whisper-stt-ct2",
            "format": "ct2",
            "description": "OpenAI Whisper Large V3 via CTranslate2",
            "license": "MIT",
            "parameters": {"sample_rate": 16000, "beam_size": 5},
            "adapter_package": "vox-whisper",
        },
        "large-v3-turbo": {
            "source": "deepdml/faster-whisper-large-v3-turbo-ct2",
            "architecture": "whisper",
            "type": "stt",
            "adapter": "whisper-stt-ct2",
            "format": "ct2",
            "description": "OpenAI Whisper Large V3 Turbo — faster with minor quality loss",
            "license": "MIT",
            "parameters": {"sample_rate": 16000, "beam_size": 5},
            "adapter_package": "vox-whisper",
        },
        "base.en": {
            "source": "Systran/faster-whisper-base.en",
            "architecture": "whisper",
            "type": "stt",
            "adapter": "whisper-stt-ct2",
            "format": "ct2",
            "description": "OpenAI Whisper Base English — small and fast",
            "license": "MIT",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-whisper",
        },
        "small.en": {
            "source": "Systran/faster-whisper-small.en",
            "architecture": "whisper",
            "type": "stt",
            "adapter": "whisper-stt-ct2",
            "format": "ct2",
            "description": "OpenAI Whisper Small English — stronger English transcription than base.en",
            "license": "MIT",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-whisper",
        },
        "medium.en": {
            "source": "Systran/faster-whisper-medium.en",
            "architecture": "whisper",
            "type": "stt",
            "adapter": "whisper-stt-ct2",
            "format": "ct2",
            "description": "OpenAI Whisper Medium English — larger English CTranslate2 checkpoint",
            "license": "MIT",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-whisper",
        },
    },
    "piper-tts-onnx": {
        "en-us-lessac-medium": {
            "source": "rhasspy/piper-voices",
            "architecture": "piper",
            "type": "tts",
            "adapter": "piper-tts-onnx",
            "format": "onnx",
            "description": "Piper English US Lessac Medium — single-voice ONNX TTS",
            "license": "MIT",
            "parameters": {"sample_rate": 22050},
            "files": [
                "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
                "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
            ],
            "adapter_package": "vox-piper",
        },
    },
    "dia-tts-torch": {
        "1.6b": {
            "source": "nari-labs/Dia-1.6B",
            "architecture": "dia",
            "type": "tts",
            "adapter": "dia-tts-torch",
            "format": "pytorch",
            "description": "Dia 1.6B — multi-speaker dialogue TTS with non-verbal sounds",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 44000},
            "adapter_package": "vox-dia",
        },
    },
    "sesame-tts-torch": {
        "csm-1b": {
            "source": "sesame/csm-1b",
            "architecture": "sesame",
            "type": "tts",
            "adapter": "sesame-tts-torch",
            "format": "pytorch",
            "description": "Sesame CSM 1B — context-aware conversational speech model",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 24000, "default_voice": "0"},
            "adapter_package": "vox-sesame",
        },
    },
    "openvoice-tts-torch": {
        "v1": {
            "source": "myshell-ai/OpenVoice",
            "architecture": "openvoice",
            "type": "tts",
            "adapter": "openvoice-tts-torch",
            "format": "pytorch",
            "description": "MyShell OpenVoice V1 — instant voice cloning and style control",
            "license": "MIT",
            "parameters": {"sample_rate": 22050, "default_voice": "en/default"},
            "files": [
                "checkpoints/base_speakers/EN/checkpoint.pth",
                "checkpoints/base_speakers/EN/config.json",
                "checkpoints/base_speakers/EN/en_default_se.pth",
                "checkpoints/base_speakers/EN/en_style_se.pth",
                "checkpoints/base_speakers/ZH/checkpoint.pth",
                "checkpoints/base_speakers/ZH/config.json",
                "checkpoints/base_speakers/ZH/zh_default_se.pth",
                "checkpoints/converter/checkpoint.pth",
                "checkpoints/converter/config.json",
            ],
            "adapter_package": "vox-openvoice",
        },
    },
    "voxtral-stt-torch": {
        "mini-3b": {
            "source": "mistralai/Voxtral-Mini-3B-2507",
            "architecture": "voxtral-mini",
            "type": "stt",
            "adapter": "voxtral-stt-torch",
            "format": "pytorch",
            "description": "Mistral Voxtral Mini 3B — audio understanding and transcription",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-voxtral",
        },
        "small-24b": {
            "source": "mistralai/Voxtral-Small-24B-2507",
            "architecture": "voxtral-small",
            "type": "stt",
            "adapter": "voxtral-stt-torch",
            "format": "pytorch",
            "description": "Mistral Voxtral Small 24B — high-quality audio understanding and transcription",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-voxtral",
        },
        "24b": {
            "source": "mistralai/Voxtral-Small-24B-2507",
            "architecture": "voxtral-small",
            "type": "stt",
            "adapter": "voxtral-stt-torch",
            "format": "pytorch",
            "description": "Mistral Voxtral 24B — size alias for the larger Small 24B transcription model",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-voxtral",
        },
        "realtime-4b": {
            "source": "mistralai/Voxtral-Mini-4B-Realtime-2602",
            "architecture": "voxtral-realtime",
            "type": "stt",
            "adapter": "voxtral-stt-torch",
            "format": "pytorch",
            "description": "Mistral Voxtral Realtime 4B — sub-500ms latency transcription, 13 languages",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-voxtral",
        },
    },
    "voxtral-tts-vllm": {
        "4b": {
            "source": "mistralai/Voxtral-4B-TTS-2603",
            "architecture": "voxtral-tts",
            "type": "tts",
            "adapter": "voxtral-tts-vllm",
            "format": "pytorch",
            "description": "Mistral Voxtral TTS 4B — preset-voice TTS, 9 languages, ~90ms TTFA",
            "license": "CC-BY-NC-4.0",
            "parameters": {"sample_rate": 24000, "default_voice": "neutral_female"},
            "adapter_package": "vox-voxtral",
        },
    },
    "speecht5-tts-torch": {
        "base": {
            "source": "microsoft/speecht5_tts",
            "architecture": "speecht5-tts",
            "type": "tts",
            "adapter": "speecht5-tts-torch",
            "format": "pytorch",
            "description": "Microsoft SpeechT5 TTS — lightweight text-to-speech with speaker embeddings",
            "license": "MIT",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-microsoft",
        },
    },
    "speecht5-stt-torch": {
        "base": {
            "source": "microsoft/speecht5_asr",
            "architecture": "speecht5-asr",
            "type": "stt",
            "adapter": "speecht5-stt-torch",
            "format": "pytorch",
            "description": "Microsoft SpeechT5 ASR — lightweight speech recognition",
            "license": "MIT",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-microsoft",
        },
    },
    "vibevoice-tts-torch": {
        "realtime-0.5b": {
            "source": "microsoft/VibeVoice-Realtime-0.5B",
            "architecture": "vibevoice-realtime",
            "type": "tts",
            "adapter": "vibevoice-tts-torch",
            "format": "pytorch",
            "description": "Microsoft VibeVoice Realtime 0.5B — ~300ms streaming TTS",
            "license": "MIT",
            "parameters": {"sample_rate": 24000},
            "adapter_package": "vox-microsoft",
        },
        "1.5b": {
            "source": "microsoft/VibeVoice-1.5B",
            "architecture": "vibevoice",
            "type": "tts",
            "adapter": "vibevoice-tts-torch",
            "format": "pytorch",
            "description": "Microsoft VibeVoice 1.5B — high-quality TTS with 64K token context",
            "license": "MIT",
            "parameters": {"sample_rate": 24000},
            "adapter_package": "vox-microsoft",
        },
    },
    "qwen3-stt-torch": {
        "1.7b": {
            "source": "Qwen/Qwen3-ASR-1.7B",
            "architecture": "qwen3-asr",
            "type": "stt",
            "adapter": "qwen3-stt-torch",
            "format": "pytorch",
            "description": "Alibaba Qwen3-ASR 1.7B — 52 languages, word timestamps",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-qwen",
        },
        "0.6b": {
            "source": "Qwen/Qwen3-ASR-0.6B",
            "architecture": "qwen3-asr",
            "type": "stt",
            "adapter": "qwen3-stt-torch",
            "format": "pytorch",
            "description": "Alibaba Qwen3-ASR 0.6B — lightweight 52-language speech recognition",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-qwen",
        },
    },
    "qwen3-tts-torch": {
        "1.7b": {
            "source": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "architecture": "qwen3-tts",
            "type": "tts",
            "adapter": "qwen3-tts-torch",
            "format": "pytorch",
            "description": "Alibaba Qwen3-TTS 1.7B — multilingual speaker-based CustomVoice TTS with streaming",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 24000, "default_voice": "Ryan"},
            "adapter_package": "vox-qwen",
        },
        "0.6b": {
            "source": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "architecture": "qwen3-tts",
            "type": "tts",
            "adapter": "qwen3-tts-torch",
            "format": "pytorch",
            "description": "Alibaba Qwen3-TTS 0.6B — lightweight multilingual speaker-based CustomVoice TTS",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 24000, "default_voice": "Ryan"},
            "adapter_package": "vox-qwen",
        },
    },
}


# ---------------------------------------------------------------------------
# B. Adapter discovery via entry points
# ---------------------------------------------------------------------------

ADAPTERS_DIR = "adapters"  # relative to vox home
REGISTRY_BASE_URL = "https://raw.githubusercontent.com/eleven-am/vox-registry/main"
BUNDLED_ADAPTERS_ENV = "VOX_BUNDLED_ADAPTERS"
BUNDLED_ADAPTERS_NO_DEPS_ENV = "VOX_BUNDLED_ADAPTERS_NO_DEPS"
DISABLE_BUNDLED_ADAPTERS_ENV = "VOX_DISABLE_BUNDLED_ADAPTERS"
ADAPTERS_NO_DEPS_ENV = "VOX_ADAPTERS_NO_DEPS"
IMPLICIT_MODEL_ALIASES: dict[str, dict[str, tuple[str, str]]] = {
    "parakeet": {
        "spark": ("parakeet-stt-nemo", "tdt-0.6b-v3"),
        "default": ("parakeet-stt-onnx", "tdt-0.6b-v3"),
    },
    "parakeet-stt": {
        "spark": ("parakeet-stt-nemo", "tdt-0.6b-v3"),
        "default": ("parakeet-stt-onnx", "tdt-0.6b-v3"),
    },
    "kokoro": {
        "spark": ("kokoro-tts-torch", "v1.0"),
        "default": ("kokoro-tts-onnx", "v1.0"),
    },
    "kokoro-tts": {
        "spark": ("kokoro-tts-torch", "v1.0"),
        "default": ("kokoro-tts-onnx", "v1.0"),
    },
    "whisper": {
        "spark": ("whisper-stt-ct2", "base.en"),
        "default": ("whisper-stt-ct2", "base.en"),
    },
    "whisper-stt": {
        "spark": ("whisper-stt-ct2", "base.en"),
        "default": ("whisper-stt-ct2", "base.en"),
    },
    "piper": {
        "spark": ("piper-tts-onnx", "en-us-lessac-medium"),
        "default": ("piper-tts-onnx", "en-us-lessac-medium"),
    },
    "piper-tts": {
        "spark": ("piper-tts-onnx", "en-us-lessac-medium"),
        "default": ("piper-tts-onnx", "en-us-lessac-medium"),
    },
    "openvoice": {
        "spark": ("openvoice-tts-torch", "v1"),
        "default": ("openvoice-tts-torch", "v1"),
    },
    "openvoice-tts": {
        "spark": ("openvoice-tts-torch", "v1"),
        "default": ("openvoice-tts-torch", "v1"),
    },
    "dia": {
        "spark": ("dia-tts-torch", "1.6b"),
        "default": ("dia-tts-torch", "1.6b"),
    },
    "dia-tts": {
        "spark": ("dia-tts-torch", "1.6b"),
        "default": ("dia-tts-torch", "1.6b"),
    },
    "sesame": {
        "spark": ("sesame-tts-torch", "csm-1b"),
        "default": ("sesame-tts-torch", "csm-1b"),
    },
    "sesame-tts": {
        "spark": ("sesame-tts-torch", "csm-1b"),
        "default": ("sesame-tts-torch", "csm-1b"),
    },
    "speecht5-stt": {
        "spark": ("speecht5-stt-torch", "base"),
        "default": ("speecht5-stt-torch", "base"),
    },
    "speecht5-tts": {
        "spark": ("speecht5-tts-torch", "base"),
        "default": ("speecht5-tts-torch", "base"),
    },
    "vibevoice": {
        "spark": ("vibevoice-tts-torch", "realtime-0.5b"),
        "default": ("vibevoice-tts-torch", "realtime-0.5b"),
    },
    "vibevoice-tts": {
        "spark": ("vibevoice-tts-torch", "realtime-0.5b"),
        "default": ("vibevoice-tts-torch", "realtime-0.5b"),
    },
    "qwen3-stt": {
        "spark": ("qwen3-stt-torch", "0.6b"),
        "default": ("qwen3-stt-torch", "0.6b"),
    },
    "qwen3-tts": {
        "spark": ("qwen3-tts-torch", "0.6b"),
        "default": ("qwen3-tts-torch", "0.6b"),
    },
    "xtts": {
        "spark": ("xtts-tts-torch", "v2"),
        "default": ("xtts-tts-torch", "v2"),
    },
    "xtts-tts": {
        "spark": ("xtts-tts-torch", "v2"),
        "default": ("xtts-tts-torch", "v2"),
    },
    "voxtral-stt": {
        "spark": ("voxtral-stt-torch", "mini-3b"),
        "default": ("voxtral-stt-torch", "mini-3b"),
    },
    "voxtral-tts": {
        "spark": ("voxtral-tts-vllm", "4b"),
        "default": ("voxtral-tts-vllm", "4b"),
    },
}

LEGACY_MODEL_REF_ALIASES: dict[tuple[str, str], tuple[str, str]] = {
    ("kokoro", "v1.0"): ("kokoro-tts-onnx", "v1.0"),
    ("kokoro", "v1.0-torch"): ("kokoro-tts-torch", "v1.0"),
    ("parakeet", "tdt-0.6b"): ("parakeet-stt-onnx", "tdt-0.6b"),
    ("parakeet", "tdt-0.6b-v3"): ("parakeet-stt-onnx", "tdt-0.6b-v3"),
    ("parakeet", "tdt-0.6b-v3-nemo"): ("parakeet-stt-nemo", "tdt-0.6b-v3"),
    ("parakeet", "tdt-0.6b-v3-cuda"): ("parakeet-stt-nemo", "tdt-0.6b-v3"),
    ("parakeet", "tdt-1.1b-nemo"): ("parakeet-stt-nemo", "tdt-1.1b"),
    ("parakeet", "tdt-1.1b-cuda"): ("parakeet-stt-nemo", "tdt-1.1b"),
    ("speecht5", "asr"): ("speecht5-stt-torch", "base"),
    ("speecht5", "tts"): ("speecht5-tts-torch", "base"),
    ("voxtral", "mini-3b"): ("voxtral-stt-torch", "mini-3b"),
    ("voxtral", "small-24b"): ("voxtral-stt-torch", "small-24b"),
    ("voxtral", "24b"): ("voxtral-stt-torch", "24b"),
    ("voxtral", "realtime-4b"): ("voxtral-stt-torch", "realtime-4b"),
    ("voxtral", "tts-4b"): ("voxtral-tts-vllm", "4b"),
}

LEGACY_NAME_ALIASES: dict[str, str] = {
    "dia": "dia-tts-torch",
    "kokoro-torch": "kokoro-tts-torch",
    "openvoice": "openvoice-tts-torch",
    "parakeet-nemo": "parakeet-stt-nemo",
    "piper": "piper-tts-onnx",
    "qwen3-asr": "qwen3-stt-torch",
    "qwen3-tts": "qwen3-tts-torch",
    "sesame": "sesame-tts-torch",
    "speecht5-stt": "speecht5-stt-torch",
    "speecht5-tts": "speecht5-tts-torch",
    "vibevoice": "vibevoice-tts-torch",
    "vibevoice-tts": "vibevoice-tts-torch",
    "voxtral-stt": "voxtral-stt-torch",
    "voxtral-tts": "voxtral-tts-vllm",
    "whisper": "whisper-stt-ct2",
    "xtts": "xtts-tts-torch",
}


def _runtime_profile() -> str:
    """Return a coarse runtime profile used for family alias selection."""
    device = os.environ.get("VOX_DEVICE", "auto").strip().lower()
    machine = platform.machine().strip().lower()
    if device == "cuda" and machine in {"arm64", "aarch64"}:
        return "spark"
    return "default"


def resolve_family_alias(name: str, tag: str = "latest", *, explicit_tag: bool = False) -> tuple[str, str]:
    """Resolve model references to canonical names.

    Canonical names follow ``<family>-<task>-<backend>:<variant>``.
    Legacy names and bare family shorthands are mapped to canonical entries.
    """
    if not explicit_tag and tag == "latest":
        aliases = IMPLICIT_MODEL_ALIASES.get(name)
        if aliases is not None:
            profile = _runtime_profile()
            return aliases.get(profile) or aliases["default"]

    exact_alias = LEGACY_MODEL_REF_ALIASES.get((name, tag))
    if exact_alias is not None:
        return exact_alias

    resolved_name = LEGACY_NAME_ALIASES.get(name, name)
    return resolved_name, tag


def fetch_from_registry(name: str, tag: str) -> dict[str, Any] | None:
    """Fetch model metadata from the remote GitHub registry."""
    import httpx

    url = f"{REGISTRY_BASE_URL}/library/{name}/{tag}.json"
    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        if resp.status_code == 200:
            return resp.json()
        return None
    except (httpx.HTTPError, ValueError) as e:
        logger.warning(f"Failed to fetch from registry: {url}: {e}")
        return None


def fetch_registry_index() -> list[dict[str, Any]] | None:
    """Fetch the full model index from the remote registry."""
    import httpx

    url = f"{REGISTRY_BASE_URL}/index.json"
    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        if resp.status_code == 200:
            return resp.json()
        return None
    except (httpx.HTTPError, ValueError):
        return None


def _find_bundled_adapter_source(package_name: str) -> Path | None:
    """Return a local adapter source tree when one is bundled with the app."""
    if os.environ.get(DISABLE_BUNDLED_ADAPTERS_ENV, "").lower() in {"1", "true", "yes", "on"}:
        return None

    candidates: list[Path] = []

    bundled_root = os.environ.get(BUNDLED_ADAPTERS_ENV)
    if bundled_root:
        candidates.append(Path(bundled_root))

    candidates.append(Path(__file__).resolve().parents[3] / "adapters")

    for base_dir in candidates:
        candidate = base_dir / package_name
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _adapter_install_dir(vox_home: Path, package_name: str) -> Path:
    """Return the isolated install directory for a single adapter package."""
    return vox_home / ADAPTERS_DIR / package_name


def install_adapter_package(package_name: str, vox_home: Path) -> bool:
    """Install an adapter package into the vox adapters directory.

    Returns True if installation succeeded, False otherwise.
    """
    target_dir = _adapter_install_dir(vox_home, package_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    package_spec = package_name

    bundled_source = _find_bundled_adapter_source(package_name)
    install_no_deps = os.environ.get(ADAPTERS_NO_DEPS_ENV, "").lower() in {"1", "true", "yes", "on"}
    if bundled_source is not None:
        package_spec = str(bundled_source)
        install_no_deps = install_no_deps or os.environ.get(BUNDLED_ADAPTERS_NO_DEPS_ENV, "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        logger.info("Installing bundled adapter package from %s", bundled_source)

    installers = [
        ["uv", "pip", "install", "--python", sys.executable],
        [sys.executable, "-m", "pip", "install"],
    ]
    for installer in installers:
        try:
            cmd = [*installer, "--target", str(target_dir), "--upgrade"]
            if installer[:2] == ["uv", "pip"]:
                cmd.extend(["--refresh-package", package_name])
            if install_no_deps:
                cmd.append("--no-deps")
            cmd.append(package_spec)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                logger.info("Installed adapter package: %s", package_name)
                return True
            logger.warning("%s failed: %s", " ".join(installer), result.stderr)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    logger.error("Failed to install adapter package: %s", package_name)
    return False


def _ensure_adapters_on_path(vox_home: Path) -> None:
    """Add the vox adapters directory to sys.path if not already present."""
    adapters_root = vox_home / ADAPTERS_DIR
    adapters_root.mkdir(parents=True, exist_ok=True)

    candidate_paths = [adapters_root]
    candidate_paths.extend(
        sorted(path for path in adapters_root.iterdir() if path.is_dir())
    )

    for path in candidate_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    importlib.invalidate_caches()


def discover_adapters() -> dict[str, type]:
    """Discover installed adapter classes via 'vox.adapters' entry point group."""
    adapters: dict[str, type] = {}
    for ep in entry_points(group="vox.adapters"):
        try:
            adapters[ep.name] = ep.load()
        except Exception as e:
            logger.warning(f"Skipping broken adapter plugin '{ep.name}': {e}")
    return adapters


# ---------------------------------------------------------------------------
# C. Registry class
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Ties the built-in catalog, blob store, and adapter discovery together."""

    def __init__(self, store: BlobStore) -> None:
        self._store = store
        _ensure_adapters_on_path(self._store.root)
        self._adapters = discover_adapters()

    # -- catalog helpers -----------------------------------------------------

    def resolve_model_ref(self, name: str, tag: str = "latest", *, explicit_tag: bool = False) -> tuple[str, str]:
        """Resolve a possibly-bare model reference to a concrete catalog tag."""
        return resolve_family_alias(name, tag, explicit_tag=explicit_tag)

    def lookup(self, name: str, tag: str = "latest", *, explicit_tag: bool = False) -> dict | None:
        """Look up a model — local catalog first, then remote registry."""
        name, tag = self.resolve_model_ref(name, tag, explicit_tag=explicit_tag)
        tags = CATALOG.get(name)
        if tags is not None:
            entry = tags.get(tag)
            if entry is not None:
                return entry

        # Try remote registry
        entry = fetch_from_registry(name, tag)
        if entry is not None:
            # Cache locally for this session
            if name not in CATALOG:
                CATALOG[name] = {}
            CATALOG[name][tag] = entry
            logger.info(f"Fetched {name}:{tag} from remote registry")
            return entry

        return None

    def available_models(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return local catalog merged with remote index if available."""
        remote = fetch_registry_index()
        if remote:
            for entry in remote:
                name, tag = entry["name"], entry["tag"]
                if name not in CATALOG:
                    CATALOG[name] = {}
                if tag not in CATALOG[name]:
                    CATALOG[name][tag] = entry
        return CATALOG

    # -- adapter helpers -----------------------------------------------------

    def ensure_adapter(self, adapter_name: str, package_name: str) -> bool:
        """Ensure an adapter is installed. Auto-installs if needed."""
        if adapter_name in self._adapters:
            return True

        logger.info(f"Adapter '{adapter_name}' not found, installing {package_name}...")
        if not install_adapter_package(package_name, self._store.root):
            return False

        # Re-discover after install
        _ensure_adapters_on_path(self._store.root)
        self._adapters = discover_adapters()
        return adapter_name in self._adapters

    def get_adapter_class(self, adapter_name: str) -> type:
        """Return the adapter class for *adapter_name*.

        Raises :class:`AdapterNotFoundError` if no matching adapter is installed.
        """
        cls = self._adapters.get(adapter_name)
        if cls is None:
            raise AdapterNotFoundError(adapter_name)
        return cls

    # -- resolution ----------------------------------------------------------

    def resolve(self, name: str, tag: str = "latest", *, explicit_tag: bool = False) -> tuple[ModelInfo, Path]:
        """Resolve a model to its :class:`ModelInfo` and a model directory path.

        The model must already be pulled (i.e. a manifest must exist in the
        store).  Raises :class:`ModelNotFoundError` otherwise.

        The returned path is a directory where blobs are symlinked to their
        original filenames, so adapters can load files by name (e.g.
        ``model.onnx``, ``voices.bin``).
        """
        name, tag = self.resolve_model_ref(name, tag, explicit_tag=explicit_tag)
        manifest = self._store.resolve_model(name, tag)
        if manifest is None:
            raise ModelNotFoundError(f"{name}:{tag}")

        cfg = manifest.config
        size = sum(layer.size for layer in manifest.layers)

        info = ModelInfo.from_manifest_config(name, tag, cfg, size_bytes=size)

        # Inject the catalog source (e.g. HuggingFace repo ID) so adapters
        # that need a model ID rather than a local path can retrieve it.
        source = cfg.get("source")
        if source:
            updated_params = {**info.parameters, "_source": source}
            info = replace(info, parameters=updated_params)

        if not manifest.layers:
            raise ModelNotFoundError(f"{name}:{tag} (manifest has no layers)")

        # Build a model directory with symlinks to blobs using original filenames.
        model_dir = self._store.root / "models" / "links" / name / tag
        model_dir.mkdir(parents=True, exist_ok=True)

        for layer in manifest.layers:
            link_path = model_dir / layer.filename
            blob_path = self._store.get_blob_path(layer.digest)
            link_path.parent.mkdir(parents=True, exist_ok=True)
            # Remove stale/broken symlinks before recreating
            if link_path.is_symlink() and not link_path.exists():
                link_path.unlink()
            if not link_path.exists():
                try:
                    link_path.symlink_to(blob_path)
                except OSError as e:
                    raise ModelLoadError(
                        f"Failed to create symlink {link_path} -> {blob_path}: {e}"
                    ) from e

        return info, model_dir
