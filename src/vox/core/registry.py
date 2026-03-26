"""Built-in model catalog, adapter discovery, and model registry for Vox."""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from vox.core.errors import AdapterNotFoundError, ModelNotFoundError
from vox.core.store import BlobStore
from vox.core.types import ModelFormat, ModelInfo, ModelType

# ---------------------------------------------------------------------------
# A. Built-in model catalog
# ---------------------------------------------------------------------------

CATALOG: dict[str, dict[str, dict[str, Any]]] = {
    "parakeet": {
        "tdt-0.6b": {
            "source": "nvidia/parakeet-tdt-0.6b-v2",
            "architecture": "parakeet",
            "type": "stt",
            "adapter": "parakeet",
            "format": "onnx",
            "description": "NVIDIA Parakeet TDT 0.6B — top Open ASR Leaderboard model",
            "license": "CC-BY-4.0",
            "parameters": {"sample_rate": 16000},
        },
        "tdt-0.6b-v3": {
            "source": "nvidia/parakeet-tdt-0.6b-v3",
            "architecture": "parakeet",
            "type": "stt",
            "adapter": "parakeet",
            "format": "onnx",
            "description": "NVIDIA Parakeet TDT 0.6B v3 — 25 languages, streaming support",
            "license": "CC-BY-4.0",
            "parameters": {"sample_rate": 16000},
        },
    },
    "kokoro": {
        "v1.0": {
            "source": "hexgrad/Kokoro-82M-v1.0-ONNX",
            "architecture": "kokoro",
            "type": "tts",
            "adapter": "kokoro",
            "format": "onnx",
            "description": "Kokoro 82M ONNX — fast, lightweight TTS with preset voices",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 24000, "default_voice": "af_heart"},
            "files": ["model.onnx", "voices.bin"],
        },
    },
    "whisper": {
        "large-v3": {
            "source": "Systran/faster-whisper-large-v3",
            "architecture": "whisper",
            "type": "stt",
            "adapter": "whisper",
            "format": "ct2",
            "description": "OpenAI Whisper Large V3 via CTranslate2",
            "license": "MIT",
            "parameters": {"sample_rate": 16000, "beam_size": 5},
        },
        "large-v3-turbo": {
            "source": "Systran/faster-whisper-large-v3-turbo",
            "architecture": "whisper",
            "type": "stt",
            "adapter": "whisper",
            "format": "ct2",
            "description": "OpenAI Whisper Large V3 Turbo — faster with minor quality loss",
            "license": "MIT",
            "parameters": {"sample_rate": 16000, "beam_size": 5},
        },
        "base.en": {
            "source": "Systran/faster-whisper-base.en",
            "architecture": "whisper",
            "type": "stt",
            "adapter": "whisper",
            "format": "ct2",
            "description": "OpenAI Whisper Base English — small and fast",
            "license": "MIT",
            "parameters": {"sample_rate": 16000},
        },
    },
    "piper": {
        "en-us-lessac-medium": {
            "source": "rhasspy/piper-voices",
            "architecture": "piper",
            "type": "tts",
            "adapter": "piper",
            "format": "onnx",
            "description": "Piper English US Lessac Medium voice",
            "license": "MIT",
            "parameters": {"sample_rate": 22050},
        },
    },
    "fish-speech": {
        "v1.4": {
            "source": "fishaudio/fish-speech-1.4",
            "architecture": "fish-speech",
            "type": "tts",
            "adapter": "fish-speech",
            "format": "pytorch",
            "description": "Fish Speech 1.4 — high quality multilingual TTS with voice cloning",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 44100},
        },
    },
    "orpheus": {
        "3b": {
            "source": "canopylabs/orpheus-3b-0.1-ft",
            "architecture": "orpheus",
            "type": "tts",
            "adapter": "orpheus",
            "format": "pytorch",
            "description": "Orpheus 3B — LLM-based emotional TTS",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 24000},
        },
    },
}


# ---------------------------------------------------------------------------
# B. Adapter discovery via entry points
# ---------------------------------------------------------------------------

def discover_adapters() -> dict[str, type]:
    """Discover installed adapter classes via 'vox.adapters' entry point group."""
    adapters: dict[str, type] = {}
    for ep in entry_points(group="vox.adapters"):
        adapters[ep.name] = ep.load()
    return adapters


# ---------------------------------------------------------------------------
# C. Registry class
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Ties the built-in catalog, blob store, and adapter discovery together."""

    def __init__(self, store: BlobStore) -> None:
        self._store = store
        self._adapters = discover_adapters()

    # -- catalog helpers -----------------------------------------------------

    def lookup(self, name: str, tag: str = "latest") -> dict | None:
        """Look up a model in the built-in catalog.

        Returns the catalog entry dict, or ``None`` if not found.
        """
        tags = CATALOG.get(name)
        if tags is None:
            return None
        return tags.get(tag)

    def available_models(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return the full built-in catalog for browsing."""
        return CATALOG

    # -- adapter helpers -----------------------------------------------------

    def get_adapter_class(self, adapter_name: str) -> type:
        """Return the adapter class for *adapter_name*.

        Raises :class:`AdapterNotFoundError` if no matching adapter is installed.
        """
        cls = self._adapters.get(adapter_name)
        if cls is None:
            raise AdapterNotFoundError(adapter_name)
        return cls

    # -- resolution ----------------------------------------------------------

    def resolve(self, name: str, tag: str = "latest") -> tuple[ModelInfo, Path]:
        """Resolve a model to its :class:`ModelInfo` and a model directory path.

        The model must already be pulled (i.e. a manifest must exist in the
        store).  Raises :class:`ModelNotFoundError` otherwise.

        The returned path is a directory where blobs are symlinked to their
        original filenames, so adapters can load files by name (e.g.
        ``model.onnx``, ``voices.bin``).
        """
        manifest = self._store.resolve_model(name, tag)
        if manifest is None:
            raise ModelNotFoundError(f"{name}:{tag}")

        cfg = manifest.config
        size = sum(layer.size for layer in manifest.layers)

        info = ModelInfo(
            name=name,
            tag=tag,
            type=ModelType(cfg.get("type", "stt")),
            format=ModelFormat(cfg.get("format", "onnx")),
            architecture=cfg.get("architecture", ""),
            adapter=cfg.get("adapter", ""),
            size_bytes=size,
            description=cfg.get("description", ""),
            license=cfg.get("license", ""),
            parameters=cfg.get("parameters", {}),
        )

        if not manifest.layers:
            raise ModelNotFoundError(f"{name}:{tag} (manifest has no layers)")

        # Build a model directory with symlinks to blobs using original filenames.
        model_dir = self._store.root / "models" / "links" / name / tag
        model_dir.mkdir(parents=True, exist_ok=True)

        for layer in manifest.layers:
            link_path = model_dir / layer.filename
            blob_path = self._store.get_blob_path(layer.digest)
            if not link_path.exists():
                link_path.symlink_to(blob_path)

        return info, model_dir
