"""Built-in model catalog, adapter discovery, and model registry for Vox."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import logging

from vox.core.errors import AdapterNotFoundError, ModelLoadError, ModelNotFoundError
from vox.core.store import BlobStore
from vox.core.types import ModelInfo

logger = logging.getLogger(__name__)

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
            "adapter_package": "vox-parakeet",
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
            "adapter_package": "vox-parakeet",
        },
    },
    "kokoro": {
        "v1.0": {
            "source": "onnx-community/Kokoro-82M-v1.0-ONNX",
            "architecture": "kokoro",
            "type": "tts",
            "adapter": "kokoro",
            "format": "onnx",
            "description": "Kokoro 82M ONNX — fast, lightweight TTS with preset voices",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 24000, "default_voice": "af_heart"},
            "files": ["model.onnx", "voices.bin"],
            "adapter_package": "vox-kokoro",
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
            "adapter_package": "vox-whisper",
        },
        "large-v3-turbo": {
            "source": "deepdml/faster-whisper-large-v3-turbo-ct2",
            "architecture": "whisper",
            "type": "stt",
            "adapter": "whisper",
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
            "adapter": "whisper",
            "format": "ct2",
            "description": "OpenAI Whisper Base English — small and fast",
            "license": "MIT",
            "parameters": {"sample_rate": 16000},
            "adapter_package": "vox-whisper",
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
            "adapter_package": "vox-piper",
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
            "license": "CC-BY-NC-SA-4.0",
            "parameters": {"sample_rate": 44100},
            "adapter_package": "vox-fish-speech",
        },
    },
    "orpheus": {
        "3b": {
            "source": "canopylabs/orpheus-3b-0.1-ft",
            "architecture": "orpheus",
            "type": "tts",
            "adapter": "orpheus",
            "format": "pytorch",
            "description": "Orpheus 3B — LLM-based emotional TTS with inline emotion tags",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 24000},
            "adapter_package": "vox-orpheus",
        },
    },
    "dia": {
        "1.6b": {
            "source": "nari-labs/Dia-1.6B",
            "architecture": "dia",
            "type": "tts",
            "adapter": "dia",
            "format": "pytorch",
            "description": "Dia 1.6B — multi-speaker dialogue TTS with non-verbal sounds",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 44000},
            "adapter_package": "vox-dia",
        },
    },
    "sesame": {
        "csm-1b": {
            "source": "sesame/csm-1b",
            "architecture": "sesame",
            "type": "tts",
            "adapter": "sesame",
            "format": "pytorch",
            "description": "Sesame CSM 1B — context-aware conversational speech model",
            "license": "Apache-2.0",
            "parameters": {"sample_rate": 24000},
            "adapter_package": "vox-sesame",
        },
    },
}


# ---------------------------------------------------------------------------
# B. Adapter discovery via entry points
# ---------------------------------------------------------------------------

ADAPTERS_DIR = "adapters"  # relative to vox home
REGISTRY_BASE_URL = "https://raw.githubusercontent.com/eleven-am/vox-registry/main"


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


def install_adapter_package(package_name: str, vox_home: Path) -> bool:
    """Install an adapter package into the vox adapters directory.

    Returns True if installation succeeded, False otherwise.
    """
    target_dir = vox_home / ADAPTERS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Try uv first (faster), fall back to pip
    for installer in ["uv pip install", "pip install"]:
        try:
            cmd = f"{installer} --target {target_dir} {package_name}"
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                logger.info(f"Installed adapter package: {package_name}")
                return True
            logger.warning(f"{installer} failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    logger.error(f"Failed to install adapter package: {package_name}")
    return False


def _ensure_adapters_on_path(vox_home: Path) -> None:
    """Add the vox adapters directory to sys.path if not already present."""
    adapters_dir = str(vox_home / ADAPTERS_DIR)
    if adapters_dir not in sys.path:
        sys.path.insert(0, adapters_dir)


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

    def lookup(self, name: str, tag: str = "latest") -> dict | None:
        """Look up a model — local catalog first, then remote registry."""
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
