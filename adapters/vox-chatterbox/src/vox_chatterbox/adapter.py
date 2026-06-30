from __future__ import annotations

import importlib
import logging
import subprocess
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from vox.core.adapter import TTSAdapter
from vox.core.adapter_runtime import (
    activate_runtime_path,
    install_target_runtime_requirements,
    purge_runtime_modules,
)
from vox.core.adapter_runtime import (
    runtime_root as vox_runtime_root,
)
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo

logger = logging.getLogger(__name__)

CHATTERBOX_SAMPLE_RATE = 24_000
CHATTERBOX_RUNTIME_DEPS = ("chatterbox-tts>=0.1.7,<0.2.0",)


def _runtime_root() -> Path:
    return vox_runtime_root() / "chatterbox"


def _ensure_runtime_path() -> str:
    runtime_dir = _runtime_root()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return activate_runtime_path(runtime_dir, root=runtime_dir.parent)


def _run_install_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _install_chatterbox_runtime() -> None:
    runtime_path = _ensure_runtime_path()
    if not install_target_runtime_requirements(
        runtime_path,
        CHATTERBOX_RUNTIME_DEPS,
        timeout=900,
        install_runner=_run_install_command,
        context="Chatterbox runtime install",
    ):
        raise RuntimeError("Failed to install Chatterbox runtime package.")


def _clear_chatterbox_modules() -> None:
    purge_runtime_modules(("chatterbox", "s3tokenizer"))


def _load_chatterbox_class(module_name: str, class_name: str) -> type[Any]:
    _ensure_runtime_path()
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        _install_chatterbox_runtime()
        _clear_chatterbox_modules()
        module = importlib.import_module(module_name)

    cls = getattr(module, class_name, None)
    if cls is None:
        raise RuntimeError(
            f"Chatterbox runtime is installed, but {module_name}.{class_name} was not found."
        )
    return cls


def _float_audio(audio: Any) -> NDArray[np.float32]:
    if isinstance(audio, dict):
        for key in ("audio", "wav", "waveform"):
            if key in audio:
                return _float_audio(audio[key])
        raise RuntimeError("Chatterbox returned a dict without audio data.")

    if isinstance(audio, tuple | list) and audio and not isinstance(audio[0], (int, float, np.number)):
        return _float_audio(audio[0])

    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()

    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise RuntimeError("Chatterbox produced no audio.")
    return array


def _sample_rate(model: Any) -> int:
    for attr in ("sr", "sample_rate", "sampling_rate"):
        value = getattr(model, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return CHATTERBOX_SAMPLE_RATE


def _load_model(cls: type[Any], device: str) -> Any:
    if hasattr(cls, "from_pretrained"):
        try:
            return cls.from_pretrained(device=device)
        except TypeError:
            return cls.from_pretrained()
    try:
        return cls(device=device)
    except TypeError:
        return cls()


def _voice_path(voice: str | None) -> str | None:
    if not voice:
        return None
    path = Path(voice).expanduser()
    return str(path) if path.is_file() else None


def _write_reference_audio(path: Path, reference_audio: NDArray[np.float32], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(reference_audio, dtype=np.float32), sample_rate)


class _BaseChatterboxAdapter(TTSAdapter):
    adapter_name = "chatterbox-tts"
    architectures = ("chatterbox-tts", "chatterbox")
    runtime_module = "chatterbox.tts"
    runtime_class = "ChatterboxTTS"
    supported_languages: tuple[str, ...] = ("en",)
    supports_streaming = False

    def __init__(self) -> None:
        self._model: Any | None = None
        self._device = "cpu"
        self._sample_rate = CHATTERBOX_SAMPLE_RATE

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name=self.adapter_name,
            type=ModelType.TTS,
            architectures=self.architectures,
            default_sample_rate=CHATTERBOX_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=self.supports_streaming,
            supports_voice_cloning=True,
            supported_languages=self.supported_languages,
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._model is not None:
            return

        kwargs.pop("_source", None)
        self._device = device
        cls = _load_chatterbox_class(self.runtime_module, self.runtime_class)
        logger.info("Loading Chatterbox runtime %s (device=%s)", self.runtime_class, self._device)
        self._model = _load_model(cls, self._device)
        self._sample_rate = _sample_rate(self._model)

    def unload(self) -> None:
        self._model = None
        self._device = "cpu"
        self._sample_rate = CHATTERBOX_SAMPLE_RATE

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
    ) -> AsyncIterator[SynthesizeChunk]:
        if self._model is None:
            raise RuntimeError("Chatterbox model is not loaded — call load() first")
        if not text or not text.strip():
            return

        kwargs: dict[str, Any] = {}
        if speed and speed != 1.0:
            kwargs["speed"] = speed
        if reference_text:
            kwargs["audio_prompt_text"] = reference_text
        if self.supported_languages != ("en",):
            kwargs["language_id"] = language or "en"

        voice_file = _voice_path(voice)
        if voice_file is not None:
            kwargs["audio_prompt_path"] = voice_file

        if reference_audio is not None:
            with tempfile.TemporaryDirectory(prefix="vox-chatterbox-") as tmpdir:
                ref_path = Path(tmpdir) / "reference.wav"
                _write_reference_audio(ref_path, reference_audio, self._sample_rate)
                kwargs["audio_prompt_path"] = str(ref_path)
                audio = _float_audio(self._model.generate(text, **kwargs))
        else:
            audio = _float_audio(self._model.generate(text, **kwargs))

        chunk_size = self._sample_rate * 2
        for i in range(0, len(audio), chunk_size):
            yield SynthesizeChunk(
                audio=audio[i:i + chunk_size].tobytes(),
                sample_rate=self._sample_rate,
                is_final=False,
            )

        yield SynthesizeChunk(audio=b"", sample_rate=self._sample_rate, is_final=True)

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id="reference",
                name="Reference audio",
                language=None,
                description="Pass reference_audio or a voice path to clone a speaker.",
                is_cloned=True,
            )
        ]

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 2_000_000_000


class ChatterboxTurboAdapter(_BaseChatterboxAdapter):
    adapter_name = "chatterbox-tts-turbo"
    architectures = ("chatterbox-tts-turbo", "chatterbox-turbo")
    runtime_module = "chatterbox.tts_turbo"
    runtime_class = "ChatterboxTurboTTS"


class ChatterboxAdapter(_BaseChatterboxAdapter):
    adapter_name = "chatterbox-tts"
    architectures = ("chatterbox-tts", "chatterbox")
    runtime_class = "ChatterboxTTS"


class ChatterboxMultilingualAdapter(_BaseChatterboxAdapter):
    adapter_name = "chatterbox-tts-multilingual"
    architectures = ("chatterbox-tts-multilingual", "chatterbox-multilingual")
    runtime_module = "chatterbox.mtl_tts"
    runtime_class = "ChatterboxMultilingualTTS"
    supported_languages = (
        "ar",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "he",
        "hi",
        "it",
        "ja",
        "ko",
        "ms",
        "nl",
        "no",
        "pl",
        "pt",
        "ru",
        "sv",
        "sw",
        "tr",
        "zh",
    )
