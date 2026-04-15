from __future__ import annotations

import importlib
import importlib.util
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray

from vox.core.adapter import TTSAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)

logger = logging.getLogger(__name__)

OPENVOICE_SAMPLE_RATE = 22_050
OPENVOICE_REPO = "git+https://github.com/myshell-ai/OpenVoice.git"
OPENVOICE_RUNTIME_DEPS = (
    "unidecode==1.3.7",
    "eng_to_ipa==0.0.2",
    "inflect==7.0.0",
    "pypinyin==0.50.0",
    "cn2an==0.5.22",
    "proces==0.1.7",
    "jieba==0.42.1",
    "pydub==0.25.1",
    "wavmark==0.0.3",
    "langid==1.1.6",
)
OPENVOICE_VOICE_NAMES = (
    "default",
    "whispering",
    "shouting",
    "excited",
    "cheerful",
    "terrified",
    "angry",
    "sad",
    "friendly",
)
OPENVOICE_LANGUAGES = ("en", "zh")


def _select_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda:0"
    if device in ("mps", "auto") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _language_name(language: str | None) -> str:
    if language is None:
        return "English"
    key = language.strip().lower()
    if key in {"en", "english", "en-us", "en-gb"}:
        return "English"
    if key in {"zh", "zh-cn", "zh-hans", "chinese", "cmn"}:
        return "Chinese"
    raise ValueError(
        "OpenVoice V1 only supports English and Chinese base generation. "
        "Use language='en' or language='zh'."
    )


def _language_code(language: str) -> str:
    return "zh" if language == "Chinese" else "en"


def _parse_voice(voice: str | None, language: str) -> tuple[str, str]:
    default_voice = f"{_language_code(language)}/default"
    voice_id = (voice or default_voice).strip() or default_voice
    if "/" in voice_id:
        language_code, speaker = voice_id.split("/", 1)
        language_code = language_code.strip().lower()
        speaker = speaker.strip() or "default"
        if language_code not in OPENVOICE_LANGUAGES:
            raise ValueError(
                f"Unknown OpenVoice language prefix '{language_code}'. "
                "Use 'en/<speaker>' or 'zh/<speaker>'."
            )
        return language_code, speaker
    return _language_code(language), voice_id


def _build_voice_list() -> list[VoiceInfo]:
    voices: list[VoiceInfo] = []
    for language in OPENVOICE_LANGUAGES:
        for speaker in OPENVOICE_VOICE_NAMES:
            voices.append(
                VoiceInfo(
                    id=f"{language}/{speaker}",
                    name=f"{language.upper()} {speaker}",
                    language=language,
                    description="OpenVoice V1 preset speaker",
                )
            )
    return voices


def _install_openvoice_runtime() -> None:
    logger.info("Installing OpenVoice runtime from %s", OPENVOICE_REPO)
    uv_executable = shutil.which("uv") or "/usr/bin/uv"
    install_cmd = [
        uv_executable,
        "pip",
        "install",
        "--python",
        sys.executable,
        "--no-build-isolation",
        "--no-deps",
        OPENVOICE_REPO,
        *OPENVOICE_RUNTIME_DEPS,
    ]
    result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        if importlib.util.find_spec("pip") is None:
            bootstrap = subprocess.run(
                [sys.executable, "-m", "ensurepip", "--default-pip"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if bootstrap.returncode != 0:
                raise RuntimeError(
                    "Failed to bootstrap pip for OpenVoice runtime install. "
                    f"stderr: {bootstrap.stderr.strip()}"
                )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                OPENVOICE_REPO,
                *OPENVOICE_RUNTIME_DEPS,
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to install OpenVoice runtime from GitHub. "
            f"stderr: {result.stderr.strip()}"
        )


def _clear_openvoice_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "openvoice" or module_name.startswith(
            ("openvoice.", "cn2an", "cn2an.", "proces", "proces.", "jieba", "jieba.")
        ):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _load_openvoice_api() -> tuple[type[Any], type[Any]]:
    try:
        module = importlib.import_module("openvoice.api")
    except ImportError:
        _install_openvoice_runtime()
        _clear_openvoice_modules()
        module = importlib.import_module("openvoice.api")

    try:
        return module.BaseSpeakerTTS, module.ToneColorConverter
    except AttributeError as exc:
        raise RuntimeError(
            "OpenVoice runtime is installed, but the expected API was not found. "
            "The adapter requires openvoice.api.BaseSpeakerTTS and openvoice.api.ToneColorConverter."
        ) from exc


def _move_to_device(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs


def _save_audio_file(path: Path, audio: NDArray[np.float32], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def _load_audio_file(path: Path, sample_rate: int | None = None) -> NDArray[np.float32]:
    audio, sr = librosa.load(path, sr=sample_rate)
    return audio.astype(np.float32)


class OpenVoiceTTSAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._loaded = False
        self._device = "cpu"
        self._model_root: Path | None = None
        self._converter: Any | None = None
        self._base_models: dict[str, Any] = {}

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="openvoice",
            type=ModelType.TTS,
            architectures=("openvoice", "openvoice-v1"),
            default_sample_rate=OPENVOICE_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=False,
            supports_voice_cloning=True,
            supported_languages=OPENVOICE_LANGUAGES,
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        self._device = _select_device(device)
        self._model_root = Path(kwargs.pop("_source", None) or model_path)
        BaseSpeakerTTS, ToneColorConverter = _load_openvoice_api()

        converter_config = self._model_root / "checkpoints" / "converter" / "config.json"
        converter_ckpt = self._model_root / "checkpoints" / "converter" / "checkpoint.pth"
        if not converter_config.is_file():
            raise FileNotFoundError(f"Missing OpenVoice converter config: {converter_config}")
        if not converter_ckpt.is_file():
            raise FileNotFoundError(f"Missing OpenVoice converter checkpoint: {converter_ckpt}")

        logger.info("Loading OpenVoice converter from %s (device=%s)", self._model_root, self._device)
        start = time.perf_counter()

        try:
            self._converter = ToneColorConverter(
                str(converter_config),
                device=self._device,
                enable_watermark=False,
            )
        except TypeError:
            self._converter = ToneColorConverter(str(converter_config), device=self._device)
        self._converter.load_ckpt(str(converter_ckpt))

        elapsed = time.perf_counter() - start
        logger.info("OpenVoice runtime loaded in %.2fs", elapsed)
        self._loaded = True
        self._BaseSpeakerTTS = BaseSpeakerTTS  # type: ignore[attr-defined]

    def unload(self) -> None:
        self._loaded = False
        self._converter = None
        self._base_models.clear()
        self._model_root = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _resolve_base_paths(self, language_code: str) -> tuple[Path, Path]:
        assert self._model_root is not None
        subdir = "EN" if language_code == "en" else "ZH"
        base_dir = self._model_root / "checkpoints" / "base_speakers" / subdir
        config = base_dir / "config.json"
        ckpt = base_dir / "checkpoint.pth"
        if not config.is_file():
            raise FileNotFoundError(f"Missing OpenVoice base speaker config: {config}")
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing OpenVoice base speaker checkpoint: {ckpt}")
        return config, ckpt

    def _get_base_model(self, language_code: str) -> Any:
        model = self._base_models.get(language_code)
        if model is not None:
            return model

        BaseSpeakerTTS = getattr(self, "_BaseSpeakerTTS", None)
        if BaseSpeakerTTS is None:
            raise RuntimeError("OpenVoice runtime not loaded; call load() first")

        config, ckpt = self._resolve_base_paths(language_code)
        model = BaseSpeakerTTS(str(config), device=self._device)
        model.load_ckpt(str(ckpt))
        self._base_models[language_code] = model
        return model

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
        if not self._loaded or self._converter is None or self._model_root is None:
            raise RuntimeError("OpenVoice model is not loaded — call load() first")

        if not text or not text.strip():
            return

        language_name = _language_name(language)
        language_code, speaker = _parse_voice(voice, language_name)
        base_model = self._get_base_model(language_code)
        speed = max(0.5, min(speed, 2.0))

        generated = base_model.tts(
            text,
            output_path=None,
            speaker=speaker,
            language=language_name,
            speed=speed,
        )
        base_audio = np.asarray(generated, dtype=np.float32)
        sample_rate = getattr(base_model.hps.data, "sampling_rate", OPENVOICE_SAMPLE_RATE)

        if reference_audio is not None:
            with tempfile.TemporaryDirectory(prefix="vox-openvoice-") as tmpdir:
                tmpdir_path = Path(tmpdir)
                base_wav = tmpdir_path / "base.wav"
                ref_wav = tmpdir_path / "reference.wav"
                _save_audio_file(base_wav, base_audio, sample_rate)
                _save_audio_file(ref_wav, np.asarray(reference_audio, dtype=np.float32), OPENVOICE_SAMPLE_RATE)

                src_se = self._converter.extract_se(str(base_wav))
                tgt_se = self._converter.extract_se(str(ref_wav))
                converted = self._converter.convert(str(base_wav), src_se, tgt_se, output_path=None)
                base_audio = np.asarray(converted, dtype=np.float32)
                sample_rate = getattr(self._converter.hps.data, "sampling_rate", sample_rate)

        chunk_size = sample_rate * 2
        for i in range(0, len(base_audio), chunk_size):
            chunk = base_audio[i:i + chunk_size]
            yield SynthesizeChunk(
                audio=chunk.tobytes(),
                sample_rate=sample_rate,
                is_final=False,
            )

        yield SynthesizeChunk(
            audio=b"",
            sample_rate=sample_rate,
            is_final=True,
        )

    def list_voices(self) -> list[VoiceInfo]:
        return _build_voice_list()

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 4_000_000_000
