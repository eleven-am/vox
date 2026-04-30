from __future__ import annotations

import logging
import platform
from collections.abc import AsyncIterator
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
from kokoro_onnx import Kokoro, KoKoroConfig, Tokenizer
from onnxruntime import InferenceSession, get_available_providers

from vox.core.adapter import TTSAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)
from vox_kokoro.common import SAMPLE_RATE, SUPPORTED_LANGUAGES, voice_info, voice_lang_tag

logger = logging.getLogger(__name__)


def _select_audio_output(session: InferenceSession, outputs: list[Any]) -> np.ndarray:
    output_metas = list(session.get_outputs())
    preferred_tokens = ("audio", "wave", "wav")
    for idx, meta in enumerate(output_metas):
        if idx >= len(outputs):
            break
        if any(token in meta.name.lower() for token in preferred_tokens):
            return np.asarray(outputs[idx], dtype=np.float32)

    candidates: list[tuple[int, np.ndarray]] = []
    for output in outputs:
        array = np.asarray(output)
        if array.dtype.kind == "f" and array.size > 1:
            candidates.append((array.size, np.asarray(array, dtype=np.float32)))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return np.asarray(outputs[0], dtype=np.float32)

def _get_onnx_providers(device: str) -> tuple[list[tuple[str, dict]], str]:
    """Choose ONNX execution providers based on *device* and platform."""
    available = get_available_providers()
    system = platform.system()
    machine = platform.machine()

    logger.info("Available ONNX providers: %s", available)

    if device == "cpu":
        return [("CPUExecutionProvider", {})], "cpu"

    providers: list[tuple[str, dict]] = []
    resolved_device = "cpu"

    if system == "Darwin" and machine == "arm64" and "CoreMLExecutionProvider" in available:
        providers.append(("CoreMLExecutionProvider", {}))
        resolved_device = "coreml"

    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {}))
        resolved_device = "cuda"

    if not providers:
        if device == "auto":
            return [("CPUExecutionProvider", {})], "cpu"
        raise RuntimeError(
            "Kokoro requires a GPU-capable ONNX Runtime provider for non-CPU devices; "
            "CPU fallback is disabled"
        )

    providers.append(("CPUExecutionProvider", {}))

    logger.info("Using ONNX providers: %s", providers)
    return providers, resolved_device


def _voice_lang(voice_id: str) -> str:
    """Return the Kokoro language tag for a voice ID."""
    return voice_lang_tag(voice_id)


def _voice_info(voice_id: str) -> VoiceInfo:
    """Build a VoiceInfo from a Kokoro voice ID like ``af_heart``."""
    return voice_info(voice_id)


class KokoroAdapter(TTSAdapter):
    """Vox TTS adapter backed by Kokoro-82M via ``kokoro-onnx``."""

    def __init__(self) -> None:
        self._kokoro: Kokoro | None = None
        self._device: str = "cpu"





    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="kokoro-tts-onnx",
            type=ModelType.TTS,
            architectures=("kokoro-tts-onnx", "kokoro"),
            default_sample_rate=SAMPLE_RATE,
            supported_formats=(ModelFormat.ONNX,),
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages=SUPPORTED_LANGUAGES,
            max_input_chars=250,
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        """Load the Kokoro model from *model_path*.

        Supports both the legacy layout:
        - ``model.onnx``
        - ``voices.bin``

        and the current Hugging Face layout:
        - ``onnx/model.onnx``
        - ``voices/*.bin``

        *device* is mapped to ONNX execution providers (``"cpu"``,
        ``"cuda"``, or ``"auto"``).
        """
        model_dir = Path(model_path)
        model_file = self._resolve_model_file(model_dir)
        voices_file = model_dir / "voices.bin"
        voices_dir = model_dir / "voices"

        if model_file is None:
            raise FileNotFoundError(f"No Kokoro ONNX model file found in {model_dir}")
        if not voices_file.exists() and not voices_dir.is_dir():
            raise FileNotFoundError(f"No Kokoro voices found in {model_dir}")

        providers, resolved_device = _get_onnx_providers(device)
        self._device = resolved_device

        logger.info("Loading Kokoro model from %s (device=%s)", model_dir, self._device)
        session = InferenceSession(str(model_file), providers=providers)
        if voices_file.exists():
            self._kokoro = Kokoro.from_session(session, str(voices_file))
        else:
            self._kokoro = self._load_directory_layout(session, model_file, voices_dir)
        self._patch_runtime_compat()
        logger.info("Kokoro model loaded")

    def _resolve_model_file(self, model_dir: Path) -> Path | None:
        for candidate in (model_dir / "model.onnx", model_dir / "onnx" / "model.onnx"):
            if candidate.exists():
                return candidate
        return None

    def _load_directory_layout(self, session: InferenceSession, model_file: Path, voices_dir: Path) -> Kokoro:
        voices = self._load_voice_tensors(voices_dir)
        kokoro = Kokoro.__new__(Kokoro)
        kokoro.sess = session
        kokoro.config = KoKoroConfig(str(model_file), str(voices_dir))
        kokoro.voices = voices
        kokoro.tokenizer = Tokenizer(None, vocab={})
        return kokoro

    def _load_voice_tensors(self, voices_dir: Path) -> dict[str, np.ndarray]:
        voices: dict[str, np.ndarray] = {}
        for voice_file in sorted(voices_dir.glob("*.bin")):
            data = np.fromfile(voice_file, dtype=np.float32)
            if data.size == 0 or data.size % 256 != 0:
                raise ValueError(f"Unexpected Kokoro voice tensor shape in {voice_file}")
            voices[voice_file.stem] = data.reshape(-1, 1, 256)
        if not voices:
            raise FileNotFoundError(f"No Kokoro voice bins found in {voices_dir}")
        return voices

    def _patch_runtime_compat(self) -> None:
        if self._kokoro is None:
            return

        input_types = {
            input_meta.name: input_meta.type
            for input_meta in self._kokoro.sess.get_inputs()
        }
        if input_types.get("input_ids") and input_types.get("speed") == "tensor(float)":
            logger.info("Patching Kokoro runtime for float speed input")
            self._kokoro._create_audio = MethodType(_create_audio_float_speed, self._kokoro)

    def unload(self) -> None:
        if self._kokoro is not None:
            self._kokoro = None
            logger.info("Kokoro model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._kokoro is not None

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        reference_audio: np.ndarray | None = None,
        reference_text: str | None = None,
    ) -> AsyncIterator[SynthesizeChunk]:
        if self._kokoro is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not text or not text.strip():
            return

        voice_id = voice or "af_heart"
        speed = max(0.5, min(speed, 2.0))

        lang = language or _voice_lang(voice_id)

        async for audio_chunk, _token in self._kokoro.create_stream(
            text, voice_id, lang=lang, speed=speed, trim=False
        ):
            yield SynthesizeChunk(
                audio=audio_chunk.astype(np.float32).tobytes(),
                sample_rate=SAMPLE_RATE,
                is_final=False,
            )


        yield SynthesizeChunk(
            audio=b"",
            sample_rate=SAMPLE_RATE,
            is_final=True,
        )

    def list_voices(self) -> list[VoiceInfo]:
        if self._kokoro is None:
            return []
        return [_voice_info(v) for v in self._kokoro.get_voices()]

    def estimate_vram_bytes(self, **kwargs: Any) -> int:

        return 330 * 1024 * 1024


def _create_audio_float_speed(self, phonemes: str, voice: np.ndarray, speed: float) -> tuple[np.ndarray, int]:
    tokens = np.array(self.tokenizer.tokenize(phonemes), dtype=np.int64)
    voice = voice[len(tokens)]
    padded_tokens = np.array([[0, *tokens.tolist(), 0]], dtype=np.int64)
    inputs = {
        "input_ids": padded_tokens,
        "style": np.array(voice, dtype=np.float32),
        "speed": np.array([speed], dtype=np.float32),
    }
    outputs = self.sess.run(None, inputs)
    audio = _select_audio_output(self.sess, outputs)
    logger.info(
        "kokoro_create_audio phonemes=%d tokens=%d audio_samples=%d outputs=%s",
        len(phonemes),
        len(tokens),
        int(audio.size),
        [meta.name for meta in self.sess.get_outputs()],
    )
    return audio, SAMPLE_RATE
