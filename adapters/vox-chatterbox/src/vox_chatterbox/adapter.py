from __future__ import annotations

import importlib
import importlib.util
import logging
import shutil
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
from vox.operations.errors import InvalidConfigError

logger = logging.getLogger(__name__)

CHATTERBOX_SAMPLE_RATE = 24_000
CHATTERBOX_PACKAGE = "chatterbox-tts>=0.1.7,<0.2.0"
CHATTERBOX_ONNX_RUNTIME_DEPS = (
    "onnxruntime>=1.27.0,<2.0.0",
    "transformers>=4.56.0,<6.0.0",
    "librosa==0.11.0",
)
CHATTERBOX_RUNTIME_DEPS = (
    "numpy>=1.26.0,<2.0.0",
    "librosa==0.11.0",
    "s3tokenizer",
    "transformers==5.2.0",
    "diffusers==0.29.0",
    "resemble-perth>=1.0.0",
    "conformer==0.3.2",
    "safetensors==0.5.3",
    "spacy-pkuseg",
    "pykakasi==2.3.0",
    "gradio==6.8.0",
    "pyloudnorm",
    "omegaconf",
)
CHATTERBOX_APP_RUNTIME_PACKAGES = ("torch", "torchgen", "torchaudio", "nvidia")
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SILENCE_TOKEN = 4299
NUM_KV_HEADS = 16
HEAD_DIM = 64


def _runtime_root() -> Path:
    return vox_runtime_root() / "chatterbox"


def _ensure_runtime_path() -> str:
    runtime_dir = _runtime_root()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return activate_runtime_path(runtime_dir, root=runtime_dir.parent)


def _run_install_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _runtime_module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _purge_chatterbox_app_runtime_packages(runtime_path: str | Path) -> None:
    runtime_dir = Path(runtime_path)
    if not runtime_dir.exists():
        return

    normalized_names = {
        name.replace("-", "_").lower()
        for name in CHATTERBOX_APP_RUNTIME_PACKAGES
    }
    for child in runtime_dir.iterdir():
        child_name = child.name.replace("-", "_").lower()
        if child_name in normalized_names or any(
            child_name.startswith(f"{name}_") or child_name.startswith(f"{name}.")
            for name in normalized_names
        ):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)


def _install_chatterbox_runtime() -> None:
    runtime_path = _ensure_runtime_path()
    _purge_chatterbox_app_runtime_packages(runtime_path)
    if not install_target_runtime_requirements(
        runtime_path,
        (CHATTERBOX_PACKAGE,),
        no_deps=True,
        timeout=900,
        install_runner=_run_install_command,
        context="Chatterbox runtime package install",
    ):
        raise RuntimeError("Failed to install Chatterbox runtime package.")
    if not install_target_runtime_requirements(
        runtime_path,
        CHATTERBOX_RUNTIME_DEPS,
        timeout=900,
        install_runner=_run_install_command,
        context="Chatterbox runtime dependency install",
    ):
        raise RuntimeError("Failed to install Chatterbox runtime dependencies.")
    _purge_chatterbox_app_runtime_packages(runtime_path)


def _install_chatterbox_onnx_runtime() -> None:
    runtime_path = _ensure_runtime_path()
    if _runtime_module_available("onnxruntime") and _runtime_module_available("transformers"):
        return
    if not install_target_runtime_requirements(
        runtime_path,
        CHATTERBOX_ONNX_RUNTIME_DEPS,
        timeout=900,
        install_runner=_run_install_command,
        context="Chatterbox ONNX runtime dependency install",
    ):
        raise RuntimeError("Failed to install Chatterbox ONNX runtime dependencies.")


def _clear_chatterbox_modules() -> None:
    purge_runtime_modules((
        "chatterbox",
        "diffusers",
        "huggingface_hub",
        "s3tokenizer",
        "tokenizers",
        "transformers",
    ))


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


def _load_reference_audio(
    *,
    voice: str | None,
    reference_audio: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    if reference_audio is not None:
        audio = np.asarray(reference_audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError("Chatterbox ONNX reference_audio is empty.")
        return audio

    voice_file = _voice_path(voice)
    if voice_file is None:
        raise RuntimeError(
            "Chatterbox Turbo ONNX requires reference_audio or a voice WAV path. "
            "Use the default chatterbox-tts-turbo variant for default-voice synthesis."
        )

    import librosa

    audio, _ = librosa.load(voice_file, sr=CHATTERBOX_SAMPLE_RATE, mono=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise RuntimeError(f"Chatterbox ONNX voice file has no audio: {voice_file}")
    return audio


def _model_file(model_dir: Path, name: str, dtype: str) -> Path:
    suffix = {
        "fp32": "",
        "fp16": "_fp16",
        "q8": "_quantized",
        "quantized": "_quantized",
        "q4": "_q4",
        "q4f16": "_q4f16",
    }.get(dtype)
    if suffix is None:
        raise ValueError(
            "Unsupported Chatterbox ONNX dtype "
            f"{dtype!r}; expected fp32, fp16, q8, q4, or q4f16"
        )
    path = model_dir / "onnx" / f"{name}{suffix}.onnx"
    if not path.is_file():
        raise RuntimeError(f"Chatterbox ONNX model file is missing: {path}")
    return path


def _session_providers(device: str, ort: Any) -> list[str]:
    available = set(ort.get_available_providers())
    providers: list[str] = []
    if device == "cuda" and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


class _RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float) -> None:
        if penalty <= 0:
            raise ValueError("repetition penalty must be positive")
        self.penalty = float(penalty)

    def __call__(self, input_ids: NDArray[np.int64], scores: NDArray[np.float32]) -> NDArray[np.float32]:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed


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


class ChatterboxTurboOnnxAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._loaded = False
        self._model_dir: Path | None = None
        self._device = "cpu"
        self._onnx_dtype = "fp32"
        self._sample_rate = CHATTERBOX_SAMPLE_RATE
        self._tokenizer: Any = None
        self._speech_encoder_session: Any = None
        self._embed_tokens_session: Any = None
        self._language_model_session: Any = None
        self._conditional_decoder_session: Any = None

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="chatterbox-tts-turbo-onnx",
            type=ModelType.TTS,
            architectures=("chatterbox-tts-turbo-onnx", "chatterbox-turbo"),
            default_sample_rate=CHATTERBOX_SAMPLE_RATE,
            supported_formats=(ModelFormat.ONNX,),
            supports_streaming=False,
            supports_voice_cloning=True,
            supported_languages=("en",),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        _install_chatterbox_onnx_runtime()

        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._model_dir = Path(model_path)
        self._device = device
        self._onnx_dtype = str(kwargs.pop("onnx_dtype", "fp32") or "fp32")
        providers = _session_providers(self._device, ort)

        logger.info(
            "Loading Chatterbox Turbo ONNX runtime from %s (dtype=%s, providers=%s)",
            self._model_dir,
            self._onnx_dtype,
            ",".join(providers),
        )
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
        self._speech_encoder_session = ort.InferenceSession(
            str(_model_file(self._model_dir, "speech_encoder", self._onnx_dtype)),
            providers=providers,
        )
        self._embed_tokens_session = ort.InferenceSession(
            str(_model_file(self._model_dir, "embed_tokens", self._onnx_dtype)),
            providers=providers,
        )
        self._language_model_session = ort.InferenceSession(
            str(_model_file(self._model_dir, "language_model", self._onnx_dtype)),
            providers=providers,
        )
        self._conditional_decoder_session = ort.InferenceSession(
            str(_model_file(self._model_dir, "conditional_decoder", self._onnx_dtype)),
            providers=providers,
        )
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False
        self._model_dir = None
        self._device = "cpu"
        self._onnx_dtype = "fp32"
        self._tokenizer = None
        self._speech_encoder_session = None
        self._embed_tokens_session = None
        self._language_model_session = None
        self._conditional_decoder_session = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def validate_synthesis_request(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
    ) -> None:
        if reference_audio is not None:
            if np.asarray(reference_audio, dtype=np.float32).size == 0:
                raise InvalidConfigError("Chatterbox Turbo ONNX reference_audio is empty")
            return
        if _voice_path(voice) is not None:
            return
        raise InvalidConfigError(
            "Chatterbox Turbo ONNX requires reference_audio or a stored voice/reference WAV"
        )

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
        if not self._loaded:
            raise RuntimeError("Chatterbox Turbo ONNX model is not loaded — call load() first")
        if not text or not text.strip():
            return

        audio = self._generate(
            text=text,
            voice=voice,
            reference_audio=reference_audio,
            max_new_tokens=1024,
            repetition_penalty=1.2,
        )
        chunk_size = self._sample_rate * 2
        for i in range(0, len(audio), chunk_size):
            yield SynthesizeChunk(
                audio=audio[i:i + chunk_size].astype(np.float32).tobytes(),
                sample_rate=self._sample_rate,
                is_final=False,
            )
        yield SynthesizeChunk(audio=b"", sample_rate=self._sample_rate, is_final=True)

    def _generate(
        self,
        *,
        text: str,
        voice: str | None,
        reference_audio: NDArray[np.float32] | None,
        max_new_tokens: int,
        repetition_penalty: float,
    ) -> NDArray[np.float32]:
        assert self._tokenizer is not None
        assert self._speech_encoder_session is not None
        assert self._embed_tokens_session is not None
        assert self._language_model_session is not None
        assert self._conditional_decoder_session is not None

        audio_values = _load_reference_audio(
            voice=voice,
            reference_audio=reference_audio,
        )[np.newaxis, :].astype(np.float32)
        input_ids = self._tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
        repetition_penalty_processor = _RepetitionPenaltyLogitsProcessor(repetition_penalty)
        past_key_values: dict[str, NDArray[np.float32]] | None = None
        attention_mask: NDArray[np.int64] | None = None
        position_ids: NDArray[np.int64] | None = None
        prompt_token = None
        speaker_embeddings = None
        speaker_features = None

        for i in range(max_new_tokens):
            inputs_embeds = self._embed_tokens_session.run(None, {"input_ids": input_ids})[0]
            if i == 0:
                cond_emb, prompt_token, speaker_embeddings, speaker_features = (
                    self._speech_encoder_session.run(None, {"audio_values": audio_values})
                )
                inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)
                batch_size, seq_len, _ = inputs_embeds.shape
                past_key_values = {
                    inp.name: np.zeros(
                        [batch_size, NUM_KV_HEADS, 0, HEAD_DIM],
                        dtype=np.float16 if inp.type == "tensor(float16)" else np.float32,
                    )
                    for inp in self._language_model_session.get_inputs()
                    if "past_key_values" in inp.name
                }
                attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
                position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1).repeat(batch_size, axis=0)

            assert past_key_values is not None
            assert attention_mask is not None
            assert position_ids is not None
            logits, *present_key_values = self._language_model_session.run(
                None,
                {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    **past_key_values,
                },
            )
            next_token_logits = repetition_penalty_processor(generate_tokens, logits[:, -1, :])
            input_ids = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
            generate_tokens = np.concatenate((generate_tokens, input_ids), axis=-1)
            if (input_ids.flatten() == STOP_SPEECH_TOKEN).all():
                break

            attention_mask = np.concatenate(
                [attention_mask, np.ones((attention_mask.shape[0], 1), dtype=np.int64)],
                axis=1,
            )
            position_ids = position_ids[:, -1:] + 1
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]

        if prompt_token is None or speaker_embeddings is None or speaker_features is None:
            raise RuntimeError("Chatterbox ONNX speech encoder did not produce conditioning outputs.")

        speech_tokens = generate_tokens[:, 1:-1]
        silence_tokens = np.full((speech_tokens.shape[0], 3), SILENCE_TOKEN, dtype=np.int64)
        speech_tokens = np.concatenate([prompt_token, speech_tokens, silence_tokens], axis=1)
        wav = self._conditional_decoder_session.run(
            None,
            {
                "speech_tokens": speech_tokens,
                "speaker_embeddings": speaker_embeddings,
                "speaker_features": speaker_features,
            },
        )[0]
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError("Chatterbox Turbo ONNX produced no audio.")
        return audio

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id="reference",
                name="Reference audio",
                language="en",
                description="Pass reference_audio or a voice path to clone a speaker.",
                is_cloned=True,
            )
        ]

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 2_000_000_000


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
