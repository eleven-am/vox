from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from vox.core.adapter import TTSAdapter
from vox.core.device_placement import PlacementTier
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)
from vox_voxtral.backends import (
    InProcessOmniBackend,
    OmniBackend,
    SubprocessOmniBackend,
)
from vox_voxtral.runtime import (
    VOXTRAL_TIER_DEFAULT,
    VOXTRAL_TIER_SMALL_24GB,
    VOXTRAL_TIER_SPARK_16GB,
    ensure_voxtral_tts_runtime,
    recommended_voxtral_tts_vram_bytes,
    voxtral_tts_tier_extras,
)


VOXTRAL_TTS_TIERS: tuple[PlacementTier, ...] = (
    PlacementTier(
        name=VOXTRAL_TIER_SPARK_16GB,
        total_memory_max_bytes=16 * 1024**3,
        extras=voxtral_tts_tier_extras(VOXTRAL_TIER_SPARK_16GB),
    ),
    PlacementTier(
        name=VOXTRAL_TIER_SMALL_24GB,
        total_memory_max_bytes=24 * 1024**3,
        extras=voxtral_tts_tier_extras(VOXTRAL_TIER_SMALL_24GB),
    ),
    PlacementTier(
        name=VOXTRAL_TIER_DEFAULT,
        total_memory_max_bytes=None,
        extras=voxtral_tts_tier_extras(VOXTRAL_TIER_DEFAULT),
    ),
)

logger = logging.getLogger(__name__)

VOXTRAL_TTS_SAMPLE_RATE = 24_000

PRESET_VOICES: list[VoiceInfo] = [
    VoiceInfo(id="neutral_female", name="neutral_female", description="Official Voxtral preset voice"),
    VoiceInfo(id="cheerful_female", name="cheerful_female", description="Official Voxtral preset voice"),
    VoiceInfo(id="casual_male", name="casual_male", description="Official Voxtral preset voice"),
]

SUPPORTED_LANGUAGES = ("en", "fr", "es", "de", "it", "pt", "nl", "ar", "hi")


def _resolve_stage_configs_path(explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path

    try:
        stage_configs = files("vllm_omni").joinpath("model_executor/stage_configs/voxtral_tts.yaml")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Voxtral TTS requires vLLM-Omni stage configs. "
            "Install vllm-omni>=0.18.0 or pass `_stage_configs_path` explicitly."
        ) from exc

    if stage_configs.is_file():
        return str(stage_configs)

    raise RuntimeError(
        "Voxtral TTS requires vLLM-Omni stage configs. "
        "Pass `_stage_configs_path` when loading or install a vLLM-Omni build "
        "that ships `model_executor/stage_configs/voxtral_tts.yaml`."
    )


def _load_voxtral_tts_runtime() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    try:
        from mistral_common.protocol.speech.request import SpeechRequest
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from vllm import SamplingParams
        from vllm_omni import AsyncOmni
    except ImportError as exc:
        raise RuntimeError(
            "Voxtral TTS requires vllm>=0.18.0, vllm-omni>=0.18.0, and "
            "mistral-common[audio]>=1.10.0"
        ) from exc

    return AsyncOmni, SamplingParams, SpeechRequest, MistralTokenizer


class VoxtralTTSAdapter(TTSAdapter):

    def __init__(self) -> None:
        self._backend: OmniBackend | None = None
        self._runtime: Any | None = None
        self._tokenizer: Any | None = None
        self._speech_request_cls: Any | None = None
        self._subprocess_only = False
        self._loaded = False
        self._model_id: str = ""
        self._model_ref: str = ""
        self._device: str = "cpu"
        self._default_voice: str | None = None
        self._placement_tier: str | None = None
        self._placement_extras: dict[str, Any] = {}

    def placement_tiers(self) -> tuple[PlacementTier, ...]:
        return VOXTRAL_TTS_TIERS

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="voxtral-tts-vllm",
            type=ModelType.TTS,
            architectures=("voxtral-tts-vllm", "voxtral-tts"),
            default_sample_rate=VOXTRAL_TTS_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages=SUPPORTED_LANGUAGES,
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        source = kwargs.pop("_source", None)
        self._model_id = source if source else model_path
        self._model_ref = model_path
        self._default_voice = kwargs.pop("default_voice", None)
        explicit_stage_configs_path = kwargs.pop("_stage_configs_path", None)
        self._placement_tier = kwargs.pop("_placement_tier", None)
        self._placement_extras = dict(kwargs.pop("_placement_extras", {}) or {})
        if device == "cpu":
            raise RuntimeError(
                "Voxtral TTS on Vox requires CUDA + vLLM-Omni; CPU and MPS are not supported"
            )
        self._device = "cuda"

        log_stats = bool(kwargs.pop("log_stats", False))

        try:
            stage_configs_path = _resolve_stage_configs_path(explicit_stage_configs_path)
            AsyncOmni, SamplingParams, SpeechRequest, MistralTokenizer = _load_voxtral_tts_runtime()
            self._speech_request_cls = SpeechRequest

            logger.info("Loading Voxtral TTS model: %s (device=%s)", self._model_ref, self._device)
            start = time.perf_counter()

            self._tokenizer = self._load_tokenizer(MistralTokenizer)
            self._runtime = AsyncOmni(
                model=self._model_ref,
                stage_configs_path=stage_configs_path,
                log_stats=log_stats,
            )
            sampling_params = SamplingParams(max_tokens=2500)
            self._sampling_params = [sampling_params, sampling_params]

            elapsed = time.perf_counter() - start
            logger.info("Voxtral TTS model loaded in %.2fs", elapsed)
            self._backend = InProcessOmniBackend(
                runtime=self._runtime,
                tokenizer=self._tokenizer,
                speech_request_cls=self._speech_request_cls,
                sampling_params=self._sampling_params,
            )
            self._subprocess_only = False
            self._loaded = True
            return
        except Exception as exc:
            logger.warning(
                "Falling back to subprocess-isolated Voxtral TTS runtime for %s: %s",
                self._model_id,
                exc,
            )

        self._backend = self._make_subprocess_backend(explicit_stage_configs_path=explicit_stage_configs_path)
        self._subprocess_only = True
        self._loaded = True

    def _load_tokenizer(self, mistral_tokenizer_cls: Any) -> Any:
        model_path = self._model_ref or self._model_id
        model_dir = None
        try:
            from pathlib import Path

            candidate = Path(model_path)
            if candidate.is_dir() and (candidate / "tekken.json").is_file():
                model_dir = candidate
        except Exception:
            model_dir = None

        if model_dir is not None:
            return mistral_tokenizer_cls.from_file(str(model_dir / "tekken.json"))
        return mistral_tokenizer_cls.from_hf_hub(self._model_id)

    def _make_subprocess_backend(self, *, explicit_stage_configs_path: str | None) -> SubprocessOmniBackend:
        worker_script = str(Path(__file__).with_name("voxtral_tts_worker.py"))
        runtime = ensure_voxtral_tts_runtime()
        stage_configs_path = explicit_stage_configs_path or runtime.stage_configs_path
        return SubprocessOmniBackend.from_worker_script(
            python_executable=runtime.python_executable,
            worker_script=worker_script,
            model_id=self._model_ref or self._model_id,
            stage_configs_path=stage_configs_path,
            default_voice=self._default_voice or "neutral_female",
            env=runtime.env,
        )

    def unload(self) -> None:
        import asyncio

        if self._backend is not None:
            backend = self._backend
            self._backend = None
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(backend.close())
            except RuntimeError:
                try:
                    asyncio.run(backend.close())
                except Exception:
                    pass

        self._runtime = None
        self._tokenizer = None
        self._speech_request_cls = None
        self._subprocess_only = False
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Voxtral TTS adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

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
            raise RuntimeError("Voxtral TTS model is not loaded — call load() first")

        if not text or not text.strip():
            return

        if reference_audio is not None or reference_text is not None:
            raise NotImplementedError(
                "Voxtral TTS reference-audio cloning is not yet released in the current "
                "vLLM-Omni path; use a preset voice such as 'neutral_female'"
            )

        voice_id = voice or self._default_voice or "neutral_female"
        _ = max(0.5, min(speed, 2.0))

        if self._backend is None:
            raise RuntimeError("Voxtral TTS model is not loaded — call load() first")

        async for chunk in self._backend.generate(text=text, voice=voice_id):
            yield chunk

    def list_voices(self) -> list[VoiceInfo]:
        if self._runtime is not None:
            getter = getattr(self._runtime, "get_supported_voices", None)
            if callable(getter):
                voices = getter()
                normalized: list[VoiceInfo] = []
                for voice in voices or []:
                    if isinstance(voice, str):
                        normalized.append(VoiceInfo(id=voice, name=voice))
                        continue
                    if isinstance(voice, dict):
                        voice_id = voice.get("id") or voice.get("name") or voice.get("voice")
                        if voice_id:
                            normalized.append(
                                VoiceInfo(
                                    id=str(voice_id),
                                    name=str(voice.get("name") or voice_id),
                                    description=voice.get("description"),
                                )
                            )
                        continue
                    voice_id = (
                        getattr(voice, "id", None)
                        or getattr(voice, "name", None)
                        or getattr(voice, "voice", None)
                    )
                    if voice_id:
                        normalized.append(
                            VoiceInfo(
                                id=str(voice_id),
                                name=str(getattr(voice, "name", voice_id)),
                                description=getattr(voice, "description", None),
                            )
                        )
                if normalized:
                    return normalized
        return list(PRESET_VOICES)

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        del kwargs
        return recommended_voxtral_tts_vram_bytes()
