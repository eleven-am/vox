from __future__ import annotations

import asyncio
import base64
import json
import logging
import subprocess
import threading
import time
from collections.abc import AsyncIterator
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
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
from vox_voxtral.runtime import ensure_voxtral_tts_runtime, recommended_voxtral_tts_vram_bytes

logger = logging.getLogger(__name__)

VOXTRAL_TTS_SAMPLE_RATE = 24_000

PRESET_VOICES: list[VoiceInfo] = [
    VoiceInfo(id="neutral_female", name="neutral_female", description="Official Voxtral preset voice"),
    VoiceInfo(id="cheerful_female", name="cheerful_female", description="Official Voxtral preset voice"),
    VoiceInfo(id="casual_male", name="casual_male", description="Official Voxtral preset voice"),
]

SUPPORTED_LANGUAGES = ("en", "fr", "es", "de", "it", "pt", "nl", "ar", "hi")


def _select_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    if device in ("mps", "auto") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


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
        self._runtime: Any | None = None
        self._sampling_params: list[Any] = []
        self._tokenizer: Any | None = None
        self._speech_request_cls: Any | None = None
        self._worker_proc: subprocess.Popen[str] | None = None
        self._worker_lock = threading.Lock()
        self._subprocess_only = False
        self._loaded = False
        self._model_id: str = ""
        self._device: str = "cpu"
        self._default_voice: str | None = None

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
        self._default_voice = kwargs.pop("default_voice", None)
        explicit_stage_configs_path = kwargs.pop("_stage_configs_path", None)
        self._device = _select_device(device)
        if self._device != "cuda":
            raise RuntimeError(
                "Voxtral TTS on Vox requires CUDA + vLLM-Omni; CPU and MPS are not supported"
            )

        log_stats = bool(kwargs.pop("log_stats", False))

        try:
            stage_configs_path = _resolve_stage_configs_path(explicit_stage_configs_path)
            AsyncOmni, SamplingParams, SpeechRequest, MistralTokenizer = _load_voxtral_tts_runtime()
            self._speech_request_cls = SpeechRequest

            logger.info("Loading Voxtral TTS model: %s (device=%s)", self._model_id, self._device)
            start = time.perf_counter()

            self._tokenizer = self._load_tokenizer(MistralTokenizer)
            self._runtime = AsyncOmni(
                model=self._model_id,
                stage_configs_path=stage_configs_path,
                log_stats=log_stats,
            )
            sampling_params = SamplingParams(max_tokens=2500)
            self._sampling_params = [sampling_params, sampling_params]

            elapsed = time.perf_counter() - start
            logger.info("Voxtral TTS model loaded in %.2fs", elapsed)
            self._subprocess_only = False
            self._loaded = True
            return
        except Exception as exc:
            logger.warning(
                "Falling back to subprocess-isolated Voxtral TTS runtime for %s: %s",
                self._model_id,
                exc,
            )

        self._start_worker(explicit_stage_configs_path=explicit_stage_configs_path)
        self._subprocess_only = True
        self._loaded = True

    def _load_tokenizer(self, mistral_tokenizer_cls: Any) -> Any:
        model_path = self._model_id
        model_dir = None
        try:
            from pathlib import Path

            candidate = Path(model_path)
            if candidate.is_dir() and (candidate / "tekken.json").is_file():
                model_dir = candidate
        except Exception:  # pragma: no cover - defensive
            model_dir = None

        if model_dir is not None:
            return mistral_tokenizer_cls.from_file(str(model_dir / "tekken.json"))
        return mistral_tokenizer_cls.from_hf_hub(self._model_id)

    def unload(self) -> None:
        shutdown = getattr(self._runtime, "shutdown", None)
        if callable(shutdown):
            shutdown()
        if self._worker_proc is not None:
            try:
                if self._worker_proc.stdin is not None:
                    self._worker_proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    self._worker_proc.stdin.flush()
            except Exception:
                pass
            try:
                self._worker_proc.terminate()
                self._worker_proc.wait(timeout=10)
            except Exception:
                self._worker_proc.kill()
            self._worker_proc = None
        self._runtime = None
        self._sampling_params = []
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

        if self._subprocess_only:
            payload = await asyncio.to_thread(
                self._worker_request,
                text=text,
                voice=voice_id,
            )
            audio = base64.b64decode(payload["audio_b64"])
            yield SynthesizeChunk(
                audio=audio,
                sample_rate=int(payload.get("sample_rate", VOXTRAL_TTS_SAMPLE_RATE)),
                is_final=False,
            )
            yield SynthesizeChunk(
                audio=b"",
                sample_rate=int(payload.get("sample_rate", VOXTRAL_TTS_SAMPLE_RATE)),
                is_final=True,
            )
            return

        if self._runtime is None or self._tokenizer is None:
            raise RuntimeError("Voxtral TTS model is not loaded — call load() first")

        instruct_tokenizer = self._tokenizer.instruct_tokenizer
        if self._speech_request_cls is None:
            raise RuntimeError("Voxtral TTS speech request class is not initialized")
        tokenized = instruct_tokenizer.encode_speech_request(
            self._speech_request_cls(input=text, voice=voice_id)
        )
        inputs: dict[str, Any] = {
            "prompt_token_ids": tokenized.tokens,
            "additional_information": {"voice": [voice_id]},
        }

        accumulated_sample = 0
        chunk_idx = 0
        async for stage_output in self._runtime.generate(
            inputs,
            request_id=str(time.time_ns()),
            sampling_params_list=self._sampling_params,
        ):
            multimodal_output = getattr(stage_output, "multimodal_output", None)
            finished = bool(getattr(stage_output, "finished", False))
            if not multimodal_output or "audio" not in multimodal_output:
                continue

            audio_chunk = multimodal_output["audio"]
            audio_array = self._to_numpy_audio_chunk(audio_chunk, chunk_idx)
            if finished and accumulated_sample and len(audio_array) > accumulated_sample:
                audio_array = audio_array[accumulated_sample:]

            accumulated_sample += len(audio_array)
            chunk_idx += 1

            yield SynthesizeChunk(
                audio=audio_array.astype(np.float32, copy=False).tobytes(),
                sample_rate=VOXTRAL_TTS_SAMPLE_RATE,
                is_final=False,
            )

        yield SynthesizeChunk(
            audio=b"",
            sample_rate=VOXTRAL_TTS_SAMPLE_RATE,
            is_final=True,
        )

    @staticmethod
    def _to_numpy_audio_chunk(audio_chunk: Any, chunk_idx: int) -> NDArray[np.float32]:
        if isinstance(audio_chunk, list):
            if not audio_chunk:
                return np.asarray([], dtype=np.float32)
            audio_chunk = audio_chunk[chunk_idx] if chunk_idx < len(audio_chunk) else audio_chunk[-1]

        if hasattr(audio_chunk, "detach"):
            audio_chunk = audio_chunk.float().detach().cpu().numpy()
            return np.asarray(audio_chunk, dtype=np.float32)

        if isinstance(audio_chunk, np.ndarray):
            return audio_chunk.astype(np.float32, copy=False)

        return np.asarray(audio_chunk, dtype=np.float32)

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

    def _start_worker(self, *, explicit_stage_configs_path: str | None) -> None:
        worker_script = str(Path(__file__).with_name("voxtral_tts_worker.py"))
        runtime = ensure_voxtral_tts_runtime()
        stage_configs_path = explicit_stage_configs_path or runtime.stage_configs_path

        self._worker_proc = subprocess.Popen(
            [
                runtime.python_executable,
                "-u",
                worker_script,
                "--model-id",
                self._model_id,
                "--stage-configs-path",
                stage_configs_path,
                "--default-voice",
                self._default_voice or "neutral_female",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=runtime.env,
        )
        startup_logs: list[str] = []
        while True:
            line = self._read_worker_line(allow_empty=True)
            if not line:
                stderr = ""
                if self._worker_proc.stderr is not None:
                    stderr = self._worker_proc.stderr.read()
                message = stderr or "\n".join(startup_logs) or "Failed to start Voxtral TTS worker"
                raise RuntimeError(message)

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                startup_logs.append(line.strip())
                continue

            if payload.get("status") == "ready":
                return

            stderr = ""
            if self._worker_proc.stderr is not None:
                stderr = self._worker_proc.stderr.read()
            message = payload.get("error") or stderr or "\n".join(startup_logs) or "Failed to start Voxtral TTS worker"
            raise RuntimeError(message)

    def _worker_request(self, *, text: str, voice: str) -> dict[str, Any]:
        if self._worker_proc is None or self._worker_proc.stdin is None:
            raise RuntimeError("Voxtral TTS worker is not running")

        with self._worker_lock:
            self._worker_proc.stdin.write(
                json.dumps(
                    {
                        "op": "synthesize",
                        "text": text,
                        "voice": voice,
                    }
                )
                + "\n"
            )
            self._worker_proc.stdin.flush()
            request_logs: list[str] = []
            while True:
                line = self._read_worker_line()
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    request_logs.append(line.strip())
                    continue

                if payload.get("status") == "ok":
                    return payload

                message = (
                    payload.get("error")
                    or "\n".join(log for log in request_logs if log)
                    or "Voxtral TTS worker request failed"
                )
                raise RuntimeError(message)

    def _read_worker_line(self, *, allow_empty: bool = False) -> str:
        if self._worker_proc is None or self._worker_proc.stdout is None:
            raise RuntimeError("Voxtral TTS worker is not running")

        line = self._worker_proc.stdout.readline()
        if line:
            return line
        if allow_empty:
            return ""

        stderr = ""
        if self._worker_proc.stderr is not None:
            stderr = self._worker_proc.stderr.read()
        raise RuntimeError(stderr or "Voxtral TTS worker exited unexpectedly")
