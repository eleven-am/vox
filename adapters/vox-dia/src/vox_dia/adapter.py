from __future__ import annotations

import asyncio
import importlib
import logging
import math
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vox.core.adapter import TTSAdapter
from vox.core.adapter_runtime import (
    activate_runtime_path,
    install_target_runtime_requirements,
    write_app_fallback_path,
)
from vox.core.adapter_runtime import (
    runtime_root as vox_runtime_root,
)
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesisParameterInfo, SynthesizeChunk, VoiceInfo
from vox.operations.errors import InvalidConfigError

logger = logging.getLogger(__name__)

DIA_SAMPLE_RATE = 44_100
_DEFAULT_MAX_NEW_TOKENS = 3_072
_DEFAULT_GUIDANCE_SCALE = 3.0
_DEFAULT_TEMPERATURE = 1.8
_DEFAULT_TOP_P = 0.90
_DEFAULT_TOP_K = 45
_DEFAULT_REFERENCE_PROMPT_SECONDS = 5.0
_DIA_TRANSFORMERS_SPEC = "transformers==4.57.6"
_DIA_RUNTIME_IMPORT = "transformers.models.dia.modeling_dia"
_DIA_RUNTIME_PACKAGES = (
    _DIA_TRANSFORMERS_SPEC,
    "sentencepiece>=0.2.0,<0.3",
    "tiktoken>=0.9.0,<1",
)
_SPEAKER_TAG_RE = re.compile(r"\[S[12]\]")


def _torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "Dia requires PyTorch at runtime. Use a Vox image/runtime with torch and CUDA support."
        ) from exc


def _runtime_root() -> Path:
    return vox_runtime_root() / "dia"


def _ensure_runtime_path() -> str:
    runtime_dir = _runtime_root()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_app_fallback_path(runtime_dir)
    return activate_runtime_path(runtime_dir, root=runtime_dir.parent)



def _load_transformers_runtime() -> tuple[Any, Any]:
    runtime_path = _ensure_runtime_path()
    if not _runtime_has_dia_support():
        _install_transformers_runtime()
        _clear_transformers_modules()
        _ensure_runtime_path()
    if not _runtime_has_dia_support():
        raise RuntimeError(
            "Dia requires Hugging Face Transformers with DiaForConditionalGeneration support. "
            f"The isolated Dia runtime at {runtime_path} could not expose the required symbols."
        )

    from transformers import AutoProcessor, DiaForConditionalGeneration

    return AutoProcessor, DiaForConditionalGeneration


def _clear_transformers_modules() -> None:
    for name in list(sys.modules):
        if name == "transformers" or name.startswith(("transformers.", "sentencepiece", "tiktoken", "regex")):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _install_transformers_runtime() -> None:
    runtime_path = _ensure_runtime_path()
    if not install_target_runtime_requirements(
        runtime_path,
        _DIA_RUNTIME_PACKAGES,
        no_deps=False,
        upgrade=True,
        timeout=900,
        install_runner=_run_install_command,
        context="Dia runtime install",
    ):
        raise RuntimeError(
            f"Failed to install Dia runtime package {_DIA_TRANSFORMERS_SPEC}."
        ) from None
    _clear_transformers_modules()
    _ensure_runtime_path()
    if not _runtime_has_dia_support():
        raise RuntimeError("Dia runtime install did not expose DiaForConditionalGeneration")


def _run_install_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _runtime_has_dia_support() -> bool:
    runtime_path = Path(_ensure_runtime_path()).resolve()
    try:
        dia_module = importlib.import_module(_DIA_RUNTIME_IMPORT)
        transformers = importlib.import_module("transformers")
    except Exception:
        logger.debug("Dia runtime probe failed", exc_info=True)
        return False
    if not _module_loaded_from_runtime(transformers, runtime_path):
        return False
    if not _module_loaded_from_runtime(dia_module, runtime_path):
        return False
    return hasattr(transformers, "DiaForConditionalGeneration") and hasattr(transformers, "AutoProcessor")


def _module_loaded_from_runtime(module: Any, runtime_path: Path) -> bool:
    candidate_paths: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        candidate_paths.append(Path(module_file))

    spec = getattr(module, "__spec__", None)
    spec_origin = getattr(spec, "origin", None)
    if spec_origin and spec_origin not in {"built-in", "frozen"}:
        candidate_paths.append(Path(spec_origin))
    search_locations = getattr(spec, "submodule_search_locations", None)
    if search_locations:
        candidate_paths.extend(Path(path) for path in search_locations)

    return any(_is_relative_to(path, runtime_path) for path in candidate_paths)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent)
    except (OSError, ValueError):
        return False
    return True


def _decode_dia_output(
    *,
    processor: Any,
    model: Any,
    inputs: Any,
    temp_path: Path,
    audio_prompt_len: Any | None,
    max_new_tokens: int,
    guidance_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[np.ndarray, int]:
    torch = _torch_module()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            guidance_scale=guidance_scale,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    if audio_prompt_len is not None:
        decoded = processor.batch_decode(output, audio_prompt_len=audio_prompt_len)
    else:
        decoded = processor.batch_decode(output)
    processor.save_audio(decoded, str(temp_path))

    import soundfile as sf

    audio, sample_rate = sf.read(str(temp_path), dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sample_rate)


def _dia_text_with_reference(text: str, reference_text: str | None) -> str:
    target = _ensure_speaker_tag(text.strip())
    if reference_text is None or not reference_text.strip():
        return target
    return f"{_ensure_speaker_tag(reference_text.strip())} {target}".strip()


def _ensure_speaker_tag(text: str) -> str:
    if not text:
        return text
    if _SPEAKER_TAG_RE.search(text):
        return text
    return f"[S1] {text}".strip()


def _reference_prompt_limit_seconds(params: dict[str, Any] | None) -> float:
    if not params:
        return _DEFAULT_REFERENCE_PROMPT_SECONDS
    raw = params.get("reference_prompt_seconds", _DEFAULT_REFERENCE_PROMPT_SECONDS)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_REFERENCE_PROMPT_SECONDS
    return max(1.0, min(value, 12.0))


def _trim_reference_text(reference_text: str | None, ratio: float) -> str | None:
    if reference_text is None or not reference_text.strip():
        return reference_text

    text = reference_text.strip()
    speaker_prefix = ""
    if text.startswith("[S1]") or text.startswith("[S2]"):
        speaker_prefix = text[:4]
        text = text[4:].strip()

    words = text.split()
    if not words:
        return speaker_prefix or reference_text
    keep = max(1, min(len(words), int(math.ceil(len(words) * ratio))))
    trimmed = " ".join(words[:keep]).strip()
    if speaker_prefix:
        return f"{speaker_prefix} {trimmed}".strip()
    return trimmed


def _trim_reference_prompt(
    *,
    reference_audio: NDArray[np.float32],
    reference_text: str | None,
    max_seconds: float,
    sample_rate: int = DIA_SAMPLE_RATE,
) -> tuple[NDArray[np.float32], str | None]:
    audio = np.asarray(reference_audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio, reference_text

    max_samples = max(1, int(sample_rate * max_seconds))
    if audio.size <= max_samples:
        return audio, reference_text

    ratio = max_samples / float(audio.size)
    return audio[:max_samples], _trim_reference_text(reference_text, ratio)


def _dia_inputs(
    *,
    processor: Any,
    text: str,
    device: str,
    reference_audio: NDArray[np.float32] | None,
    reference_text: str | None,
    params: dict[str, Any] | None = None,
) -> tuple[Any, Any | None]:
    if reference_audio is not None:
        audio = np.asarray(reference_audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise InvalidConfigError("Dia reference_audio is empty")
        if reference_text is None or not reference_text.strip():
            raise InvalidConfigError(
                "Dia voice cloning requires reference_text for the reference_audio prompt."
            )
        audio, reference_text = _trim_reference_prompt(
            reference_audio=audio,
            reference_text=reference_text,
            max_seconds=_reference_prompt_limit_seconds(params),
        )
        inputs = processor(
            text=[_dia_text_with_reference(text, reference_text)],
            audio=audio,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device) if hasattr(inputs, "to") else inputs
        prompt_len = processor.get_audio_prompt_len(inputs["decoder_attention_mask"])
        return inputs, prompt_len

    inputs = processor(text=[text], padding=True, return_tensors="pt")
    inputs = inputs.to(device) if hasattr(inputs, "to") else inputs
    return inputs, None


class DiaAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False
        self._model_id: str = ""
        self._model_ref: str = ""
        self._device: str = "cpu"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="dia-tts-torch",
            type=ModelType.TTS,
            architectures=("dia-tts-torch", "dia"),
            default_sample_rate=DIA_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=False,
            supports_voice_cloning=True,
            supported_languages=("en",),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        source = kwargs.pop("_source", None)
        self._model_id = source if source else model_path
        path = Path(model_path)
        self._model_ref = str(path) if path.exists() else self._model_id
        self._device = device
        if self._device != "cuda":
            raise RuntimeError(
                "Dia requires a CUDA-capable GPU. CPU execution is not supported by the official runtime."
            )
        _torch_module()

        AutoProcessor, DiaForConditionalGeneration = _load_transformers_runtime()

        logger.info("Loading Dia model: %s (device=%s)", self._model_ref, self._device)
        start = time.perf_counter()

        self._processor = AutoProcessor.from_pretrained(self._model_ref)
        self._model = DiaForConditionalGeneration.from_pretrained(self._model_ref).to(self._device)
        self._model.eval()

        elapsed = time.perf_counter() - start
        logger.info("Dia model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        self._model_ref = ""
        try:
            torch = _torch_module()
        except RuntimeError:
            torch = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Dia adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def prepare_runtime(self) -> None:
        _load_transformers_runtime()

    def validate_synthesis_request(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        if reference_audio is not None:
            audio = np.asarray(reference_audio, dtype=np.float32)
            if audio.size == 0:
                raise InvalidConfigError("Dia reference_audio is empty")
            if reference_text is None or not reference_text.strip():
                raise InvalidConfigError(
                    "Dia voice cloning requires reference_text for the reference_audio prompt."
                )
            return
        if reference_text is not None:
            raise InvalidConfigError(
                "Dia reference_text can only be used together with reference_audio."
            )

    def synthesis_parameters(self) -> tuple[SynthesisParameterInfo, ...]:
        return (
            SynthesisParameterInfo(
                name="max_new_tokens",
                type="integer",
                default=_DEFAULT_MAX_NEW_TOKENS,
                min_value=1,
                max_value=8192,
                description="Maximum number of Dia audio tokens generated for the request.",
            ),
            SynthesisParameterInfo(
                name="guidance_scale",
                type="number",
                default=_DEFAULT_GUIDANCE_SCALE,
                min_value=0.0,
                max_value=10.0,
                description="Classifier-free guidance scale used by Dia generation.",
            ),
            SynthesisParameterInfo(
                name="temperature",
                type="number",
                default=_DEFAULT_TEMPERATURE,
                min_value=0.0,
                max_value=3.0,
                description="Sampling temperature used by Dia generation.",
            ),
            SynthesisParameterInfo(
                name="top_p",
                type="number",
                default=_DEFAULT_TOP_P,
                min_value=0.0,
                max_value=1.0,
                description="Nucleus sampling probability used by Dia generation.",
            ),
            SynthesisParameterInfo(
                name="top_k",
                type="integer",
                default=_DEFAULT_TOP_K,
                min_value=0,
                max_value=200,
                description="Top-k sampling cutoff used by Dia generation.",
            ),
            SynthesisParameterInfo(
                name="reference_prompt_seconds",
                type="number",
                default=_DEFAULT_REFERENCE_PROMPT_SECONDS,
                min_value=1.0,
                max_value=12.0,
                description="Maximum cloned reference audio seconds used as Dia conditioning prompt.",
            ),
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
        params: dict[str, Any] | None = None,
    ) -> AsyncIterator[SynthesizeChunk]:
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("Dia model is not loaded — call load() first")

        self.validate_synthesis_request(
            voice=voice,
            language=language,
            reference_audio=reference_audio,
            reference_text=reference_text,
            params=params,
        )

        if not text or not text.strip():
            return

        inputs, audio_prompt_len = _dia_inputs(
            processor=self._processor,
            text=text,
            device=self._device,
            reference_audio=reference_audio,
            reference_text=reference_text,
            params=params,
        )
        params = dict(params or {})

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = Path(tmp.name)
            audio, sample_rate = await asyncio.to_thread(
                _decode_dia_output,
                processor=self._processor,
                model=self._model,
                inputs=inputs,
                temp_path=temp_path,
                audio_prompt_len=audio_prompt_len,
                max_new_tokens=int(params.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS)),
                guidance_scale=float(params.get("guidance_scale", _DEFAULT_GUIDANCE_SCALE)),
                temperature=float(params.get("temperature", _DEFAULT_TEMPERATURE)),
                top_p=float(params.get("top_p", _DEFAULT_TOP_P)),
                top_k=int(params.get("top_k", _DEFAULT_TOP_K)),
            )

            chunk_size = sample_rate * 2
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                yield SynthesizeChunk(
                    audio=chunk.tobytes(),
                    sample_rate=sample_rate,
                    is_final=False,
                )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        yield SynthesizeChunk(audio=b"", sample_rate=DIA_SAMPLE_RATE, is_final=True)

    def list_voices(self) -> list[VoiceInfo]:
        return []

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 10_000_000_000
