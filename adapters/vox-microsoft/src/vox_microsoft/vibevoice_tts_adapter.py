# ruff: noqa: E402

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import torch
from numpy.typing import NDArray

from vox_microsoft._hf_compat import ensure_huggingface_hub_compat

ensure_huggingface_hub_compat()

from vox.core.adapter import TTSAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)

logger = logging.getLogger(__name__)

VIBEVOICE_SAMPLE_RATE = 24_000
VIBEVOICE_RUNTIME_PACKAGE = "vibevoice"
VIBEVOICE_RUNTIME_SPEC = "vibevoice[streamingtts] @ git+https://github.com/microsoft/VibeVoice.git@main"
VIBEVOICE_MIN_VERSIONS: dict[str, str] = {
    VIBEVOICE_RUNTIME_PACKAGE: "0.1.0",
}
VIBEVOICE_PINNED_VERSIONS: dict[str, str] = {
    "transformers": "4.51.3",
    "accelerate": "1.6.0",
    "huggingface_hub": "0.35.3",
    "tokenizers": "0.21.4",
}
VIBEVOICE_BOOTSTRAP_SPECS: dict[str, str] = {
    VIBEVOICE_RUNTIME_PACKAGE: VIBEVOICE_RUNTIME_SPEC,
    "diffusers": "diffusers",
    "PIL": "pillow",
    "transformers": "transformers==4.51.3",
    "accelerate": "accelerate==1.6.0",
    "huggingface_hub": "huggingface-hub==0.35.3",
    "tokenizers": "tokenizers==0.21.4",
}
VIBEVOICE_STREAMING_DEFAULT_VOICE = "en-Carter_man"
VIBEVOICE_STREAMING_VOICE_URL = (
    "https://raw.githubusercontent.com/microsoft/VibeVoice/main/"
    "demo/voices/streaming_model/{voice}.pt"
)
VIBEVOICE_STREAMING_VOICE_API_URL = (
    "https://api.github.com/repos/microsoft/VibeVoice/contents/"
    "demo/voices/streaming_model/{voice}.pt?ref=main"
)
VIBEVOICE_STREAMING_VOICE_ALIASES: dict[str, str] = {
    "default": VIBEVOICE_STREAMING_DEFAULT_VOICE,
    "alloy": VIBEVOICE_STREAMING_DEFAULT_VOICE,
}

_VRAM_ESTIMATES: dict[str, int] = {
    "0.5b": 2_000_000_000,
    "1.5b": 6_000_000_000,
}
VIBEVOICE_STREAMING_PRESET_VOICES: tuple[str, ...] = (
    "en-Carter_man",
    "en-Davis_man",
    "en-Emma_woman",
    "en-Frank_man",
    "en-Grace_woman",
    "en-Mike_man",
)


def _select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def _estimate_vram(model_id: str) -> int:
    lower = model_id.lower()
    for size_key, vram in _VRAM_ESTIMATES.items():
        if size_key in lower:
            return vram
    return _VRAM_ESTIMATES["0.5b"]


def _normalize_streaming_voice_id(voice: str | None) -> str:
    voice_key = (voice or VIBEVOICE_STREAMING_DEFAULT_VOICE).strip() or VIBEVOICE_STREAMING_DEFAULT_VOICE
    return VIBEVOICE_STREAMING_VOICE_ALIASES.get(voice_key.lower(), voice_key)


def _download_streaming_prompt(url: str, target: Path, *, accept: str | None = None) -> None:
    headers = {"User-Agent": "vox-runtime/1.0"}
    if accept:
        headers["Accept"] = accept
    request = Request(url, headers=headers)
    with urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in version.split("."):
        digits = "".join(character for character in token if character.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def _runtime_root() -> Path:
    vox_home = Path(os.environ.get("VOX_HOME", str(Path.home() / ".vox")))
    return vox_home / "runtime" / "vibevoice"


def _ensure_runtime_path() -> str:
    runtime_path = str(_runtime_root())
    _runtime_root().mkdir(parents=True, exist_ok=True)
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    return runtime_path


def _runtime_dist_version(dist_name: str) -> str | None:
    normalized = dist_name.replace("-", "_")
    candidates = sorted(_runtime_root().glob(f"{normalized}-*.dist-info"))
    if not candidates:
        return None
    metadata_file = candidates[-1] / "METADATA"
    if metadata_file.is_file():
        for line in metadata_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("Version: "):
                return line.split("Version: ", 1)[1].strip()
    suffix = candidates[-1].name.removeprefix(f"{normalized}-").removesuffix(".dist-info")
    return suffix or None


def _runtime_has_package_path(package_name: str) -> bool:
    runtime_root = _runtime_root()
    package_paths = {
        "vibevoice": [runtime_root / "vibevoice"],
        "diffusers": [runtime_root / "diffusers"],
        "PIL": [runtime_root / "PIL"],
        "transformers": [runtime_root / "transformers"],
        "accelerate": [runtime_root / "accelerate"],
        "huggingface_hub": [runtime_root / "huggingface_hub"],
        "tokenizers": [runtime_root / "tokenizers", runtime_root / "tokenizers.libs"],
    }
    return any(path.exists() for path in package_paths.get(package_name, [runtime_root / package_name]))

def _clear_runtime_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "vibevoice" or module_name.startswith(
            (
                "vibevoice.",
                "diffusers.",
                "PIL.",
                "transformers.",
                "huggingface_hub.",
                "accelerate.",
                "tokenizers.",
            )
        ) or module_name in {
            "diffusers",
            "PIL",
            "transformers",
            "huggingface_hub",
            "accelerate",
            "tokenizers",
        }:
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _require_runtime() -> None:
    _ensure_runtime_path()
    problems: list[str] = []
    for package_name, minimum_version in VIBEVOICE_MIN_VERSIONS.items():
        installed_version = _runtime_dist_version(package_name)
        if installed_version is None:
            problems.append(f"{package_name} is not installed")
            continue
        if not _runtime_has_package_path(package_name):
            problems.append(f"{package_name} runtime files are missing")
            continue
        if _version_tuple(installed_version) < _version_tuple(minimum_version):
            problems.append(
                f"{package_name}>={minimum_version} required (found {installed_version})"
            )
    for package_name, required_version in VIBEVOICE_PINNED_VERSIONS.items():
        installed_version = _runtime_dist_version(package_name)
        if installed_version is None:
            problems.append(f"{package_name} is not installed")
            continue
        if not _runtime_has_package_path(package_name):
            problems.append(f"{package_name} runtime files are missing")
            continue
        if _version_tuple(installed_version) != _version_tuple(required_version):
            problems.append(
                f"{package_name}=={required_version} required (found {installed_version})"
            )

    if problems:
        details = "; ".join(problems)
        required_versions = ", ".join(
            [f"{name}>={version}" for name, version in VIBEVOICE_MIN_VERSIONS.items()]
            + [f"{name}=={version}" for name, version in VIBEVOICE_PINNED_VERSIONS.items()]
        )
        raise RuntimeError(
            "VibeVoice requires the community vibevoice runtime package and exact pinned dependencies "
            f"({required_versions}). "
            "Install the community VibeVoice codebase before pulling or serving this model. "
            f"Problems: {details}"
        )


def _bootstrap_runtime() -> None:
    runtime_path = _ensure_runtime_path()
    packages_to_install: list[str] = []
    for package_name, _package_spec in VIBEVOICE_BOOTSTRAP_SPECS.items():
        if not _runtime_has_package_path(package_name):
            packages_to_install.append(package_name)
            continue
        minimum_version = VIBEVOICE_MIN_VERSIONS.get(package_name)
        runtime_version = _runtime_dist_version(package_name)
        if runtime_version is None:
            packages_to_install.append(package_name)
            continue
        if minimum_version is not None and _version_tuple(runtime_version) < _version_tuple(minimum_version):
            packages_to_install.append(package_name)
            continue
        required_version = VIBEVOICE_PINNED_VERSIONS.get(package_name)
        if required_version is not None and _version_tuple(runtime_version) != _version_tuple(required_version):
            packages_to_install.append(package_name)
            continue
    if not packages_to_install:
        return

    installers = [
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            runtime_path,
            "--no-deps",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            runtime_path,
            "--no-deps",
        ],
    ]
    for package_name in packages_to_install:
        package_spec = VIBEVOICE_BOOTSTRAP_SPECS[package_name]
        for installer in installers:
            cmd = [*installer, package_spec]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0 and _runtime_has_package_path(package_name):
                logger.info(
                    "Bootstrapped VibeVoice runtime package into %s: %s",
                    runtime_path,
                    package_spec,
                )
                break
            logger.warning("%s failed: %s", " ".join(installer), result.stderr)
        else:
            raise RuntimeError(
                f"VibeVoice runtime package is missing and could not be bootstrapped: {package_name}. "
                "Install the community VibeVoice codebase before pulling or serving this model."
            )

    _clear_runtime_modules()
    _ensure_runtime_path()


def _prime_runtime(model_id: str) -> None:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    @contextmanager
    def ignore_duplicate_registrations() -> Any:
        original_config_register = AutoConfig.register
        original_model_register = AutoModel.register
        original_causal_lm_register = AutoModelForCausalLM.register

        def _wrap_register(register_fn: Any) -> Any:
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                try:
                    return register_fn(*args, **kwargs)
                except ValueError as exc:
                    message = str(exc)
                    if "already used by a Transformers model" not in message:
                        raise
                    logger.info("Ignoring duplicate VibeVoice Transformers registration: %s", message)
                    return None

            return _wrapped

        AutoConfig.register = _wrap_register(original_config_register)
        AutoModel.register = _wrap_register(original_model_register)
        AutoModelForCausalLM.register = _wrap_register(original_causal_lm_register)
        try:
            yield
        finally:
            AutoConfig.register = original_config_register
            AutoModel.register = original_model_register
            AutoModelForCausalLM.register = original_causal_lm_register

    lower = model_id.lower()
    is_streaming = "realtime" in lower or "streaming" in lower
    module_name = (
        "vibevoice.modular.modeling_vibevoice_streaming_inference"
        if is_streaming
        else "vibevoice.modular.modeling_vibevoice"
    )
    config_module_name = (
        "vibevoice.modular.configuration_vibevoice_streaming"
        if is_streaming
        else "vibevoice.modular.configuration_vibevoice"
    )
    with ignore_duplicate_registrations():
        config_module = importlib.import_module(config_module_name)
        config_class = config_module.VibeVoiceStreamingConfig if is_streaming else config_module.VibeVoiceConfig
        AutoConfig.register(config_class.model_type, config_class, exist_ok=True)
        importlib.import_module(module_name)


class VibeVoiceTTSAdapter(TTSAdapter):

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False
        self._model_id: str = ""
        self._device: str = "cpu"
        self._streaming_prompt_cache: dict[str, Any] = {}

    def _streaming_prompt_dir(self) -> Path:
        vox_home = Path(os.environ.get("VOX_HOME", str(Path.home() / ".vox")))
        return vox_home / "runtime" / "vibevoice" / "voices" / "streaming_model"

    def _ensure_streaming_prompt_file(self, voice: str | None) -> Path:
        voice_key = _normalize_streaming_voice_id(voice)
        prompt_dir = self._streaming_prompt_dir()
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = prompt_dir / f"{voice_key}.pt"
        if not prompt_path.exists():
            errors: list[str] = []
            for url, accept in (
                (VIBEVOICE_STREAMING_VOICE_URL.format(voice=voice_key), None),
                (
                    VIBEVOICE_STREAMING_VOICE_API_URL.format(voice=voice_key),
                    "application/vnd.github.raw",
                ),
            ):
                logger.info("Downloading VibeVoice streaming prompt %s from %s", voice_key, url)
                try:
                    _download_streaming_prompt(url, prompt_path, accept=accept)
                    break
                except Exception as exc:
                    prompt_path.unlink(missing_ok=True)
                    errors.append(f"{url}: {exc}")
            else:
                raise RuntimeError(
                    "Failed to download VibeVoice streaming prompt "
                    f"{voice_key}: {'; '.join(errors)}"
                )
        return prompt_path

    def _load_streaming_prompt(self, voice: str | None) -> Any:
        voice_key = _normalize_streaming_voice_id(voice)
        if voice_key not in self._streaming_prompt_cache:
            prompt_path = self._ensure_streaming_prompt_file(voice_key)
            target_device = self._device if self._device != "cpu" else "cpu"
            self._streaming_prompt_cache[voice_key] = torch.load(
                prompt_path,
                map_location=target_device,
                weights_only=False,
            )
        return self._streaming_prompt_cache[voice_key]

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="vibevoice-tts-torch",
            type=ModelType.TTS,
            architectures=("vibevoice-tts-torch", "vibevoice", "vibevoice-realtime"),
            default_sample_rate=VIBEVOICE_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),

            supports_streaming=False,
            supports_voice_cloning=False,
            supported_languages=("en",),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        source = kwargs.pop("_source", None)
        self._model_id = source if source else model_path
        self._device = device
        model_ref = str(Path(model_path)) if Path(model_path).exists() else self._model_id
        dtype = _select_dtype(self._device)
        _bootstrap_runtime()
        _require_runtime()
        _prime_runtime(self._model_id)

        lower = self._model_id.lower()
        is_streaming = "realtime" in lower or "streaming" in lower
        processor_module_name = (
            "vibevoice.processor.vibevoice_streaming_processor"
            if is_streaming
            else "vibevoice.processor.vibevoice_processor"
        )
        model_module_name = (
            "vibevoice.modular.modeling_vibevoice_streaming_inference"
            if is_streaming
            else "vibevoice.modular.modeling_vibevoice"
        )
        processor_module = importlib.import_module(processor_module_name)
        model_module = importlib.import_module(model_module_name)
        processor_class = (
            processor_module.VibeVoiceStreamingProcessor if is_streaming else processor_module.VibeVoiceProcessor
        )
        model_class = (
            model_module.VibeVoiceStreamingForConditionalGenerationInference
            if is_streaming
            else model_module.VibeVoiceForConditionalGeneration
        )

        attn_implementation = "flash_attention_2" if self._device == "cuda" else "sdpa"
        device_map: str | None = self._device if self._device in ("cuda", "cpu") else None

        logger.info("Loading VibeVoice TTS model: %s (device=%s, dtype=%s)", model_ref, self._device, dtype)
        start = time.perf_counter()

        self._processor = processor_class.from_pretrained(model_ref)
        try:
            self._model = model_class.from_pretrained(
                model_ref,
                torch_dtype=dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
        except Exception:
            if attn_implementation != "flash_attention_2":
                raise
            logger.info("Falling back to SDPA for VibeVoice model load")
            self._model = model_class.from_pretrained(
                model_ref,
                torch_dtype=dtype,
                device_map=device_map,
                attn_implementation="sdpa",
            )

        if self._device == "mps":
            self._model = self._model.to("mps")
        self._model.eval()

        if is_streaming and hasattr(self._model, "set_ddpm_inference_steps"):
            self._model.set_ddpm_inference_steps(num_steps=5)

        elapsed = time.perf_counter() - start
        logger.info("VibeVoice TTS model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        self._streaming_prompt_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VibeVoice TTS adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @staticmethod
    def _coerce_audio_array(output: Any) -> NDArray[np.float32]:
        candidate = getattr(output, "speech_outputs", None)
        if candidate is None:
            candidate = getattr(output, "audio", None)
        if candidate is None:
            candidate = getattr(output, "waveform", None)
        if candidate is None:
            candidate = output[0] if isinstance(output, (list, tuple)) else output

        if isinstance(candidate, (list, tuple)):
            if not candidate:
                return np.zeros(0, dtype=np.float32)
            candidate = candidate[0]

        if isinstance(candidate, (list, tuple)):
            parts: list[NDArray[np.float32]] = []
            for part in candidate:
                if part is None:
                    continue
                if hasattr(part, "detach"):
                    part = part.float().detach().cpu().numpy()
                array = np.asarray(part, dtype=np.float32).reshape(-1)
                if array.size:
                    parts.append(array)
            if not parts:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(parts).astype(np.float32, copy=False)

        if hasattr(candidate, "detach"):
            candidate = candidate.float().detach().cpu().numpy()

        return np.asarray(candidate, dtype=np.float32).reshape(-1)

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
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("VibeVoice TTS model is not loaded — call load() first")

        if not text or not text.strip():
            return

        script = text.strip()
        if "speaker" not in script.lower():
            script = f"Speaker 1: {script}"

        if "streaming" in self._model_id.lower() or "realtime" in self._model_id.lower():
            cached_prompt = self._load_streaming_prompt(voice)
            inputs = self._processor.process_input_with_cached_prompt(
                text=script,
                cached_prompt=cached_prompt,
                padding=True,
                truncation=False,
                return_tensors="pt",
                return_attention_mask=True,
            )
        else:
            voice_samples = [reference_audio] if reference_audio is not None else None
            inputs = self._processor(
                text=script,
                voice_samples=voice_samples,
                padding=True,
                truncation=False,
                return_tensors="pt",
                return_attention_mask=True,
            )

        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                tokenizer=getattr(self._processor, "tokenizer", None),
                generation_config={"do_sample": False},
                is_prefill=reference_audio is not None,
                cfg_scale=1.0,
                all_prefilled_outputs=(
                    deepcopy(cached_prompt)
                    if "streaming" in self._model_id.lower() or "realtime" in self._model_id.lower()
                    else inputs.get("speech_tensors")
                ),
                verbose=False,
            )

        audio = self._coerce_audio_array(output)

        chunk_size = VIBEVOICE_SAMPLE_RATE * 2
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            yield SynthesizeChunk(
                audio=chunk.tobytes(),
                sample_rate=VIBEVOICE_SAMPLE_RATE,
                is_final=False,
            )

        yield SynthesizeChunk(
            audio=b"",
            sample_rate=VIBEVOICE_SAMPLE_RATE,
            is_final=True,
        )

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id="default",
                name="default",
                description=f"Alias for {VIBEVOICE_STREAMING_DEFAULT_VOICE}",
            ),
            *[
                VoiceInfo(
                    id=voice_id,
                    name=voice_id,
                    description="Official VibeVoice streaming prompt",
                )
                for voice_id in VIBEVOICE_STREAMING_PRESET_VOICES
            ],
        ]

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        model_id = kwargs.get("_source") or kwargs.get("model_id") or self._model_id
        return _estimate_vram(str(model_id))
