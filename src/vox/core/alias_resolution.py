from __future__ import annotations

import os

from vox.core.device_placement import runtime_profile_for_alias


_IMPLICIT_MODEL_ALIASES: dict[str, dict[str, tuple[str, str]]] = {
    "parakeet": {
        "spark": ("parakeet-stt-nemo", "tdt-0.6b-v3"),
        "default": ("parakeet-stt-onnx", "tdt-0.6b-v3"),
    },
    "parakeet-stt": {
        "spark": ("parakeet-stt-nemo", "tdt-0.6b-v3"),
        "default": ("parakeet-stt-onnx", "tdt-0.6b-v3"),
    },
    "kokoro": {
        "spark": ("kokoro-tts-torch", "v1.0"),
        "default": ("kokoro-tts-onnx", "v1.0"),
    },
    "kokoro-tts": {
        "spark": ("kokoro-tts-torch", "v1.0"),
        "default": ("kokoro-tts-onnx", "v1.0"),
    },
    "whisper": {
        "spark": ("whisper-stt-ct2", "base.en"),
        "default": ("whisper-stt-ct2", "base.en"),
    },
    "whisper-stt": {
        "spark": ("whisper-stt-ct2", "base.en"),
        "default": ("whisper-stt-ct2", "base.en"),
    },
    "piper": {
        "spark": ("piper-tts-onnx", "en-us-lessac-medium"),
        "default": ("piper-tts-onnx", "en-us-lessac-medium"),
    },
    "piper-tts": {
        "spark": ("piper-tts-onnx", "en-us-lessac-medium"),
        "default": ("piper-tts-onnx", "en-us-lessac-medium"),
    },
    "openvoice": {
        "spark": ("openvoice-tts-torch", "v1"),
        "default": ("openvoice-tts-torch", "v1"),
    },
    "openvoice-tts": {
        "spark": ("openvoice-tts-torch", "v1"),
        "default": ("openvoice-tts-torch", "v1"),
    },
    "dia": {
        "spark": ("dia-tts-torch", "1.6b"),
        "default": ("dia-tts-torch", "1.6b"),
    },
    "dia-tts": {
        "spark": ("dia-tts-torch", "1.6b"),
        "default": ("dia-tts-torch", "1.6b"),
    },
    "sesame": {
        "spark": ("sesame-tts-torch", "csm-1b"),
        "default": ("sesame-tts-torch", "csm-1b"),
    },
    "sesame-tts": {
        "spark": ("sesame-tts-torch", "csm-1b"),
        "default": ("sesame-tts-torch", "csm-1b"),
    },
    "speecht5-stt": {
        "spark": ("speecht5-stt-torch", "base"),
        "default": ("speecht5-stt-torch", "base"),
    },
    "speecht5-tts": {
        "spark": ("speecht5-tts-torch", "base"),
        "default": ("speecht5-tts-torch", "base"),
    },
    "vibevoice": {
        "spark": ("vibevoice-tts-torch", "realtime-0.5b"),
        "default": ("vibevoice-tts-torch", "realtime-0.5b"),
    },
    "vibevoice-tts": {
        "spark": ("vibevoice-tts-torch", "realtime-0.5b"),
        "default": ("vibevoice-tts-torch", "realtime-0.5b"),
    },
    "qwen3-stt": {
        "spark": ("qwen3-stt-torch", "0.6b"),
        "default": ("qwen3-stt-torch", "0.6b"),
    },
    "qwen3-tts": {
        "spark": ("qwen3-tts-torch", "0.6b"),
        "default": ("qwen3-tts-torch", "0.6b"),
    },
    "xtts": {
        "spark": ("xtts-tts-torch", "v2"),
        "default": ("xtts-tts-torch", "v2"),
    },
    "xtts-tts": {
        "spark": ("xtts-tts-torch", "v2"),
        "default": ("xtts-tts-torch", "v2"),
    },
    "voxtral-stt": {
        "spark": ("voxtral-stt-torch", "mini-3b"),
        "default": ("voxtral-stt-torch", "mini-3b"),
    },
    "voxtral-tts": {
        "spark": ("voxtral-tts-vllm", "4b"),
        "default": ("voxtral-tts-vllm", "4b"),
    },
}

_LEGACY_MODEL_REF_ALIASES: dict[tuple[str, str], tuple[str, str]] = {
    ("kokoro", "v1.0"): ("kokoro-tts-onnx", "v1.0"),
    ("kokoro", "v1.0-torch"): ("kokoro-tts-torch", "v1.0"),
    ("parakeet", "tdt-0.6b"): ("parakeet-stt-onnx", "tdt-0.6b"),
    ("parakeet", "tdt-0.6b-v3"): ("parakeet-stt-onnx", "tdt-0.6b-v3"),
    ("parakeet", "tdt-0.6b-v3-nemo"): ("parakeet-stt-nemo", "tdt-0.6b-v3"),
    ("parakeet", "tdt-0.6b-v3-cuda"): ("parakeet-stt-nemo", "tdt-0.6b-v3"),
    ("parakeet", "tdt-1.1b-nemo"): ("parakeet-stt-nemo", "tdt-1.1b"),
    ("parakeet", "tdt-1.1b-cuda"): ("parakeet-stt-nemo", "tdt-1.1b"),
    ("speecht5", "asr"): ("speecht5-stt-torch", "base"),
    ("speecht5", "tts"): ("speecht5-tts-torch", "base"),
    ("voxtral", "mini-3b"): ("voxtral-stt-torch", "mini-3b"),
    ("voxtral", "small-24b"): ("voxtral-stt-torch", "small-24b"),
    ("voxtral", "24b"): ("voxtral-stt-torch", "24b"),
    ("voxtral", "realtime-4b"): ("voxtral-stt-torch", "realtime-4b"),
    ("voxtral", "tts-4b"): ("voxtral-tts-vllm", "4b"),
}

_LEGACY_NAME_ALIASES: dict[str, str] = {
    "dia": "dia-tts-torch",
    "kokoro-torch": "kokoro-tts-torch",
    "openvoice": "openvoice-tts-torch",
    "parakeet-nemo": "parakeet-stt-nemo",
    "piper": "piper-tts-onnx",
    "qwen3-asr": "qwen3-stt-torch",
    "qwen3-tts": "qwen3-tts-torch",
    "sesame": "sesame-tts-torch",
    "speecht5-stt": "speecht5-stt-torch",
    "speecht5-tts": "speecht5-tts-torch",
    "vibevoice": "vibevoice-tts-torch",
    "vibevoice-tts": "vibevoice-tts-torch",
    "voxtral-stt": "voxtral-stt-torch",
    "voxtral-tts": "voxtral-tts-vllm",
    "whisper": "whisper-stt-ct2",
    "xtts": "xtts-tts-torch",
}


def _runtime_profile() -> str:
    device = os.environ.get("VOX_DEVICE", "auto").strip().lower()
    return runtime_profile_for_alias(device_hint=device)


def resolve_family_alias(
    name: str, tag: str = "latest", *, explicit_tag: bool = False
) -> tuple[str, str]:
    if not explicit_tag and tag == "latest":
        aliases = _IMPLICIT_MODEL_ALIASES.get(name)
        if aliases is not None:
            profile = _runtime_profile()
            return aliases.get(profile) or aliases["default"]

    exact_alias = _LEGACY_MODEL_REF_ALIASES.get((name, tag))
    if exact_alias is not None:
        return exact_alias

    resolved_name = _LEGACY_NAME_ALIASES.get(name, name)
    return resolved_name, tag


__all__ = ["resolve_family_alias"]
