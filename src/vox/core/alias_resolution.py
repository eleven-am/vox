from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from vox.core.device_placement import runtime_profile_for_alias


class AliasResolutionKind(StrEnum):
    NONE = "none"
    FAMILY = "family"
    LEGACY_MODEL_REF = "legacy_model_ref"
    LEGACY_NAME = "legacy_name"


@dataclass(frozen=True)
class ModelAliasResolution:
    name: str
    tag: str
    kind: AliasResolutionKind
    original_name: str
    original_tag: str
    profile: str | None = None
    resolved_profile: str | None = None

    @property
    def rewritten(self) -> bool:
        return self.kind is not AliasResolutionKind.NONE


@dataclass(frozen=True)
class FamilyAliasPolicy:
    name: str
    profiles: Mapping[str, tuple[str, str]]


@dataclass(frozen=True)
class LegacyModelRefAliasPolicy:
    name: str
    tag: str
    target_name: str
    target_tag: str


@dataclass(frozen=True)
class LegacyNameAliasPolicy:
    name: str
    target_name: str


_IMPLICIT_MODEL_ALIASES: dict[str, dict[str, tuple[str, str]]] = {
    "parakeet": {
        "spark": ("parakeet-stt", "tdt-0.6b-v3"),
        "default": ("parakeet-stt", "tdt-0.6b-v3"),
    },
    "parakeet-stt": {
        "spark": ("parakeet-stt", "tdt-0.6b-v3"),
        "default": ("parakeet-stt", "tdt-0.6b-v3"),
    },
    "kokoro": {
        "spark": ("kokoro-tts", "v1.0"),
        "default": ("kokoro-tts", "v1.0"),
    },
    "kokoro-tts": {
        "spark": ("kokoro-tts", "v1.0"),
        "default": ("kokoro-tts", "v1.0"),
    },
    "whisper": {
        "spark": ("whisper-stt", "base.en"),
        "default": ("whisper-stt", "base.en"),
    },
    "whisper-stt": {
        "spark": ("whisper-stt", "base.en"),
        "default": ("whisper-stt", "base.en"),
    },
    "piper": {
        "spark": ("piper-tts", "en-us-lessac-medium"),
        "default": ("piper-tts", "en-us-lessac-medium"),
    },
    "piper-tts": {
        "spark": ("piper-tts", "en-us-lessac-medium"),
        "default": ("piper-tts", "en-us-lessac-medium"),
    },
    "openvoice": {
        "spark": ("openvoice-tts", "v1"),
        "default": ("openvoice-tts", "v1"),
    },
    "openvoice-tts": {
        "spark": ("openvoice-tts", "v1"),
        "default": ("openvoice-tts", "v1"),
    },
    "chatterbox": {
        "spark": ("chatterbox-tts-turbo", "0.1.7"),
        "default": ("chatterbox-tts-turbo", "0.1.7"),
    },
    "chatterbox-tts": {
        "spark": ("chatterbox-tts-turbo", "0.1.7"),
        "default": ("chatterbox-tts-turbo", "0.1.7"),
    },
    "chatterbox-multilingual": {
        "spark": ("chatterbox-tts-multilingual", "0.1.7"),
        "default": ("chatterbox-tts-multilingual", "0.1.7"),
    },
    "cosyvoice": {
        "spark": ("cosyvoice2-tts", "0.5b"),
        "default": ("cosyvoice2-tts", "0.5b"),
    },
    "cosyvoice2": {
        "spark": ("cosyvoice2-tts", "0.5b"),
        "default": ("cosyvoice2-tts", "0.5b"),
    },
    "cosyvoice-tts": {
        "spark": ("cosyvoice2-tts", "0.5b"),
        "default": ("cosyvoice2-tts", "0.5b"),
    },
    "orpheus": {
        "spark": ("orpheus-tts", "medium-3b"),
        "default": ("orpheus-tts", "medium-3b"),
    },
    "orpheus-tts": {
        "spark": ("orpheus-tts", "medium-3b"),
        "default": ("orpheus-tts", "medium-3b"),
    },
    "indextts": {
        "spark": ("indextts-tts", "2"),
        "default": ("indextts-tts", "2"),
    },
    "indextts-tts": {
        "spark": ("indextts-tts", "2"),
        "default": ("indextts-tts", "2"),
    },
    "spark": {
        "spark": ("spark-tts", "0.5b"),
        "default": ("spark-tts", "0.5b"),
    },
    "spark-tts": {
        "spark": ("spark-tts", "0.5b"),
        "default": ("spark-tts", "0.5b"),
    },
    "neutts": {
        "spark": ("neutts-air-tts", "air"),
        "default": ("neutts-air-tts", "air"),
    },
    "neutts-air": {
        "spark": ("neutts-air-tts", "air"),
        "default": ("neutts-air-tts", "air"),
    },
    "neutts-tts": {
        "spark": ("neutts-air-tts", "air"),
        "default": ("neutts-air-tts", "air"),
    },
    "dia": {
        "spark": ("dia-tts", "1.6b"),
        "default": ("dia-tts", "1.6b"),
    },
    "dia-tts": {
        "spark": ("dia-tts", "1.6b"),
        "default": ("dia-tts", "1.6b"),
    },
    "sesame": {
        "spark": ("sesame-tts", "csm-1b"),
        "default": ("sesame-tts", "csm-1b"),
    },
    "sesame-tts": {
        "spark": ("sesame-tts", "csm-1b"),
        "default": ("sesame-tts", "csm-1b"),
    },
    "speecht5-stt": {
        "spark": ("speecht5-stt", "base"),
        "default": ("speecht5-stt", "base"),
    },
    "speecht5-tts": {
        "spark": ("speecht5-tts", "base"),
        "default": ("speecht5-tts", "base"),
    },
    "vibevoice": {
        "spark": ("vibevoice-tts", "realtime-0.5b"),
        "default": ("vibevoice-tts", "realtime-0.5b"),
    },
    "vibevoice-tts": {
        "spark": ("vibevoice-tts", "realtime-0.5b"),
        "default": ("vibevoice-tts", "realtime-0.5b"),
    },
    "qwen3-stt": {
        "spark": ("qwen3-stt", "0.6b"),
        "default": ("qwen3-stt", "0.6b"),
    },
    "qwen3-tts": {
        "spark": ("qwen3-tts", "0.6b"),
        "default": ("qwen3-tts", "0.6b"),
    },
    "xtts": {
        "spark": ("xtts-tts", "v2"),
        "default": ("xtts-tts", "v2"),
    },
    "xtts-tts": {
        "spark": ("xtts-tts", "v2"),
        "default": ("xtts-tts", "v2"),
    },
    "voxtral-stt": {
        "spark": ("voxtral-stt", "mini-3b"),
        "default": ("voxtral-stt", "mini-3b"),
    },
    "voxtral-tts": {
        "spark": ("voxtral-tts", "4b"),
        "default": ("voxtral-tts", "4b"),
    },
}

_LEGACY_MODEL_REF_ALIASES: dict[tuple[str, str], tuple[str, str]] = {
    ("kokoro", "v1.0"): ("kokoro-tts", "v1.0"),
    ("parakeet", "tdt-0.6b"): ("parakeet-stt", "tdt-0.6b"),
    ("parakeet", "tdt-0.6b-v3"): ("parakeet-stt", "tdt-0.6b-v3"),
    ("parakeet", "tdt-1.1b"): ("parakeet-stt", "tdt-1.1b"),
    ("speecht5", "asr"): ("speecht5-stt", "base"),
    ("speecht5", "tts"): ("speecht5-tts", "base"),
    ("voxtral", "mini-3b"): ("voxtral-stt", "mini-3b"),
    ("voxtral", "small-24b"): ("voxtral-stt", "small-24b"),
    ("voxtral", "24b"): ("voxtral-stt", "24b"),
    ("voxtral", "realtime-4b"): ("voxtral-stt", "realtime-4b"),
    ("voxtral", "tts-4b"): ("voxtral-tts", "4b"),
}

_LEGACY_NAME_ALIASES: dict[str, str] = {
    "chatterbox": "chatterbox-tts-turbo",
    "chatterbox-multilingual": "chatterbox-tts-multilingual",
    "cosyvoice": "cosyvoice2-tts",
    "cosyvoice2": "cosyvoice2-tts",
    "dia": "dia-tts",
    "indextts": "indextts-tts",
    "neutts": "neutts-air-tts",
    "neutts-air": "neutts-air-tts",
    "openvoice": "openvoice-tts",
    "orpheus": "orpheus-tts",
    "piper": "piper-tts",
    "qwen3-asr": "qwen3-stt",
    "qwen3-tts": "qwen3-tts",
    "sesame": "sesame-tts",
    "spark": "spark-tts",
    "speecht5-stt": "speecht5-stt",
    "speecht5-tts": "speecht5-tts",
    "vibevoice": "vibevoice-tts",
    "vibevoice-tts": "vibevoice-tts",
    "voxtral-stt": "voxtral-stt",
    "voxtral-tts": "voxtral-tts",
    "whisper": "whisper-stt",
    "xtts": "xtts-tts",
}


def _runtime_profile() -> str:
    device = os.environ.get("VOX_DEVICE", "auto").strip().lower()
    return runtime_profile_for_alias(device_hint=device)


def resolve_family_alias(
    name: str, tag: str = "latest", *, explicit_tag: bool = False
) -> tuple[str, str]:
    resolution = resolve_model_alias(name, tag, explicit_tag=explicit_tag)
    return resolution.name, resolution.tag


def family_alias_policy() -> tuple[FamilyAliasPolicy, ...]:
    return tuple(
        FamilyAliasPolicy(
            name=name,
            profiles=MappingProxyType(dict(profiles)),
        )
        for name, profiles in sorted(_IMPLICIT_MODEL_ALIASES.items())
    )


def legacy_model_ref_alias_policy() -> tuple[LegacyModelRefAliasPolicy, ...]:
    return tuple(
        LegacyModelRefAliasPolicy(
            name=name,
            tag=tag,
            target_name=target_name,
            target_tag=target_tag,
        )
        for (name, tag), (target_name, target_tag)
        in sorted(_LEGACY_MODEL_REF_ALIASES.items())
    )


def legacy_name_alias_policy() -> tuple[LegacyNameAliasPolicy, ...]:
    return tuple(
        LegacyNameAliasPolicy(name=name, target_name=target_name)
        for name, target_name in sorted(_LEGACY_NAME_ALIASES.items())
    )


def resolve_model_alias(
    name: str, tag: str = "latest", *, explicit_tag: bool = False
) -> ModelAliasResolution:
    if not explicit_tag and tag == "latest":
        aliases = _IMPLICIT_MODEL_ALIASES.get(name)
        if aliases is not None:
            profile = _runtime_profile()
            resolved_profile = profile if profile in aliases else "default"
            resolved_name, resolved_tag = aliases[resolved_profile]
            return ModelAliasResolution(
                name=resolved_name,
                tag=resolved_tag,
                kind=AliasResolutionKind.FAMILY,
                original_name=name,
                original_tag=tag,
                profile=profile,
                resolved_profile=resolved_profile,
            )

    exact_alias = _LEGACY_MODEL_REF_ALIASES.get((name, tag))
    if exact_alias is not None:
        resolved_name, resolved_tag = exact_alias
        return ModelAliasResolution(
            name=resolved_name,
            tag=resolved_tag,
            kind=AliasResolutionKind.LEGACY_MODEL_REF,
            original_name=name,
            original_tag=tag,
        )

    resolved_name = _LEGACY_NAME_ALIASES.get(name, name)
    if resolved_name != name:
        return ModelAliasResolution(
            name=resolved_name,
            tag=tag,
            kind=AliasResolutionKind.LEGACY_NAME,
            original_name=name,
            original_tag=tag,
        )

    return ModelAliasResolution(
        name=name,
        tag=tag,
        kind=AliasResolutionKind.NONE,
        original_name=name,
        original_tag=tag,
    )


__all__ = [
    "AliasResolutionKind",
    "FamilyAliasPolicy",
    "LegacyModelRefAliasPolicy",
    "LegacyNameAliasPolicy",
    "ModelAliasResolution",
    "family_alias_policy",
    "legacy_model_ref_alias_policy",
    "legacy_name_alias_policy",
    "resolve_family_alias",
    "resolve_model_alias",
]
