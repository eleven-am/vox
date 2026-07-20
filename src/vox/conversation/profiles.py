from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from vox.conversation.types import TurnPolicy

DEFAULT_TURN_PROFILE = "default"

TURN_PROFILE_ALIASES: dict[str, str] = {
    "browser": "browser_default",
    "web": "browser_default",
    "headphones": "headset",
    "loudspeaker": "speakerphone",
    "speaker": "speakerphone",
    "noisy": "noisy_room",
}

TURN_PROFILES: dict[str, TurnPolicy] = {
    DEFAULT_TURN_PROFILE: TurnPolicy(),
    "browser_default": TurnPolicy(
        min_interrupt_duration_ms=220,
        false_interruption_timeout_ms=1700,
        min_endpointing_delay_ms=350,
        speaking_interrupt_min_duration_ms=420,
        speaking_interrupt_min_words=2,
        self_echo_min_words=3,
        self_echo_min_overlap=0.72,
        aec_warmup_ms=900,
        backchannel_end_cooldown_ms=1600,
        vad_min_silence_ms=800,
    ),
    "headset": TurnPolicy(
        min_interrupt_duration_ms=180,
        false_interruption_timeout_ms=1200,
        min_endpointing_delay_ms=300,
        speaking_interrupt_min_duration_ms=250,
        speaking_interrupt_min_words=1,
        aec_warmup_ms=250,
        backchannel_end_cooldown_ms=900,
        vad_min_silence_ms=600,
    ),
    "speakerphone": TurnPolicy(
        min_interrupt_duration_ms=260,
        false_interruption_timeout_ms=1900,
        min_endpointing_delay_ms=450,
        speaking_interrupt_min_duration_ms=520,
        speaking_interrupt_min_words=2,
        self_echo_min_words=2,
        self_echo_min_overlap=0.82,
        aec_warmup_ms=1100,
        backchannel_end_cooldown_ms=1800,
    ),
    "noisy_room": TurnPolicy(
        min_interrupt_duration_ms=320,
        false_interruption_timeout_ms=2400,
        min_endpointing_delay_ms=700,
        speaking_interrupt_min_duration_ms=700,
        speaking_interrupt_min_words=2,
        self_echo_min_words=3,
        self_echo_min_overlap=0.78,
        aec_warmup_ms=900,
        backchannel_end_cooldown_ms=2200,
        vad_min_silence_ms=1200,
    ),
}

TURN_POLICY_FIELD_NAMES = tuple(field.name for field in fields(TurnPolicy))


def list_turn_profiles() -> tuple[str, ...]:
    return tuple(TURN_PROFILES.keys())


def normalize_turn_profile_name(name: str | None) -> str:
    candidate = (name or DEFAULT_TURN_PROFILE).strip().lower()
    if not candidate:
        return DEFAULT_TURN_PROFILE
    return TURN_PROFILE_ALIASES.get(candidate, candidate)


def resolve_turn_profile(name: str | None) -> tuple[str, TurnPolicy]:
    canonical = normalize_turn_profile_name(name)
    try:
        return canonical, TURN_PROFILES[canonical]
    except KeyError as exc:
        known = ", ".join(sorted(TURN_PROFILES))
        raise ValueError(f"unknown turn profile '{name}'; expected one of: {known}") from exc


def resolve_turn_policy(
    profile_name: str | None,
    overrides: dict[str, Any] | None = None,
) -> tuple[str, TurnPolicy]:
    canonical, base = resolve_turn_profile(profile_name)
    if not overrides:
        return canonical, base
    unknown = set(overrides) - set(TURN_POLICY_FIELD_NAMES)
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"unknown turn policy override(s): {joined}")
    return canonical, replace(base, **overrides)
