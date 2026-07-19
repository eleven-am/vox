from __future__ import annotations

import dataclasses

import pytest

from vox.core.alias_resolution import (
    AliasResolutionKind,
    family_alias_policy,
    legacy_model_ref_alias_policy,
    legacy_name_alias_policy,
    resolve_family_alias,
    resolve_model_alias,
)


class TestBareNameResolution:
    def test_bare_name_resolves_family_default(self):
        assert resolve_family_alias("parakeet") == ("parakeet-stt", "tdt-0.6b-v3")
        assert resolve_family_alias("kokoro") == ("kokoro-tts", "v1.0")
        assert resolve_family_alias("voxtral-stt") == ("voxtral-stt", "mini-3b")
        assert resolve_family_alias("voxtral-tts") == ("voxtral-tts", "4b")

    def test_bare_name_resolution_reports_family_alias_metadata(self):
        resolution = resolve_model_alias("parakeet")

        assert resolution.kind is AliasResolutionKind.FAMILY
        assert resolution.rewritten is True
        assert resolution.original_name == "parakeet"
        assert resolution.original_tag == "latest"
        assert (resolution.name, resolution.tag) == ("parakeet-stt", "tdt-0.6b-v3")

    def test_all_bare_family_names_resolve(self):
        assert resolve_family_alias("whisper") == ("whisper-stt", "base.en")
        assert resolve_family_alias("whisper-stt") == ("whisper-stt", "base.en")
        assert resolve_family_alias("piper") == ("piper-tts", "en-us-lessac-medium")
        assert resolve_family_alias("openvoice") == ("openvoice-tts", "v1")
        assert resolve_family_alias("chatterbox") == ("chatterbox-tts-turbo", "0.1.7")
        assert resolve_family_alias("chatterbox-multilingual") == (
            "chatterbox-tts-multilingual",
            "0.1.7",
        )
        assert resolve_family_alias("cosyvoice") == ("cosyvoice2-tts", "0.5b")
        assert resolve_family_alias("cosyvoice2") == ("cosyvoice2-tts", "0.5b")
        assert resolve_family_alias("orpheus") == ("orpheus-tts", "medium-3b")
        assert resolve_family_alias("indextts") == ("indextts-tts", "2")
        assert resolve_family_alias("spark") == ("spark-tts", "0.5b")
        assert resolve_family_alias("neutts") == ("neutts-air-tts", "air")
        assert resolve_family_alias("neutts-air") == ("neutts-air-tts", "air")
        assert resolve_family_alias("dia") == ("dia-tts", "1.6b")
        assert resolve_family_alias("sesame") == ("sesame-tts", "csm-1b")
        assert resolve_family_alias("speecht5-stt") == ("speecht5-stt", "base")
        assert resolve_family_alias("speecht5-tts") == ("speecht5-tts", "base")
        assert resolve_family_alias("vibevoice") == ("vibevoice-tts", "realtime-0.5b")
        assert resolve_family_alias("qwen3-stt") == ("qwen3-stt", "0.6b")
        assert resolve_family_alias("qwen3-tts") == ("qwen3-tts", "0.6b")
        assert resolve_family_alias("xtts") == ("xtts-tts", "v2")


class TestExplicitTags:
    def test_explicit_latest_is_not_rewritten(self):
        assert resolve_family_alias("parakeet", "latest", explicit_tag=True) == (
            "parakeet",
            "latest",
        )

    def test_explicit_legacy_pair_still_rewrites(self):
        assert resolve_family_alias("kokoro", "v1.0", explicit_tag=True) == (
            "kokoro-tts",
            "v1.0",
        )


class TestLegacyAliasRewrite:
    def test_legacy_model_ref_pairs_rewrite_to_canonical(self):
        assert resolve_family_alias("kokoro", "v1.0") == ("kokoro-tts", "v1.0")
        # The explicit-backend legacy ref is a hard cutoff: it no longer resolves,
        # so it 404s at pull instead of silently picking the wrong backend.
        assert resolve_family_alias("kokoro", "v1.0-torch") == ("kokoro", "v1.0-torch")
        assert resolve_family_alias("parakeet", "tdt-0.6b") == (
            "parakeet-stt",
            "tdt-0.6b",
        )
        # The explicit-backend parakeet legacy refs are a hard cutoff: they no longer
        # rewrite and pass through unchanged to 404 at pull.
        assert resolve_family_alias("parakeet", "tdt-0.6b-v3-cuda") == (
            "parakeet",
            "tdt-0.6b-v3-cuda",
        )
        assert resolve_family_alias("parakeet", "tdt-1.1b-cuda") == (
            "parakeet",
            "tdt-1.1b-cuda",
        )
        assert resolve_family_alias("speecht5", "asr") == ("speecht5-stt", "base")
        assert resolve_family_alias("speecht5", "tts") == ("speecht5-tts", "base")
        assert resolve_family_alias("voxtral", "tts-4b") == ("voxtral-tts", "4b")
        assert resolve_family_alias("voxtral", "24b") == ("voxtral-stt", "24b")

    def test_legacy_name_aliases_rewrite_canonical_with_tag_passthrough(self):
        assert resolve_family_alias("qwen3-asr", "0.6b", explicit_tag=True) == (
            "qwen3-stt",
            "0.6b",
        )
        # The legacy "parakeet-nemo" name was removed as a hard cutoff: it no longer
        # rewrites and passes through unchanged.
        assert resolve_family_alias("parakeet-nemo", "tdt-1.1b", explicit_tag=True) == (
            "parakeet-nemo",
            "tdt-1.1b",
        )

    def test_legacy_model_ref_resolution_reports_legacy_metadata(self):
        resolution = resolve_model_alias("voxtral", "tts-4b", explicit_tag=True)

        assert resolution.kind is AliasResolutionKind.LEGACY_MODEL_REF
        assert resolution.rewritten is True
        assert resolution.original_name == "voxtral"
        assert resolution.original_tag == "tts-4b"
        assert (resolution.name, resolution.tag) == ("voxtral-tts", "4b")

    def test_legacy_name_resolution_reports_legacy_metadata(self):
        resolution = resolve_model_alias("qwen3-asr", "0.6b", explicit_tag=True)

        assert resolution.kind is AliasResolutionKind.LEGACY_NAME
        assert resolution.rewritten is True
        assert resolution.original_name == "qwen3-asr"
        assert resolution.original_tag == "0.6b"
        assert (resolution.name, resolution.tag) == ("qwen3-stt", "0.6b")

    def test_documented_legacy_model_ref_examples_are_public_policy(self):
        policies = {
            (policy.name, policy.tag): (policy.target_name, policy.target_tag)
            for policy in legacy_model_ref_alias_policy()
        }

        assert policies[("parakeet", "tdt-0.6b-v3")] == (
            "parakeet-stt",
            "tdt-0.6b-v3",
        )
        assert policies[("voxtral", "tts-4b")] == ("voxtral-tts", "4b")

    def test_documented_legacy_name_examples_are_public_policy(self):
        policies = {
            policy.name: policy.target_name
            for policy in legacy_name_alias_policy()
        }

        assert policies["qwen3-asr"] == "qwen3-stt"

    def test_family_alias_policy_is_read_only(self):
        parakeet = next(policy for policy in family_alias_policy() if policy.name == "parakeet")

        assert (parakeet.target_name, parakeet.target_tag) == ("parakeet-stt", "tdt-0.6b-v3")
        with pytest.raises(dataclasses.FrozenInstanceError):
            parakeet.target_name = "different"


class TestUnknownNameFallthrough:
    def test_unknown_name_passes_through_unchanged(self):
        assert resolve_family_alias("totally-made-up", "latest", explicit_tag=True) == (
            "totally-made-up",
            "latest",
        )
        assert resolve_family_alias("does-not-exist", "v9") == ("does-not-exist", "v9")

    def test_unknown_name_reports_no_alias_metadata(self):
        resolution = resolve_model_alias("does-not-exist", "v9")

        assert resolution.kind is AliasResolutionKind.NONE
        assert resolution.rewritten is False
        assert resolution.original_name == "does-not-exist"
        assert resolution.original_tag == "v9"
        assert (resolution.name, resolution.tag) == ("does-not-exist", "v9")

    def test_unknown_bare_name_with_latest_passes_through(self):
        assert resolve_family_alias("nope") == ("nope", "latest")
