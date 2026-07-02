from __future__ import annotations

from unittest.mock import patch

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
    def test_bare_name_uses_default_profile_off_spark(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VOX_DEVICE", "auto")
        monkeypatch.setattr("vox.core.device_placement.platform.machine", lambda: "arm64")

        assert resolve_family_alias("parakeet") == ("parakeet-stt-onnx", "tdt-0.6b-v3")
        assert resolve_family_alias("kokoro") == ("kokoro-tts-onnx", "v1.0")
        assert resolve_family_alias("voxtral-stt") == ("voxtral-stt-torch", "mini-3b")
        assert resolve_family_alias("voxtral-tts") == ("voxtral-tts-vllm", "4b")

    def test_bare_name_resolution_reports_family_alias_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("VOX_DEVICE", "auto")
        monkeypatch.setattr("vox.core.device_placement.platform.machine", lambda: "arm64")

        resolution = resolve_model_alias("parakeet")

        assert resolution.kind is AliasResolutionKind.FAMILY
        assert resolution.rewritten is True
        assert resolution.original_name == "parakeet"
        assert resolution.original_tag == "latest"
        assert resolution.profile == "default"
        assert resolution.resolved_profile == "default"
        assert (resolution.name, resolution.tag) == ("parakeet-stt-onnx", "tdt-0.6b-v3")

    def test_bare_name_uses_spark_profile_when_cuda_on_arm(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VOX_DEVICE", "cuda")
        monkeypatch.setattr("vox.core.device_placement.platform.machine", lambda: "arm64")

        assert resolve_family_alias("parakeet") == ("parakeet-stt-nemo", "tdt-0.6b-v3")
        assert resolve_family_alias("kokoro") == ("kokoro-tts-torch", "v1.0")
        assert resolve_family_alias("parakeet-stt") == ("parakeet-stt-nemo", "tdt-0.6b-v3")
        assert resolve_family_alias("kokoro-tts") == ("kokoro-tts-torch", "v1.0")

    def test_bare_name_falls_back_to_default_when_profile_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("VOX_DEVICE", "auto")
        monkeypatch.setattr("vox.core.device_placement.platform.machine", lambda: "x86_64")
        with patch(
            "vox.core.device_placement.infer_runtime_profile",
            return_value="unknown-profile",
        ):
            resolution = resolve_model_alias("kokoro")

        assert resolution.profile == "unknown-profile"
        assert resolution.resolved_profile == "default"
        assert (resolution.name, resolution.tag) == ("kokoro-tts-onnx", "v1.0")

    def test_all_bare_family_names_resolve(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VOX_DEVICE", "cuda")
        monkeypatch.setattr("vox.core.device_placement.platform.machine", lambda: "arm64")

        assert resolve_family_alias("whisper") == ("whisper-stt-ct2", "base.en")
        assert resolve_family_alias("whisper-stt") == ("whisper-stt-ct2", "base.en")
        assert resolve_family_alias("piper") == ("piper-tts-onnx", "en-us-lessac-medium")
        assert resolve_family_alias("openvoice") == ("openvoice-tts-torch", "v1")
        assert resolve_family_alias("chatterbox") == ("chatterbox-tts-turbo", "0.1.7")
        assert resolve_family_alias("chatterbox-multilingual") == (
            "chatterbox-tts-multilingual",
            "0.1.7",
        )
        assert resolve_family_alias("cosyvoice") == ("cosyvoice2-tts-torch", "0.5b")
        assert resolve_family_alias("cosyvoice2") == ("cosyvoice2-tts-torch", "0.5b")
        assert resolve_family_alias("orpheus") == ("orpheus-tts-vllm", "medium-3b")
        assert resolve_family_alias("indextts") == ("indextts-tts-torch", "2")
        assert resolve_family_alias("spark") == ("spark-tts-torch", "0.5b")
        assert resolve_family_alias("neutts") == ("neutts-air-tts-torch", "air")
        assert resolve_family_alias("neutts-air") == ("neutts-air-tts-torch", "air")
        assert resolve_family_alias("dia") == ("dia-tts-torch", "1.6b")
        assert resolve_family_alias("sesame") == ("sesame-tts-torch", "csm-1b")
        assert resolve_family_alias("speecht5-stt") == ("speecht5-stt-torch", "base")
        assert resolve_family_alias("speecht5-tts") == ("speecht5-tts-torch", "base")
        assert resolve_family_alias("vibevoice") == ("vibevoice-tts-torch", "realtime-0.5b")
        assert resolve_family_alias("qwen3-stt") == ("qwen3-stt-torch", "0.6b")
        assert resolve_family_alias("qwen3-tts") == ("qwen3-tts-torch", "0.6b")
        assert resolve_family_alias("xtts") == ("xtts-tts-torch", "v2")


class TestProfileInference:
    def test_inferred_spark_profile_used_without_explicit_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("VOX_DEVICE", "auto")
        monkeypatch.setattr("vox.core.device_placement.platform.machine", lambda: "x86_64")
        with patch(
            "vox.core.device_placement.infer_runtime_profile",
            return_value="spark",
        ):
            parakeet = resolve_model_alias("parakeet")
            kokoro = resolve_model_alias("kokoro")

        assert parakeet.resolved_profile == "spark"
        assert kokoro.resolved_profile == "spark"
        assert (parakeet.name, parakeet.tag) == ("parakeet-stt-nemo", "tdt-0.6b-v3")
        assert (kokoro.name, kokoro.tag) == ("kokoro-tts-torch", "v1.0")

    def test_cuda_hint_on_arm_forces_spark_regardless_of_inference(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("VOX_DEVICE", "cuda")
        monkeypatch.setattr("vox.core.device_placement.platform.machine", lambda: "aarch64")
        with patch(
            "vox.core.device_placement.infer_runtime_profile",
            return_value="default",
        ):
            assert resolve_family_alias("parakeet") == ("parakeet-stt-nemo", "tdt-0.6b-v3")


class TestExplicitTags:
    def test_explicit_latest_is_not_rewritten(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VOX_DEVICE", "cuda")
        monkeypatch.setattr("vox.core.device_placement.platform.machine", lambda: "arm64")

        assert resolve_family_alias("parakeet", "latest", explicit_tag=True) == (
            "parakeet",
            "latest",
        )

    def test_explicit_legacy_pair_still_rewrites(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VOX_DEVICE", "cuda")
        monkeypatch.setattr("vox.core.device_placement.platform.machine", lambda: "arm64")

        assert resolve_family_alias("kokoro", "v1.0", explicit_tag=True) == (
            "kokoro-tts-onnx",
            "v1.0",
        )


class TestLegacyAliasRewrite:
    def test_legacy_model_ref_pairs_rewrite_to_canonical(self):
        assert resolve_family_alias("kokoro", "v1.0") == ("kokoro-tts-onnx", "v1.0")
        assert resolve_family_alias("kokoro", "v1.0-torch") == ("kokoro-tts-torch", "v1.0")
        assert resolve_family_alias("parakeet", "tdt-0.6b") == (
            "parakeet-stt-onnx",
            "tdt-0.6b",
        )
        assert resolve_family_alias("parakeet", "tdt-0.6b-v3-cuda") == (
            "parakeet-stt-nemo",
            "tdt-0.6b-v3",
        )
        assert resolve_family_alias("parakeet", "tdt-1.1b-cuda") == (
            "parakeet-stt-nemo",
            "tdt-1.1b",
        )
        assert resolve_family_alias("speecht5", "asr") == ("speecht5-stt-torch", "base")
        assert resolve_family_alias("speecht5", "tts") == ("speecht5-tts-torch", "base")
        assert resolve_family_alias("voxtral", "tts-4b") == ("voxtral-tts-vllm", "4b")
        assert resolve_family_alias("voxtral", "24b") == ("voxtral-stt-torch", "24b")

    def test_legacy_name_aliases_rewrite_canonical_with_tag_passthrough(self):
        assert resolve_family_alias("qwen3-asr", "0.6b", explicit_tag=True) == (
            "qwen3-stt-torch",
            "0.6b",
        )
        assert resolve_family_alias("kokoro-torch", "v1.0", explicit_tag=True) == (
            "kokoro-tts-torch",
            "v1.0",
        )
        assert resolve_family_alias("parakeet-nemo", "tdt-1.1b", explicit_tag=True) == (
            "parakeet-stt-nemo",
            "tdt-1.1b",
        )

    def test_legacy_model_ref_resolution_reports_legacy_metadata(self):
        resolution = resolve_model_alias("voxtral", "tts-4b", explicit_tag=True)

        assert resolution.kind is AliasResolutionKind.LEGACY_MODEL_REF
        assert resolution.rewritten is True
        assert resolution.original_name == "voxtral"
        assert resolution.original_tag == "tts-4b"
        assert (resolution.name, resolution.tag) == ("voxtral-tts-vllm", "4b")

    def test_legacy_name_resolution_reports_legacy_metadata(self):
        resolution = resolve_model_alias("qwen3-asr", "0.6b", explicit_tag=True)

        assert resolution.kind is AliasResolutionKind.LEGACY_NAME
        assert resolution.rewritten is True
        assert resolution.original_name == "qwen3-asr"
        assert resolution.original_tag == "0.6b"
        assert (resolution.name, resolution.tag) == ("qwen3-stt-torch", "0.6b")

    def test_documented_legacy_model_ref_examples_are_public_policy(self):
        policies = {
            (policy.name, policy.tag): (policy.target_name, policy.target_tag)
            for policy in legacy_model_ref_alias_policy()
        }

        assert policies[("parakeet", "tdt-0.6b-v3-cuda")] == (
            "parakeet-stt-nemo",
            "tdt-0.6b-v3",
        )
        assert policies[("voxtral", "tts-4b")] == ("voxtral-tts-vllm", "4b")

    def test_documented_legacy_name_examples_are_public_policy(self):
        policies = {
            policy.name: policy.target_name
            for policy in legacy_name_alias_policy()
        }

        assert policies["qwen3-asr"] == "qwen3-stt-torch"
        assert policies["kokoro-torch"] == "kokoro-tts-torch"

    def test_family_alias_policy_is_read_only(self):
        parakeet = next(policy for policy in family_alias_policy() if policy.name == "parakeet")

        with pytest.raises(TypeError):
            parakeet.profiles["default"] = ("different", "latest")


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

    def test_unknown_bare_name_with_latest_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("VOX_DEVICE", "cuda")
        monkeypatch.setattr("vox.core.device_placement.platform.machine", lambda: "arm64")

        assert resolve_family_alias("nope") == ("nope", "latest")
