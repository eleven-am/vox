"""Tests for ModelRegistry: catalog lookup and model resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from vox.core.adapter_resolution import AdapterResolver
from vox.core.errors import ModelNotFoundError
from vox.core.registry import CATALOG, ModelRegistry
from vox.core.store import BlobStore, Manifest, ManifestLayer


def _make_store(tmp_path: Path) -> BlobStore:
    return BlobStore(root=tmp_path)


def _write_manifest(
    store: BlobStore,
    name: str,
    tag: str,
    *,
    adapter: str = "fake",
    model_type: str = "stt",
    fmt: str = "onnx",
    source: str | None = None,
    layers: list[ManifestLayer] | None = None,
) -> Manifest:
    if layers is None:
        digest = "sha256-" + "ab" * 32
        blob_path = store.blobs_dir / digest
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"fake-model-data")
        layers = [
            ManifestLayer(
                media_type="application/vox.model.onnx",
                digest=digest,
                size=15,
                filename="model.onnx",
            )
        ]

    config: dict = {
        "architecture": "test-arch",
        "type": model_type,
        "adapter": adapter,
        "format": fmt,
        "parameters": {"sample_rate": 16000},
    }
    if source is not None:
        config["source"] = source

    manifest = Manifest(layers=layers, config=config)
    store.save_manifest(name, tag, manifest)
    return manifest


def _make_registry(store: BlobStore, adapters: dict | None = None) -> ModelRegistry:
    resolver = AdapterResolver(
        store.root, bundled_adapters_root=store.root / "_no_bundled"
    )
    if adapters is not None:
        resolver._adapters = dict(adapters)
    return ModelRegistry(store, resolver=resolver)


class TestLookup:
    def test_lookup_existing_model(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        entry = registry.lookup("whisper-stt-ct2", "large-v3")
        assert entry is not None
        assert entry["source"] == "Systran/faster-whisper-large-v3"
        assert entry["type"] == "stt"
        assert entry["adapter"] == "whisper-stt-ct2"

    def test_lookup_missing_model(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        assert registry.lookup("nonexistent-model") is None

    def test_lookup_missing_tag(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        assert registry.lookup("whisper-stt-ct2", "no-such-tag") is None

    def test_lookup_bare_family_prefers_spark_alias_when_cuda(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        store = _make_store(tmp_path)
        registry = _make_registry(store)
        monkeypatch.setenv("VOX_DEVICE", "cuda")
        monkeypatch.setattr("vox.core.registry.platform.machine", lambda: "arm64")

        entry = registry.lookup("parakeet")
        assert entry is not None
        assert entry is CATALOG["parakeet-stt-nemo"]["tdt-0.6b-v3"]

        kokoro = registry.lookup("kokoro")
        assert kokoro is not None
        assert kokoro is CATALOG["kokoro-tts-torch"]["v1.0"]

        voxtral_tts = registry.lookup("voxtral-tts")
        assert voxtral_tts is not None
        assert voxtral_tts is CATALOG["voxtral-tts-vllm"]["4b"]

        parakeet_stt = registry.lookup("parakeet-stt")
        assert parakeet_stt is not None
        assert parakeet_stt is CATALOG["parakeet-stt-nemo"]["tdt-0.6b-v3"]

    def test_lookup_bare_family_prefers_default_alias_off_spark(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        store = _make_store(tmp_path)
        registry = _make_registry(store)
        monkeypatch.setenv("VOX_DEVICE", "auto")
        monkeypatch.setattr("vox.core.registry.platform.machine", lambda: "arm64")

        entry = registry.lookup("parakeet")
        assert entry is not None
        assert entry is CATALOG["parakeet-stt-onnx"]["tdt-0.6b-v3"]

        kokoro = registry.lookup("kokoro")
        assert kokoro is not None
        assert kokoro is CATALOG["kokoro-tts-onnx"]["v1.0"]

        voxtral_tts = registry.lookup("voxtral-tts")
        assert voxtral_tts is not None
        assert voxtral_tts is CATALOG["voxtral-tts-vllm"]["4b"]

        parakeet_stt = registry.lookup("parakeet-stt")
        assert parakeet_stt is not None
        assert parakeet_stt is CATALOG["parakeet-stt-onnx"]["tdt-0.6b-v3"]

    def test_lookup_explicit_latest_is_not_rewritten(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        store = _make_store(tmp_path)
        registry = _make_registry(store)
        monkeypatch.setenv("VOX_DEVICE", "cuda")
        monkeypatch.setattr("vox.core.registry.platform.machine", lambda: "arm64")

        assert registry.lookup("parakeet", "latest", explicit_tag=True) is None
        assert registry.resolve_model_ref("parakeet", "latest", explicit_tag=True) == ("parakeet", "latest")

    def test_lookup_bare_family_infers_spark_profile_without_explicit_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        store = _make_store(tmp_path)
        registry = _make_registry(store)
        monkeypatch.setenv("VOX_DEVICE", "auto")
        monkeypatch.setattr("vox.core.registry.platform.machine", lambda: "arm64")
        with patch("vox.core.device_placement.infer_runtime_profile", return_value="spark"):
            entry = registry.lookup("parakeet")
            assert entry is CATALOG["parakeet-stt-nemo"]["tdt-0.6b-v3"]


class TestAdapterForwarders:
    def test_get_adapter_class_delegates_to_resolver(self, tmp_path: Path):
        store = _make_store(tmp_path)

        class FakeAdapter:
            pass

        registry = _make_registry(store, adapters={"fake": FakeAdapter})
        assert registry.get_adapter_class("fake") is FakeAdapter

    def test_ensure_adapter_delegates_to_resolver(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        with patch.object(registry.adapter_resolver, "ensure", return_value=True) as ensure_mock:
            assert registry.ensure_adapter("fake", "vox-fake") is True
        ensure_mock.assert_called_once_with("fake", "vox-fake")


class TestResolve:
    def test_resolve_raises_model_not_found_when_no_manifest(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        with pytest.raises(ModelNotFoundError):
            registry.resolve("nothing", "latest")

    def test_resolve_creates_symlinks(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _write_manifest(store, "mymodel", "v1")
        registry = _make_registry(store)

        info, model_dir = registry.resolve("mymodel", "v1")

        assert info.name == "mymodel"
        assert info.tag == "v1"
        assert model_dir.is_dir()

        link = model_dir / "model.onnx"
        assert link.is_symlink()
        assert link.resolve().exists()

    def test_resolve_ensures_adapter_package_from_manifest(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _write_manifest(store, "mymodel", "v1", adapter="fake-adapter")
        manifest = store.resolve_model("mymodel", "v1")
        assert manifest is not None
        manifest.config["adapter_package"] = "vox-fake"
        store.save_manifest("mymodel", "v1", manifest)
        registry = _make_registry(store)

        with patch.object(registry, "ensure_adapter", return_value=True) as ensure_mock:
            info, _ = registry.resolve("mymodel", "v1")

        ensure_mock.assert_called_once_with("fake-adapter", "vox-fake")
        assert info.adapter == "fake-adapter"

    def test_resolve_creates_parent_dirs_for_nested_filenames(self, tmp_path: Path):
        store = _make_store(tmp_path)
        digest = "sha256-" + "cd" * 32
        blob_path = store.blobs_dir / digest
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"nested-model")
        manifest = Manifest(
            layers=[
                ManifestLayer(
                    media_type="application/vox.model.onnx",
                    digest=digest,
                    size=12,
                    filename="onnx/model.onnx",
                )
            ],
            config={
                "architecture": "test-arch",
                "type": "tts",
                "adapter": "kokoro",
                "format": "onnx",
                "parameters": {"sample_rate": 24000},
            },
        )
        store.save_manifest("kokoro", "v1", manifest)
        registry = _make_registry(store)

        _, model_dir = registry.resolve("kokoro", "v1")

        link = model_dir / "onnx" / "model.onnx"
        assert link.is_symlink()
        assert link.resolve().exists()

    def test_resolve_handles_stale_symlinks(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _write_manifest(store, "mymodel", "v1")
        registry = _make_registry(store)

        model_dir = store.root / "models" / "links" / "mymodel" / "v1"
        model_dir.mkdir(parents=True, exist_ok=True)
        stale_link = model_dir / "model.onnx"
        stale_link.symlink_to("/nonexistent/path/that/does/not/exist")
        assert stale_link.is_symlink()
        assert not stale_link.exists()

        info, resolved_dir = registry.resolve("mymodel", "v1")

        link = resolved_dir / "model.onnx"
        assert link.is_symlink()
        assert link.exists()

    def test_resolve_injects_source_into_parameters(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _write_manifest(
            store, "mymodel", "v1", source="huggingface/some-repo"
        )
        registry = _make_registry(store)

        info, _ = registry.resolve("mymodel", "v1")
        assert info.parameters.get("_source") == "huggingface/some-repo"

    def test_resolve_no_source_means_no_injection(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _write_manifest(store, "mymodel", "v1")
        registry = _make_registry(store)

        info, _ = registry.resolve("mymodel", "v1")
        assert "_source" not in info.parameters


class TestAvailableModels:
    def test_available_models_returns_catalog(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        catalog = registry.available_models()
        assert catalog is CATALOG
        assert "whisper-stt-ct2" in catalog
        assert "kokoro-tts-onnx" in catalog

    def test_whisper_catalog_uses_ct2_and_whisper_adapter_package(self):
        whisper = CATALOG["whisper-stt-ct2"]

        assert set(whisper) == {"large-v3", "large-v3-turbo", "base.en", "small.en", "medium.en"}
        for _tag, entry in whisper.items():
            assert entry["adapter_package"] == "vox-whisper"
            assert entry["format"] == "ct2"
            assert entry["adapter"] == "whisper-stt-ct2"
            assert entry["type"] == "stt"
            assert entry["parameters"]["sample_rate"] == 16000

    def test_sesame_catalog_entry_has_default_voice(self):
        sesame = CATALOG["sesame-tts-torch"]["csm-1b"]

        assert sesame["adapter_package"] == "vox-sesame"
        assert sesame["parameters"]["sample_rate"] == 24_000
        assert sesame["parameters"]["default_voice"] == "0"

    def test_parakeet_nemo_catalog_entry_is_explicit_and_pytorch(self):
        parakeet_nemo = CATALOG["parakeet-stt-nemo"]["tdt-0.6b-v3"]

        assert parakeet_nemo["adapter_package"] == "vox-parakeet"
        assert parakeet_nemo["adapter"] == "parakeet-stt-nemo"
        assert parakeet_nemo["format"] == "pytorch"
        assert parakeet_nemo["files"] == ["parakeet-tdt-0.6b-v3.nemo"]
        assert parakeet_nemo["parameters"]["sample_rate"] == 16_000

    def test_parakeet_onnx_catalog_entry_uses_onnx_repo_and_runtime_source(self):
        parakeet_onnx = CATALOG["parakeet-stt-onnx"]["tdt-0.6b-v3"]

        assert parakeet_onnx["source"] == "istupakov/parakeet-tdt-0.6b-v3-onnx"
        assert parakeet_onnx["runtime_source"] == "nvidia/parakeet-tdt-0.6b-v3"
        assert parakeet_onnx["files"] == [
            "config.json",
            "decoder_joint-model.onnx",
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "nemo128.onnx",
            "vocab.txt",
        ]

    def test_parakeet_cuda_alias_points_to_nemo_backend(self):
        registry_entry = CATALOG["parakeet-stt-nemo"]["tdt-0.6b-v3"]

        assert registry_entry["adapter_package"] == "vox-parakeet"
        assert registry_entry["adapter"] == "parakeet-stt-nemo"
        assert registry_entry["format"] == "pytorch"
        assert registry_entry["files"] == ["parakeet-tdt-0.6b-v3.nemo"]

    def test_parakeet_1_1b_variants_use_nemo_backend(self):
        parakeet_nemo = CATALOG["parakeet-stt-nemo"]["tdt-1.1b"]

        assert parakeet_nemo["adapter_package"] == "vox-parakeet"
        assert parakeet_nemo["adapter"] == "parakeet-stt-nemo"
        assert parakeet_nemo["files"] == ["parakeet-tdt-1.1b.nemo"]

    def test_voxtral_24b_alias_points_to_large_stt_source(self):
        voxtral_24b = CATALOG["voxtral-stt-torch"]["24b"]

        assert voxtral_24b["adapter_package"] == "vox-voxtral"
        assert voxtral_24b["adapter"] == "voxtral-stt-torch"
        assert voxtral_24b["source"] == "mistralai/Voxtral-Small-24B-2507"

    def test_dia_catalog_entry_uses_transformers_compatible_checkpoint(self):
        dia = CATALOG["dia-tts-torch"]["1.6b"]

        assert dia["adapter_package"] == "vox-dia"
        assert dia["adapter"] == "dia-tts-torch"
        assert dia["source"] == "nari-labs/Dia-1.6B-0626"

    def test_kokoro_torch_catalog_entry_is_explicit_and_pytorch(self):
        kokoro_torch = CATALOG["kokoro-tts-torch"]["v1.0"]

        assert kokoro_torch["adapter_package"] == "vox-kokoro"
        assert kokoro_torch["adapter"] == "kokoro-tts-torch"
        assert kokoro_torch["format"] == "pytorch"
        assert kokoro_torch["files"] == ["kokoro-v1_0.pth"]
        assert kokoro_torch["parameters"]["default_voice"] == "af_heart"

    def test_resolve_model_ref_uses_aliases(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        store = _make_store(tmp_path)
        registry = _make_registry(store)
        monkeypatch.setenv("VOX_DEVICE", "cuda")
        monkeypatch.setattr("vox.core.registry.platform.machine", lambda: "arm64")

        assert registry.resolve_model_ref("parakeet") == ("parakeet-stt-nemo", "tdt-0.6b-v3")
        assert registry.resolve_model_ref("parakeet-stt") == ("parakeet-stt-nemo", "tdt-0.6b-v3")
        assert registry.resolve_model_ref("kokoro") == ("kokoro-tts-torch", "v1.0")
        assert registry.resolve_model_ref("kokoro-tts") == ("kokoro-tts-torch", "v1.0")
        assert registry.resolve_model_ref("whisper") == ("whisper-stt-ct2", "base.en")
        assert registry.resolve_model_ref("whisper-stt") == ("whisper-stt-ct2", "base.en")
        assert registry.resolve_model_ref("piper") == ("piper-tts-onnx", "en-us-lessac-medium")
        assert registry.resolve_model_ref("openvoice") == ("openvoice-tts-torch", "v1")
        assert registry.resolve_model_ref("dia") == ("dia-tts-torch", "1.6b")
        assert registry.resolve_model_ref("sesame") == ("sesame-tts-torch", "csm-1b")
        assert registry.resolve_model_ref("speecht5-stt") == ("speecht5-stt-torch", "base")
        assert registry.resolve_model_ref("speecht5-tts") == ("speecht5-tts-torch", "base")
        assert registry.resolve_model_ref("vibevoice") == ("vibevoice-tts-torch", "realtime-0.5b")
        assert registry.resolve_model_ref("qwen3-stt") == ("qwen3-stt-torch", "0.6b")
        assert registry.resolve_model_ref("qwen3-tts") == ("qwen3-tts-torch", "0.6b")
        assert registry.resolve_model_ref("xtts") == ("xtts-tts-torch", "v2")
        assert registry.resolve_model_ref("voxtral-stt") == ("voxtral-stt-torch", "mini-3b")
        assert registry.resolve_model_ref("voxtral-tts") == ("voxtral-tts-vllm", "4b")

    def test_resolve_model_ref_keeps_explicit_tags(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        store = _make_store(tmp_path)
        registry = _make_registry(store)
        monkeypatch.setenv("VOX_DEVICE", "cuda")
        monkeypatch.setattr("vox.core.registry.platform.machine", lambda: "arm64")

        assert registry.resolve_model_ref("parakeet", "latest", explicit_tag=True) == ("parakeet", "latest")
        assert registry.resolve_model_ref("kokoro", "v1.0", explicit_tag=True) == ("kokoro-tts-onnx", "v1.0")

    def test_resolve_model_ref_uses_inferred_spark_profile_without_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        store = _make_store(tmp_path)
        registry = _make_registry(store)
        monkeypatch.setenv("VOX_DEVICE", "auto")
        monkeypatch.setattr("vox.core.registry.platform.machine", lambda: "arm64")

        with patch("vox.core.device_placement.infer_runtime_profile", return_value="spark"):
            assert registry.resolve_model_ref("parakeet") == ("parakeet-stt-nemo", "tdt-0.6b-v3")
            assert registry.resolve_model_ref("kokoro") == ("kokoro-tts-torch", "v1.0")

    def test_openvoice_catalog_entry_has_checkpoint_files(self):
        openvoice = CATALOG["openvoice-tts-torch"]["v1"]

        assert openvoice["adapter_package"] == "vox-openvoice"
        assert openvoice["parameters"]["sample_rate"] == 22_050
        assert openvoice["parameters"]["default_voice"] == "en/default"
        assert "checkpoints/base_speakers/EN/config.json" in openvoice["files"]

    def test_xtts_catalog_entry_uses_huggingface_repo_id(self):
        xtts = CATALOG["xtts-tts-torch"]["v2"]

        assert xtts["source"] == "coqui/XTTS-v2"
        assert xtts["adapter_package"] == "vox-xtts"

    def test_resolve_legacy_aliases_to_canonical_names(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        assert registry.resolve_model_ref("qwen3-asr", "0.6b", explicit_tag=True) == ("qwen3-stt-torch", "0.6b")
        assert registry.resolve_model_ref("speecht5", "asr", explicit_tag=True) == ("speecht5-stt-torch", "base")
        assert registry.resolve_model_ref("voxtral", "tts-4b", explicit_tag=True) == ("voxtral-tts-vllm", "4b")
