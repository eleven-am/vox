from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tests._catalog_fixture import FIXTURE_CATALOG as CATALOG
from vox.core.adapter_resolution import AdapterResolver
from vox.core.errors import ModelLoadError, ModelNotFoundError
from vox.core.registry import ModelRegistry
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
    resolver = AdapterResolver(store.root)
    if adapters is not None:
        resolver._adapters = dict(adapters)
    return ModelRegistry(store, resolver=resolver)


class TestLookup:
    def test_lookup_existing_model(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        entry = registry.lookup("whisper-stt", "large-v3")
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

    def test_lookup_routes_bare_family_through_alias_resolver(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        store = _make_store(tmp_path)
        registry = _make_registry(store)
        monkeypatch.setenv("VOX_DEVICE", "cuda")
        monkeypatch.setattr(
            "vox.core.device_placement.platform.machine", lambda: "arm64"
        )

        entry = registry.lookup("parakeet")
        assert entry == CATALOG["parakeet-stt"]["tdt-0.6b-v3"]


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

    def test_resolve_rejects_parent_traversal_filename(self, tmp_path: Path):
        store = _make_store(tmp_path)
        digest = "sha256-" + "ef" * 32
        blob_path = store.blobs_dir / digest
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"evil")
        _write_manifest(
            store, "evil", "v1",
            layers=[
                ManifestLayer(
                    media_type="application/vox.model.onnx",
                    digest=digest,
                    size=4,
                    filename="../../../../../../tmp/vox-escape.onnx",
                )
            ],
        )
        registry = _make_registry(store)

        with pytest.raises(ModelLoadError, match="escapes model directory"):
            registry.resolve("evil", "v1")

    def test_resolve_rejects_absolute_filename(self, tmp_path: Path):
        store = _make_store(tmp_path)
        digest = "sha256-" + "12" * 32
        blob_path = store.blobs_dir / digest
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"evil")
        _write_manifest(
            store, "evil", "v2",
            layers=[
                ManifestLayer(
                    media_type="application/vox.model.onnx",
                    digest=digest,
                    size=4,
                    filename=str(tmp_path / "vox-abs-escape.onnx"),
                )
            ],
        )
        registry = _make_registry(store)

        with pytest.raises(ModelLoadError, match="escapes model directory"):
            registry.resolve("evil", "v2")

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

    def test_resolve_ignores_pull_time_runtime_diagnostics(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _write_manifest(
            store,
            "mymodel",
            "v1",
            adapter="kokoro-tts-onnx",
            model_type="tts",
            fmt="onnx",
        )
        manifest = store.resolve_model("mymodel", "v1")
        assert manifest is not None
        manifest.config["runtime"] = {
            "checked_at_pull": True,
            "resolved_variant": "onnx",
            "preferred_backend": "kokoro-onnx-cpu",
            "detected": {"torch_cuda": True},
        }
        store.save_manifest("mymodel", "v1", manifest)
        registry = _make_registry(store)

        info, _ = registry.resolve("mymodel", "v1")

        assert info.adapter == "kokoro-tts-onnx"
        assert info.format.value == "onnx"
        assert "runtime" not in info.parameters


class TestAvailableModels:
    def test_available_models_returns_catalog(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        catalog = registry.available_models()
        assert "whisper-stt" in catalog
        assert "kokoro-tts" in catalog
        assert catalog["whisper-stt"]["large-v3"]["type"] == "stt"
        assert "kokoro-tts-onnx" not in catalog
        assert "kokoro-tts-torch" not in catalog

    def test_whisper_catalog_uses_ct2_and_whisper_adapter_package(self):
        whisper = CATALOG["whisper-stt"]

        assert set(whisper) == {"large-v3", "large-v3-turbo", "base.en", "small.en", "medium.en"}
        for _tag, entry in whisper.items():
            assert entry["adapter_package"] == "vox-whisper"
            assert entry["format"] == "ct2"
            assert entry["adapter"] == "whisper-stt-ct2"
            assert entry["type"] == "stt"
            assert entry["parameters"]["sample_rate"] == 16000

    def test_sesame_catalog_entry_has_default_voice(self):
        sesame = CATALOG["sesame-tts"]["csm-1b"]

        assert sesame["adapter_package"] == "vox-sesame"
        assert sesame["parameters"]["sample_rate"] == 24_000
        assert sesame["parameters"]["default_voice"] == "0"

    def test_parakeet_nemo_catalog_entry_is_explicit_and_pytorch(self):
        entry = CATALOG["parakeet-stt"]["tdt-0.6b-v3"]
        parakeet_nemo = next(v for v in entry["variants"] if v["id"] == "nemo")

        assert parakeet_nemo["adapter_package"] == "vox-parakeet"
        assert parakeet_nemo["adapter"] == "parakeet-stt-nemo"
        assert parakeet_nemo["format"] == "pytorch"
        assert parakeet_nemo["files"] == ["parakeet-tdt-0.6b-v3.nemo"]
        assert entry["parameters"]["sample_rate"] == 16_000

    def test_parakeet_onnx_catalog_entry_uses_onnx_repo_and_runtime_source(self):
        entry = CATALOG["parakeet-stt"]["tdt-0.6b-v3"]
        parakeet_onnx = next(v for v in entry["variants"] if v["id"] == "onnx")

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
        entry = CATALOG["parakeet-stt"]["tdt-0.6b-v3"]
        registry_entry = next(v for v in entry["variants"] if v["id"] == "nemo")

        assert registry_entry["adapter_package"] == "vox-parakeet"
        assert registry_entry["adapter"] == "parakeet-stt-nemo"
        assert registry_entry["format"] == "pytorch"
        assert registry_entry["files"] == ["parakeet-tdt-0.6b-v3.nemo"]

    def test_parakeet_1_1b_variants_use_nemo_backend(self):
        entry = CATALOG["parakeet-stt"]["tdt-1.1b"]
        parakeet_nemo = next(v for v in entry["variants"] if v["id"] == "nemo")

        assert parakeet_nemo["adapter_package"] == "vox-parakeet"
        assert parakeet_nemo["adapter"] == "parakeet-stt-nemo"
        assert parakeet_nemo["files"] == ["parakeet-tdt-1.1b.nemo"]

    def test_voxtral_24b_alias_points_to_large_stt_source(self):
        voxtral_24b = CATALOG["voxtral-stt"]["24b"]

        assert voxtral_24b["adapter_package"] == "vox-voxtral"
        assert voxtral_24b["adapter"] == "voxtral-stt-torch"
        assert voxtral_24b["source"] == "mistralai/Voxtral-Small-24B-2507"

    def test_dia_catalog_entry_uses_transformers_compatible_checkpoint(self):
        dia = CATALOG["dia-tts"]["1.6b"]

        assert dia["adapter_package"] == "vox-dia"
        assert dia["adapter"] == "dia-tts-torch"
        assert dia["source"] == "nari-labs/Dia-1.6B-0626"
        assert dia["runtime"]["required"]["accelerators"] == ["cuda"]
        assert dia["runtime"]["required"]["min_vram_gb"] == 10
        assert "no CPU/ONNX path" in dia["runtime"]["required"]["notes"][0]

    def test_kokoro_logical_catalog_entry_keeps_concrete_variants(self):
        kokoro = CATALOG["kokoro-tts"]["v1.0"]

        assert kokoro["type"] == "tts"
        assert "variants" in kokoro
        variants = {variant["id"]: variant for variant in kokoro["variants"]}
        assert variants["torch"]["adapter"] == "kokoro-tts-torch"
        assert variants["torch"]["format"] == "pytorch"
        assert variants["torch"]["files"] == ["kokoro-v1_0.pth"]
        assert variants["onnx"]["adapter"] == "kokoro-tts-onnx"
        assert variants["onnx"]["format"] == "onnx"
        assert variants["onnx"]["fallback"] is True

    def test_kokoro_logical_lookup_uses_canonical_name(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        entry = registry.lookup("kokoro-tts")

        assert entry == CATALOG["kokoro-tts"]["v1.0"]
        assert entry["variants"][0]["adapter"] == "kokoro-tts-torch"

    def test_kokoro_backend_suffix_names_are_not_public_catalog_entries(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        with patch("vox.core.registry.fetch_from_registry", return_value=None):
            assert registry.lookup("kokoro-tts-onnx", "v1.0", explicit_tag=True) is None
            assert registry.lookup("kokoro-tts-torch", "v1.0", explicit_tag=True) is None

    def test_openvoice_catalog_entry_has_checkpoint_files(self):
        openvoice = CATALOG["openvoice-tts"]["v1"]

        assert openvoice["adapter_package"] == "vox-openvoice"
        assert openvoice["parameters"]["sample_rate"] == 22_050
        assert openvoice["parameters"]["default_voice"] == "en/default"
        assert "checkpoints/base_speakers/EN/config.json" in openvoice["files"]

    def test_chatterbox_catalog_entries_use_chatterbox_adapter_package(self):
        turbo = CATALOG["chatterbox-tts-turbo"]["0.1.7"]
        standard = CATALOG["chatterbox-tts"]["0.1.7"]
        multilingual = CATALOG["chatterbox-tts-multilingual"]["0.1.7"]

        variants = {variant["id"]: variant for variant in turbo["variants"]}
        assert variants["torch"]["adapter_package"] == "vox-chatterbox"
        assert variants["torch"]["adapter"] == "chatterbox-tts-turbo"
        assert variants["torch"]["format"] == "pytorch"
        assert variants["onnx"]["source"] == "ResembleAI/chatterbox-turbo-ONNX"
        assert variants["onnx"]["adapter"] == "chatterbox-tts-turbo-onnx"
        assert variants["onnx"]["format"] == "onnx"
        assert "onnx/language_model.onnx" in variants["onnx"]["files"]
        assert standard["adapter"] == "chatterbox-tts"
        assert multilingual["adapter"] == "chatterbox-tts-multilingual"
        assert multilingual["parameters"]["sample_rate"] == 24_000

    def test_indextts_catalog_entry_uses_indextts_adapter_package(self):
        indextts = CATALOG["indextts-tts"]["2"]

        assert indextts["adapter_package"] == "vox-indextts"
        assert indextts["adapter"] == "indextts-tts-torch"
        assert indextts["format"] == "pytorch"
        assert indextts["parameters"]["sample_rate"] == 24_000

    def test_qwen_tts_catalog_entries_prefer_faster_backend_with_fallback(self):
        for entry in CATALOG["qwen3-tts"].values():
            backends = entry["backends"]
            preferred = backends["preferred"][0]
            assert preferred["name"] == "faster-qwen3-tts"
            assert preferred["requires"]["python_modules"] == ["torch", "faster_qwen3_tts"]
            assert preferred["requires"]["accelerators"] == ["cuda"]
            assert preferred["requires"]["min_versions"] == {"torch": "2.5.1"}
            assert backends["fallback"]["name"] == "qwen-tts"

    def test_cosyvoice_catalog_entry_uses_cosyvoice_adapter_package(self):
        cosyvoice = CATALOG["cosyvoice2-tts"]["0.5b"]

        assert cosyvoice["source"] == "FunAudioLLM/CosyVoice2-0.5B"
        assert cosyvoice["adapter_package"] == "vox-cosyvoice"
        assert cosyvoice["adapter"] == "cosyvoice2-tts-torch"
        assert cosyvoice["parameters"]["sample_rate"] == 24_000

    def test_orpheus_catalog_entry_uses_orpheus_adapter_package(self):
        orpheus = CATALOG["orpheus-tts"]["medium-3b"]

        assert orpheus["source"] == "canopylabs/orpheus-tts-0.1-finetune-prod"
        assert orpheus["adapter_package"] == "vox-orpheus"
        assert orpheus["adapter"] == "orpheus-tts-vllm"
        assert orpheus["parameters"]["default_voice"] == "tara"
        assert orpheus["runtime"]["required"]["systems"] == ["linux"]
        assert orpheus["runtime"]["required"]["machines"] == ["x86_64"]
        assert orpheus["runtime"]["required"]["accelerators"] == ["cuda"]
        assert orpheus["runtime"]["required"]["min_vram_gb"] == 10
        assert "Spark/ARM NVIDIA" in orpheus["runtime"]["required"]["notes"][0]

    def test_spark_catalog_entry_uses_spark_adapter_package(self):
        spark = CATALOG["spark-tts"]["0.5b"]

        assert spark["source"] == "SparkAudio/Spark-TTS-0.5B"
        assert spark["adapter_package"] == "vox-spark"
        assert spark["adapter"] == "spark-tts-torch"
        assert spark["parameters"]["sample_rate"] == 16_000

    def test_neutts_catalog_entry_uses_neutts_adapter_package(self):
        neutts = CATALOG["neutts-air-tts"]["air"]

        assert neutts["source"] == "neuphonic/neutts-air"
        assert neutts["adapter_package"] == "vox-neutts"
        assert neutts["adapter"] == "neutts-air-tts-torch"
        assert neutts["parameters"]["sample_rate"] == 24_000

    def test_xtts_catalog_entry_uses_huggingface_repo_id(self):
        xtts = CATALOG["xtts-tts"]["v2"]

        assert xtts["source"] == "coqui/XTTS-v2"
        assert xtts["adapter_package"] == "vox-xtts"
