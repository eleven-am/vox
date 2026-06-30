from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILES = (REPO_ROOT / "Dockerfile", REPO_ROOT / "Dockerfile.spark")

FORBIDDEN_IMAGE_FRAGMENTS = (
    "COPY --chown=vox:vox adapters",
    "COPY adapters",
    "VOX_BUNDLED_ADAPTERS",
    "VOX_ADAPTERS_NO_DEPS",
    "vox-kokoro",
    "vox-parakeet",
    "vox-qwen",
    "vox-dia",
    "vox-sesame",
    "vox-xtts",
    "vox-piper",
    "vox-openvoice",
    "vox-whisper",
    "vox-microsoft",
    "vox-voxtral",
    "kokoro-onnx",
    "onnx-asr",
    "piper-tts",
    "coqui-tts",
    "nemo-toolkit",
    "mistral-common",
)


def test_default_images_do_not_bundle_adapter_packages_or_backend_runtimes():
    for dockerfile in DOCKERFILES:
        content = dockerfile.read_text(encoding="utf-8")
        for fragment in FORBIDDEN_IMAGE_FRAGMENTS:
            assert fragment not in content, f"{dockerfile.name} must not bundle {fragment}"


def test_default_images_keep_runtime_install_primitives():
    for dockerfile in DOCKERFILES:
        content = dockerfile.read_text(encoding="utf-8")
        assert "COPY --from=uv /uv /bin/uv" in content
        assert "python3-venv" in content
        assert "$HOME/.vox/adapters" in content
        assert "ffmpeg" in content
