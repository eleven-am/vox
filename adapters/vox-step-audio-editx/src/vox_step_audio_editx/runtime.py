from __future__ import annotations

import logging
import os
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from vox.core.adapter_runtime import (
    install_target_runtime_requirements,
    remove_target_runtime_paths,
    runtime_root,
    staged_target_runtime,
)

logger = logging.getLogger(__name__)

SOURCE_REF = "a652e87052c109e26f616d60971376ff47a829d4"
SOURCE_URL = f"https://github.com/stepfun-ai/Step-Audio-EditX/archive/{SOURCE_REF}.tar.gz"
RUNTIME_REQUIREMENTS = (
    "vllm==0.18.0",
    "conch-triton-kernels==1.3",
    "compressed-tensors==0.13.0",
    "depyf==0.20.0",
    "diskcache==5.6.3",
    "flashinfer-python==0.6.6",
    "lark==1.2.2",
    "lm-format-enforcer==0.11.3",
    "outlines_core==0.2.11",
    "anthropic==0.120.2",
    "gguf==0.19.0",
    "llguidance==1.3.0",
    "mistral_common==1.11.7",
    "model-hosting-container-standards==0.1.16",
    "openai==2.24.0",
    "openai-harmony==0.0.8",
    "opencv-python-headless==5.0.0.93",
    "opentelemetry-api==1.44.0",
    "opentelemetry-exporter-otlp==1.44.0",
    "opentelemetry-exporter-otlp-proto-grpc==1.44.0",
    "opentelemetry-exporter-otlp-proto-http==1.44.0",
    "opentelemetry-exporter-otlp-proto-common==1.44.0",
    "opentelemetry-proto==1.44.0",
    "opentelemetry-sdk==1.44.0",
    "opentelemetry-semantic-conventions==0.65b0",
    "opentelemetry-semantic-conventions-ai==0.5.1",
    "pillow==12.3.0",
    "prometheus_client==0.26.0",
    "prometheus-fastapi-instrumentator==8.1.0",
    "pyzmq==27.1.0",
    "quack-kernels==0.5.0",
    "tiktoken==0.13.0",
    "xgrammar==0.2.3",
    "blake3==1.0.9",
    "cachetools==7.1.6",
    "cbor2==6.1.3",
    "cloudpickle==3.1.2",
    "einops==0.8.2",
    "ijson==3.5.1",
    "mcp==2.0.0",
    "mcp-types==2.0.0",
    "msgspec==0.21.1",
    "ninja==1.13.0",
    "partial-json-parser==0.2.1.1.post7",
    "py-cpuinfo==9.0.0",
    "pybase64==1.4.3",
    "python-json-logger==4.1.0",
    "setproctitle==1.3.7",
    "apache-tvm-ffi==0.1.12",
    "distro==1.9.0",
    "docstring_parser==0.18.0",
    "email-validator==2.3.0",
    "fastapi-cli==0.0.32",
    "fastar==0.11.0",
    "httpx2==2.9.1",
    "httpcore2==2.9.1",
    "interegular==0.3.3",
    "jiter==0.16.0",
    "jsonschema==4.26.0",
    "pydantic-extra-types==2.11.1",
    "pydantic-settings==2.14.2",
    "PyJWT==2.13.0",
    "sse-starlette==3.4.6",
    "supervisor==4.3.0",
    "astor==0.8.1",
    "jmespath==1.1.0",
    "loguru==0.7.3",
    "sniffio==1.3.1",
    "tabulate==0.10.0",
    "cuda-python==12.9.4",
    "fastapi-cloud-cli==0.23.0",
    "googleapis-common-protos==1.75.0",
    "jsonschema-specifications==2025.9.1",
    "pycountry==26.2.16",
    "referencing==0.37.0",
    "rich-toolkit==0.20.3",
    "rpds-py==2026.6.3",
    "truststore==0.10.4",
    "detect-installer==0.1.0",
    "rignore==0.8.0",
    "sentry-sdk==2.66.1",
    "torchvision==0.25.0",
    "nvidia-cudnn-frontend==1.18.0",
    "nvidia-cutlass-dsl==4.6.1",
    "nvidia-cutlass-dsl-libs-base==4.6.1",
    "nvidia-cutlass-dsl-libs-cu12==4.6.1",
    "nvidia-cutlass-dsl-libs-core==4.6.1",
    "hyperpyyaml==1.2.3",
    "ruamel-yaml==0.18.17",
    "ruamel-yaml-clib==0.2.15",
    "openai-whisper==20250625",
    "more-itertools==10.8.0",
    "funasr==1.3.0",
    "editdistance==0.8.1",
    "hydra-core==1.3.4",
    "antlr4-python3-runtime==4.9.3",
    "kaldiio==2.18.1",
    "jaconv==0.5.0",
    "jamo==0.4.1",
    "jieba==0.42.1",
    "modelscope==1.39.0",
    "oss2==2.19.1",
    "pytorch-wpe==0.0.1",
    "tensorboardX==2.6.5",
    "umap-learn==0.5.12",
    "aliyun-python-sdk-core==2.16.0",
    "aliyun-python-sdk-kms==2.16.5",
    "crcmod==1.7",
    "modelscope-hub==0.1.8",
    "omegaconf==2.3.1",
    "pycryptodome==3.23.0",
    "pynndescent==0.6.0",
    "torch-complex==0.4.4",
    "rotary-embedding-torch==0.8.9",
    "ffmpeg-python==0.2.0",
    "future==1.0.0",
    "sox==1.5.0",
)
SHARED_RUNTIME_GLOBS = (
    "torch",
    "torch-*.dist-info",
    "torchgen",
    "functorch",
    "torchaudio",
    "torchaudio-*.dist-info",
    "torio",
    "triton",
    "triton-*.dist-info",
)
EXPECTED_RUNTIME_PATHS = (
    Path("conch") / "__init__.py",
    Path("vllm") / "__init__.py",
    Path("vllm") / "_C.abi3.so",
    Path("torchvision") / "__init__.py",
    Path("whisper") / "__init__.py",
    Path("hyperpyyaml") / "__init__.py",
)
EXPECTED_SOURCE_PATHS = (
    Path("model_loader.py"),
    Path("tokenizer.py"),
    Path("tts.py"),
    Path("stepvocoder") / "cosyvoice2" / "cli" / "cosyvoice.py",
)
RUNTIME_SENTINEL = ".vox-step-audio-editx-runtime-ready"


def runtime_dir() -> Path:
    return runtime_root() / "step-audio-editx"


def source_dir(root: Path | None = None) -> Path:
    return (root or runtime_dir()) / "source"


def _run_install_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _remove_shared_runtime_stack(path: Path) -> None:
    paths = {item for pattern in SHARED_RUNTIME_GLOBS for item in path.glob(pattern)}
    remove_target_runtime_paths(path, paths)


def _source_complete(path: Path) -> bool:
    return all((path / relative).is_file() for relative in EXPECTED_SOURCE_PATHS)


def _extract_source(target: Path) -> None:
    with tempfile.NamedTemporaryFile(prefix="vox-step-audio-editx-", suffix=".tar.gz") as archive:
        urllib.request.urlretrieve(SOURCE_URL, archive.name)
        with tarfile.open(archive.name, "r:gz") as tar:
            members = tar.getmembers()
            roots = {member.name.split("/", 1)[0] for member in members if member.name}
            if len(roots) != 1:
                raise RuntimeError("Step-Audio-EditX source archive has an unexpected layout")
            root = roots.pop()
            for member in members:
                relative = Path(member.name).relative_to(root)
                if relative == Path(".") or member.issym() or member.islnk():
                    continue
                destination = (target / relative).resolve()
                if not destination.is_relative_to(target.resolve()):
                    raise RuntimeError("Step-Audio-EditX source archive contains an unsafe path")
                member.name = str(relative)
                tar.extract(member, target, filter="data")
    if not _source_complete(target):
        raise RuntimeError("Step-Audio-EditX source archive is incomplete")
    (target / ".vox-source-ref").write_text(f"{SOURCE_REF}\n", encoding="utf-8")


def _source_matches(path: Path) -> bool:
    marker = path / ".vox-source-ref"
    return _source_complete(path) and marker.is_file() and marker.read_text(encoding="utf-8").strip() == SOURCE_REF


def _worker_paths(path: Path) -> list[str]:
    import vox

    candidates = (
        path,
        source_dir(path),
        Path(vox.__file__).resolve().parents[1],
        Path(__file__).resolve().parents[1],
        Path(sysconfig.get_paths()["purelib"]),
    )
    values: list[str] = []
    for candidate in candidates:
        value = str(candidate)
        if value not in values:
            values.append(value)
    return values


def worker_env(path: Path, device: str) -> dict[str, str]:
    names = {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
    prefixes = ("CUDA_", "NVIDIA_", "HF_", "HUGGINGFACE_", "PYTORCH_", "VLLM_")
    env = {
        name: value
        for name, value in os.environ.items()
        if name in names or name.startswith(prefixes)
    }
    env["PYTHONPATH"] = os.pathsep.join(_worker_paths(path))
    env["VOX_STEP_AUDIO_EDITX_DEVICE"] = device
    disabled_kernels = {
        name
        for name in env.get("VLLM_DISABLED_KERNELS", "").split(",")
        if name
    }
    disabled_kernels.add("MarlinLinearKernel")
    env["VLLM_DISABLED_KERNELS"] = ",".join(sorted(disabled_kernels))
    return env


def _probe_runtime(path: Path) -> bool:
    script = (
        "from pathlib import Path; "
        "import conch, hyperpyyaml, model_loader, onnxruntime, tokenizer, torch, torchaudio, "
        "torchvision, transformers, vllm, "
        "vllm._C, whisper; "
        f"root=Path({str(path)!r}).resolve(); "
        "assert torch.__version__.startswith('2.10.'); "
        "assert torchaudio.__version__.startswith('2.10.'); "
        "assert vllm.__version__ == '0.18.0'; "
        "assert not Path(torch.__file__).resolve().is_relative_to(root); "
        "assert not Path(torchaudio.__file__).resolve().is_relative_to(root); "
        "from vllm import LLM, SamplingParams"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=worker_env(path, "cuda"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        logger.warning("Step-Audio-EditX runtime probe failed: %s", result.stderr.strip())
    return result.returncode == 0


def ensure_runtime() -> Path:
    target = runtime_dir()
    sentinel = target / RUNTIME_SENTINEL
    _remove_shared_runtime_stack(target)
    if sentinel.is_file() and _source_matches(source_dir(target)) and _probe_runtime(target):
        return target

    with staged_target_runtime(target, preserve_existing=False) as stage:
        installed = install_target_runtime_requirements(
            stage,
            RUNTIME_REQUIREMENTS,
            no_deps=True,
            upgrade=False,
            timeout=1800,
            expected_paths=tuple(stage / relative for relative in EXPECTED_RUNTIME_PATHS),
            installer_order=("uv", "pip"),
            install_runner=_run_install_command,
            context="Step-Audio-EditX runtime install",
        )
        if not installed:
            raise RuntimeError("Failed to install Step-Audio-EditX runtime dependencies")
        _remove_shared_runtime_stack(stage)
        stage_source = source_dir(stage)
        stage_source.mkdir(parents=True, exist_ok=True)
        _extract_source(stage_source)
        if not _probe_runtime(stage):
            raise RuntimeError("Step-Audio-EditX runtime verification failed")
        (stage / RUNTIME_SENTINEL).touch()
    return target
