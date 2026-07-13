from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vox.core.adapter import TurnDetectorAdapter
from vox.core.adapter_runtime import (
    activate_runtime_path,
    install_target_runtime_requirements,
    module_available_in_runtime,
    target_runtime,
)
from vox.core.types import AdapterInfo, ModelFormat, ModelType

_RUNTIME_NAME = "smart-turn"
_RUNTIME_REQUIREMENTS = (
    "transformers>=4.57.6,<5.0",
    "tokenizers>=0.22,<0.23",
    "safetensors>=0.5,<1.0",
)
_RUNTIME_IMPORTS = ("transformers", "tokenizers", "safetensors")
_SAMPLE_RATE = 16_000
_MAX_AUDIO_SECONDS = 8


def _runtime_path() -> Path:
    return target_runtime(_RUNTIME_NAME).path


def _runtime_ready(runtime_path: Path) -> bool:
    return all(
        module_available_in_runtime(
            import_name,
            runtime_path,
            include_app_fallback=False,
        )
        for import_name in _RUNTIME_IMPORTS
    )


def _ensure_runtime() -> Path:
    runtime_path = _runtime_path()
    runtime_path.mkdir(parents=True, exist_ok=True)
    activate_runtime_path(runtime_path, root=runtime_path.parent)
    if _runtime_ready(runtime_path):
        return runtime_path
    installed = install_target_runtime_requirements(
        runtime_path,
        _RUNTIME_REQUIREMENTS,
        no_deps=True,
        context="Smart Turn runtime install",
    )
    activate_runtime_path(runtime_path, root=runtime_path.parent)
    if not installed or not _runtime_ready(runtime_path):
        raise RuntimeError("Smart Turn runtime dependencies could not be prepared")
    return runtime_path


class SmartTurnV3Adapter(TurnDetectorAdapter):
    def __init__(self) -> None:
        self._session: Any = None
        self._feature_extractor: Any = None
        self._device = "cpu"
        self._provider = "CPUExecutionProvider"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="smart-turn-v3",
            type=ModelType.TURN,
            architectures=("smart-turn-v3",),
            default_sample_rate=_SAMPLE_RATE,
            supported_formats=(ModelFormat.ONNX,),
        )

    def prepare_runtime(self) -> None:
        _ensure_runtime()

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        runtime_path = _ensure_runtime()
        activate_runtime_path(runtime_path, root=runtime_path.parent)

        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        provider_name = str(kwargs.get("provider", "cpu")).strip().lower()
        provider = "CUDAExecutionProvider" if provider_name in {"cuda", "gpu"} else "CPUExecutionProvider"
        available = ort.get_available_providers()
        if provider not in available:
            raise RuntimeError(
                f"Smart Turn requested {provider}, but ONNX Runtime provides {available}"
            )

        model_dir = Path(model_path)
        filename = str(
            kwargs.get(
                "model_file",
                "smart-turn-v3.2-gpu.onnx" if provider == "CUDAExecutionProvider" else "smart-turn-v3.2-cpu.onnx",
            )
        )
        onnx_path = model_dir / filename
        if not onnx_path.is_file():
            candidates = sorted(model_dir.glob("*.onnx"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"Smart Turn ONNX model not found: {onnx_path}")
            onnx_path = candidates[0]

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=_MAX_AUDIO_SECONDS)
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=[provider],
        )
        self._provider = provider
        self._device = "cuda" if provider == "CUDAExecutionProvider" else "cpu"

    def unload(self) -> None:
        self._session = None
        self._feature_extractor = None
        self._device = "cpu"
        self._provider = "CPUExecutionProvider"

    @property
    def is_loaded(self) -> bool:
        return self._session is not None and self._feature_extractor is not None

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        provider = str(kwargs.get("provider", "cpu")).strip().lower()
        return 64 * 1024 * 1024 if provider in {"cuda", "gpu"} else 0

    def memory_status(self) -> dict[str, Any]:
        return {"provider": self._provider}

    def predict(
        self,
        audio: NDArray[np.float32],
        *,
        sample_rate: int,
    ) -> float:
        if not self.is_loaded:
            raise RuntimeError("Smart Turn adapter is not loaded")
        if sample_rate != _SAMPLE_RATE:
            raise ValueError(f"Smart Turn requires {_SAMPLE_RATE}Hz audio, got {sample_rate}Hz")

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        samples = samples[-(_SAMPLE_RATE * _MAX_AUDIO_SECONDS) :]
        if samples.size == 0:
            return 0.0
        peak = float(np.max(np.abs(samples)))
        if peak > 1.0:
            samples = samples / peak

        inputs = self._feature_extractor(
            samples,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=_SAMPLE_RATE * _MAX_AUDIO_SECONDS,
            truncation=True,
            do_normalize=True,
        )
        features = np.expand_dims(
            np.asarray(inputs.input_features, dtype=np.float32).squeeze(0),
            axis=0,
        )
        outputs = self._session.run(None, {"input_features": features})
        if not outputs:
            raise RuntimeError("Smart Turn returned no outputs")
        values = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if values.size != 1:
            raise RuntimeError(f"Smart Turn returned unexpected output shape {values.shape}")
        return float(np.clip(values[0], 0.0, 1.0))
