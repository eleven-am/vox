from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from vox.core.adapter import TurnDetectorAdapter
from vox.core.adapter_acquisition import acquire_typed_adapter

logger = logging.getLogger(__name__)

EOU_MODEL_ID = "livekit/turn-detector"
EOU_MODEL_FILE = "model_q8.onnx"
EOU_MODEL_SUBFOLDER = "onnx"
EOU_MODEL_REVISION = "v1.2.2-en"
DEFAULT_TURN_DETECTOR = "livekit"
SMART_TURN_MODEL = "smart-turn:v3.2"
MAX_HISTORY_TURNS = 4


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class EOUConfig:
    model: str = DEFAULT_TURN_DETECTOR
    threshold: float = 0.5
    max_context_turns: int = MAX_HISTORY_TURNS
    max_pending_tokens: int = 60


class TurnDetector(Protocol):
    def token_count(self, text: str) -> int: ...

    def predict(
        self,
        turns: list[ConversationTurn],
        *,
        audio: NDArray[np.float32],
        sample_rate: int,
        max_context_turns: int = MAX_HISTORY_TURNS,
    ) -> float | Awaitable[float]: ...


class EOUModel:
    _instance: EOUModel | None = None
    _session: Any = None
    _tokenizer: Any = None
    _lock = threading.Lock()

    def __new__(cls) -> EOUModel:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def _ensure_loaded(self) -> None:
        if EOUModel._session is not None:
            return
        with EOUModel._lock:
            if EOUModel._session is not None:
                return

            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer

            started = time.perf_counter()
            model_path = hf_hub_download(
                repo_id=EOU_MODEL_ID,
                filename=EOU_MODEL_FILE,
                subfolder=EOU_MODEL_SUBFOLDER,
                revision=EOU_MODEL_REVISION,
            )
            EOUModel._tokenizer = AutoTokenizer.from_pretrained(
                EOU_MODEL_ID,
                revision=EOU_MODEL_REVISION,
            )
            EOUModel._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            logger.info(
                "Loaded LiveKit turn detector in %.2fs; %s remains available as the open audio-native alternative",
                time.perf_counter() - started,
                SMART_TURN_MODEL,
            )

    async def preload(self) -> None:
        await asyncio.to_thread(self._ensure_loaded)

    def token_count(self, text: str) -> int:
        if not text:
            return 0
        self._ensure_loaded()
        if EOUModel._tokenizer is None:
            return max(len(text) // 4, 1)
        try:
            return len(EOUModel._tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            logger.exception("EOU tokenizer failed; falling back to char estimate")
            return max(len(text) // 4, 1)

    def predict(
        self,
        turns: list[ConversationTurn],
        *,
        audio: NDArray[np.float32] | None = None,
        sample_rate: int = 16_000,
        max_context_turns: int = MAX_HISTORY_TURNS,
    ) -> float:
        del audio, sample_rate
        self._ensure_loaded()
        if not turns:
            return 0.0

        history_limit = max(1, int(max_context_turns or MAX_HISTORY_TURNS))
        messages = [{"role": turn.role, "content": turn.content} for turn in turns[-history_limit:]]
        text = EOUModel._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_special_tokens=False,
        )
        end_index = text.rfind("<|im_end|>")
        if end_index != -1:
            text = text[:end_index]

        inputs = EOUModel._tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=128,
        )
        outputs = EOUModel._session.run(None, {"input_ids": inputs["input_ids"]})
        if not outputs:
            raise RuntimeError("LiveKit turn detector returned no outputs")

        values = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if values.size == 1:
            return float(np.clip(values[0], 0.0, 1.0))
        if values.size == 2:
            shifted = values - np.max(values)
            probabilities = np.exp(shifted) / np.exp(shifted).sum()
            return float(probabilities[1])
        raise RuntimeError(f"LiveKit turn detector returned unexpected output shape {values.shape}")


class TenTurnDetector:
    MODEL_ID = "TEN-framework/TEN_Turn_Detection"

    _instance: TenTurnDetector | None = None
    _model: Any = None
    _tokenizer: Any = None
    _torch: Any = None
    _lock = threading.Lock()

    def __new__(cls) -> TenTurnDetector:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def _ensure_loaded(self) -> None:
        if TenTurnDetector._model is not None:
            return
        with TenTurnDetector._lock:
            if TenTurnDetector._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "TEN turn detector requires optional dependencies: torch and transformers"
                ) from exc

            started = time.perf_counter()
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            TenTurnDetector._torch = torch
            TenTurnDetector._tokenizer = tokenizer
            TenTurnDetector._model = model
            logger.info("TEN turn detector loaded in %.2fs", time.perf_counter() - started)

    def token_count(self, text: str) -> int:
        if not text:
            return 0
        self._ensure_loaded()
        try:
            return len(TenTurnDetector._tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            logger.exception("TEN tokenizer failed; falling back to char estimate")
            return max(len(text) // 4, 1)

    async def preload(self) -> None:
        await asyncio.to_thread(self._ensure_loaded)

    def classify(
        self,
        turns: list[ConversationTurn],
        *,
        max_context_turns: int = MAX_HISTORY_TURNS,
    ) -> str:
        self._ensure_loaded()
        if not turns:
            return "unfinished"

        history_limit = max(1, int(max_context_turns or MAX_HISTORY_TURNS))
        text = "\n".join(f"{turn.role}: {turn.content}" for turn in turns[-history_limit:])
        input_ids = TenTurnDetector._tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if TenTurnDetector._torch.cuda.is_available():
            input_ids = input_ids.cuda()

        with TenTurnDetector._torch.no_grad():
            outputs = TenTurnDetector._model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=TenTurnDetector._tokenizer.eos_token_id,
            )
        response = outputs[0][input_ids.shape[-1] :]
        label = TenTurnDetector._tokenizer.decode(response, skip_special_tokens=True).strip().lower()
        if label in {"finished", "finish", "done"}:
            return "finished"
        if label in {"wait", "stop", "pause"}:
            return "wait"
        return "unfinished"

    def predict(
        self,
        turns: list[ConversationTurn],
        *,
        audio: NDArray[np.float32] | None = None,
        sample_rate: int = 16_000,
        max_context_turns: int = MAX_HISTORY_TURNS,
    ) -> float:
        del audio, sample_rate
        label = self.classify(turns, max_context_turns=max_context_turns)
        return 1.0 if label == "finished" else 0.0


class ModelTurnDetector:
    def __init__(self, scheduler: Any, model: str) -> None:
        self._scheduler = scheduler
        self._model = model

    def token_count(self, text: str) -> int:
        return max(len(text) // 4, 1) if text else 0

    async def preload(self) -> None:
        async with acquire_typed_adapter(
            self._scheduler,
            model=self._model,
            adapter_type=TurnDetectorAdapter,
            expected_type="turn detector",
        ):
            pass

    async def predict(
        self,
        turns: list[ConversationTurn],
        *,
        audio: NDArray[np.float32],
        sample_rate: int,
        max_context_turns: int = MAX_HISTORY_TURNS,
    ) -> float:
        del turns, max_context_turns
        if audio.size == 0:
            return 0.0
        async with acquire_typed_adapter(
            self._scheduler,
            model=self._model,
            adapter_type=TurnDetectorAdapter,
            expected_type="turn detector",
        ) as adapter:
            probability = await asyncio.to_thread(
                adapter.predict,
                audio,
                sample_rate=sample_rate,
            )
        return float(np.clip(probability, 0.0, 1.0))


def create_turn_detector(name: str, *, scheduler: Any | None = None) -> TurnDetector:
    selected = (name or DEFAULT_TURN_DETECTOR).strip()
    normalized = selected.lower().replace("_", "-")
    if normalized in {"livekit", "livekit-eou", "eou"}:
        return EOUModel()
    if normalized in {"ten", "ten-turn", "ten-turn-detection"}:
        return TenTurnDetector()
    if scheduler is None:
        raise ValueError(f"turn detector model {selected!r} requires a scheduler")
    return ModelTurnDetector(scheduler, selected)
