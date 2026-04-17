from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

EOU_MODEL_ID = "livekit/turn-detector"
EOU_MODEL_FILE = "model_q8.onnx"
EOU_MODEL_SUBFOLDER = "onnx"
EOU_MODEL_REVISION = "v1.2.0"
MAX_HISTORY_TURNS = 4


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class EOUConfig:
    threshold: float = 0.5
    max_context_turns: int = MAX_HISTORY_TURNS




    max_pending_tokens: int = 60


class EOUModel:
    _instance: EOUModel | None = None
    _session = None
    _tokenizer = None
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
            if EOUModel._session is None:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from transformers import AutoTokenizer

                logger.info("Loading EOUModel from %s", EOU_MODEL_ID)
                start = time.perf_counter()

                model_path = hf_hub_download(
                    repo_id=EOU_MODEL_ID,
                    filename=EOU_MODEL_FILE,
                    subfolder=EOU_MODEL_SUBFOLDER,
                    revision=EOU_MODEL_REVISION,
                )

                EOUModel._tokenizer = AutoTokenizer.from_pretrained(EOU_MODEL_ID)
                EOUModel._session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )

                elapsed = time.perf_counter() - start
                logger.info("EOUModel loaded in %.2fs", elapsed)

    def token_count(self, text: str) -> int:
        """Return the number of tokens in `text` using the EOU tokenizer.

        Language-neutral because the tokenizer is the same one used for turn detection,
        trained on multilingual chat text. Returns a char-based fallback if the
        tokenizer isn't loaded yet.
        """
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
        max_context_turns: int = MAX_HISTORY_TURNS,
    ) -> float:
        self._ensure_loaded()

        if not turns:
            return 0.0

        history_limit = max(1, int(max_context_turns or MAX_HISTORY_TURNS))
        messages = [{"role": t.role, "content": t.content} for t in turns[-history_limit:]]
        text = EOUModel._tokenizer.apply_chat_template(
            messages, tokenize=False, add_special_tokens=False
        )

        ix = text.rfind("<|im_end|>")
        if ix != -1:
            text = text[:ix]

        inputs = EOUModel._tokenizer(
            text, return_tensors="np", truncation=True, max_length=512
        )

        outputs = EOUModel._session.run(None, {"input_ids": inputs["input_ids"]})

        if not outputs or len(outputs) == 0 or len(outputs[0]) == 0:
            logger.warning("EOU model returned empty output")
            return 0.0

        logits = np.atleast_1d(outputs[0][0])
        if logits.size < 2:
            if logits.size == 1:
                return float(1.0 / (1.0 + np.exp(-logits.item())))
            logger.warning("EOU model returned unexpected logits shape: %s", logits.shape)
            return 0.0

        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()
        return float(probabilities[1])
