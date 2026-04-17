from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from vox.streaming.eou import ConversationTurn, EOUConfig, EOUModel
from vox.streaming.pipeline import StreamPipeline, StreamPipelineConfig


class TestEOUConfig:
    def test_default_max_pending_tokens(self):
        config = EOUConfig()
        assert config.max_pending_tokens == 60

    def test_max_pending_tokens_overridable(self):
        config = EOUConfig(max_pending_tokens=100)
        assert config.max_pending_tokens == 100


class TestEOUModelTokenCount:
    def test_empty_text_returns_zero(self):
        model = EOUModel()
        assert model.token_count("") == 0

    def test_uses_tokenizer_when_loaded(self):
        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        model = EOUModel()
        with (
            patch.object(EOUModel, "_ensure_loaded"),
            patch.object(EOUModel, "_tokenizer", fake_tokenizer),
        ):
            assert model.token_count("hello world") == 5
        fake_tokenizer.encode.assert_called_once_with("hello world", add_special_tokens=False)

    def test_falls_back_when_tokenizer_missing(self):
        model = EOUModel()
        with (
            patch.object(EOUModel, "_ensure_loaded"),
            patch.object(EOUModel, "_tokenizer", None),
        ):

            assert model.token_count("a" * 16) == 4

            assert model.token_count("abc") == 1

    def test_handles_tokenizer_exception(self):
        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.side_effect = RuntimeError("tokenizer broken")

        model = EOUModel()
        with (
            patch.object(EOUModel, "_ensure_loaded"),
            patch.object(EOUModel, "_tokenizer", fake_tokenizer),
        ):

            assert model.token_count("a" * 20) == 5


class TestTokenFlushIsLanguageNeutral:
    """Verify the token-count check behaves consistently across scripts.

    A 200-char Chinese paragraph contains roughly 5-10x more semantic content
    than a 200-char English sentence. Token-count normalises this.
    """

    def test_chinese_text_hits_token_budget_faster_than_char_budget(self):
        fake_tokenizer = MagicMock()

        fake_tokenizer.encode.side_effect = lambda text, **_: list(range(int(len(text) * 1.5)))

        model = EOUModel()
        with (
            patch.object(EOUModel, "_ensure_loaded"),
            patch.object(EOUModel, "_tokenizer", fake_tokenizer),
        ):

            assert model.token_count("中" * 50) > 60

    def test_english_text_gets_reasonable_token_count(self):
        fake_tokenizer = MagicMock()

        fake_tokenizer.encode.side_effect = lambda text, **_: list(range(len(text) // 4))

        model = EOUModel()
        with (
            patch.object(EOUModel, "_ensure_loaded"),
            patch.object(EOUModel, "_tokenizer", fake_tokenizer),
        ):

            assert model.token_count("a" * 200) < 60


class TestEOUContextWindow:
    def test_predict_respects_overridden_max_context_turns(self):
        fake_tokenizer = MagicMock()
        fake_tokenizer.apply_chat_template.return_value = "prompt"
        fake_tokenizer.return_value = {"input_ids": np.array([[1, 2, 3]], dtype=np.int64)}
        fake_session = MagicMock()
        fake_session.run.return_value = [np.array([[0.0, 1.0]], dtype=np.float32)]

        turns = [
            ConversationTurn(role="user", content="one"),
            ConversationTurn(role="assistant", content="two"),
            ConversationTurn(role="user", content="three"),
        ]

        model = EOUModel()
        with (
            patch.object(EOUModel, "_ensure_loaded"),
            patch.object(EOUModel, "_tokenizer", fake_tokenizer),
            patch.object(EOUModel, "_session", fake_session),
        ):
            model.predict(turns, max_context_turns=2)

        fake_tokenizer.apply_chat_template.assert_called_once_with(
            [
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
            tokenize=False,
            add_special_tokens=False,
        )

    def test_pipeline_history_trim_respects_configured_context_window(self):
        pipeline = StreamPipeline(
            scheduler=MagicMock(),
            config=StreamPipelineConfig(eou_config=EOUConfig(max_context_turns=2)),
        )

        for idx in range(6):
            pipeline.add_assistant_turn(f"turn-{idx}")

        assert [turn.content for turn in pipeline._conversation_history] == [
            "turn-2",
            "turn-3",
            "turn-4",
            "turn-5",
        ]
