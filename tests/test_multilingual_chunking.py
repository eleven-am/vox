from __future__ import annotations

from vox.conversation.text_buffer import (
    split_by_chars as _split_by_chars,
    split_by_words as _split_by_words,
    split_clauses as _split_clauses,
    split_for_tts as _chunk_text,
    split_long_sentence as _split_long_sentence,
    split_sentences as _split_sentences,
)


class TestSplitSentencesLatin:
    def test_period_bang_question(self):
        sentences = _split_sentences("Hello. World! Really?")
        assert sentences == ["Hello.", "World!", "Really?"]

    def test_single_sentence_no_terminator(self):
        assert _split_sentences("just text") == ["just text"]

    def test_empty(self):
        assert _split_sentences("") == []

    def test_whitespace_only(self):
        assert _split_sentences("   \n  ") == []


class TestSplitSentencesCJK:
    def test_chinese_terminators(self):
        assert _split_sentences("你好。世界。") == ["你好。", "世界。"]

    def test_japanese_terminators(self):
        assert _split_sentences("こんにちは！元気ですか？") == ["こんにちは！", "元気ですか？"]

    def test_mixed_latin_and_chinese(self):
        assert _split_sentences("Hello. 你好。Bonjour.") == ["Hello.", "你好。", "Bonjour."]

    def test_fullwidth_period(self):
        assert _split_sentences("one．two．") == ["one．", "two．"]


class TestSplitSentencesOtherScripts:
    def test_hindi_devanagari(self):
        assert _split_sentences("नमस्ते।संसार।") == ["नमस्ते।", "संसार।"]

    def test_arabic_question(self):
        assert _split_sentences("ماذا؟ أهلاً.") == ["ماذا؟", "أهلاً."]


class TestSplitClauses:
    def test_latin_commas(self):
        assert _split_clauses("one, two, three") == ["one,", "two,", "three"]

    def test_chinese_commas(self):
        assert _split_clauses("第一，第二，第三") == ["第一，", "第二，", "第三"]

    def test_no_clause_boundary(self):
        assert _split_clauses("a single phrase") == ["a single phrase"]


class TestSplitByWords:
    def test_caps_at_max_chars(self):
        result = _split_by_words("the quick brown fox jumped over", 10)
        for chunk in result:
            assert len(chunk) <= 10

    def test_single_huge_word_exceeds_cap(self):

        result = _split_by_words("supercalifragilistic", 5)
        assert result == ["supercalifragilistic"]


class TestSplitByChars:
    def test_slices_cjk_text(self):
        text = "一二三四五六七八九十"
        assert _split_by_chars(text, 3) == ["一二三", "四五六", "七八九", "十"]

    def test_empty_input(self):
        assert _split_by_chars("", 5) == []

    def test_zero_or_negative_cap_returns_whole(self):
        assert _split_by_chars("text", 0) == ["text"]


class TestChunkTextMultilingual:
    def test_english_paragraph_fits_under_cap(self):
        text = "Hello world. This is a test."
        assert _chunk_text(text, max_chars=200) == [text]

    def test_chinese_paragraph_splits_at_sentence_boundary(self):
        text = "你好世界。这是一个测试。再见世界。"

        chunks = _chunk_text(text, max_chars=8)
        assert len(chunks) >= 2
        assert all("。" in c or len(c) <= 8 for c in chunks)

    def test_long_chinese_sentence_falls_back_to_char_split(self):

        text = "一二三四五六七八九十一二三四五六七八九十"
        chunks = _chunk_text(text, max_chars=5)
        assert len(chunks) == 4
        assert chunks == ["一二三四五", "六七八九十", "一二三四五", "六七八九十"]

    def test_long_latin_sentence_falls_back_to_word_split(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        chunks = _chunk_text(text, max_chars=15)

        for c in chunks:
            for word in c.split():
                assert word in text

    def test_empty_text_returns_empty(self):
        assert _chunk_text("", max_chars=100) == []

    def test_whitespace_only_returns_empty(self):
        assert _chunk_text("   \n\t ", max_chars=100) == []

    def test_mixed_script_paragraph(self):
        text = "First sentence. 第二句。Third sentence."
        chunks = _chunk_text(text, max_chars=20)

        assert any("First sentence" in c for c in chunks)
        assert any("第二句" in c for c in chunks)
        assert any("Third sentence" in c for c in chunks)


class TestSplitLongSentenceFallbackChoice:
    def test_whitespace_sentence_uses_word_split(self):
        sentence = "alpha beta gamma delta epsilon zeta eta"
        result = _split_long_sentence(sentence, max_chars=12)

        for c in result:
            assert all(w.isalpha() for w in c.split())

    def test_whitespace_free_sentence_uses_char_split(self):
        sentence = "一二三四五六七八九十"
        result = _split_long_sentence(sentence, max_chars=3)
        assert result == ["一二三", "四五六", "七八九", "十"]

    def test_clause_boundaries_preferred_over_char_split(self):
        sentence = "第一，第二，第三"
        result = _split_long_sentence(sentence, max_chars=3)

        assert any("，" in c for c in result) or len(result) >= 3
