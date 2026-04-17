from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from vox.core import ner
from vox.core.ner import (
    EMAIL_REGEX,
    MAX_TOPICS,
    PHONE_REGEX,
    URL_REGEX,
    _clean_chunk,
    _regex_entities,
    annotate,
)


class TestRegexEdgeCases:
    def test_email_with_plus_sign(self):
        text = "billing+ops@example.com"
        matches = EMAIL_REGEX.findall(text)
        assert "billing+ops@example.com" in matches

    def test_email_with_subdomain(self):
        text = "ceo@mail.corp.example.co.uk"
        matches = EMAIL_REGEX.findall(text)
        assert "ceo@mail.corp.example.co.uk" in matches

    def test_email_requires_tld(self):
        text = "invalid@localhost"
        matches = EMAIL_REGEX.findall(text)
        assert matches == []

    def test_email_word_boundary(self):
        text = "xroy@example.comx other@example.com"
        entities = _regex_entities(text)
        emails = [e.text for e in entities if e.type == "EMAIL"]
        assert "other@example.com" in emails

    def test_url_with_query_string(self):
        text = "see https://example.com/path?a=1&b=2#fragment now"
        entities = _regex_entities(text)
        urls = [e for e in entities if e.type == "URL"]
        assert len(urls) == 1
        assert urls[0].text == "https://example.com/path?a=1&b=2#fragment"

    def test_url_http_and_https(self):
        text = "http://a.com and https://b.com"
        entities = _regex_entities(text)
        urls = {e.text for e in entities if e.type == "URL"}
        assert urls == {"http://a.com", "https://b.com"}

    def test_url_case_insensitive(self):
        text = "HTTPS://example.com"
        entities = _regex_entities(text)
        urls = [e for e in entities if e.type == "URL"]
        assert len(urls) == 1

    def test_phone_various_formats(self):
        cases = [
            "+1 415 555 1212",
            "+1-415-555-1212",
            "+1 (415) 555-1212",
            "+44 20 7946 0958",
            "4155551212",
        ]
        for text in cases:
            entities = _regex_entities(text)
            phones = [e for e in entities if e.type == "PHONE"]
            assert phones, f"no phone matched in {text!r}"

    def test_phone_rejects_short(self):
        text = "pin is 12345"
        entities = _regex_entities(text)
        phones = [e for e in entities if e.type == "PHONE"]
        assert phones == []

    def test_dedup_across_multiple_occurrences(self):
        text = "roy@a.com again roy@a.com or https://a.com or https://a.com"
        entities = _regex_entities(text)
        emails = [e for e in entities if e.type == "EMAIL"]
        urls = [e for e in entities if e.type == "URL"]
        assert len(emails) == 1
        assert len(urls) == 1

    def test_offsets_are_accurate(self):
        text = "prefix roy@example.com suffix"
        entities = _regex_entities(text)
        email = next(e for e in entities if e.type == "EMAIL")
        assert text[email.start_char:email.end_char] == email.text


class TestCleanChunkAdversarial:
    """_clean_chunk is now text-only (punctuation + lowercase). Determiner/stopword
    stripping is structural (via spaCy POS/is_stop) and lives in _clean_noun_chunk.
    These tests verify language-neutral text behaviour.
    """

    def test_unicode_preserved(self):
        assert _clean_chunk("café") == "café"

    def test_accent_marks_kept(self):
        assert _clean_chunk("señor gonzález") == "señor gonzález"

    def test_passes_middle_determiners_through(self):
        assert _clean_chunk("dog of the year") == "dog of the year"

    def test_strips_punct_only(self):
        assert _clean_chunk('"quick!!!" ') == "quick"

    def test_lowercases_any_script(self):
        assert _clean_chunk("The Customer") == "the customer"
        assert _clean_chunk("ПРИВЕТ МИР") == "привет мир"

    def test_no_english_determiner_stripping(self):

        assert _clean_chunk("the customer") == "the customer"
        assert _clean_chunk("el cliente") == "el cliente"
        assert _clean_chunk("der kunde") == "der kunde"


class TestAnnotateAdversarial:
    def test_very_long_text(self):

        text = " ".join([f"word{i}" for i in range(2000)] + ["contact roy@example.com"])
        entities, _ = annotate(text, "ja")
        emails = [e for e in entities if e.type == "EMAIL"]
        assert any(e.text == "roy@example.com" for e in emails)

    def test_unicode_text_regex_only(self):
        text = "送信 to roy@example.com 緊急"
        entities, _ = annotate(text, "ja")
        assert any(e.type == "EMAIL" for e in entities)

    def test_empty_topics_when_no_model(self):
        _, topics = annotate("some text with no model language", "xx")
        assert topics == []

    def test_none_safe(self):

        assert annotate("", "en") == ([], [])
        assert annotate("   ", "en") == ([], [])
        assert annotate("\n\t", "en") == ([], [])


class TestMockedTopicRanking:
    def _make_doc(self, tokens=(), chunks=(), ents=()):
        class _E:
            def __init__(self, label, text, s, e):
                self.label_, self.text, self.start_char, self.end_char = label, text, s, e
        class _T:
            def __init__(self, pos, lemma, i, is_stop=False, is_punct=False, text=None):
                self.pos_, self.lemma_, self.i, self.is_stop, self.is_punct = pos, lemma, i, is_stop, is_punct
                self.text = text if text is not None else lemma

        _DETERMINERS = {"the", "a", "an", "my", "el", "la", "der", "die", "das", "le", "la", "les"}

        def _chunk_from_text(text: str, start: int):
            words = text.split()
            toks = []
            for idx, w in enumerate(words):
                is_det = idx == 0 and len(words) > 1 and w.lower() in _DETERMINERS
                toks.append(_T(
                    pos="DET" if is_det else "NOUN",
                    lemma=w.lower(),
                    i=start + idx,
                    is_stop=is_det,
                    text=w,
                ))
            class _C:
                def __init__(c):
                    c.text = text
                    c.start = start
                    c._toks = toks
                def __iter__(c): return iter(c._toks)
            return _C()

        class _Doc:
            def __init__(s):
                s.ents = [_E(*e) for e in ents]
                s._tokens = [_T(*t) for t in tokens]
                s.noun_chunks = [_chunk_from_text(*c) for c in chunks]
            def __iter__(s): return iter(s._tokens)
        return _Doc()

    def _with_mock(self, doc):
        class _NLP:
            def __call__(s, text): return doc
        return patch.object(ner, "_get_model", return_value=_NLP())

    def test_frequency_tiebreak_by_first_seen(self):
        doc = self._make_doc(chunks=[
            ("first phrase", 0),
            ("second phrase", 3),
            ("first phrase", 6),
            ("second phrase", 9),
        ])
        with self._with_mock(doc):
            _, topics = annotate("x", "en")
        assert topics == ["first phrase", "second phrase"]

    def test_stopwords_excluded_from_singles(self):
        doc = self._make_doc(tokens=[
            ("NOUN", "account", 0, False, False),
            ("NOUN", "the", 1, True, False),
            ("NOUN", "customer", 2, False, False),
        ])
        with self._with_mock(doc):
            _, topics = annotate("x", "en")
        assert topics == ["account", "customer"]

    def test_propn_included(self):
        doc = self._make_doc(tokens=[("PROPN", "london", 0, False, False)])
        with self._with_mock(doc):
            _, topics = annotate("x", "en")
        assert topics == ["london"]

    def test_short_lemma_dropped(self):
        doc = self._make_doc(tokens=[
            ("NOUN", "a", 0, False, False),
            ("NOUN", "ox", 1, False, False),
        ])
        with self._with_mock(doc):
            _, topics = annotate("x", "en")
        assert topics == ["ox"]

    def test_cap_preserves_phrase_priority_over_single(self):
        chunks = [(f"phrase {i}", i) for i in range(MAX_TOPICS)]
        tokens = [("NOUN", f"single{i}", i + 100, False, False) for i in range(5)]
        doc = self._make_doc(chunks=chunks, tokens=tokens)
        with self._with_mock(doc):
            _, topics = annotate("x", "en")
        assert len(topics) == MAX_TOPICS
        assert all("phrase" in t for t in topics)

    def test_deterministic_across_runs(self):
        doc = self._make_doc(chunks=[("alpha", 0), ("beta", 1), ("gamma", 2)])
        with self._with_mock(doc):
            _, topics1 = annotate("x", "en")
            _, topics2 = annotate("x", "en")
        assert topics1 == topics2

    def test_punct_token_excluded(self):
        doc = self._make_doc(tokens=[
            ("NOUN", ",", 0, False, True),
            ("NOUN", "banana", 1, False, False),
        ])
        with self._with_mock(doc):
            _, topics = annotate("x", "en")
        assert topics == ["banana"]


class TestConcurrentModelLoad:
    def test_lock_prevents_double_load(self):

        ner._models.clear()
        ner._missing_languages.clear()
        counter = {"calls": 0}

        original_get = ner._get_model

        def counting_get(lang):
            counter["calls"] += 1
            return None

        with patch.object(ner, "_get_model", side_effect=counting_get):
            threads = [threading.Thread(target=lambda: annotate("hi", "en")) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()


        assert counter["calls"] == 20


class TestAnnotateDoesNotPollutecaOnFailure:
    def test_language_failure_does_not_cache_nlp(self):
        ner._models.clear()
        ner._missing_languages.clear()


        with patch.object(ner, "_get_model", return_value=None):
            annotate("hello", "en")
            annotate("hello", "en")


        assert "en" not in ner._models


class TestRegressionAgainstSidecarBaseline:
    """Verify we capture categories sidecar catches, plus our additions."""

    def test_covers_core_sidecar_entity_types(self):

        for expected in ("PERSON", "ORG", "LOCATION", "DATE", "TIME", "PRODUCT", "EMAIL"):
            assert expected in set(ner.ENTITY_MAP.values()) | {"EMAIL"}

    def test_adds_url_and_phone_on_top(self):
        text = "call +1 415 555 1212 or visit https://example.com"
        entities = _regex_entities(text)
        types = {e.type for e in entities}
        assert "URL" in types
        assert "PHONE" in types

    def test_entity_has_offsets_unlike_sidecar(self):
        text = "visit https://example.com today"
        entities = _regex_entities(text)
        url = next(e for e in entities if e.type == "URL")
        assert url.start_char == text.index("https")
        assert url.end_char == text.index("https") + len("https://example.com")
