from __future__ import annotations

from unittest.mock import patch

import pytest

from vox.core import ner
from vox.core.ner import (
    MAX_TOPICS,
    Entity,
    _clean_chunk,
    _regex_entities,
    annotate,
    entity_to_dict,
)


class TestRegexEntities:
    def test_email_detected(self):
        entities = _regex_entities("contact me at roy@example.com please")
        emails = [e for e in entities if e.type == "EMAIL"]
        assert len(emails) == 1
        assert emails[0].text == "roy@example.com"
        assert emails[0].start_char == 14
        assert emails[0].end_char == 29

    def test_url_detected(self):
        entities = _regex_entities("visit https://example.com/docs for details")
        urls = [e for e in entities if e.type == "URL"]
        assert len(urls) == 1
        assert urls[0].text == "https://example.com/docs"

    def test_phone_detected(self):
        entities = _regex_entities("call me at +1 (415) 555-1212 tomorrow")
        phones = [e for e in entities if e.type == "PHONE"]
        assert len(phones) >= 1
        assert any("415" in p.text for p in phones)

    def test_multiple_types(self):
        text = "ping roy@example.com or https://example.com or +1-415-555-1212"
        entities = _regex_entities(text)
        types = {e.type for e in entities}
        assert types == {"EMAIL", "URL", "PHONE"}

    def test_dedup_same_email(self):
        entities = _regex_entities("roy@example.com and roy@example.com")
        emails = [e for e in entities if e.type == "EMAIL"]
        assert len(emails) == 1


class TestCleanChunk:
    """_clean_chunk is a text-only fallback: lowercases and strips punctuation.
    Determiner stripping is done structurally in _clean_noun_chunk using spaCy POS
    tags, so this function is language-neutral and does NOT know about English words.
    """

    def test_lowercases(self):
        assert _clean_chunk("Customer Account") == "customer account"

    def test_strips_punctuation(self):
        assert _clean_chunk("customer,") == "customer"

    def test_empty(self):
        assert _clean_chunk("") == ""

    def test_no_english_only_stripping(self):

        assert _clean_chunk("the customer") == "the customer"
        assert _clean_chunk("my dog") == "my dog"


class TestAnnotateWithoutSpacyModel:
    """Verify graceful behavior when spaCy model for a given language is missing."""

    def test_unsupported_language_returns_regex_only(self):
        text = "contact roy@example.com please"
        entities, topics = annotate(text, "ja")
        assert topics == []
        assert any(e.type == "EMAIL" for e in entities)

    def test_empty_text(self):
        entities, topics = annotate("", "en")
        assert entities == []
        assert topics == []

    def test_whitespace_only(self):
        entities, topics = annotate("   \n  ", "en")
        assert entities == []
        assert topics == []


class TestAnnotateWithMockedModel:

    def _make_doc(self, ents=(), tokens=(), chunks=()):
        class _Ent:
            def __init__(self, label, text, start_char, end_char):
                self.label_ = label
                self.text = text
                self.start_char = start_char
                self.end_char = end_char

        class _Token:
            def __init__(self, pos, lemma, i, is_stop=False, is_punct=False, text=None):
                self.pos_ = pos
                self.lemma_ = lemma
                self.i = i
                self.is_stop = is_stop
                self.is_punct = is_punct
                self.text = text if text is not None else lemma

        def _chunk_from_text(text: str, start: int):



            words = text.split()
            tokens_ = []
            for idx, w in enumerate(words):
                is_leading_det = idx == 0 and len(words) > 1 and w.lower() in {"the", "a", "an", "my", "these", "those", "this", "that"}
                tokens_.append(_Token(
                    pos="DET" if is_leading_det else "NOUN",
                    lemma=w.lower(),
                    i=start + idx,
                    is_stop=is_leading_det,
                    text=w,
                ))

            class _Chunk:
                def __init__(self):
                    self.text = text
                    self.start = start
                    self._tokens = tokens_

                def __iter__(self):
                    return iter(self._tokens)

            return _Chunk()

        class _Doc:
            def __init__(self):
                self.ents = [_Ent(*e) for e in ents]
                self._tokens = [_Token(*t) for t in tokens]
                self.noun_chunks = [_chunk_from_text(*c) for c in chunks]

            def __iter__(self):
                return iter(self._tokens)

        return _Doc()

    def test_topic_ranking_deterministic(self):
        doc = self._make_doc(
            chunks=[
                ("the customer", 0),
                ("customer", 3),
                ("the customer", 6),
            ],
            tokens=[],
        )

        class _NLP:
            def __call__(self, text):
                return doc

        with patch.object(ner, "_get_model", return_value=_NLP()):
            entities, topics = annotate("some text", "en")

        assert topics[0] == "customer"

    def test_phrases_ranked_before_singles(self):
        doc = self._make_doc(
            chunks=[("customer account", 0)],
            tokens=[("NOUN", "order", 2, False, False)],
        )

        class _NLP:
            def __call__(self, text):
                return doc

        with patch.object(ner, "_get_model", return_value=_NLP()):
            _, topics = annotate("customer account order", "en")

        assert topics == ["customer account", "order"]

    def test_singles_skip_if_in_phrases(self):
        doc = self._make_doc(
            chunks=[("customer", 0)],
            tokens=[("NOUN", "customer", 0, False, False)],
        )

        class _NLP:
            def __call__(self, text):
                return doc

        with patch.object(ner, "_get_model", return_value=_NLP()):
            _, topics = annotate("customer", "en")

        assert topics == ["customer"]

    def test_topics_capped_at_max(self):
        chunks = [(f"phrase {i}", i) for i in range(MAX_TOPICS + 5)]
        doc = self._make_doc(chunks=chunks, tokens=[])

        class _NLP:
            def __call__(self, text):
                return doc

        with patch.object(ner, "_get_model", return_value=_NLP()):
            _, topics = annotate("x", "en")

        assert len(topics) == MAX_TOPICS

    def test_entity_with_offsets(self):
        doc = self._make_doc(
            ents=[("PERSON", "Alice", 0, 5)],
            chunks=[],
            tokens=[],
        )

        class _NLP:
            def __call__(self, text):
                return doc

        with patch.object(ner, "_get_model", return_value=_NLP()):
            entities, _ = annotate("Alice went home", "en")

        person_entities = [e for e in entities if e.type == "PERSON"]
        assert len(person_entities) == 1
        assert person_entities[0].text == "Alice"
        assert person_entities[0].start_char == 0
        assert person_entities[0].end_char == 5

    def test_gpe_maps_to_location(self):
        doc = self._make_doc(ents=[("GPE", "Paris", 0, 5)], chunks=[], tokens=[])

        class _NLP:
            def __call__(self, text):
                return doc

        with patch.object(ner, "_get_model", return_value=_NLP()):
            entities, _ = annotate("Paris", "en")

        assert entities[0].type == "LOCATION"

    def test_unknown_entity_label_dropped(self):
        doc = self._make_doc(ents=[("ORDINAL", "first", 0, 5)], chunks=[], tokens=[])

        class _NLP:
            def __call__(self, text):
                return doc

        with patch.object(ner, "_get_model", return_value=_NLP()):
            entities, _ = annotate("first place", "en")


        assert entities == []

    def test_regex_entities_added_alongside_spacy(self):
        doc = self._make_doc(
            ents=[("PERSON", "Alice", 0, 5)],
            chunks=[],
            tokens=[],
        )

        class _NLP:
            def __call__(self, text):
                return doc

        text = "Alice wrote to roy@example.com"
        with patch.object(ner, "_get_model", return_value=_NLP()):
            entities, _ = annotate(text, "en")

        types = {e.type for e in entities}
        assert "PERSON" in types
        assert "EMAIL" in types


class TestEntityToDict:
    def test_serializes_all_fields(self):
        e = Entity(type="PERSON", text="Alice", start_char=0, end_char=5)
        assert entity_to_dict(e) == {
            "type": "PERSON",
            "text": "Alice",
            "start_char": 0,
            "end_char": 5,
        }
