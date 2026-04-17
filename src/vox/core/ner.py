from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

LANG_MODELS: dict[str, str] = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
}

ENTITY_MAP: dict[str, str] = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE",
    "TIME": "TIME",
    "PRODUCT": "PRODUCT",
    "MONEY": "MONEY",
}

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_REGEX = re.compile(r"\bhttps?://[^\s<>\"']+", re.IGNORECASE)
PHONE_REGEX = re.compile(r"\+?\d[\d\s().\-]{6,18}\d")

REGEX_ENTITIES: tuple[tuple[re.Pattern[str], str], ...] = (
    (EMAIL_REGEX, "EMAIL"),
    (URL_REGEX, "URL"),
    (PHONE_REGEX, "PHONE"),
)

MAX_TOPICS = 10






_NON_CONTENT_POS = frozenset({
    "DET", "PRON", "ADP", "CCONJ", "SCONJ", "PART", "AUX", "PUNCT", "SPACE", "SYM", "X",
})

_PUNCT_STRIP = ".,;:!?\"'()[]{}"


@dataclass(frozen=True)
class Entity:
    type: str
    text: str
    start_char: int = 0
    end_char: int = 0


_models: dict[str, Any] = {}
_missing_languages: set[str] = set()
_spacy_unavailable: bool = False
_lock = threading.Lock()


def _get_model(lang: str) -> Any | None:
    global _spacy_unavailable
    if _spacy_unavailable:
        return None

    if lang not in LANG_MODELS:
        with _lock:
            if lang not in _missing_languages:
                _missing_languages.add(lang)
                logger.warning("NER: unsupported language %r, returning regex-only annotations", lang)
        return None

    with _lock:
        cached = _models.get(lang)
        if cached is not None:
            return cached

        model_name = LANG_MODELS[lang]
        try:
            import spacy
            from spacy.util import is_package
        except ImportError as exc:
            _spacy_unavailable = True
            logger.warning("NER: spaCy not installed, falling back to regex-only (%s)", exc)
            return None
        except Exception as exc:
            _spacy_unavailable = True
            logger.warning(
                "NER: spaCy import failed (%s: %s), falling back to regex-only",
                type(exc).__name__, exc,
            )
            return None

        try:
            if not is_package(model_name):
                logger.warning(
                    "NER: spaCy model %r not installed; run 'python -m spacy download %s'",
                    model_name, model_name,
                )
                return None
        except Exception:
            logger.exception("NER: is_package probe failed for %s", model_name)
            return None

        try:
            nlp = spacy.load(model_name)
        except Exception:
            logger.exception("NER: failed to load spaCy model %s", model_name)
            return None

        _models[lang] = nlp
        logger.info("NER: loaded spaCy model %s", model_name)
        return nlp


def _clean_chunk(text: str) -> str:
    """Text-only fallback cleaner (used when we don't have the spaCy chunk object).

    Strips only punctuation; relies on `_clean_noun_chunk` for structural filtering.
    """
    tokens = [t.strip(_PUNCT_STRIP) for t in text.lower().strip().split()]
    return " ".join(t for t in tokens if t).strip()


def _clean_noun_chunk(chunk: Any) -> str:
    """Strip leading non-content tokens (determiners, pronouns, particles) from a spaCy
    noun chunk using POS + is_stop flags. Language-neutral because spaCy's language
    model tags these correctly for every loaded language.
    """
    tokens = list(chunk)

    while tokens and (tokens[0].pos_ in _NON_CONTENT_POS or tokens[0].is_stop):
        tokens = tokens[1:]

    while tokens and (tokens[-1].pos_ in _NON_CONTENT_POS or tokens[-1].is_stop):
        tokens = tokens[:-1]
    if not tokens:
        return ""
    parts = [t.text.lower().strip(_PUNCT_STRIP) for t in tokens]
    return " ".join(p for p in parts if p).strip()


def _regex_entities(text: str, seen: set[tuple[str, str]] | None = None) -> list[Entity]:
    entities: list[Entity] = []
    if seen is None:
        seen = set()
    for regex, label in REGEX_ENTITIES:
        for match in regex.finditer(text):
            value = match.group().strip()
            if not value:
                continue
            key = (label, value)
            if key in seen:
                continue
            seen.add(key)
            entities.append(Entity(
                type=label,
                text=value,
                start_char=match.start(),
                end_char=match.end(),
            ))
    return entities


def _extract_entities(doc: Any, text: str) -> list[Entity]:
    entities: list[Entity] = []
    seen: set[tuple[str, str]] = set()

    for ent in doc.ents:
        ent_type = ENTITY_MAP.get(ent.label_)
        if ent_type is None:
            continue
        value = ent.text
        key = (ent_type, value)
        if key in seen:
            continue
        seen.add(key)
        entities.append(Entity(
            type=ent_type,
            text=value,
            start_char=int(ent.start_char),
            end_char=int(ent.end_char),
        ))

    entities.extend(_regex_entities(text, seen))
    return entities


def _extract_topics(doc: Any) -> list[str]:
    phrases: Counter[str] = Counter()
    phrase_first_seen: dict[str, int] = {}

    for chunk in doc.noun_chunks:
        cleaned = _clean_noun_chunk(chunk)
        if len(cleaned) < 2:
            continue
        phrases[cleaned] += 1
        phrase_first_seen.setdefault(cleaned, chunk.start)

    singles: Counter[str] = Counter()
    single_first_seen: dict[str, int] = {}
    for token in doc:
        if token.pos_ not in ("NOUN", "PROPN", "VERB"):
            continue
        if token.is_stop or token.is_punct:
            continue
        lemma = token.lemma_.lower().strip(_PUNCT_STRIP)
        if len(lemma) < 2:
            continue
        if lemma in phrases:
            continue
        singles[lemma] += 1
        single_first_seen.setdefault(lemma, token.i)

    ranked_phrases = sorted(
        phrases.items(),
        key=lambda kv: (-kv[1], phrase_first_seen[kv[0]]),
    )
    ranked_singles = sorted(
        singles.items(),
        key=lambda kv: (-kv[1], single_first_seen[kv[0]]),
    )

    topics: list[str] = []
    for term, _ in ranked_phrases:
        topics.append(term)
        if len(topics) >= MAX_TOPICS:
            return topics
    for term, _ in ranked_singles:
        topics.append(term)
        if len(topics) >= MAX_TOPICS:
            return topics
    return topics


def annotate(text: str, lang: str = "en") -> tuple[list[Entity], list[str]]:
    if not text or not text.strip():
        return [], []

    nlp = _get_model(lang)
    if nlp is None:
        return _regex_entities(text), []

    try:
        doc = nlp(text)
    except Exception:
        logger.exception("NER: spaCy processing failed (lang=%s)", lang)
        return _regex_entities(text), []

    return _extract_entities(doc, text), _extract_topics(doc)


def entity_to_dict(e: Entity) -> dict[str, Any]:
    return {
        "type": e.type,
        "text": e.text,
        "start_char": e.start_char,
        "end_char": e.end_char,
    }
