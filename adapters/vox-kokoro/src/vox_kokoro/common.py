from __future__ import annotations

from vox.core.types import VoiceInfo

SAMPLE_RATE = 24000

# Maps the first 3 characters of a Kokoro voice ID (e.g. "af_") to the
# language tag expected by the adapter metadata.
VOICE_PREFIX_TO_LANGUAGE: dict[str, str] = {
    "af_": "en-us",  # American English female
    "am_": "en-us",  # American English male
    "bf_": "en-gb",  # British English female
    "bm_": "en-gb",  # British English male
    "jf_": "ja",  # Japanese female
    "jm_": "ja",  # Japanese male
    "zf_": "zh",  # Chinese female
    "zm_": "zh",  # Chinese male
    "ef_": "es",  # Spanish female
    "em_": "es",  # Spanish male
    "ff_": "fr",  # French female
    "fm_": "fr",  # French male
    "hf_": "hi",  # Hindi female
    "hm_": "hi",  # Hindi male
    "if_": "it",  # Italian female
    "im_": "it",  # Italian male
    "pf_": "pt",  # Portuguese female
    "pm_": "pt",  # Portuguese male
}

# Kokoro's native runtime uses single-letter language codes.
VOICE_PREFIX_TO_PIPELINE_LANG: dict[str, str] = {
    "af_": "a",
    "am_": "a",
    "bf_": "b",
    "bm_": "b",
    "ef_": "e",
    "em_": "e",
    "ff_": "f",
    "fm_": "f",
    "hf_": "h",
    "hm_": "h",
    "if_": "i",
    "im_": "i",
    "jf_": "j",
    "jm_": "j",
    "pf_": "p",
    "pm_": "p",
    "zf_": "z",
    "zm_": "z",
}

LANGUAGE_TO_PIPELINE_LANG: dict[str, str] = {
    "a": "a",
    "en": "a",
    "en-us": "a",
    "american-english": "a",
    "b": "b",
    "en-gb": "b",
    "british-english": "b",
    "e": "e",
    "es": "e",
    "es-es": "e",
    "f": "f",
    "fr": "f",
    "fr-fr": "f",
    "h": "h",
    "hi": "h",
    "hi-in": "h",
    "i": "i",
    "it": "i",
    "it-it": "i",
    "j": "j",
    "ja": "j",
    "ja-jp": "j",
    "p": "p",
    "pt": "p",
    "pt-br": "p",
    "z": "z",
    "zh": "z",
    "zh-cn": "z",
}

SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(sorted(set(VOICE_PREFIX_TO_LANGUAGE.values())))

_GENDER_MAP: dict[str, str] = {"f": "female", "m": "male"}


def voice_lang_tag(voice_id: str) -> str:
    prefix = voice_id[:3] if len(voice_id) >= 3 else ""
    return VOICE_PREFIX_TO_LANGUAGE.get(prefix, "en-us")


def voice_info(voice_id: str) -> VoiceInfo:
    prefix = voice_id[:3] if len(voice_id) >= 3 else ""
    language = VOICE_PREFIX_TO_LANGUAGE.get(prefix, "en-us")
    gender = _GENDER_MAP.get(prefix[1:2]) if len(prefix) >= 2 else None
    return VoiceInfo(
        id=voice_id,
        name=voice_id,
        language=language,
        gender=gender,
    )


def pipeline_lang_code(voice_id: str, language: str | None = None) -> str:
    if language:
        normalized = language.strip().lower()
        if normalized in LANGUAGE_TO_PIPELINE_LANG:
            return LANGUAGE_TO_PIPELINE_LANG[normalized]

    prefix = voice_id[:3] if len(voice_id) >= 3 else ""
    return VOICE_PREFIX_TO_PIPELINE_LANG.get(prefix, "a")
