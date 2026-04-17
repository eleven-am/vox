from __future__ import annotations

from vox.core.types import VoiceInfo

SAMPLE_RATE = 24000

OFFICIAL_VOICE_IDS: tuple[str, ...] = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
)



VOICE_PREFIX_TO_LANGUAGE: dict[str, str] = {
    "af_": "en-us",
    "am_": "en-us",
    "bf_": "en-gb",
    "bm_": "en-gb",
    "jf_": "ja",
    "jm_": "ja",
    "zf_": "zh",
    "zm_": "zh",
    "ef_": "es",
    "em_": "es",
    "ff_": "fr",
    "fm_": "fr",
    "hf_": "hi",
    "hm_": "hi",
    "if_": "it",
    "im_": "it",
    "pf_": "pt",
    "pm_": "pt",
}


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
