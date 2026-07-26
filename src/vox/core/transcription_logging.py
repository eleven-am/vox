from __future__ import annotations

import logging


def log_transcription_result(
    logger: logging.Logger,
    *,
    audio_duration_ms: int,
    text: str,
) -> None:
    if text:
        logger.info(
            "Transcribed %dms audio (%d chars)",
            audio_duration_ms,
            len(text),
        )
    else:
        logger.warning("Empty transcription for %dms audio", audio_duration_ms)
