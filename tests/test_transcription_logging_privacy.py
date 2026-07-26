import logging
from pathlib import Path

from vox.core.transcription_logging import log_transcription_result

ADAPTER_SOURCES = (
    Path("adapters/vox-parakeet/src/vox_parakeet/adapter.py"),
    Path("adapters/vox-qwen/src/vox_qwen/asr_adapter.py"),
    Path("adapters/vox-voxtral/src/vox_voxtral/stt_adapter.py"),
    Path("adapters/vox-microsoft/src/vox_microsoft/speecht5_stt_adapter.py"),
)


def test_stt_adapters_do_not_log_transcript_content_at_info():
    for path in ADAPTER_SOURCES:
        source = path.read_text()
        assert "text[:80]" not in source
        assert "Transcribed %dms audio: %s" not in source


def test_transcription_summary_logs_metadata_without_text(caplog):
    secret = "private transcript contents"
    logger = logging.getLogger("vox.test.transcription")

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_transcription_result(
            logger,
            audio_duration_ms=1234,
            text=secret,
        )

    assert secret not in caplog.text
    assert "1234ms" in caplog.text
    assert f"{len(secret)} chars" in caplog.text
