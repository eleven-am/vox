from __future__ import annotations

import argparse
import base64
import contextlib
import json
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("custom", "clone"), default="custom")
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="English")

    parser.add_argument("--speaker")
    parser.add_argument("--instruct")

    parser.add_argument("--ref-audio-path")
    parser.add_argument("--ref-text")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    original_stdout = sys.stdout

    with contextlib.redirect_stdout(sys.stderr):
        sys.path.insert(0, args.runtime_dir)
        from qwen_tts import Qwen3TTSModel

        dtype = "bfloat16" if args.device == "cuda" else "float32"
        device_map = "cuda:0" if args.device == "cuda" else "cpu"
        model = Qwen3TTSModel.from_pretrained(
            args.model_id,
            device_map=device_map,
            dtype=dtype,
        )

        if args.mode == "clone":
            if not args.ref_audio_path:
                raise SystemExit("clone mode requires --ref-audio-path")
            import soundfile as sf

            ref_audio, ref_sr = sf.read(args.ref_audio_path, dtype="float32", always_2d=False)
            wavs, sample_rate = model.generate_voice_clone(
                text=args.text,
                language=args.language,
                ref_audio=(ref_audio, int(ref_sr)),
                ref_text=args.ref_text,
            )
        else:
            if not args.speaker:
                raise SystemExit("custom mode requires --speaker")
            wavs, sample_rate = model.generate_custom_voice(
                text=args.text,
                language=args.language,
                speaker=args.speaker,
                instruct=args.instruct,
            )

    if not isinstance(wavs, (list, tuple)):
        wavs = [wavs]

    audio = b"".join(
        wav.astype("float32", copy=False).tobytes()
        for wav in wavs
        if getattr(wav, "size", 0)
    )
    payload = {
        "sample_rate": int(sample_rate),
        "audio_b64": base64.b64encode(audio).decode("ascii"),
    }
    print(json.dumps(payload), file=original_stdout, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
