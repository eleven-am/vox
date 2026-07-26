from __future__ import annotations

import argparse
import base64
import contextlib
import json
import random
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("custom", "clone"), default="custom")
    parser.add_argument("--language", default="English")

    parser.add_argument("--speaker")

    parser.add_argument("--ref-audio-path")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    request = json.load(sys.stdin)
    text = str(request["text"])
    reference_text = request.get("reference_text")
    original_stdout = sys.stdout

    with contextlib.redirect_stdout(sys.stderr):
        sys.path.insert(0, args.runtime_dir)
        if args.seed is not None:
            random.seed(args.seed)
            try:
                import numpy as np

                np.random.seed(args.seed % (2**32))
            except ImportError:
                pass
            try:
                import torch

                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)
            except ImportError:
                pass
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
                text=text,
                language=args.language,
                ref_audio=(ref_audio, int(ref_sr)),
                ref_text=reference_text,
            )
        else:
            if not args.speaker:
                raise SystemExit("custom mode requires --speaker")
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language=args.language,
                speaker=args.speaker,
                instruct=reference_text,
            )

    if not isinstance(wavs, (list, tuple)):
        wavs = [wavs]

    audio = b"".join(wav.astype("float32", copy=False).tobytes() for wav in wavs if getattr(wav, "size", 0))
    payload = {
        "sample_rate": int(sample_rate),
        "audio_b64": base64.b64encode(audio).decode("ascii"),
    }
    print(json.dumps(payload), file=original_stdout, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
