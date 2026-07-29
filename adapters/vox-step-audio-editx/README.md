# vox-step-audio-editx

`vox-step-audio-editx` provides Step-Audio-EditX voice-cloning synthesis for Vox.

## Install

```bash
pip install vox-step-audio-editx
```

## Runtime Dependencies

The package contains only the Vox adapter. `vox pull` installs the pinned Step-Audio-EditX inference runtime, including its matching TorchVision package, into `$VOX_HOME/runtime/step-audio-editx`. The Vox application environment remains the owner of Torch, TorchAudio, Triton, and CUDA runtime libraries.

The supported backend is Linux x86_64 CUDA with at least 10GiB of usable VRAM budget. CPU, ONNX, and Spark/ARM NVIDIA execution are not currently production-supported.

## Use with Vox

```bash
vox pull step-audio-editx:3b-awq
vox run step-audio-editx:3b-awq "Hello from a cloned voice" --voice /path/to/reference.wav
```

The model requires reference audio and its transcript. Stored Vox voices provide both through the standard voice API. Inline paralinguistic markers supported by Step-Audio-EditX, such as `[Laughter]` and `[Sigh]`, may be included in the input text.
