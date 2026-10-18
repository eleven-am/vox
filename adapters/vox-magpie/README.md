# vox-magpie

`vox-magpie` provides a streaming NVIDIA MagpieTTS Multilingual adapter for Vox.

The adapter runs the stable NeMo-Speech.cpp C ABI in an isolated worker process.
It emits each NanoCodec PCM chunk as soon as it is available and propagates Vox
response cancellation into the native synthesis callback.

## Install

```bash
pip install vox-magpie
```

## Runtime Dependencies

The lightweight package downloads the pinned NeMo-Speech.cpp v0.2.0 Linux
x86_64 CPU runtime into `$VOX_HOME/runtime/magpie-native`. Downloads are
checksum-verified and published atomically. Only the required tokenizer members
are range-downloaded from the pinned Magpie v2607 checkpoint.

The initial adapter is CPU-only. It does not allocate CUDA memory or compete
with GPU-resident STT models. The official binary supports English, Spanish,
German, French, Italian, Vietnamese, Hindi, Arabic, Korean, and Brazilian
Portuguese. Japanese and Mandarin require a custom NeMo-Speech.cpp build and
are intentionally not advertised by this package.

`VOX_MAGPIE_THREADS` sets a total native pool budget. Vox divides that budget
between the Magpie generator and NanoCodec worker and reports both allocations
in adapter memory diagnostics. The default is cgroup-aware and leaves one CPU
from the container quota available when the host has more than one CPU.

## Use with Vox

```bash
vox pull magpie-tts:v2607-f16
vox run magpie-tts:v2607-f16 "Hello from Magpie." --voice john
```

Magpie provides five preset voices: Aria, Jason, John, Leo, and Sofia. It does
not support reference-audio voice cloning or speed values other than `1.0`.
