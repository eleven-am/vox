# vox-smart-turn

Pipecat Smart Turn v3 endpoint detection for Vox.

## Install

```bash
pip install vox-smart-turn
```

## Runtime Dependencies

The lightweight adapter package installs Transformers feature-extraction support into
`$VOX_HOME/runtime/smart-turn` during `vox pull`. ONNX Runtime remains a generic Vox
compute dependency. Model weights remain in the Vox model store.

## Use with Vox

```bash
vox pull smart-turn:v3.2
```

Use `smart-turn:v3.2` as the conversation `turn_detector`. The CPU int8 variant is
the default. Use `--variant gpu` to request the larger fp32 ONNX model explicitly.
