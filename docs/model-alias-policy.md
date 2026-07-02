# Model Alias Policy

Vox model references should be canonical inside the runtime. The canonical form
is:

```text
<model-family-backend>:<tag>
```

Examples:

```text
parakeet-stt-onnx:tdt-0.6b-v3
kokoro-tts-onnx:v1.0
chatterbox-tts-turbo:0.1.7
```

Alias resolution is owned by `vox.core.alias_resolution`. Other modules should
not invent model-name translation rules.

## Alias Classes

Vox currently supports three explicit alias classes.

### Family Aliases

Family aliases are user-facing conveniences for the default model in a family.
They apply only when the caller uses a bare model name with the implicit
`latest` tag.

Examples:

```text
parakeet -> parakeet-stt-onnx:tdt-0.6b-v3
kokoro   -> kokoro-tts-onnx:v1.0
```

Some family aliases are runtime-profile aware. On Spark/CUDA-style systems, a
family alias may resolve to a backend better suited for that runtime. This is a
product choice owned by the alias resolver, not by HTTP, gRPC, scheduler, or
adapter code.

### Legacy Model Reference Aliases

Legacy model reference aliases rewrite older `(name, tag)` pairs to canonical
model references.

Examples:

```text
parakeet:tdt-0.6b-v3-cuda -> parakeet-stt-nemo:tdt-0.6b-v3
voxtral:tts-4b            -> voxtral-tts-vllm:4b
```

These are compatibility paths. They must be listed in
`vox.core.alias_resolution`, covered by tests, and treated as deliberate
compatibility behavior.

### Legacy Name Aliases

Legacy name aliases rewrite older names while preserving the requested tag.

Examples:

```text
qwen3-asr:0.6b     -> qwen3-stt-torch:0.6b
kokoro-torch:v1.0  -> kokoro-tts-torch:v1.0
```

These are also compatibility paths and must stay explicit and tested.

## Unknown Names

Unknown names are not silently translated. They pass through unchanged so the
registry or store can fail with the normal model-not-found behavior. This keeps
alias resolution from becoming a hidden fallback layer.

## Metadata

`resolve_model_alias(...)` returns structured metadata describing whether a
reference was rewritten and which alias class performed the rewrite.
`resolve_family_alias(...)` remains as the tuple-returning compatibility wrapper
for existing call sites.
