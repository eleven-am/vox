# Model Alias Policy

Vox model references should be canonical inside the runtime. The canonical form
is a logical name (task, not backend) with a tag:

```text
<model-name>:<tag>
```

Examples:

```text
parakeet-stt:tdt-0.6b-v3
kokoro-tts:v1.0
chatterbox-tts-turbo:0.1.7
```

Backend-specific names (`-onnx`, `-torch`, `-ct2`, `-nemo`, `-vllm`) are not
public model references. Vox picks the backend from the detected runtime at pull
time, so those names are hard-cut and return not-found.

Alias resolution is owned by `vox.core.alias_resolution`. Other modules should
not invent model-name translation rules.

The resolver also exposes read-only policy snapshots:

- `family_alias_policy()`
- `legacy_model_ref_alias_policy()`
- `legacy_name_alias_policy()`

Tests and tooling should use these snapshots instead of reaching into private
alias tables. This keeps compatibility aliases visible while preserving
`vox.core.alias_resolution` as the single owner of rewrite behavior.

## Alias Classes

Vox currently supports three explicit alias classes.

### Family Aliases

Family aliases are user-facing conveniences for the default model in a family.
They apply only when the caller uses a bare model name with the implicit
`latest` tag.

Examples:

```text
parakeet -> parakeet-stt:tdt-0.6b-v3
kokoro   -> kokoro-tts:v1.0
```

Hardware-appropriate backend selection is handled at pull time by variant
resolution (`vox.core.model_resolution`), not by the alias layer: a logical name
like `parakeet-stt` resolves to its ONNX variant on CPU and its NeMo variant on
CUDA. Family aliases only choose the default model and tag for a bare family
name.

If the runtime profile cannot be matched, family alias resolution falls back to
the `default` profile. This fallback is part of the alias policy and is exposed
in `ModelAliasResolution.resolved_profile`; callers can compare it with
`ModelAliasResolution.profile` to see that fallback occurred.

### Legacy Model Reference Aliases

Legacy model reference aliases rewrite older `(name, tag)` pairs to canonical
model references.

Examples:

```text
voxtral:tts-4b   -> voxtral-tts:4b
parakeet:tdt-0.6b -> parakeet-stt:tdt-0.6b
```

Backend-specific legacy refs (the old `-cuda`/`-nemo`/`-torch` tag forms) are
hard-cut: they no longer rewrite and return not-found.

These are compatibility paths. They must be listed in
`vox.core.alias_resolution`, covered by tests, and treated as deliberate
compatibility behavior.

### Legacy Name Aliases

Legacy name aliases rewrite older names while preserving the requested tag.

Examples:

```text
qwen3-asr:0.6b   -> qwen3-stt:0.6b
whisper:large-v3 -> whisper-stt:large-v3
```

These are also compatibility paths and must stay explicit and tested.

## Unknown Names

Unknown names are not silently translated. They pass through unchanged so the
registry or store can fail with the normal model-not-found behavior. This keeps
alias resolution from becoming a hidden fallback layer.

## Metadata

`resolve_model_alias(...)` returns structured metadata describing whether a
reference was rewritten and which alias class performed the rewrite.
For family aliases, `profile` records the inferred/requested runtime profile and
`resolved_profile` records the profile entry that was actually used.
`resolve_family_alias(...)` remains as the tuple-returning compatibility wrapper
for existing call sites.
