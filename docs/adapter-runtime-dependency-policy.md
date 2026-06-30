# Adapter Runtime Dependency Policy

This document defines how Vox adapter runtime dependencies are specified,
installed, verified, upgraded, and repaired.

It extends the packaging boundary described in
[the adapter contract](adapter-contract.md).

## Dependency Classes

Adapter runtime dependencies fall into three classes.

### Exact Pins

Use exact pins when an adapter depends on a known-compatible package version:

```text
transformers==4.51.3
tokenizers==0.21.4
```

Exact pins are preferred for fragile ML runtimes, packages installed from a
source repository, packages with fast-moving APIs, and any runtime dependency
that has previously broken model loading or inference.

### Bounded Ranges

Use bounded ranges when an adapter is known to work across a narrow compatible
range:

```text
transformers>=4.57.6,<4.58
faster-whisper>=1.2.1,<2.0.0
```

Bounded ranges are acceptable when the upstream package follows compatible
release behavior inside the bound and tests cover the adapter's load path.

### Broad Ranges

Broad ranges should be avoided for heavyweight runtime dependencies:

```text
nemo-toolkit[asr]
coqui-tts>=0.27.5
```

They are allowed only as a temporary compatibility bridge. When a broad runtime
dependency is used, the adapter must verify the runtime after install and fail
with a clear error if the backend cannot be imported or loaded.

## `--upgrade` Policy

`--upgrade` is not a style choice; it changes dependency resolution.

Use `--upgrade` when:

- the runtime dependency is exact-pinned
- the runtime dependency is tightly bounded
- the adapter is repairing a stale or broken runtime directory
- the adapter is installing a package that should replace an older compatible
  version already present in the runtime directory

Avoid `--upgrade` when:

- the dependency is broad or unbounded
- the package resolves a large transitive ML stack
- the package is installed from a moving source such as a GitHub branch
- the adapter intentionally wants to keep an already-working runtime stable

If an adapter uses `upgrade=False`, that should be deliberate and visible in
tests. The current examples are broad or source-based runtimes such as Parakeet
NeMo, XTTS, Whisper, and VibeVoice.

## Install Location

Heavy runtime dependencies must be installed into:

```text
$VOX_HOME/runtime/<runtime-name>
```

Adapters should use `vox.core.adapter_runtime` for target-directory installs:

- `activate_runtime_path(...)`
- `install_target_runtime_requirements(...)`
- `ensure_target_runtime(...)`
- `purge_runtime_modules(...)`
- `write_app_fallback_path(...)`

If a backend requires a full virtual environment or process-level isolation,
the adapter may use a custom runtime layout, but that exception must be
documented and tested.

## Verification After Install

A runtime install is valid only after verification. A successful `pip` or `uv`
exit code is not enough.

Adapters must verify at least one of:

- the expected import is available
- expected package directories exist
- expected `.dist-info` metadata exists and has an acceptable version
- required runtime symbols exist
- a backend-specific load probe succeeds

Examples:

- Qwen verifies the import requested by the adapter.
- Kokoro verifies critical expected package paths for grouped installs.
- VibeVoice verifies package paths and pinned versions.
- Dia verifies that `DiaForConditionalGeneration` and `AutoProcessor` are
  exposed by the isolated Transformers runtime.

Runtime sentinels may be used only as a cache hint. A sentinel must not bypass
import or package verification. If a sentinel exists but verification fails, the
runtime is stale and must be repaired.

## Repairing Stale Or Broken Runtime Directories

Runtime directories are adapter-owned and disposable. They may be deleted and
recreated when dependency resolution is broken.

The normal repair flow is:

1. Activate the runtime directory.
2. Verify imports, package paths, versions, or required symbols.
3. If verification passes, use the runtime.
4. If verification fails, purge relevant imported modules.
5. Reinstall the required packages into the same runtime directory.
6. Verify again.
7. If verification still fails, raise a clear runtime error.

For severe corruption, an adapter may delete the runtime directory before
reinstalling. This is appropriate when:

- metadata says one version is installed but imports resolve to another
- expected package paths are missing after a reported successful install
- the runtime contains incompatible leftovers that cannot be safely overwritten
- a moving source package changed layout incompatibly

Adapters should not silently continue with a failed runtime. A broken runtime
must either be repaired or reported.

## Installer Order

The default installer order is:

1. `uv pip install --python <current-python> --target <runtime>`
2. `python -m pip install --target <runtime>`

Adapters may override the order only for compatibility with known runtime
constraints. The reason should be covered by a test.

If `python -m pip` is used and `pip` is missing, Vox bootstraps pip with
`ensurepip` before retrying.

## Tests Required For Runtime Dependencies

Adapters with runtime bootstrap logic should have tests for:

- installer command shape
- `--upgrade` policy
- `--no-deps` policy
- fallback from `uv` to `python -m pip`
- missing `pip` bootstrap when relevant
- post-install verification
- stale sentinel or stale runtime repair behavior
- module purging when runtime dependencies are replaced

New runtime policy behavior should be covered in `tests/test_adapter_runtime.py`
when it belongs to the shared helper, or in the adapter-specific test file when
it is adapter behavior.

