from __future__ import annotations

import tomllib
from pathlib import Path

ADAPTER_PACKAGE_DIRS = {
    "vox-cosyvoice": Path("adapters/vox-cosyvoice"),
    "vox-dia": Path("adapters/vox-dia"),
    "vox-orpheus": Path("adapters/vox-orpheus"),
    "vox-indextts": Path("adapters/vox-indextts"),
}


def adapter_package_specs() -> tuple[str, ...]:
    specs: list[str] = []
    for package_name, package_dir in ADAPTER_PACKAGE_DIRS.items():
        pyproject = tomllib.loads((package_dir / "pyproject.toml").read_text())
        project = pyproject["project"]
        assert project["name"] == package_name
        specs.append(f"{package_name}=={project['version']}")
    return tuple(specs)


def test_expressive_adapter_smoke_script_refuses_production_and_requires_create():
    script = Path("scripts/expressive-adapter-smoke.sh").read_text()
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()
    makefile = Path("Makefile").read_text()

    assert "--variant VARIANT" in script
    assert "--voice VOICE" in script
    assert "--voice VOICE              Voice id or WAV path to pass to vox run; required for indextts-tts" in script
    assert "--audio-usable yes|no" in script
    assert "--failure-class CLASS" in script
    assert "--cpu-only" in script
    assert "VARIANT=\"\"" in script
    assert "VOICE=\"\"" in script
    assert "AUDIO_USABLE=\"unchecked\"" in script
    assert "AUDIO_USABLE_PROVIDED=0" in script
    assert "AUDIO_USABLE_PROVIDED=1" in script
    assert "FAILURE_CLASS=\"none\"" in script
    assert '[[ "$AUDIO_USABLE_PROVIDED" == "1" ]]' in script
    assert "Manual audio usable: $AUDIO_USABLE" in script
    assert "validate_audio_usability()" in script
    assert "validate_failure_classification()" in script
    assert "Manual audio usability not confirmed" in script
    assert "Manual audio usability rejected" in script
    assert "Failing smoke run must set --failure-class" in script
    assert "Passing smoke run must use --failure-class none" in script
    assert "none|Vox|adapter|dependency|upstream|hardware" in script
    assert "requires_voice_reference()" in script
    assert "indextts-tts:*)" in script
    assert 'requires_voice_reference "$MODEL" && [[ -z "$VOICE" ]]' in script
    assert "requires --voice with a disposable reference WAV or voice id" in script
    assert "GPU=\"${VOX_SMOKE_GPU:-1}\"" in script
    assert 'resources_json=\'{"limits": {"nvidia.com/gpu": "1"}}\'' in script
    assert "Accelerator request: $accelerator_request" in script
    assert "pull_command=\"vox pull $MODEL\"" in script
    assert 'vox pull "$MODEL_REF" --variant "$VARIANT_REF"' in script
    assert '[[ "$NS" == "vox" || "$PVC" == "vox-data" ]]' in script
    assert "refusing to use production Vox namespace/PVC" in script
    assert "namespace $NS does not exist; rerun with --create" in script
    assert "pvc $NS/$PVC does not exist; rerun with --create" in script
    assert "pod $NS/$POD does not exist; rerun with --create" in script
    assert "existing_image=" in script
    assert "already exists with image" in script
    assert "existing_pvc=" in script
    assert "already exists with PVC" in script
    assert "existing_has_gpu=0" in script
    assert "already exists with a GPU limit, but this run requested --cpu-only" in script
    assert "already exists without a GPU limit, but this run requested GPU validation" in script
    assert "exit 6" in script
    assert "allow_incompatible=" in script
    assert "allow_incompatible_normalized=" in script
    assert 'FAILED_STEPS+=("VOX_ALLOW_INCOMPATIBLE is enabled in the smoke pod")' in script
    assert "Used VOX_ALLOW_INCOMPATIBLE: $allow_incompatible_evidence" in script
    assert "expected_adapter_package=" in script
    assert "validate_adapter_package_baseline()" in script
    assert "Expected adapter package: $expected_adapter_package" in script
    assert 'validate_adapter_package_baseline "$packages"' in script
    assert 'FAILED_STEPS+=("Expected adapter package baseline missing: $expected_adapter_package")' in script
    assert "verifies that it was created\nwith the requested image, PVC, and accelerator mode" in runbook
    assert "switching image tags, PVCs, or switching between GPU and\n`--cpu-only` validation" in runbook
    assert "VOX_ALLOW_INCOMPATIBLE=1" not in script
    assert "scripts/expressive-adapter-smoke.sh --model dia-tts:1.6b --audio-usable yes" in runbook
    assert (
        "scripts/expressive-adapter-smoke.sh --model chatterbox-tts-turbo:0.1.7 "
        "--variant onnx --audio-usable yes"
    ) in runbook
    assert (
        "scripts/expressive-adapter-smoke.sh --model chatterbox-tts-turbo:0.1.7 "
        "--variant onnx --cpu-only --audio-usable yes"
    ) in runbook
    assert "make smoke-expressive MODEL=dia-tts:1.6b SMOKE_AUDIO_USABLE=yes" in runbook
    assert "make smoke-expressive MODEL=chatterbox-tts-turbo:0.1.7 SMOKE_VARIANT=onnx SMOKE_AUDIO_USABLE=yes" in runbook
    assert (
        "make smoke-expressive MODEL=chatterbox-tts-turbo:0.1.7 "
        "SMOKE_VARIANT=onnx SMOKE_CPU_ONLY=1 SMOKE_AUDIO_USABLE=yes"
    ) in runbook
    assert (
        "make smoke-expressive MODEL=indextts-tts:2 "
        "SMOKE_VOICE=/home/vox/.vox/smoke-voices/reference.wav SMOKE_AUDIO_USABLE=yes"
    ) in runbook
    assert "make smoke-expressive MODEL=dia-tts:1.6b SMOKE_CREATE=1 SMOKE_AUDIO_USABLE=yes" in runbook
    assert "unless `--create` is\npassed explicitly after approval" in runbook
    assert "`indextts-tts:*` smoke validation requires `--voice`" in runbook
    assert "fails before touching\nKubernetes resources" in runbook
    assert "rerun with `--audio-usable yes` only when both short and long outputs\nare usable" in runbook
    assert "Failing smoke runs must be rerun or recorded with a concrete failure class" in runbook
    assert "SMOKE_VARIANT=onnx" in makefile
    assert "SMOKE_VOICE=/home/vox/.vox/smoke-voices/reference.wav" in makefile
    assert '$(if $(SMOKE_VOICE),--voice "$(SMOKE_VOICE)",)' in makefile
    assert "SMOKE_CPU_ONLY=1" in makefile
    assert "SMOKE_AUDIO_USABLE=yes" in makefile
    assert "SMOKE_FAILURE_CLASS=dependency" in makefile
    assert '--audio-usable "$(SMOKE_AUDIO_USABLE)"' in makefile
    assert '--failure-class "$(SMOKE_FAILURE_CLASS)"' in makefile
    assert '--variant "$(SMOKE_VARIANT)"' in makefile
    assert "--cpu-only" in makefile
    assert "smoke-expressive-served" in makefile
    assert "scripts/expressive-adapter-served-smoke.py" in makefile
    assert "SMOKE_BASE_URL=http://127.0.0.1:8000" in makefile
    assert "SMOKE_API_KEY=..." in makefile
    assert "SMOKE_PARAMS_JSON='{}'" in makefile
    assert "SMOKE_INSPECT_ONLY=1" in makefile
    assert "--inspect-only" in makefile


def test_expressive_adapter_served_smoke_script_uses_existing_server_only():
    script = Path("scripts/expressive-adapter-served-smoke.py").read_text()
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    assert "already-running Vox HTTP server" in script
    assert "does not call Kubernetes" in script
    assert "v1/audio/speech" in script
    assert "v1/health" in script
    assert "urllib.parse.quote(args.model, safe='')" in script
    assert "model_detail" in script
    assert "v1/models/loaded" in script
    assert "_response_evidence" in script
    assert "loaded_before" in script
    assert "loaded_after" in script
    assert "--base-url" in script
    assert "--api-key" in script
    assert "VOX_API_KEY" in script
    assert "\"x-api-key\"" in script
    assert "api_key_provided" in script
    assert "--params-json" in script
    assert "--audio-usable" in script
    assert "--inspect-only" in script
    assert "synthesis_skipped" in script
    assert "inspect_only" in script
    assert "/v1/audio/speech" in script
    assert "/tmp/vox-served-smoke" in script
    assert "kubectl" not in script
    assert "vox pull" not in script
    assert "Existing Server Smoke" in runbook
    assert "For read-only inspection, use `--inspect-only`" in runbook
    assert "stops before synthesis" in runbook
    assert "SMOKE_INSPECT_ONLY=1" in runbook
    assert "do not create\na new namespace or PVC" in runbook
    assert "does not call `kubectl`, `vox pull`, or mutate adapter,\nruntime, model, or PVC contents" in runbook
    assert "`GET /v1/models/{model}` for the requested model" in runbook
    assert "`/v1/models/loaded` before synthesis" in runbook
    assert "`/v1/models/loaded` after synthesis" in runbook
    assert "Full existing-server smoke can change in-memory scheduler state and VRAM usage" in runbook
    assert "Use `--inspect-only` for read-only checks" in runbook
    assert "not enough to mark an adapter fully production-ready" in runbook
    assert "never the key value" in runbook
    assert "SMOKE_API_KEY=..." in runbook
    assert "make smoke-expressive-served MODEL=dia-tts:1.6b" in runbook


def test_expressive_adapter_smoke_script_preserves_evidence_after_step_failures():
    script = Path("scripts/expressive-adapter-smoke.sh").read_text()

    assert "FAILED=0" in script
    assert "FAILED_STEPS=()" in script
    assert "record_timed()" in script
    assert "validate_timed_output()" in script
    assert "record_resources()" in script
    assert "record_storage_usage()" in script
    assert "validate_storage_usage()" in script
    assert "record_artifact_stats()" in script
    assert "validate_copied_artifacts()" in script
    assert "record_audio_durations()" in script
    assert "validate_audio_durations()" in script
    assert "record_audio_streams()" in script
    assert "validate_audio_streams()" in script
    assert "record_audio_signal()" in script
    assert "validate_audio_signal()" in script
    assert "validate_audio_usability()" in script
    assert "validate_failure_classification()" in script
    assert "validate_model_resolution()" in script
    assert "record_smoke_result()" in script
    assert 'env MODEL_REF="$MODEL" TEXT_REF="$SHORT_TEXT"' in script
    assert 'env MODEL_REF="$MODEL" TEXT_REF="$LONG_TEXT"' in script
    assert 'VOICE_REF="$VOICE"' in script
    assert 'vox run "$MODEL_REF" "$TEXT_REF" --voice "$VOICE_REF" --output /tmp/short.wav' in script
    assert 'vox run "$MODEL_REF" "$TEXT_REF" --voice "$VOICE_REF" --output /tmp/long.wav' in script
    assert 'vox run "$MODEL_REF" "$TEXT_REF" --output /tmp/short.wav' in script
    assert 'vox run "$MODEL_REF" "$TEXT_REF" --output /tmp/long.wav' in script
    assert "vox run '$MODEL' '$SHORT_TEXT'" not in script
    assert "vox run '$MODEL' '$LONG_TEXT'" not in script
    assert 'FAILED=1' in script
    assert 'record_timed "Pull Output"' in script
    assert 'record_timed "Short Synthesis Output"' in script
    assert 'record_timed "Long Synthesis Output"' in script
    assert "^real[[:space:]]+[0-9]+([.][0-9]+)?$" in script
    assert 'FAILED_STEPS+=("$label missing machine-readable real duration")' in script
    assert 'validate_timed_output "$label" "$output"' in script
    assert 'FAILED_STEPS+=("$label")' in script
    assert 'append_section "Smoke Result"' in script
    assert script.count("## Smoke Result") == 0
    assert script.count('append_section "Smoke Result"') == 1
    assert "failed_steps=none" in script
    assert 'record_resources "Resource Snapshot After Pull"' in script
    assert 'record_resources "Resource Snapshot After Short Synthesis"' in script
    assert 'record_resources "Resource Snapshot After Long Synthesis"' in script
    assert "pod_metrics=" in script
    assert "gpu_metrics=" in script
    assert "Pod metrics:" in script
    assert "GPU metrics:" in script
    assert '[[ "$pod_metrics" != *"$POD"* ]]' in script
    assert 'FAILED_STEPS+=("$label missing pod memory telemetry")' in script
    assert '[[ "$GPU" != "0" && "$gpu_metrics" != *"NVIDIA-SMI"* ]]' in script
    assert 'FAILED_STEPS+=("$label missing GPU telemetry")' in script
    assert 'append_section "Storage Snapshot After Pull"' in script
    assert 'record_storage_usage' in script
    assert 'df -h "$vox_home"' in script
    assert 'du -sh "$path"' in script
    assert '"adapters:$vox_home/adapters"' in script
    assert '"runtime:$vox_home/runtime"' in script
    assert '"models:$vox_home/models"' in script
    assert '"manifests:$vox_home/manifests"' in script
    assert '"blobs:$vox_home/blobs"' in script
    assert 'FAILED_STEPS+=("Storage snapshot missing filesystem usage")' in script
    assert 'FAILED_STEPS+=("Storage snapshot missing $label usage")' in script
    assert 'record_storage_usage' in script
    assert "voice_path_check=\"file\"" in script
    assert "voice_path_check=\"voice-id\"" in script
    assert "voice_path_exists=\"yes\"" in script
    assert "voice_path_exists=\"no\"" in script
    assert 'FAILED_STEPS+=("Voice reference path missing: $VOICE")' in script
    assert 'kubectl -n "$NS" exec "$POD" -- test -f "$VOICE"' in script
    assert "record_audio_durations" in script
    assert 'validate_audio_durations "$body"' in script
    assert 'FAILED_STEPS+=("Missing or invalid $label audio duration: $duration")' in script
    assert 'FAILED_STEPS+=("Non-positive $label audio duration: $duration")' in script
    assert "record_audio_streams" in script
    assert 'append_section "Audio Stream Metadata"' in script
    assert "codec_name,sample_rate,channels" in script
    assert "sample_rate=" in script
    assert "channels=" in script
    assert 'validate_audio_streams "$body"' in script
    assert 'FAILED_STEPS+=("Missing or invalid $label audio stream metadata")' in script
    assert 'FAILED_STEPS+=("Invalid $label audio sample rate: ${sample_rate:-missing}")' in script
    assert 'FAILED_STEPS+=("Invalid $label audio channel count: ${channels:-missing}")' in script
    assert "record_audio_signal" in script
    assert 'append_section "Audio Signal Stats"' in script
    assert "volumedetect" in script
    assert "mean_volume=" in script
    assert "max_volume=" in script
    assert 'validate_audio_signal "$body"' in script
    assert 'FAILED_STEPS+=("Missing or invalid $label audio signal stats")' in script
    assert 'FAILED_STEPS+=("Silent $label audio output")' in script
    assert 'record_audio_signal' in script
    assert 'validate_model_resolution "$model_resolution"' in script
    assert 'FAILED_STEPS+=("Model resolution reported an error")' in script
    assert 'FAILED_STEPS+=("Model manifest missing after pull")' in script
    assert 'FAILED_STEPS+=("Model manifest layers missing after pull")' in script
    assert 'FAILED_STEPS+=("Model adapter package missing from resolved entry")' in script
    assert 'FAILED_STEPS+=("Resolved adapter package is not installed")' in script
    assert 'FAILED_STEPS+=("Expected adapter runtime path is missing after pull")' in script
    assert 'FAILED_STEPS+=("Expected adapter runtime path has no adapter-owned contents")' in script
    assert "validate_pre_pull_clean_state()" in script
    assert 'append_section "Pre-Pull Clean State"' in script
    assert 'validate_pre_pull_clean_state "$pre_pull_state"' in script
    assert 'FAILED_STEPS+=("Pre-pull clean-state probe reported an error")' in script
    assert 'FAILED_STEPS+=("Pre-pull target model/runtime state is not clean")' in script
    assert "'dirty'" in script
    assert "'manifest_exists'" in script
    assert "'model_link_exists'" in script
    assert "meaningful_entry_count" in script
    assert "short:/tmp/short.wav long:/tmp/long.wav" in script
    assert "printf \"%s=%s\\n\"" in script
    assert "printf \"%s=missing\\n\"" in script
    disallowed_multi_probe = (
        "ffprobe -v error -show_entries format=duration "
        "-of default=nw=1:nk=1 /tmp/short.wav /tmp/long.wav"
    )
    assert disallowed_multi_probe not in script
    assert 'record_artifact_stats "Copied Artifact Stats"' in script
    assert "artifact_sha256()" in script
    assert "sha256sum" in script
    assert "shasum -a 256" in script
    assert "sha256=$digest" in script
    assert 'validate_copied_artifacts "$short_wav" "$long_wav"' in script
    assert "validate_audio_usability" in script
    assert "validate_failure_classification" in script
    assert 'FAILED_STEPS+=("Missing or empty copied artifact: $path")' in script
    assert 'one or more smoke steps failed; inspect evidence' in script
    assert "exit 5" in script


def test_expressive_adapter_smoke_script_records_model_resolution_evidence():
    script = Path("scripts/expressive-adapter-smoke.sh").read_text()

    assert 'append_section "Model Resolution"' in script
    assert "resolve_model_reference" in script
    assert "resolve_catalog_entry" in script
    assert "'registry_entry'" in script
    assert "requested_variant" in script
    assert "variant=os.environ.get('VARIANT_REF') or None" in script
    assert "'resolved_variant'" in script
    assert "'preferred_backend'" in script
    assert "'manifest_path'" in script
    assert "'model_link_path'" in script
    assert "'runtime_root'" in script
    assert "'manifest_exists'" in script
    assert "'manifest_config'" in script
    assert "'manifest_layers'" in script
    assert "'adapter_package'" in script
    assert "'adapter_package_version'" in script
    assert "'adapter_package_installed'" in script
    assert "registry.adapter_resolver.installed_version(adapter_package)" in script
    assert "'vox-cosyvoice': ['cosyvoice']" in script
    assert "'vox-dia': ['dia']" in script
    assert "'vox-orpheus': ['orpheus']" in script
    assert "'vox-indextts': ['indextts']" in script
    assert "'adapter_runtime_paths'" in script
    assert "'adapter_runtime_missing'" in script
    assert "'adapter_runtime_empty'" in script
    assert "'meaningful_entry_count'" in script
    assert "_vox_runtime_fallback_paths.pth" in script
    assert "store.root / 'runtime' / runtime_name" in script


def test_expressive_adapter_smoke_runbook_keeps_production_safety_boundary():
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    assert "The default operational path is to test the\nalready-running Vox HTTP endpoint" in runbook
    assert "Clean-pull Kubernetes smoke" in runbook
    assert "must be explicitly approved before creating or using any nonstandard\nnamespace or PVC" in runbook
    assert "VOX_SMOKE_NS=vox-adapter-smoke" in runbook
    assert "VOX_SMOKE_PVC=vox-adapter-smoke-data" in runbook
    assert "Do not create a namespace or PVC just\nbecause a model needs testing" in runbook
    assert "Do not mutate, clean, reinstall, restart, or scale" in runbook
    assert "kubectl get namespace \"$VOX_SMOKE_NS\"" in runbook
    assert "kubectl -n \"$VOX_SMOKE_NS\" get pvc \"$VOX_SMOKE_PVC\"" in runbook
    assert "Do not\ninvent new namespace/PVC names" in runbook


def test_expressive_adapter_smoke_runbook_lists_required_models_and_evidence():
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    for model in ("cosyvoice2-tts:0.5b", "dia-tts:1.6b", "orpheus-tts:medium-3b", "indextts-tts:2"):
        assert model in runbook

    for evidence in (
        "image tag and digest",
        "adapter package version resolved from PyPI",
        "target model's expected adapter package baseline from this runbook",
        "registry entry used",
        "accelerator request (`gpu` or `cpu-only`)",
        "voice id, voice path, or `none`",
        "manual audio usability verdict (`--audio-usable yes` for a passing run)",
        "voice path existence inside the disposable pod when `--voice` is a file path",
        "runtime capability snapshot from the pod",
        "pre-pull clean-state probe proving the target manifest, model link, "
        "and adapter runtime are not already present",
        "`vox pull <model>` output",
        "machine-readable `real` durations for pull, short synthesis, and long synthesis",
        "adapter, runtime, model, manifest, and blob storage usage after pull",
        "short synthesis wall time",
        "long synthesis wall time",
        "generated audio duration",
        "generated audio stream metadata, including codec, sample rate, and channels",
        "generated audio signal stats proving the WAV is not silent",
        "pod memory and GPU memory snapshots after pull, short synthesis, and long synthesis",
        "output WAV artifact",
        "copied WAV byte size and SHA-256 digest",
        "smoke result and failed-step summary",
        "audio is usable",
        "exact failure output",
    ):
        assert evidence in runbook


def test_expressive_adapter_smoke_runbook_pins_published_adapter_baseline():
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    for package in adapter_package_specs():
        assert package in runbook

    assert "resolve adapter packages from PyPI" in runbook
    assert "not from a local\nsource tree or a patched live cluster directory" in runbook


def test_expressive_adapter_status_uses_current_adapter_package_versions():
    status = Path("docs/expressive-adapter-status.md").read_text()

    for package in adapter_package_specs():
        assert package in status


def test_expressive_adapter_smoke_runbook_preserves_runtime_and_artifact_boundaries():
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    for requirement in (
        "Runtime dependencies are installed under `$VOX_HOME/runtime/<adapter>`",
        "Model files are stored in the model store, not in the adapter package or base image",
        "reference WAV copied into the disposable\nPVC or mounted as test-only data, then pass it with `--voice`",
        "fails the run if that path does not exist\ninside the disposable pod",
        "The adapter is expected to reject\nrequests without reference audio or a voice-path prompt",
        "Do not delete or modify production voice data under `$VOX_HOME/voices`",
        "Failures, if any, are classified as Vox, adapter, dependency, upstream, or hardware",
    ):
        assert requirement in runbook


def test_expressive_adapter_status_tracks_all_goal_targets_and_smoke_gap():
    status = Path("docs/expressive-adapter-status.md").read_text()

    for model in (
        "cosyvoice2-tts:0.5b",
        "dia-tts:1.6b",
        "orpheus-tts:medium-3b",
        "indextts-tts:2",
    ):
        assert model in status

    assert "Previously cluster-smoked successfully, but slow" in status
    assert "Pending isolated GPU smoke" in status
    assert "registry requires `min_vram_gb=8`" in status
    assert "registry requires `min_vram_gb=12`" in status
    assert status.count("registry requires `min_vram_gb=10`") == 2
    assert "Existing-server smoke is available for the currently running Vox endpoint" in status
    assert "That path is the default for\nchecking the existing Vox service" in status
    assert "requested model detail from\n`GET /v1/models/{model}`" in status
    assert "`/v1/models/loaded` before and after synthesis" in status
    assert "without creating namespaces, PVCs, or running `vox pull`" in status
    assert "Use `--inspect-only` when the goal is read-only inspection" in status
    assert "skips\n`/v1/audio/speech` and records no synthesis cases" in status
    assert "Full served smoke can still\nchange in-memory loaded model state and VRAM" in status
    assert "not sufficient to mark a model production-ready" in status
    assert "Do not create a new namespace or PVC just because a\nmodel needs testing" in status
    assert "vox pull` succeeds without `VOX_ALLOW_INCOMPATIBLE" in status
    assert "Model files are stored in the model store and storage usage is recorded" in status
    assert "Adapter package, runtime, manifest, and blob storage usage is recorded" in status
    assert "pre-pull clean-state probe proves the target manifest, model link" in status


def test_expressive_adapter_status_names_local_regression_evidence():
    status = Path("docs/expressive-adapter-status.md").read_text()

    for evidence in (
        "`tests/test_cosyvoice_adapter.py`; the test proves `prepare_runtime()` can",
        "bootstrap the isolated runtime without loading model weights",
        "rejects `cosyvoice.cli.cosyvoice` modules loaded\n  from outside `$VOX_HOME/runtime/cosyvoice`",
        "`tests/test_dia_adapter.py`; the test\n  proves the isolated Transformers runtime",
        "without loading\n  processors or model weights",
        "rejects\n  Dia-capable Transformers modules loaded from the Vox app environment",
        "Pull atomicity across adapter runtime preparation is covered by",
        "Vox does not save a model\n  manifest when `prepare_runtime()` fails",
        "`tests/test_orpheus_adapter.py`",
        "a stale `orpheus_tts` module missing `OrpheusModel` and a\n  broken runtime import probe are repaired",
        "rejects `orpheus_tts` modules loaded from\n  outside `$VOX_HOME/runtime/orpheus`",
        "`tests/test_indextts_adapter.py`",
        "a stale `indextts.infer_v2` module missing `IndexTTS2` and a\n  broken runtime import probe are repaired",
        "rejects `indextts.infer_v2` modules loaded\n  from outside `$VOX_HOME/runtime/indextts`",
    ):
        assert evidence in status


def test_expressive_adapter_status_records_dia_budget_finding():
    status = Path("docs/expressive-adapter-status.md").read_text()

    for evidence in (
        "Current Dia Cluster Finding",
        "`ghcr.io/eleven-am/vox:v0.2.86`",
        "`vox-dia==0.2.11`",
        "`/home/vox/.vox/runtime/dia` only contained",
        "`--max-vram 10GiB --vram-headroom 1GiB`",
        "`min_vram_gb=12`",
        "not a successful smoke test",
        "fresh pull and synthesis in the disposable smoke namespace",
    ):
        assert evidence in status


def test_expressive_adapter_smoke_runbook_requires_durable_evidence_record():
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    assert "Store one evidence file per model next to the copied WAV artifacts" in runbook
    assert "Do not keep\nthe only record in chat logs or terminal scrollback" in runbook
    assert "${MODEL//[:\\/]/-}-evidence.md" in runbook

    for field in (
        "Variant:",
        "Accelerator request:",
        "Voice:",
        "Manual audio usable:",
        "Image digest:",
        "Adapter package:",
        "Expected adapter package:",
        "Runtime capability snapshot:",
        "## Pre-Pull Clean State",
        "## Voice Reference",
        "Voice value:",
        "Voice path check:",
        "Voice path exists:",
        "## Model Resolution",
        "Resolved variant:",
        "Preferred backend:",
        "Manifest path:",
        "Model store path:",
        "Model store path exists:",
        "Runtime root:",
        "Runtime paths:",
        "Manifest exists:",
        "Dirty:",
        "## Adapter Packages",
        "Used VOX_ALLOW_INCOMPATIBLE: no",
        "## Resource Snapshot After Pull",
        "## Storage Snapshot After Pull",
        "Filesystem:",
        "Adapter package storage:",
        "Runtime storage:",
        "Model storage:",
        "Manifest storage:",
        "Blob storage:",
        "## Resource Snapshot After Short Synthesis",
        "## Resource Snapshot After Long Synthesis",
        "Pod memory:",
        "GPU memory:",
        "## Audio Durations",
        "## Audio Stream Metadata",
        "## Audio Signal Stats",
        "## Copied Artifact Stats",
        "Short WAV bytes:",
        "Short WAV sha256:",
        "Long WAV bytes:",
        "Long WAV sha256:",
        "## Smoke Result",
        "Result:",
        "Failed steps:",
        "Failure class: Vox / adapter / dependency / upstream / hardware / none",
        "Exact error:",
    ):
        assert field in runbook


def test_expressive_adapter_smoke_script_uses_current_adapter_package_baselines():
    script = Path("scripts/expressive-adapter-smoke.sh").read_text()
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    for package_spec in adapter_package_specs():
        assert f'expected_adapter_package="{package_spec}"' in script
        assert package_spec in runbook
