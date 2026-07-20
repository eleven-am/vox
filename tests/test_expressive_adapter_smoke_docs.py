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
    assert "disabled by default" in script
    assert "VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1" in script
    assert '[[ "${VOX_ENABLE_DISPOSABLE_K8S_SMOKE:-0}" != "1" ]]' in script
    assert "Use scripts/expressive-adapter-served-smoke.py" in script
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
    assert (
        "refuses to create the disposable namespace, PVC, or pod unless the guard\n"
        "environment variable is set"
    ) in runbook
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
    assert "smoke-expressive-local" in makefile
    assert "make smoke-expressive-local MODEL=<model>" in makefile
    assert "SMOKE_PROOF_TARGET=cosyvoice|dia|orpheus|indextts" in makefile
    assert '$(if $(SMOKE_PROOF_TARGET),--proof-target "$(SMOKE_PROOF_TARGET)",)' in makefile
    assert "scripts/expressive-adapter-served-smoke.py" in makefile
    assert "scripts/expressive-adapter-local-smoke.py" in makefile
    assert "SMOKE_EXPECT_ADAPTER=vox-dia" in makefile
    assert "SMOKE_EXPECT_ADAPTER_PACKAGE=vox-dia==0.2.15" in makefile
    assert "SMOKE_EXPECT_RUNTIME=dia" in makefile
    assert "VERIFY_MODEL=dia-tts:1.6b" in makefile
    assert "VERIFY_PROOF_TARGET=dia" in makefile
    assert "SMOKE_EXPECT_MODEL_LINK=dia-tts" in makefile
    assert "SMOKE_RESOURCE_SAMPLE_INTERVAL=1.0" in makefile
    assert '--expect-adapter "$(SMOKE_EXPECT_ADAPTER)"' in makefile
    assert '--expect-adapter-package "$(SMOKE_EXPECT_ADAPTER_PACKAGE)"' in makefile
    assert '--expect-runtime "$(SMOKE_EXPECT_RUNTIME)"' in makefile
    assert '--expect-model-link "$(SMOKE_EXPECT_MODEL_LINK)"' in makefile
    assert '--resource-sample-interval "$(SMOKE_RESOURCE_SAMPLE_INTERVAL)"' in makefile
    assert "SMOKE_ESTIMATE_ONLY=1" in makefile
    assert "--estimate-only" in makefile
    assert "SMOKE_BASE_URL=http://127.0.0.1:8000" in makefile
    assert "SMOKE_API_KEY=..." in makefile
    assert "SMOKE_PARAMS_JSON='{}'" in makefile
    assert "SMOKE_MEMORY_SAMPLE_INTERVAL=1.0" in makefile
    assert '--memory-sample-interval "$(SMOKE_MEMORY_SAMPLE_INTERVAL)"' in makefile
    assert "SMOKE_INSPECT_ONLY=1" in makefile
    assert "SMOKE_FAILURE_CLASS=dependency" in makefile
    assert "SMOKE_FAILURE_NOTE='missing runtime package'" in makefile
    assert '--failure-note "$(SMOKE_FAILURE_NOTE)"' in makefile
    assert "--inspect-only" in makefile


def test_expressive_adapter_served_smoke_script_uses_existing_server_only():
    script = Path("scripts/expressive-adapter-served-smoke.py").read_text()
    evidence_helpers = Path("src/vox/smoke/evidence.py").read_text()
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
    assert "clean_pull_proof" in script
    assert "clean_pull_blockers" in script
    assert "existing-server smoke cannot prove a clean model pull" in script
    assert "--params-json" in script
    assert "--audio-usable" in script
    assert "--failure-class" in script
    assert "--failure-note" in script
    assert "failure_class" in script
    assert "failure_note" in script
    assert "failing smoke run must set --failure-class" in evidence_helpers
    assert "classified failing smoke run must include --failure-note" in evidence_helpers
    assert "passing smoke run must not set --failure-note" in evidence_helpers
    assert "passing smoke run must use --failure-class none" in evidence_helpers
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
    assert "`/v1/system/memory` before synthesis" in runbook
    assert "`/v1/system/memory` after synthesis" in runbook
    assert "per-synthesis `/v1/system/memory` samples under `memory_samples`" in runbook
    assert "`clean_pull_proof: false`" in runbook
    assert "`clean_pull_blockers`" in runbook
    assert (
        "cannot prove a\nclean `vox pull`, a clean adapter package install, or a clean adapter runtime\ninstall"
        in runbook
    )
    assert "--memory-sample-interval 0" in runbook
    assert "SMOKE_MEMORY_SAMPLE_INTERVAL=0" in runbook
    assert "explicit `failure_reasons`" in runbook
    assert "--failure-class adapter" in runbook
    assert "--failure-note \"runtime verification failed after install\"" in runbook
    assert "Full existing-server smoke can change in-memory scheduler state and VRAM usage" in runbook
    assert "Use `--inspect-only` for read-only checks" in runbook
    assert "not enough to mark an adapter fully production-ready" in runbook
    assert "never the key value" in runbook
    assert "SMOKE_API_KEY=..." in runbook
    assert "make smoke-expressive-served MODEL=dia-tts:1.6b" in runbook
    status = Path("docs/expressive-adapter-status.md").read_text()
    assert "plus `/v1/system/memory` before and after synthesis" in status
    assert "per-request `/v1/system/memory` samples under each synthesis case" in status
    assert "RAM/VRAM evidence is recorded" in status
    assert "continuous sample summaries with peak observed RAM\n    and GPU memory" in status
    assert "The local estimate-only clean-pull preflight was run on Roy's Mac" in status
    assert "current live `vox` pod no longer has the model installed" in status
    assert "GET\n/v1/models/cosyvoice2-tts:0.5b` returned HTTP 404" in status
    assert "Do not treat the previous\nsuccessful cluster smoke as current served evidence" in status
    assert "--estimate-only" in status
    assert "did not run Docker, did not run `vox pull`, and did not download\nmodel files" in status
    assert "nari-labs/Dia-1.6B-0626" in status
    assert "reported 12 selected files but no file sizes" in status
    assert "`--allow-large-download`" in status
    assert "The local estimate-only clean-pull preflight was run on Roy's Mac with:" in status
    assert (
        "uv run python scripts/expressive-adapter-local-smoke.py \\\n"
        "  --model orpheus-tts:medium-3b"
    ) in status
    assert "`orpheus-tts:medium-3b` to `canopylabs/orpheus-tts-0.1-finetune-prod`" in status
    assert "not\na valid Orpheus target: missing Torch, missing CUDA, Darwin/arm64 host, and\nunknown VRAM" in status
    assert "reported 20 selected files but no file sizes" in status
    assert "would not bypass the missing runtime\nrequirements" in status
    assert "GET\n/v1/models/orpheus-tts:medium-3b` returned HTTP 404" in status
    assert "provides no Orpheus load\nor synthesis evidence" in status
    assert (
        "uv run python scripts/expressive-adapter-local-smoke.py \\\n"
        "  --model indextts-tts:2"
    ) in status
    assert "`indextts-tts:2` to\n`IndexTeam/IndexTTS-2`" in status
    assert "not a valid IndexTTS target: missing Torch,\nmissing CUDA, Darwin/arm64 host, and unknown VRAM" in status
    assert "reported 21\nselected files but no file sizes" in status
    assert "The proof-target preset path was exercised on Roy's Mac without Docker" in status
    assert "  --proof-target cosyvoice \\" in status
    assert "automatically applied the expected clean-pull state: `vox-cosyvoice==0.1.10`" in status
    assert "all 18 Hugging Face\nfiles reported unknown sizes" in status
    assert status.count("The newer proof-target preset path was also exercised without Docker") == 3
    assert "  --proof-target dia \\" in status
    assert "  --proof-target orpheus \\" in status
    assert "  --proof-target indextts \\" in status
    assert "automatically applied the expected clean-pull\nstate: `vox-dia==0.2.15`" in status
    assert "automatically applied the expected clean-pull\nstate: `vox-orpheus==0.1.7`" in status
    assert "automatically applied the expected clean-pull\nstate: `vox-indextts==0.1.21`" in status
    assert "default voice `samantha`" in status
    assert "not acceptable completion evidence for IndexTTS" in status
    assert "actual cloned voice id `44a66a38` (`Samantha (Her)`)" in status
    assert (
        "The first invalid check with `voice=samantha` loaded IndexTTS in\nabout 16.7s but returned HTTP 400"
        in status
    )
    assert "Short text: HTTP 200, 37 input characters, 215596-byte WAV" in status
    assert "Long text: HTTP 200, 241 input characters, 905772-byte WAV" in status
    assert "estimated loaded VRAM at 8.5GB" in status
    assert "Python\n`urllib.request` in `scripts/expressive-adapter-served-smoke.py` hung" in status
    assert "Bounded `curl` against the same endpoint completed normally" in status


def test_expressive_adapter_smoke_runbook_documents_local_docker_clean_pull():
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()
    local_script = Path("scripts/expressive-adapter-local-smoke.py").read_text()

    for expected in (
        "Local Docker Clean-Pull Smoke",
        "without touching the live\ncluster",
        "disposable scratch directory",
        "`$VOX_HOME` -> `<scratch>/<model>/vox-home`",
        "`HF_HOME` / `HUGGINGFACE_HUB_CACHE` -> `<scratch>/<model>/hf-cache`",
        "refuses to run `docker` or `vox pull` unless `--allow-download` is\npassed",
        "df -h /tmp",
        "Before allowing any Docker or pull work, run the host-side estimate-only path",
        "--estimate-only",
        "SMOKE_ESTIMATE_ONLY=1",
        "`--estimate-only` is not a smoke pass",
        "before Docker can pull\nan image or Vox can download model files",
        "If variant resolution reports missing runtime requirements",
        "`download_guard_failures`",
        "`--allow-large-download` does not bypass missing runtime\nrequirements",
        "For CUDA-only adapters such as Dia, Orpheus, and IndexTTS",
        "Do not\nuse the `:cpu` or `:lean` images for these GPU-only clean-pull checks",
        "scripts/expressive-adapter-local-smoke.py",
        "--image ghcr.io/eleven-am/vox:latest",
        "make smoke-expressive-local MODEL=dia-tts:1.6b",
        "SMOKE_IMAGE=ghcr.io/eleven-am/vox:latest",
        "--expect-adapter vox-dia",
        "--expect-adapter-package vox-dia==0.2.15",
        "--expect-runtime dia",
        "--expect-model-link dia-tts",
        "SMOKE_EXPECT_ADAPTER=vox-dia",
        "SMOKE_EXPECT_ADAPTER_PACKAGE=vox-dia==0.2.15",
        "SMOKE_EXPECT_RUNTIME=dia",
        "SMOKE_EXPECT_MODEL_LINK=dia-tts",
        "SMOKE_RESOURCE_SAMPLE_INTERVAL=1.0",
        "SMOKE_MAX_DOWNLOAD_GB=20",
        "SMOKE_ESTIMATE_ONLY=1",
        "SMOKE_ALLOW_DOWNLOAD=1",
        "SMOKE_ALLOW_LARGE_DOWNLOAD=1",
        "SMOKE_CLEANUP=1",
        "`evidence_schema_version`",
        "`download_estimate`",
        "`proof_ready`",
        "`proof_blockers`",
        "`--max-download-gb 20`",
        "would leave less than `--min-free-gb` free on the scratch\nfilesystem",
        "`--allow-large-download`",
        "would breach the\n`--min-free-gb` reserve",
        "This is separate from\n`--allow-download`",
        "local-smoke-evidence.json",
        "`evidence_schema_version: 2`",
        "schema 2 adds the\nrequired `goal_checklist` proof map",
        "resource snapshots before and after smoke\nwhere measurable",
        "Docker image inspect metadata including image ID and repo\ndigests when available",
        "Linux RAM totals/available bytes from `/proc/meminfo`",
        "optional\nVRAM usage from `nvidia-smi`",
        "samples RAM and `nvidia-smi` VRAM during `vox pull`,\nshort synthesis, and long synthesis",
        "`resource_samples`",
        "`resource_sample_interval_s`",
        "`--resource-sample-interval 0` to disable\ncontinuous sampling",
        "`goal_checklist`",
        "adapter production-readiness goal to the concrete proof fields",
        "Each check has\n`status`, `evidence`, and `blockers` keys",
        "blocked checks explain the missing command, state, artifact,\n"
        "resource, usability, or failure-classification proof",
        "copied audio stats including sample\nwidth, peak, RMS, and silence detection",
        "Silent generated WAVs fail the run even if",
        "post-pull adapter/runtime/manifest/model-store state",
        "Use `--expect-adapter`, `--expect-adapter-package`, `--expect-runtime`, and",
        "`--expect-adapter-package NAME==VERSION`",
        "matches the expected runbook baseline",
        "`state_failures`",
        "entries and adapter package metadata must be absent before pull and present",
        "Missing post-pull state or reused scratch state fails the smoke run",
        "When `--voice` looks like a file path, the local helper checks that the path\n"
        "exists inside the disposable container before `vox pull`",
        "`voice_reference_failures`",
        "Voice IDs such as `samantha` are passed through without a file-existence check",
        "Failing local smoke runs must set `--failure-class`",
        "`adapter`, `dependency`, `upstream`, or `hardware`; passing runs must leave the\nclass as `none`",
        "Classified failures must also\nset `--failure-note`",
        "`--failure-note \"upstream wheel is unavailable for linux/arm64\"`",
        "If the runtime snapshot or pre-pull clean-state check fails, the helper skips\n`vox pull`",
        "If the download estimate fails, is not parseable, exceeds\n`--max-download-gb`",
        "reserve-breaching",
        "unless\n`--allow-large-download` is passed",
        "If variant resolution reports missing\nruntime requirements",
        "skips `vox pull` regardless of\n`--allow-large-download`",
        "use a compatible image/host",
        "recorded as `pre-pull guard failed`",
        "If `vox pull` fails, the helper skips short/long synthesis",
        "`skipped_commands`",
        "The helper also evaluates the full clean-pull proof contract before writing\n"
        "evidence",
        "`proof_ready: true` means the run had a fresh pull",
        "`proof_ready: false` is paired with `proof_blockers`",
        "It copies generated WAV files into the `artifacts` directory before optional\ncleanup",
        "With `--cleanup`, it removes `vox-home`, Hugging Face cache, XDG cache,\nand temp directories",
        "rm -rf /tmp/vox-adapter-lab/dia-tts-1.6b",
    ):
        assert expected in runbook

    assert "would breach the configured free-space reserve" in local_script
    assert "EVIDENCE_SCHEMA_VERSION = 2" in local_script
    assert "def _proof_readiness(" in local_script


def test_expressive_adapter_smoke_runbook_documents_clean_pull_proof_queue():
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    for expected in (
        "Approved Clean-Pull Proof Queue",
        "remaining expressive-adapter proof targets",
        "approved non-production Linux x86_64 CUDA host",
        "not for Roy's live Vox PVC",
        "not for the local Mac when the estimate-only guard reports missing",
        "df -h /tmp",
        "Run the estimate-only command first",
        "If it reports missing runtime\nrequirements, do not pass `--allow-download`",
        "guarded proof presets for all four goal targets",
        "SMOKE_PROOF_TARGET=cosyvoice SMOKE_ESTIMATE_ONLY=1",
        "SMOKE_PROOF_TARGET=dia SMOKE_ESTIMATE_ONLY=1",
        "SMOKE_PROOF_TARGET=orpheus SMOKE_ESTIMATE_ONLY=1",
        "SMOKE_PROOF_TARGET=indextts SMOKE_ESTIMATE_ONLY=1",
        "CosyVoice baseline clean-pull proof",
        "SMOKE_PROOF_TARGET=cosyvoice SMOKE_RESOURCE_SAMPLE_INTERVAL=1.0",
        "Dia clean-pull proof",
        "SMOKE_PROOF_TARGET=dia SMOKE_RESOURCE_SAMPLE_INTERVAL=1.0",
        "Orpheus clean-pull proof",
        "SMOKE_PROOF_TARGET=orpheus SMOKE_RESOURCE_SAMPLE_INTERVAL=1.0",
        "Orpheus uses gated Hugging Face assets",
        "`--failure-class upstream`",
        "IndexTTS clean-pull proof requires a voice id",
        "SMOKE_PROOF_TARGET=indextts SMOKE_RESOURCE_SAMPLE_INTERVAL=1.0",
        "--proof-target dia",
        "Do not use or delete production voice data under `$VOX_HOME/voices`",
        "`--failure-class Vox`",
        "`--failure-class adapter`",
        "`--failure-class dependency`",
        "`--failure-class hardware`",
        "Passing proof requires `proof_ready: true`",
        "If `proof_ready` is false, use `proof_blockers` as the\n"
        "authoritative list",
        "make verify-expressive-local-evidence "
        "EVIDENCE=/tmp/vox-adapter-lab/dia-tts-1.6b/artifacts/local-smoke-evidence.json "
        "VERIFY_PROOF_TARGET=dia VERIFY_MODEL=dia-tts:1.6b",
        "scripts/verify-expressive-adapter-evidence.py",
        "checks\n`evidence_schema_version`, requires `proof_ready: true`, and independently",
        "checks the requested model/proof target, required command results",
        "clean pre-pull manifest/runtime/model state",
        "actual post-pull adapter/runtime/manifest/model state",
        "positive model blob storage",
        "copied artifact file presence, artifact byte counts",
        "artifact SHA-256 digests",
        "`goal_checklist`",
        "file is for the wrong model",
        "reused scratch state",
        "missing expected post-pull state",
        "missing its copied WAV artifacts",
        "has blocked goal checks",
        "manually edited into an inconsistent state",
        "`--audio-usable yes`",
    ):
        assert expected in runbook

    assert runbook.count("SMOKE_ALLOW_DOWNLOAD=1") >= 3
    assert runbook.count("SMOKE_ALLOW_LARGE_DOWNLOAD=1") >= 3
    assert runbook.count("SMOKE_CLEANUP=1") >= 3
    assert runbook.count("SMOKE_RESOURCE_SAMPLE_INTERVAL=1.0") >= 3


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
    assert "Do not create a new namespace or PVC for\ncluster testing" in runbook
    assert "If cluster testing is needed, use the existing Vox namespace,\nservice, and HTTP port" in runbook
    assert "Legacy Disposable Kubernetes Smoke" in runbook
    assert "disabled by default" in runbook
    assert "It is not the workflow for Roy's current Vox deployment" in runbook
    assert "VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1" in runbook
    assert "VOX_SMOKE_NS=vox-adapter-smoke" in runbook
    assert "VOX_SMOKE_PVC=vox-adapter-smoke-data" in runbook
    assert "Do not create one as a\nshortcut" in runbook
    assert "do not mutate, clean, reinstall, restart, or scale the live Vox" in runbook
    assert "kubectl get namespace \"$VOX_SMOKE_NS\"" in runbook
    assert "kubectl -n \"$VOX_SMOKE_NS\" get pvc \"$VOX_SMOKE_PVC\"" in runbook
    assert "Do not invent new namespace/PVC names" in runbook


def test_expressive_adapter_smoke_runbook_lists_required_models_and_evidence():
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    for model in ("cosyvoice2-tts:0.5b", "dia-tts:1.6b", "orpheus-tts:medium-3b", "indextts-tts:2"):
        assert model in runbook

    for evidence in (
        "image tag and digest",
        "adapter package version resolved from PyPI",
        "target model's expected adapter package baseline from this runbook",
        "registry entry used",
        "Docker image inspect metadata including image ID and repo\n"
        "digests when available",
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
        "invalid, non-positive, silent, or shorter-than-expected long output fails the smoke",
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
        "RAM/VRAM evidence is recorded. Existing-server smoke records",
        "`/v1/system/memory` before and after synthesis plus per-request memory",
        "Output durations are readable, positive, and plausible for the text length",
        "the smoke helpers fail a long output that is shorter than the short output",
        "Failures, if any, are classified as Vox, adapter, dependency, upstream, or hardware",
        "and include a concrete failure note with the likely cause or next fix",
        "`vox pull` succeeds in the isolated clean-pull environment without `VOX_ALLOW_INCOMPATIBLE`",
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
    assert "Not proven; current live `vox` pod does not have the model installed" in status
    assert "Current live `vox` pod serves Samantha voice successfully" in status
    assert "registry requires `min_vram_gb=8`" in status
    assert "registry requires `min_vram_gb=12`" in status
    assert status.count("registry requires `min_vram_gb=10`") == 2
    assert "Existing-server smoke is available for the currently running Vox endpoint" in status
    assert "That path is the default for\nchecking the existing Vox service" in status
    assert "requested model detail from\n`GET /v1/models/{model}`" in status
    assert "`/v1/models/loaded` before and after synthesis" in status
    assert "plus `/v1/system/memory` before and after synthesis" in status
    assert "per-request `/v1/system/memory` samples under each synthesis case" in status
    assert "without creating\nnamespaces, PVCs, or running `vox pull`" in status
    assert "Use `--inspect-only` when the goal is read-only inspection" in status
    assert "skips\n`/v1/audio/speech` and records no synthesis cases" in status
    assert "Full served smoke can still\nchange in-memory loaded model state and VRAM" in status
    assert "not sufficient to mark a model production-ready" in status
    assert "Do not create a new namespace or PVC in the live\ncluster just because a model needs testing" in status
    assert "Approved Clean-Pull Proof Queue" in status
    assert "expressive-adapter-smoke.md#approved-clean-pull-proof-queue" in status
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
        "Current Dia Finding",
        "only been tested on\nGPUs with PyTorch/CUDA",
        "CPU support is future work",
        "https://github.com/nari-labs/dia#hardware-and-inference-speed",
        "requires around 10GB of VRAM",
        "transformers==4.57.6",
        "only claims Linux x86_64 CUDA/Torch today",
        "Spark/ARM NVIDIA remain unsupported unless a real upstream-compatible backend\nis identified and smoked",
        "`ghcr.io/eleven-am/vox:v0.2.86`",
        "`vox-dia==0.2.11`",
        "`/home/vox/.vox/runtime/dia` only contained",
        "`--max-vram 10GiB --vram-headroom 1GiB`, while that release budgeted Dia",
        "`min_vram_gb=12`",
        "Current Vox has removed the VRAM budget flags entirely",
        "optional `--idle-trim-ttl` auto trim plus the",
        "manual `POST /v1/system/trim` endpoint) and by TTL unload",
        "wrote evidence to `/tmp/vox-served-smoke/evidence.json`",
        "Short and long Dia synthesis both returned HTTP 500",
        "Cannot satisfy VRAM budget: projected 12500000000 bytes plus headroom\n"
        "1073741824 exceeds max 10737418240 bytes",
        "loaded model state before and\nafter the run still contained only `parakeet-stt:tdt-0.6b-v3`",
        "current served\nfailure as deployment hardware/budget",
        "generic `Internal synthesis error`",
        "HTTP 507",
        "not a successful smoke test",
        "fresh pull and synthesis in an approved non-production clean-pull environment",
        "Do not create a new\nnamespace or PVC in the live cluster",
    ):
        assert evidence in status


def test_expressive_adapter_status_records_orpheus_finding():
    status = Path("docs/expressive-adapter-status.md").read_text()

    for evidence in (
        "Current Orpheus Finding",
        "https://github.com/canopyai/Orpheus-TTS",
        "installs `orpheus-speech`, which uses vLLM under the hood",
        "requires accepting gated model access",
        "`canopylabs/orpheus-tts-0.1-finetune-prod`\ncurrently resolves to the same Hugging Face revision as",
        "`canopylabs/orpheus-3b-0.1-ft`",
        "only claims Linux x86_64 CUDA/Torch today",
        "does not have a portable non-vLLM backend",
        "only wires preset voices and\nrejects Vox `reference_audio` / `reference_text` clearly",
    ):
        assert evidence in status


def test_expressive_adapter_status_records_indextts_finding():
    status = Path("docs/expressive-adapter-status.md").read_text()

    for evidence in (
        "Current IndexTTS Finding",
        "https://github.com/index-tts/index-tts",
        "zero-shot speaker cloning",
        "duration control is not enabled in the current release",
        "does not expose\na duration target parameter",
        "now exposes the upstream advanced generation\ncontrols",
        "`do_sample`,\n`temperature`, `top_p`, `top_k`, `num_beams`, `repetition_penalty`,",
        "`length_penalty`, `max_mel_tokens`, and `max_text_tokens_per_segment`",
        "not a clean-pull proof",
        "fresh\nclean-pull smoke in an approved non-production environment",
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
