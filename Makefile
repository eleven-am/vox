REGISTRY := docker.io
IMAGE := $(REGISTRY)/elevenam/vox
PLATFORMS := linux/amd64,linux/arm64
GPU_PLATFORM := linux/amd64
SPARK_PLATFORM := linux/arm64
VERSION = $(shell git tag --list 'v*' --sort=-version:refname | head -n1 || echo "v0.0.0")
APP_VERSION = $(shell sed -nE 's/^version = "([0-9]+\.[0-9]+\.[0-9]+)".*/\1/p' pyproject.toml | head -n1)

GPU_BASE := nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
CPU_BASE := python:3.12-slim
SPARK_BASE := nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
SPARK_ORT_PACKAGE ?= onnxruntime==1.23.0
SPARK_ORT_INDEX_URL ?=
SPARK_ORT_EXTRA_INDEX_URL ?=
SPARK_ORT_WHEEL ?=
SPARK_TORCH_WHEEL ?=
SPARK_TORCHAUDIO_WHEEL ?=
SPARK_TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu129
SPARK_TORCH_EXTRA_INDEX_URL ?=

.PHONY: build build-lean build-cpu build-spark build-local build-local-lean build-local-cpu build-local-spark push tag clean setup-buildx current-version bump-patch bump-minor bump-major test test-smoke-runner smoke-expressive smoke-expressive-served smoke-expressive-local proto

build:
	@test "$(patsubst v%,%,$(VERSION))" = "$(APP_VERSION)" || \
		(echo "pyproject.toml version $(APP_VERSION) does not match release tag $(VERSION)"; exit 1)
	docker buildx build \
		--platform $(GPU_PLATFORM) \
		--build-arg BASE_IMAGE=$(GPU_BASE) \
		--build-arg VOX_ACCELERATOR=gpu \
		--build-arg VOX_DEFAULT_DEVICE=auto \
		--tag $(IMAGE):$(VERSION) \
		--tag $(IMAGE):latest \
		--push \
		.

# Lean CPU image (multi-arch): onnxruntime + CT2/ONNX models, no torch stack.
# Runs on Linux amd64/arm64 and Apple-Silicon Docker. Streaming/VAD works;
# torch-based models are refused at pull time.
build-lean:
	@test "$(patsubst v%,%,$(VERSION))" = "$(APP_VERSION)" || \
		(echo "pyproject.toml version $(APP_VERSION) does not match release tag $(VERSION)"; exit 1)
	docker buildx build \
		--platform $(PLATFORMS) \
		--build-arg BASE_IMAGE=$(CPU_BASE) \
		--build-arg VOX_ACCELERATOR=cpu \
		--build-arg VOX_INCLUDE_TORCH=0 \
		--build-arg VOX_DEFAULT_DEVICE=cpu \
		--tag $(IMAGE):$(VERSION)-lean \
		--tag $(IMAGE):lean \
		--push \
		.

# CPU image with the torch stack (multi-arch): runs every model, but torch
# models run on CPU only (slow). Use build-lean unless you need torch on CPU.
build-cpu:
	@test "$(patsubst v%,%,$(VERSION))" = "$(APP_VERSION)" || \
		(echo "pyproject.toml version $(APP_VERSION) does not match release tag $(VERSION)"; exit 1)
	docker buildx build \
		--platform $(PLATFORMS) \
		--build-arg BASE_IMAGE=$(CPU_BASE) \
		--build-arg VOX_ACCELERATOR=cpu \
		--build-arg VOX_INCLUDE_TORCH=1 \
		--build-arg VOX_DEFAULT_DEVICE=cpu \
		--tag $(IMAGE):$(VERSION)-cpu \
		--tag $(IMAGE):cpu \
		--push \
		.

build-spark:
	@test "$(patsubst v%,%,$(VERSION))" = "$(APP_VERSION)" || \
		(echo "pyproject.toml version $(APP_VERSION) does not match release tag $(VERSION)"; exit 1)
	docker buildx build \
		--platform $(SPARK_PLATFORM) \
		-f Dockerfile.spark \
		--build-arg BASE_IMAGE=$(SPARK_BASE) \
		--build-arg SPARK_ORT_PACKAGE=$(SPARK_ORT_PACKAGE) \
		--build-arg SPARK_ORT_INDEX_URL=$(SPARK_ORT_INDEX_URL) \
		--build-arg SPARK_ORT_EXTRA_INDEX_URL=$(SPARK_ORT_EXTRA_INDEX_URL) \
		--build-arg SPARK_ORT_WHEEL=$(SPARK_ORT_WHEEL) \
		--build-arg SPARK_TORCH_WHEEL=$(SPARK_TORCH_WHEEL) \
		--build-arg SPARK_TORCHAUDIO_WHEEL=$(SPARK_TORCHAUDIO_WHEEL) \
		--build-arg SPARK_TORCH_INDEX_URL=$(SPARK_TORCH_INDEX_URL) \
		--build-arg SPARK_TORCH_EXTRA_INDEX_URL=$(SPARK_TORCH_EXTRA_INDEX_URL) \
		--tag $(IMAGE):$(VERSION)-spark \
		--tag $(IMAGE):spark \
		--push \
		.

build-local:
	docker build --build-arg BASE_IMAGE=$(GPU_BASE) --build-arg VOX_ACCELERATOR=gpu --build-arg VOX_DEFAULT_DEVICE=auto -t vox:local .

build-local-lean:
	docker build --build-arg BASE_IMAGE=$(CPU_BASE) --build-arg VOX_ACCELERATOR=cpu --build-arg VOX_INCLUDE_TORCH=0 --build-arg VOX_DEFAULT_DEVICE=cpu -t vox:local-lean .

build-local-cpu:
	docker build --build-arg BASE_IMAGE=$(CPU_BASE) --build-arg VOX_ACCELERATOR=cpu --build-arg VOX_INCLUDE_TORCH=1 --build-arg VOX_DEFAULT_DEVICE=cpu -t vox:local-cpu .

build-local-spark:
	docker build \
		--platform $(SPARK_PLATFORM) \
		-f Dockerfile.spark \
		--build-arg BASE_IMAGE=$(SPARK_BASE) \
		--build-arg SPARK_ORT_PACKAGE=$(SPARK_ORT_PACKAGE) \
		--build-arg SPARK_ORT_INDEX_URL=$(SPARK_ORT_INDEX_URL) \
		--build-arg SPARK_ORT_EXTRA_INDEX_URL=$(SPARK_ORT_EXTRA_INDEX_URL) \
		--build-arg SPARK_ORT_WHEEL=$(SPARK_ORT_WHEEL) \
		--build-arg SPARK_TORCH_WHEEL=$(SPARK_TORCH_WHEEL) \
		--build-arg SPARK_TORCHAUDIO_WHEEL=$(SPARK_TORCHAUDIO_WHEEL) \
		--build-arg SPARK_TORCH_INDEX_URL=$(SPARK_TORCH_INDEX_URL) \
		--build-arg SPARK_TORCH_EXTRA_INDEX_URL=$(SPARK_TORCH_EXTRA_INDEX_URL) \
		-t vox:spark-local \
		.

push:
	docker push $(IMAGE):$(VERSION)
	docker push $(IMAGE):latest

tag:
	docker tag vox:local $(IMAGE):$(VERSION)
	docker tag vox:local $(IMAGE):latest

clean:
	docker rmi -f \
		$(IMAGE):latest \
		$(IMAGE):$(VERSION) \
		$(IMAGE):lean \
		$(IMAGE):$(VERSION)-lean \
		$(IMAGE):cpu \
		$(IMAGE):$(VERSION)-cpu \
		$(IMAGE):spark \
		$(IMAGE):$(VERSION)-spark \
		vox:local \
		vox:local-lean \
		vox:local-cpu \
		vox:spark-local 2>/dev/null || true

setup-buildx:
	docker buildx create --name multiarch --driver docker-container --use || true
	docker buildx inspect --bootstrap

current-version:
	@echo tag=$(VERSION) pyproject=$(APP_VERSION)

bump-patch:
	@current=$$(git tag --list 'v*' --sort=-version:refname | head -n1 || echo "v0.0.0"); \
	if [ "$$current" = "v0.0.0" ]; then \
		new="v0.0.1"; \
	else \
		new=$$(echo $$current | awk -F. '{$$NF = $$NF + 1;} 1' | sed 's/ /./g'); \
	fi; \
	app=$$(sed -nE 's/^version = "([0-9]+\.[0-9]+\.[0-9]+)".*/\1/p' pyproject.toml | head -n1); \
	if [ "$$app" != "$${new#v}" ]; then \
		echo "pyproject.toml version $$app does not match $$new; update pyproject.toml before tagging"; \
		exit 1; \
	fi; \
	echo "Bumping version from $$current to $$new"; \
	git tag $$new; \
	git push origin $$new; \
	echo "Tagged and pushed $$new"

bump-minor:
	@current=$$(git tag --list 'v*' --sort=-version:refname | head -n1 || echo "v0.0.0"); \
	if [ "$$current" = "v0.0.0" ]; then \
		new="v0.1.0"; \
	else \
		new=$$(echo $$current | awk -F. '{$$(NF-1) = $$(NF-1) + 1; $$NF = 0;} 1' | sed 's/ /./g'); \
	fi; \
	app=$$(sed -nE 's/^version = "([0-9]+\.[0-9]+\.[0-9]+)".*/\1/p' pyproject.toml | head -n1); \
	if [ "$$app" != "$${new#v}" ]; then \
		echo "pyproject.toml version $$app does not match $$new; update pyproject.toml before tagging"; \
		exit 1; \
	fi; \
	echo "Bumping version from $$current to $$new"; \
	git tag $$new; \
	git push origin $$new; \
	echo "Tagged and pushed $$new"

bump-major:
	@current=$$(git tag --list 'v*' --sort=-version:refname | head -n1 || echo "v0.0.0"); \
	if [ "$$current" = "v0.0.0" ]; then \
		new="v1.0.0"; \
	else \
		new=$$(echo $$current | awk -F. '{$$(NF-2) = $$(NF-2) + 1; $$(NF-1) = 0; $$NF = 0;} 1' | sed 's/ /./g'); \
	fi; \
	app=$$(sed -nE 's/^version = "([0-9]+\.[0-9]+\.[0-9]+)".*/\1/p' pyproject.toml | head -n1); \
	if [ "$$app" != "$${new#v}" ]; then \
		echo "pyproject.toml version $$app does not match $$new; update pyproject.toml before tagging"; \
		exit 1; \
	fi; \
	echo "Bumping version from $$current to $$new"; \
	git tag $$new; \
	git push origin $$new; \
	echo "Tagged and pushed $$new"

test:
	uv run python -m pytest tests/ -q

test-smoke-runner:
	bash -n scripts/expressive-adapter-smoke.sh
	uv run python -m py_compile scripts/expressive-adapter-served-smoke.py
	uv run python -m py_compile scripts/expressive-adapter-local-smoke.py
	uv run python -m py_compile scripts/verify-expressive-adapter-evidence.py
	uv run python -m pytest tests/test_expressive_adapter_smoke_docs.py tests/test_expressive_adapter_served_smoke.py tests/test_expressive_adapter_local_smoke.py -q
	@bash scripts/expressive-adapter-smoke.sh --model dia-tts:1.6b; \
		rc=$$?; \
		if [ "$$rc" -ne 4 ]; then \
			echo "expected legacy Kubernetes smoke runner to fail closed with exit 4 when not explicitly enabled, got $$rc"; \
			exit 1; \
		fi

smoke-expressive:
	@test -n "$(MODEL)" || (echo "legacy disposable Kubernetes runner; use make smoke-expressive-served for existing Vox. Usage only for approved non-production labs: VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1 make smoke-expressive MODEL=dia-tts:1.6b [SMOKE_VARIANT=onnx] [SMOKE_VOICE=/home/vox/.vox/smoke-voices/reference.wav] [SMOKE_CPU_ONLY=1] [SMOKE_CREATE=1] [SMOKE_AUDIO_USABLE=yes] [SMOKE_FAILURE_CLASS=dependency]"; exit 2)
	bash scripts/expressive-adapter-smoke.sh --model "$(MODEL)" $(if $(SMOKE_VARIANT),--variant "$(SMOKE_VARIANT)",) $(if $(SMOKE_VOICE),--voice "$(SMOKE_VOICE)",) $(if $(SMOKE_CPU_ONLY),--cpu-only,) $(if $(SMOKE_CREATE),--create,) $(if $(SMOKE_AUDIO_USABLE),--audio-usable "$(SMOKE_AUDIO_USABLE)",) $(if $(SMOKE_FAILURE_CLASS),--failure-class "$(SMOKE_FAILURE_CLASS)",)

smoke-expressive-served:
	@test -n "$(MODEL)" || (echo "usage: make smoke-expressive-served MODEL=dia-tts:1.6b [SMOKE_BASE_URL=http://127.0.0.1:8000] [SMOKE_API_KEY=...] [SMOKE_VOICE=voice-id] [SMOKE_PARAMS_JSON='{}'] [SMOKE_MEMORY_SAMPLE_INTERVAL=1.0] [SMOKE_AUDIO_USABLE=yes] [SMOKE_FAILURE_CLASS=dependency] [SMOKE_FAILURE_NOTE='missing runtime package'] [SMOKE_INSPECT_ONLY=1]"; exit 2)
	uv run python scripts/expressive-adapter-served-smoke.py --model "$(MODEL)" --base-url "$(or $(SMOKE_BASE_URL),http://127.0.0.1:8000)" $(if $(SMOKE_API_KEY),--api-key "$(SMOKE_API_KEY)",) $(if $(SMOKE_VOICE),--voice "$(SMOKE_VOICE)",) $(if $(SMOKE_PARAMS_JSON),--params-json '$(SMOKE_PARAMS_JSON)',) $(if $(SMOKE_MEMORY_SAMPLE_INTERVAL),--memory-sample-interval "$(SMOKE_MEMORY_SAMPLE_INTERVAL)",) $(if $(SMOKE_AUDIO_USABLE),--audio-usable "$(SMOKE_AUDIO_USABLE)",) $(if $(SMOKE_FAILURE_CLASS),--failure-class "$(SMOKE_FAILURE_CLASS)",) $(if $(SMOKE_FAILURE_NOTE),--failure-note "$(SMOKE_FAILURE_NOTE)",) $(if $(SMOKE_INSPECT_ONLY),--inspect-only,)

smoke-expressive-local:
	@test -n "$(MODEL)$(SMOKE_PROOF_TARGET)" || (echo "usage: make smoke-expressive-local MODEL=<model> or SMOKE_PROOF_TARGET=cosyvoice|dia|orpheus|indextts [SMOKE_IMAGE=ghcr.io/eleven-am/vox:lean] [SMOKE_VARIANT=onnx] [SMOKE_SCRATCH_ROOT=/tmp/vox-adapter-lab] [SMOKE_EXPECT_ADAPTER=vox-dia] [SMOKE_EXPECT_ADAPTER_PACKAGE=vox-dia==0.2.15] [SMOKE_EXPECT_RUNTIME=dia] [SMOKE_EXPECT_MODEL_LINK=dia-tts] [SMOKE_RESOURCE_SAMPLE_INTERVAL=1.0] [SMOKE_MAX_DOWNLOAD_GB=20] [SMOKE_ESTIMATE_ONLY=1] [SMOKE_ALLOW_DOWNLOAD=1] [SMOKE_ALLOW_LARGE_DOWNLOAD=1] [SMOKE_AUDIO_USABLE=yes] [SMOKE_FAILURE_CLASS=dependency] [SMOKE_FAILURE_NOTE='missing runtime package'] [SMOKE_CLEANUP=1]"; exit 2)
	uv run python scripts/expressive-adapter-local-smoke.py $(if $(MODEL),--model "$(MODEL)",) $(if $(SMOKE_PROOF_TARGET),--proof-target "$(SMOKE_PROOF_TARGET)",) $(if $(SMOKE_IMAGE),--image "$(SMOKE_IMAGE)",) $(if $(SMOKE_VARIANT),--variant "$(SMOKE_VARIANT)",) $(if $(SMOKE_SCRATCH_ROOT),--scratch-root "$(SMOKE_SCRATCH_ROOT)",) $(if $(SMOKE_EXPECT_ADAPTER),--expect-adapter "$(SMOKE_EXPECT_ADAPTER)",) $(if $(SMOKE_EXPECT_ADAPTER_PACKAGE),--expect-adapter-package "$(SMOKE_EXPECT_ADAPTER_PACKAGE)",) $(if $(SMOKE_EXPECT_RUNTIME),--expect-runtime "$(SMOKE_EXPECT_RUNTIME)",) $(if $(SMOKE_EXPECT_MODEL_LINK),--expect-model-link "$(SMOKE_EXPECT_MODEL_LINK)",) $(if $(SMOKE_RESOURCE_SAMPLE_INTERVAL),--resource-sample-interval "$(SMOKE_RESOURCE_SAMPLE_INTERVAL)",) $(if $(SMOKE_MAX_DOWNLOAD_GB),--max-download-gb "$(SMOKE_MAX_DOWNLOAD_GB)",) $(if $(SMOKE_ESTIMATE_ONLY),--estimate-only,) $(if $(SMOKE_ALLOW_DOWNLOAD),--allow-download,) $(if $(SMOKE_ALLOW_LARGE_DOWNLOAD),--allow-large-download,) $(if $(SMOKE_AUDIO_USABLE),--audio-usable "$(SMOKE_AUDIO_USABLE)",) $(if $(SMOKE_FAILURE_CLASS),--failure-class "$(SMOKE_FAILURE_CLASS)",) $(if $(SMOKE_FAILURE_NOTE),--failure-note "$(SMOKE_FAILURE_NOTE)",) $(if $(SMOKE_CLEANUP),--cleanup,)

verify-expressive-local-evidence:
	@test -n "$(EVIDENCE)" || (echo "usage: make verify-expressive-local-evidence EVIDENCE=/path/to/local-smoke-evidence.json [VERIFY_MODEL=dia-tts:1.6b] [VERIFY_PROOF_TARGET=dia]"; exit 2)
	uv run python scripts/verify-expressive-adapter-evidence.py $(if $(VERIFY_MODEL),--expect-model "$(VERIFY_MODEL)",) $(if $(VERIFY_PROOF_TARGET),--expect-proof-target "$(VERIFY_PROOF_TARGET)",) $(EVIDENCE)

proto:
	uv run python -m grpc_tools.protoc \
		-I proto \
		--python_out=src/vox/grpc \
		--pyi_out=src/vox/grpc \
		--grpc_python_out=src/vox/grpc \
		proto/vox.proto
	sed -i.bak 's/^import vox_pb2 as vox__pb2$$/from . import vox_pb2 as vox__pb2/' src/vox/grpc/vox_pb2_grpc.py
	rm -f src/vox/grpc/vox_pb2_grpc.py.bak
