REGISTRY := docker.io
IMAGE := $(REGISTRY)/elevenam/vox
PLATFORMS := linux/amd64,linux/arm64
SPARK_PLATFORM := linux/arm64
VERSION = $(shell git tag --list 'v*' --sort=-version:refname | head -n1 || echo "v0.0.0")
APP_VERSION = $(shell sed -nE 's/^version = "([0-9]+\.[0-9]+\.[0-9]+)".*/\1/p' pyproject.toml | head -n1)

GPU_BASE := nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
CPU_BASE := python:3.12-slim
SPARK_BASE := nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
SPARK_ORT_PACKAGE ?= onnxruntime-gpu
SPARK_ORT_INDEX_URL ?=
SPARK_ORT_EXTRA_INDEX_URL ?=
SPARK_ORT_WHEEL ?= https://pypi.jetson-ai-lab.io/jp6/cu129/+f/2e3/a07114007df15/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl

.PHONY: build build-cpu build-spark build-local build-local-cpu build-local-spark push tag clean setup-buildx current-version bump-patch bump-minor bump-major test proto

build:
	@test "$(patsubst v%,%,$(VERSION))" = "$(APP_VERSION)" || \
		(echo "pyproject.toml version $(APP_VERSION) does not match release tag $(VERSION)"; exit 1)
	docker buildx build \
		--platform $(PLATFORMS) \
		--build-arg BASE_IMAGE=$(GPU_BASE) \
		--tag $(IMAGE):$(VERSION) \
		--tag $(IMAGE):latest \
		--push \
		.

build-cpu:
	@test "$(patsubst v%,%,$(VERSION))" = "$(APP_VERSION)" || \
		(echo "pyproject.toml version $(APP_VERSION) does not match release tag $(VERSION)"; exit 1)
	docker buildx build \
		--platform $(PLATFORMS) \
		--build-arg BASE_IMAGE=$(CPU_BASE) \
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
		--tag $(IMAGE):$(VERSION)-spark \
		--tag $(IMAGE):spark \
		--push \
		.

build-local:
	docker build --build-arg BASE_IMAGE=$(GPU_BASE) -t vox:local .

build-local-cpu:
	docker build --build-arg BASE_IMAGE=$(CPU_BASE) -t vox:local-cpu .

build-local-spark:
	docker build \
		--platform $(SPARK_PLATFORM) \
		-f Dockerfile.spark \
		--build-arg BASE_IMAGE=$(SPARK_BASE) \
		--build-arg SPARK_ORT_PACKAGE=$(SPARK_ORT_PACKAGE) \
		--build-arg SPARK_ORT_INDEX_URL=$(SPARK_ORT_INDEX_URL) \
		--build-arg SPARK_ORT_EXTRA_INDEX_URL=$(SPARK_ORT_EXTRA_INDEX_URL) \
		--build-arg SPARK_ORT_WHEEL=$(SPARK_ORT_WHEEL) \
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
		$(IMAGE):cpu \
		$(IMAGE):$(VERSION)-cpu \
		$(IMAGE):spark \
		$(IMAGE):$(VERSION)-spark \
		vox:local \
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

proto:
	uv run python -m grpc_tools.protoc \
		-I proto \
		--python_out=src/vox/grpc \
		--grpc_python_out=src/vox/grpc \
		proto/vox.proto
	sed -i.bak 's/^import vox_pb2 as vox__pb2$$/from . import vox_pb2 as vox__pb2/' src/vox/grpc/vox_pb2_grpc.py
	rm -f src/vox/grpc/vox_pb2_grpc.py.bak
