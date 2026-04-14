REGISTRY := docker.io
IMAGE := $(REGISTRY)/elevenam/vox
PLATFORMS := linux/amd64,linux/arm64
VERSION = $(shell git tag --list 'v*' --sort=-version:refname | head -n1 || echo "v0.0.0")
APP_VERSION = $(shell sed -nE 's/^version = "([0-9]+\.[0-9]+\.[0-9]+)".*/\1/p' pyproject.toml | head -n1)

GPU_BASE := nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
CPU_BASE := python:3.12-slim
ORT_REF ?= v1.24.0

.PHONY: build build-cpu build-local build-local-cpu push tag clean setup-buildx current-version bump-patch bump-minor bump-major test

build:
	@test "$(patsubst v%,%,$(VERSION))" = "$(APP_VERSION)" || \
		(echo "pyproject.toml version $(APP_VERSION) does not match release tag $(VERSION)"; exit 1)
	docker buildx build \
		--platform $(PLATFORMS) \
		--target vox \
		--build-arg BASE_IMAGE=$(GPU_BASE) \
		--build-arg ORT_GIT_REF=$(ORT_REF) \
		--tag $(IMAGE):$(VERSION) \
		--tag $(IMAGE):latest \
		--push \
		.

build-cpu:
	@test "$(patsubst v%,%,$(VERSION))" = "$(APP_VERSION)" || \
		(echo "pyproject.toml version $(APP_VERSION) does not match release tag $(VERSION)"; exit 1)
	docker buildx build \
		--platform $(PLATFORMS) \
		--target vox-runtime \
		--build-arg BASE_IMAGE=$(CPU_BASE) \
		--tag $(IMAGE):$(VERSION)-cpu \
		--tag $(IMAGE):cpu \
		--push \
		.

build-local:
	docker build \
		--target vox \
		--build-arg BASE_IMAGE=$(GPU_BASE) \
		--build-arg ORT_GIT_REF=$(ORT_REF) \
		-t vox:local \
		.

build-local-cpu:
	docker build --target vox-runtime --build-arg BASE_IMAGE=$(CPU_BASE) -t vox:local-cpu .

push:
	docker push $(IMAGE):$(VERSION)
	docker push $(IMAGE):latest

tag:
	docker tag vox:local $(IMAGE):$(VERSION)
	docker tag vox:local $(IMAGE):latest

clean:
	docker rmi -f $(IMAGE):latest $(IMAGE):$(VERSION) $(IMAGE):cpu $(IMAGE):$(VERSION)-cpu vox:local vox:local-cpu 2>/dev/null || true

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
