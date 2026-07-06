# Adapter Runtime Contract

This document is a compatibility pointer for older goals, notes, and review
threads that still reference `docs/adapter-runtime-contract.md`.

The adapter runtime contract now lives in two more precise documents:

- [Vox Adapter Contract](adapter-contract.md) defines the packaging boundary
  between `vox-runtime`, adapter packages, adapter runtime directories, model
  storage, and Docker images.
- [Adapter Runtime Dependency Policy](adapter-runtime-dependency-policy.md)
  defines runtime dependency pinning, `--upgrade`, install verification,
  repair, and pull-time preparation rules for `$VOX_HOME/runtime/<runtime-name>`.

Read both documents before changing adapter packaging, runtime bootstrap,
model pull behavior, or Docker image contents.
