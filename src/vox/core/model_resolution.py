"""Hardware-aware catalog entry resolution.

The registry can contain either today's concrete entries or logical entries with
pull-time variants. This module keeps that selection pure so operations can gate
and download only the resolved concrete entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vox.core.capabilities import (
    CapabilityCheck,
    RuntimeRequirement,
    check_model_capabilities,
    check_runtime_requirement,
    infer_model_requirements,
    runtime_requirement_from_mapping,
)
from vox.core.runtime import RuntimeCapabilities, detect_runtime_capabilities


@dataclass(frozen=True)
class ParsedModelVariantRef:
    name: str
    tag: str
    explicit_tag: bool = False


@dataclass(frozen=True)
class VariantResolution:
    entry: dict[str, Any]
    variant_id: str | None = None
    preferred_backend: str | None = None
    missing: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    snapshot: RuntimeCapabilities | None = None


def parse_model_variant_ref(value: str) -> ParsedModelVariantRef:
    """Parse ``<name>[:<tag>]`` without changing concrete names."""
    explicit_tag = ":" in value
    if explicit_tag:
        name_part, tag = value.split(":", 1)
    else:
        name_part, tag = value, "latest"

    return ParsedModelVariantRef(name=name_part, tag=tag, explicit_tag=explicit_tag)


def resolve_catalog_entry(
    entry: dict[str, Any],
    *,
    snapshot: RuntimeCapabilities | None = None,
    forced_variant: str | None = None,
) -> VariantResolution:
    """Resolve a concrete or logical catalog entry for the current runtime."""
    snapshot = snapshot or detect_runtime_capabilities()
    variants = entry.get("variants")
    if not isinstance(variants, list):
        preferred_backend, backend_warnings = _preferred_backend_for_variant(entry, snapshot)
        check = check_model_capabilities(entry, snapshot=snapshot)
        return VariantResolution(
            entry=dict(entry),
            missing=check.missing,
            warnings=(*check.warnings, *backend_warnings),
            preferred_backend=preferred_backend or check.preferred_backend,
            snapshot=snapshot,
        )

    variant_results = _evaluate_variants(entry, variants, snapshot)
    if forced_variant:
        for variant, concrete, check in variant_results:
            if _variant_matches(variant, forced_variant):
                return VariantResolution(
                    entry=concrete,
                    variant_id=str(variant.get("id", "")) or forced_variant,
                    missing=check.missing,
                    warnings=check.warnings,
                    preferred_backend=check.preferred_backend,
                    snapshot=snapshot,
                )
        return VariantResolution(
            entry={},
            variant_id=forced_variant,
            missing=(f"variant {forced_variant!r} is not defined",),
            snapshot=snapshot,
        )

    candidates = [
        (index, variant, concrete, check)
        for index, (variant, concrete, check) in enumerate(variant_results)
        if check.runnable
    ]
    if not candidates:
        fallback = _fallback_variant(variant_results)
        return VariantResolution(
            entry=fallback[1] if fallback else {},
            variant_id=str(fallback[0].get("id", "")) if fallback else None,
            missing=_variant_failure_reasons(variant_results),
            snapshot=snapshot,
        )

    _, variant, concrete, check = max(
        candidates,
        key=lambda item: (int(item[1].get("priority", 0) or 0), -item[0]),
    )
    return VariantResolution(
        entry=concrete,
        variant_id=str(variant.get("id", "")) or None,
        missing=check.missing,
        warnings=check.warnings,
        preferred_backend=check.preferred_backend,
        snapshot=snapshot,
    )


def _evaluate_variants(
    entry: dict[str, Any],
    variants: list[Any],
    snapshot: RuntimeCapabilities,
) -> list[tuple[dict[str, Any], dict[str, Any], CapabilityCheck]]:
    results: list[tuple[dict[str, Any], dict[str, Any], CapabilityCheck]] = []
    for raw_variant in variants:
        if not isinstance(raw_variant, dict):
            continue
        concrete = _concrete_entry_for_variant(entry, raw_variant)
        requirement = _variant_requirement(concrete, raw_variant)
        preferred_backend, backend_warnings = _preferred_backend_for_variant(
            raw_variant,
            snapshot,
        )
        check = check_runtime_requirement(
            requirement,
            snapshot=snapshot,
            preferred_backend=preferred_backend,
        )
        check = CapabilityCheck(
            runnable=check.runnable,
            preferred_backend=check.preferred_backend,
            missing=check.missing,
            warnings=(*check.warnings, *backend_warnings),
        )
        results.append((raw_variant, concrete, check))
    return results


def _concrete_entry_for_variant(entry: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    concrete = {
        key: value
        for key, value in entry.items()
        if key not in {"variants", "backends", "requires"}
    }
    concrete.update(
        {
            key: value
            for key, value in variant.items()
            if key not in {"id", "aliases", "priority", "fallback", "requires"}
        }
    )
    return concrete


def _variant_requirement(
    concrete: dict[str, Any],
    variant: dict[str, Any],
) -> RuntimeRequirement:
    if isinstance(variant.get("requires"), dict):
        concrete = {**concrete, "runtime": {"required": variant.get("requires")}}
    return infer_model_requirements(concrete).requirement


def _preferred_backend_for_variant(
    variant: dict[str, Any],
    snapshot: RuntimeCapabilities,
) -> tuple[str | None, tuple[str, ...]]:
    backends = variant.get("backends")
    if not isinstance(backends, dict):
        return None, ()

    warnings: list[str] = []
    preferred = backends.get("preferred")
    if isinstance(preferred, list):
        for backend in preferred:
            if not isinstance(backend, dict):
                continue
            name = str(backend.get("name") or "")
            requirement = runtime_requirement_from_mapping(backend.get("requires"))
            check = check_runtime_requirement(requirement, snapshot=snapshot)
            if check.runnable:
                return name or None, ()
            if name:
                warnings.append(
                    f"preferred backend {name!r} unavailable: "
                    + "; ".join(check.missing)
                )

    fallback = backends.get("fallback")
    if isinstance(fallback, dict):
        return str(fallback.get("name") or "") or None, tuple(warnings)
    return None, tuple(warnings)


def _variant_failure_reasons(
    variant_results: list[tuple[dict[str, Any], dict[str, Any], CapabilityCheck]],
) -> tuple[str, ...]:
    if not variant_results:
        return ("no variants are defined",)
    reasons: list[str] = []
    for variant, _concrete, check in variant_results:
        variant_id = str(variant.get("id", "unknown"))
        if check.missing:
            reasons.append(f"variant {variant_id}: " + "; ".join(check.missing))
        else:
            reasons.append(f"variant {variant_id}: not runnable")
    return tuple(reasons)


def _fallback_variant(
    variant_results: list[tuple[dict[str, Any], dict[str, Any], CapabilityCheck]],
) -> tuple[dict[str, Any], dict[str, Any], CapabilityCheck] | None:
    for result in variant_results:
        if bool(result[0].get("fallback")):
            return result
    return variant_results[0] if variant_results else None


def _variant_matches(variant: dict[str, Any], forced_variant: str) -> bool:
    names = [str(variant.get("id", ""))]
    aliases = variant.get("aliases")
    if isinstance(aliases, list):
        names.extend(str(alias) for alias in aliases)
    return forced_variant in {name for name in names if name}
