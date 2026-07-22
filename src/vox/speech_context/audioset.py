from __future__ import annotations

from typing import Any

from vox.speech_context.reducer import SpeechContextReductionError


def enrich_audioset_classes(
    classes: list[dict[str, Any]],
    ontology: Any,
) -> list[dict[str, Any]]:
    if not isinstance(ontology, list):
        raise SpeechContextReductionError("AudioSet ontology must be a list")

    child_ids_by_id: dict[str, tuple[str, ...]] = {}
    parents_by_id: dict[str, set[str]] = {}
    for position, item in enumerate(ontology):
        if not isinstance(item, dict):
            raise SpeechContextReductionError(f"AudioSet ontology item {position} must be an object")
        item_id = item.get("id")
        child_ids = item.get("child_ids")
        if not isinstance(item_id, str) or not item_id:
            raise SpeechContextReductionError(f"AudioSet ontology item {position} id must be non-empty")
        if item_id in child_ids_by_id:
            raise SpeechContextReductionError(f"AudioSet ontology contains duplicate id {item_id}")
        if not isinstance(child_ids, list) or not all(isinstance(child_id, str) and child_id for child_id in child_ids):
            raise SpeechContextReductionError(f"AudioSet ontology item {item_id} child_ids must be non-empty strings")
        if len(set(child_ids)) != len(child_ids):
            raise SpeechContextReductionError(f"AudioSet ontology item {item_id} has duplicate child ids")
        child_ids_by_id[item_id] = tuple(child_ids)
        parents_by_id[item_id] = set()

    for item_id, child_ids in child_ids_by_id.items():
        for child_id in child_ids:
            if child_id not in child_ids_by_id:
                raise SpeechContextReductionError(
                    f"AudioSet ontology child {child_id} referenced by {item_id} is missing"
                )
            parents_by_id[child_id].add(item_id)

    ancestors_by_id: dict[str, tuple[str, ...]] = {}
    visiting: set[str] = set()

    def collect_ancestors(item_id: str) -> tuple[str, ...]:
        cached = ancestors_by_id.get(item_id)
        if cached is not None:
            return cached
        if item_id in visiting:
            raise SpeechContextReductionError(f"AudioSet ontology contains a cycle at {item_id}")
        visiting.add(item_id)
        ordered: list[str] = []
        seen: set[str] = set()
        for parent_id in sorted(parents_by_id[item_id]):
            for ancestor_id in (*collect_ancestors(parent_id), parent_id):
                if ancestor_id not in seen:
                    seen.add(ancestor_id)
                    ordered.append(ancestor_id)
        visiting.remove(item_id)
        ancestors_by_id[item_id] = tuple(ordered)
        return ancestors_by_id[item_id]

    for item_id in child_ids_by_id:
        collect_ancestors(item_id)

    enriched: list[dict[str, Any]] = []
    for position, item in enumerate(classes):
        if not isinstance(item, dict):
            raise SpeechContextReductionError(f"audio event class {position} must be an object")
        class_id = item.get("id")
        if not isinstance(class_id, str) or not class_id:
            raise SpeechContextReductionError(f"audio event class {position} id must be non-empty")
        if class_id not in ancestors_by_id:
            raise SpeechContextReductionError(f"audio event class {class_id} is missing from the AudioSet ontology")
        enriched.append({**item, "ancestor_ids": list(ancestors_by_id[class_id])})
    return enriched
