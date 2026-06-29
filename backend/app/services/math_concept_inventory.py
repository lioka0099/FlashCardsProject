"""
Math concept inventory.

Before authoring calculation problems we extract, grounded in the document
excerpts, *what the document actually teaches*: concepts, formulas (with a
source quote), solution methods/techniques, and the problem archetypes the
document demonstrates. Problems are then built from these concepts rather than
by copying surface examples, which is what makes the calculation cards deep and
diverse across any math domain (algebra, linear algebra, probability,
statistics, discrete, geometry, optimization, calculus).

The extraction is LLM-based but domain-agnostic. Results are cached on the
topic's ``info`` (same pattern as ``route_candidate``) so we extract once per
topic. All parsing/merging helpers are pure and unit-tested without a live LLM.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.services.llm import CHAT_MODEL, chat_completions_create


INVENTORY_INFO_KEY = "math_concept_inventory"
_INVENTORY_VERSION = 1


@dataclass
class ConceptInventory:
    concepts: List[Dict[str, str]] = field(default_factory=list)
    formulas: List[Dict[str, str]] = field(default_factory=list)
    methods: List[Dict[str, str]] = field(default_factory=list)
    archetypes: List[Dict[str, str]] = field(default_factory=list)
    source: str = "llm"
    version: int = _INVENTORY_VERSION

    def is_empty(self) -> bool:
        return not (self.concepts or self.formulas or self.methods or self.archetypes)

    def to_info(self) -> Dict[str, Any]:
        return asdict(self)

    def as_prompt(self) -> str:
        """Render the inventory as grounding material for the author prompt."""
        def _block(title: str, items: List[Dict[str, str]], keys: List[str]) -> str:
            if not items:
                return f"{title}: (none extracted)"
            lines = [f"{title}:"]
            for item in items:
                parts = [str(item.get(k, "")).strip() for k in keys]
                parts = [p for p in parts if p]
                lines.append("- " + " — ".join(parts))
            return "\n".join(lines)

        return "\n\n".join(
            [
                _block("CONCEPTS", self.concepts, ["name", "summary"]),
                _block("FORMULAS", self.formulas, ["name", "expression", "source_quote"]),
                _block("METHODS/TECHNIQUES", self.methods, ["name", "when_to_use"]),
                _block("PROBLEM ARCHETYPES", self.archetypes, ["name", "description"]),
            ]
        )


def _coerce_items(value: Any, keys: List[str], *, limit: int) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for raw in value:
        if isinstance(raw, dict):
            item = {k: str(raw.get(k, "")).strip() for k in keys}
        else:
            item = {keys[0]: str(raw).strip()}
            for k in keys[1:]:
                item[k] = ""
        name = item.get(keys[0], "")
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(item)
        if len(out) >= limit:
            break
    return out


def parse_inventory_payload(payload: Optional[Dict[str, Any]]) -> ConceptInventory:
    """Pure parser: coerce a raw LLM/cached payload into a ConceptInventory."""
    raw = payload or {}
    if not isinstance(raw, dict):
        return ConceptInventory(source="empty")
    return ConceptInventory(
        concepts=_coerce_items(raw.get("concepts"), ["name", "summary"], limit=12),
        formulas=_coerce_items(raw.get("formulas"), ["name", "expression", "source_quote"], limit=12),
        methods=_coerce_items(raw.get("methods"), ["name", "when_to_use"], limit=12),
        archetypes=_coerce_items(raw.get("archetypes"), ["name", "description"], limit=10),
        source=str(raw.get("source") or "llm"),
        version=int(raw.get("version") or _INVENTORY_VERSION),
    )


def _safe_json_load(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class MathConceptInventoryService:
    """Extracts (and caches) the math concept inventory for a topic."""

    def __init__(self, *, model: str = CHAT_MODEL) -> None:
        self.model = model

    def get_or_build(
        self,
        *,
        topic_label: str,
        context_pack: str,
        cached: Optional[Dict[str, Any]] = None,
    ) -> ConceptInventory:
        if isinstance(cached, dict) and cached:
            inventory = parse_inventory_payload(cached)
            if not inventory.is_empty() and int(inventory.version) >= _INVENTORY_VERSION:
                return inventory
        return self.build(topic_label=topic_label, context_pack=context_pack)

    def build(self, *, topic_label: str, context_pack: str) -> ConceptInventory:
        sys_prompt = (
            "You extract the teachable mathematical content of a document section.\n"
            "Work for ANY area of mathematics (algebra, linear algebra, probability, "
            "statistics, discrete math, geometry, optimization, calculus, etc.).\n"
            "Only report what the excerpts actually teach. Return JSON only."
        )
        user_prompt = (
            f"TOPIC: {topic_label}\n\n"
            f"EXCERPTS:\n{context_pack}\n\n"
            "Extract what this section teaches that a student could be asked to CALCULATE with.\n"
            "Return JSON with this schema:\n"
            "{\n"
            '  "concepts": [{"name": "...", "summary": "one sentence"}],\n'
            '  "formulas": [{"name": "...", "expression": "as written, SymPy-style if possible", "source_quote": "short quote from excerpts"}],\n'
            '  "methods": [{"name": "...", "when_to_use": "one sentence"}],\n'
            '  "archetypes": [{"name": "short-kebab-label", "description": "a type of calculation problem this section supports"}]\n'
            "}\n"
            "Rules:\n"
            "- Ground every item in the excerpts; do not invent topics not present.\n"
            "- Prefer methods/techniques and problem archetypes over restating definitions.\n"
            "- If the section is not calculational, return empty arrays."
        )
        try:
            resp = chat_completions_create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=900,
                response_format={"type": "json_object"},
            )
            payload = _safe_json_load(resp.choices[0].message.content or "{}")
        except Exception:
            return ConceptInventory(source="error")
        payload["source"] = "llm"
        payload["version"] = _INVENTORY_VERSION
        return parse_inventory_payload(payload)
