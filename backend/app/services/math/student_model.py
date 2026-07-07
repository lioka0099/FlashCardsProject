"""Generate a math question adapted to the student (archetype diversity from prior fingerprints)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.services.generation.difficulty_frameworks import get_level
from app.services.llm import CHAT_MODEL, chat_completions_create, safe_json_load
from app.services.math.compound_spec import (
    math_structural_fingerprint,
    normalize_compound_spec,
)
from app.services.math.concept_inventory import ConceptInventory, MathConceptInventoryService
from app.services.math.verification import verify_compound_solution


# Machine-checkable step kinds and the fields each requires. This is the contract
# the author must follow so every computational step can be verified by SymPy.
_CHECK_KIND_GUIDE = (
    "Machine-checkable step kinds (each step's `check` must be ONE of these, with `expected` omitted — the backend computes it):\n"
    "- arithmetic/substitution: {kind, expression, substitutions?}\n"
    "- simplify/expand/factor/equivalence: {kind, expression}\n"
    "- equation: {kind:'equation', equation:'LHS = RHS', variable}\n"
    "- system: {kind:'system', equations:[...], variables:[...]}\n"
    "- derivative/integral: {kind, expression, variable, lower?, upper? (integral, for definite)}\n"
    "- limit: {kind:'limit', expression, variable, point, direction? ('+'/'-')}\n"
    "- summation: {kind:'summation', expression, variable, lower, upper}\n"
    "- matrix: {kind:'matrix', operation:'det|inverse|transpose|rank|eigenvals|multiply', matrix:[[..],[..]], matrix_b?:[[..]]}\n"
)


@dataclass
class MathQuestionResult:
    question: str
    payload: Dict[str, Any]


def _archetypes_from_fingerprints(fingerprints: List[str]) -> List[str]:
    out: List[str] = []
    for fp in fingerprints:
        for part in str(fp).split("|"):
            if part.startswith("arch=") and part[5:].strip():
                arch = part[5:].strip()
                if arch not in out:
                    out.append(arch)
    return out


@dataclass
class MathStudentModelService:
    """Authors grounded, multi-step (compound) math calculation problems.

    Problems are built from concepts extracted from the document and authored as
    a chain of steps, each carrying a SymPy-checkable assertion. Difficulty is
    conceptual depth (number of concepts/steps, indirectness, method selection),
    not larger numbers. The computational spine and final answer are verified
    deterministically before the problem is accepted.
    """

    model: str = CHAT_MODEL

    def generate_question(
        self,
        *,
        topic_label: str,
        context_pack: str,
        difficulty: int,
        memory: Optional[Dict[str, Any]] = None,
        route_metadata: Optional[Dict[str, Any]] = None,
        avoid_questions: Optional[List[str]] = None,
        avoid_fingerprints: Optional[List[str]] = None,
        concept_inventory: Optional[ConceptInventory] = None,
        attempt_no: int = 1,
    ) -> MathQuestionResult:
        level = get_level("tag", difficulty)
        depth = level.depth
        mem = memory or {}
        known = [str(x) for x in mem.get("known_facts", [])[:6]]
        misconceptions = [str(x) for x in mem.get("misconceptions", [])[:6]]
        blocked_texts = [str(x).strip() for x in (avoid_questions or []) if str(x or "").strip()][:20]
        blocked_fps = [str(x).strip() for x in (avoid_fingerprints or []) if str(x or "").strip()][:30]
        avoid_archetypes = _archetypes_from_fingerprints(blocked_fps)

        inventory = concept_inventory or MathConceptInventoryService().build(
            topic_label=topic_label, context_pack=context_pack
        )
        if inventory.is_empty():
            raise ValueError("Insufficient evidence: no calculational concepts extracted from the document.")

        retry_hint = (
            "This is a retry. Produce a STRUCTURALLY different problem (different archetype/concepts/operations), "
            "not the same template with new numbers."
            if int(attempt_no) > 1
            else "Produce a fresh problem."
        )

        sys_prompt = (
            "You are a mathematics problem author. You design rigorous, document-grounded "
            "calculation problems for flashcards across ANY area of mathematics.\n"
            "You output a structured, machine-verifiable solution plan as strict JSON.\n"
            "Return JSON only."
        )
        user_prompt = (
            f"TOPIC: {topic_label}\n\n"
            f"DIFFICULTY: TAG level {level.level} ({level.name})\n"
            f"{level.instruction}\n"
            f"DEPTH REQUIREMENTS:\n{depth.as_prompt() if depth else '(none)'}\n\n"
            "DOCUMENT CONCEPT INVENTORY (build the problem from THESE):\n"
            f"{inventory.as_prompt()}\n\n"
            f"STUDENT KNOWN FACTS:\n- " + ("\n- ".join(known) if known else "(none)") + "\n"
            f"STUDENT MISCONCEPTIONS:\n- " + ("\n- ".join(misconceptions) if misconceptions else "(none)") + "\n\n"
            f"ATTEMPT: {int(attempt_no)} — {retry_hint}\n"
            f"AVOID THESE ARCHETYPES (already used): {avoid_archetypes or '(none)'}\n"
            "AVOID PARAPHRASING THESE EXISTING QUESTIONS:\n- "
            + ("\n- ".join(blocked_texts) if blocked_texts else "(none)")
            + "\n\n"
            f"{_CHECK_KIND_GUIDE}\n"
            "Design ONE calculation problem and its step-by-step solution plan.\n"
            "Rules:\n"
            "- Ground the problem in the concepts/formulas/methods above; do not introduce outside topics.\n"
            "- Build difficulty through conceptual depth and connections — NOT larger numbers.\n"
            "- Decompose the solution into ordered steps. Every computational step MUST carry a `check`.\n"
            "- A purely explanatory step may set `check` to null, but the FINAL answer must come from a checked step.\n"
            "- Mark the step that produces the final result with `is_final: true`.\n"
            "- The final step's `check` MUST be self-contained: inline the numeric results of earlier steps "
            "(e.g. write `91/6 - (7/2)**2`, NOT `EX2 - EX**2`) so it evaluates on its own.\n"
            "- In `check` expressions and `final_answer`, use Python/SymPy syntax "
            "(** for powers, * for multiplication, sqrt(x) for roots); NO LaTeX in those machine-verified fields.\n"
            "- In `problem_statement` (the text the student reads), format all math with inline "
            "LaTeX \\( ... \\) — e.g. \\(f(x) = x^3 - 3x^2 + 4\\) — never bare ASCII like x^3.\n"
            "- Keep numbers small and clean; depth must come from the mathematics, not arithmetic size.\n"
            "- Return status='insufficient_evidence' with a reason if the document does not support a grounded problem.\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            '  "status": "ready|insufficient_evidence",\n'
            '  "reason": "short reason when insufficient_evidence",\n'
            '  "problem_statement": "the full natural-language problem the student reads (math in inline LaTeX \\\\( ... \\\\))",\n'
            '  "concepts_used": ["concept name", "..."],\n'
            '  "archetype": "short-kebab-label for the problem type",\n'
            '  "source_rule": "rule/formula/method from the inventory this problem rests on",\n'
            '  "final_answer": "the final answer (SymPy-style)",\n'
            '  "steps": [\n'
            '    {"description": "what this step does and why", "check": {"kind": "...", "...": "..."}, "is_final": false}\n'
            "  ]\n"
            "}"
        )

        temperature = 0.4 if int(attempt_no) <= 1 else min(0.85, 0.4 + 0.15 * (int(attempt_no) - 1))
        resp = chat_completions_create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1600,
            response_format={"type": "json_object"},
        )
        payload = safe_json_load(resp.choices[0].message.content or "{}")

        # source_rule is descriptive metadata, not a correctness signal — don't
        # reject an otherwise-solvable problem just because the author omitted it.
        normalized = normalize_compound_spec(payload, require_source_rule=False)
        if not normalized.ok:
            reason = normalized.reason or "Compound spec was not usable."
            if normalized.status == "insufficient_evidence":
                raise ValueError(f"Insufficient evidence: {reason}")
            raise ValueError(f"{normalized.status}: {reason}")

        spec = normalized.spec
        # Deterministic safeguard: every checkable step must solve, and the solver
        # is authoritative for the final answer. We do NOT gate on the LLM's own
        # stated answer here (declared_final_answer="" skips that check) — its
        # arithmetic is advisory and we overwrite it with the CAS-canonical value.
        # Re-verification of the *displayed* answer still happens downstream.
        verification = verify_compound_solution(spec, declared_final_answer="")
        if verification.status != "verified":
            raise ValueError(
                f"Math compound spec could not solve generated question: {verification.details.get('reason', verification.status)}"
            )
        canonical = verification.details.get("canonical_final_answer")
        if canonical:
            spec["final_answer"] = canonical
            spec["expected_final_answer"] = canonical

        # Expose the final step's target for the downstream verifier/teacher.
        final_step = next((s for s in spec["steps"] if s.get("is_final") and s.get("check")), None)
        if final_step and isinstance(final_step.get("check"), dict):
            check = final_step["check"]
            spec["verification_target"] = check.get("verification_target") or check
            spec["problem_type"] = spec["verification_target"].get("kind")

        question = spec["problem_statement"]
        spec["question"] = question
        spec["compound"] = True
        spec["fingerprint"] = math_structural_fingerprint(spec)
        spec["compound_verification"] = verification.to_info()
        spec.setdefault("tag_level", level.level)
        spec.setdefault("tag_level_name", level.name)
        return MathQuestionResult(question=question, payload=spec)
