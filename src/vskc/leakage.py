"""Leakage detection: does the input already contain the answer?

Two checks:
1. Scenario narratives that name an action or give advice. Matching sentences are removed
   to produce the "stripped" variant; scores are reported before and after.
2. Knowledge items that closely paraphrase a gold rationale, which would let retrieval hand
   the answer to the model.
"""
import re
from typing import Dict, List, Sequence, Tuple

from .retriever import tokenize
from .schema import Action, KnowledgeItem, Scenario

PLACEHOLDER = "[all sentences removed by the leakage filter]"

LEAK_TERMS: Dict[str, List[str]] = {
    Action.qualify_budget_authority.value: [
        r"qualif(y|ying|ication)\b[^.]{0,40}\b(budget|authority)", r"\bBANT\b"],
    Action.request_spec_review.value: [
        r"spec(ification)?s? review", r"review (of )?(the |their )?spec(ification)?s?\b"],
    Action.propose_pilot_order.value: [
        r"\bpilot\b", r"trial (order|batch|run|shipment)", r"sample order"],
    Action.send_case_study.value: [
        r"case stud(y|ies)", r"success stor(y|ies)", r"reference customers?"],
    Action.schedule_site_visit.value: [
        r"(site|factory|plant|on[- ]?site) visit", r"visit (the|their|our) (site|plant|factory)"],
    Action.escalate_pricing.value: [
        r"\bescalat\w*", r"(special )?pricing approval"],
    Action.nurture.value: [
        r"\bnurtur\w*", r"(keep|stay) in touch", r"check back (in|next)"],
    Action.disqualify.value: [
        r"\bdisqualif\w*", r"walk away", r"(not|stop) pursu\w+", r"drop (the|this) (lead|opportunity|account)"],
    "action_name": [r"\b(%s)\b" % "|".join(a.value for a in Action)],
    "advice": [
        r"\b(best|recommended|right|logical|obvious) next (step|move|action)\b",
        r"\bnext[- ]best\b", r"\b(we|they|you|sales) should\b", r"\bit would be best to\b"],
}

_COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in LEAK_TERMS.items()}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def find_leaks(sentence: str) -> List[Tuple[str, str]]:
    hits = []
    for category, patterns in _COMPILED.items():
        for rx in patterns:
            m = rx.search(sentence)
            if m:
                hits.append((category, m.group(0)))
                break
    return hits


def strip_narrative(narrative: str) -> Tuple[str, List[Dict[str, str]]]:
    """Remove every sentence that matches a leak pattern. Returns (stripped text, hits)."""
    kept, hits = [], []
    for sentence in _SENTENCE_SPLIT.split(narrative.strip()):
        found = find_leaks(sentence)
        if found:
            for category, match in found:
                hits.append({"category": category, "match": match, "sentence": sentence})
        else:
            kept.append(sentence)
    stripped = " ".join(kept).strip() or PLACEHOLDER
    return stripped, hits


def strip_scenario(s: Scenario) -> Tuple[Scenario, List[Dict[str, str]]]:
    stripped, hits = strip_narrative(s.narrative)
    meta = dict(s.meta)
    meta["leakage_hits"] = len(hits)
    return s.model_copy(update={"narrative": stripped, "meta": meta}), hits


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def knowledge_rationale_overlap(
    knowledge: Sequence[KnowledgeItem], scenarios: Sequence[Scenario], threshold: float = 0.5
) -> List[Dict[str, object]]:
    """Knowledge items whose wording overlaps heavily with a gold rationale."""
    flagged = []
    for s in scenarios:
        if not s.label_rationale:
            continue
        for k in knowledge:
            j = jaccard(k.text, s.label_rationale)
            if j >= threshold:
                flagged.append({"scenario_id": s.id, "knowledge_id": k.id, "jaccard": round(j, 3)})
    return sorted(flagged, key=lambda x: -float(x["jaccard"]))
