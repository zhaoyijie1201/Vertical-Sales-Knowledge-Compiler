"""Deterministic confidence gate. Decides whether a recommendation is shown or escalated.

Thresholds are chosen on the dev split (scripts/tune_gate.py) and only reported on held-out.
"""
from typing import Optional, Sequence, Tuple

PASS = "pass"
HUMAN_REVIEW = "human_review"

REASON_SYSTEM_ERROR = "system_error"
REASON_UNKNOWN_EVIDENCE = "cited_unknown_evidence"
REASON_NOT_COVERED = "knowledge_not_covering"
REASON_LOW_CONFIDENCE = "low_confidence"

REASONS = (REASON_SYSTEM_ERROR, REASON_UNKNOWN_EVIDENCE, REASON_NOT_COVERED, REASON_LOW_CONFIDENCE)


def gate(
    *,
    has_recommendation: bool,
    confidence: Optional[float],
    evidence_ids: Sequence[str],
    retrieved_ids: Sequence[str],
    top_score: Optional[float],
    tau_conf: float,
    tau_ret: float,
) -> Tuple[str, Optional[str]]:
    """Return (decision, reason). `top_score` of None means the system does no retrieval."""
    if not has_recommendation or confidence is None:
        return HUMAN_REVIEW, REASON_SYSTEM_ERROR
    if not set(evidence_ids) <= set(retrieved_ids):
        return HUMAN_REVIEW, REASON_UNKNOWN_EVIDENCE
    if top_score is not None and top_score < tau_ret:
        return HUMAN_REVIEW, REASON_NOT_COVERED
    if confidence < tau_conf:
        return HUMAN_REVIEW, REASON_LOW_CONFIDENCE
    return PASS, None
