"""Metrics computed from prediction rows. Pure functions, no model calls.

A prediction row is a dict with at least:
  system, scenario_id, gold, action, correct, confidence, evidence_ids, retrieved_ids,
  top_score, error, input_tokens, output_tokens, latency_s, sales_stage, customer_size, ambiguous

A row whose system failed to produce a recommendation has action=None and correct=False.
It stays in every denominator.
"""
import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .gate import HUMAN_REVIEW, PASS, gate

Row = Dict[str, Any]


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def accuracy(rows: Sequence[Row]) -> Tuple[int, int]:
    return sum(1 for r in rows if r["correct"]), len(rows)


def majority_baseline(golds: Sequence[str]) -> Tuple[Optional[str], int, int]:
    """Score obtained by always answering the most common gold label."""
    if not golds:
        return None, 0, 0
    label, k = sorted(Counter(golds).items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return label, k, len(golds)


def gate_row(r: Row, tau_conf: float, tau_ret: float) -> Tuple[str, Optional[str]]:
    return gate(
        has_recommendation=r["action"] is not None,
        confidence=r["confidence"],
        evidence_ids=r.get("evidence_ids") or [],
        retrieved_ids=r.get("retrieved_ids") or [],
        top_score=r.get("top_score"),
        tau_conf=tau_conf,
        tau_ret=tau_ret,
    )


def abstention_summary(rows: Sequence[Row], tau_conf: float, tau_ret: float) -> Dict[str, Any]:
    """The two numbers the course asks for, plus the accuracy on what was answered.

    abstain_rate            : share of cases escalated to human review
    abstained_would_be_wrong: among escalated cases, share that would have been answered wrongly
    """
    n = len(rows)
    abstained: List[Row] = []
    answered: List[Row] = []
    reasons: Counter = Counter()
    for r in rows:
        decision, reason = gate_row(r, tau_conf, tau_ret)
        if decision == PASS:
            answered.append(r)
        else:
            abstained.append(r)
            reasons[reason] += 1
    wrong_in_abstained = sum(1 for r in abstained if not r["correct"])
    ans_k, ans_n = accuracy(answered)
    return {
        "n": n,
        "tau_conf": tau_conf,
        "tau_ret": tau_ret,
        "abstained": len(abstained),
        "abstain_rate": (len(abstained) / n) if n else 0.0,
        "abstained_wrong": wrong_in_abstained,
        "abstained_would_be_wrong": (wrong_in_abstained / len(abstained)) if abstained else None,
        "answered": ans_n,
        "answered_correct": ans_k,
        "answered_accuracy": (ans_k / ans_n) if ans_n else None,
        "reasons": dict(reasons),
    }


def slice_accuracy(rows: Sequence[Row], key: str) -> Dict[str, Tuple[int, int]]:
    groups: Dict[str, List[Row]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key)), []).append(r)
    return {g: accuracy(rs) for g, rs in sorted(groups.items())}


def confusion(rows: Sequence[Row]) -> Dict[Tuple[str, str], int]:
    c: Counter = Counter()
    for r in rows:
        c[(r["gold"], r["action"] if r["action"] is not None else "<error>")] += 1
    return dict(c)


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return (sum(vals) / len(vals)) if vals else None


def cost_per_scenario(rows: Sequence[Row], price_in: Optional[float],
                      price_out: Optional[float]) -> Optional[float]:
    """Mean cost in the price currency, from logged token usage. None when prices are not set."""
    if price_in is None or price_out is None or not rows:
        return None
    total = sum(r["input_tokens"] * price_in + r["output_tokens"] * price_out for r in rows) / 1e6
    return total / len(rows)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    vs = sorted(values)
    idx = min(len(vs) - 1, max(0, int(round(q * (len(vs) - 1)))))
    return vs[idx]
