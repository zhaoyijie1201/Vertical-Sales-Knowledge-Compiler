"""The three systems under comparison. All take a Scenario and return a SystemOutput.

rule : deterministic mapping from structured fields only (the non-AI baseline)
llm  : the model with the shared prompt and an empty <knowledge> block
rag  : the model with the shared prompt and the top-k retrieved knowledge items
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import Settings
from .llm import complete_json
from .prompts import NBA_JSON_SCHEMA, PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from .retriever import BM25Retriever, scenario_query
from .schema import Action, NBARecommendation, SalesStage, Scenario

SYSTEM_NAMES = ("rule", "llm", "rag")


@dataclass
class Context:
    settings: Settings
    retriever: Optional[BM25Retriever]
    run_id: str
    raw_dir: Path


@dataclass
class SystemOutput:
    system: str
    scenario_id: str
    recommendation: Optional[NBARecommendation]
    retrieved_ids: List[str] = field(default_factory=list)
    top_score: Optional[float] = None
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 0


# --------------------------------------------------------------------------- rule baseline

def _mentions(phrases: Sequence[str], pattern: str) -> bool:
    rx = re.compile(pattern, re.IGNORECASE)
    return any(rx.search(p) for p in phrases)


def _late(stage: SalesStage) -> bool:
    return stage in (SalesStage.technical_review, SalesStage.proposal, SalesStage.negotiation)


Rule = Tuple[str, Callable[[Scenario], bool], Action]

# DRAFT. Ordered, first match wins. Reads structured fields only, never the narrative.
# Adjust on the dev split only.
RULES: List[Rule] = [
    ("hard_requirement_unmet",
     lambda s: _mentions(s.objections + s.pain_points,
                         r"certif|atex|spec(ification)? mismatch|below (the )?(moq|minimum)|cannot meet"),
     Action.disqualify),
    ("price_objection_late_stage",
     lambda s: s.sales_stage in (SalesStage.proposal, SalesStage.negotiation)
     and _mentions(s.objections, r"price|pricing|discount|cost|expensive|quote"),
     Action.escalate_pricing),
    ("budget_or_authority_unclear",
     lambda s: _mentions(s.objections, r"budget|authority|decision[- ]?maker|approval|funding"),
     Action.qualify_budget_authority),
    ("no_near_term_plan",
     lambda s: _mentions(s.objections, r"contract|no plan|not this year|next year|no timeline|locked in"),
     Action.nurture),
    ("supplier_audit_required",
     lambda s: s.customer_size in ("medium", "large")
     and _mentions(s.objections + s.pain_points, r"audit|qualification|on[- ]?site|supplier approval"),
     Action.schedule_site_visit),
    ("trust_concern_with_incumbent_late",
     lambda s: _late(s.sales_stage) and s.has_incumbent
     and _mentions(s.objections, r"reliab|unproven|risk|switch|field performance"),
     Action.propose_pilot_order),
    ("trust_concern",
     lambda s: _mentions(s.objections, r"reliab|unproven|experience|reference|track record|trust"),
     Action.send_case_study),
    ("prospecting_default", lambda s: s.sales_stage == SalesStage.prospecting,
     Action.qualify_budget_authority),
    ("qualifying_default", lambda s: s.sales_stage == SalesStage.qualifying,
     Action.request_spec_review),
    ("technical_review_with_incumbent",
     lambda s: s.sales_stage == SalesStage.technical_review and s.has_incumbent,
     Action.propose_pilot_order),
    ("technical_review_default", lambda s: s.sales_stage == SalesStage.technical_review,
     Action.request_spec_review),
    ("proposal_default", lambda s: s.sales_stage == SalesStage.proposal, Action.send_case_study),
    ("negotiation_default", lambda s: s.sales_stage == SalesStage.negotiation,
     Action.escalate_pricing),
    ("fallback", lambda s: True, Action.nurture),
]


def system_rule(s: Scenario, ctx: Optional[Context] = None) -> SystemOutput:
    for name, predicate, action in RULES:
        if predicate(s):
            rec = NBARecommendation(action=action, rationale="rule:%s" % name,
                                    evidence_ids=[], confidence=1.0)
            return SystemOutput(system="rule", scenario_id=s.id, recommendation=rec)
    raise AssertionError("the fallback rule always matches")


# --------------------------------------------------------------------------- model systems

def _run_model(s: Scenario, ctx: Context, with_knowledge: bool) -> SystemOutput:
    name = "rag" if with_knowledge else "llm"
    safe = s.without_label()

    hits = []
    if with_knowledge:
        if ctx.retriever is None:
            raise RuntimeError("the rag system needs a retriever")
        hits = ctx.retriever.search(scenario_query(safe), ctx.settings.top_k)
    items = [h[0] for h in hits]
    retrieved_ids = [i.id for i in items]
    top_score = float(hits[0][1]) if hits else (0.0 if with_knowledge else None)

    rec, stats, error = complete_json(
        settings=ctx.settings,
        model=ctx.settings.model_under_test,
        system=SYSTEM_PROMPT,
        user=build_user_message(safe, items),
        schema=NBA_JSON_SCHEMA,
        schema_name="nba_recommendation",
        validate=NBARecommendation.model_validate,
        run_id=ctx.run_id,
        raw_dir=ctx.raw_dir,
        tag={"kind": "system", "system": name, "scenario_id": s.id,
             "prompt_version": PROMPT_VERSION, "retrieved_ids": retrieved_ids},
    )
    return SystemOutput(
        system=name, scenario_id=s.id, recommendation=rec,
        retrieved_ids=retrieved_ids, top_score=top_score, error=error,
        input_tokens=stats["input_tokens"], output_tokens=stats["output_tokens"],
        latency_s=stats["latency_s"], attempts=stats["attempts"],
    )


def system_llm(s: Scenario, ctx: Context) -> SystemOutput:
    return _run_model(s, ctx, with_knowledge=False)


def system_rag(s: Scenario, ctx: Context) -> SystemOutput:
    return _run_model(s, ctx, with_knowledge=True)


SYSTEMS: Dict[str, Callable[[Scenario, Context], SystemOutput]] = {
    "rule": system_rule,
    "llm": system_llm,
    "rag": system_rag,
}
