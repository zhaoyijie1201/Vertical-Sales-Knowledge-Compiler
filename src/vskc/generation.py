"""Synthetic data generation: the sampling grid and the generation prompts.

The grid never contains a target action. Generated scenarios are unlabeled; a person
assigns the gold label afterwards. This keeps the model that writes the scenarios from
also writing the answers.
"""
import random
from typing import Any, Dict, List

GRID: Dict[str, List[Any]] = {
    "sales_stage": ["prospecting", "qualifying", "technical_review", "proposal", "negotiation"],
    "customer_size": ["small", "medium", "large"],
    "export_oriented": [True, False],
    "has_incumbent": [True, False],
    "objection_type": ["price", "reliability", "budget", "certification", "lead_time",
                       "supplier_audit", "none"],
    "spec_status": ["unknown", "confirmed", "partial_mismatch"],
    "purchase_timing": ["immediate", "within_year", "no_plan"],
    "length": ["short", "medium", "long"],
}

LENGTH_WORDS = {"short": "60 to 100", "medium": "100 to 160", "long": "160 to 220"}
AMBIGUOUS_SHARE = 0.15


def sample_grid(n: int, seed: int) -> List[Dict[str, Any]]:
    """n grid points with near-uniform marginals on every factor, reproducible from the seed."""
    rng = random.Random(seed)
    columns: Dict[str, List[Any]] = {}
    for factor, values in GRID.items():
        col = [values[i % len(values)] for i in range(n)]
        rng.shuffle(col)
        columns[factor] = col
    n_amb = int(round(n * AMBIGUOUS_SHARE))
    amb = [True] * n_amb + [False] * (n - n_amb)
    rng.shuffle(amb)
    points = []
    for i in range(n):
        p = {factor: columns[factor][i] for factor in GRID}
        p["ambiguous"] = amb[i]
        points.append(p)
    return points


SCENARIO_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["narrative", "pain_points", "objections"],
    "properties": {
        "narrative": {"type": "string"},
        "pain_points": {"type": "array", "items": {"type": "string"}},
        "objections": {"type": "array", "items": {"type": "string"}},
    },
}

_COMMON_RULES = """Rules:
- The supplier sells industrial sensors (pressure, temperature, displacement, flow, proximity).
- State facts only. Do not name, hint at, or recommend any next step for the sales team.
- Include one or two details that are irrelevant to the decision.
- Use invented company descriptions. No real company names, no real product model numbers.
- pain_points and objections are short phrases taken from the narrative, zero to three each.
- The narrative must agree with every attribute given below.
Return only the JSON object."""

TEMPLATES: Dict[str, str] = {
    "A": "You write synthetic B2B sales scenarios for evaluating a decision-support system.\n"
         "Style: a neutral third-person account record, as a CRM summary would read.\n"
         + _COMMON_RULES,
    "B": "You write synthetic B2B sales scenarios for evaluating a decision-support system.\n"
         "Style: informal first-person notes a salesperson typed after a call or a meeting, "
         "with uneven detail and the occasional abbreviation.\n"
         + _COMMON_RULES,
}

_AMBIGUOUS_NOTE = ("Make this case ambiguous: include two statements that conflict with each other, "
                   "or leave out a fact that would be needed to decide.")


def scenario_user_prompt(point: Dict[str, Any]) -> str:
    lines = ["Write one scenario with these attributes:"]
    for k in ("sales_stage", "customer_size", "export_oriented", "has_incumbent",
              "objection_type", "spec_status", "purchase_timing"):
        lines.append("- %s: %s" % (k, point[k]))
    lines.append("- length: %s words" % LENGTH_WORDS[point["length"]])
    if point.get("ambiguous"):
        lines.append(_AMBIGUOUS_NOTE)
    return "\n".join(lines)


TEXT_ITEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {"text": {"type": "string"}},
}

SOP_SYSTEM = """You write synthetic procurement procedures for invented industrial customers.
Each one describes, in 60 to 120 words, how that customer buys components: approval levels,
sample testing requirements, payment terms, supplier qualification steps.
State the procedure only. Do not advise the seller on what to do.
No real company names. Return only the JSON object."""

OUTCOME_SYSTEM = """You write synthetic records of past B2B sales opportunities for an industrial
sensor supplier. Each record, in 50 to 100 words, states the customer situation, the result
(won, lost or stalled) and the main reason for that result.
Describe what happened. Do not phrase it as a rule or a recommendation.
No real company names. Return only the JSON object."""

OUTCOME_RESULTS = ["won", "lost", "stalled"]


def outcome_user_prompt(point: Dict[str, Any], result: str) -> str:
    lines = ["Write one record with these attributes:"]
    for k in ("sales_stage", "customer_size", "export_oriented", "has_incumbent",
              "objection_type", "purchase_timing"):
        lines.append("- %s: %s" % (k, point[k]))
    lines.append("- result: %s" % result)
    return "\n".join(lines)


def sop_user_prompt(point: Dict[str, Any]) -> str:
    return ("Write one procurement procedure for a %s, %s customer.%s"
            % (point["customer_size"],
               "export-oriented" if point["export_oriented"] else "domestic-market",
               " They have a formal supplier audit." if point["objection_type"] == "supplier_audit" else ""))
