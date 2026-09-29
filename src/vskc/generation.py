"""Synthetic data generation: the sampling grid and the generation prompts.

The grid never contains a target action. Generated scenarios are unlabeled; a person
assigns the gold label afterwards. This keeps the model that writes the scenarios from
also writing the answers.
"""
import math
import random
from typing import Any, Dict, List

GRID: Dict[str, List[Any]] = {
    "sales_stage": ["prospecting", "qualifying", "technical_review", "proposal", "negotiation"],
    "customer_size": ["small", "medium", "large"],
    "export_oriented": [True, False],
    "has_incumbent": [True, False],
    "objection_type": ["price", "product_reliability", "budget", "industry_experience",
                       "lead_time", "supplier_audit", "none"],
    "spec_status": ["not_yet_checked", "confirmed_by_customer", "open_question"],
    "purchase_timing": ["immediate", "within_year", "no_plan"],
    "length": ["short", "medium", "long"],
}

LENGTH_WORDS = {"short": "60 to 100", "medium": "100 to 160", "long": "160 to 220"}
AMBIGUOUS_SHARE = 0.15


# --------------------------------------------------------------------------- supplier-fact hooks
#
# A hook is a customer-side requirement whose consequence depends on a fact that only the
# knowledge base states. Each hook has two sides: a value the supplier can meet ("within")
# and one it cannot ("beyond"). The generator is given the customer's requirement only. It
# is never told what the supplier can do, so the narrative cannot leak that fact.
HOOKS: List[Dict[str, Any]] = [
    {"id": "temperature", "fact_refs": ["pd-003"],
     "within": ["the sensor head has to run continuously at {v} degrees Celsius", [90, 120, 150, 170]],
     "beyond": ["the sensor head has to run continuously at {v} degrees Celsius", [220, 260, 350, 480]]},
    {"id": "viscosity", "fact_refs": ["pd-004"],
     "within": ["the process fluid measured by the flow sensor has a viscosity of {v} millipascal-seconds",
                [200, 800, 1500]],
     "beyond": ["the process fluid measured by the flow sensor has a viscosity of {v} millipascal-seconds",
                [3500, 6000, 15000]]},
    {"id": "interface", "fact_refs": ["pd-009"],
     "within": ["the machine controller accepts linear position sensors with {v}", ["IO-Link", "SSI"]],
     "beyond": ["the machine controller accepts linear position sensors only with {v}",
                ["the controller maker's own proprietary protocol"]]},
    {"id": "rotary_position", "fact_refs": ["pd-008"],
     "within": ["the machine runs a homing movement after every power-up, so {v} from the rotary sensor "
                "are sufficient", ["incremental signals"]],
     "beyond": ["the rotary sensor has to report {v} immediately after a power loss, and a homing "
                "movement is not allowed", ["the absolute angular position"]]},
    {"id": "hygienic_certificate", "fact_refs": ["pd-011", "pd-017"],
     "within": ["the end customer asks for {v} for the level sensor's wetted parts",
                ["a manufacturer's declaration of materials"]],
     "beyond": ["the end customer requires {v} for the level sensor's wetted parts",
                ["an independent third-party hygienic design certificate"]]},
    {"id": "hazardous_area", "fact_refs": ["pd-017"],
     "within": ["the sensors are installed {v}", ["outside any hazardous area, with CE marking required"]],
     "beyond": ["the sensors are installed {v}",
                ["inside a hazardous area where ATEX-certified components are mandatory"]]},
    {"id": "custom_housing_quantity", "fact_refs": ["pb-002", "pd-010"],
     "within": ["the displacement sensor needs a housing shape that is not in the catalogue, and the "
                "customer expects to buy {v} units per year", [400, 900, 2500]],
     "beyond": ["the displacement sensor needs a housing shape that is not in the catalogue, and the "
                "customer will need {v} units in total", [6, 40, 120]]},
    {"id": "price_reduction", "fact_refs": ["pb-001"],
     "beyond": ["all other terms are agreed and the buyer requires a price reduction of {v} percent",
                [8, 11, 15]]},
]

FLEXIBILITY = {
    "fixed": "The customer states that this requirement cannot be changed.",
    "open": "The customer states that this requirement could be revised if there is a good reason.",
}
FIXED_SHARE = 0.7


def hook_plan(n_hooks: int, rng: random.Random) -> List[Dict[str, Any]]:
    """n_hooks hook assignments, cycling through every hook and side so that each is used."""
    combos = [(h, side) for h in HOOKS for side in ("within", "beyond") if side in h]
    rng.shuffle(combos)
    plan = []
    for i in range(n_hooks):
        h, side = combos[i % len(combos)]
        template, values = h[side]
        value = values[rng.randrange(len(values))]
        plan.append({
            "hook": h["id"], "side": side, "fact_refs": list(h["fact_refs"]),
            "requirement": template.format(v=value), "value": value,
            "flexibility": "fixed" if rng.random() < FIXED_SHARE else "open",
        })
    return plan


def sample_plan(n: int, seed: int, hook_share: float = 0.0) -> List[Dict[str, Any]]:
    """Grid points for n scenarios, of which ceil(n * hook_share) carry a supplier-fact hook."""
    points = sample_grid(n, seed)
    n_hooks = int(math.ceil(n * hook_share - 1e-9)) if hook_share > 0 else 0
    rng = random.Random(seed + 1)
    chosen = sorted(rng.sample(range(n), n_hooks))
    for idx, hook in zip(chosen, hook_plan(n_hooks, rng)):
        p = points[idx]
        p["hook"] = hook
        p["spec_status"] = "given_by_requirement"
        p["objection_type"] = "price" if hook["hook"] == "price_reduction" else "none"
        if hook["hook"] == "price_reduction":
            p["sales_stage"] = "negotiation" if idx % 2 else "proposal"
    for p in points:
        p["depends_on_supplier_fact"] = "hook" in p
    return points


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
- Write only what the customer said, did or documented. Do not state what the supplier's
  products can or cannot do, and do not state the supplier's discount limits, minimum
  quantities or certifications.
- Do not say which details are relevant and which are not.
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
    hook = point.get("hook")
    if hook:
        lines.append("The customer's requirement, to be stated as a fact in the narrative: %s. %s"
                     % (hook["requirement"], FLEXIBILITY[hook["flexibility"]]))
        lines.append("Do not comment on whether the supplier can meet this requirement.")
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
