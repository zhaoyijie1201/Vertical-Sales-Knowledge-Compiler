"""Data model: the closed action set, scenarios, knowledge items and the recommendation.

The action set is the label space of the evaluation. Accuracy is exact match on `Action`,
so no LLM judge is needed.
"""
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Action(str, Enum):
    qualify_budget_authority = "qualify_budget_authority"
    request_spec_review = "request_spec_review"
    propose_pilot_order = "propose_pilot_order"
    send_case_study = "send_case_study"
    schedule_site_visit = "schedule_site_visit"
    escalate_pricing = "escalate_pricing"
    nurture = "nurture"
    disqualify = "disqualify"


# DRAFT definitions. Freeze them once the 40 gold labels are written.
# Each entry: (when to choose it, how it differs from its nearest neighbour).
ACTION_DEFINITIONS: Dict[Action, Dict[str, str]] = {
    Action.qualify_budget_authority: {
        "when": "There is an interest signal but budget, decision maker or purchase timing is unclear.",
        "boundary": "Unlike nurture, the customer shows a near-term intent to buy.",
    },
    Action.request_spec_review: {
        "when": "The customer has concrete technical requirements and the product fit is not yet confirmed.",
        "boundary": "Unlike propose_pilot_order, the specification match is still open.",
    },
    Action.propose_pilot_order: {
        "when": "Specifications are confirmed, there is an incumbent supplier and a motive to switch, "
                "but the customer is worried about switching risk.",
        "boundary": "Unlike schedule_site_visit, the concern is product performance, not supplier qualification.",
    },
    Action.send_case_study: {
        "when": "The customer doubts the supplier's experience or reliability in comparable industries.",
        "boundary": "Unlike propose_pilot_order, trust has to be established before any trial.",
    },
    Action.schedule_site_visit: {
        "when": "A medium or large customer requires a supplier qualification audit, "
                "or the application environment has to be assessed on site.",
        "boundary": "Unlike request_spec_review, a document review is not enough.",
    },
    Action.escalate_pricing: {
        "when": "There is an explicit price objection and the required discount exceeds the seller's authority.",
        "boundary": "Unlike propose_pilot_order, price is the only remaining barrier.",
    },
    Action.nurture: {
        "when": "The fit is good but there is no purchase plan in the near term, or a contract has not expired.",
        "boundary": "Unlike disqualify, there is a future opportunity.",
    },
    Action.disqualify: {
        "when": "A hard requirement cannot be met: missing certification, fundamental specification "
                "mismatch, or volume far below the minimum order quantity.",
        "boundary": "Unlike nurture, time will not resolve the mismatch.",
    },
}


class SalesStage(str, Enum):
    prospecting = "prospecting"
    qualifying = "qualifying"
    technical_review = "technical_review"
    proposal = "proposal"
    negotiation = "negotiation"


CustomerSize = Literal["small", "medium", "large"]
KnowledgeType = Literal["playbook", "product", "sop", "outcome"]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    narrative: str = Field(min_length=1)
    sales_stage: SalesStage
    customer_size: CustomerSize
    export_oriented: bool
    has_incumbent: bool
    pain_points: List[str] = Field(default_factory=list)
    objections: List[str] = Field(default_factory=list)
    label: Optional[Action] = None
    label_rationale: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    def without_label(self) -> "Scenario":
        """Copy that is safe to hand to a system under test."""
        return self.model_copy(update={"label": None, "label_rationale": None, "meta": {}})

    @property
    def ambiguous(self) -> bool:
        return bool(self.meta.get("ambiguous", False))


class KnowledgeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: KnowledgeType
    text: str = Field(min_length=1)
    source: str
    version: str


class NBARecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    rationale: str
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
