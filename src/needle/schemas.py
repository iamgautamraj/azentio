from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    txn_id: str
    card_id: str
    ts: datetime
    amount: float = Field(..., gt=0)
    merchant_id: str
    mcc: int
    channel: Literal["POS", "ECOM", "ATM"]
    country: str
    device_id: str | None = None


class ScoreResponse(BaseModel):
    txn_id: str
    fraud_score: float
    decision: Literal["ALLOW", "REVIEW"] = "ALLOW"
    model_version: str
    features: dict[str, float]


class Claim(BaseModel):
    feature: str
    value: float | int | str
    direction: Literal["raises_risk", "lowers_risk"]


class EvidenceItem(BaseModel):
    feature: str
    label: str
    value: float | int | str
    contribution: float
    direction: Literal["raises_risk", "lowers_risk"]


class LLMExplanationOutput(BaseModel):
    narrative: str
    claims: list[Claim]


class FaithfulnessResult(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)


class AlertExplanationResponse(BaseModel):
    txn_id: str
    narrative: str
    claims: list[Claim]
    evidence: list[EvidenceItem]
    explanation_source: Literal["llm", "template"]
    faithfulness: FaithfulnessResult
