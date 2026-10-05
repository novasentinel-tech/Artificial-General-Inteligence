from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class StimulusSource(str, Enum):
    EXTERNAL = "external"
    INTERNAL = "internal"
    SELF = "self"


class StimulusEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source: StimulusSource
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    intensity: float = Field(ge=0.0, le=1.0)
    context_id: str | None = None


class ValenceResult(BaseModel):
    stimulus_id: str
    score: float
    confidence: float
    components: dict[str, float]


class PatternRecord(BaseModel):
    stimulus_type: str
    feature_vector: list[float]
    frequency: int = 1
    average_valence: float = 0.0


class ActionCandidate(BaseModel):
    name: str
    base_utility: float = 0.0
    risk_penalty: float = 0.0


class DecisionResult(BaseModel):
    action: str
    score: float
    reason: str
