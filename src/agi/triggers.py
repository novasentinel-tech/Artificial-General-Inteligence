from __future__ import annotations

from dataclasses import dataclass

from .models import StimulusEvent, ValenceResult


@dataclass(frozen=True)
class Trigger:
    name: str
    priority: int
    reason: str


class TriggerSystem:
    def evaluate(self, event: StimulusEvent, valence: ValenceResult) -> list[Trigger]:
        fired: list[Trigger] = []

        risk = float(event.payload.get("risk_level", 0.0))
        urgency = float(event.payload.get("urgency", 0.0))

        if risk >= 0.8:
            fired.append(Trigger("high_risk", 100, f"risk={risk:.2f}"))

        if urgency >= 0.8:
            fired.append(Trigger("high_urgency", 80, f"urgency={urgency:.2f}"))

        if valence.score <= -0.65:
            fired.append(Trigger("strong_negative_valence", 70, f"valence={valence.score:.2f}"))

        return sorted(fired, key=lambda t: t.priority, reverse=True)
