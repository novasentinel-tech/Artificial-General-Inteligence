from __future__ import annotations

from .models import ActionCandidate, DecisionResult, StimulusEvent, ValenceResult
from .triggers import Trigger


class DecisionModule:
    def choose(
        self,
        event: StimulusEvent,
        valence: ValenceResult,
        triggers: list[Trigger],
        actions: list[ActionCandidate],
    ) -> DecisionResult:
        if not actions:
            return DecisionResult(
                action="observe",
                score=0.0,
                reason="No candidate action supplied.",
            )

        risk = float(event.payload.get("risk_level", 0.0))
        goal_alignment = float(event.payload.get("goal_alignment", 0.0))
        trigger_bonus = min(sum(t.priority for t in triggers) / 200.0, 1.0)

        ranked: list[tuple[float, ActionCandidate]] = []
        for action in actions:
            score = (
                action.base_utility
                + 0.4 * goal_alignment
                + 0.25 * valence.score
                + 0.2 * trigger_bonus
                - action.risk_penalty * risk
            )
            ranked.append((score, action))

        ranked.sort(key=lambda item: item[0], reverse=True)
        score, winner = ranked[0]

        return DecisionResult(
            action=winner.name,
            score=float(score),
            reason=(
                f"utility={winner.base_utility:.2f}; "
                f"goal={goal_alignment:.2f}; "
                f"valence={valence.score:.2f}; "
                f"triggers={len(triggers)}"
            ),
        )
