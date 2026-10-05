from __future__ import annotations

from pprint import pprint

from .engine import CognitiveEngine
from .models import ActionCandidate, StimulusEvent, StimulusSource


def main() -> None:
    engine = CognitiveEngine()

    actions = [
        ActionCandidate(name="observe", base_utility=0.20, risk_penalty=0.0),
        ActionCandidate(name="investigate", base_utility=0.55, risk_penalty=0.15),
        ActionCandidate(name="avoid", base_utility=0.35, risk_penalty=-0.35),
    ]

    event = StimulusEvent(
        source=StimulusSource.EXTERNAL,
        type="environment_event",
        intensity=0.9,
        payload={
            "urgency": 0.85,
            "goal_alignment": -0.2,
            "risk_level": 0.9,
        },
    )

    first = engine.process(event, actions)
    print("FIRST EXPERIENCE")
    pprint(first)

    # Simulate a negative real-world consequence.
    engine.learn_from_outcome(event, first.novelty, outcome=-0.9)

    second = engine.process(event, actions)
    print("\nSECOND EXPERIENCE AFTER LEARNING")
    pprint(second)


if __name__ == "__main__":
    main()
