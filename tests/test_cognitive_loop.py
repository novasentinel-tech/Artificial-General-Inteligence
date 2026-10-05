from agi.engine import CognitiveEngine
from agi.models import ActionCandidate, StimulusEvent, StimulusSource


def test_full_cognitive_cycle_runs():
    engine = CognitiveEngine()

    event = StimulusEvent(
        source=StimulusSource.EXTERNAL,
        type="test",
        intensity=0.8,
        payload={
            "urgency": 0.9,
            "risk_level": 0.9,
            "goal_alignment": -0.2,
        },
    )

    actions = [
        ActionCandidate(name="observe", base_utility=0.2),
        ActionCandidate(name="avoid", base_utility=0.3, risk_penalty=-0.4),
    ]

    step = engine.process(event, actions)

    assert -1.0 <= step.valence.score <= 1.0
    assert 0.0 <= step.novelty <= 1.0
    assert step.decision.action in {"observe", "avoid"}
    assert any(trigger.name == "high_risk" for trigger in step.triggers)


def test_valence_learns_from_negative_outcome():
    engine = CognitiveEngine()

    event = StimulusEvent(
        source=StimulusSource.EXTERNAL,
        type="danger",
        intensity=1.0,
        payload={
            "urgency": 1.0,
            "risk_level": 1.0,
            "goal_alignment": -1.0,
        },
    )

    actions = [ActionCandidate(name="avoid", base_utility=0.5)]

    before = engine.process(event, actions)
    engine.learn_from_outcome(event, before.novelty, outcome=-1.0)
    after = engine.process(event, actions)

    assert after.valence.score < before.valence.score
