from __future__ import annotations

from dataclasses import dataclass

from .decision import DecisionModule
from .features import extract_features
from .memory import PatternMemory
from .models import ActionCandidate, DecisionResult, StimulusEvent, ValenceResult
from .triggers import Trigger, TriggerSystem
from .valence import ValenceEngine


@dataclass
class CognitiveStep:
    stimulus: StimulusEvent
    novelty: float
    valence: ValenceResult
    triggers: list[Trigger]
    decision: DecisionResult


class CognitiveEngine:
    def __init__(self) -> None:
        self.memory = PatternMemory()
        self.valence = ValenceEngine()
        self.triggers = TriggerSystem()
        self.decisions = DecisionModule()

    def process(
        self,
        event: StimulusEvent,
        actions: list[ActionCandidate],
    ) -> CognitiveStep:
        # First-pass novelty is computed using intensity as a neutral placeholder.
        # We then recompute the final feature vector including novelty.
        seed_features = extract_features(event, novelty=0.5)
        novelty = self.memory.novelty(seed_features)

        features = extract_features(event, novelty=novelty)
        valence = self.valence.evaluate(event, features)
        triggers = self.triggers.evaluate(event, valence)
        decision = self.decisions.choose(event, valence, triggers, actions)

        self.memory.remember(event.type, features, valence.score)

        return CognitiveStep(
            stimulus=event,
            novelty=novelty,
            valence=valence,
            triggers=triggers,
            decision=decision,
        )

    def learn_from_outcome(self, event: StimulusEvent, novelty: float, outcome: float) -> None:
        features = extract_features(event, novelty=novelty)
        self.valence.learn(features, outcome)
