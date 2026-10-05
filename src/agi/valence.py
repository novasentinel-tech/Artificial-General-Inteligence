from __future__ import annotations

import numpy as np

from .features import DIMENSIONS
from .models import StimulusEvent, ValenceResult


class ValenceEngine:
    def __init__(self) -> None:
        self.weights = np.zeros(len(DIMENSIONS), dtype=np.float64)
        self.experience_count = 0

    def evaluate(self, event: StimulusEvent, features: np.ndarray) -> ValenceResult:
        raw = float(np.dot(features, self.weights))
        score = float(np.tanh(raw))

        confidence = 1.0 - (1.0 / (1.0 + self.experience_count * 0.05))

        components = {
            name: float(features[i] * self.weights[i])
            for i, name in enumerate(DIMENSIONS)
        }

        return ValenceResult(
            stimulus_id=event.id,
            score=score,
            confidence=confidence,
            components=components,
        )

    def learn(self, features: np.ndarray, outcome: float, learning_rate: float = 0.05) -> None:
        outcome = float(np.clip(outcome, -1.0, 1.0))
        predicted = float(np.dot(features, self.weights))
        error = outcome - predicted
        self.weights += learning_rate * error * features
        self.experience_count += 1
