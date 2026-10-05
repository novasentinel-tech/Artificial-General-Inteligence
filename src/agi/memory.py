from __future__ import annotations

import numpy as np

from .models import PatternRecord


class PatternMemory:
    def __init__(self) -> None:
        self._patterns: list[PatternRecord] = []

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def novelty(self, features: np.ndarray) -> float:
        if not self._patterns:
            return 1.0

        similarities = [
            self._cosine_similarity(features, np.asarray(p.feature_vector, dtype=np.float64))
            for p in self._patterns
        ]
        best = max(similarities, default=0.0)
        return float(np.clip(1.0 - max(best, 0.0), 0.0, 1.0))

    def remember(self, stimulus_type: str, features: np.ndarray, valence: float) -> PatternRecord:
        best_index = None
        best_similarity = -1.0

        for i, pattern in enumerate(self._patterns):
            if pattern.stimulus_type != stimulus_type:
                continue
            similarity = self._cosine_similarity(
                features,
                np.asarray(pattern.feature_vector, dtype=np.float64),
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = i

        if best_index is not None and best_similarity >= 0.97:
            old = self._patterns[best_index]
            new_frequency = old.frequency + 1
            new_average = (
                old.average_valence * old.frequency + valence
            ) / new_frequency
            updated = PatternRecord(
                stimulus_type=old.stimulus_type,
                feature_vector=old.feature_vector,
                frequency=new_frequency,
                average_valence=new_average,
            )
            self._patterns[best_index] = updated
            return updated

        record = PatternRecord(
            stimulus_type=stimulus_type,
            feature_vector=features.tolist(),
            frequency=1,
            average_valence=valence,
        )
        self._patterns.append(record)
        return record

    @property
    def patterns(self) -> tuple[PatternRecord, ...]:
        return tuple(self._patterns)
