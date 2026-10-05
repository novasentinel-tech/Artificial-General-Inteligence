from __future__ import annotations

import numpy as np

from .models import StimulusEvent


DIMENSIONS = (
    "intensity",
    "novelty",
    "urgency",
    "goal_alignment",
    "risk",
)


def extract_features(event: StimulusEvent, novelty: float) -> np.ndarray:
    payload = event.payload
    return np.asarray(
        [
            event.intensity,
            float(novelty),
            float(payload.get("urgency", 0.5)),
            float(payload.get("goal_alignment", 0.0)),
            float(payload.get("risk_level", 0.0)),
        ],
        dtype=np.float64,
    )
