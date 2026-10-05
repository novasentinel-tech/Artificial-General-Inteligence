from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

import json
import os

import numpy as np

from .features import DIMENSIONS
from .models import StimulusEvent, ValenceResult

VALENCE_SCHEMA_VERSION = 1

DEFAULT_LEARNING_RATE = 0.05

DEFAULT_CONFIDENCE_HALF_SATURATION = 20.0

@dataclass(frozen=True)
class ValenceLearningUpdate:

    outcome: float

    raw_prediction_before: float

    squashed_prediction_before: float

    prediction_error: float

    learning_rate: float

    delta: list[float]

    learned_weights_before: list[float]

    learned_weights_after: list[float]

class ValenceEngine:

    def __init__(
        self,
        intrinsic_weights: (
            Mapping[str, float]
            | Sequence[float]
            | np.ndarray
            | None
        ) = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        confidence_half_saturation: float = (
            DEFAULT_CONFIDENCE_HALF_SATURATION
        ),
        storage_path: str | Path | None = None,
        load_state: bool = True,
    ) -> None:

        if learning_rate <= 0.0:

            raise ValueError(
                "learning_rate deve ser maior que zero."
            )

        if confidence_half_saturation <= 0.0:

            raise ValueError(
                "confidence_half_saturation deve ser maior "
                "que zero."
            )

        self.learning_rate = float(
            learning_rate
        )

        self.confidence_half_saturation = float(
            confidence_half_saturation
        )

        self._lock = RLock()

        self._intrinsic_explicitly_provided = (
            intrinsic_weights is not None
        )

        self.intrinsic_weights = (
            self._coerce_weights(
                intrinsic_weights
            )
        )

        self.learned_weights = np.zeros(
            len(DIMENSIONS),
            dtype=np.float64,
        )

        self.experience_count: int = 0

        self.last_prediction_error: float | None = None

        self.last_outcome: float | None = None

        if storage_path is None:

            project_root = (
                Path(__file__)
                .resolve()
                .parents[2]
            )

            storage_path = (
                project_root
                /
                "data"
                /
                "brain"
                /
                "valence.json"
            )

        self.storage_path = Path(
            storage_path
        )

        if load_state:

            self._load()

    @property
    def weights(
        self,
    ) -> np.ndarray:

        return (
            self.intrinsic_weights
            +
            self.learned_weights
        ).copy()

    def evaluate(
        self,
        event: StimulusEvent,
        features: np.ndarray,
    ) -> ValenceResult:

        phi = self._validate_features(
            features
        )

        with self._lock:

            total_weights = (
                self.intrinsic_weights
                +
                self.learned_weights
            )

            raw_activation = float(

                np.dot(
                    total_weights,
                    phi,
                )

            )

            score = float(
                np.tanh(
                    raw_activation
                )
            )

            components = {

                dimension:
                    float(
                        total_weights[index]
                        *
                        phi[index]
                    )

                for index, dimension
                in enumerate(
                    DIMENSIONS
                )

            }

            confidence = (
                self._calculate_confidence()
            )

        return ValenceResult(

            stimulus_id=
                event.id,

            score=
                score,

            confidence=
                confidence,

            components=
                components,

            raw_activation=
                raw_activation,

        )

    def learn(
        self,
        features: np.ndarray,
        outcome: float,
        learning_rate: float | None = None,
    ) -> ValenceLearningUpdate:

        phi = self._validate_features(
            features
        )

        outcome = float(
            outcome
        )

        if not np.isfinite(
            outcome
        ):

            raise ValueError(
                "outcome precisa ser um número finito."
            )

        if not -1.0 <= outcome <= 1.0:

            raise ValueError(
                "outcome deve estar entre -1.0 e +1.0."
            )

        if learning_rate is None:

            eta = self.learning_rate

        else:

            eta = float(
                learning_rate
            )

            if eta <= 0.0:

                raise ValueError(
                    "learning_rate deve ser maior que zero."
                )

        with self._lock:

            learned_before = (
                self.learned_weights.copy()
            )

            total_weights = (
                self.intrinsic_weights
                +
                self.learned_weights
            )

            raw_prediction = float(

                np.dot(
                    total_weights,
                    phi,
                )

            )

            squashed_prediction = float(

                np.tanh(
                    raw_prediction
                )

            )

            prediction_error = (

                outcome
                -
                raw_prediction

            )

            delta = (

                eta

                *

                prediction_error

                *

                phi

            )

            self.learned_weights += (
                delta
            )

            self.experience_count += 1

            self.last_prediction_error = (
                float(
                    prediction_error
                )
            )

            self.last_outcome = (
                outcome
            )

            learned_after = (
                self.learned_weights.copy()
            )

            self._save_locked()

        return ValenceLearningUpdate(

            outcome=
                outcome,

            raw_prediction_before=
                raw_prediction,

            squashed_prediction_before=
                squashed_prediction,

            prediction_error=
                float(
                    prediction_error
                ),

            learning_rate=
                eta,

            delta=
                delta.tolist(),

            learned_weights_before=
                learned_before.tolist(),

            learned_weights_after=
                learned_after.tolist(),

        )

    def intrinsic_activation(
        self,
        features: np.ndarray,
    ) -> float:

        phi = self._validate_features(
            features
        )

        with self._lock:

            return float(

                np.dot(
                    self.intrinsic_weights,
                    phi,
                )

            )

    def learned_activation(
        self,
        features: np.ndarray,
    ) -> float:

        phi = self._validate_features(
            features
        )

        with self._lock:

            return float(

                np.dot(
                    self.learned_weights,
                    phi,
                )

            )

    def activation_breakdown(
        self,
        features: np.ndarray,
    ) -> dict[str, Any]:

        phi = self._validate_features(
            features
        )

        with self._lock:

            intrinsic = (
                self.intrinsic_weights
                *
                phi
            )

            learned = (
                self.learned_weights
                *
                phi
            )

            total = (
                intrinsic
                +
                learned
            )

        return {

            "features": {

                dimension:
                    float(
                        phi[index]
                    )

                for index, dimension
                in enumerate(
                    DIMENSIONS
                )

            },

            "intrinsic": {

                dimension:
                    float(
                        intrinsic[index]
                    )

                for index, dimension
                in enumerate(
                    DIMENSIONS
                )

            },

            "learned": {

                dimension:
                    float(
                        learned[index]
                    )

                for index, dimension
                in enumerate(
                    DIMENSIONS
                )

            },

            "total": {

                dimension:
                    float(
                        total[index]
                    )

                for index, dimension
                in enumerate(
                    DIMENSIONS
                )

            },

            "raw_activation":
                float(
                    np.sum(
                        total
                    )
                ),

            "valence":
                float(
                    np.tanh(
                        np.sum(
                            total
                        )
                    )
                ),

        }

    def _calculate_confidence(
        self,
    ) -> float:

        n = float(
            self.experience_count
        )

        k = (
            self.confidence_half_saturation
        )

        return float(

            n
            /
            (
                n
                +
                k
            )

        )

    def set_intrinsic_weights(
        self,
        weights: (
            Mapping[str, float]
            | Sequence[float]
            | np.ndarray
        ),
        save: bool = True,
    ) -> None:

        vector = self._coerce_weights(
            weights
        )

        with self._lock:

            self.intrinsic_weights = (
                vector
            )

            self._intrinsic_explicitly_provided = (
                True
            )

            if save:

                self._save_locked()

    def reset_learning(
        self,
        save: bool = True,
    ) -> None:

        with self._lock:

            self.learned_weights = np.zeros(
                len(
                    DIMENSIONS
                ),
                dtype=np.float64,
            )

            self.experience_count = 0

            self.last_prediction_error = None

            self.last_outcome = None

            if save:

                self._save_locked()

    def snapshot(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            total = (

                self.intrinsic_weights
                +
                self.learned_weights

            )

            return {

                "experience_count":
                    self.experience_count,

                "learning_rate":
                    self.learning_rate,

                "confidence":
                    self._calculate_confidence(),

                "last_prediction_error":
                    self.last_prediction_error,

                "last_outcome":
                    self.last_outcome,

                "intrinsic_weights": {

                    dimension:
                        float(
                            self.intrinsic_weights[
                                index
                            ]
                        )

                    for index, dimension
                    in enumerate(
                        DIMENSIONS
                    )

                },

                "learned_weights": {

                    dimension:
                        float(
                            self.learned_weights[
                                index
                            ]
                        )

                    for index, dimension
                    in enumerate(
                        DIMENSIONS
                    )

                },

                "total_weights": {

                    dimension:
                        float(
                            total[
                                index
                            ]
                        )

                    for index, dimension
                    in enumerate(
                        DIMENSIONS
                    )

                },

                "storage":
                    str(
                        self.storage_path
                    ),

            }

    @staticmethod
    def _coerce_weights(
        weights: (
            Mapping[str, float]
            | Sequence[float]
            | np.ndarray
            | None
        ),
    ) -> np.ndarray:

        if weights is None:

            return np.zeros(
                len(
                    DIMENSIONS
                ),
                dtype=np.float64,
            )

        if isinstance(
            weights,
            Mapping,
        ):

            unknown_dimensions = (

                set(
                    weights.keys()
                )

                -

                set(
                    DIMENSIONS
                )

            )

            if unknown_dimensions:

                raise ValueError(
                    "Dimensões desconhecidas nos pesos: "
                    f"{sorted(unknown_dimensions)}"
                )

            vector = np.asarray(

                [

                    float(
                        weights.get(
                            dimension,
                            0.0,
                        )
                    )

                    for dimension
                    in DIMENSIONS

                ],

                dtype=np.float64,

            )

        else:

            vector = np.asarray(
                weights,
                dtype=np.float64,
            ).reshape(
                -1
            )

        if (
            vector.size
            !=
            len(
                DIMENSIONS
            )
        ):

            raise ValueError(

                "O vetor de pesos deve possuir "
                f"{len(DIMENSIONS)} dimensões."

            )

        if not np.all(
            np.isfinite(
                vector
            )
        ):

            raise ValueError(
                "Todos os pesos devem ser números finitos."
            )

        return (
            vector.copy()
        )

    @staticmethod
    def _validate_features(
        features: np.ndarray,
    ) -> np.ndarray:

        phi = np.asarray(
            features,
            dtype=np.float64,
        ).reshape(
            -1
        )

        expected_size = len(
            DIMENSIONS
        )

        if (
            phi.size
            !=
            expected_size
        ):

            raise ValueError(

                "φ(E) deve possuir "
                f"{expected_size} dimensões: "
                f"{DIMENSIONS}"

            )

        if not np.all(
            np.isfinite(
                phi
            )
        ):

            raise ValueError(
                "φ(E) contém NaN ou infinito."
            )

        intensity = phi[0]
        novelty = phi[1]
        urgency = phi[2]
        goal_alignment = phi[3]
        risk = phi[4]

        if not 0.0 <= intensity <= 1.0:

            raise ValueError(
                "intensity deve estar entre 0 e 1."
            )

        if not 0.0 <= novelty <= 1.0:

            raise ValueError(
                "novelty deve estar entre 0 e 1."
            )

        if not 0.0 <= urgency <= 1.0:

            raise ValueError(
                "urgency deve estar entre 0 e 1."
            )

        if not -1.0 <= goal_alignment <= 1.0:

            raise ValueError(
                "goal_alignment deve estar entre -1 e 1."
            )

        if not 0.0 <= risk <= 1.0:

            raise ValueError(
                "risk deve estar entre 0 e 1."
            )

        return (
            phi.copy()
        )

    def save(
        self,
    ) -> None:

        with self._lock:

            self._save_locked()

    def _save_locked(
        self,
    ) -> None:

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {

            "schema_version":
                VALENCE_SCHEMA_VERSION,

            "dimensions":
                list(
                    DIMENSIONS
                ),

            "learning_rate":
                self.learning_rate,

            "confidence_half_saturation":
                self.confidence_half_saturation,

            "experience_count":
                self.experience_count,

            "last_prediction_error":
                self.last_prediction_error,

            "last_outcome":
                self.last_outcome,

            "intrinsic_weights":
                self.intrinsic_weights.tolist(),

            "learned_weights":
                self.learned_weights.tolist(),

        }

        temporary_path = (
            self.storage_path
            .with_suffix(
                ".tmp"
            )
        )

        temporary_path.write_text(

            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),

            encoding="utf-8",

        )

        os.replace(
            temporary_path,
            self.storage_path,
        )

    def _load(
        self,
    ) -> None:

        if not self.storage_path.exists():

            return

        try:

            raw = (
                self.storage_path
                .read_text(
                    encoding="utf-8"
                )
            )

            data = json.loads(
                raw
            )

        except Exception as exc:

            raise RuntimeError(

                "Não foi possível carregar "
                "o estado do Valence Engine em "
                f"{self.storage_path}"

            ) from exc

        schema_version = int(
            data.get(
                "schema_version",
                0,
            )
        )

        if (
            schema_version
            !=
            VALENCE_SCHEMA_VERSION
        ):

            raise RuntimeError(

                "Versão incompatível do estado "
                "do Valence Engine. "
                f"Encontrada: {schema_version}. "
                f"Esperada: {VALENCE_SCHEMA_VERSION}."

            )

        stored_dimensions = tuple(
            data.get(
                "dimensions",
                [],
            )
        )

        if (
            stored_dimensions
            !=
            tuple(
                DIMENSIONS
            )
        ):

            raise RuntimeError(

                "As dimensões cognitivas do estado salvo "
                "não correspondem às dimensões atuais."

            )

        learned = self._coerce_weights(

            data.get(
                "learned_weights",
                None,
            )

        )

        if not self._intrinsic_explicitly_provided:

            self.intrinsic_weights = (
                self._coerce_weights(

                    data.get(
                        "intrinsic_weights",
                        None,
                    )

                )
            )

        self.learned_weights = (
            learned
        )

        self.experience_count = int(

            data.get(
                "experience_count",
                0,
            )

        )

        self.last_prediction_error = (
            data.get(
                "last_prediction_error",
                None,
            )
        )

        self.last_outcome = (
            data.get(
                "last_outcome",
                None,
            )
        )
