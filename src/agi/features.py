from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import StimulusEvent


# ============================================================
# DIMENSÕES COGNITIVAS DO MODELO
# ============================================================

DIMENSIONS = (
    "intensity",
    "novelty",
    "urgency",
    "goal_alignment",
    "risk",
)


@dataclass(frozen=True)
class FeatureVector:
    """
    Representação estruturada das características cognitivas
    extraídas de um estímulo.

    Cada dimensão possui significado explícito:

    intensity:
        Magnitude percebida do estímulo.
        Intervalo: [0, 1]

    novelty:
        Grau de diferença em relação a padrões conhecidos.
        Intervalo: [0, 1]

    urgency:
        Pressão temporal para resposta.
        Intervalo: [0, 1]

    goal_alignment:
        Grau de compatibilidade com os objetivos atuais.
        Intervalo: [-1, 1]

        -1 -> fortemente contrário aos objetivos
         0 -> neutro
        +1 -> fortemente favorável

    risk:
        Potencial estimado de consequência negativa.
        Intervalo: [0, 1]
    """

    intensity: float
    novelty: float
    urgency: float
    goal_alignment: float
    risk: float

    def as_array(self) -> np.ndarray:
        """
        Converte a estrutura para vetor NumPy na ordem
        definida em DIMENSIONS.
        """

        return np.asarray(
            [
                self.intensity,
                self.novelty,
                self.urgency,
                self.goal_alignment,
                self.risk,
            ],
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, float]:
        """
        Retorna representação nomeada das features.
        """

        return {
            "intensity": self.intensity,
            "novelty": self.novelty,
            "urgency": self.urgency,
            "goal_alignment": self.goal_alignment,
            "risk": self.risk,
        }


# ============================================================
# HELPERS DE NORMALIZAÇÃO
# ============================================================

def clamp01(value: float) -> float:
    """
    Limita um valor ao intervalo [0, 1].
    """

    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def clamp_signed(value: float) -> float:
    """
    Limita um valor ao intervalo [-1, 1].
    """

    return max(
        -1.0,
        min(
            1.0,
            float(value),
        ),
    )


# ============================================================
# EXTRAÇÃO
# ============================================================

def extract_feature_vector(
    event: StimulusEvent,
    novelty: float,
) -> FeatureVector:
    """
    Extrai as dimensões cognitivas principais de um estímulo.

    A função não aprende nada.

    Ela apenas converte informações do StimulusEvent
    em um espaço numérico coerente para processamento
    posterior pelo Valence Engine.

    A novidade é fornecida externamente porque depende
    da comparação com a memória.
    """

    payload = event.payload

    intensity = clamp01(
        event.intensity
    )

    novelty_value = clamp01(
        novelty
    )

    urgency = clamp01(
        payload.get(
            "urgency",
            0.0,
        )
    )

    goal_alignment = clamp_signed(
        payload.get(
            "goal_alignment",
            0.0,
        )
    )

    risk = clamp01(
        payload.get(
            "risk_level",
            0.0,
        )
    )

    return FeatureVector(
        intensity=intensity,
        novelty=novelty_value,
        urgency=urgency,
        goal_alignment=goal_alignment,
        risk=risk,
    )


def extract_features(
    event: StimulusEvent,
    novelty: float,
) -> np.ndarray:
    """
    Interface principal usada pelos outros módulos.

    Retorna:

        φ(E) = [
            intensity,
            novelty,
            urgency,
            goal_alignment,
            risk
        ]

    onde:

        intensity      ∈ [0, 1]
        novelty        ∈ [0, 1]
        urgency        ∈ [0, 1]
        goal_alignment ∈ [-1, 1]
        risk           ∈ [0, 1]
    """

    return extract_feature_vector(
        event=event,
        novelty=novelty,
    ).as_array()


# ============================================================
# FEATURE VECTOR PARA COMPARAÇÃO DE PADRÕES
# ============================================================

def extract_pattern_features(
    event: StimulusEvent,
) -> np.ndarray:
    """
    Extrai apenas características que podem ser comparadas
    ANTES de calcular novidade.

    Isso evita um problema lógico circular:

        novidade depende da memória

    mas:

        memória não deve depender da novidade para descobrir
        se algo é novo.

    Por isso, para comparação de padrões usamos:

        [
            intensity,
            urgency,
            goal_alignment,
            risk
        ]

    sem incluir novelty.
    """

    payload = event.payload

    return np.asarray(
        [
            clamp01(
                event.intensity
            ),

            clamp01(
                payload.get(
                    "urgency",
                    0.0,
                )
            ),

            clamp_signed(
                payload.get(
                    "goal_alignment",
                    0.0,
                )
            ),

            clamp01(
                payload.get(
                    "risk_level",
                    0.0,
                )
            ),
        ],
        dtype=np.float64,
    )