from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import json
import math
import os

import numpy as np

from .features import extract_pattern_features
from .models import StimulusEvent


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

MEMORY_SCHEMA_VERSION = 1

DEFAULT_SIGNATURE_DIMENSION = 128

DEFAULT_SEARCH_THRESHOLD = 0.75

DEFAULT_MERGE_THRESHOLD = 0.93

DEFAULT_MAX_RESULTS = 5

DEFAULT_MAX_CONSEQUENCE_HISTORY = 2000


# ============================================================
# UTILIDADES
# ============================================================

def utc_now() -> str:
    """
    Timestamp UTC padronizado.

    Sempre usamos UTC para impedir inconsistências temporais
    quando o sistema for executado em máquinas diferentes.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Limita um valor a determinado intervalo.
    """

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


# ============================================================
# CONSEQUÊNCIA DE UMA AÇÃO
# ============================================================

@dataclass
class PatternConsequence:
    """
    Representa uma consequência real observada.

    Relação:

        padrão
          ↓
        ação
          ↓
        consequência

    outcome:

        -1.0 = extremamente negativo
         0.0 = neutro
        +1.0 = extremamente positivo
    """

    action_id: str

    outcome: float

    timestamp: str = field(
        default_factory=utc_now
    )


# ============================================================
# ESTATÍSTICA POR AÇÃO
# ============================================================

@dataclass
class ActionOutcomeStats:
    """
    Memória agregada das consequências de uma ação
    aplicada sobre determinado padrão.

    Isso permite responder futuramente:

        "Quando encontrei esse padrão anteriormente
         e escolhi esta ação, o que normalmente aconteceu?"
    """

    count: int = 0

    mean_outcome: float = 0.0

    last_outcome: float = 0.0

    last_seen: str = field(
        default_factory=utc_now
    )

    def update(
        self,
        outcome: float,
    ) -> None:
        """
        Atualização incremental da média.

        nova_média =
            média_antiga
            +
            (novo_valor - média_antiga)
            / número_de_amostras

        Essa fórmula evita precisar recalcular todo
        o histórico sempre que uma experiência chega.
        """

        self.count += 1

        self.mean_outcome += (
            outcome - self.mean_outcome
        ) / self.count

        self.last_outcome = outcome

        self.last_seen = utc_now()


# ============================================================
# PADRÃO COGNITIVO
# ============================================================

@dataclass
class PatternRecord:
    """
    Representação persistente de um padrão cognitivo.

    Conceitualmente:

        P = <σ, S, f, v̄, ω>

    onde:

        σ = assinatura vetorial
        S = tipos de estímulo
        f = frequência
        v̄ = valência histórica
        ω = peso/importância
    """

    id: str

    stimulus_type: str

    stimulus_types: list[str]

    # Vetor estrutural interpretável:
    #
    # [
    #   intensity,
    #   urgency,
    #   goal_alignment,
    #   risk
    # ]
    feature_vector: list[float]

    # Assinatura matemática usada para similaridade.
    #
    # Dimensão padrão:
    #
    # 128
    signature: list[float]

    frequency: int = 1

    # Média das valências PREDITAS quando
    # o padrão foi reconhecido.
    predicted_valence_mean: float = 0.0

    # Média das consequências REALMENTE observadas.
    experienced_valence_mean: float = 0.0

    outcome_count: int = 0

    importance_weight: float = 0.0

    first_seen: str = field(
        default_factory=utc_now
    )

    last_seen: str = field(
        default_factory=utc_now
    )

    action_stats: dict[
        str,
        ActionOutcomeStats
    ] = field(
        default_factory=dict
    )

    consequences: list[
        PatternConsequence
    ] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # VALÊNCIA HISTÓRICA
    # --------------------------------------------------------

    @property
    def average_valence(
        self,
    ) -> float:
        """
        Se já existem consequências reais,
        usamos a experiência observada.

        Caso contrário, usamos a expectativa prevista.

        Isso evita misturar:

            previsão

        com:

            consequência real
        """

        if self.outcome_count > 0:

            return self.experienced_valence_mean

        return self.predicted_valence_mean

    # --------------------------------------------------------
    # SERIALIZAÇÃO
    # --------------------------------------------------------

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {

            "id":
                self.id,

            "stimulus_type":
                self.stimulus_type,

            "stimulus_types":
                self.stimulus_types,

            "feature_vector":
                self.feature_vector,

            "signature":
                self.signature,

            "frequency":
                self.frequency,

            "predicted_valence_mean":
                self.predicted_valence_mean,

            "experienced_valence_mean":
                self.experienced_valence_mean,

            "outcome_count":
                self.outcome_count,

            "importance_weight":
                self.importance_weight,

            "first_seen":
                self.first_seen,

            "last_seen":
                self.last_seen,

            "action_stats": {

                action_id:
                    asdict(stats)

                for action_id, stats
                in self.action_stats.items()

            },

            "consequences": [

                asdict(consequence)

                for consequence
                in self.consequences

            ],

        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> PatternRecord:

        action_stats = {

            action_id:
                ActionOutcomeStats(
                    **stats
                )

            for action_id, stats
            in data.get(
                "action_stats",
                {},
            ).items()

        }

        consequences = [

            PatternConsequence(
                **item
            )

            for item
            in data.get(
                "consequences",
                [],
            )

        ]

        return cls(

            id=data["id"],

            stimulus_type=data[
                "stimulus_type"
            ],

            stimulus_types=list(
                data.get(
                    "stimulus_types",
                    [
                        data[
                            "stimulus_type"
                        ]
                    ],
                )
            ),

            feature_vector=list(
                data[
                    "feature_vector"
                ]
            ),

            signature=list(
                data[
                    "signature"
                ]
            ),

            frequency=int(
                data.get(
                    "frequency",
                    1,
                )
            ),

            predicted_valence_mean=float(
                data.get(
                    "predicted_valence_mean",
                    0.0,
                )
            ),

            experienced_valence_mean=float(
                data.get(
                    "experienced_valence_mean",
                    0.0,
                )
            ),

            outcome_count=int(
                data.get(
                    "outcome_count",
                    0,
                )
            ),

            importance_weight=float(
                data.get(
                    "importance_weight",
                    0.0,
                )
            ),

            first_seen=data.get(
                "first_seen",
                utc_now(),
            ),

            last_seen=data.get(
                "last_seen",
                utc_now(),
            ),

            action_stats=action_stats,

            consequences=consequences,
        )


# ============================================================
# RESULTADO DE BUSCA NA MEMÓRIA
# ============================================================

@dataclass(frozen=True)
class PatternMatch:
    """
    Resultado de uma recuperação de memória.
    """

    pattern_id: str

    stimulus_type: str

    similarity: float

    average_valence: float

    frequency: int

    weight: float

    action_expectations: dict[
        str,
        float
    ]


# ============================================================
# PATTERN MEMORY
# ============================================================

class PatternMemory:
    """
    Sistema de memória episódica/padrões.

    Responsabilidades:

    1. representar experiências;
    2. gerar assinaturas vetoriais;
    3. calcular similaridade;
    4. calcular novidade;
    5. reconhecer padrões;
    6. consolidar experiências repetidas;
    7. registrar consequências;
    8. persistir memória entre execuções.
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        signature_dimension: int = DEFAULT_SIGNATURE_DIMENSION,
        search_threshold: float = DEFAULT_SEARCH_THRESHOLD,
        merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
        max_consequence_history: int = DEFAULT_MAX_CONSEQUENCE_HISTORY,
        state_weight: float = 0.60,
        type_weight: float = 0.40,
    ) -> None:

        if signature_dimension < 32:

            raise ValueError(
                "signature_dimension deve ser >= 32"
            )

        if not 0.0 <= search_threshold <= 1.0:

            raise ValueError(
                "search_threshold deve estar entre 0 e 1"
            )

        if not 0.0 <= merge_threshold <= 1.0:

            raise ValueError(
                "merge_threshold deve estar entre 0 e 1"
            )

        if merge_threshold <= search_threshold:

            raise ValueError(
                "merge_threshold deve ser maior "
                "que search_threshold"
            )

        weight_sum = (
            state_weight
            +
            type_weight
        )

        if weight_sum <= 0:

            raise ValueError(
                "Os pesos da assinatura devem "
                "somar valor maior que zero."
            )

        # Normalização automática.
        self.state_weight = (
            state_weight
            /
            weight_sum
        )

        self.type_weight = (
            type_weight
            /
            weight_sum
        )

        self.signature_dimension = (
            signature_dimension
        )

        self.search_threshold = (
            search_threshold
        )

        self.merge_threshold = (
            merge_threshold
        )

        self.max_consequence_history = (
            max_consequence_history
        )

        self._lock = RLock()

        self._patterns: list[
            PatternRecord
        ] = []

        # ----------------------------------------------------
        # LOCAL DE PERSISTÊNCIA
        # ----------------------------------------------------

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
                "memory.json"
            )

        self.storage_path = Path(
            storage_path
        )

        self.created_at = utc_now()

        self.updated_at = utc_now()

        self._load()

    # ========================================================
    # ACESSO À MEMÓRIA
    # ========================================================

    @property
    def patterns(
        self,
    ) -> tuple[PatternRecord, ...]:
        """
        Retorna visão somente-leitura dos padrões.
        """

        with self._lock:

            return tuple(
                self._patterns
            )

    # ========================================================
    # NOVIDADE
    # ========================================================

    def estimate_novelty(
        self,
        event: StimulusEvent,
    ) -> float:
        """
        Calcula novidade:

            N(E) = 1 - max(sim(E, P))

        Interpretação:

            1.0 -> totalmente desconhecido

            0.0 -> praticamente idêntico
                   a padrão conhecido
        """

        structural_features = (
            extract_pattern_features(
                event
            )
        )

        signature = (
            self._build_signature(
                stimulus_type=event.type,
                structural_features=structural_features,
            )
        )

        with self._lock:

            if not self._patterns:

                return 1.0

            similarities = [

                self._cosine_similarity(
                    signature,
                    np.asarray(
                        pattern.signature,
                        dtype=np.float64,
                    ),
                )

                for pattern
                in self._patterns

            ]

        best_similarity = max(
            similarities
        )

        novelty = (
            1.0
            -
            best_similarity
        )

        return clamp(
            novelty,
            0.0,
            1.0,
        )

    # ========================================================
    # COMPATIBILIDADE
    # ========================================================

    def novelty(
        self,
        features: np.ndarray,
        stimulus_type: str = "unknown",
    ) -> float:
        """
        Interface alternativa para compatibilidade.

        Preferir estimate_novelty(event).
        """

        structural = (
            self._extract_structural_features(
                features
            )
        )

        signature = (
            self._build_signature(
                stimulus_type=stimulus_type,
                structural_features=structural,
            )
        )

        with self._lock:

            if not self._patterns:

                return 1.0

            best_similarity = max(

                self._cosine_similarity(
                    signature,
                    np.asarray(
                        pattern.signature,
                        dtype=np.float64,
                    ),
                )

                for pattern
                in self._patterns

            )

        return clamp(
            1.0 - best_similarity,
            0.0,
            1.0,
        )

    # ========================================================
    # BUSCA DE PADRÕES
    # ========================================================

    def find_similar(
        self,
        stimulus_type: str,
        features: np.ndarray,
        threshold: float | None = None,
        limit: int = DEFAULT_MAX_RESULTS,
    ) -> list[PatternMatch]:
        """
        Recupera padrões semanticamente/estruturalmente
        semelhantes.

        A ordenação é feita por similaridade cosseno
        decrescente.
        """

        if threshold is None:

            threshold = (
                self.search_threshold
            )

        threshold = clamp(
            threshold,
            0.0,
            1.0,
        )

        structural = (
            self._extract_structural_features(
                features
            )
        )

        signature = (
            self._build_signature(
                stimulus_type=stimulus_type,
                structural_features=structural,
            )
        )

        matches: list[
            PatternMatch
        ] = []

        with self._lock:

            for pattern in self._patterns:

                similarity = (
                    self._cosine_similarity(
                        signature,
                        np.asarray(
                            pattern.signature,
                            dtype=np.float64,
                        ),
                    )
                )

                if similarity < threshold:

                    continue

                expectations = {

                    action_id:
                        stats.mean_outcome

                    for action_id, stats
                    in pattern.action_stats.items()

                }

                matches.append(

                    PatternMatch(

                        pattern_id=
                            pattern.id,

                        stimulus_type=
                            pattern.stimulus_type,

                        similarity=
                            similarity,

                        average_valence=
                            pattern.average_valence,

                        frequency=
                            pattern.frequency,

                        weight=
                            pattern.importance_weight,

                        action_expectations=
                            expectations,

                    )

                )

        matches.sort(
            key=lambda item:
                item.similarity,
            reverse=True,
        )

        return matches[
            :max(
                1,
                int(limit),
            )
        ]

    # ========================================================
    # FORMAÇÃO / CONSOLIDAÇÃO DE PADRÕES
    # ========================================================

    def remember(
        self,
        stimulus_type: str,
        features: np.ndarray,
        valence: float,
    ) -> PatternRecord:
        """
        Registra nova experiência.

        Se já existe padrão suficientemente semelhante,
        consolidamos a experiência nele.

        Caso contrário, formamos novo padrão.
        """

        valence = clamp(
            valence,
            -1.0,
            1.0,
        )

        structural = (
            self._extract_structural_features(
                features
            )
        )

        signature = (
            self._build_signature(
                stimulus_type=stimulus_type,
                structural_features=structural,
            )
        )

        now = utc_now()

        with self._lock:

            best_pattern: (
                PatternRecord
                |
                None
            ) = None

            best_similarity = -1.0

            for pattern in self._patterns:

                similarity = (
                    self._cosine_similarity(
                        signature,
                        np.asarray(
                            pattern.signature,
                            dtype=np.float64,
                        ),
                    )
                )

                if similarity > best_similarity:

                    best_similarity = (
                        similarity
                    )

                    best_pattern = (
                        pattern
                    )

            # ------------------------------------------------
            # CONSOLIDAÇÃO
            # ------------------------------------------------

            if (
                best_pattern
                is not None
                and
                best_similarity
                >= self.merge_threshold
            ):

                old_frequency = (
                    best_pattern.frequency
                )

                new_frequency = (
                    old_frequency
                    +
                    1
                )

                # --------------------------------------------
                # CENTROIDE DAS FEATURES
                # --------------------------------------------

                old_features = np.asarray(
                    best_pattern.feature_vector,
                    dtype=np.float64,
                )

                new_features = (

                    old_features
                    *
                    old_frequency

                    +

                    structural

                ) / new_frequency

                best_pattern.feature_vector = (
                    new_features.tolist()
                )

                # --------------------------------------------
                # CENTROIDE DA ASSINATURA
                # --------------------------------------------

                old_signature = np.asarray(
                    best_pattern.signature,
                    dtype=np.float64,
                )

                centroid = (

                    old_signature
                    *
                    old_frequency

                    +

                    signature

                ) / new_frequency

                centroid_norm = (
                    np.linalg.norm(
                        centroid
                    )
                )

                if centroid_norm > 0:

                    centroid = (
                        centroid
                        /
                        centroid_norm
                    )

                best_pattern.signature = (
                    centroid.tolist()
                )

                # --------------------------------------------
                # MÉDIA DA VALÊNCIA PREDITA
                # --------------------------------------------

                best_pattern.predicted_valence_mean += (

                    valence
                    -
                    best_pattern.predicted_valence_mean

                ) / new_frequency

                best_pattern.frequency = (
                    new_frequency
                )

                best_pattern.last_seen = (
                    now
                )

                if (
                    stimulus_type
                    not in
                    best_pattern.stimulus_types
                ):

                    best_pattern.stimulus_types.append(
                        stimulus_type
                    )

                best_pattern.importance_weight = (
                    self._calculate_importance(
                        best_pattern
                    )
                )

                self._save_locked()

                return best_pattern

            # ------------------------------------------------
            # FORMAÇÃO DE NOVO PADRÃO
            # ------------------------------------------------

            pattern = PatternRecord(

                id=str(
                    uuid4()
                ),

                stimulus_type=
                    stimulus_type,

                stimulus_types=[
                    stimulus_type
                ],

                feature_vector=
                    structural.tolist(),

                signature=
                    signature.tolist(),

                frequency=1,

                predicted_valence_mean=
                    valence,

                experienced_valence_mean=
                    0.0,

                outcome_count=0,

                first_seen=
                    now,

                last_seen=
                    now,

            )

            pattern.importance_weight = (
                self._calculate_importance(
                    pattern
                )
            )

            self._patterns.append(
                pattern
            )

            self._save_locked()

            return pattern

    # ========================================================
    # CONSEQUÊNCIA
    # ========================================================

    def record_outcome(
        self,
        pattern_id: str,
        action_id: str,
        outcome: float,
    ) -> None:
        """
        Registra consequência observada para:

            padrão + ação

        Esta função transforma memória episódica
        em memória útil para decisão.
        """

        outcome = clamp(
            outcome,
            -1.0,
            1.0,
        )

        with self._lock:

            pattern = (
                self._get_pattern_locked(
                    pattern_id
                )
            )

            if pattern is None:

                raise KeyError(
                    f"Padrão não encontrado: "
                    f"{pattern_id}"
                )

            # ------------------------------------------------
            # HISTÓRICO BRUTO
            # ------------------------------------------------

            consequence = (
                PatternConsequence(
                    action_id=action_id,
                    outcome=outcome,
                )
            )

            pattern.consequences.append(
                consequence
            )

            # Evita crescimento infinito do arquivo.
            if (
                len(
                    pattern.consequences
                )
                >
                self.max_consequence_history
            ):

                pattern.consequences = (
                    pattern.consequences[
                        -self.max_consequence_history:
                    ]
                )

            # ------------------------------------------------
            # MÉDIA REAL DE CONSEQUÊNCIAS
            # ------------------------------------------------

            pattern.outcome_count += 1

            pattern.experienced_valence_mean += (

                outcome
                -
                pattern.experienced_valence_mean

            ) / pattern.outcome_count

            # ------------------------------------------------
            # ESTATÍSTICA ESPECÍFICA DA AÇÃO
            # ------------------------------------------------

            stats = (
                pattern.action_stats.get(
                    action_id
                )
            )

            if stats is None:

                stats = (
                    ActionOutcomeStats()
                )

                pattern.action_stats[
                    action_id
                ] = stats

            stats.update(
                outcome
            )

            pattern.last_seen = (
                utc_now()
            )

            pattern.importance_weight = (
                self._calculate_importance(
                    pattern
                )
            )

            self._save_locked()

    # ========================================================
    # EXPECTATIVA DE RESULTADO
    # ========================================================

    def expected_outcome(
        self,
        pattern_id: str,
        action_id: str,
    ) -> float | None:
        """
        Recupera a expectativa histórica:

            E[outcome | padrão, ação]
        """

        with self._lock:

            pattern = (
                self._get_pattern_locked(
                    pattern_id
                )
            )

            if pattern is None:

                return None

            stats = (
                pattern.action_stats.get(
                    action_id
                )
            )

            if stats is None:

                return None

            return (
                stats.mean_outcome
            )

    # ========================================================
    # IMPORTÂNCIA DO PADRÃO
    # ========================================================

    @staticmethod
    def _calculate_importance(
        pattern: PatternRecord,
    ) -> float:
        """
        Peso de importância:

            ω = f × (ε + |v̄|)

        onde:

            f  = frequência

            |v̄| = magnitude da valência histórica

            ε = 0.10

        Por quê?

        Frequência representa recorrência.

        Magnitude da valência representa relevância
        comportamental.

        O epsilon impede que um padrão frequente,
        porém emocionalmente neutro, tenha importância zero.
        """

        epsilon = 0.10

        return (

            pattern.frequency

            *

            (
                epsilon
                +
                abs(
                    pattern.average_valence
                )
            )

        )

    # ========================================================
    # ASSINATURA VETORIAL DE 128 DIMENSÕES
    # ========================================================

    def _build_signature(
        self,
        stimulus_type: str,
        structural_features: np.ndarray,
    ) -> np.ndarray:
        """
        Constrói σ ∈ R¹²⁸.

        A assinatura possui dois componentes:

        1. estado estrutural;
        2. identidade do tipo de estímulo.

        O resultado é normalizado.

        Assim a similaridade cosseno possui
        interpretação consistente.
        """

        structural_features = (
            self._normalize_structural_features(
                structural_features
            )
        )

        # ----------------------------------------------------
        # PARTE 1 — ESTADO
        # ----------------------------------------------------

        state_slots = 16

        state_vector = np.zeros(
            state_slots,
            dtype=np.float64,
        )

        intensity = (
            structural_features[0]
        )

        urgency = (
            structural_features[1]
        )

        goal = (
            structural_features[2]
        )

        risk = (
            structural_features[3]
        )

        # intensity
        state_vector[0] = intensity

        state_vector[1] = (
            1.0
            -
            intensity
        )

        # urgency
        state_vector[2] = urgency

        state_vector[3] = (
            1.0
            -
            urgency
        )

        # goal alignment
        #
        # Codificamos o sinal em canais independentes.
        state_vector[4] = max(
            goal,
            0.0,
        )

        state_vector[5] = max(
            -goal,
            0.0,
        )

        state_vector[6] = (
            1.0
            -
            abs(
                goal
            )
        )

        # risk
        state_vector[7] = risk

        state_vector[8] = (
            1.0
            -
            risk
        )

        state_norm = np.linalg.norm(
            state_vector
        )

        if state_norm > 0:

            state_vector /= (
                state_norm
            )

        # ----------------------------------------------------
        # PARTE 2 — TIPO DE ESTÍMULO
        # ----------------------------------------------------

        type_slots = (

            self.signature_dimension
            -
            state_slots

        )

        type_vector = np.zeros(
            type_slots,
            dtype=np.float64,
        )

        type_index = (
            self._stable_type_index(
                stimulus_type,
                type_slots,
            )
        )

        type_vector[
            type_index
        ] = 1.0

        # ----------------------------------------------------
        # COMBINAÇÃO
        # ----------------------------------------------------
        #
        # sqrt(weight) é usado porque:
        #
        # cosine(concat)
        #
        # passa a refletir aproximadamente:
        #
        # state_weight * sim_estado
        #
        # +
        #
        # type_weight * sim_tipo
        # ----------------------------------------------------

        signature = np.zeros(
            self.signature_dimension,
            dtype=np.float64,
        )

        signature[
            :state_slots
        ] = (

            math.sqrt(
                self.state_weight
            )

            *

            state_vector

        )

        signature[
            state_slots:
        ] = (

            math.sqrt(
                self.type_weight
            )

            *

            type_vector

        )

        total_norm = (
            np.linalg.norm(
                signature
            )
        )

        if total_norm > 0:

            signature /= (
                total_norm
            )

        return signature

    # ========================================================
    # HASH ESTÁVEL DO TIPO
    # ========================================================

    @staticmethod
    def _stable_type_index(
        stimulus_type: str,
        slots: int,
    ) -> int:
        """
        Python hash() muda entre execuções.

        Portanto não pode ser usado para memória persistente.

        Criamos hash determinístico com BLAKE2b.
        """

        import hashlib

        digest = hashlib.blake2b(
            stimulus_type
            .strip()
            .lower()
            .encode(
                "utf-8"
            ),
            digest_size=8,
        ).digest()

        integer = int.from_bytes(
            digest,
            byteorder="big",
            signed=False,
        )

        return (
            integer
            %
            slots
        )

    # ========================================================
    # SIMILARIDADE COSSENO
    # ========================================================

    @staticmethod
    def _cosine_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        sim(a,b) = (a·b) / (||a|| ||b||)
        """

        denominator = (

            np.linalg.norm(
                a
            )

            *

            np.linalg.norm(
                b
            )

        )

        if denominator == 0.0:

            return 0.0

        similarity = float(

            np.dot(
                a,
                b,
            )

            /
            denominator

        )

        return clamp(
            similarity,
            0.0,
            1.0,
        )

    # ========================================================
    # EXTRAÇÃO E NORMALIZAÇÃO
    # ========================================================

    @staticmethod
    def _extract_structural_features(
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Aceita:

        vetor cognitivo completo:

            [
                intensity,
                novelty,
                urgency,
                goal_alignment,
                risk
            ]

        ou vetor estrutural:

            [
                intensity,
                urgency,
                goal_alignment,
                risk
            ]

        A novidade NÃO faz parte da assinatura utilizada
        para calcular a própria novidade.
        """

        array = np.asarray(
            features,
            dtype=np.float64,
        ).reshape(
            -1
        )

        if array.size == 5:

            return np.asarray(
                [
                    array[0],
                    array[2],
                    array[3],
                    array[4],
                ],
                dtype=np.float64,
            )

        if array.size == 4:

            return array.copy()

        raise ValueError(

            "features deve possuir "
            "4 características estruturais "
            "ou 5 características cognitivas."

        )

    @staticmethod
    def _normalize_structural_features(
        features: np.ndarray,
    ) -> np.ndarray:

        array = np.asarray(
            features,
            dtype=np.float64,
        ).copy()

        if array.size != 4:

            raise ValueError(
                "Vetor estrutural deve "
                "possuir 4 dimensões."
            )

        array[0] = clamp(
            array[0],
            0.0,
            1.0,
        )

        array[1] = clamp(
            array[1],
            0.0,
            1.0,
        )

        array[2] = clamp(
            array[2],
            -1.0,
            1.0,
        )

        array[3] = clamp(
            array[3],
            0.0,
            1.0,
        )

        return array

    # ========================================================
    # PERSISTÊNCIA
    # ========================================================

    def save(
        self,
    ) -> None:

        with self._lock:

            self._save_locked()

    def _save_locked(
        self,
    ) -> None:
        """
        Escrita atômica.

        Primeiro salvamos em arquivo temporário.

        Depois substituímos o original.

        Isso reduz o risco de corrupção caso o programa
        seja encerrado durante a escrita.
        """

        self.updated_at = (
            utc_now()
        )

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {

            "schema_version":
                MEMORY_SCHEMA_VERSION,

            "created_at":
                self.created_at,

            "updated_at":
                self.updated_at,

            "signature_dimension":
                self.signature_dimension,

            "patterns": [

                pattern.to_dict()

                for pattern
                in self._patterns

            ],

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

    # ========================================================
    # CARREGAMENTO
    # ========================================================

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
                f"a memória em "
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
            MEMORY_SCHEMA_VERSION
        ):

            raise RuntimeError(

                "Versão incompatível da memória. "
                f"Encontrada: {schema_version}. "
                f"Esperada: "
                f"{MEMORY_SCHEMA_VERSION}."

            )

        stored_dimension = int(

            data.get(
                "signature_dimension",
                self.signature_dimension,
            )

        )

        if (
            stored_dimension
            !=
            self.signature_dimension
        ):

            raise RuntimeError(

                "A dimensão das assinaturas "
                "salvas é diferente da atual."

            )

        self.created_at = data.get(
            "created_at",
            self.created_at,
        )

        self.updated_at = data.get(
            "updated_at",
            self.updated_at,
        )

        self._patterns = [

            PatternRecord.from_dict(
                pattern
            )

            for pattern
            in data.get(
                "patterns",
                []
            )

        ]

    # ========================================================
    # RECUPERAÇÃO INTERNA
    # ========================================================

    def _get_pattern_locked(
        self,
        pattern_id: str,
    ) -> PatternRecord | None:

        for pattern in self._patterns:

            if (
                pattern.id
                ==
                pattern_id
            ):

                return pattern

        return None

    # ========================================================
    # SNAPSHOT DO "CÉREBRO"
    # ========================================================

    def snapshot(
        self,
    ) -> dict[str, Any]:
        """
        Estado resumido da memória.

        Esse método será usado posteriormente pelo
        monitor visual no terminal.
        """

        with self._lock:

            total_exposures = sum(

                pattern.frequency

                for pattern
                in self._patterns

            )

            total_outcomes = sum(

                pattern.outcome_count

                for pattern
                in self._patterns

            )

            patterns = [

                {

                    "id":
                        pattern.id,

                    "type":
                        pattern.stimulus_type,

                    "frequency":
                        pattern.frequency,

                    "valence":
                        pattern.average_valence,

                    "importance":
                        pattern.importance_weight,

                    "outcomes":
                        pattern.outcome_count,

                    "actions":
                        {

                            action:
                                stats.mean_outcome

                            for action, stats
                            in pattern.action_stats.items()

                        },

                }

                for pattern
                in self._patterns

            ]

        return {

            "storage":
                str(
                    self.storage_path
                ),

            "patterns":
                patterns,

            "pattern_count":
                len(
                    patterns
                ),

            "total_exposures":
                total_exposures,

            "total_outcomes":
                total_outcomes,

            "created_at":
                self.created_at,

            "updated_at":
                self.updated_at,

        }