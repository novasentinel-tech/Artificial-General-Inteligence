from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

import math

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# ============================================================
# UTILIDADES
# ============================================================


def utc_now() -> datetime:
    """
    Retorna um timestamp timezone-aware em UTC.

    Todos os eventos cognitivos utilizam uma referência
    temporal comum para impedir ambiguidades entre ambientes
    ou máquinas diferentes.
    """

    return datetime.now(
        timezone.utc
    )


# ============================================================
# CONFIGURAÇÃO BASE DOS MODELOS
# ============================================================


class EVPModel(BaseModel):
    """
    Classe base para os schemas do sistema EVP.

    extra="forbid":
        impede campos desconhecidos de entrarem silenciosamente.

    validate_assignment=True:
        valida alterações feitas depois da criação do objeto.

    str_strip_whitespace=True:
        remove espaços acidentais no início/fim de strings.

    allow_inf_nan=False:
        impede NaN e infinito em valores numéricos.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


# ============================================================
# ORIGEM DO ESTÍMULO
# ============================================================


class StimulusSource(str, Enum):
    """
    Fonte funcional de um estímulo.

    EXTERNAL
        Informação proveniente do ambiente.

    INTERNAL
        Estado interno produzido pelo próprio sistema.

    SELF
        Representação explícita do estado do próprio agente.

    O tipo SELF será utilizado posteriormente pelo
    Self-Monitor para implementar o ciclo reflexivo:

        estado interno
            ↓
        StimulusEvent(source=SELF)
            ↓
        valência
            ↓
        memória
            ↓
        resposta
            ↓
        novo estado interno
    """

    EXTERNAL = "external"

    INTERNAL = "internal"

    SELF = "self"


# ============================================================
# ESTÍMULO
# ============================================================


class StimulusEvent(EVPModel):
    """
    Unidade fundamental de entrada do sistema.

    Formalmente:

        E = <t, s, p, τ, I>

    onde:

        t
            tipo do estímulo

        s
            fonte

        p
            payload

        τ
            timestamp

        I
            intensidade normalizada


    IMPORTANTE:

    O estímulo NÃO possui valência embutida.

    A valência é uma avaliação produzida internamente
    posteriormente pelo Valence Engine.
    """

    id: str = Field(
        default_factory=lambda: str(
            uuid4()
        )
    )

    source: StimulusSource

    type: str = Field(
        min_length=1,
        max_length=128,
    )

    payload: dict[str, Any] = Field(
        default_factory=dict
    )

    # I ∈ [0,1]
    intensity: float = Field(
        ge=0.0,
        le=1.0,
    )

    timestamp: datetime = Field(
        default_factory=utc_now
    )

    # Permite agrupar estímulos pertencentes ao mesmo
    # contexto/episódio.
    context_id: str | None = None

    # --------------------------------------------------------
    # NORMALIZAÇÃO DO TIPO
    # --------------------------------------------------------

    @field_validator(
        "type"
    )
    @classmethod
    def normalize_type(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value
            .strip()
            .lower()
        )

        if not normalized:

            raise ValueError(
                "StimulusEvent.type não pode estar vazio."
            )

        return normalized

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    @field_validator(
        "timestamp"
    )
    @classmethod
    def ensure_utc_timestamp(
        cls,
        value: datetime,
    ) -> datetime:

        if value.tzinfo is None:

            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )


# ============================================================
# RESULTADO DA VALÊNCIA
# ============================================================


class ValenceResult(EVPModel):
    """
    Resultado produzido pelo Valence Engine.

    Formalmente:

        V(E) = tanh(wᵀφ(E))

    com:

        V(E) ∈ [-1,+1]


    score:

        -1
            valência extremamente negativa

         0
            estado neutro

        +1
            valência extremamente positiva


    confidence:

        confiança operacional da avaliação.

    IMPORTANTE:

        confidence != valence

    Exemplo:

        score = -0.90
        confidence = 0.10

    significa:

        "a avaliação atual é fortemente negativa,
         mas o agente ainda possui pouca experiência
         para confiar nessa estimativa."
    """

    stimulus_id: str = Field(
        min_length=1
    )

    score: float = Field(
        ge=-1.0,
        le=1.0,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    # Contribuição de cada dimensão:
    #
    # w_i * φ_i(E)
    components: dict[str, float] = Field(
        default_factory=dict
    )

    # z = wᵀφ(E)
    #
    # antes:
    #
    # V(E) = tanh(z)
    raw_activation: float | None = None

    @field_validator(
        "components"
    )
    @classmethod
    def validate_components(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:

        for name, component in value.items():

            numeric = float(
                component
            )

            if not math.isfinite(
                numeric
            ):

                raise ValueError(
                    "ValenceResult.components contém "
                    f"valor inválido em '{name}'."
                )

        return value


# ============================================================
# AÇÃO CANDIDATA
# ============================================================


class ActionCandidate(EVPModel):
    """
    Ação disponível ao agente em determinado ciclo.

    Essa classe representa o espaço comportamental possível.

    ==========================================================
    EQUAÇÃO DE DECISÃO DO EVP
    ==========================================================

        U(a|E) =

            O(a)
            *
            (
                αV(E)
                +
                βv̄_memory
            )

            -

            C(a)


    portanto esta estrutura possui diretamente:

        O(a)
            expected_outcome

        C(a)
            cost


    ==========================================================
    expected_outcome
    ==========================================================

        O(a) ∈ [-1,+1]

    É a expectativa associada ao resultado da ação.

    Mantemos exatamente o intervalo descrito pelo modelo
    técnico EVP.

    A interpretação matemática do sinal de O(a) será estudada
    formalmente nos experimentos, especialmente porque a
    multiplicação entre dois valores negativos pode produzir
    utilidade positiva.


    ==========================================================
    cost
    ==========================================================

        C(a) ∈ [0,1]

    Representa custo computacional, energético ou operacional
    da execução da ação.


    ==========================================================
    IMPORTANTE
    ==========================================================

    Removemos:

        base_utility
        risk_penalty

    porque eles pertenciam ao protótipo inicial e não fazem
    parte da função de decisão atualmente implementada.
    """

    name: str = Field(
        min_length=1,
        max_length=128,
    )

    # O(a)
    expected_outcome: float = Field(
        ge=-1.0,
        le=1.0,
    )

    # C(a)
    cost: float = Field(
        ge=0.0,
        le=1.0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator(
        "name"
    )
    @classmethod
    def normalize_name(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value
            .strip()
            .lower()
        )

        if not normalized:

            raise ValueError(
                "ActionCandidate.name não pode estar vazio."
            )

        return normalized


# ============================================================
# RESULTADO DA DECISÃO
# ============================================================


class DecisionResult(EVPModel):
    """
    Resultado comportamental produzido pelo sistema.

    A ação pode ser escolhida através de dois caminhos:

        deliberative

    ou:

        automatic trigger

    O DecisionResult não precisa saber qual módulo produziu
    a decisão.

    Essa informação é armazenada pelo CognitiveEngine em:

        CognitiveStep.decision_path


    score:

        NÃO é obrigatoriamente limitado a [-1,+1].

    No caminho deliberativo representa:

        U(a|E)

    No caminho automático pode representar, por exemplo,
    magnitude da valência que originou o trigger.
    """

    action: str = Field(
        min_length=1,
        max_length=128,
    )

    score: float

    reason: str = ""

    # Breakdown numérico da decisão.
    #
    # Exemplo:
    #
    # {
    #     "expected_outcome": 0.6,
    #     "current_valence": -0.5,
    #     "memory_valence": -0.4,
    #     "integrated_valence": -0.46,
    #     "cost": 0.1,
    #     "calculated_utility": -0.376
    # }
    components: dict[str, float] = Field(
        default_factory=dict
    )

    @field_validator(
        "action"
    )
    @classmethod
    def normalize_action(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value
            .strip()
            .lower()
        )

        if not normalized:

            raise ValueError(
                "DecisionResult.action não pode estar vazio."
            )

        return normalized

    @field_validator(
        "components"
    )
    @classmethod
    def validate_decision_components(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:

        for name, component in value.items():

            numeric = float(
                component
            )

            if not math.isfinite(
                numeric
            ):

                raise ValueError(
                    "DecisionResult.components contém "
                    f"valor inválido em '{name}'."
                )

        return value


# ============================================================
# CONSEQUÊNCIA REAL
# ============================================================


class OutcomeEvent(EVPModel):
    """
    Consequência observada APÓS uma ação.

    A sequência causal é:

        StimulusEvent
            ↓
        avaliação
            ↓
        decisão
            ↓
        ação
            ↓
        OutcomeEvent


    Portanto OutcomeEvent nunca deve existir cognitivamente
    antes da seleção da ação correspondente.


    outcome:

        r ∈ [-1,+1]

    onde:

        -1
            consequência extremamente negativa

         0
            consequência neutra

        +1
            consequência extremamente positiva


    Esse valor alimentará:

        Δw =
            η
            *
            (
                r - wᵀφ(E)
            )
            *
            φ(E)
    """

    id: str = Field(
        default_factory=lambda: str(
            uuid4()
        )
    )

    stimulus_id: str = Field(
        min_length=1
    )

    # Atualmente action_id corresponde ao nome normalizado
    # da ActionCandidate selecionada.
    action_id: str = Field(
        min_length=1,
        max_length=128,
    )

    outcome: float = Field(
        ge=-1.0,
        le=1.0,
    )

    timestamp: datetime = Field(
        default_factory=utc_now
    )

    context_id: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator(
        "action_id"
    )
    @classmethod
    def normalize_action_id(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value
            .strip()
            .lower()
        )

        if not normalized:

            raise ValueError(
                "OutcomeEvent.action_id não pode estar vazio."
            )

        return normalized

    @field_validator(
        "timestamp"
    )
    @classmethod
    def ensure_outcome_timestamp_utc(
        cls,
        value: datetime,
    ) -> datetime:

        if value.tzinfo is None:

            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )


# ============================================================
# OBJETIVO INTERNO
# ============================================================


class GoalState(EVPModel):
    """
    Representação preliminar de um objetivo do agente.

    Ainda NÃO constitui um sistema de planejamento.

    Ele existe porque:

        goal_alignment

    atualmente é fornecido durante a extração de features,
    mas futuramente deverá ser calculado comparando o estado
    atual do agente com seus objetivos ativos.

    Exemplo conceitual:

        objetivo:
            manter energy >= 0.30

        estado atual:
            energy = 0.12

        estímulo:
            resource_detected

        então:

            goal_alignment > 0

    O valor de alinhamento não deveria permanecer para sempre
    como um número escrito manualmente pelo ambiente.
    """

    id: str = Field(
        default_factory=lambda: str(
            uuid4()
        )
    )

    name: str = Field(
        min_length=1,
        max_length=256,
    )

    # Importância relativa do objetivo.
    priority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    active: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator(
        "name"
    )
    @classmethod
    def normalize_goal_name(
        cls,
        value: str,
    ) -> str:

        normalized = value.strip()

        if not normalized:

            raise ValueError(
                "GoalState.name não pode estar vazio."
            )

        return normalized


# ============================================================
# ESTADO COGNITIVO
# ============================================================


class CognitiveState(EVPModel):
    """
    Snapshot explícito de propriedades internas do agente.

    Isso NÃO deve ser confundido com consciência.

    É apenas uma representação computacional do estado
    observável do sistema.

    Posteriormente o Self-Monitor poderá transformar mudanças
    importantes deste estado em:

        StimulusEvent(
            source=StimulusSource.SELF
        )
    """

    cycle: int = Field(
        ge=0
    )

    known_patterns: int = Field(
        ge=0
    )

    learned_experiences: int = Field(
        ge=0
    )

    pending_experiences: int = Field(
        ge=0
    )

    active_goals: int = Field(
        default=0,
        ge=0,
    )

    current_valence: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )

    average_valence: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )

    timestamp: datetime = Field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator(
        "timestamp"
    )
    @classmethod
    def ensure_cognitive_timestamp_utc(
        cls,
        value: datetime,
    ) -> datetime:

        if value.tzinfo is None:

            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )