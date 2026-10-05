from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# ============================================================
# UTILIDADES TEMPORAIS
# ============================================================

def utc_now() -> datetime:
    """
    Retorna timestamp timezone-aware em UTC.

    Evitamos datetime.utcnow(), pois ele retorna um datetime
    sem timezone explícito.
    """

    return datetime.now(
        timezone.utc
    )


# ============================================================
# ORIGEM DO ESTÍMULO
# ============================================================

class StimulusSource(str, Enum):
    """
    Origem funcional de um estímulo.

    EXTERNAL
        Informação proveniente do ambiente.

        Exemplos:
        - visão
        - texto
        - sensores
        - APIs
        - eventos externos

    INTERNAL
        Estado produzido internamente pelo organismo artificial.

        Exemplos:
        - necessidade energética
        - erro de processamento
        - estado de memória
        - conflito entre objetivos

    SELF
        Informação referente à representação que o sistema
        mantém de si próprio.

        Essa categoria será particularmente importante quando
        implementarmos o Self-Monitor.

        Exemplos futuros:
        - "minha confiança está baixa"
        - "minha memória está saturando"
        - "meu objetivo mudou"
    """

    EXTERNAL = "external"
    INTERNAL = "internal"
    SELF = "self"


# ============================================================
# ESTÍMULO
# ============================================================

class StimulusEvent(BaseModel):
    """
    Unidade fundamental de entrada do sistema cognitivo.

    Formalmente:

        E = <t, s, p, τ, I>

    onde:

        t = tipo
        s = fonte
        p = payload
        τ = instante temporal
        I = intensidade

    O estímulo NÃO contém valência.

    Isso é proposital.

    A valência deve ser uma interpretação produzida pelo
    sistema após receber o estímulo, e não uma propriedade
    previamente conhecida da entrada.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

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

    intensity: float = Field(
        ge=0.0,
        le=1.0,
    )

    timestamp: datetime = Field(
        default_factory=utc_now
    )

    # Permite agrupar estímulos pertencentes
    # à mesma experiência ou episódio.
    context_id: str | None = None

    @field_validator(
        "type"
    )
    @classmethod
    def normalize_type(
        cls,
        value: str,
    ) -> str:
        """
        Evita tipos semanticamente iguais com diferenças
        puramente textuais.

        Exemplo:

            " Danger "
            "danger"

        tornam-se:

            "danger"
        """

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

    @field_validator(
        "timestamp"
    )
    @classmethod
    def ensure_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        """
        Garante consistência temporal.

        Se chegar um datetime sem timezone, assumimos UTC.
        """

        if value.tzinfo is None:

            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )


# ============================================================
# RESULTADO DE VALÊNCIA
# ============================================================

class ValenceResult(BaseModel):
    """
    Resultado produzido pelo Valence Engine.

    score:

        V(E) ∈ [-1, +1]

        -1 -> fortemente negativo
         0 -> neutro
        +1 -> fortemente positivo

    confidence:

        C(E) ∈ [0, 1]

        Representa quanto o sistema confia na estimativa.

        Essa confiança NÃO é a valência.

        Um sistema pode, por exemplo:

            valência = -0.90
            confiança = 0.15

        significando:

            "parece muito negativo,
             mas quase não tenho experiência suficiente
             para confiar nisso."
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    stimulus_id: str

    score: float = Field(
        ge=-1.0,
        le=1.0,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    # Decomposição interpretável da ativação.
    #
    # Exemplo:
    #
    # {
    #     "intensity": -0.10,
    #     "novelty": -0.30,
    #     "urgency": -0.20,
    #     "goal_alignment": 0.40,
    #     "risk": -0.55
    # }
    components: dict[str, float] = Field(
        default_factory=dict
    )

    # Ativação anterior à compressão por tanh.
    #
    # Se:
    #
    #     z = wᵀφ(E)
    #
    # então:
    #
    #     score = tanh(z)
    #
    # Vamos usar isso no valence.py definitivo.
    raw_activation: float | None = None


# ============================================================
# AÇÃO CANDIDATA
# ============================================================

class ActionCandidate(BaseModel):
    """
    Ação que o sistema pode escolher em determinado ciclo.

    O módulo de decisão NÃO cria ações arbitrariamente.

    Ele recebe um espaço de ações possíveis e seleciona
    aquela cuja expectativa de consequência é mais adequada.

    base_utility:

        preferência basal pela ação.

        Intervalo:
            [-1, +1]

        Não representa experiência aprendida.

        O histórico aprendido pertence à Pattern Memory.

    risk_penalty:

        sensibilidade da ação ao risco ambiental.

        Intervalo:
            [0, 1]

        Exemplo:

            investigate:
                risk_penalty = 0.8

            avoid:
                risk_penalty = 0.0
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(
        min_length=1,
        max_length=128,
    )

    base_utility: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
    )

    risk_penalty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    # Campo opcional para informações futuras.
    #
    # Exemplos:
    #
    # {
    #     "motor_command": "move_back",
    #     "energy_cost": 0.2
    # }
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

class DecisionResult(BaseModel):
    """
    Resposta produzida pelo Decision Module.

    decision.score NÃO está limitado a [-1, +1].

    Isso é intencional.

    Diferentemente da valência, que é normalizada por definição,
    o score de decisão representa uma função composta por vários
    fatores.

    Exemplo futuro:

        Q(a) =

            utilidade_base

            + alinhamento_com_objetivo

            + valência

            + experiência_histórica

            + gatilhos

            - penalidade_de_risco
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    action: str = Field(
        min_length=1,
        max_length=128,
    )

    score: float

    reason: str = ""

    # Futuramente podemos colocar aqui a decomposição completa
    # da função de decisão.
    components: dict[str, float] = Field(
        default_factory=dict
    )


# ============================================================
# FEEDBACK DO AMBIENTE
# ============================================================

class OutcomeEvent(BaseModel):
    """
    Representa a consequência observada APÓS uma ação.

    Essa separação é fundamental:

        estímulo
            ↓
        decisão
            ↓
        ação
            ↓
        consequência

    O agente não pode conhecer o outcome antes de agir.

    outcome:

        R ∈ [-1, +1]

        -1 -> consequência extremamente negativa
         0 -> consequência neutra
        +1 -> consequência extremamente positiva

    O OutcomeEvent vai permitir futuramente que o aprendizado
    seja desacoplado completamente do ambiente.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    id: str = Field(
        default_factory=lambda: str(
            uuid4()
        )
    )

    stimulus_id: str

    action_id: str

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


# ============================================================
# OBJETIVO COGNITIVO
# ============================================================

class GoalState(BaseModel):
    """
    Representação básica de um objetivo interno.

    Ainda NÃO estamos implementando planejamento.

    Esse modelo existe para preparar uma evolução importante:

        goal_alignment

    atualmente chega no payload do estímulo.

    No futuro, isso deve ser CALCULADO comparando o estado
    percebido com os objetivos ativos do sistema.

    Portanto GoalState prepara essa migração sem alterar
    a arquitetura fundamental.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    id: str = Field(
        default_factory=lambda: str(
            uuid4()
        )
    )

    name: str = Field(
        min_length=1,
        max_length=256,
    )

    priority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    active: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# ESTADO COGNITIVO OBSERVÁVEL
# ============================================================

class CognitiveState(BaseModel):
    """
    Snapshot observável do sistema.

    IMPORTANTE:

    Isso não significa consciência.

    É apenas uma representação computacional explícita
    do próprio estado interno.

    Posteriormente o Self-Monitor poderá converter mudanças
    nesse estado em estímulos SELF.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

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

    timestamp: datetime = Field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )