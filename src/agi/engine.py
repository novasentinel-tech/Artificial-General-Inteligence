from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .decision import DecisionModule
from .features import extract_features
from .memory import (
    PatternMatch,
    PatternMemory,
)
from .models import (
    ActionCandidate,
    DecisionResult,
    OutcomeEvent,
    StimulusEvent,
    ValenceResult,
)
from .triggers import (
    Trigger,
    TriggerResolution,
    TriggerSystem,
)
from .valence import (
    ValenceEngine,
    ValenceLearningUpdate,
)


# ============================================================
# RESULTADO DE UM CICLO COGNITIVO
# ============================================================

@dataclass
class CognitiveStep:
    """
    Snapshot completo de um ciclo cognitivo EVP.

    Permite observar:

        estímulo
            ↓
        novidade
            ↓
        φ(E)
            ↓
        valência
            ↓
        padrões recuperados
            ↓
        gatilhos
            ↓
        decisão

    Um estímulo pode ser filtrado antes da avaliação completa.
    Nesse caso vários campos permanecem None/vazios.
    """

    cycle: int

    stimulus: StimulusEvent

    filtered: bool

    filter_reason: str | None

    novelty: float | None

    features: list[float]

    valence: ValenceResult | None

    patterns: list[PatternMatch]

    triggers: list[Trigger]

    trigger_resolution: TriggerResolution | None

    decision: DecisionResult | None

    decision_path: str | None

    pattern_id: str | None


# ============================================================
# EXPERIÊNCIA PENDENTE
# ============================================================

@dataclass
class ExperienceRecord:
    """
    Experiência que já gerou uma ação, mas cuja consequência
    real ainda não foi observada.

    Guardamos o vetor original φ(E).

    Isso é fundamental.

    Depois de processar o estímulo, a memória pode mudar.
    Portanto NÃO devemos recalcular novidade/features
    quando a consequência chegar.
    """

    stimulus: StimulusEvent

    cycle: int

    novelty: float

    features: np.ndarray

    valence: ValenceResult

    patterns: list[PatternMatch]

    decision: DecisionResult

    decision_path: str

    pattern_ids: list[str]


# ============================================================
# RESULTADO DO FEEDBACK
# ============================================================

@dataclass(frozen=True)
class FeedbackResult:
    """
    Resultado do aprendizado causado por uma consequência.
    """

    stimulus_id: str

    action_id: str

    outcome: float

    prediction_error: float

    weight_delta: list[float]

    updated_pattern_ids: tuple[str, ...]


# ============================================================
# COGNITIVE ENGINE
# ============================================================

class CognitiveEngine:
    """
    Orquestrador central da arquitetura EVP.

    ==========================================================
    CICLO
    ==========================================================

        StimulusEvent
             │
             ▼
        filtro sensorial
             │
             ▼
        Pattern Memory
        calcula novidade
             │
             ▼
           φ(E)
             │
             ▼
        Valence Engine
             │
             ▼
       recuperação histórica
             │
             ▼
        formação/consolidação
           de padrão
             │
             ▼
        Trigger System
             │
             ├───────────────┐
             │               │
        automático?          não
             │               │
             ▼               ▼
       ação automática   DecisionModule
                             │
                             ▼
                          U(a|E)

             └───────┬───────┘
                     ▼
                 ação escolhida
                     │
                     ▼
                OutcomeEvent
                     │
             ┌───────┴────────┐
             ▼                ▼
        Valence Learn    Pattern Outcome
             │                │
             └───────┬────────┘
                     ▼
               próximo ciclo


    ==========================================================
    PRINCÍPIO TEMPORAL
    ==========================================================

    process(...)

        NÃO conhece a consequência futura.

    feedback(...)

        ocorre apenas depois da ação.

    Dessa forma não existe vazamento de informação entre
    decisão e aprendizado.
    """

    def __init__(
        self,
        memory: PatternMemory | None = None,
        valence: ValenceEngine | None = None,
        triggers: TriggerSystem | None = None,
        decisions: DecisionModule | None = None,
        sensory_threshold: float | None = 0.20,
    ) -> None:

        self.memory = (
            memory
            if memory is not None
            else PatternMemory()
        )

        self.valence = (
            valence
            if valence is not None
            else ValenceEngine()
        )

        self.triggers = (
            triggers
            if triggers is not None
            else TriggerSystem()
        )

        self.decisions = (
            decisions
            if decisions is not None
            else DecisionModule()
        )

        # ----------------------------------------------------
        # FILTRO PRÉ-ATENCIONAL
        # ----------------------------------------------------
        #
        # O artigo propõe que estímulos de baixa intensidade
        # podem ser filtrados antes da avaliação completa.
        #
        # O valor 0.20 vem desse cenário experimental,
        # mas permanece configurável.
        # ----------------------------------------------------

        if sensory_threshold is not None:

            if not 0.0 <= sensory_threshold <= 1.0:

                raise ValueError(
                    "sensory_threshold deve estar entre 0 e 1."
                )

        self.sensory_threshold = (
            sensory_threshold
        )

        # ----------------------------------------------------
        # EXPERIÊNCIAS AGUARDANDO CONSEQUÊNCIA
        # ----------------------------------------------------

        self._pending_experiences: dict[
            str,
            ExperienceRecord
        ] = {}

        # Quantidade total de estímulos recebidos.
        self.cycle_count: int = 0

    # ========================================================
    # CICLO COGNITIVO
    # ========================================================

    def process(
        self,
        event: StimulusEvent,
        actions: Sequence[ActionCandidate],
    ) -> CognitiveStep:
        """
        Processa um estímulo até produzir uma resposta.

        Nenhum aprendizado por consequência ocorre aqui.
        """

        self._validate_event(
            event
        )

        self.cycle_count += 1

        cycle = (
            self.cycle_count
        )

        # ====================================================
        # 1. FILTRO PRÉ-ATENCIONAL
        # ====================================================

        if self._should_filter(
            event
        ):

            return CognitiveStep(

                cycle=
                    cycle,

                stimulus=
                    event,

                filtered=
                    True,

                filter_reason=(
                    "Stimulus intensity below "
                    "pre-attentional threshold."
                ),

                novelty=
                    None,

                features=
                    [],

                valence=
                    None,

                patterns=
                    [],

                triggers=
                    [],

                trigger_resolution=
                    None,

                decision=
                    None,

                decision_path=
                    None,

                pattern_id=
                    None,

            )

        # O cérebro precisa possuir um espaço de ações
        # explícito para produzir comportamento.
        actions = list(
            actions
        )

        self._validate_action_space(
            actions
        )

        # ====================================================
        # 2. NOVIDADE
        # ====================================================
        #
        # N(E) =
        #
        #   1
        #   -
        #   max sim(
        #       σ(E),
        #       σ(P_i)
        #   )
        #
        # O cálculo ocorre ANTES de inserir a experiência
        # atual na memória.
        #
        # Caso contrário o estímulo seria comparado consigo
        # mesmo e sua novidade tenderia artificialmente a zero.
        # ====================================================

        novelty = float(

            self.memory.estimate_novelty(
                event
            )

        )

        novelty = self._clamp(
            novelty,
            0.0,
            1.0,
        )

        # ====================================================
        # 3. FEATURE VECTOR
        # ====================================================
        #
        # φ(E) =
        #
        # [
        #   intensity,
        #   novelty,
        #   urgency,
        #   goal_alignment,
        #   risk
        # ]
        # ====================================================

        features = extract_features(
            event=event,
            novelty=novelty,
        )

        # ====================================================
        # 4. RECUPERAÇÃO DE MEMÓRIA ANTERIOR
        # ====================================================
        #
        # Buscamos os padrões ANTES de registrar o evento atual.
        #
        # Portanto DecisionModule recebe apenas história
        # anterior ao momento atual.
        #
        # Isso evita um ciclo artificial:
        #
        #     evento atual
        #       ↓
        #     memória
        #       ↓
        #     decisão influenciada pelo próprio evento
        #     como se ele já fosse experiência passada.
        # ====================================================

        patterns = self.memory.find_similar(

            stimulus_type=
                event.type,

            features=
                features,

        )

        # ====================================================
        # 5. VALÊNCIA
        # ====================================================
        #
        # V(E) =
        #
        # tanh(
        #     wᵀ φ(E)
        # )
        #
        # Neste momento V(E) é uma expectativa interna.
        #
        # A consequência real ainda não existe.
        # ====================================================

        valence = self.valence.evaluate(

            event=
                event,

            features=
                features,

        )

        # ====================================================
        # 6. FORMAÇÃO / CONSOLIDAÇÃO DO PADRÃO ATUAL
        # ====================================================
        #
        # O padrão é registrado com a valência PREDITA atual.
        #
        # A consequência REAL será acrescentada somente
        # posteriormente em feedback().
        # ====================================================

        current_pattern = self.memory.remember(

            stimulus_type=
                event.type,

            features=
                features,

            valence=
                valence.score,

        )

        pattern_id = str(
            current_pattern.id
        )

        # ====================================================
        # 7. TRIGGER SYSTEM
        # ====================================================

        fired_triggers = self.triggers.evaluate(

            event=
                event,

            valence=
                valence,

            cycle=
                cycle,

        )

        trigger_resolution = (
            self.triggers.resolve(
                fired_triggers
            )
        )

        # ====================================================
        # 8. SELEÇÃO DO CAMINHO COMPORTAMENTAL
        # ====================================================

        automatic_action = (
            self._resolve_automatic_action(
                resolution=trigger_resolution,
                actions=actions,
            )
        )

        # ----------------------------------------------------
        # CAMINHO AUTOMÁTICO
        # ----------------------------------------------------

        if automatic_action is not None:

            dominant_trigger = (
                trigger_resolution.trigger
            )

            if dominant_trigger is None:

                raise RuntimeError(
                    "Trigger resolution inconsistente: "
                    "automatic_action sem trigger dominante."
                )

            decision = DecisionResult(

                action=
                    automatic_action.name,

                score=
                    float(
                        dominant_trigger.magnitude
                    ),

                reason=(

                    "automatic_trigger; "
                    f"trigger={dominant_trigger.rule_id}; "
                    f"valence={dominant_trigger.valence:+.4f}; "
                    f"magnitude={dominant_trigger.magnitude:.4f}; "
                    f"priority={dominant_trigger.priority}; "
                    f"action={automatic_action.name}"

                ),

                components={

                    "automatic":
                        1.0,

                    "trigger_valence":
                        float(
                            dominant_trigger.valence
                        ),

                    "trigger_magnitude":
                        float(
                            dominant_trigger.magnitude
                        ),

                    "trigger_priority":
                        float(
                            dominant_trigger.priority
                        ),

                },

            )

            decision_path = (
                "automatic"
            )

        # ----------------------------------------------------
        # CAMINHO DELIBERATIVO
        # ----------------------------------------------------

        else:

            trigger_ids = [

                trigger.rule_id

                for trigger
                in fired_triggers

            ]

            decision = self.decisions.choose(

                event=
                    event,

                valence=
                    valence,

                patterns=
                    patterns,

                actions=
                    actions,

                triggered_rules=
                    trigger_ids,

            )

            # Se um trigger existiu, mas não havia resposta
            # automática configurada/permitida, registramos
            # essa condição explicitamente.
            if (
                trigger_resolution
                .requires_automatic_path
            ):

                decision_path = (
                    "deliberative_after_unresolved_trigger"
                )

            else:

                decision_path = (
                    "deliberative"
                )

        # ====================================================
        # 9. EXPERIÊNCIA PENDENTE
        # ====================================================
        #
        # A partir daqui:
        #
        #     estímulo
        #     decisão
        #     ação
        #
        # já existem.
        #
        # Mas:
        #
        #     consequência
        #
        # ainda não.
        #
        # Guardamos exatamente o estado cognitivo usado
        # durante a decisão.
        # ====================================================

        experience = ExperienceRecord(

            stimulus=
                event,

            cycle=
                cycle,

            novelty=
                novelty,

            features=
                features.copy(),

            valence=
                valence,

            patterns=
                list(
                    patterns
                ),

            decision=
                decision,

            decision_path=
                decision_path,

            pattern_ids=[
                pattern_id
            ],

        )

        self._pending_experiences[
            event.id
        ] = experience

        # ====================================================
        # 10. RESULTADO DO CICLO
        # ====================================================

        return CognitiveStep(

            cycle=
                cycle,

            stimulus=
                event,

            filtered=
                False,

            filter_reason=
                None,

            novelty=
                novelty,

            features=
                features.tolist(),

            valence=
                valence,

            patterns=
                list(
                    patterns
                ),

            triggers=
                list(
                    fired_triggers
                ),

            trigger_resolution=
                trigger_resolution,

            decision=
                decision,

            decision_path=
                decision_path,

            pattern_id=
                pattern_id,

        )

    # ========================================================
    # FEEDBACK FORMAL
    # ========================================================

    def feedback(
        self,
        outcome_event: OutcomeEvent,
    ) -> FeedbackResult:
        """
        Recebe a consequência real após a execução da ação.

        Esse é o segundo momento fundamental do aprendizado:

            experiência
                ↓
            ação
                ↓
            resultado real r
                ↓
            erro de predição
                ↓
            alteração de pesos
                +
            alteração da memória
        """

        experience = (
            self._pending_experiences.get(
                outcome_event.stimulus_id
            )
        )

        if experience is None:

            raise KeyError(

                "Nenhuma experiência pendente encontrada "
                f"para stimulus_id="
                f"'{outcome_event.stimulus_id}'."

            )

        # ----------------------------------------------------
        # GARANTIA CAUSAL
        # ----------------------------------------------------
        #
        # Não permitimos atribuir ao agente a consequência
        # de uma ação diferente daquela realmente escolhida.
        # ----------------------------------------------------

        selected_action = (
            experience.decision.action
        )

        if (
            outcome_event.action_id
            !=
            selected_action
        ):

            raise ValueError(

                "OutcomeEvent.action_id não corresponde "
                "à ação escolhida durante o ciclo. "
                f"Esperado='{selected_action}', "
                f"recebido='{outcome_event.action_id}'."

            )

        outcome = float(
            outcome_event.outcome
        )

        # ====================================================
        # 1. VALENCE LEARNING
        # ====================================================
        #
        # Utilizamos φ(E) ORIGINAL.
        #
        # NÃO recalculamos novelty.
        #
        # Entre process() e feedback(), a memória já recebeu
        # esse estímulo, portanto recalcular novidade produziria
        # um vetor diferente daquele usado durante a decisão.
        # ====================================================

        update: ValenceLearningUpdate = (
            self.valence.learn(

                features=
                    experience.features,

                outcome=
                    outcome,

            )
        )

        # ====================================================
        # 2. MEMÓRIA DA CONSEQUÊNCIA
        # ====================================================
        #
        # Pattern
        #    +
        # Action
        #    +
        # Outcome
        # ====================================================

        updated_patterns: list[str] = []

        for pattern_id in (
            experience.pattern_ids
        ):

            self.memory.record_outcome(

                pattern_id=
                    pattern_id,

                action_id=
                    selected_action,

                outcome=
                    outcome,

            )

            updated_patterns.append(
                pattern_id
            )

        # ====================================================
        # 3. FINALIZA EXPERIÊNCIA
        # ====================================================

        self._pending_experiences.pop(
            outcome_event.stimulus_id,
            None,
        )

        return FeedbackResult(

            stimulus_id=
                outcome_event.stimulus_id,

            action_id=
                selected_action,

            outcome=
                outcome,

            prediction_error=
                float(
                    update.prediction_error
                ),

            weight_delta=
                list(
                    update.delta
                ),

            updated_pattern_ids=
                tuple(
                    updated_patterns
                ),

        )

    # ========================================================
    # COMPATIBILIDADE COM demo.py ANTIGO
    # ========================================================

    def learn_from_outcome(
        self,
        event: StimulusEvent,
        novelty: float | None = None,
        outcome: float = 0.0,
    ) -> FeedbackResult:
        """
        Interface de compatibilidade.

        O parâmetro novelty é mantido temporariamente para que
        o demo.py antigo não quebre.

        Ele NÃO é utilizado quando existe uma experiência
        pendente, porque o engine já armazenou o φ(E) original.

        A API preferida daqui para frente é:

            engine.feedback(
                OutcomeEvent(...)
            )
        """

        experience = (
            self._pending_experiences.get(
                event.id
            )
        )

        if experience is None:

            raise KeyError(

                "learn_from_outcome() exige uma experiência "
                "gerada previamente por process(). "
                f"Nenhum registro pendente para '{event.id}'."

            )

        feedback_event = OutcomeEvent(

            stimulus_id=
                event.id,

            action_id=
                experience.decision.action,

            outcome=
                outcome,

            context_id=
                event.context_id,

        )

        return self.feedback(
            feedback_event
        )

    # ========================================================
    # RESOLUÇÃO DE AÇÃO AUTOMÁTICA
    # ========================================================

    @staticmethod
    def _resolve_automatic_action(
        resolution: TriggerResolution,
        actions: Sequence[ActionCandidate],
    ) -> ActionCandidate | None:
        """
        Um Trigger pode solicitar uma resposta automática.

        Porém o TriggerSystem NÃO possui autoridade para criar
        ações que não pertencem ao espaço disponível.

        Isso é importante por segurança arquitetural.

        Portanto:

            trigger diz:
                "execute X"

            engine verifica:
                "X existe entre as ações permitidas?"

        somente então X pode ser selecionada.
        """

        if not (
            resolution
            .requires_automatic_path
        ):

            return None

        action_name = (
            resolution.response_action
        )

        # Trigger foi ativado, mas nenhuma resposta concreta
        # foi vinculada à regra.
        #
        # Nesse caso não inventamos uma ação.
        if action_name is None:

            return None

        for action in actions:

            if (
                action.name
                ==
                action_name
            ):

                return action

        # Resposta automática solicitada não existe no
        # espaço atual de ações.
        #
        # Falhamos de forma segura:
        #
        # não executamos ação inexistente e permitimos que
        # o DecisionModule escolha entre ações válidas.
        return None

    # ========================================================
    # FILTRO SENSORIAL
    # ========================================================

    def _should_filter(
        self,
        event: StimulusEvent,
    ) -> bool:

        if (
            self.sensory_threshold
            is None
        ):

            return False

        return (
            event.intensity
            <
            self.sensory_threshold
        )

    # ========================================================
    # VALIDAÇÃO DO ESTÍMULO
    # ========================================================

    @staticmethod
    def _validate_event(
        event: StimulusEvent,
    ) -> None:

        if not event.type.strip():

            raise ValueError(
                "StimulusEvent.type não pode estar vazio."
            )

    # ========================================================
    # VALIDAÇÃO DO ESPAÇO DE AÇÕES
    # ========================================================

    @staticmethod
    def _validate_action_space(
        actions: Sequence[ActionCandidate],
    ) -> None:
        """
        O espaço comportamental precisa ser explícito.

        Também impedimos nomes duplicados, pois eles tornariam
        o feedback ambíguo:

            qual "avoid" realmente foi executado?
        """

        if not actions:

            raise ValueError(
                "O agente não possui nenhuma ação disponível."
            )

        names = [

            action.name

            for action
            in actions

        ]

        if (
            len(names)
            !=
            len(
                set(names)
            )
        ):

            raise ValueError(
                "ActionCandidate.name deve ser único "
                "dentro do espaço de ações."
            )

    # ========================================================
    # EXPERIÊNCIAS PENDENTES
    # ========================================================

    @property
    def pending_experiences(
        self,
    ) -> tuple[str, ...]:
        """
        IDs dos estímulos que aguardam consequência.
        """

        return tuple(
            self._pending_experiences.keys()
        )

    def get_pending_experience(
        self,
        stimulus_id: str,
    ) -> ExperienceRecord | None:

        return (
            self._pending_experiences.get(
                stimulus_id
            )
        )

    # ========================================================
    # ESTADO INTERNO
    # ========================================================

    def internal_state(
        self,
    ) -> dict[str, Any]:
        """
        Estado observável do organismo artificial.

        Isso ainda NÃO é o Self-Monitor.

        É apenas a interface que o Self-Monitor utilizará
        futuramente para transformar estados internos em:

            StimulusEvent(
                source=SELF
            )
        """

        return {

            "cognitive_cycles":
                self.cycle_count,

            "pending_experiences":
                len(
                    self._pending_experiences
                ),

            "memory":
                self.memory.snapshot(),

            "valence":
                self.valence.snapshot(),

            "triggers":
                self.triggers.snapshot(),

            "decision":
                self.decisions.snapshot(),

        }

    # ========================================================
    # UTILIDADE
    # ========================================================

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:

        return max(
            minimum,
            min(
                maximum,
                float(
                    value
                ),
            ),
        )