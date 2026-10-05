from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Sequence

import math

from .memory import PatternMatch
from .models import (
    ActionCandidate,
    DecisionResult,
    StimulusEvent,
    ValenceResult,
)


# ============================================================
# PARÂMETROS PADRÃO DA IMPLEMENTAÇÃO TÉCNICA
# ============================================================

DEFAULT_ALPHA = 0.60

DEFAULT_BETA = 0.40

DEFAULT_CONFIDENCE_THRESHOLD = 0.60

DEFAULT_EXPLORATION_RATE = 0.10


# ============================================================
# AVALIAÇÃO DE UMA AÇÃO
# ============================================================

@dataclass(frozen=True)
class ActionEvaluation:
    """
    Decomposição completa da avaliação de uma ação.

    Para cada ação a:

        U(a|E) =
            O(a)
            *
            (
                α * V(E)
                +
                β * v_memory
            )
            -
            C(a)

    Nenhum termo extra é adicionado silenciosamente.
    """

    action: ActionCandidate

    # O(a)
    expected_outcome: float

    # Indica de onde O(a) veio.
    expected_outcome_source: str

    # V(E), após aplicação do critério de confiança.
    current_valence: float

    # v̄_memory
    memory_valence: float

    # αV(E) + βv̄_memory
    integrated_valence: float

    # C(a)
    cost: float

    # U(a|E)
    utility: float


# ============================================================
# TRAÇO DA DECISÃO
# ============================================================

@dataclass(frozen=True)
class DecisionTrace:
    """
    Registro científico de uma decisão.

    Permite posteriormente investigar:

    - quais alternativas existiam;
    - quanto cada uma pontuou;
    - se houve exploração;
    - quais valores internos influenciaram a escolha.
    """

    stimulus_id: str

    mode: str

    exploration_rate: float

    random_value: float

    valence_score: float

    valence_confidence: float

    memory_valence: float

    evaluations: tuple[ActionEvaluation, ...]

    selected_action: str


# ============================================================
# DECISION MODULE
# ============================================================

class DecisionModule:
    """
    Camada deliberativa de seleção comportamental do EVP.

    ==========================================================
    IMPORTANTE
    ==========================================================

    Este módulo NÃO deve ser chamado quando um Trigger automático
    dominante já determinou uma resposta.

    O fluxo correto é:

        Valence
            ↓
        Trigger System
            ↓
        trigger automático?
          /        \\
        sim        não
         │          │
         ▼          ▼
      resposta   DecisionModule

    Portanto gatilhos NÃO recebem bônus nesta função de utilidade.


    ==========================================================
    EQUAÇÃO CENTRAL
    ==========================================================

        U(a|E) =
            O(a)
            *
            (
                α * V(E)
                +
                β * v̄_memory
            )
            -
            C(a)

    onde:

        O(a)
            resultado esperado da ação

        V(E)
            valência atual do estímulo

        v̄_memory
            valência histórica dos padrões semelhantes

        C(a)
            custo da ação

        α + β = 1


    ==========================================================
    EXPLORAÇÃO
    ==========================================================

    Utilizamos ε-greedy:

        P(explorar) = ε

    Isso impede que o agente fique preso permanentemente
    às primeiras respostas que funcionaram.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        confidence_threshold: float = (
            DEFAULT_CONFIDENCE_THRESHOLD
        ),
        exploration_rate: float = (
            DEFAULT_EXPLORATION_RATE
        ),
        random_seed: int | None = None,
        strict_action_schema: bool = True,
    ) -> None:

        self.alpha = float(
            alpha
        )

        self.beta = float(
            beta
        )

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.exploration_rate = float(
            exploration_rate
        )

        self.strict_action_schema = bool(
            strict_action_schema
        )

        self._rng = Random(
            random_seed
        )

        self.last_trace: DecisionTrace | None = None

        self._validate_configuration()

    # ========================================================
    # SELEÇÃO PRINCIPAL
    # ========================================================

    def choose(
        self,
        event: StimulusEvent,
        valence: ValenceResult,
        patterns: Sequence[PatternMatch],
        actions: Sequence[ActionCandidate],
        triggered_rules: Sequence[str] | None = None,
        exploration_rate: float | None = None,
    ) -> DecisionResult:
        """
        Seleciona uma resposta comportamental.

        Esta função representa SOMENTE o caminho deliberativo.

        Se existir um Trigger automático dominante,
        o CognitiveEngine deve interceptá-lo antes daqui.
        """

        actions = list(
            actions
        )

        patterns = list(
            patterns
        )

        if not actions:

            raise ValueError(
                "DecisionModule recebeu zero ações disponíveis."
            )

        self._validate_valence(
            valence
        )

        epsilon = (
            self.exploration_rate
            if exploration_rate is None
            else float(
                exploration_rate
            )
        )

        if not 0.0 <= epsilon <= 1.0:

            raise ValueError(
                "exploration_rate deve estar entre 0 e 1."
            )

        # ----------------------------------------------------
        # 1. VALÊNCIA ATUAL CONFIÁVEL
        # ----------------------------------------------------
        #
        # A implementação técnica do artigo utiliza V(E)
        # apenas quando a confiança atinge determinado limiar.
        #
        # Caso contrário:
        #
        #     V_effective(E) = 0
        #
        # Isso evita que uma avaliação ainda pouco sustentada
        # domine uma decisão deliberativa.
        # ----------------------------------------------------

        if (
            valence.confidence
            >=
            self.confidence_threshold
        ):

            current_valence = float(
                valence.score
            )

        else:

            current_valence = 0.0

        # ----------------------------------------------------
        # 2. VALÊNCIA DA MEMÓRIA
        # ----------------------------------------------------
        #
        # Conforme a arquitetura técnica:
        #
        # v_memory =
        #
        #   Σ(v_i × similarity_i)
        #   ---------------------
        #             n
        #
        # Não adicionamos frequency, importance etc. aqui,
        # pois eles não aparecem nesta equação do artigo.
        # ----------------------------------------------------

        memory_valence = (
            self._calculate_memory_valence(
                patterns
            )
        )

        # ----------------------------------------------------
        # 3. COMPONENTE INTEGRADO
        # ----------------------------------------------------
        #
        # B(E) =
        #
        #   αV(E)
        #   +
        #   βv_memory
        #
        # ----------------------------------------------------

        integrated_valence = (

            self.alpha
            *
            current_valence

            +

            self.beta
            *
            memory_valence

        )

        # ----------------------------------------------------
        # 4. CALCULA TODAS AS AÇÕES
        # ----------------------------------------------------

        evaluations = [

            self._evaluate_action(

                action=action,

                current_valence=current_valence,

                memory_valence=memory_valence,

                integrated_valence=integrated_valence,

            )

            for action
            in actions

        ]

        # ----------------------------------------------------
        # 5. ε-GREEDY
        # ----------------------------------------------------

        random_value = (
            self._rng.random()
        )

        exploring = (
            random_value
            <
            epsilon
        )

        if exploring:

            selected = (
                self._rng.choice(
                    evaluations
                )
            )

            # A implementação original retorna score = 0
            # durante exploração.
            #
            # A utilidade real continua disponível no trace
            # e nos componentes para auditoria.
            result_score = 0.0

            mode = "exploration"

        else:

            # max() preserva a primeira ação em caso de empate.
            #
            # Isso evita adicionar aleatoriedade escondida
            # fora do mecanismo ε-greedy.
            selected = max(

                evaluations,

                key=lambda item:
                    item.utility,

            )

            result_score = (
                selected.utility
            )

            mode = "exploitation"

        # ----------------------------------------------------
        # 6. RATIONALE
        # ----------------------------------------------------

        triggered_rules = list(
            triggered_rules
            or
            []
        )

        reason = (
            self._build_rationale(
                event=event,
                valence=valence,
                selected=selected,
                mode=mode,
                triggered_rules=triggered_rules,
            )
        )

        # ----------------------------------------------------
        # 7. TRACE COMPLETO
        # ----------------------------------------------------

        self.last_trace = DecisionTrace(

            stimulus_id=
                event.id,

            mode=
                mode,

            exploration_rate=
                epsilon,

            random_value=
                random_value,

            valence_score=
                float(
                    valence.score
                ),

            valence_confidence=
                float(
                    valence.confidence
                ),

            memory_valence=
                memory_valence,

            evaluations=
                tuple(
                    evaluations
                ),

            selected_action=
                selected.action.name,

        )

        # ----------------------------------------------------
        # 8. RESULTADO
        # ----------------------------------------------------

        return DecisionResult(

            action=
                selected.action.name,

            score=
                float(
                    result_score
                ),

            reason=
                reason,

            components={

                "expected_outcome":
                    selected.expected_outcome,

                "current_valence":
                    selected.current_valence,

                "memory_valence":
                    selected.memory_valence,

                "integrated_valence":
                    selected.integrated_valence,

                "cost":
                    selected.cost,

                # Mesmo durante exploration,
                # registramos a utilidade contrafactual.
                "calculated_utility":
                    selected.utility,

                "exploration":
                    1.0
                    if exploring
                    else 0.0,

            },

        )

    # ========================================================
    # AVALIAÇÃO INDIVIDUAL
    # ========================================================

    def _evaluate_action(
        self,
        action: ActionCandidate,
        current_valence: float,
        memory_valence: float,
        integrated_valence: float,
    ) -> ActionEvaluation:
        """
        Calcula exatamente:

            U(a|E) =
                O(a)
                *
                integrated_valence
                -
                C(a)
        """

        (
            expected_outcome,
            expected_outcome_source,
        ) = self._expected_outcome(
            action
        )

        cost = self._action_cost(
            action
        )

        utility = (

            expected_outcome

            *

            integrated_valence

            -

            cost

        )

        if not math.isfinite(
            utility
        ):

            raise ValueError(
                f"Utilidade não finita para ação "
                f"'{action.name}'."
            )

        return ActionEvaluation(

            action=
                action,

            expected_outcome=
                expected_outcome,

            expected_outcome_source=
                expected_outcome_source,

            current_valence=
                current_valence,

            memory_valence=
                memory_valence,

            integrated_valence=
                integrated_valence,

            cost=
                cost,

            utility=
                float(
                    utility
                ),

        )

    # ========================================================
    # O(a)
    # ========================================================

    def _expected_outcome(
        self,
        action: ActionCandidate,
    ) -> tuple[float, str]:
        """
        Recupera O(a).

        O artigo técnico trata expectedOutcome como propriedade
        explícita da Action.

        Nosso ActionCandidate atual ainda é uma estrutura de
        transição, então aceitamos:

            action.expected_outcome

        quando o models.py definitivo possuir esse atributo,

        OU:

            action.metadata["expected_outcome"]

        durante a migração.

        Não usamos base_utility como substituto.

        Isso misturaria dois conceitos diferentes.
        """

        value: Any | None = None

        source = ""

        # ----------------------------------------------------
        # MODELO FUTURO / CORRETO
        # ----------------------------------------------------

        if hasattr(
            action,
            "expected_outcome",
        ):

            candidate = getattr(
                action,
                "expected_outcome"
            )

            if candidate is not None:

                value = candidate

                source = (
                    "action.expected_outcome"
                )

        # ----------------------------------------------------
        # COMPATIBILIDADE COM NOSSO models.py ATUAL
        # ----------------------------------------------------

        if value is None:

            if (
                "expected_outcome"
                in
                action.metadata
            ):

                value = (
                    action.metadata[
                        "expected_outcome"
                    ]
                )

                source = (
                    "metadata.expected_outcome"
                )

        # ----------------------------------------------------
        # AUSÊNCIA DE O(a)
        # ----------------------------------------------------

        if value is None:

            if self.strict_action_schema:

                raise ValueError(

                    f"A ação '{action.name}' não possui "
                    "expected_outcome. "

                    "O Decision Module EVP exige O(a) "
                    "explicitamente."

                )

            # Prior neutro, somente quando o modo estrito
            # foi deliberadamente desativado.
            value = 0.0

            source = "neutral_prior"

        value = float(
            value
        )

        if not math.isfinite(
            value
        ):

            raise ValueError(
                f"expected_outcome de '{action.name}' "
                "não é finito."
            )

        if not -1.0 <= value <= 1.0:

            raise ValueError(

                f"expected_outcome de '{action.name}' "
                "deve estar entre -1 e +1."

            )

        return (
            value,
            source,
        )

    # ========================================================
    # C(a)
    # ========================================================

    def _action_cost(
        self,
        action: ActionCandidate,
    ) -> float:
        """
        Recupera C(a).

        O artigo técnico define:

            cost ∈ [0, 1]

        Representa custo computacional/energético
        da execução da ação.
        """

        value: Any | None = None

        # models.py futuro
        if hasattr(
            action,
            "cost",
        ):

            candidate = getattr(
                action,
                "cost"
            )

            if candidate is not None:

                value = candidate

        # models.py atual
        if value is None:

            if (
                "cost"
                in
                action.metadata
            ):

                value = (
                    action.metadata[
                        "cost"
                    ]
                )

        if value is None:

            if self.strict_action_schema:

                raise ValueError(

                    f"A ação '{action.name}' não possui cost. "

                    "O Decision Module EVP exige C(a) "
                    "explicitamente."

                )

            value = 0.0

        value = float(
            value
        )

        if not math.isfinite(
            value
        ):

            raise ValueError(
                f"cost de '{action.name}' "
                "não é finito."
            )

        if not 0.0 <= value <= 1.0:

            raise ValueError(

                f"cost de '{action.name}' "
                "deve estar entre 0 e 1."

            )

        return value

    # ========================================================
    # v̄_memory
    # ========================================================

    @staticmethod
    def _calculate_memory_valence(
        patterns: Sequence[PatternMatch],
    ) -> float:
        """
        Implementa a forma utilizada no artigo técnico:

                      n
                     Σ v_i * s_i
                     i=1
            v_mem = --------------
                         n

        onde:

            v_i =
                valência histórica do padrão

            s_i =
                similaridade do padrão com o estado atual

            n =
                quantidade de padrões recuperados


        Se memória estiver vazia:

            v_mem = 0

        ou seja, ausência de experiência não é interpretada
        como experiência positiva nem negativa.
        """

        if not patterns:

            return 0.0

        total = 0.0

        for pattern in patterns:

            similarity = float(
                pattern.similarity
            )

            if not math.isfinite(
                similarity
            ):

                raise ValueError(
                    "PatternMatch contém similaridade "
                    "não finita."
                )

            if not 0.0 <= similarity <= 1.0:

                raise ValueError(
                    "PatternMatch.similarity deve estar "
                    "entre 0 e 1."
                )

            # Nosso memory.py usa average_valence.
            #
            # Mantemos fallback para avg_valence porque esse é
            # o nome usado no documento técnico.
            if hasattr(
                pattern,
                "average_valence",
            ):

                historical_valence = float(
                    pattern.average_valence
                )

            elif hasattr(
                pattern,
                "avg_valence",
            ):

                historical_valence = float(
                    getattr(
                        pattern,
                        "avg_valence"
                    )
                )

            else:

                raise ValueError(
                    "PatternMatch não contém valência histórica."
                )

            if not -1.0 <= historical_valence <= 1.0:

                raise ValueError(
                    "Valência histórica do padrão deve "
                    "estar entre -1 e +1."
                )

            total += (
                historical_valence
                *
                similarity
            )

        memory_valence = (

            total
            /
            len(
                patterns
            )

        )

        # Numericamente deveria permanecer em [-1, 1],
        # mas limitamos pequenos erros de ponto flutuante.
        return max(
            -1.0,
            min(
                1.0,
                float(
                    memory_valence
                ),
            ),
        )

    # ========================================================
    # RATIONALE
    # ========================================================

    @staticmethod
    def _build_rationale(
        event: StimulusEvent,
        valence: ValenceResult,
        selected: ActionEvaluation,
        mode: str,
        triggered_rules: Sequence[str],
    ) -> str:
        """
        Explicação auditável.

        Evitamos frases vagas como:

            "a IA achou melhor".

        Registramos os números que produziram a decisão.
        """

        trigger_text = (

            ",".join(
                triggered_rules
            )

            if triggered_rules

            else "none"

        )

        return (

            f"mode={mode}; "
            f"stimulus={event.id}; "
            f"action={selected.action.name}; "
            f"O(a)={selected.expected_outcome:+.4f}; "
            f"V(E)={valence.score:+.4f}; "
            f"confidence={valence.confidence:.4f}; "
            f"V_used={selected.current_valence:+.4f}; "
            f"V_memory={selected.memory_valence:+.4f}; "
            f"integrated={selected.integrated_valence:+.4f}; "
            f"C(a)={selected.cost:.4f}; "
            f"U(a|E)={selected.utility:+.4f}; "
            f"triggers={trigger_text}"

        )

    # ========================================================
    # CONFIGURAÇÃO
    # ========================================================

    def _validate_configuration(
        self,
    ) -> None:

        for name, value in (

            ("alpha", self.alpha),
            ("beta", self.beta),
            (
                "confidence_threshold",
                self.confidence_threshold,
            ),
            (
                "exploration_rate",
                self.exploration_rate,
            ),

        ):

            if not math.isfinite(
                value
            ):

                raise ValueError(
                    f"{name} precisa ser finito."
                )

        if self.alpha < 0.0:

            raise ValueError(
                "alpha não pode ser negativo."
            )

        if self.beta < 0.0:

            raise ValueError(
                "beta não pode ser negativo."
            )

        if not math.isclose(

            self.alpha
            +
            self.beta,

            1.0,

            rel_tol=1e-9,

            abs_tol=1e-9,

        ):

            raise ValueError(
                "alpha + beta deve ser exatamente 1."
            )

        if not (
            0.0
            <=
            self.confidence_threshold
            <=
            1.0
        ):

            raise ValueError(
                "confidence_threshold deve estar "
                "entre 0 e 1."
            )

        if not (
            0.0
            <=
            self.exploration_rate
            <=
            1.0
        ):

            raise ValueError(
                "exploration_rate deve estar "
                "entre 0 e 1."
            )

    # ========================================================
    # VALIDAÇÃO DE VALÊNCIA
    # ========================================================

    @staticmethod
    def _validate_valence(
        valence: ValenceResult,
    ) -> None:

        score = float(
            valence.score
        )

        confidence = float(
            valence.confidence
        )

        if not math.isfinite(
            score
        ):

            raise ValueError(
                "ValenceResult.score não é finito."
            )

        if not math.isfinite(
            confidence
        ):

            raise ValueError(
                "ValenceResult.confidence não é finito."
            )

        if not -1.0 <= score <= 1.0:

            raise ValueError(
                "ValenceResult.score precisa estar "
                "entre -1 e +1."
            )

        if not 0.0 <= confidence <= 1.0:

            raise ValueError(
                "ValenceResult.confidence precisa estar "
                "entre 0 e 1."
            )

    # ========================================================
    # SNAPSHOT
    # ========================================================

    def snapshot(
        self,
    ) -> dict[str, Any]:
        """
        Estado observável do motor de decisão.
        """

        data: dict[str, Any] = {

            "alpha":
                self.alpha,

            "beta":
                self.beta,

            "confidence_threshold":
                self.confidence_threshold,

            "exploration_rate":
                self.exploration_rate,

            "strict_action_schema":
                self.strict_action_schema,

        }

        if self.last_trace is None:

            data["last_decision"] = None

            return data

        data["last_decision"] = {

            "stimulus_id":
                self.last_trace.stimulus_id,

            "mode":
                self.last_trace.mode,

            "random_value":
                self.last_trace.random_value,

            "memory_valence":
                self.last_trace.memory_valence,

            "selected_action":
                self.last_trace.selected_action,

            "alternatives": [

                {

                    "action":
                        evaluation.action.name,

                    "expected_outcome":
                        evaluation.expected_outcome,

                    "expected_outcome_source":
                        evaluation.expected_outcome_source,

                    "current_valence":
                        evaluation.current_valence,

                    "memory_valence":
                        evaluation.memory_valence,

                    "integrated_valence":
                        evaluation.integrated_valence,

                    "cost":
                        evaluation.cost,

                    "utility":
                        evaluation.utility,

                }

                for evaluation
                in self.last_trace.evaluations

            ],

        }

        return data