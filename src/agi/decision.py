from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .models import (
    ActionCandidate,
    DecisionResult,
    StimulusEvent,
    ValenceResult,
)

from .triggers import Trigger


@dataclass
class ScoredAction:
    action: ActionCandidate
    score: float
    explanation: str


class DecisionModule:
    """
    Módulo responsável por escolher uma ação com base em:

    - utilidade base da ação
    - alinhamento com objetivos
    - valência do estímulo
    - risco percebido
    - gatilhos ativados

    A ideia é que a decisão não seja fixa:
    ela depende do estado atual do agente.
    """

    def __init__(
        self,
        goal_weight: float = 0.40,
        valence_weight: float = 0.25,
        trigger_weight: float = 0.20,
    ):
        self.goal_weight = goal_weight
        self.valence_weight = valence_weight
        self.trigger_weight = trigger_weight

    def choose(
        self,
        event: StimulusEvent,
        valence: ValenceResult,
        triggers: List[Trigger],
        actions: List[ActionCandidate],
    ) -> DecisionResult:
        """
        Escolhe a melhor ação disponível.

        Parameters
        ----------
        event:
            Estímulo atual sendo processado.

        valence:
            Resultado da avaliação de valência.

        triggers:
            Lista de gatilhos ativados.

        actions:
            Lista de ações disponíveis.

        Returns
        -------
        DecisionResult
            A ação escolhida, score final e justificativa.
        """

        if not actions:
            return DecisionResult(
                action="observe",
                score=0.0,
                reason="Nenhuma ação disponível. Mantendo observação.",
            )

        ranked_actions: List[ScoredAction] = []

        for action in actions:
            scored = self._score_action(
                event=event,
                valence=valence,
                triggers=triggers,
                action=action,
            )

            ranked_actions.append(scored)

        ranked_actions.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        best = ranked_actions[0]

        return DecisionResult(
            action=best.action.name,
            score=best.score,
            reason=best.explanation,
        )

    def _score_action(
        self,
        event: StimulusEvent,
        valence: ValenceResult,
        triggers: List[Trigger],
        action: ActionCandidate,
    ) -> ScoredAction:
        """
        Calcula o score de uma ação específica.
        """

        payload = event.payload

        goal_alignment = float(
            payload.get("goal_alignment", 0.0)
        )

        risk_level = float(
            payload.get("risk_level", 0.0)
        )

        trigger_score = self._calculate_trigger_score(
            triggers
        )

        base_score = float(action.base_utility)

        goal_component = (
            goal_alignment
            * self.goal_weight
        )

        valence_component = (
            valence.score
            * self.valence_weight
        )

        trigger_component = (
            trigger_score
            * self.trigger_weight
        )

        risk_component = (
            action.risk_penalty
            * risk_level
        )

        final_score = (
            base_score
            + goal_component
            + valence_component
            + trigger_component
            - risk_component
        )

        explanation = (
            f"base={base_score:.3f}; "
            f"goal={goal_component:.3f}; "
            f"valence={valence_component:.3f}; "
            f"trigger={trigger_component:.3f}; "
            f"risk_penalty={risk_component:.3f}; "
            f"final={final_score:.3f}"
        )

        return ScoredAction(
            action=action,
            score=float(final_score),
            explanation=explanation,
        )

    @staticmethod
    def _calculate_trigger_score(
        triggers: List[Trigger],
    ) -> float:
        """
        Converte a prioridade acumulada dos gatilhos
        em um valor normalizado entre 0 e 1.
        """

        if not triggers:
            return 0.0

        total_priority = sum(
            trigger.priority
            for trigger in triggers
        )

        # Evita score absurdo quando há muitos gatilhos.
        normalized = total_priority / 200.0

        return max(
            0.0,
            min(1.0, normalized),
        )