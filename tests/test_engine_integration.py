from __future__ import annotations

import numpy as np
import pytest

from agi.decision import DecisionModule
from agi.engine import CognitiveEngine
from agi.memory import PatternMemory
from agi.models import (
    ActionCandidate,
    OutcomeEvent,
    StimulusEvent,
    StimulusSource,
)
from agi.triggers import (
    CooldownScope,
    TriggerPolarity,
    TriggerRule,
    TriggerSystem,
)
from agi.valence import ValenceEngine


# ============================================================
# HELPERS
# ============================================================


def make_danger_event(
    *,
    intensity: float = 0.90,
    context_id: str = "danger-context",
    event_type: str = "environment_danger",
) -> StimulusEvent:
    """Cria um estímulo adverso sem embutir valência manual."""

    return StimulusEvent(
        source=StimulusSource.EXTERNAL,
        type=event_type,
        intensity=intensity,
        context_id=context_id,
        payload={
            "urgency": 0.90,
            "goal_alignment": -0.80,
            "risk_level": 0.95,
        },
    )



def make_actions() -> list[ActionCandidate]:
    """Espaço de ações explícito usado nos testes de integração."""

    return [
        ActionCandidate(
            name="observe",
            expected_outcome=0.25,
            cost=0.05,
        ),
        ActionCandidate(
            name="investigate",
            expected_outcome=0.80,
            cost=0.25,
        ),
        ActionCandidate(
            name="avoid",
            expected_outcome=0.60,
            cost=0.10,
        ),
    ]



def make_engine(
    tmp_path,
    *,
    trigger_system: TriggerSystem | None = None,
    decision_module: DecisionModule | None = None,
    valence_engine: ValenceEngine | None = None,
    sensory_threshold: float | None = 0.20,
) -> CognitiveEngine:
    """
    Cria um CognitiveEngine isolado do estado real do projeto.

    Cada teste usa arquivos temporários próprios para impedir que
    memory.json/valence.json reais contaminem os resultados.
    """

    memory = PatternMemory(
        storage_path=tmp_path / "memory.json",
    )

    if valence_engine is None:
        valence_engine = ValenceEngine(
            storage_path=tmp_path / "valence.json",
            load_state=False,
        )

    if trigger_system is None:
        trigger_system = TriggerSystem()

    if decision_module is None:
        decision_module = DecisionModule(
            exploration_rate=0.0,
            random_seed=123,
        )

    return CognitiveEngine(
        memory=memory,
        valence=valence_engine,
        triggers=trigger_system,
        decisions=decision_module,
        sensory_threshold=sensory_threshold,
    )


# ============================================================
# TEST DOUBLES
# ============================================================


class FailIfCalledDecisionModule(DecisionModule):
    """Falha imediatamente se o caminho deliberativo for executado."""

    def choose(self, *args, **kwargs):  # type: ignore[override]
        raise AssertionError(
            "DecisionModule não deveria ser chamado no caminho automático."
        )


class CountingDecisionModule(DecisionModule):
    """Conta quantas vezes o caminho deliberativo foi utilizado."""

    def __init__(self) -> None:
        super().__init__(
            exploration_rate=0.0,
            random_seed=123,
        )
        self.calls = 0

    def choose(self, *args, **kwargs):  # type: ignore[override]
        self.calls += 1
        return super().choose(*args, **kwargs)


class CapturingValenceEngine(ValenceEngine):
    """Registra o vetor usado efetivamente no momento do aprendizado."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_learning_features: np.ndarray | None = None

    def learn(self, features, outcome, learning_rate=None):  # type: ignore[override]
        self.last_learning_features = np.asarray(
            features,
            dtype=np.float64,
        ).copy()

        return super().learn(
            features=features,
            outcome=outcome,
            learning_rate=learning_rate,
        )


# ============================================================
# 1. FILTRO PRÉ-ATENCIONAL
# ============================================================


def test_low_intensity_stimulus_is_filtered_before_cognition(tmp_path):
    engine = make_engine(tmp_path)

    event = make_danger_event(
        intensity=0.19,
    )

    step = engine.process(
        event,
        make_actions(),
    )

    assert step.filtered is True
    assert step.filter_reason is not None
    assert step.novelty is None
    assert step.features == []
    assert step.valence is None
    assert step.patterns == []
    assert step.triggers == []
    assert step.decision is None
    assert step.pattern_id is None

    # Um estímulo filtrado não deve entrar na memória nem aguardar feedback.
    assert len(engine.memory.patterns) == 0
    assert engine.pending_experiences == ()



def test_sensory_threshold_boundary_is_inclusive_for_processing(tmp_path):
    """
    O engine filtra apenas I < threshold.

    Portanto I == 0.20 deve ser processado.
    """

    engine = make_engine(tmp_path)

    event = make_danger_event(
        intensity=0.20,
    )

    step = engine.process(
        event,
        make_actions(),
    )

    assert step.filtered is False
    assert step.valence is not None
    assert step.decision is not None
    assert step.pattern_id is not None


# ============================================================
# 2. PROCESS() PRODUZ EXPERIÊNCIA PENDENTE
# ============================================================


def test_process_creates_pending_experience_and_pattern(tmp_path):
    engine = make_engine(tmp_path)

    event = make_danger_event()

    step = engine.process(
        event,
        make_actions(),
    )

    assert step.filtered is False
    assert step.cycle == 1
    assert step.novelty == pytest.approx(1.0)
    assert step.valence is not None
    assert step.decision is not None
    assert step.pattern_id is not None

    assert event.id in engine.pending_experiences
    assert len(engine.memory.patterns) == 1

    pending = engine.get_pending_experience(
        event.id
    )

    assert pending is not None
    assert pending.stimulus.id == event.id
    assert pending.decision.action == step.decision.action
    assert np.allclose(
        pending.features,
        np.asarray(step.features),
    )


# ============================================================
# 3. EVENTO ATUAL NÃO PODE VAZAR PARA A MEMÓRIA HISTÓRICA
# ============================================================


def test_first_decision_does_not_retrieve_current_event_as_past_memory(tmp_path):
    """
    O primeiro evento deve ser recuperado da memória somente em ciclos futuros.

    Isso testa a ordem:

        retrieve previous patterns
        BEFORE
        remember current pattern
    """

    engine = make_engine(tmp_path)

    first = make_danger_event(
        context_id="episode-1",
    )

    step_1 = engine.process(
        first,
        make_actions(),
    )

    assert step_1.patterns == []

    engine.feedback(
        OutcomeEvent(
            stimulus_id=first.id,
            action_id=step_1.decision.action,
            outcome=-0.80,
            context_id=first.context_id,
        )
    )

    second = make_danger_event(
        context_id="episode-2",
    )

    step_2 = engine.process(
        second,
        make_actions(),
    )

    assert len(step_2.patterns) >= 1


# ============================================================
# 4. FEEDBACK FECHA O CICLO E ATUALIZA OS DOIS SISTEMAS
# ============================================================


def test_feedback_updates_valence_and_pattern_memory(tmp_path):
    engine = make_engine(tmp_path)

    event = make_danger_event()

    step = engine.process(
        event,
        make_actions(),
    )

    assert step.decision is not None
    assert step.pattern_id is not None

    weights_before = engine.valence.learned_weights.copy()

    feedback = engine.feedback(
        OutcomeEvent(
            stimulus_id=event.id,
            action_id=step.decision.action,
            outcome=-0.90,
            context_id=event.context_id,
        )
    )

    assert feedback.stimulus_id == event.id
    assert feedback.action_id == step.decision.action
    assert feedback.outcome == pytest.approx(-0.90)
    assert feedback.updated_pattern_ids == (step.pattern_id,)

    # O ciclo foi concluído.
    assert event.id not in engine.pending_experiences

    # O motor de valência realmente aprendeu.
    assert engine.valence.experience_count == 1
    assert not np.allclose(
        engine.valence.learned_weights,
        weights_before,
    )

    # A consequência foi ligada ao padrão e à ação escolhida.
    expected_outcome = engine.memory.expected_outcome(
        pattern_id=step.pattern_id,
        action_id=step.decision.action,
    )

    assert expected_outcome == pytest.approx(-0.90)


# ============================================================
# 5. GARANTIA CAUSAL DA AÇÃO
# ============================================================


def test_feedback_rejects_outcome_for_action_that_was_not_selected(tmp_path):
    engine = make_engine(tmp_path)

    event = make_danger_event()

    step = engine.process(
        event,
        make_actions(),
    )

    assert step.decision is not None

    wrong_action = (
        "avoid"
        if step.decision.action != "avoid"
        else "observe"
    )

    with pytest.raises(
        ValueError,
        match="não corresponde",
    ):
        engine.feedback(
            OutcomeEvent(
                stimulus_id=event.id,
                action_id=wrong_action,
                outcome=-0.50,
            )
        )

    # A tentativa inválida não deve destruir a experiência pendente.
    assert event.id in engine.pending_experiences
    assert engine.valence.experience_count == 0



def test_second_feedback_for_same_experience_is_rejected(tmp_path):
    engine = make_engine(tmp_path)

    event = make_danger_event()

    step = engine.process(
        event,
        make_actions(),
    )

    outcome_event = OutcomeEvent(
        stimulus_id=event.id,
        action_id=step.decision.action,
        outcome=-0.50,
    )

    engine.feedback(
        outcome_event
    )

    with pytest.raises(KeyError):
        engine.feedback(
            outcome_event
        )


# ============================================================
# 6. APRENDIZADO ALTERA A EXPERIÊNCIA FUTURA
# ============================================================


def test_negative_feedback_changes_future_valence_and_reduces_novelty(tmp_path):
    engine = make_engine(tmp_path)

    first_event = make_danger_event(
        context_id="first",
    )

    first_step = engine.process(
        first_event,
        make_actions(),
    )

    assert first_step.valence is not None
    assert first_step.novelty is not None
    assert first_step.valence.score == pytest.approx(0.0)

    engine.feedback(
        OutcomeEvent(
            stimulus_id=first_event.id,
            action_id=first_step.decision.action,
            outcome=-1.0,
        )
    )

    second_event = make_danger_event(
        context_id="second",
    )

    second_step = engine.process(
        second_event,
        make_actions(),
    )

    assert second_step.valence is not None
    assert second_step.novelty is not None

    assert second_step.novelty < first_step.novelty
    assert second_step.valence.score < first_step.valence.score

    # A mesma estrutura deve ter sido consolidada.
    assert len(engine.memory.patterns) == 1
    assert engine.memory.patterns[0].frequency == 2


# ============================================================
# 7. FEEDBACK USA φ(E) ORIGINAL
# ============================================================


def test_feedback_uses_original_feature_vector_from_decision_time(tmp_path):
    valence = CapturingValenceEngine(
        storage_path=tmp_path / "valence.json",
        load_state=False,
    )

    engine = make_engine(
        tmp_path,
        valence_engine=valence,
    )

    event = make_danger_event()

    step = engine.process(
        event,
        make_actions(),
    )

    original_features = np.asarray(
        step.features,
        dtype=np.float64,
    )

    # Neste ponto o evento atual já entrou na memória.
    # Recalcular novelty agora produziria outro vetor.
    novelty_after_remember = engine.memory.estimate_novelty(
        event
    )

    assert novelty_after_remember < step.novelty

    engine.feedback(
        OutcomeEvent(
            stimulus_id=event.id,
            action_id=step.decision.action,
            outcome=-0.75,
        )
    )

    assert valence.last_learning_features is not None

    assert np.allclose(
        valence.last_learning_features,
        original_features,
    )


# ============================================================
# 8. TRIGGER AUTOMÁTICO DESVIA DO DECISION MODULE
# ============================================================


def test_configured_automatic_trigger_bypasses_deliberative_decision(tmp_path):
    """
    Produz uma valência fortemente negativa usando um prior experimental
    explícito para risk, apenas para testar o roteamento do engine.
    """

    valence = ValenceEngine(
        intrinsic_weights={
            "intensity": 0.0,
            "novelty": 0.0,
            "urgency": 0.0,
            "goal_alignment": 0.0,
            "risk": -3.0,
        },
        storage_path=tmp_path / "valence.json",
        load_state=False,
    )

    trigger_system = TriggerSystem(
        rules=[
            TriggerRule(
                id="auto_avoid",
                name="Automatic Avoid",
                polarity=TriggerPolarity.NEGATIVE,
                threshold=0.75,
                cooldown_cycles=0,
                cooldown_scope=CooldownScope.CONTEXT,
                response_action="avoid",
            )
        ]
    )

    engine = make_engine(
        tmp_path,
        trigger_system=trigger_system,
        decision_module=FailIfCalledDecisionModule(),
        valence_engine=valence,
    )

    event = make_danger_event()

    step = engine.process(
        event,
        make_actions(),
    )

    assert step.valence is not None
    assert step.valence.score < -0.75
    assert len(step.triggers) == 1
    assert step.decision is not None
    assert step.decision.action == "avoid"
    assert step.decision_path == "automatic"
    assert step.decision.components["automatic"] == pytest.approx(1.0)


# ============================================================
# 9. TRIGGER INVÁLIDO NÃO CRIA AÇÃO FORA DO ESPAÇO PERMITIDO
# ============================================================


def test_unavailable_automatic_action_falls_back_to_deliberation(tmp_path):
    valence = ValenceEngine(
        intrinsic_weights={
            "intensity": 0.0,
            "novelty": 0.0,
            "urgency": 0.0,
            "goal_alignment": 0.0,
            "risk": -3.0,
        },
        storage_path=tmp_path / "valence.json",
        load_state=False,
    )

    trigger_system = TriggerSystem(
        rules=[
            TriggerRule(
                id="invalid_auto_action",
                name="Invalid Automatic Action",
                polarity=TriggerPolarity.NEGATIVE,
                threshold=0.75,
                cooldown_cycles=0,
                response_action="shutdown_everything",
            )
        ]
    )

    decisions = CountingDecisionModule()

    engine = make_engine(
        tmp_path,
        trigger_system=trigger_system,
        decision_module=decisions,
        valence_engine=valence,
    )

    step = engine.process(
        make_danger_event(),
        make_actions(),
    )

    assert len(step.triggers) == 1
    assert decisions.calls == 1
    assert step.decision is not None
    assert step.decision.action in {
        "observe",
        "investigate",
        "avoid",
    }
    assert step.decision_path == "deliberative_after_unresolved_trigger"



def test_trigger_without_response_action_falls_back_to_deliberation(tmp_path):
    valence = ValenceEngine(
        intrinsic_weights={
            "intensity": 0.0,
            "novelty": 0.0,
            "urgency": 0.0,
            "goal_alignment": 0.0,
            "risk": -3.0,
        },
        storage_path=tmp_path / "valence.json",
        load_state=False,
    )

    trigger_system = TriggerSystem(
        rules=[
            TriggerRule(
                id="unbound_trigger",
                name="Unbound Trigger",
                polarity=TriggerPolarity.NEGATIVE,
                threshold=0.75,
                cooldown_cycles=0,
                response_action=None,
            )
        ]
    )

    decisions = CountingDecisionModule()

    engine = make_engine(
        tmp_path,
        trigger_system=trigger_system,
        decision_module=decisions,
        valence_engine=valence,
    )

    step = engine.process(
        make_danger_event(),
        make_actions(),
    )

    assert len(step.triggers) == 1
    assert decisions.calls == 1
    assert step.decision is not None
    assert step.decision_path == "deliberative_after_unresolved_trigger"


# ============================================================
# 10. VALIDAÇÃO DO ESPAÇO DE AÇÕES
# ============================================================


def test_unfiltered_stimulus_requires_at_least_one_action(tmp_path):
    engine = make_engine(tmp_path)

    with pytest.raises(
        ValueError,
        match="nenhuma ação",
    ):
        engine.process(
            make_danger_event(),
            [],
        )



def test_duplicate_action_names_are_rejected(tmp_path):
    engine = make_engine(tmp_path)

    actions = [
        ActionCandidate(
            name="observe",
            expected_outcome=0.2,
            cost=0.1,
        ),
        ActionCandidate(
            name="observe",
            expected_outcome=0.7,
            cost=0.2,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="único",
    ):
        engine.process(
            make_danger_event(),
            actions,
        )


# ============================================================
# 11. PERSISTÊNCIA END-TO-END
# ============================================================


def test_engine_learning_survives_restart(tmp_path):
    memory_path = tmp_path / "memory.json"
    valence_path = tmp_path / "valence.json"

    first_memory = PatternMemory(
        storage_path=memory_path,
    )

    first_valence = ValenceEngine(
        storage_path=valence_path,
        load_state=False,
    )

    first_engine = CognitiveEngine(
        memory=first_memory,
        valence=first_valence,
        triggers=TriggerSystem(),
        decisions=DecisionModule(
            exploration_rate=0.0,
            random_seed=123,
        ),
    )

    event = make_danger_event()

    step = first_engine.process(
        event,
        make_actions(),
    )

    first_engine.feedback(
        OutcomeEvent(
            stimulus_id=event.id,
            action_id=step.decision.action,
            outcome=-0.85,
        )
    )

    weights_before_restart = (
        first_engine.valence.learned_weights.copy()
    )

    pattern_count_before_restart = len(
        first_engine.memory.patterns
    )

    # --------------------------------------------------------
    # NOVO PROCESSO / RESTART
    # --------------------------------------------------------

    second_engine = CognitiveEngine(
        memory=PatternMemory(
            storage_path=memory_path,
        ),
        valence=ValenceEngine(
            storage_path=valence_path,
            load_state=True,
        ),
        triggers=TriggerSystem(),
        decisions=DecisionModule(
            exploration_rate=0.0,
            random_seed=123,
        ),
    )

    assert second_engine.valence.experience_count == 1

    assert np.allclose(
        second_engine.valence.learned_weights,
        weights_before_restart,
    )

    assert len(
        second_engine.memory.patterns
    ) == pattern_count_before_restart

    # O conhecimento persistido precisa influenciar o próximo ciclo.
    next_event = make_danger_event(
        context_id="after-restart",
    )

    next_step = second_engine.process(
        next_event,
        make_actions(),
    )

    assert next_step.novelty is not None
    assert next_step.novelty < 1.0

    assert next_step.valence is not None
    assert next_step.valence.score < 0.0


# ============================================================
# 12. ESTADO INTERNO OBSERVÁVEL
# ============================================================


def test_internal_state_reflects_current_engine_state(tmp_path):
    engine = make_engine(tmp_path)

    event = make_danger_event()

    step = engine.process(
        event,
        make_actions(),
    )

    state_before_feedback = engine.internal_state()

    assert state_before_feedback["cognitive_cycles"] == 1
    assert state_before_feedback["pending_experiences"] == 1
    assert "memory" in state_before_feedback
    assert "valence" in state_before_feedback
    assert "triggers" in state_before_feedback
    assert "decision" in state_before_feedback

    engine.feedback(
        OutcomeEvent(
            stimulus_id=event.id,
            action_id=step.decision.action,
            outcome=-0.40,
        )
    )

    state_after_feedback = engine.internal_state()

    assert state_after_feedback["pending_experiences"] == 0
    assert state_after_feedback["valence"]["experience_count"] == 1


# ============================================================
# 13. CICLO END-TO-END COMPLETO
# ============================================================


def test_complete_evp_cycle_experience_decision_outcome_learning_reexperience(
    tmp_path,
):
    """
    Teste de integração principal.

    Verifica o ciclo:

        experiência
            ↓
        decisão
            ↓
        consequência
            ↓
        aprendizado
            ↓
        nova experiência semelhante

    O segundo ciclo deve carregar efeitos reais do primeiro.
    """

    engine = make_engine(tmp_path)

    # --------------------------------------------------------
    # CICLO 1
    # --------------------------------------------------------

    event_1 = make_danger_event(
        context_id="cycle-1",
    )

    step_1 = engine.process(
        event_1,
        make_actions(),
    )

    assert step_1.filtered is False
    assert step_1.novelty == pytest.approx(1.0)
    assert step_1.valence is not None
    assert step_1.valence.score == pytest.approx(0.0)
    assert step_1.decision is not None

    engine.feedback(
        OutcomeEvent(
            stimulus_id=event_1.id,
            action_id=step_1.decision.action,
            outcome=-1.0,
            context_id=event_1.context_id,
        )
    )

    assert engine.valence.experience_count == 1
    assert event_1.id not in engine.pending_experiences

    # --------------------------------------------------------
    # CICLO 2
    # --------------------------------------------------------

    event_2 = make_danger_event(
        context_id="cycle-2",
    )

    step_2 = engine.process(
        event_2,
        make_actions(),
    )

    assert step_2.filtered is False
    assert step_2.novelty is not None
    assert step_2.novelty < step_1.novelty

    assert step_2.valence is not None
    assert step_2.valence.score < step_1.valence.score

    assert len(step_2.patterns) >= 1

    assert engine.memory.patterns[0].frequency >= 2

    # O segundo ciclo agora aguarda sua própria consequência.
    assert event_2.id in engine.pending_experiences
