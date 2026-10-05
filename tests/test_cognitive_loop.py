from __future__ import annotations

import numpy as np
import pytest

from agi.decision import DecisionModule
from agi.features import (
    DIMENSIONS,
    extract_features,
    extract_pattern_features,
)
from agi.memory import (
    PatternMatch,
    PatternMemory,
)
from agi.models import (
    ActionCandidate,
    StimulusEvent,
    StimulusSource,
    ValenceResult,
)
from agi.triggers import (
    TriggerPolarity,
    TriggerSystem,
)
from agi.valence import ValenceEngine


# ============================================================
# HELPERS
# ============================================================


def make_danger_event(
    *,
    intensity: float = 0.9,
    context_id: str = "danger-context",
) -> StimulusEvent:
    """
    Cria um estímulo experimental potencialmente adverso.

    IMPORTANTE:

    Nenhuma valência é colocada manualmente no StimulusEvent.

    O estímulo contém apenas características observáveis
    que posteriormente poderão participar de:

        φ(E)
            ↓
        V(E)
    """

    return StimulusEvent(
        source=StimulusSource.EXTERNAL,
        type="environment_danger",
        intensity=intensity,
        context_id=context_id,
        payload={
            "urgency": 0.9,
            "goal_alignment": -0.8,
            "risk_level": 0.95,
        },
    )


def make_actions() -> list[ActionCandidate]:
    """
    Espaço experimental de ações conforme o modelo EVP.

    Para cada ação:

        O(a) = expected_outcome
        C(a) = cost

    Esses valores são propriedades explícitas da ação
    e NÃO ficam mais dentro de metadata.
    """

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


# ============================================================
# 1. FEATURE VECTOR EVP
# ============================================================


def test_feature_vector_follows_evp_dimensions():
    """
    φ(E) deve possuir exatamente as cinco dimensões
    atualmente definidas pelo modelo EVP:

        intensity
        novelty
        urgency
        goal_alignment
        risk
    """

    event = make_danger_event()

    features = extract_features(
        event,
        novelty=0.75,
    )

    assert tuple(DIMENSIONS) == (
        "intensity",
        "novelty",
        "urgency",
        "goal_alignment",
        "risk",
    )

    assert features.shape == (5,)

    assert features[0] == pytest.approx(
        0.90
    )

    assert features[1] == pytest.approx(
        0.75
    )

    assert features[2] == pytest.approx(
        0.90
    )

    assert features[3] == pytest.approx(
        -0.80
    )

    assert features[4] == pytest.approx(
        0.95
    )


# ============================================================
# 2. NOVIDADE NÃO PODE CALCULAR A SI PRÓPRIA
# ============================================================


def test_pattern_features_exclude_novelty():
    """
    O vetor estrutural utilizado para comparar padrões
    não pode conter novelty.

    Caso contrário teríamos circularidade:

        novelty
            ↓
        comparação de padrões
            ↓
        novelty
    """

    event = make_danger_event()

    features = extract_pattern_features(
        event
    )

    assert features.shape == (4,)

    assert np.allclose(

        features,

        np.array(
            [
                0.9,
                0.9,
                -0.8,
                0.95,
            ]
        ),

    )


# ============================================================
# 3. PRIMEIRA EXPERIÊNCIA É TOTALMENTE NOVA
# ============================================================


def test_first_experience_is_maximally_novel(
    tmp_path,
):
    memory = PatternMemory(

        storage_path=(
            tmp_path
            /
            "memory.json"
        )

    )

    event = make_danger_event()

    novelty = memory.estimate_novelty(
        event
    )

    assert novelty == pytest.approx(
        1.0
    )


# ============================================================
# 4. REPETIÇÃO REDUZ NOVIDADE
# ============================================================


def test_repeated_experience_reduces_novelty(
    tmp_path,
):
    memory = PatternMemory(

        storage_path=(
            tmp_path
            /
            "memory.json"
        )

    )

    event = make_danger_event()

    novelty_before = (
        memory.estimate_novelty(
            event
        )
    )

    features = extract_features(
        event,
        novelty=novelty_before,
    )

    memory.remember(

        stimulus_type=
            event.type,

        features=
            features,

        valence=
            -0.2,

    )

    novelty_after = (
        memory.estimate_novelty(
            event
        )
    )

    assert novelty_before == pytest.approx(
        1.0
    )

    assert (
        novelty_after
        <
        novelty_before
    )

    # Para um evento praticamente idêntico,
    # esperamos novidade próxima de zero.
    assert novelty_after < 0.10


# ============================================================
# 5. CONSOLIDAÇÃO DE PADRÕES
# ============================================================


def test_similar_experiences_consolidate_same_pattern(
    tmp_path,
):
    memory = PatternMemory(

        storage_path=(
            tmp_path
            /
            "memory.json"
        )

    )

    event = make_danger_event()

    features = extract_features(
        event,
        novelty=1.0,
    )

    first = memory.remember(

        stimulus_type=
            event.type,

        features=
            features,

        valence=
            -0.4,

    )

    second = memory.remember(

        stimulus_type=
            event.type,

        features=
            features,

        valence=
            -0.6,

    )

    assert first.id == second.id

    assert len(
        memory.patterns
    ) == 1

    assert (
        memory.patterns[0].frequency
        ==
        2
    )

    assert (

        memory.patterns[0]
        .predicted_valence_mean

        ==

        pytest.approx(
            -0.5
        )

    )


# ============================================================
# 6. PADRÃO + AÇÃO + CONSEQUÊNCIA
# ============================================================


def test_memory_learns_action_outcome(
    tmp_path,
):
    memory = PatternMemory(

        storage_path=(
            tmp_path
            /
            "memory.json"
        )

    )

    event = make_danger_event()

    features = extract_features(
        event,
        novelty=1.0,
    )

    pattern = memory.remember(

        stimulus_type=
            event.type,

        features=
            features,

        valence=
            -0.5,

    )

    memory.record_outcome(

        pattern_id=
            pattern.id,

        action_id=
            "avoid",

        outcome=
            0.8,

    )

    memory.record_outcome(

        pattern_id=
            pattern.id,

        action_id=
            "avoid",

        outcome=
            0.6,

    )

    expected = memory.expected_outcome(

        pattern_id=
            pattern.id,

        action_id=
            "avoid",

    )

    assert expected == pytest.approx(
        0.7
    )


# ============================================================
# 7. PRIOR NEUTRO
# ============================================================


def test_neutral_weights_produce_neutral_initial_valence(
    tmp_path,
):
    """
    Sem experiência e sem pesos intrínsecos configurados:

        w = 0

    portanto:

        wᵀφ(E) = 0

    e:

        tanh(0) = 0
    """

    engine = ValenceEngine(

        storage_path=(
            tmp_path
            /
            "valence.json"
        ),

        load_state=False,

    )

    event = make_danger_event()

    features = extract_features(
        event,
        novelty=1.0,
    )

    result = engine.evaluate(
        event,
        features,
    )

    assert result.raw_activation == pytest.approx(
        0.0
    )

    assert result.score == pytest.approx(
        0.0
    )

    assert result.confidence == pytest.approx(
        0.0
    )


# ============================================================
# 8. REGRA DELTA DO ARTIGO
# ============================================================


def test_valence_delta_rule_matches_article(
    tmp_path,
):
    """
    Fórmula:

        Δw =
            η
            *
            (
                r - wᵀφ(E)
            )
            *
            φ(E)


    Como inicialmente:

        w = 0

    então:

        Δw =
            η
            *
            r
            *
            φ(E)
    """

    engine = ValenceEngine(

        learning_rate=
            0.05,

        storage_path=(
            tmp_path
            /
            "valence.json"
        ),

        load_state=False,

    )

    event = make_danger_event()

    phi = extract_features(
        event,
        novelty=1.0,
    )

    outcome = -0.9

    update = engine.learn(

        features=
            phi,

        outcome=
            outcome,

    )

    expected_delta = (

        0.05

        *

        (-0.9)

        *

        phi

    )

    assert np.allclose(
        update.delta,
        expected_delta,
    )

    assert np.allclose(
        engine.learned_weights,
        expected_delta,
    )


# ============================================================
# 9. EXPERIÊNCIA ALTERA VALÊNCIA
# ============================================================


def test_negative_experience_makes_similar_stimulus_more_negative(
    tmp_path,
):
    engine = ValenceEngine(

        learning_rate=
            0.05,

        storage_path=(
            tmp_path
            /
            "valence.json"
        ),

        load_state=False,

    )

    event = make_danger_event()

    phi = extract_features(
        event,
        novelty=1.0,
    )

    before = engine.evaluate(
        event,
        phi,
    )

    engine.learn(

        features=
            phi,

        outcome=
            -1.0,

    )

    after = engine.evaluate(
        event,
        phi,
    )

    assert before.score == pytest.approx(
        0.0
    )

    assert (
        after.score
        <
        before.score
    )


# ============================================================
# 10. GATILHO NEGATIVO
# ============================================================


def test_high_negative_valence_activates_trigger():
    system = TriggerSystem()

    event = make_danger_event()

    valence = ValenceResult(

        stimulus_id=
            event.id,

        score=
            -0.90,

        confidence=
            0.90,

        components={},

        raw_activation=
            -1.47,

    )

    fired = system.evaluate(

        event,

        valence,

        cycle=1,

    )

    assert len(
        fired
    ) == 1

    assert (

        fired[0].polarity

        ==

        TriggerPolarity.NEGATIVE

    )

    assert fired[0].priority == 90


# ============================================================
# 11. GATILHO POSITIVO
# ============================================================


def test_high_positive_valence_also_activates_trigger():
    system = TriggerSystem()

    event = StimulusEvent(

        source=
            StimulusSource.EXTERNAL,

        type=
            "resource",

        intensity=
            0.8,

        context_id=
            "resource-context",

        payload={},

    )

    valence = ValenceResult(

        stimulus_id=
            event.id,

        score=
            0.88,

        confidence=
            0.80,

        components={},

        raw_activation=
            1.38,

    )

    fired = system.evaluate(

        event,

        valence,

        cycle=1,

    )

    assert len(
        fired
    ) == 1

    assert (

        fired[0].polarity

        ==

        TriggerPolarity.POSITIVE

    )


# ============================================================
# 12. COOLDOWN
# ============================================================


def test_trigger_cooldown_prevents_repeated_loop():
    system = TriggerSystem()

    event = make_danger_event(
        context_id="same-loop"
    )

    valence = ValenceResult(

        stimulus_id=
            event.id,

        score=
            -0.95,

        confidence=
            1.0,

        components={},

        raw_activation=
            -2.0,

    )

    first = system.evaluate(

        event,

        valence,

        cycle=1,

    )

    second = system.evaluate(

        event,

        valence,

        cycle=2,

    )

    assert len(
        first
    ) == 1

    assert second == []

    assert len(
        system.suppressed_last_cycle
    ) == 1


# ============================================================
# 13. COOLDOWN SENSÍVEL AO CONTEXTO
# ============================================================


def test_cooldown_is_context_sensitive():
    system = TriggerSystem()

    first_event = make_danger_event(
        context_id="fire-A"
    )

    second_event = make_danger_event(
        context_id="danger-B"
    )

    first_valence = ValenceResult(

        stimulus_id=
            first_event.id,

        score=
            -0.95,

        confidence=
            1.0,

        components={},

        raw_activation=
            -2.0,

    )

    second_valence = ValenceResult(

        stimulus_id=
            second_event.id,

        score=
            -0.97,

        confidence=
            1.0,

        components={},

        raw_activation=
            -2.1,

    )

    first = system.evaluate(

        first_event,

        first_valence,

        cycle=1,

    )

    second = system.evaluate(

        second_event,

        second_valence,

        cycle=2,

    )

    assert len(
        first
    ) == 1

    assert len(
        second
    ) == 1


# ============================================================
# 14. VALÊNCIA DA MEMÓRIA
# ============================================================


def test_memory_valence_is_similarity_weighted():
    patterns = [

        PatternMatch(

            pattern_id=
                "p1",

            stimulus_type=
                "danger",

            similarity=
                1.0,

            average_valence=
                -0.8,

            frequency=
                10,

            weight=
                8.0,

            action_expectations={},

        ),

        PatternMatch(

            pattern_id=
                "p2",

            stimulus_type=
                "danger",

            similarity=
                0.5,

            average_valence=
                0.4,

            frequency=
                4,

            weight=
                2.0,

            action_expectations={},

        ),

    ]

    result = (
        DecisionModule
        ._calculate_memory_valence(
            patterns
        )
    )

    expected = (

        (
            -0.8
            *
            1.0
        )

        +

        (
            0.4
            *
            0.5
        )

    ) / 2

    assert result == pytest.approx(
        expected
    )


# ============================================================
# 15. EQUAÇÃO DE UTILIDADE EVP
# ============================================================


def test_decision_uses_exact_evp_utility_equation():
    """
    Testa:

        U(a|E) =
            O(a)
            *
            (
                αV(E)
                +
                βv_memory
            )
            -
            C(a)
    """

    event = StimulusEvent(

        source=
            StimulusSource.EXTERNAL,

        type=
            "resource",

        intensity=
            0.7,

        payload={},

    )

    valence = ValenceResult(

        stimulus_id=
            event.id,

        score=
            0.50,

        confidence=
            0.90,

        components={},

        raw_activation=
            0.5493,

    )

    patterns = [

        PatternMatch(

            pattern_id=
                "pattern-1",

            stimulus_type=
                "resource",

            similarity=
                1.0,

            average_valence=
                0.50,

            frequency=
                5,

            weight=
                3.0,

            action_expectations={},

        )

    ]

    actions = [

        ActionCandidate(
            name="action_a",
            expected_outcome=0.8,
            cost=0.1,
        ),

        ActionCandidate(
            name="action_b",
            expected_outcome=0.2,
            cost=0.0,
        ),

    ]

    decision = DecisionModule(

        alpha=
            0.6,

        beta=
            0.4,

        exploration_rate=
            0.0,

    )

    result = decision.choose(

        event=
            event,

        valence=
            valence,

        patterns=
            patterns,

        actions=
            actions,

    )

    # --------------------------------------------------------
    # VALÊNCIA INTEGRADA
    # --------------------------------------------------------
    #
    # 0.6(0.5) + 0.4(0.5)
    #
    # =
    #
    # 0.5
    #
    #
    # ACTION A:
    #
    # U =
    #
    # 0.8(0.5) - 0.1
    #
    # =
    #
    # 0.3
    #
    #
    # ACTION B:
    #
    # U =
    #
    # 0.2(0.5) - 0
    #
    # =
    #
    # 0.1
    # --------------------------------------------------------

    assert result.action == "action_a"

    assert result.score == pytest.approx(
        0.30
    )

    assert (

        result.components[
            "integrated_valence"
        ]

        ==

        pytest.approx(
            0.50
        )

    )


# ============================================================
# 16. BAIXA CONFIANÇA
# ============================================================


def test_low_confidence_disables_current_valence():
    """
    A arquitetura técnica utiliza V(E) no DecisionModule
    somente quando a confiança ultrapassa o limiar.

    Portanto:

        confidence < threshold

    implica:

        V_used = 0
    """

    event = make_danger_event()

    valence = ValenceResult(

        stimulus_id=
            event.id,

        score=
            -0.99,

        confidence=
            0.10,

        components={},

        raw_activation=
            -2.64,

    )

    actions = make_actions()

    decision = DecisionModule(
        exploration_rate=0.0
    )

    result = decision.choose(

        event=
            event,

        valence=
            valence,

        patterns=[],

        actions=
            actions,

    )

    assert (

        result.components[
            "current_valence"
        ]

        ==

        pytest.approx(
            0.0
        )

    )

    assert (

        result.components[
            "memory_valence"
        ]

        ==

        pytest.approx(
            0.0
        )

    )


# ============================================================
# 17. ε-GREEDY
# ============================================================


def test_epsilon_greedy_can_force_exploration():
    """
    ε = 1.0

    significa:

        explorar em 100% das decisões.
    """

    event = make_danger_event()

    valence = ValenceResult(

        stimulus_id=
            event.id,

        score=
            0.5,

        confidence=
            1.0,

        components={},

        raw_activation=
            0.55,

    )

    decision = DecisionModule(

        exploration_rate=
            1.0,

        random_seed=
            42,

    )

    result = decision.choose(

        event=
            event,

        valence=
            valence,

        patterns=[],

        actions=
            make_actions(),

    )

    assert (
        decision.last_trace
        is not None
    )

    assert (

        decision.last_trace.mode

        ==

        "exploration"

    )

    # Conforme nossa implementação:
    #
    # durante exploração, result.score = 0,
    # mas a utilidade calculada continua disponível
    # no DecisionTrace/components.
    assert result.score == pytest.approx(
        0.0
    )


# ============================================================
# 18. PERSISTÊNCIA DA PATTERN MEMORY
# ============================================================


def test_pattern_memory_survives_restart(
    tmp_path,
):
    path = (
        tmp_path
        /
        "memory.json"
    )

    first_memory = PatternMemory(
        storage_path=path
    )

    event = make_danger_event()

    features = extract_features(
        event,
        novelty=1.0,
    )

    first_memory.remember(

        stimulus_type=
            event.type,

        features=
            features,

        valence=
            -0.7,

    )

    # --------------------------------------------------------
    # SIMULA UM NOVO PROCESSO / RESTART
    # --------------------------------------------------------

    second_memory = PatternMemory(
        storage_path=path
    )

    assert len(
        second_memory.patterns
    ) == 1

    assert (

        second_memory.patterns[0]
        .average_valence

        ==

        pytest.approx(
            -0.7
        )

    )


# ============================================================
# 19. PERSISTÊNCIA DA VALÊNCIA APRENDIDA
# ============================================================


def test_valence_learning_survives_restart(
    tmp_path,
):
    path = (
        tmp_path
        /
        "valence.json"
    )

    event = make_danger_event()

    phi = extract_features(
        event,
        novelty=1.0,
    )

    first_engine = ValenceEngine(

        storage_path=
            path,

        load_state=
            False,

    )

    first_engine.learn(

        phi,

        outcome=
            -1.0,

    )

    learned_before = (

        first_engine
        .learned_weights
        .copy()

    )

    # --------------------------------------------------------
    # SIMULA NOVA EXECUÇÃO
    # --------------------------------------------------------

    second_engine = ValenceEngine(

        storage_path=
            path,

        load_state=
            True,

    )

    assert np.allclose(

        second_engine.learned_weights,

        learned_before,

    )

    assert (

        second_engine.experience_count

        ==

        1

    )