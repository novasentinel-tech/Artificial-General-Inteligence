from __future__ import annotations

from pprint import pprint
from time import sleep

from .engine import CognitiveEngine
from .models import (
    ActionCandidate,
    StimulusEvent,
    StimulusSource,
)


def print_separator(title: str) -> None:
    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def show_step(step) -> None:
    """
    Exibe o estado cognitivo produzido pelo motor.

    O objetivo aqui é permitir observar:

    - estímulo recebido
    - novidade percebida
    - valência atribuída
    - confiança da avaliação
    - gatilhos ativados
    - decisão tomada
    """

    print("\n[ESTÍMULO]")
    pprint(step.stimulus.model_dump())

    print("\n[NOVIDADE]")
    print(f"{step.novelty:.4f}")

    print("\n[VALÊNCIA]")
    print(f"Score:      {step.valence.score:.4f}")
    print(f"Confiança:  {step.valence.confidence:.4f}")

    print("\nComponentes da valência:")

    for component, value in step.valence.components.items():
        print(
            f"  {component:<20} "
            f"{value:+.4f}"
        )

    print("\n[GATILHOS]")

    if not step.triggers:
        print("Nenhum gatilho ativado.")

    else:
        for trigger in step.triggers:
            print(
                f"- {trigger.name} "
                f"(prioridade={trigger.priority}) "
                f"-> {trigger.reason}"
            )

    print("\n[DECISÃO]")
    print(f"Ação escolhida: {step.decision.action}")
    print(f"Score:          {step.decision.score:.4f}")
    print(f"Motivo:         {step.decision.reason}")


def main() -> None:

    print_separator(
        "AGI EVP — MODELO EXPERIMENTAL"
    )

    print(
        "Ciclo Cognitivo:\n"
        "\n"
        "Estímulo\n"
        "   ↓\n"
        "Avaliação de novidade\n"
        "   ↓\n"
        "Extração de características\n"
        "   ↓\n"
        "Avaliação de valência\n"
        "   ↓\n"
        "Reconhecimento de padrões\n"
        "   ↓\n"
        "Ativação de gatilhos\n"
        "   ↓\n"
        "Seleção de resposta\n"
        "   ↓\n"
        "Consequência\n"
        "   ↓\n"
        "Aprendizado\n"
    )

    engine = CognitiveEngine()

    # ---------------------------------------------------------
    # AÇÕES DISPONÍVEIS AO AGENTE
    # ---------------------------------------------------------

    actions = [

        ActionCandidate(
            name="observe",
            base_utility=0.20,
            risk_penalty=0.10,
        ),

        ActionCandidate(
            name="investigate",
            base_utility=0.70,
            risk_penalty=0.80,
        ),

        ActionCandidate(
            name="avoid",
            base_utility=0.40,
            risk_penalty=0.00,
        ),

    ]

    # =========================================================
    # EXPERIÊNCIA 1
    #
    # O agente nunca viu esta situação.
    #
    # Consequentemente:
    #
    # novelty ≈ alta
    # confiança da valência ≈ baixa
    # memória ainda vazia
    # =========================================================

    print_separator(
        "EXPERIÊNCIA 1 — PRIMEIRO CONTATO"
    )

    stimulus_1 = StimulusEvent(

        source=StimulusSource.EXTERNAL,

        type="unknown_environment_event",

        intensity=0.90,

        payload={

            "description":
                "Um evento desconhecido ocorre próximo ao agente.",

            "urgency": 0.85,

            "goal_alignment": -0.20,

            "risk_level": 0.90,

        },

        context_id="experiment-danger-001",
    )

    step_1 = engine.process(
        stimulus_1,
        actions,
    )

    show_step(step_1)

    # ---------------------------------------------------------
    # CONSEQUÊNCIA REAL
    # ---------------------------------------------------------
    #
    # Após agir, o ambiente fornece uma consequência.
    #
    # Neste experimento:
    #
    # outcome = -0.9
    #
    # Isso significa que a situação produziu
    # uma consequência fortemente negativa.
    #
    # O Valence Engine deve então atualizar seus pesos.
    # ---------------------------------------------------------

    outcome_1 = -0.90

    print("\n[CONSEQUÊNCIA OBSERVADA]")

    print(
        f"Resultado real da experiência: "
        f"{outcome_1:+.2f}"
    )

    engine.learn_from_outcome(
        event=stimulus_1,
        novelty=step_1.novelty,
        outcome=outcome_1,
    )

    print(
        "\nO sistema atualizou seus pesos "
        "com base no erro de predição."
    )

    sleep(1)

    # =========================================================
    # EXPERIÊNCIA 2
    #
    # O agente encontra situação muito semelhante.
    #
    # Esperamos:
    #
    # menor novidade
    # valência mais negativa
    # maior confiança
    # reconhecimento do padrão
    # comportamento potencialmente diferente
    # =========================================================

    print_separator(
        "EXPERIÊNCIA 2 — SITUAÇÃO SEMELHANTE"
    )

    stimulus_2 = StimulusEvent(

        source=StimulusSource.EXTERNAL,

        type="unknown_environment_event",

        intensity=0.88,

        payload={

            "description":
                "Evento semelhante ao anterior aparece novamente.",

            "urgency": 0.82,

            "goal_alignment": -0.25,

            "risk_level": 0.87,

        },

        context_id="experiment-danger-002",
    )

    step_2 = engine.process(
        stimulus_2,
        actions,
    )

    show_step(step_2)

    outcome_2 = -0.75

    print("\n[CONSEQUÊNCIA OBSERVADA]")

    print(
        f"Resultado real: "
        f"{outcome_2:+.2f}"
    )

    engine.learn_from_outcome(
        event=stimulus_2,
        novelty=step_2.novelty,
        outcome=outcome_2,
    )

    sleep(1)

    # =========================================================
    # EXPERIÊNCIA 3
    #
    # Agora apresentamos um estímulo diferente,
    # associado a consequência positiva.
    #
    # Queremos observar se o sistema
    # consegue formar outra associação.
    # =========================================================

    print_separator(
        "EXPERIÊNCIA 3 — ESTÍMULO POSITIVO"
    )

    stimulus_3 = StimulusEvent(

        source=StimulusSource.EXTERNAL,

        type="resource_found",

        intensity=0.70,

        payload={

            "description":
                "O agente encontra um recurso útil.",

            "urgency": 0.20,

            "goal_alignment": 0.90,

            "risk_level": 0.05,

        },

        context_id="experiment-resource-001",
    )

    step_3 = engine.process(
        stimulus_3,
        actions,
    )

    show_step(step_3)

    outcome_3 = 0.90

    print("\n[CONSEQUÊNCIA OBSERVADA]")

    print(
        f"Resultado real: "
        f"{outcome_3:+.2f}"
    )

    engine.learn_from_outcome(
        event=stimulus_3,
        novelty=step_3.novelty,
        outcome=outcome_3,
    )

    sleep(1)

    # =========================================================
    # EXPERIÊNCIA 4
    #
    # O estímulo positivo aparece novamente.
    #
    # A arquitetura deve começar a atribuir
    # valência positiva ao padrão.
    # =========================================================

    print_separator(
        "EXPERIÊNCIA 4 — RECONHECIMENTO POSITIVO"
    )

    stimulus_4 = StimulusEvent(

        source=StimulusSource.EXTERNAL,

        type="resource_found",

        intensity=0.72,

        payload={

            "description":
                "Outro recurso semelhante é encontrado.",

            "urgency": 0.15,

            "goal_alignment": 0.95,

            "risk_level": 0.02,

        },

        context_id="experiment-resource-002",
    )

    step_4 = engine.process(
        stimulus_4,
        actions,
    )

    show_step(step_4)

    # =========================================================
    # ESTADO DA MEMÓRIA
    # =========================================================

    print_separator(
        "MEMÓRIA DE PADRÕES"
    )

    patterns = engine.memory.patterns

    if not patterns:

        print(
            "Nenhum padrão armazenado."
        )

    else:

        for index, pattern in enumerate(
            patterns,
            start=1,
        ):

            print(
                f"\nPadrão #{index}"
            )

            print(
                f"Tipo: "
                f"{pattern.stimulus_type}"
            )

            print(
                f"Frequência: "
                f"{pattern.frequency}"
            )

            print(
                f"Valência média: "
                f"{pattern.average_valence:+.4f}"
            )

            print(
                "Vetor:"
            )

            print(
                [
                    round(value, 4)
                    for value
                    in pattern.feature_vector
                ]
            )

    # =========================================================
    # ESTADO DO MOTOR DE VALÊNCIA
    # =========================================================

    print_separator(
        "ESTADO INTERNO DO VALENCE ENGINE"
    )

    print(
        "Número de experiências aprendidas:",
        engine.valence.experience_count,
    )

    print(
        "\nPesos atuais:"
    )

    for index, weight in enumerate(
        engine.valence.weights
    ):

        print(
            f"w{index}: "
            f"{weight:+.6f}"
        )

    # =========================================================
    # FIM
    # =========================================================

    print_separator(
        "EXPERIMENTO FINALIZADO"
    )

    print(
        "O agente processou estímulos, "
        "formou padrões, recebeu consequências "
        "e modificou seu estado interno."
    )


if __name__ == "__main__":
    main()