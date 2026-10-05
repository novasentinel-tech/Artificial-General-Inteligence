from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .decision import DecisionModule
from .features import extract_features
from .memory import PatternMemory
from .models import (
    ActionCandidate,
    DecisionResult,
    StimulusEvent,
    ValenceResult,
)
from .triggers import Trigger, TriggerSystem
from .valence import ValenceEngine


@dataclass
class CognitiveStep:
    """
    Representa um ciclo cognitivo completo.

    Não é apenas o resultado final da decisão.

    Ele preserva o estado intermediário da cognição para que
    possamos observar cientificamente como a resposta foi formada.
    """

    stimulus: StimulusEvent

    novelty: float

    features: list[float]

    valence: ValenceResult

    patterns: list[Any]

    triggers: list[Trigger]

    decision: DecisionResult


@dataclass
class ExperienceRecord:
    """
    Registro temporário de uma experiência ainda sem consequência.

    O sistema primeiro percebe e age.

    Só posteriormente recebe a consequência real do ambiente.
    """

    stimulus: StimulusEvent

    novelty: float

    features: np.ndarray

    decision: DecisionResult

    pattern_ids: list[str]


class CognitiveEngine:
    """
    Motor cognitivo central.

    Implementa o ciclo:

        Estímulo
            ↓
        Recuperação de memória
            ↓
        Novidade
            ↓
        Features cognitivas
            ↓
        Valência
            ↓
        Padrões
            ↓
        Gatilhos
            ↓
        Decisão
            ↓
        Ação
            ↓
        Consequência
            ↓
        Aprendizado

    O CognitiveEngine não contém a lógica interna de cada camada.

    Ele apenas coordena os subsistemas cognitivos.

    Isso mantém a arquitetura modular e coerente com o artigo.
    """

    def __init__(
        self,
        memory: PatternMemory | None = None,
        valence: ValenceEngine | None = None,
        triggers: TriggerSystem | None = None,
        decisions: DecisionModule | None = None,
    ) -> None:

        self.memory = memory or PatternMemory()

        self.valence = valence or ValenceEngine()

        self.triggers = triggers or TriggerSystem()

        self.decisions = decisions or DecisionModule()

        # Experiências que já produziram uma decisão,
        # mas ainda não receberam consequência.
        self._pending_experiences: dict[
            str,
            ExperienceRecord
        ] = {}

        # Quantidade de ciclos processados.
        self.cycle_count: int = 0

    # =========================================================
    # CICLO COGNITIVO PRINCIPAL
    # =========================================================

    def process(
        self,
        event: StimulusEvent,
        actions: list[ActionCandidate],
    ) -> CognitiveStep:
        """
        Processa um estímulo e produz uma resposta comportamental.

        Importante:

        Este método NÃO realiza aprendizado por consequência.

        O aprendizado ocorre depois através de:

            learn_from_outcome(...)

        Isso separa corretamente:

            percepção / decisão

        de:

            consequência / aprendizado
        """

        self._validate_event(event)

        # -----------------------------------------------------
        # 1. ESTIMATIVA DE NOVIDADE
        # -----------------------------------------------------
        #
        # Novidade representa o quanto a experiência atual
        # difere das experiências armazenadas anteriormente.
        #
        # A memória deve calcular isso comparando características
        # estruturais do estímulo com padrões conhecidos.
        # -----------------------------------------------------

        novelty = self._estimate_novelty(event)

        # -----------------------------------------------------
        # 2. EXTRAÇÃO DAS FEATURES COGNITIVAS
        # -----------------------------------------------------
        #
        # Segundo o modelo:
        #
        # φ(E) =
        #
        # [
        #   intensidade,
        #   novidade,
        #   urgência,
        #   alinhamento com objetivos,
        #   risco
        # ]
        #
        # Essas dimensões não são arbitrárias.
        #
        # Cada uma representa uma propriedade relevante
        # para avaliação adaptativa.
        # -----------------------------------------------------

        features = extract_features(
            event=event,
            novelty=novelty,
        )

        # -----------------------------------------------------
        # 3. AVALIAÇÃO DE VALÊNCIA
        # -----------------------------------------------------
        #
        # Valência:
        #
        # V(E) = tanh(wᵀ φ(E))
        #
        # Ela estima se o estímulo tende a produzir
        # consequências positivas ou negativas.
        #
        # Neste momento isso é uma PREDIÇÃO.
        #
        # A consequência real ainda não aconteceu.
        # -----------------------------------------------------

        valence = self.valence.evaluate(
            event=event,
            features=features,
        )

        # -----------------------------------------------------
        # 4. RECUPERAÇÃO DE PADRÕES RELACIONADOS
        # -----------------------------------------------------
        #
        # Agora buscamos experiências estruturalmente semelhantes.
        #
        # Isso permite ao sistema utilizar contexto histórico
        # antes da tomada de decisão.
        # -----------------------------------------------------

        patterns = self._find_similar_patterns(
            event=event,
            features=features,
        )

        # -----------------------------------------------------
        # 5. GATILHOS
        # -----------------------------------------------------
        #
        # Gatilhos existem para respostas rápidas.
        #
        # Exemplo:
        #
        # risco extremamente alto
        #
        # não deveria exigir raciocínio complexo para produzir
        # uma resposta defensiva.
        # -----------------------------------------------------

        triggers = self.triggers.evaluate(
            event=event,
            valence=valence,
        )

        # -----------------------------------------------------
        # 6. DECISÃO
        # -----------------------------------------------------
        #
        # A decisão integra:
        #
        # - estado do estímulo
        # - valência prevista
        # - gatilhos
        # - ações possíveis
        #
        # Futuramente também vamos incorporar explicitamente
        # os padrões recuperados no scoring.
        # -----------------------------------------------------

        decision = self.decisions.choose(
            event=event,
            valence=valence,
            triggers=triggers,
            actions=actions,
        )

        # -----------------------------------------------------
        # 7. REGISTRO DA EXPERIÊNCIA
        # -----------------------------------------------------
        #
        # A experiência precisa entrar na memória mesmo antes
        # da consequência.
        #
        # Porém sua consequência permanece desconhecida.
        #
        # Assim evitamos ensinar algo que ainda não aconteceu.
        # -----------------------------------------------------

        pattern_ids = self._remember_experience(
            event=event,
            features=features,
            valence=valence.score,
        )

        experience = ExperienceRecord(
            stimulus=event,
            novelty=novelty,
            features=features.copy(),
            decision=decision,
            pattern_ids=pattern_ids,
        )

        self._pending_experiences[event.id] = experience

        self.cycle_count += 1

        return CognitiveStep(
            stimulus=event,
            novelty=novelty,
            features=features.tolist(),
            valence=valence,
            patterns=patterns,
            triggers=triggers,
            decision=decision,
        )

    # =========================================================
    # APRENDIZADO PÓS-CONSEQUÊNCIA
    # =========================================================

    def learn_from_outcome(
        self,
        event: StimulusEvent,
        novelty: float | None = None,
        outcome: float = 0.0,
    ) -> None:
        """
        Atualiza o sistema após observar a consequência real.

        outcome deve estar no intervalo:

            -1.0 <= outcome <= +1.0

        Interpretação:

            -1.0 = consequência extremamente negativa

             0.0 = neutra

            +1.0 = extremamente positiva


        O aprendizado de valência segue conceitualmente:

            erro = resultado_real - resultado_previsto

            Δw = η × erro × φ(E)

        Isso permite que o sistema altere gradualmente
        sua avaliação de estímulos semelhantes.
        """

        if not -1.0 <= outcome <= 1.0:
            raise ValueError(
                "outcome deve estar entre -1.0 e +1.0"
            )

        # -----------------------------------------------------
        # Recuperamos a experiência que realmente produziu
        # aquela decisão.
        # -----------------------------------------------------

        experience = self._pending_experiences.get(
            event.id
        )

        if experience is not None:

            features = experience.features

            actual_novelty = experience.novelty

        else:

            # Compatibilidade para experimentos em que
            # learn_from_outcome seja chamado separadamente.

            if novelty is None:

                actual_novelty = self._estimate_novelty(
                    event
                )

            else:

                actual_novelty = novelty

            features = extract_features(
                event=event,
                novelty=actual_novelty,
            )

        # -----------------------------------------------------
        # 1. ATUALIZA VALÊNCIA APRENDIDA
        # -----------------------------------------------------

        self.valence.learn(
            features=features,
            outcome=outcome,
        )

        # -----------------------------------------------------
        # 2. REGISTRA CONSEQUÊNCIA NA MEMÓRIA
        # -----------------------------------------------------
        #
        # No artigo, padrões possuem consequências associadas:
        #
        # Pattern
        #     ↓
        # Action
        #     ↓
        # Outcome
        #
        # O nosso memory.py será atualizado para implementar
        # isso de forma explícita.
        # -----------------------------------------------------

        if experience is not None:

            self._record_pattern_outcome(
                experience=experience,
                outcome=outcome,
            )

        # -----------------------------------------------------
        # 3. FINALIZA EXPERIÊNCIA
        # -----------------------------------------------------

        self._pending_experiences.pop(
            event.id,
            None,
        )

    # =========================================================
    # NOVIDADE
    # =========================================================

    def _estimate_novelty(
        self,
        event: StimulusEvent,
    ) -> float:
        """
        Estima a novidade do estímulo.

        Idealmente:

            novelty = 1 - similaridade_com_melhor_padrão

        Portanto:

            padrão idêntico:
                novidade ≈ 0

            experiência totalmente nova:
                novidade ≈ 1


        Enquanto o novo memory.py não estiver implementado,
        mantemos compatibilidade com o método atual.
        """

        # A memória futura terá este método.
        if hasattr(
            self.memory,
            "estimate_novelty",
        ):

            novelty = self.memory.estimate_novelty(
                event
            )

            return self._clamp01(
                float(novelty)
            )

        # -----------------------------------------------------
        # Compatibilidade temporária.
        #
        # Criamos features estruturais sem atribuir novidade,
        # pois novidade é justamente o que estamos tentando
        # calcular.
        #
        # IMPORTANTE:
        #
        # No memory.py definitivo, a dimensão "novelty"
        # será ignorada durante essa comparação.
        # -----------------------------------------------------

        provisional_features = extract_features(
            event=event,
            novelty=0.0,
        )

        novelty = self.memory.novelty(
            provisional_features
        )

        return self._clamp01(
            float(novelty)
        )

    # =========================================================
    # MEMÓRIA
    # =========================================================

    def _find_similar_patterns(
        self,
        event: StimulusEvent,
        features: np.ndarray,
    ) -> list[Any]:
        """
        Recupera padrões relacionados à experiência atual.

        A implementação definitiva ficará em memory.py.

        O engine apenas solicita a busca.
        """

        if hasattr(
            self.memory,
            "find_similar",
        ):

            result = self.memory.find_similar(
                stimulus_type=event.type,
                features=features,
            )

            return list(result)

        return []

    def _remember_experience(
        self,
        event: StimulusEvent,
        features: np.ndarray,
        valence: float,
    ) -> list[str]:
        """
        Registra a experiência na memória.

        Retorna IDs dos padrões afetados quando o sistema
        de memória disponibilizar identificadores.
        """

        record = self.memory.remember(
            stimulus_type=event.type,
            features=features,
            valence=valence,
        )

        pattern_id = getattr(
            record,
            "id",
            None,
        )

        if pattern_id is None:
            return []

        return [
            str(pattern_id)
        ]

    def _record_pattern_outcome(
        self,
        experience: ExperienceRecord,
        outcome: float,
    ) -> None:
        """
        Liga:

            padrão
              +
            ação
              +
            consequência

        Essa associação é crucial.

        O agente não precisa apenas lembrar:

            "isso aconteceu"

        Ele precisa aprender:

            "quando isso aconteceu e eu fiz X,
             o resultado foi Y".
        """

        if not hasattr(
            self.memory,
            "record_outcome",
        ):
            return

        for pattern_id in experience.pattern_ids:

            self.memory.record_outcome(
                pattern_id=pattern_id,
                action_id=experience.decision.action,
                outcome=outcome,
            )

    # =========================================================
    # ESTADO INTERNO
    # =========================================================

    def internal_state(
        self,
    ) -> dict[str, Any]:
        """
        Fornece uma representação simples do próprio estado.

        Isso NÃO é ainda o Self-Monitor completo.

        Serve apenas como base observável para quando
        implementarmos a sexta camada da arquitetura.
        """

        pattern_count = 0

        if hasattr(
            self.memory,
            "patterns",
        ):

            try:
                pattern_count = len(
                    self.memory.patterns
                )

            except TypeError:
                pattern_count = 0

        return {

            "cognitive_cycles":
                self.cycle_count,

            "pending_experiences":
                len(
                    self._pending_experiences
                ),

            "learned_experiences":
                self.valence.experience_count,

            "known_patterns":
                pattern_count,

            "valence_weights":
                self.valence.weights.tolist(),

        }

    # =========================================================
    # VALIDAÇÕES
    # =========================================================

    @staticmethod
    def _validate_event(
        event: StimulusEvent,
    ) -> None:
        """
        Faz validações semânticas adicionais.

        Pydantic já verifica os tipos e intensidade.
        """

        if not event.type.strip():

            raise ValueError(
                "O estímulo precisa possuir um tipo."
            )

    @staticmethod
    def _clamp01(
        value: float,
    ) -> float:
        """
        Garante valor dentro de:

            [0, 1]
        """

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )