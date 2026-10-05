from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .models import (
    StimulusEvent,
    ValenceResult,
)


# ============================================================
# POLARIDADE DO GATILHO
# ============================================================

class TriggerPolarity(str, Enum):
    """
    Direção da valência responsável pela ativação.

    NEGATIVE:
        V(E) <= -θ

    POSITIVE:
        V(E) >= +θ

    ABSOLUTE:
        |V(E)| >= θ

    O modo ABSOLUTE é útil para regras experimentais que não
    precisam distinguir inicialmente a direção da valência.
    """

    NEGATIVE = "negative"
    POSITIVE = "positive"
    ABSOLUTE = "absolute"


# ============================================================
# ESCOPO DO COOLDOWN
# ============================================================

class CooldownScope(str, Enum):
    """
    Define o que significa "o mesmo gatilho" durante cooldown.

    GLOBAL:
        A regra inteira entra em cooldown.

    CONTEXT:
        O cooldown ocorre por context_id.

        Se não existir context_id, utiliza stimulus.type.

    STIMULUS_TYPE:
        O cooldown ocorre por tipo de estímulo.

    CONTEXT é o padrão porque reduz loops repetitivos sem
    impedir necessariamente uma reação a outro contexto.
    """

    GLOBAL = "global"
    CONTEXT = "context"
    STIMULUS_TYPE = "stimulus_type"


# ============================================================
# REGRA DE GATILHO
# ============================================================

@dataclass(frozen=True)
class TriggerRule:
    """
    Define quando uma valência suficientemente intensa deve
    entrar no caminho automático de processamento.

    IMPORTANTE:

    threshold e cooldown_cycles são parâmetros experimentais
    da implementação.

    O artigo estabelece a existência conceitual de um limiar
    de alta magnitude e de cooldown, mas não fixa valores
    numéricos universais.
    """

    id: str

    name: str

    polarity: TriggerPolarity

    threshold: float = 0.75

    cooldown_cycles: int = 3

    cooldown_scope: CooldownScope = (
        CooldownScope.CONTEXT
    )

    enabled: bool = True

    # A arquitetura central não determina qual ação concreta
    # corresponde a uma valência positiva/negativa.
    #
    # Isso depende do corpo/ambiente do agente.
    #
    # Exemplos possíveis em experimentos:
    #
    # "move_away"
    # "consume_resource"
    # "stop"
    #
    # Portanto None é perfeitamente válido.
    response_action: str | None = None

    def __post_init__(self) -> None:

        if not self.id.strip():
            raise ValueError(
                "TriggerRule.id não pode estar vazio."
            )

        if not self.name.strip():
            raise ValueError(
                "TriggerRule.name não pode estar vazio."
            )

        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(
                "threshold deve estar no intervalo (0, 1]."
            )

        if self.cooldown_cycles < 0:
            raise ValueError(
                "cooldown_cycles não pode ser negativo."
            )


# ============================================================
# GATILHO DISPARADO
# ============================================================

@dataclass(frozen=True)
class Trigger:
    """
    Instância concreta de uma regra ativada.

    Este é o objeto que circula pelo CognitiveEngine.

    priority deriva diretamente da magnitude da valência:

        priority ≈ 100 × |V(E)|

    Isso dá significado claro ao número.

    Valência de magnitude:

        0.76 -> prioridade ~76
        0.91 -> prioridade ~91
        0.99 -> prioridade ~99
    """

    name: str

    priority: int

    reason: str

    rule_id: str

    stimulus_id: str

    valence: float

    magnitude: float

    polarity: TriggerPolarity

    cycle: int

    automatic: bool = True

    response_action: str | None = None


# ============================================================
# RESULTADO DE RESOLUÇÃO
# ============================================================

@dataclass(frozen=True)
class TriggerResolution:
    """
    Resultado da hierarquia de gatilhos.

    Quando vários gatilhos estão ativos, o de maior prioridade
    domina o caminho automático.

    Isso NÃO significa que os outros desapareçam.

    Eles permanecem observáveis para memória, monitoramento
    e investigação experimental.
    """

    trigger: Trigger | None

    requires_automatic_path: bool

    response_action: str | None

    reason: str


# ============================================================
# EVENTO SUPRIMIDO POR COOLDOWN
# ============================================================

@dataclass(frozen=True)
class SuppressedTrigger:
    """
    Registro observável de um gatilho que teria sido disparado,
    mas foi impedido pelo mecanismo de cooldown.

    Isso será útil para detectar loops patológicos.
    """

    rule_id: str

    stimulus_id: str

    cycle: int

    remaining_cycles: int

    reason: str


# ============================================================
# SISTEMA DE GATILHOS
# ============================================================

class TriggerSystem:
    """
    Implementa o mecanismo de respostas automáticas do EVP.

    Fluxo:

        ValenceResult
             │
             ▼
        |V(E)| suficientemente alto?
             │
        ┌────┴─────┐
       não        sim
        │           │
        ▼           ▼
    processamento  Trigger
    deliberativo     │
                     ▼
                cooldown?
                     │
              ┌──────┴──────┐
             sim            não
              │              │
              ▼              ▼
          suprimir       disparar
                             │
                             ▼
                       hierarquia
                             │
                             ▼
                    resposta automática


    =========================================================
    PRINCÍPIO IMPORTANTE
    =========================================================

    Este módulo NÃO recalcula:

        risk
        urgency
        novelty
        goal_alignment

    Esses fatores já participaram de:

        φ(E)

    e consequentemente:

        V(E)

    Utilizá-los novamente diretamente no TriggerSystem
    causaria dupla contagem de evidência.
    """

    def __init__(
        self,
        rules: list[TriggerRule] | None = None,
    ) -> None:

        # Ciclo interno utilizado para cooldown determinístico.
        #
        # Preferimos ciclos cognitivos a segundos reais porque:
        #
        # - experimentos ficam reproduzíveis;
        # - velocidade da CPU não altera o comportamento;
        # - simulações aceleradas continuam coerentes.
        self._cycle: int = 0

        # key:
        #
        # (rule_id, cooldown_scope_key)
        #
        # value:
        #
        # último ciclo em que disparou.
        self._last_fired: dict[
            tuple[str, str],
            int
        ] = {}

        self._suppressed_last_cycle: list[
            SuppressedTrigger
        ] = []

        if rules is None:

            self.rules = (
                self._default_rules()
            )

        else:

            self.rules = list(
                rules
            )

        self._validate_rule_ids()

    # ========================================================
    # REGRAS PADRÃO
    # ========================================================

    @staticmethod
    def _default_rules(
        threshold: float = 0.75,
    ) -> list[TriggerRule]:
        """
        Regras mínimas coerentes com o artigo.

        O threshold 0.75 NÃO é uma constante teórica do EVP.

        É somente um valor inicial experimental.

        Posteriormente devemos validar diferentes thresholds
        empiricamente.
        """

        return [

            TriggerRule(

                id="high_negative_valence",

                name="High Negative Valence",

                polarity=TriggerPolarity.NEGATIVE,

                threshold=threshold,

                cooldown_cycles=3,

                cooldown_scope=(
                    CooldownScope.CONTEXT
                ),

            ),

            TriggerRule(

                id="high_positive_valence",

                name="High Positive Valence",

                polarity=TriggerPolarity.POSITIVE,

                threshold=threshold,

                cooldown_cycles=3,

                cooldown_scope=(
                    CooldownScope.CONTEXT
                ),

            ),

        ]

    # ========================================================
    # AVALIAÇÃO
    # ========================================================

    def evaluate(
        self,
        event: StimulusEvent,
        valence: ValenceResult,
        cycle: int | None = None,
    ) -> list[Trigger]:
        """
        Avalia as regras de gatilho para uma valência já
        calculada.

        A entrada principal é:

            V(E)

        e não as features individuais do estímulo.

        Isso é proposital e coerente com o modelo EVP.
        """

        current_cycle = (
            self._resolve_cycle(
                cycle
            )
        )

        score = float(
            valence.score
        )

        magnitude = abs(
            score
        )

        fired: list[
            Trigger
        ] = []

        suppressed: list[
            SuppressedTrigger
        ] = []

        for rule in self.rules:

            if not rule.enabled:
                continue

            # ------------------------------------------------
            # 1. A REGRA É COMPATÍVEL COM A VALÊNCIA?
            # ------------------------------------------------

            if not self._matches_rule(
                rule=rule,
                score=score,
                magnitude=magnitude,
            ):
                continue

            # ------------------------------------------------
            # 2. VERIFICA COOLDOWN
            # ------------------------------------------------

            cooldown_key = (
                self._cooldown_key(
                    rule=rule,
                    event=event,
                )
            )

            remaining = (
                self._remaining_cooldown(
                    rule=rule,
                    key=cooldown_key,
                    current_cycle=current_cycle,
                )
            )

            if remaining > 0:

                suppressed.append(

                    SuppressedTrigger(

                        rule_id=
                            rule.id,

                        stimulus_id=
                            event.id,

                        cycle=
                            current_cycle,

                        remaining_cycles=
                            remaining,

                        reason=(
                            "Gatilho semanticamente válido, "
                            "mas suprimido por cooldown."
                        ),

                    )

                )

                continue

            # ------------------------------------------------
            # 3. PRIORIDADE
            # ------------------------------------------------
            #
            # P = round(100 × |V(E)|)
            #
            # Não introduzimos pesos secretos.
            #
            # Quanto maior a magnitude da valência,
            # maior a prioridade.
            # ------------------------------------------------

            priority = (
                self._priority_from_magnitude(
                    magnitude
                )
            )

            # ------------------------------------------------
            # 4. REGISTRA DISPARO
            # ------------------------------------------------

            self._last_fired[
                cooldown_key
            ] = current_cycle

            trigger = Trigger(

                name=
                    rule.name,

                priority=
                    priority,

                reason=
                    self._build_reason(
                        rule=rule,
                        score=score,
                        magnitude=magnitude,
                    ),

                rule_id=
                    rule.id,

                stimulus_id=
                    event.id,

                valence=
                    score,

                magnitude=
                    magnitude,

                polarity=
                    self._score_polarity(
                        score
                    ),

                cycle=
                    current_cycle,

                automatic=True,

                response_action=
                    rule.response_action,

            )

            fired.append(
                trigger
            )

        # ----------------------------------------------------
        # HIERARQUIA
        # ----------------------------------------------------

        fired.sort(

            key=lambda trigger:
                trigger.priority,

            reverse=True,

        )

        self._suppressed_last_cycle = (
            suppressed
        )

        return fired

    # ========================================================
    # RESOLUÇÃO DO CAMINHO AUTOMÁTICO
    # ========================================================

    def resolve(
        self,
        triggers: list[Trigger],
    ) -> TriggerResolution:
        """
        Determina se o processamento deliberativo deve ser
        interrompido.

        Conforme o modelo:

            alta magnitude de valência
            ->
            mecanismo automático

        Se vários gatilhos estiverem ativos, a hierarquia
        seleciona o de maior prioridade.
        """

        if not triggers:

            return TriggerResolution(

                trigger=None,

                requires_automatic_path=False,

                response_action=None,

                reason=(
                    "Nenhum gatilho de alta magnitude ativo."
                ),

            )

        dominant = max(

            triggers,

            key=lambda trigger:
                trigger.priority,

        )

        if dominant.response_action is None:

            reason = (

                "Gatilho automático ativado, mas nenhuma "
                "ação corporal/ambiental foi associada "
                "a esta regra."

            )

        else:

            reason = (

                "Gatilho automático dominante selecionado "
                "pela hierarquia de prioridade."

            )

        return TriggerResolution(

            trigger=
                dominant,

            requires_automatic_path=True,

            response_action=
                dominant.response_action,

            reason=
                reason,

        )

    # ========================================================
    # MATCH DA REGRA
    # ========================================================

    @staticmethod
    def _matches_rule(
        rule: TriggerRule,
        score: float,
        magnitude: float,
    ) -> bool:
        """
        Determina formalmente se uma regra deve ser candidata.

        NEGATIVE:

            V(E) <= -θ

        POSITIVE:

            V(E) >= +θ

        ABSOLUTE:

            |V(E)| >= θ
        """

        if (
            rule.polarity
            ==
            TriggerPolarity.NEGATIVE
        ):

            return (
                score
                <=
                -rule.threshold
            )

        if (
            rule.polarity
            ==
            TriggerPolarity.POSITIVE
        ):

            return (
                score
                >=
                rule.threshold
            )

        if (
            rule.polarity
            ==
            TriggerPolarity.ABSOLUTE
        ):

            return (
                magnitude
                >=
                rule.threshold
            )

        return False

    # ========================================================
    # PRIORIDADE
    # ========================================================

    @staticmethod
    def _priority_from_magnitude(
        magnitude: float,
    ) -> int:
        """
        Converte magnitude de valência em prioridade.

            priority =
                round(100 × |V(E)|)

        Portanto a prioridade preserva uma interpretação
        diretamente observável.
        """

        magnitude = max(
            0.0,
            min(
                1.0,
                float(
                    magnitude
                ),
            ),
        )

        return int(
            round(
                100.0
                *
                magnitude
            )
        )

    # ========================================================
    # COOLDOWN
    # ========================================================

    @staticmethod
    def _cooldown_scope_value(
        rule: TriggerRule,
        event: StimulusEvent,
    ) -> str:
        """
        Produz a identidade contextual utilizada pelo cooldown.
        """

        if (
            rule.cooldown_scope
            ==
            CooldownScope.GLOBAL
        ):

            return "__global__"

        if (
            rule.cooldown_scope
            ==
            CooldownScope.STIMULUS_TYPE
        ):

            return event.type

        # CONTEXT
        #
        # context_id agrupa estímulos pertencentes ao mesmo
        # episódio.
        #
        # Se ele não existir, usamos stimulus.type como fallback.
        return (
            event.context_id
            or
            event.type
        )

    def _cooldown_key(
        self,
        rule: TriggerRule,
        event: StimulusEvent,
    ) -> tuple[str, str]:

        scope = (
            self._cooldown_scope_value(
                rule=rule,
                event=event,
            )
        )

        return (
            rule.id,
            scope,
        )

    def _remaining_cooldown(
        self,
        rule: TriggerRule,
        key: tuple[str, str],
        current_cycle: int,
    ) -> int:
        """
        Calcula quantos ciclos faltam.

        Exemplo:

            cooldown = 3
            disparou ciclo 10

        ciclos bloqueados:

            11
            12

        novo disparo possível:

            13
        """

        if rule.cooldown_cycles == 0:

            return 0

        last_cycle = (
            self._last_fired.get(
                key
            )
        )

        if last_cycle is None:

            return 0

        elapsed = (
            current_cycle
            -
            last_cycle
        )

        remaining = (
            rule.cooldown_cycles
            -
            elapsed
        )

        return max(
            0,
            remaining,
        )

    # ========================================================
    # CICLO
    # ========================================================

    def _resolve_cycle(
        self,
        cycle: int | None,
    ) -> int:
        """
        Se CognitiveEngine fornecer o número do ciclo,
        utilizamos ele.

        Caso contrário o TriggerSystem mantém seu próprio
        contador interno para compatibilidade.
        """

        if cycle is None:

            self._cycle += 1

            return (
                self._cycle
            )

        if cycle < 0:

            raise ValueError(
                "cycle não pode ser negativo."
            )

        self._cycle = max(
            self._cycle,
            cycle,
        )

        return cycle

    # ========================================================
    # POLARIDADE REAL DO SCORE
    # ========================================================

    @staticmethod
    def _score_polarity(
        score: float,
    ) -> TriggerPolarity:

        if score < 0.0:

            return (
                TriggerPolarity.NEGATIVE
            )

        if score > 0.0:

            return (
                TriggerPolarity.POSITIVE
            )

        return (
            TriggerPolarity.ABSOLUTE
        )

    # ========================================================
    # EXPLICAÇÃO
    # ========================================================

    @staticmethod
    def _build_reason(
        rule: TriggerRule,
        score: float,
        magnitude: float,
    ) -> str:
        """
        Produz explicação legível para auditoria do estado
        cognitivo.
        """

        return (

            f"V(E)={score:+.4f}; "
            f"|V(E)|={magnitude:.4f}; "
            f"threshold={rule.threshold:.4f}; "
            f"polarity={rule.polarity.value}"

        )

    # ========================================================
    # CONFIGURAÇÃO DINÂMICA
    # ========================================================

    def register_rule(
        self,
        rule: TriggerRule,
    ) -> None:
        """
        Registra nova regra experimental.

        IDs duplicados são proibidos.
        """

        if any(
            existing.id
            ==
            rule.id

            for existing
            in self.rules
        ):

            raise ValueError(
                f"Já existe uma TriggerRule com id "
                f"'{rule.id}'."
            )

        self.rules.append(
            rule
        )

    def remove_rule(
        self,
        rule_id: str,
    ) -> None:
        """
        Remove uma regra do sistema.
        """

        self.rules = [

            rule

            for rule
            in self.rules

            if rule.id
            !=
            rule_id

        ]

    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    def _validate_rule_ids(
        self,
    ) -> None:

        ids = [
            rule.id
            for rule
            in self.rules
        ]

        if (
            len(ids)
            !=
            len(
                set(ids)
            )
        ):

            raise ValueError(
                "TriggerRule IDs precisam ser únicos."
            )

    # ========================================================
    # OBSERVABILIDADE
    # ========================================================

    @property
    def suppressed_last_cycle(
        self,
    ) -> tuple[SuppressedTrigger, ...]:

        return tuple(
            self._suppressed_last_cycle
        )

    def snapshot(
        self,
    ) -> dict[str, Any]:
        """
        Estado observável do TriggerSystem.

        Ideal para o futuro monitor em Rich.
        """

        rules = [

            {

                "id":
                    rule.id,

                "name":
                    rule.name,

                "polarity":
                    rule.polarity.value,

                "threshold":
                    rule.threshold,

                "cooldown_cycles":
                    rule.cooldown_cycles,

                "cooldown_scope":
                    rule.cooldown_scope.value,

                "response_action":
                    rule.response_action,

                "enabled":
                    rule.enabled,

            }

            for rule
            in self.rules

        ]

        cooldowns = [

            {

                "rule_id":
                    rule_id,

                "scope":
                    scope,

                "last_fired_cycle":
                    cycle,

            }

            for (
                rule_id,
                scope
            ),
            cycle
            in self._last_fired.items()

        ]

        suppressed = [

            {

                "rule_id":
                    item.rule_id,

                "stimulus_id":
                    item.stimulus_id,

                "cycle":
                    item.cycle,

                "remaining_cycles":
                    item.remaining_cycles,

                "reason":
                    item.reason,

            }

            for item
            in self._suppressed_last_cycle

        ]

        return {

            "cycle":
                self._cycle,

            "rules":
                rules,

            "cooldowns":
                cooldowns,

            "suppressed_last_cycle":
                suppressed,

        }