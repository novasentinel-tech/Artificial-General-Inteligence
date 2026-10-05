from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

import json
import os

import numpy as np

from .features import DIMENSIONS
from .models import StimulusEvent, ValenceResult


# ============================================================
# CONFIGURAÇÕES
# ============================================================

VALENCE_SCHEMA_VERSION = 1

DEFAULT_LEARNING_RATE = 0.05

# Esse valor NÃO faz parte da teoria EVP.
#
# É apenas uma constante operacional utilizada para transformar
# quantidade de experiências em uma medida observável de
# confiança.
#
# experience_count = 20
# confidence = 0.5
DEFAULT_CONFIDENCE_HALF_SATURATION = 20.0


# ============================================================
# RESULTADO DE APRENDIZADO
# ============================================================

@dataclass(frozen=True)
class ValenceLearningUpdate:
    """
    Registro de uma atualização do motor de valência.

    Esse objeto existe principalmente para:

    - monitoramento;
    - experimentos;
    - auditoria matemática;
    - visualização futura no terminal.

    Ele não participa diretamente da cognição.
    """

    outcome: float

    raw_prediction_before: float

    squashed_prediction_before: float

    prediction_error: float

    learning_rate: float

    delta: list[float]

    learned_weights_before: list[float]

    learned_weights_after: list[float]


# ============================================================
# MOTOR DE VALÊNCIA
# ============================================================

class ValenceEngine:
    """
    Motor responsável pela construção interna da valência.

    No modelo EVP:

        estímulo
            ↓
        características φ(E)
            ↓
        ativação linear
            ↓
        z = wᵀφ(E)
            ↓
        tanh
            ↓
        V(E) ∈ [-1, +1]


    A valência NÃO é fornecida pelo ambiente.

    Ela é produzida internamente pelo sistema.


    =========================================================
    DOIS COMPONENTES
    =========================================================

    O artigo descreve:

    1. valência intrínseca;
    2. valência aprendida.

    Representamos isso como:

        w_total =
            w_intrinsic
            +
            w_learned

    portanto:

        z =
            (w_intrinsic + w_learned)ᵀ φ(E)

    Essa decomposição continua matematicamente equivalente a:

        z = wᵀφ(E)


    =========================================================
    APRENDIZADO
    =========================================================

    Após observar uma consequência real r:

        δ = r - wᵀφ(E)

    e:

        Δw =
            η × δ × φ(E)

    Somente o componente aprendido é atualizado:

        w_learned ← w_learned + Δw

    Os pesos intrínsecos permanecem constantes durante
    esse mecanismo de aprendizado.
    """

    def __init__(
        self,
        intrinsic_weights: (
            Mapping[str, float]
            | Sequence[float]
            | np.ndarray
            | None
        ) = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        confidence_half_saturation: float = (
            DEFAULT_CONFIDENCE_HALF_SATURATION
        ),
        storage_path: str | Path | None = None,
        load_state: bool = True,
    ) -> None:

        if learning_rate <= 0.0:

            raise ValueError(
                "learning_rate deve ser maior que zero."
            )

        if confidence_half_saturation <= 0.0:

            raise ValueError(
                "confidence_half_saturation deve ser maior "
                "que zero."
            )

        self.learning_rate = float(
            learning_rate
        )

        self.confidence_half_saturation = float(
            confidence_half_saturation
        )

        self._lock = RLock()

        # ----------------------------------------------------
        # PESOS
        # ----------------------------------------------------

        # Mantemos a informação de se os pesos intrínsecos
        # foram fornecidos explicitamente.
        #
        # Isso será importante durante carregamento do estado.
        self._intrinsic_explicitly_provided = (
            intrinsic_weights is not None
        )

        self.intrinsic_weights = (
            self._coerce_weights(
                intrinsic_weights
            )
        )

        self.learned_weights = np.zeros(
            len(DIMENSIONS),
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # ESTADO DE APRENDIZADO
        # ----------------------------------------------------

        self.experience_count: int = 0

        self.last_prediction_error: float | None = None

        self.last_outcome: float | None = None

        # ----------------------------------------------------
        # PERSISTÊNCIA
        # ----------------------------------------------------

        if storage_path is None:

            project_root = (
                Path(__file__)
                .resolve()
                .parents[2]
            )

            storage_path = (
                project_root
                /
                "data"
                /
                "brain"
                /
                "valence.json"
            )

        self.storage_path = Path(
            storage_path
        )

        if load_state:

            self._load()

    # ========================================================
    # PESO TOTAL
    # ========================================================

    @property
    def weights(
        self,
    ) -> np.ndarray:
        """
        Vetor total utilizado na função:

            V(E) = tanh(wᵀφ(E))

        onde:

            w =
                w_intrinsic
                +
                w_learned
        """

        return (
            self.intrinsic_weights
            +
            self.learned_weights
        ).copy()

    # ========================================================
    # AVALIAÇÃO
    # ========================================================

    def evaluate(
        self,
        event: StimulusEvent,
        features: np.ndarray,
    ) -> ValenceResult:
        """
        Calcula a valência interna atribuída ao estímulo.

        Importante:

        Nenhuma consequência real é conhecida neste momento.

        Portanto esta função representa uma PREDIÇÃO
        construída pelo estado interno atual do sistema.
        """

        phi = self._validate_features(
            features
        )

        with self._lock:

            total_weights = (
                self.intrinsic_weights
                +
                self.learned_weights
            )

            # ------------------------------------------------
            # ATIVAÇÃO LINEAR
            # ------------------------------------------------
            #
            # z = wᵀφ(E)
            # ------------------------------------------------

            raw_activation = float(

                np.dot(
                    total_weights,
                    phi,
                )

            )

            # ------------------------------------------------
            # VALÊNCIA
            # ------------------------------------------------
            #
            # V(E) = tanh(z)
            #
            # tanh garante:
            #
            # -1 < V(E) < +1
            # ------------------------------------------------

            score = float(
                np.tanh(
                    raw_activation
                )
            )

            # ------------------------------------------------
            # DECOMPOSIÇÃO
            # ------------------------------------------------
            #
            # contribuição_i =
            #
            #     w_i × φ_i(E)
            #
            # A soma das contribuições é exatamente:
            #
            #     wᵀφ(E)
            # ------------------------------------------------

            components = {

                dimension:
                    float(
                        total_weights[index]
                        *
                        phi[index]
                    )

                for index, dimension
                in enumerate(
                    DIMENSIONS
                )

            }

            confidence = (
                self._calculate_confidence()
            )

        return ValenceResult(

            stimulus_id=
                event.id,

            score=
                score,

            confidence=
                confidence,

            components=
                components,

            raw_activation=
                raw_activation,

        )

    # ========================================================
    # APRENDIZADO
    # ========================================================

    def learn(
        self,
        features: np.ndarray,
        outcome: float,
        learning_rate: float | None = None,
    ) -> ValenceLearningUpdate:
        """
        Atualiza os pesos aprendidos após observar
        uma consequência real.

        A equação implementada é exatamente:

            Δw =
                η
                ×
                (r - wᵀφ(E))
                ×
                φ(E)

        onde:

            η = taxa de aprendizado

            r = consequência observada

            w = vetor total atual

            φ(E) = vetor de características


        IMPORTANTE:

        O erro utiliza:

            wᵀφ(E)

        e NÃO:

            tanh(wᵀφ(E))

        porque essa é a formulação matemática apresentada
        no modelo EVP original.
        """

        phi = self._validate_features(
            features
        )

        outcome = float(
            outcome
        )

        if not np.isfinite(
            outcome
        ):

            raise ValueError(
                "outcome precisa ser um número finito."
            )

        if not -1.0 <= outcome <= 1.0:

            raise ValueError(
                "outcome deve estar entre -1.0 e +1.0."
            )

        if learning_rate is None:

            eta = self.learning_rate

        else:

            eta = float(
                learning_rate
            )

            if eta <= 0.0:

                raise ValueError(
                    "learning_rate deve ser maior que zero."
                )

        with self._lock:

            # ------------------------------------------------
            # PESOS ANTES DO APRENDIZADO
            # ------------------------------------------------

            learned_before = (
                self.learned_weights.copy()
            )

            total_weights = (
                self.intrinsic_weights
                +
                self.learned_weights
            )

            # ------------------------------------------------
            # PREDIÇÃO LINEAR
            # ------------------------------------------------
            #
            # z = wᵀφ(E)
            # ------------------------------------------------

            raw_prediction = float(

                np.dot(
                    total_weights,
                    phi,
                )

            )

            squashed_prediction = float(

                np.tanh(
                    raw_prediction
                )

            )

            # ------------------------------------------------
            # ERRO DE PREDIÇÃO
            # ------------------------------------------------
            #
            # δ = r - wᵀφ(E)
            # ------------------------------------------------

            prediction_error = (

                outcome
                -
                raw_prediction

            )

            # ------------------------------------------------
            # REGRA DELTA
            # ------------------------------------------------
            #
            # Δw =
            #
            # η × δ × φ(E)
            # ------------------------------------------------

            delta = (

                eta

                *

                prediction_error

                *

                phi

            )

            # ------------------------------------------------
            # SOMENTE O COMPONENTE APRENDIDO MUDA
            # ------------------------------------------------

            self.learned_weights += (
                delta
            )

            self.experience_count += 1

            self.last_prediction_error = (
                float(
                    prediction_error
                )
            )

            self.last_outcome = (
                outcome
            )

            learned_after = (
                self.learned_weights.copy()
            )

            # Persistimos imediatamente.
            self._save_locked()

        return ValenceLearningUpdate(

            outcome=
                outcome,

            raw_prediction_before=
                raw_prediction,

            squashed_prediction_before=
                squashed_prediction,

            prediction_error=
                float(
                    prediction_error
                ),

            learning_rate=
                eta,

            delta=
                delta.tolist(),

            learned_weights_before=
                learned_before.tolist(),

            learned_weights_after=
                learned_after.tolist(),

        )

    # ========================================================
    # COMPONENTE INTRÍNSECO
    # ========================================================

    def intrinsic_activation(
        self,
        features: np.ndarray,
    ) -> float:
        """
        Retorna apenas:

            z_intrinsic =
                w_intrinsicᵀ φ(E)

        Útil para experimentos que desejem distinguir
        componente intrínseco do aprendido.
        """

        phi = self._validate_features(
            features
        )

        with self._lock:

            return float(

                np.dot(
                    self.intrinsic_weights,
                    phi,
                )

            )

    # ========================================================
    # COMPONENTE APRENDIDO
    # ========================================================

    def learned_activation(
        self,
        features: np.ndarray,
    ) -> float:
        """
        Retorna apenas:

            z_learned =
                w_learnedᵀ φ(E)
        """

        phi = self._validate_features(
            features
        )

        with self._lock:

            return float(

                np.dot(
                    self.learned_weights,
                    phi,
                )

            )

    # ========================================================
    # BREAKDOWN
    # ========================================================

    def activation_breakdown(
        self,
        features: np.ndarray,
    ) -> dict[str, Any]:
        """
        Expõe de forma interpretável a origem da valência.

        Isso será extremamente útil no monitor.py.
        """

        phi = self._validate_features(
            features
        )

        with self._lock:

            intrinsic = (
                self.intrinsic_weights
                *
                phi
            )

            learned = (
                self.learned_weights
                *
                phi
            )

            total = (
                intrinsic
                +
                learned
            )

        return {

            "features": {

                dimension:
                    float(
                        phi[index]
                    )

                for index, dimension
                in enumerate(
                    DIMENSIONS
                )

            },

            "intrinsic": {

                dimension:
                    float(
                        intrinsic[index]
                    )

                for index, dimension
                in enumerate(
                    DIMENSIONS
                )

            },

            "learned": {

                dimension:
                    float(
                        learned[index]
                    )

                for index, dimension
                in enumerate(
                    DIMENSIONS
                )

            },

            "total": {

                dimension:
                    float(
                        total[index]
                    )

                for index, dimension
                in enumerate(
                    DIMENSIONS
                )

            },

            "raw_activation":
                float(
                    np.sum(
                        total
                    )
                ),

            "valence":
                float(
                    np.tanh(
                        np.sum(
                            total
                        )
                    )
                ),

        }

    # ========================================================
    # CONFIANÇA
    # ========================================================

    def _calculate_confidence(
        self,
    ) -> float:
        """
        Métrica operacional de confiança.

        ATENÇÃO:

        Essa equação NÃO faz parte da formulação matemática
        original do EVP.

        Ela existe apenas para observabilidade.

        Utilizamos:

                    n
            C = ---------
                n + k

        onde:

            n = experiências aprendidas

            k = confidence_half_saturation

        Consequentemente:

            n = 0
            C = 0

            n = k
            C = 0.5

            n → ∞
            C → 1
        """

        n = float(
            self.experience_count
        )

        k = (
            self.confidence_half_saturation
        )

        return float(

            n
            /
            (
                n
                +
                k
            )

        )

    # ========================================================
    # PESOS INTRÍNSECOS
    # ========================================================

    def set_intrinsic_weights(
        self,
        weights: (
            Mapping[str, float]
            | Sequence[float]
            | np.ndarray
        ),
        save: bool = True,
    ) -> None:
        """
        Permite configurar experimentalmente os pesos
        intrínsecos.

        IMPORTANTE:

        A arquitetura EVP original não especifica um protocolo
        definitivo para inicialização destes pesos.

        Portanto qualquer valor definido aqui deve ser
        documentado como condição experimental.
        """

        vector = self._coerce_weights(
            weights
        )

        with self._lock:

            self.intrinsic_weights = (
                vector
            )

            self._intrinsic_explicitly_provided = (
                True
            )

            if save:

                self._save_locked()

    # ========================================================
    # RESET DE APRENDIZADO
    # ========================================================

    def reset_learning(
        self,
        save: bool = True,
    ) -> None:
        """
        Remove APENAS experiência aprendida.

        Os pesos intrínsecos permanecem intactos.

        Útil para iniciar novos experimentos científicos
        mantendo a mesma configuração inicial.
        """

        with self._lock:

            self.learned_weights = np.zeros(
                len(
                    DIMENSIONS
                ),
                dtype=np.float64,
            )

            self.experience_count = 0

            self.last_prediction_error = None

            self.last_outcome = None

            if save:

                self._save_locked()

    # ========================================================
    # SNAPSHOT
    # ========================================================

    def snapshot(
        self,
    ) -> dict[str, Any]:
        """
        Estado atual do motor de valência.

        Será utilizado futuramente pelo monitor do
        cérebro artificial.
        """

        with self._lock:

            total = (

                self.intrinsic_weights
                +
                self.learned_weights

            )

            return {

                "experience_count":
                    self.experience_count,

                "learning_rate":
                    self.learning_rate,

                "confidence":
                    self._calculate_confidence(),

                "last_prediction_error":
                    self.last_prediction_error,

                "last_outcome":
                    self.last_outcome,

                "intrinsic_weights": {

                    dimension:
                        float(
                            self.intrinsic_weights[
                                index
                            ]
                        )

                    for index, dimension
                    in enumerate(
                        DIMENSIONS
                    )

                },

                "learned_weights": {

                    dimension:
                        float(
                            self.learned_weights[
                                index
                            ]
                        )

                    for index, dimension
                    in enumerate(
                        DIMENSIONS
                    )

                },

                "total_weights": {

                    dimension:
                        float(
                            total[
                                index
                            ]
                        )

                    for index, dimension
                    in enumerate(
                        DIMENSIONS
                    )

                },

                "storage":
                    str(
                        self.storage_path
                    ),

            }

    # ========================================================
    # CONVERSÃO DE PESOS
    # ========================================================

    @staticmethod
    def _coerce_weights(
        weights: (
            Mapping[str, float]
            | Sequence[float]
            | np.ndarray
            | None
        ),
    ) -> np.ndarray:
        """
        Converte diferentes representações para o vetor
        matemático utilizado pelo motor.

        Aceita:

        None

        ou:

        {
            "intensity": 0.0,
            "novelty": 0.0,
            ...
        }

        ou:

        [0, 0, 0, 0, 0]
        """

        if weights is None:

            return np.zeros(
                len(
                    DIMENSIONS
                ),
                dtype=np.float64,
            )

        if isinstance(
            weights,
            Mapping,
        ):

            unknown_dimensions = (

                set(
                    weights.keys()
                )

                -

                set(
                    DIMENSIONS
                )

            )

            if unknown_dimensions:

                raise ValueError(
                    "Dimensões desconhecidas nos pesos: "
                    f"{sorted(unknown_dimensions)}"
                )

            vector = np.asarray(

                [

                    float(
                        weights.get(
                            dimension,
                            0.0,
                        )
                    )

                    for dimension
                    in DIMENSIONS

                ],

                dtype=np.float64,

            )

        else:

            vector = np.asarray(
                weights,
                dtype=np.float64,
            ).reshape(
                -1
            )

        if (
            vector.size
            !=
            len(
                DIMENSIONS
            )
        ):

            raise ValueError(

                "O vetor de pesos deve possuir "
                f"{len(DIMENSIONS)} dimensões."

            )

        if not np.all(
            np.isfinite(
                vector
            )
        ):

            raise ValueError(
                "Todos os pesos devem ser números finitos."
            )

        return (
            vector.copy()
        )

    # ========================================================
    # VALIDAÇÃO DAS FEATURES
    # ========================================================

    @staticmethod
    def _validate_features(
        features: np.ndarray,
    ) -> np.ndarray:

        phi = np.asarray(
            features,
            dtype=np.float64,
        ).reshape(
            -1
        )

        expected_size = len(
            DIMENSIONS
        )

        if (
            phi.size
            !=
            expected_size
        ):

            raise ValueError(

                "φ(E) deve possuir "
                f"{expected_size} dimensões: "
                f"{DIMENSIONS}"

            )

        if not np.all(
            np.isfinite(
                phi
            )
        ):

            raise ValueError(
                "φ(E) contém NaN ou infinito."
            )

        # ----------------------------------------------------
        # VALIDAMOS A SEMÂNTICA DEFINIDA EM features.py
        # ----------------------------------------------------

        intensity = phi[0]
        novelty = phi[1]
        urgency = phi[2]
        goal_alignment = phi[3]
        risk = phi[4]

        if not 0.0 <= intensity <= 1.0:

            raise ValueError(
                "intensity deve estar entre 0 e 1."
            )

        if not 0.0 <= novelty <= 1.0:

            raise ValueError(
                "novelty deve estar entre 0 e 1."
            )

        if not 0.0 <= urgency <= 1.0:

            raise ValueError(
                "urgency deve estar entre 0 e 1."
            )

        if not -1.0 <= goal_alignment <= 1.0:

            raise ValueError(
                "goal_alignment deve estar entre -1 e 1."
            )

        if not 0.0 <= risk <= 1.0:

            raise ValueError(
                "risk deve estar entre 0 e 1."
            )

        return (
            phi.copy()
        )

    # ========================================================
    # PERSISTÊNCIA
    # ========================================================

    def save(
        self,
    ) -> None:

        with self._lock:

            self._save_locked()

    def _save_locked(
        self,
    ) -> None:
        """
        Persistência atômica do estado aprendido.

        Arquivo:

            data/
            └── brain/
                └── valence.json

        Isso permite que o sistema continue aprendendo
        entre diferentes execuções.
        """

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {

            "schema_version":
                VALENCE_SCHEMA_VERSION,

            "dimensions":
                list(
                    DIMENSIONS
                ),

            "learning_rate":
                self.learning_rate,

            "confidence_half_saturation":
                self.confidence_half_saturation,

            "experience_count":
                self.experience_count,

            "last_prediction_error":
                self.last_prediction_error,

            "last_outcome":
                self.last_outcome,

            "intrinsic_weights":
                self.intrinsic_weights.tolist(),

            "learned_weights":
                self.learned_weights.tolist(),

        }

        temporary_path = (
            self.storage_path
            .with_suffix(
                ".tmp"
            )
        )

        temporary_path.write_text(

            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),

            encoding="utf-8",

        )

        os.replace(
            temporary_path,
            self.storage_path,
        )

    # ========================================================
    # CARREGAMENTO
    # ========================================================

    def _load(
        self,
    ) -> None:

        if not self.storage_path.exists():

            return

        try:

            raw = (
                self.storage_path
                .read_text(
                    encoding="utf-8"
                )
            )

            data = json.loads(
                raw
            )

        except Exception as exc:

            raise RuntimeError(

                "Não foi possível carregar "
                "o estado do Valence Engine em "
                f"{self.storage_path}"

            ) from exc

        schema_version = int(
            data.get(
                "schema_version",
                0,
            )
        )

        if (
            schema_version
            !=
            VALENCE_SCHEMA_VERSION
        ):

            raise RuntimeError(

                "Versão incompatível do estado "
                "do Valence Engine. "
                f"Encontrada: {schema_version}. "
                f"Esperada: {VALENCE_SCHEMA_VERSION}."

            )

        stored_dimensions = tuple(
            data.get(
                "dimensions",
                [],
            )
        )

        if (
            stored_dimensions
            !=
            tuple(
                DIMENSIONS
            )
        ):

            raise RuntimeError(

                "As dimensões cognitivas do estado salvo "
                "não correspondem às dimensões atuais."

            )

        learned = self._coerce_weights(

            data.get(
                "learned_weights",
                None,
            )

        )

        # ----------------------------------------------------
        # PESOS INTRÍNSECOS
        # ----------------------------------------------------
        #
        # Se foram fornecidos explicitamente no construtor,
        # respeitamos a condição experimental atual.
        #
        # Caso contrário, restauramos os pesos salvos.
        # ----------------------------------------------------

        if not self._intrinsic_explicitly_provided:

            self.intrinsic_weights = (
                self._coerce_weights(

                    data.get(
                        "intrinsic_weights",
                        None,
                    )

                )
            )

        self.learned_weights = (
            learned
        )

        self.experience_count = int(

            data.get(
                "experience_count",
                0,
            )

        )

        self.last_prediction_error = (
            data.get(
                "last_prediction_error",
                None,
            )
        )

        self.last_outcome = (
            data.get(
                "last_outcome",
                None,
            )
        )