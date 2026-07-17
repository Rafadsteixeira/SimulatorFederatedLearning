# -*- coding: utf-8 -*-
"""
models/federated/experiment_config.py
ExperimentConfig — configuração de um experimento FL, carregável por JSON.
"""
import json
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ExperimentConfig:
    """
    Parâmetros que controlam um experimento de Aprendizado Federado.

    Atributos:
        n_clients (int):                  Número de clientes por round.
        n_rounds (int):                   Número de rounds FL.
        weight_mode (str):                Modo de pesos: 'random' ou 'fixed'.
        fixed_weights (Dict[int,float]):  Latências fixas por bs_id (modo fixed).
        high_latency_periodicity (int):   Cada N rounds usa clientes HL (0 = desativado).
        high_latency_fraction (float):    Fração de clientes HL no round HL.
        seed (int):                       Semente aleatória para reprodutibilidade.
        local_epochs (int):               Épocas locais de treino por round.
        batch_size (int):                 Tamanho de batch no DataLoader.
    """

    n_clients: int = 5
    n_rounds: int = 5
    weight_mode: str = "random"
    fixed_weights: Dict[int, float] = field(default_factory=dict)
    high_latency_periodicity: int = 0
    high_latency_fraction: float = 0.3
    seed: int = 42
    local_epochs: int = 1
    batch_size: int = 32

    # ------------------------------------------------------------------
    # Construtores alternativos
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        """
        Carrega a configuração a partir de um arquivo JSON.

        Args:
            path: Caminho para o arquivo .json.

        Returns:
            Nova instância de ExperimentConfig.

        Raises:
            ValueError: Se weight_mode for inválido.
            FileNotFoundError: Se o arquivo não existir.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = cls()
        cfg.n_clients = int(data.get("n_clients", cfg.n_clients))
        cfg.n_rounds = int(data.get("n_rounds", cfg.n_rounds))
        cfg.weight_mode = str(data.get("weight_mode", cfg.weight_mode))

        # fixed_weights: filtra chaves não numéricas (ex.: comentários)
        fw = {
            k: v
            for k, v in data.get("fixed_weights", {}).items()
            if k.lstrip("-").isdigit()
        }
        cfg.fixed_weights = {int(k): float(v) for k, v in fw.items()}

        cfg.high_latency_periodicity = int(
            data.get("high_latency_periodicity", cfg.high_latency_periodicity)
        )
        cfg.high_latency_fraction = float(
            data.get("high_latency_fraction", cfg.high_latency_fraction)
        )
        cfg.seed = int(data.get("seed", cfg.seed))
        cfg.local_epochs = int(data.get("local_epochs", cfg.local_epochs))
        cfg.batch_size = int(data.get("batch_size", cfg.batch_size))

        if cfg.weight_mode not in ("random", "fixed"):
            raise ValueError(
                f"weight_mode deve ser 'random' ou 'fixed', recebido: {cfg.weight_mode!r}"
            )

        return cfg

    # ------------------------------------------------------------------
    # Serialização
    # ------------------------------------------------------------------

    def to_summary_lines(self) -> List[str]:
        """Retorna linhas de texto resumindo a configuração."""
        mode_str = "Aleatório" if self.weight_mode == "random" else "Fixo"
        lines = [
            f"Clientes  : {self.n_clients}",
            f"Rounds    : {self.n_rounds}",
            f"Pesos     : {mode_str}",
        ]
        if self.high_latency_periodicity > 0:
            lines.append(f"Alta-lat. : a cada {self.high_latency_periodicity} rounds")
            lines.append(f"Fração    : {int(self.high_latency_fraction * 100)}%")
        else:
            lines.append("Alta-lat. : desativada")
        lines.append(f"Seed      : {self.seed}")
        lines.append(f"Épocas loc: {self.local_epochs}")
        lines.append(f"Batch     : {self.batch_size}")
        return lines
