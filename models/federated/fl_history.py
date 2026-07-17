# -*- coding: utf-8 -*-
"""
models/federated/fl_history.py
FlHistory — armazena métricas de todos os rounds de um experimento FL.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class FlHistory:
    """
    Histórico de métricas de um experimento de Aprendizado Federado.

    Atributos:
        losses_distributed (List[Tuple[int,float]]):  (round, loss) de validação por round.
        fit_metrics (Dict[str, List[Tuple[int,float]]]):  Métricas de treino por chave.
        eval_metrics (Dict[str, List[Tuple[int,float]]]):  Métricas de avaliação por chave.
        num_rounds (int):   Total de rounds planejados.
        num_clients (int):  Número de clientes por round.
    """

    losses_distributed: List[Tuple[int, float]] = field(default_factory=list)
    fit_metrics: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    eval_metrics: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    num_rounds: int = 0
    num_clients: int = 0

    # ------------------------------------------------------------------
    # Métodos de atualização
    # ------------------------------------------------------------------

    def add_fit_metrics(self, round_num: int, metrics: Dict) -> None:
        """Registra métricas de treino (FedAvg agregado) de um round."""
        for k, v in metrics.items():
            self.fit_metrics.setdefault(k, []).append((round_num, float(v)))

    def add_eval_metrics(self, round_num: int, metrics: Dict) -> None:
        """Registra métricas de avaliação de um round."""
        for k, v in metrics.items():
            self.eval_metrics.setdefault(k, []).append((round_num, float(v)))

    # ------------------------------------------------------------------
    # Resumo
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Retorna um resumo textual do histórico."""
        lines = [f"FL Concluído — {self.num_rounds} rounds, {self.num_clients} clientes"]
        if self.losses_distributed:
            lines.append(f"  Loss final    : {self.losses_distributed[-1][1]:.4f}")
        if "accuracy" in self.eval_metrics and self.eval_metrics["accuracy"]:
            acc = self.eval_metrics["accuracy"][-1][1] * 100
            lines.append(f"  Acurácia final: {acc:.2f}%")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        """Retorna True se nenhum round foi registrado ainda."""
        return len(self.losses_distributed) == 0
