# -*- coding: utf-8 -*-
"""
views/metrics_window.py
MetricsWindow — janela Toplevel com gráficos de convergência FL.

Nota sobre imports matplotlib:
    Os imports de matplotlib.pyplot e FigureCanvasTkAgg são feitos de forma
    *lazy* (dentro do método _build), evitando:
      - Erro "cannot find module" do Pylance quando o interpretador
        selecionado no VS Code não tem matplotlib instalado.
      - Conflito de backend quando o módulo é importado antes de
        matplotlib.use() ser chamado em main.py.
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING

import tkinter as tk
from tkinter import ttk

if TYPE_CHECKING:
    from models.federated.fl_history import FlHistory


class MetricsWindow(tk.Toplevel):
    """
    Janela Toplevel que exibe 4 gráficos de métricas FL:
        [0,0] Loss de validação por round.
        [0,1] Acurácia de validação por round.
        [1,0] Train Loss (FedAvg agregado) por round.
        [1,1] Train Accuracy agregada por round.

    Salva os gráficos em PNG no diretório de resultados.

    Args:
        parent:      Widget pai (MainWindow).
        history:     FlHistory com as métricas do experimento.
        results_dir: Diretório para salvar o PNG.
    """

    def __init__(
        self,
        parent,
        history: "FlHistory",
        results_dir: str = "./fl_results_v15",
    ) -> None:
        super().__init__(parent)
        self.title("Métricas — FL CIFAR-10 com Dijkstra (migration_fdl)")
        self.geometry("1000x680")
        self.configure(bg="#1A1A2E")
        self._history = history
        self._results_dir = results_dir
        self._fig = None
        self._plt = None

        self._build()

    # ------------------------------------------------------------------
    # Construção
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # Lazy imports — resolvem o aviso do Pylance/pyright sem afetar o runtime.
        # O backend já foi configurado em main.py antes de qualquer import.
        import matplotlib.pyplot as plt                                   # noqa: PLC0415
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: PLC0415

        # Guarda referência ao módulo plt para uso em _on_close
        self._plt = plt

        h = self._history

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        self._fig = fig
        fig.patch.set_facecolor("#1A1A2E")
        fig.suptitle(
            f"Digital Twin FL — CIFAR-10 | {h.num_rounds} rounds | "
            f"{h.num_clients} clientes | Distribuição Dijkstra",
            fontsize=12, fontweight="bold", color="#E0E0E0",
        )

        rounds_e = [r for r, _ in h.losses_distributed]
        losses_e = [l for _, l in h.losses_distributed]

        # [0,0] Val Loss
        axes[0, 0].plot(rounds_e, losses_e, "o-", color="#4FC3F7",
                        linewidth=2, markersize=7, label="Val Loss")
        axes[0, 0].fill_between(rounds_e, losses_e, alpha=0.15, color="#4FC3F7")
        self._style_ax(axes[0, 0], "Loss de Validação por Round", "Round", "Loss")
        axes[0, 0].legend(facecolor="#1A1A2E", labelcolor="#E0E0E0")

        # [0,1] Val Accuracy
        if "accuracy" in h.eval_metrics:
            acc_rounds = [r for r, _ in h.eval_metrics["accuracy"]]
            acc_vals   = [v * 100 for _, v in h.eval_metrics["accuracy"]]
            axes[0, 1].plot(acc_rounds, acc_vals, "s-", color="#66BB6A",
                            linewidth=2, markersize=7, label="Val Acc")
            axes[0, 1].fill_between(acc_rounds, acc_vals, alpha=0.15, color="#66BB6A")
            axes[0, 1].set_ylim(0, 100)
            self._style_ax(axes[0, 1], "Acurácia de Validação por Round", "Round", "Acurácia (%)")
            axes[0, 1].legend(facecolor="#1A1A2E", labelcolor="#E0E0E0")

        # [1,0] Train Loss
        if "train_loss" in h.fit_metrics:
            tl_r = [r for r, _ in h.fit_metrics["train_loss"]]
            tl_v = [v for _, v in h.fit_metrics["train_loss"]]
            axes[1, 0].plot(tl_r, tl_v, "D-", color="#FFA726",
                            linewidth=2, markersize=7, label="Train Loss")
            axes[1, 0].fill_between(tl_r, tl_v, alpha=0.15, color="#FFA726")
            self._style_ax(axes[1, 0], "Train Loss (FedAvg Agregado)", "Round", "Loss")
            axes[1, 0].legend(facecolor="#1A1A2E", labelcolor="#E0E0E0")

        # [1,1] Train Accuracy
        if "train_acc" in h.fit_metrics:
            ta_r = [r for r, _ in h.fit_metrics["train_acc"]]
            ta_v = [v * 100 for _, v in h.fit_metrics["train_acc"]]
            axes[1, 1].plot(ta_r, ta_v, "^-", color="#CE93D8",
                            linewidth=2, markersize=7, label="Train Acc")
            axes[1, 1].fill_between(ta_r, ta_v, alpha=0.15, color="#CE93D8")
            axes[1, 1].set_ylim(0, 100)
            self._style_ax(axes[1, 1], "Train Accuracy (Agregada)", "Round", "Acurácia (%)")
            axes[1, 1].legend(facecolor="#1A1A2E", labelcolor="#E0E0E0")

        plt.tight_layout()

        # Salva PNG
        os.makedirs(self._results_dir, exist_ok=True)
        out_path = os.path.join(self._results_dir, "fl_cifar10_metrics.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Gráfico salvo → {out_path}")

        # Embute no Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Resumo final
        self._build_summary()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_summary(self) -> None:
        h = self._history
        sum_frame = ttk.LabelFrame(self, text="Resumo Final", padding=6)
        sum_frame.pack(fill=tk.X, padx=10, pady=6)

        final_r    = h.losses_distributed[-1][0]
        final_loss = h.losses_distributed[-1][1]
        final_acc  = (
            h.eval_metrics["accuracy"][-1][1] * 100
            if "accuracy" in h.eval_metrics
            else 0.0
        )
        ttk.Label(
            sum_frame,
            text=(
                f"Round {final_r}  |  Val Loss: {final_loss:.4f}  |  "
                f"Val Acc: {final_acc:.2f}%  |  "
                f"Clientes: {h.num_clients}  |  "
                f"Distribuição: Dijkstra"
            ),
            font=("Arial", 10, "bold"),
        ).pack()

    # ------------------------------------------------------------------
    # Estilo dos eixos
    # ------------------------------------------------------------------

    @staticmethod
    def _style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_facecolor("#0F0F1A")
        ax.set_title(title, color="#E0E0E0", fontsize=10)
        ax.set_xlabel(xlabel, color="#9E9E9E", fontsize=9)
        ax.set_ylabel(ylabel, color="#9E9E9E", fontsize=9)
        ax.tick_params(colors="#9E9E9E")
        ax.grid(True, alpha=0.2, color="#333355")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    # ------------------------------------------------------------------
    # Fechamento
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        if self._fig is not None and self._plt is not None:
            self._plt.close(self._fig)
        self.destroy()
