# -*- coding: utf-8 -*-
"""
views/control_bar.py
ControlBar — barra de botões e status na parte superior da janela.
"""
import tkinter as tk
from tkinter import ttk
from typing import Callable


class ControlBar(ttk.Frame):
    """
    Frame superior com botões de ação e label de status FL.

    Callbacks injetados via construtor:
        on_load_dataset():   Aciona carregamento do CIFAR-10.
        on_start_training(): Aciona o treinamento FL.
        on_show_metrics():   Abre janela de métricas.
        on_load_config():    Abre diálogo de configuração JSON.
    """

    def __init__(
        self,
        parent,
        on_load_dataset: Callable,
        on_start_training: Callable,
        on_show_metrics: Callable,
        on_load_config: Callable,
        **kwargs,
    ) -> None:
        super().__init__(parent, padding="6", **kwargs)

        self._status_var = tk.StringVar(value="FL: Não treinado")

        # --- Botão: Carregar CIFAR-10 ---
        self._load_btn = ttk.Button(
            self, text="Carregar CIFAR-10", command=on_load_dataset
        )
        self._load_btn.pack(side=tk.LEFT, padx=4)

        # --- Botão: Treinar Modelo ---
        self._fl_btn = ttk.Button(
            self, text="Treinar Modelo", command=on_start_training
        )
        self._fl_btn.pack(side=tk.LEFT, padx=4)

        # --- Label de status ---
        ttk.Label(
            self,
            textvariable=self._status_var,
            foreground="#9ECDFF",
        ).pack(side=tk.LEFT, padx=6)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # --- Botão: Ver Métricas ---
        self._metrics_btn = ttk.Button(
            self, text="Ver Métricas FL", command=on_show_metrics
        )
        self._metrics_btn.pack(side=tk.LEFT, padx=4)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # --- Botão: Config JSON ---
        self._json_btn = ttk.Button(
            self, text="⚙ Carregar Config JSON", command=on_load_config
        )
        self._json_btn.pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def update_status(self, text: str) -> None:
        """Atualiza o label de status FL."""
        self._status_var.set(f"FL: {text}")

    def set_buttons_state(self, enabled: bool) -> None:
        """Habilita ou desabilita os botões de ação principal."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self._fl_btn.config(state=state)
        self._metrics_btn.config(state=state)
        self._load_btn.config(state=state)
