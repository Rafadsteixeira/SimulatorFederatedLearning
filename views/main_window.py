# -*- coding: utf-8 -*-
"""
views/main_window.py
MainWindow — janela principal da aplicação, monta o layout e delega eventos.
"""
from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Callable, Optional

import tkinter as tk
from tkinter import ttk, messagebox

if TYPE_CHECKING:
    from models.network.topology_builder import TopologyResult
    from models.federated.dataset_manager import DatasetManager
    from models.federated.experiment_config import ExperimentConfig
    from models.federated.fl_history import FLHistory

from .control_bar import ControlBar
from .network_map_view import NetworkMapView
from .rounds_panel import RoundsPanel


class MainWindow:
    """
    Janela principal da aplicação Digital Twin FL.

    Monta o layout (barra de controle, mapa, painel lateral) e expõe
    métodos que o AppController usa para atualizar a UI.

    Args:
        topo_result:          Topologia construída pelo TopologyBuilder.
        on_load_dataset:      Callback para carregar CIFAR-10.
        on_start_training:    Callback para iniciar treinamento FL.
        on_load_config_file:  Callback(path) para carregar config JSON.
        on_load_config_url:   Callback(url) para carregar config JSON por URL.
        on_show_metrics:      Callback para abrir métricas FL.
    """

    BG_COLOR    = "#1A1A2E"
    TITLE       = (
        "migration_fdl — Digital Twin FL com CIFAR-10 | "
        "1 Servidor Central + Dijkstra"
    )

    def __init__(
        self,
        topo_result: "TopologyResult",
        on_load_dataset: Callable,
        on_start_training: Callable,
        on_load_config_file: Callable,
        on_load_config_url: Callable,
        on_show_metrics: Callable,
    ) -> None:
        self._topo_result = topo_result
        self._on_load_config_file = on_load_config_file
        self._on_load_config_url  = on_load_config_url

        # Dataset manager será injetado externamente (referência)
        self._dataset_mgr: Optional["DatasetManager"] = None

        # Constrói a janela Tkinter
        self._root = tk.Tk()
        self._root.title(self.TITLE)
        self._root.geometry("1480x920")
        self._root.configure(bg=self.BG_COLOR)
        self._root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self._apply_style()

        # --- Barra de controle ---
        self._ctrl_bar = ControlBar(
            self._root,
            on_load_dataset=on_load_dataset,
            on_start_training=on_start_training,
            on_show_metrics=on_show_metrics,
            on_load_config=self._open_config_dialog,
        )
        self._ctrl_bar.pack(fill=tk.X, side=tk.TOP)

        # --- Área de conteúdo ---
        content = ttk.Frame(self._root)
        content.pack(fill=tk.BOTH, expand=True)

        # --- Placeholder dataset_mgr (será definido depois) ---
        from models.federated.dataset_manager import DatasetManager
        self._dataset_mgr = DatasetManager()  # vazio; será substituído via injeção

        # --- Mapa de rede ---
        self._map_view = NetworkMapView(
            content,
            topo_result=topo_result,
            dataset_mgr=self._dataset_mgr,
        )
        self._map_view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Painel lateral ---
        side = ttk.Frame(content, width=380)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        # Painel de configuração ativa
        self._config_panel = _ConfigPanel(side)
        self._config_panel.pack(fill=tk.X, padx=8, pady=(8, 2))

        # Painel de rounds
        self._rounds_panel = RoundsPanel(side)
        self._rounds_panel.pack(fill=tk.BOTH, expand=True)

        # Legenda
        _LegendPanel(side).pack(fill=tk.X, padx=8, pady=4)

        # Desenho inicial
        self._map_view.draw()

    # ------------------------------------------------------------------
    # Injeção de dataset_mgr (chamada pelo AppController após criação)
    # ------------------------------------------------------------------

    def inject_dataset_manager(self, mgr: "DatasetManager") -> None:
        """Injeta o DatasetManager real no NetworkMapView."""
        self._dataset_mgr = mgr
        self._map_view._dataset_mgr = mgr

    # ------------------------------------------------------------------
    # API pública (chamada pelo AppController)
    # ------------------------------------------------------------------

    def mainloop(self) -> None:
        """Inicia o loop principal Tkinter."""
        self._root.mainloop()

    def update_status(self, text: str) -> None:
        """Atualiza o label de status FL na barra de controle."""
        self._root.after(0, lambda: self._ctrl_bar.update_status(text))

    def set_training_buttons_state(self, enabled: bool) -> None:
        """Habilita/desabilita botões de ação."""
        self._root.after(0, lambda: self._ctrl_bar.set_buttons_state(enabled))

    def refresh_map(self) -> None:
        """Redesenha o mapa de rede."""
        self._root.after(0, self._map_view.draw)

    def append_round(self, entry: dict) -> None:
        """Adiciona um round ao painel lateral."""
        self._root.after(0, lambda: self._rounds_panel.append_round(entry))

    def clear_round_list(self) -> None:
        """Limpa a lista de rounds."""
        self._root.after(0, self._rounds_panel.clear)

    def update_config_panel(self, cfg: "ExperimentConfig", display_name: str) -> None:
        """Atualiza o painel de configuração ativa."""
        self._root.after(
            0, lambda: self._config_panel.refresh(cfg, display_name)
        )

    def open_metrics_window(
        self, history: "FLHistory", results_dir: str
    ) -> None:
        """Abre a janela de métricas FL."""
        from .metrics_window import MetricsWindow
        self._root.after(
            0,
            lambda: MetricsWindow(self._root, history=history, results_dir=results_dir),
        )

    # ------------------------------------------------------------------
    # Diálogos auxiliares
    # ------------------------------------------------------------------

    def show_info(self, title: str, message: str) -> None:
        self._root.after(0, lambda: messagebox.showinfo(title, message))

    def show_warning(self, title: str, message: str) -> None:
        self._root.after(0, lambda: messagebox.showwarning(title, message))

    def show_error(self, title: str, message: str) -> None:
        self._root.after(0, lambda: messagebox.showerror(title, message))

    def ask_yes_no(self, title: str, message: str) -> bool:
        return messagebox.askyesno(title, message)

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _open_config_dialog(self) -> None:
        from .config_dialog import ConfigDialog
        ConfigDialog(
            self._root,
            on_file=self._on_load_config_file,
            on_url=self._on_load_config_url,
        )

    def _on_closing(self) -> None:
        self._root.quit()
        self._root.destroy()
        sys.exit(0)

    @staticmethod
    def _apply_style() -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame",           background="#1A1A2E")
        style.configure("TLabel",           background="#1A1A2E", foreground="#E0E0E0")
        style.configure("TButton",          background="#2A2A5A", foreground="#E0E0E0")
        style.configure("TCombobox",        fieldbackground="#2A2A5A", foreground="#E0E0E0")
        style.configure("TSeparator",       background="#3A3A6A")
        style.configure("TLabelframe",      background="#1A1A2E", foreground="#E0E0E0")
        style.configure("TLabelframe.Label",background="#1A1A2E", foreground="#FFD700")


# ---------------------------------------------------------------------------
# Widgets privados da MainWindow
# ---------------------------------------------------------------------------

class _ConfigPanel(ttk.LabelFrame):
    """Painel de exibição da configuração ativa."""

    def __init__(self, parent) -> None:
        super().__init__(parent, text="⚙ Configuração Ativa", padding=6)
        self._var = tk.StringVar(value="Nenhum JSON carregado.\nUsando valores padrão.")
        ttk.Label(
            self,
            textvariable=self._var,
            justify=tk.LEFT,
            foreground="#B0BEC5",
            font=("Consolas", 8),
        ).pack(fill=tk.X)

    def refresh(self, cfg: "ExperimentConfig", display_name: str) -> None:
        lines = [f"Fonte: {display_name}"] + cfg.to_summary_lines()
        self._var.set("\n".join(lines))


class _LegendPanel(ttk.LabelFrame):
    """Painel de legenda das cores do mapa de rede."""

    ITEMS = [
        ("★  Servidor Central", "#FFD700"),
        ("●  Cliente (perto)",  "#3060DC"),
        ("●  Cliente (longe)",  "#5A2A8A"),
        ("━  Link rápido",      "#32CD32"),
        ("━  Link médio",       "#FFA500"),
        ("━  Link lento",       "#FF4444"),
    ]

    def __init__(self, parent) -> None:
        super().__init__(parent, text="Legenda", padding=6)
        for text, color in self.ITEMS:
            row = ttk.Frame(self)
            row.pack(fill=tk.X, pady=1)
            tk.Label(
                row, text="â– ", foreground=color,
                background="#1A1A2E", font=("Arial", 11),
            ).pack(side=tk.LEFT)
            ttk.Label(
                row, text=text, foreground="#CFD8DC",
                font=("Arial", 9),
            ).pack(side=tk.LEFT, padx=4)


