"""Minimal Tkinter dashboard for monitoring the local FedAvg training loop."""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flower import run_local_fedavg_training
from topology import draw_hexagonal_topology


class FLDashboard:
    """Simple dashboard showing per-round metrics for a local FedAvg run."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SimulatorFederatedLearning — FL Dashboard")
        self.root.geometry("1040x620")
        self.root.minsize(900, 520)

        self.rounds_var = tk.IntVar(value=3)
        self.clients_var = tk.IntVar(value=2)
        self.batch_size_var = tk.IntVar(value=32)
        self.learning_rate_var = tk.StringVar(value="0.01")

        self._build_widgets()
        self._metrics: list[dict[str, float]] = []

    def _build_widgets(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="Rounds").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Spinbox(top, from_=1, to=20, textvariable=self.rounds_var, width=8).grid(row=0, column=1, padx=(0, 12))

        ttk.Label(top, text="Clientes").grid(row=0, column=2, sticky="w", padx=(0, 8), pady=4)
        ttk.Spinbox(top, from_=1, to=10, textvariable=self.clients_var, width=8).grid(row=0, column=3, padx=(0, 12))

        ttk.Label(top, text="Batch").grid(row=0, column=4, sticky="w", padx=(0, 8), pady=4)
        ttk.Spinbox(top, from_=8, to=256, textvariable=self.batch_size_var, width=8).grid(row=0, column=5, padx=(0, 12))

        ttk.Label(top, text="LR").grid(row=0, column=6, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(top, textvariable=self.learning_rate_var, width=10).grid(row=0, column=7)

        self.start_button = ttk.Button(top, text="Treinar Modelo", command=self.start_training)
        self.start_button.grid(row=0, column=8, padx=(16, 0))

        body = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        left_panel = ttk.Frame(body)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        columns = ("round", "accuracy")
        self.table = ttk.Treeview(left_panel, columns=columns, show="headings", height=12)
        self.table.heading("round", text="Round")
        self.table.heading("accuracy", text="Accuracy")
        self.table.column("round", width=120, anchor="center")
        self.table.column("accuracy", width=180, anchor="center")
        self.table.pack(fill="both", expand=True, pady=(0, 10))

        self.log_box = tk.Text(left_panel, height=10, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True)

        right_panel = ttk.Frame(body)
        right_panel.grid(row=0, column=1, sticky="nsew")
        ttk.Label(right_panel, text="Topologia Hexagonal").pack(anchor="w", pady=(0, 8))

        self.topology_figure = Figure(figsize=(4.5, 4.5), dpi=100)
        self.topology_ax = self.topology_figure.add_subplot(111)
        self.topology_canvas = FigureCanvasTkAgg(self.topology_figure, master=right_panel)
        self.topology_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._render_initial_topology()

    def _render_initial_topology(self) -> None:
        self.topology_ax.clear()
        draw_hexagonal_topology(
            self.topology_ax,
            radius=2,
            num_base_stations=6,
            grid_size=5,
            training_history=[{"round": 1.0, "accuracy": 0.0}],
            model_name="FedAvg",
        )
        self.topology_ax.set_title("Topologia Hexagonal")
        self.topology_canvas.draw()

    def log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def start_training(self) -> None:
        self.start_button.configure(state="disabled")
        self.log("Iniciando treinamento federado...")
        self._metrics.clear()
        for row in self.table.get_children():
            self.table.delete(row)

        thread = threading.Thread(target=self._training_worker, daemon=True)
        thread.start()

    def _training_worker(self) -> None:
        try:
            rounds = int(self.rounds_var.get())
            clients = int(self.clients_var.get())
            batch = int(self.batch_size_var.get())
            lr = float(self.learning_rate_var.get())

            metrics = run_local_fedavg_training(
                num_rounds=rounds,
                num_clients=clients,
                batch_size=batch,
                local_epochs=1,
                learning_rate=lr,
            )
            self.root.after(0, self._show_results, metrics)
        except Exception as exc:  # pragma: no cover - UI-only guard
            self.root.after(0, self._show_error, str(exc))

    def _show_results(self, metrics: list[dict[str, float]]) -> None:
        self._metrics = metrics
        for row in metrics:
            self.table.insert("", "end", values=(int(row["round"]), f"{row['accuracy']:.4f}"))
        self.log(f"Treinamento concluído com {len(metrics)} rounds.")
        self._show_topology(metrics)
        self.start_button.configure(state="normal")

    def _show_topology(self, metrics: list[dict[str, float]]) -> None:
        self.topology_ax.clear()
        draw_hexagonal_topology(
            self.topology_ax,
            radius=2,
            num_base_stations=max(4, min(len(metrics) or 4, 6)),
            grid_size=5,
            training_history=metrics,
            model_name="FedAvg",
        )
        self.topology_canvas.draw()
        self.log("Topologia exibida diretamente na interface.")

    def _show_error(self, message: str) -> None:
        self.log(f"Erro: {message}")
        self.start_button.configure(state="normal")


def run_dashboard() -> None:
    """Open the dashboard window for local monitoring of the FL baseline."""
    root = tk.Tk()
    FLDashboard(root)
    root.mainloop()
