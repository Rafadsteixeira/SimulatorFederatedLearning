# -*- coding: utf-8 -*-
"""
views/rounds_panel.py
RoundsPanel â€” painel lateral com listbox de rounds e detalhes do round selecionado.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import ttk


class RoundsPanel(ttk.Frame):
    """
    Painel lateral que exibe:
        - Lista scrollable de rounds FL (RoundsListbox).
        - Painel de detalhes do round selecionado.

    O painel Ã© passivo: recebe dados via append_round() e clear().
    """

    def __init__(self, parent, total_rounds: int = 5, **kwargs) -> None:
        super().__init__(parent, **kwargs)

        self._total_rounds = total_rounds
        self._round_log: List[Dict] = []

        self._build()

    # ------------------------------------------------------------------
    # ConstruÃ§Ã£o
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # --- Listbox de rounds ---
        rounds_lf = ttk.LabelFrame(self, text="Rodadas de Treinamento", padding=6)
        rounds_lf.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 4))

        list_frame = ttk.Frame(rounds_lf)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self._listbox = tk.Listbox(
            list_frame,
            bg="#0F0F1A", fg="#9ECDFF",
            selectbackground="#2A4A8A", selectforeground="#FFFFFF",
            font=("Consolas", 9),
            activestyle="none",
            borderwidth=0, highlightthickness=0,
            relief=tk.FLAT,
        )
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                  command=self._listbox.yview)
        self._listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._listbox.bind("<<ListboxSelect>>", self._on_select)

        # --- Painel de detalhes ---
        details_lf = ttk.LabelFrame(self, text="Detalhes da Rodada", padding=8)
        details_lf.pack(fill=tk.X, padx=8, pady=(0, 4))

        self._details_var = tk.StringVar(value="Clique em uma rodada para\nver os detalhes.")
        ttk.Label(
            details_lf,
            textvariable=self._details_var,
            justify=tk.LEFT,
            wraplength=340,
            foreground="#CFD8DC",
            font=("Consolas", 9),
        ).pack(fill=tk.X)

    # ------------------------------------------------------------------
    # API pÃºblica
    # ------------------------------------------------------------------

    def append_round(self, entry: Dict) -> None:
        """
        Adiciona um round Ã  lista.

        Args:
            entry: DicionÃ¡rio com mÃ©tricas do round (mesmo formato do fl_round_log).
        """
        self._round_log.append(entry)

        r    = entry["round"]
        acc  = entry["val_acc"] * 100
        loss = entry["val_loss"]
        n    = entry["n_clients"]
        mode = entry.get("delay_mode", "?")[0].upper()
        hl   = " âš " if entry.get("is_hl_round") else ""
        line = f" R{r:>2} [{mode}]{hl}  Loss {loss:.4f}  Acc {acc:5.1f}%  C:{n}"

        self._listbox.insert(tk.END, line)

        if entry.get("is_hl_round"):
            color = "#3A1A0A"
        else:
            color = "#1A2A4A" if r % 2 == 0 else "#0F1A30"
        self._listbox.itemconfig(tk.END, background=color)

        self._listbox.selection_clear(0, tk.END)
        self._listbox.selection_set(tk.END)
        self._listbox.see(tk.END)
        self._show_details(len(self._round_log) - 1)

    def clear(self) -> None:
        """Limpa a lista de rounds e o painel de detalhes."""
        self._round_log = []
        self._listbox.delete(0, tk.END)
        self._details_var.set("Clique em uma rodada para\nver os detalhes.")

    def set_total_rounds(self, n: int) -> None:
        """Atualiza o total de rounds esperado (usado nos detalhes)."""
        self._total_rounds = n

    # ------------------------------------------------------------------
    # Evento interno
    # ------------------------------------------------------------------

    def _on_select(self, event) -> None:
        sel = self._listbox.curselection()
        if sel:
            self._show_details(sel[0])

    def _show_details(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._round_log):
            self._details_var.set("Nenhum dado disponÃ­vel.")
            return

        e = self._round_log[idx]
        r        = e["round"]
        n_cli    = e["n_clients"]
        samples  = e["total_samples"]
        tl       = e["train_loss"]
        ta       = e["train_acc"] * 100
        vl       = e["val_loss"]
        va       = e["val_acc"] * 100

        delta_loss_str = ""
        delta_acc_str  = ""
        if idx > 0:
            prev   = self._round_log[idx - 1]
            d_loss = vl - prev["val_loss"]
            d_acc  = va - prev["val_acc"] * 100
            sign_l = "+" if d_loss >= 0 else ""
            sign_a = "+" if d_acc  >= 0 else ""
            delta_loss_str = f"  ({sign_l}{d_loss:.4f})"
            delta_acc_str  = f"  ({sign_a}{d_acc:.2f}%)"

        mode  = e.get("delay_mode", "?").upper()
        is_hl = e.get("is_hl_round", False)
        sep   = "=" * 31

        lines = [
            f"+{sep}+",
            f"  Rodada  : {r} / {self._total_rounds}  [{mode}]{'  âš  HL' if is_hl else ''}",
            f"  Clientes: {n_cli}",
            f"  Amostras: {samples:,}",
            f"+{sep}+",
            f"  TREINO  (FedAvg agregado)",
            f"  Loss  : {tl:.4f}",
            f"  Acc   : {ta:.2f}%",
            f"+{sep}+",
            f"  VALIDACAO",
            f"  Loss  : {vl:.4f}{delta_loss_str}",
            f"  Acc   : {va:.2f}%{delta_acc_str}",
            f"+{sep}+",
            f"  CLIENTES (latÃªncia {mode})",
        ]

        for cd in e.get("clients_detail", []):
            cid     = cd["cid"]
            bs_id   = cd["bs_id"]
            dist_ms = cd["dist_ms"]
            c_loss  = cd["loss"]
            c_acc   = cd["acc"] * 100
            lines.append(
                f"  [C{cid:02d}|BS{bs_id:<2}|{dist_ms:5.1f}ms]"
                f" L={c_loss:.4f} A={c_acc:.1f}%"
            )

        lines.append(f"+{sep}+")
        self._details_var.set("\n".join(lines))

