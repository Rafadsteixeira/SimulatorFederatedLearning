# -*- coding: utf-8 -*-
"""
views/config_dialog.py
ConfigDialog — diálogo de seleção da fonte do JSON de configuração.
Oferece duas opções: arquivo local ou URL remota.
"""
import tkinter as tk
from tkinter import ttk, filedialog
from typing import Callable


class ConfigDialog(tk.Toplevel):
    """
    Janela modal para escolher como carregar o JSON de configuração.

    Ao confirmar um arquivo local, chama on_file(path).
    Ao confirmar uma URL, chama on_url(url) via URLDialog.

    Args:
        parent:    Widget pai.
        on_file:   Callback(path: str) para arquivo local.
        on_url:    Callback(url: str) para URL remota.
    """

    def __init__(
        self,
        parent,
        on_file: Callable[[str], None],
        on_url: Callable[[str], None],
    ) -> None:
        super().__init__(parent)
        self.title("Carregar Configuração JSON")
        self.geometry("380x170")
        self.configure(bg="#1A1A2E")
        self.resizable(False, False)
        self.grab_set()

        self._on_file = on_file
        self._on_url  = on_url

        self._build()

    def _build(self) -> None:
        ttk.Label(
            self,
            text="Como deseja carregar o JSON?",
            font=("Arial", 11, "bold"),
            foreground="#FFD700",
            background="#1A1A2E",
        ).pack(pady=(20, 12))

        btn_f = ttk.Frame(self)
        btn_f.pack(pady=4)

        ttk.Button(
            btn_f, text="📁  Arquivo Local",
            command=self._pick_file, width=18,
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            btn_f, text="🌐  URL / Link",
            command=self._pick_url, width=18,
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            self, text="Cancelar",
            command=self.destroy,
        ).pack(pady=8)

    def _pick_file(self) -> None:
        self.destroy()
        path = filedialog.askopenfilename(
            title="Selecionar configuração JSON",
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")],
        )
        if path:
            self._on_file(path)

    def _pick_url(self) -> None:
        self.destroy()
        URLDialog(self.master, self._on_url)


class URLDialog(tk.Toplevel):
    """
    Diálogo para digitar uma URL e baixar o JSON de configuração.

    Args:
        parent:   Widget pai.
        on_url:   Callback(url: str) chamado ao confirmar.
    """

    def __init__(self, parent, on_url: Callable[[str], None]) -> None:
        super().__init__(parent)
        self.title("Carregar JSON por URL")
        self.geometry("500x160")
        self.configure(bg="#1A1A2E")
        self.resizable(False, False)
        self.grab_set()

        self._on_url = on_url
        self._build()

    def _build(self) -> None:
        ttk.Label(
            self,
            text="URL do arquivo JSON:",
            foreground="#E0E0E0",
        ).pack(pady=(18, 4))

        self._entry = ttk.Entry(self, width=58)
        self._entry.pack(padx=20)
        self._entry.insert(0, "https://")

        btn_f = ttk.Frame(self)
        btn_f.pack(pady=14)
        ttk.Button(btn_f, text="Carregar", command=self._fetch).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_f, text="Cancelar", command=self.destroy).pack(side=tk.LEFT, padx=10)

    def _fetch(self) -> None:
        url = self._entry.get().strip()
        if url and url not in ("https://", "http://"):
            self.destroy()
            self._on_url(url)
