# -*- coding: utf-8 -*-
"""
models/network/edge_server.py
Entidade EdgeServer — servidor de borda responsável pelo treino federado.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_station import BaseStation


class EdgeServer:
    """
    Servidor de borda conectado a uma BaseStation.

    Atributos:
        cpu (int):      Número de CPUs disponíveis.
        memory (int):   Memória RAM em GB.
        power_model:    Modelo de consumo de energia (opcional).

    Atributos de classe:
        _all (list):        Registro global de todas as instâncias.
        _id_counter (int):  Contador auto-incrementado de IDs.
    """

    _all: list = []
    _id_counter: int = 1

    def __init__(self, cpu: int = 32, memory: int = 128) -> None:
        self.id: int = EdgeServer._id_counter
        EdgeServer._id_counter += 1
        self.cpu: int = cpu
        self.memory: int = memory
        self.base_station: "BaseStation | None" = None
        self.power_model = None
        EdgeServer._all.append(self)

    # ------------------------------------------------------------------
    # Métodos de classe (registro global)
    # ------------------------------------------------------------------

    @classmethod
    def all(cls) -> list["EdgeServer"]:
        """Retorna cópia da lista de todas as instâncias criadas."""
        return list(cls._all)

    @classmethod
    def reset(cls) -> None:
        """Limpa o registro global e reinicia o contador de IDs."""
        cls._all = []
        cls._id_counter = 1

    # ------------------------------------------------------------------
    # Representação
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"EdgeServer(id={self.id}, cpu={self.cpu}, mem={self.memory})"
