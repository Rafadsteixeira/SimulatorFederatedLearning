# -*- coding: utf-8 -*-
"""
models/network/base_station.py
Entidade BaseStation — estação base sem fio da rede móvel.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .network_switch import NetworkSwitch
    from .edge_server import EdgeServer


class BaseStation:
    """
    Estação base que gerencia clientes sem fio e se conecta
    a um NetworkSwitch e a EdgeServers.

    Atributos de classe:
        _all (list):        Registro global de todas as instâncias.
        _id_counter (int):  Contador auto-incrementado de IDs.
    """

    _all: list = []
    _id_counter: int = 1

    def __init__(self) -> None:
        self.id: int = BaseStation._id_counter
        BaseStation._id_counter += 1
        self.wireless_delay: float = 0.0
        self.coordinates: tuple | None = None
        self.network_switch: "NetworkSwitch | None" = None
        self.edge_servers: list["EdgeServer"] = []
        BaseStation._all.append(self)

    # ------------------------------------------------------------------
    # Métodos de domínio
    # ------------------------------------------------------------------

    def connect_to_network_switch(self, switch: "NetworkSwitch") -> None:
        """Conecta esta BS ao switch e sincroniza coordenadas."""
        self.network_switch = switch
        switch.base_station = self
        switch.coordinates = self.coordinates

    def connect_to_edge_server(self, server: "EdgeServer") -> None:
        """Registra um EdgeServer como filho desta BS."""
        self.edge_servers.append(server)
        server.base_station = self

    # ------------------------------------------------------------------
    # Métodos de classe (registro global)
    # ------------------------------------------------------------------

    @classmethod
    def all(cls) -> list["BaseStation"]:
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
        return f"BaseStation(id={self.id}, coords={self.coordinates})"
