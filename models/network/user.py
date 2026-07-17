# -*- coding: utf-8 -*-
"""
models/network/user.py
Entidade User — dispositivo móvel cliente na rede.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_station import BaseStation


class User:
    """
    Usuário móvel conectado a uma BaseStation.

    Atributos de classe:
        _all (list):        Registro global de todas as instâncias.
        _id_counter (int):  Contador auto-incrementado de IDs.
    """

    _all: list = []
    _id_counter: int = 1

    def __init__(self) -> None:
        self.id: int = User._id_counter
        User._id_counter += 1
        self.coordinates: tuple | None = None
        self.base_station: "BaseStation | None" = None
        User._all.append(self)

    # ------------------------------------------------------------------
    # Métodos de domínio
    # ------------------------------------------------------------------

    def set_initial_position(
        self,
        coordinates: tuple,
        base_stations: list["BaseStation"]
    ) -> None:
        """
        Define a posição inicial do usuário e encontra a BS mais próxima
        (por coincidência exata de coordenadas).

        Args:
            coordinates:   Par (x, y) de posição.
            base_stations: Lista de BaseStation disponíveis.
        """
        self.coordinates = coordinates
        for bs in base_stations:
            if (
                abs(bs.coordinates[0] - coordinates[0]) < 1e-6
                and abs(bs.coordinates[1] - coordinates[1]) < 1e-6
            ):
                self.base_station = bs
                break

    # ------------------------------------------------------------------
    # Métodos de classe (registro global)
    # ------------------------------------------------------------------

    @classmethod
    def all(cls) -> list["User"]:
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
        return f"User(id={self.id}, coords={self.coordinates})"
