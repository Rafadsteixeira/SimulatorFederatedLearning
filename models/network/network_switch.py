# -*- coding: utf-8 -*-
"""
models/network/network_switch.py
Entidade NetworkSwitch — representa um switch de rede na topologia.
"""


class NetworkSwitch:
    """
    Nó de rede (switch) conectado a uma BaseStation.

    Responsabilidades:
        - Guardar ID único auto-incrementado.
        - Guardar coordenadas geográficas (herdadas da BaseStation).
        - Manter referência à BaseStation associada.

    Atributos de classe:
        _all (list):        Registro global de todas as instâncias.
        _id_counter (int):  Contador auto-incrementado de IDs.
    """

    _all: list = []
    _id_counter: int = 1

    def __init__(self) -> None:
        self.id: int = NetworkSwitch._id_counter
        NetworkSwitch._id_counter += 1
        self.coordinates: tuple | None = None
        self.base_station = None          # referência circular → BaseStation
        NetworkSwitch._all.append(self)

    # ------------------------------------------------------------------
    # Métodos de classe (registro global)
    # ------------------------------------------------------------------

    @classmethod
    def all(cls) -> list["NetworkSwitch"]:
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
        return f"NetworkSwitch(id={self.id}, coords={self.coordinates})"
