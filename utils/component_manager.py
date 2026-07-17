# -*- coding: utf-8 -*-
"""
utils/component_manager.py
ComponentManager — exportação do cenário de rede para JSON.
"""
import json
import os

from models.network.base_station import BaseStation
from models.network.edge_server import EdgeServer


class ComponentManager:
    """
    Utilitário estático para exportar o estado atual da simulação
    (BaseStations e EdgeServers) como arquivo JSON.
    """

    @staticmethod
    def export_scenario(
        save_to_file: bool = True,
        file_name: str = "scenario",
        output_dir: str = "datasets",
    ) -> None:
        """
        Serializa BaseStations e EdgeServers ativos em JSON.

        Args:
            save_to_file: Se False, não faz nada (modo dry-run).
            file_name:    Nome base do arquivo (sem extensão).
            output_dir:   Pasta de destino (criada se não existir).
        """
        if not save_to_file:
            return

        os.makedirs(output_dir, exist_ok=True)
        data = {
            "base_stations": [
                {
                    "id": bs.id,
                    "coordinates": list(bs.coordinates),
                    "wireless_delay": bs.wireless_delay,
                }
                for bs in BaseStation.all()
            ],
            "edge_servers": [
                {
                    "id": es.id,
                    "cpu": es.cpu,
                    "memory": es.memory,
                    "base_station_id": (
                        es.base_station.id if es.base_station else None
                    ),
                }
                for es in EdgeServer.all()
            ],
        }
        path = os.path.join(output_dir, f"{file_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  Dataset exportado → {path}")
