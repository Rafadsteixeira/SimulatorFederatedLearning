# -*- coding: utf-8 -*-
"""
models/network/topology_builder.py
TopologyBuilder — fábrica que constrói a topologia hexagonal 5x5,
posiciona BaseStations, NetworkSwitches, EdgeServers e calcula Dijkstra.
"""
import random
import json
import os
from typing import Dict, List, Tuple

import networkx as nx

from .network_switch import NetworkSwitch
from .base_station import BaseStation
from .edge_server import EdgeServer
from .topology import Topology


class TopologyBuilder:
    """
    Constrói e inicializa a topologia de rede completa.

    Uso:
        builder = TopologyBuilder(x_size=5, y_size=5, seed=42)
        result  = builder.build()
        # result.topology, result.central_server, result.central_bs,
        # result.central_sw, result.dijkstra_distances, result.client_stations

    O método build() é idempotente — reseta todos os registros globais
    das entidades antes de criar novas instâncias.
    """

    def __init__(
        self,
        x_size: int = 5,
        y_size: int = 5,
        seed: int = 42,
        central_cpu: int = 64,
        central_memory: int = 256,
        num_links: int = 100,
    ) -> None:
        self.x_size = x_size
        self.y_size = y_size
        self.seed = seed
        self.central_cpu = central_cpu
        self.central_memory = central_memory
        self.num_links = num_links

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def build(self) -> "TopologyResult":
        """
        Constrói a topologia completa e retorna um TopologyResult.

        Passos:
            1. Reseta todos os registros globais.
            2. Cria a grade hexagonal de BaseStations e NetworkSwitches.
            3. Constrói a malha parcialmente conectada (grafo).
            4. Calcula betweenness centrality para identificar nó central.
            5. Cria o EdgeServer central.
            6. Executa Dijkstra a partir do switch central.
            7. Ordena clientes por distância.
            8. Exporta cenário JSON.

        Returns:
            TopologyResult com todos os artefatos da topologia.
        """
        random.seed(self.seed)

        # 1. Reset
        self._reset_all()

        # 2. Grade hexagonal
        coords = self._hexagonal_grid(self.x_size, self.y_size)
        for coord in coords:
            bs = BaseStation()
            bs.wireless_delay = 0.0
            bs.coordinates = coord
            sw = NetworkSwitch()
            bs.connect_to_network_switch(sw)

        # 3. Grafo parcialmente conectado
        link_specs = self._generate_random_links(self.num_links)
        self._build_hex_mesh(NetworkSwitch.all(), link_specs)

        topology = Topology.first()

        # 4. Betweenness centrality → switch central
        betweenness = nx.betweenness_centrality(topology, weight="delay")
        central_sw = max(betweenness, key=betweenness.get)

        # 5. EdgeServer central
        central_server = EdgeServer(
            cpu=self.central_cpu, memory=self.central_memory
        )
        central_sw.base_station.connect_to_edge_server(central_server)

        central_bs = central_server.base_station

        print(f"\n  Servidor Central -> BS{central_bs.id} | Switch S{central_sw.id}")
        print(f"  Betweenness centrality: {betweenness[central_sw]:.4f}")
        print(f"  Coordenadas: {central_bs.coordinates}")
        print(f"  Clientes FL: {len(BaseStation.all()) - 1} nós\n")

        # 6. Dijkstra
        _lengths = nx.single_source_dijkstra_path_length(
            topology, central_sw, weight="delay"
        )
        dijkstra_distances: Dict = dict(_lengths)

        # 7. Clientes ordenados por distância
        client_stations: List[BaseStation] = sorted(
            [bs for bs in BaseStation.all() if bs != central_bs],
            key=lambda bs: dijkstra_distances.get(bs.network_switch, float("inf")),
        )

        print("  Ordem Dijkstra (clientes por distância ao servidor):")
        for i, bs in enumerate(client_stations):
            d = dijkstra_distances.get(bs.network_switch, float("inf"))
            print(f"    [{i:02d}] BS{bs.id}  dist={d:.2f}ms")
        print()

        # 8. Exporta cenário
        self._export_scenario(
            central_bs=central_bs,
            file_name="sample_dataset_v15",
        )

        return TopologyResult(
            topology=topology,
            central_server=central_server,
            central_bs=central_bs,
            central_sw=central_sw,
            dijkstra_distances=dijkstra_distances,
            client_stations=client_stations,
        )

    # ------------------------------------------------------------------
    # Métodos internos (privados)
    # ------------------------------------------------------------------

    @staticmethod
    def _reset_all() -> None:
        """Reseta os registros globais de todas as entidades."""
        NetworkSwitch.reset()
        BaseStation.reset()
        EdgeServer.reset()
        Topology.reset()

    @staticmethod
    def _hexagonal_grid(x_size: int, y_size: int) -> List[Tuple[float, float]]:
        """Gera coordenadas de uma grade hexagonal."""
        coords = []
        dy = round(3 ** 0.5 / 2, 6)
        for row in range(y_size):
            x_offset = 0.5 if row % 2 == 1 else 0.0
            for col in range(x_size):
                coords.append(
                    (round(col + x_offset, 6), round(row * dy, 6))
                )
        return coords

    @staticmethod
    def _generate_random_links(num_links: int) -> List[Dict]:
        """Gera especificações aleatórias de links."""
        return [
            {
                "delay": round(random.uniform(1, 5), 2),
                "bandwidth": random.randint(5, 20),
            }
            for _ in range(num_links)
        ]

    @staticmethod
    def _build_hex_mesh(
        network_nodes: List[NetworkSwitch],
        link_specs: List[Dict],
    ) -> Topology:
        """Constrói a malha hexagonal parcialmente conectada."""
        topology = Topology()
        nodes = list(network_nodes)
        for node in nodes:
            topology.add_node(node)

        link_idx = 0
        connected: set = set()

        for i, n1 in enumerate(nodes):
            for j, n2 in enumerate(nodes):
                if i >= j:
                    continue
                pair = (i, j)
                if pair in connected:
                    continue
                dx = n1.coordinates[0] - n2.coordinates[0]
                dy = n1.coordinates[1] - n2.coordinates[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= 1.01 and link_idx < len(link_specs):
                    spec = link_specs[link_idx]
                    topology.add_edge(
                        n1, n2,
                        delay=spec["delay"],
                        bandwidth=spec["bandwidth"],
                    )
                    connected.add(pair)
                    link_idx += 1

        return topology

    @staticmethod
    def _export_scenario(
        central_bs: BaseStation,
        file_name: str = "scenario",
        output_dir: str = "datasets",
    ) -> None:
        """Exporta o cenário de rede como arquivo JSON."""
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
                    "base_station_id": es.base_station.id if es.base_station else None,
                }
                for es in EdgeServer.all()
            ],
        }
        path = os.path.join(output_dir, f"{file_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  Dataset exportado -> {path}")


class TopologyResult:
    """
    Contêiner imutável com todos os artefatos produzidos pelo TopologyBuilder.

    Atributos:
        topology (Topology):                    Grafo da rede.
        central_server (EdgeServer):            Servidor central.
        central_bs (BaseStation):               BS do servidor central.
        central_sw (NetworkSwitch):             Switch central.
        dijkstra_distances (Dict):              Distâncias Dijkstra de cada switch ao central.
        client_stations (List[BaseStation]):    BSs clientes ordenadas por distância.
    """

    def __init__(
        self,
        topology: Topology,
        central_server: EdgeServer,
        central_bs: BaseStation,
        central_sw: NetworkSwitch,
        dijkstra_distances: Dict,
        client_stations: List[BaseStation],
    ) -> None:
        self.topology = topology
        self.central_server = central_server
        self.central_bs = central_bs
        self.central_sw = central_sw
        self.dijkstra_distances = dijkstra_distances
        self.client_stations = client_stations

    def get_path_to_server(
        self, base_station: BaseStation
    ) -> Tuple[List, float]:
        """
        Retorna o caminho Dijkstra e atraso de uma BS ao servidor central.

        Args:
            base_station: BaseStation de origem.

        Returns:
            Tupla (lista_de_nós, atraso_total_ms).
        """
        try:
            path = nx.shortest_path(
                self.topology,
                source=base_station.network_switch,
                target=self.central_sw,
                weight="delay",
            )
            delay = self.topology.calculate_path_delay(path)
        except Exception:
            path = []
            delay = float("inf")
        return path, delay
