# -*- coding: utf-8 -*-
"""Topologia Edge Computing, tempo virtual dos clientes e perfis de dados."""

import json
import networkx as nx
import numpy as np
import os
import random
import torch.nn as nn
from dataclasses import dataclass
from experiment_core import deadline_status, estimate_virtual_duration, label_distribution_metrics
from torch.utils.data import Subset
from typing import Any, Dict, List, Optional, Tuple
from experiment_config import ExperimentConfig, NUM_CLASSES, _SCRIPT_DIR

# ==============================================================================
#  INFRAESTRUTURA DE REDE
# ==============================================================================

class NetworkSwitch:
    _all = []
    _id_counter = 1

    def __init__(self):
        self.id = NetworkSwitch._id_counter
        NetworkSwitch._id_counter += 1
        self.coordinates = None
        self.base_station = None
        NetworkSwitch._all.append(self)

    @classmethod
    def all(cls):
        return list(cls._all)

    @classmethod
    def _reset(cls):
        cls._all = []
        cls._id_counter = 1

    def __repr__(self):
        return f"NetworkSwitch(id={self.id})"


class BaseStation:
    _all = []
    _id_counter = 1

    def __init__(self):
        self.id = BaseStation._id_counter
        BaseStation._id_counter += 1
        self.wireless_delay = 0
        self.coordinates = None
        self.network_switch = None
        self.edge_servers = []
        BaseStation._all.append(self)

    def _connect_to_network_switch(self, network_switch):
        self.network_switch = network_switch
        network_switch.base_station = self
        network_switch.coordinates = self.coordinates

    def _connect_to_edge_server(self, edge_server):
        self.edge_servers.append(edge_server)
        edge_server.base_station = self

    @classmethod
    def all(cls):
        return list(cls._all)

    @classmethod
    def _reset(cls):
        cls._all = []
        cls._id_counter = 1

    def __repr__(self):
        return f"BaseStation(id={self.id}, coords={self.coordinates})"


class EdgeServer:
    _all = []
    _id_counter = 1

    def __init__(self, cpu=32, memory=128):
        self.id = EdgeServer._id_counter
        EdgeServer._id_counter += 1
        self.cpu = cpu
        self.memory = memory
        self.base_station = None
        EdgeServer._all.append(self)

    @classmethod
    def all(cls):
        return list(cls._all)

    @classmethod
    def _reset(cls):
        cls._all = []
        cls._id_counter = 1

    def __repr__(self):
        return f"EdgeServer(id={self.id})"


class User:
    _all = []
    _id_counter = 1

    def __init__(self):
        self.id = User._id_counter
        User._id_counter += 1
        self.coordinates = None
        self.base_station = None
        User._all.append(self)

    def _set_initial_position(self, coordinates):
        self.coordinates = coordinates
        for bs in BaseStation.all():
            if (abs(bs.coordinates[0] - coordinates[0]) < 1e-6 and
                    abs(bs.coordinates[1] - coordinates[1]) < 1e-6):
                self.base_station = bs
                break

    @classmethod
    def all(cls):
        return list(cls._all)

    @classmethod
    def _reset(cls):
        cls._all = []
        cls._id_counter = 1

    def __repr__(self):
        return f"User(id={self.id})"


class Topology(nx.Graph):
    _instance = None

    def __init__(self):
        super().__init__()
        Topology._instance = self

    @classmethod
    def first(cls):
        return cls._instance

    @classmethod
    def _reset(cls):
        cls._instance = None

    def calculate_path_delay(self, path):
        delay = 0.0
        for i in range(len(path) - 1):
            delay += self.edges[path[i], path[i + 1]]['delay']
        return delay


class ScenarioExporter:
    """Exporta topologia de rede para arquivo JSON."""

    @staticmethod
    def export(save_to_file=True, file_name="scenario"):
        if not save_to_file:
            return
        _ds_dir = os.path.join(_SCRIPT_DIR, "datasets")
        os.makedirs(_ds_dir, exist_ok=True)
        data = {
            "base_stations": [
                {"id": bs.id, "coordinates": list(bs.coordinates),
                 "wireless_delay": bs.wireless_delay}
                for bs in BaseStation.all()
            ],
            "edge_servers": [
                {"id": es.id, "cpu": es.cpu, "memory": es.memory,
                 "base_station_id": es.base_station.id if es.base_station else None}
                for es in EdgeServer.all()
            ],
        }
        path = os.path.join(_ds_dir, f"{file_name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Dataset exportado -> {path}")


# ==============================================================================
#  UTILITARIOS DE TOPOLOGIA
# ==============================================================================

def hexagonal_grid(x_size, y_size):
    coords = []
    dy = round(3 ** 0.5 / 2, 6)
    for row in range(y_size):
        x_offset = 0.5 if row % 2 == 1 else 0.0
        for col in range(x_size):
            coords.append((round(col + x_offset, 6), round(row * dy, 6)))
    return coords


def partially_connected_hexagonal_mesh(network_nodes, link_specifications):
    topology = Topology()
    nodes = list(network_nodes)
    for node in nodes:
        topology.add_node(node)

    link_idx = 0
    connected = set()
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
            if dist <= 1.01 and link_idx < len(link_specifications):
                spec = link_specifications[link_idx]
                topology.add_edge(n1, n2,
                                  delay=spec['delay'],
                                  bandwidth=spec['bandwidth'])
                connected.add(pair)
                link_idx += 1
    return topology


def generate_random_links(num_links):
    return [{"delay": round(random.uniform(1, 5), 2),
             "bandwidth": random.randint(5, 20)} for _ in range(num_links)]


# ==============================================================================
#  INICIALIZACAO DA TOPOLOGIA
# ==============================================================================

random.seed(42)

map_coordinates = hexagonal_grid(x_size=5, y_size=5)

for coords in map_coordinates:
    bs = BaseStation()
    bs.wireless_delay = 0
    bs.coordinates = coords
    sw = NetworkSwitch()
    bs._connect_to_network_switch(sw)
    # Servidor de borda com capacidade heterogenea por estacao base
    es = EdgeServer(cpu=random.choice([16, 24, 32, 48]), memory=64)
    bs._connect_to_edge_server(es)

partially_connected_hexagonal_mesh(
    network_nodes=NetworkSwitch.all(),
    link_specifications=generate_random_links(100)
)

topology_ref = Topology.first()
betweenness = nx.betweenness_centrality(topology_ref, weight='delay')
central_switch = max(betweenness, key=betweenness.get)

central_server = EdgeServer(cpu=64, memory=256)
central_switch.base_station.edge_servers = [central_server]
central_server.base_station = central_switch.base_station

CENTRAL_SERVER = central_server
CENTRAL_BS     = central_server.base_station
CENTRAL_SW     = central_switch

print(f"\n  Servidor Central -> BS{CENTRAL_BS.id} | Switch S{CENTRAL_SW.id}")
print(f"  Betweenness centrality: {betweenness[CENTRAL_SW]:.4f}")
print(f"  Coordenadas: {CENTRAL_BS.coordinates}")
print(f"  Clientes FL: {len(BaseStation.all()) - 1} nos\n")

ScenarioExporter.export(save_to_file=True, file_name="sample_dataset_v15")

_dijkstra_lengths = nx.single_source_dijkstra_path_length(
    topology_ref, CENTRAL_SW, weight='delay'
)
dijkstra_distances: Dict = dict(_dijkstra_lengths)

client_stations: List[BaseStation] = sorted(
    [bs for bs in BaseStation.all() if bs != CENTRAL_BS],
    key=lambda bs: dijkstra_distances.get(bs.network_switch, float('inf'))
)

print("  Ordem Dijkstra (clientes por distancia ao servidor):")
for i, bs in enumerate(client_stations):
    d = dijkstra_distances.get(bs.network_switch, float('inf'))
    print(f"    [{i:02d}] BS{bs.id}  dist={d:.2f}ms")
print()


def get_dijkstra_path(from_switch, to_switch):
    try:
        return nx.shortest_path(topology_ref, source=from_switch,
                                target=to_switch, weight='delay')
    except Exception:
        return []


def get_path_to_server(base_station: BaseStation):
    path = get_dijkstra_path(base_station.network_switch, CENTRAL_SW)
    delay = topology_ref.calculate_path_delay(path) if len(path) >= 2 else float('inf')
    return path, delay


def get_path_network_metrics(base_station: BaseStation) -> Tuple[float, float, List]:
    """
    Retorna (latencia_total_ms, bottleneck_bandwidth_mbps, caminho_switches)
    ao longo do caminho mais curto (Dijkstra) ate o switch central.
    """
    path = get_dijkstra_path(base_station.network_switch, CENTRAL_SW)
    if len(path) < 2:
        return 0.0, 20.0, path

    total_delay_ms = 0.0
    bottleneck_bw  = float('inf')

    for i in range(len(path) - 1):
        edge_data = topology_ref.edges[path[i], path[i + 1]]
        total_delay_ms += edge_data.get('delay', 1.0)
        bw = edge_data.get('bandwidth', 10.0)
        if bw < bottleneck_bw:
            bottleneck_bw = bw

    if bottleneck_bw == float('inf'):
        bottleneck_bw = 10.0

    return total_delay_ms, bottleneck_bw, path


def get_model_size_megabits(model: nn.Module) -> float:
    """Tamanho do modelo em Megabits (Mb) com parametros float32 (32 bits)."""
    total_params = sum(p.numel() for p in model.parameters())
    return float((total_params * 32) / 1e6)


@dataclass
class ClientTimingResult:
    cid:            int
    bs_id:          int
    t_down:         float  # tempo de download do modelo (s)
    t_comp:         float  # tempo de treinamento local (s)
    t_up:           float  # tempo de upload dos pesos (s)
    t_total:        float  # tempo total da iteracao (s)
    arrival_time:   float  # momento de chegada no servidor (s)
    deadline:       float  # deadline da rodada (s)
    on_time:        bool   # se concluiu dentro do deadline
    staleness:      int    # nivel de desatualizacao / rounds atrasados (tau)
    bandwidth_mbps: float  # largura de banda do gargalo no caminho (Mbps)
    latency_ms:     float  # latencia acumulada no caminho (ms)


def compute_client_timing(
    client_station: BaseStation,
    cid: int,
    n_samples: int,
    epochs: int,
    model_size_megabits: float,
    cfg: ExperimentConfig,
    round_deadline: float,
    round_start_time: float = 0.0,
    accumulated_staleness: int = 0,
    client_profile: Optional["ClientProfile"] = None,
) -> ClientTimingResult:
    """
    Calcula analiticamente a duracao de download, treinamento local,
    upload e o momento de chegada para o cliente na rodada.
    """
    latency_ms, bottleneck_bw, _ = get_path_network_metrics(client_station)
    # CPU e variacao sao dados do perfil; o tempo permanece uma estimativa virtual.
    cpu_cores = 32
    if client_station.edge_servers:
        cpu_cores = client_station.edge_servers[0].cpu
    profile_factor = client_profile.compute_factor if client_profile else 1.0
    duration = estimate_virtual_duration(
        model_size_megabits=model_size_megabits,
        n_samples=n_samples,
        epochs=epochs,
        comp_time_per_sample=cfg.comp_time_per_sample,
        bandwidth_mbps=bottleneck_bw,
        latency_ms=latency_ms,
        cpu_cores=cpu_cores,
        compute_factor=profile_factor,
        jitter_fraction=client_profile.time_jitter if client_profile else 0.0,
    )
    t_down = duration["download_s"]
    t_comp = duration["compute_s"]
    t_up = duration["upload_s"]
    t_total = duration["total_s"]
    arrival_time = round_start_time + t_total

    on_time = (
        deadline_status(arrival_time, round_start_time + round_deadline)[0]
        if round_deadline > 0 else True
    )
    staleness = accumulated_staleness if on_time else (accumulated_staleness + 1)

    return ClientTimingResult(
        cid=cid,
        bs_id=client_station.id,
        t_down=round(t_down, 3),
        t_comp=round(t_comp, 3),
        t_up=round(t_up, 3),
        t_total=round(t_total, 3),
        arrival_time=round(arrival_time, 3),
        deadline=round(round_deadline, 3),
        on_time=on_time,
        staleness=staleness,
        bandwidth_mbps=round(bottleneck_bw, 2),
        latency_ms=round(latency_ms, 2),
    )


@dataclass
class ClientProfile:
    """Metadados verificaveis usados antes de convidar um cliente."""
    cid: int
    label_counts: List[int]
    entropy: float
    rare_coverage: float
    js_divergence: float
    compute_factor: float = 1.0
    time_jitter: float = 0.0

    @property
    def information_score(self) -> float:
        entropy_norm = self.entropy / max(np.log2(NUM_CLASSES), 1e-12)
        js_norm = self.js_divergence / max(np.log(2.0), 1e-12)
        return float(0.5 * entropy_norm + 0.5 * min(js_norm, 1.0))


@dataclass
class PendingUpdate:
    """Atualizacao que chegou apos o deadline e pode ser tratada em rodada futura."""
    cid: int
    base_version: int
    arrival_time: float
    payload: Any
    n_samples: int
    metrics: Dict
    invited_round: int = 0
    round_start_time: float = 0.0
    round_deadline: float = 0.0
    t_total: float = 0.0


def build_client_profiles(partitions: List[Subset], cfg: ExperimentConfig) -> Dict[int, ClientProfile]:
    """Pre-calcula distribuicao, entropia, cobertura rara e JS para cada cliente."""
    all_counts: List[np.ndarray] = []
    for subset in partitions:
        targets = getattr(subset.dataset, "targets", [])
        labels = [int(targets[i]) for i in subset.indices]
        all_counts.append(np.bincount(labels, minlength=NUM_CLASSES).astype(float))
    train_targets = getattr(partitions[0].dataset, "targets", []) if partitions else []
    aggregate = np.bincount(train_targets, minlength=NUM_CLASSES).astype(float)
    profiles: Dict[int, ClientProfile] = {}
    for cid, counts in enumerate(all_counts):
        metrics = label_distribution_metrics(
            counts, aggregate, cfg.rare_class_threshold
        )
        rng = random.Random(cfg.seed + cid * 1009)
        profiles[cid] = ClientProfile(
            cid=cid, label_counts=counts.astype(int).tolist(),
            entropy=metrics["entropy"], rare_coverage=metrics["rare_coverage"],
            js_divergence=metrics["js_divergence"],
            compute_factor=rng.uniform(0.7, 1.3),
            time_jitter=rng.uniform(-cfg.time_jitter_fraction, cfg.time_jitter_fraction),
        )
    return profiles
