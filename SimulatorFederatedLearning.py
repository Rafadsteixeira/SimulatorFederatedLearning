# -*- coding: utf-8 -*-
"""
SimulatorFederatedLearning — Digital Twin FL com CIFAR-10
Topologia Edge Computing hexagonal com aprendizado federado via Flower nativo.

Arquitetura Flower:
  - ClientApp  → encapsula CifarFederatedClient (NumPyClient)
  - ServerApp  → configura rodadas e strategy
  - TopologyAwareFedAvg(FedAvg) → seleção por Dijkstra, alta-latência,
    coleta de métricas e atualização da GUI
"""

import sys

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import random
import os
import json
import threading
import urllib.request
import tempfile
from pathlib import Path
from uuid import uuid4
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import asdict, dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# Flower
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg as FlowerFedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    NDArrays, Parameters,
    FitIns, FitRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
try:
    from flwr.common import Context
except ImportError:
    Context = object  # type: ignore

try:
    from flwr.simulation import run_simulation as _flwr_run_simulation
    _HAS_RUN_SIMULATION = True
except ImportError:
    _HAS_RUN_SIMULATION = False
    print("Aviso: flwr.simulation.run_simulation nao disponivel. "
          "Sincronize o ambiente com: python -m uv sync")

from agno_policy import AgnoRoundPolicy
from experiment_core import (
    choose_client_ids,
    choose_round_deadline,
    compute_staleness,
    deadline_status,
    estimate_virtual_duration,
    effective_sample_weight,
    jensen_shannon_divergence,
    label_distribution_metrics,
    partition_indices,
    update_alignment,
    weighted_average_updates,
)
from run_artifacts import RunArtifactRecorder


# ==============================================================================
#  CONFIGURACAO DO EXPERIMENTO
# ==============================================================================

@dataclass
class ExperimentConfig:
    n_clients:                int   = 5
    n_rounds:                 int   = 5
    weight_mode:              str   = "random"
    fixed_weights:            Dict  = field(default_factory=dict)
    high_latency_periodicity: int   = 0
    high_latency_fraction:    float = 0.3
    seed:                     int   = 42
    local_epochs:             int   = 1
    batch_size:               int   = 32

    # Modelo de tempo, deadlines e staleness
    deadline_mode:            str   = "percentile"  # 'fixed' ou 'percentile'
    round_deadline:           float = 12.0          # segundos simulados (se deadline_mode == 'fixed')
    deadline_percentile:      float = 80.0          # percentil de corte (se deadline_mode == 'percentile')
    comp_time_per_sample:     float = 0.005         # tempo computacional base (s/amostra/epoca)
    enable_staleness:         bool  = True          # aplicar amortecimento de staleness na agregacao
    staleness_gamma:          float = 0.5           # expoente de penalizacao: 1 / (1 + tau)^gamma
    drop_stragglers:          bool  = False         # Compatibilidade: equivale a late_update_policy='discard'
    late_update_policy:       str   = "reduced_weight"  # discard, defer ou reduced_weight
    late_update_weight_factor: float = 0.5
    max_staleness:            int   = 2

    # Baselines de selecao e score multiobjetivo fixo.
    selection_mode:           str   = "multi_objective"
    selection_info_weight:    float = 0.45
    selection_rare_weight:    float = 0.25
    selection_diversity_weight: float = 0.15
    selection_time_weight:    float = 0.15
    rare_class_threshold:     float = 0.08
    time_jitter_fraction:     float = 0.0

    # Politica opcional baseada em LLM/Agno.
    # Em shadow o agente decide e audita, mas o baseline continua sendo aplicado.
    agent_policy_mode:        str   = "shadow"  # baseline, shadow, active ou replay
    agent_model_id:           str   = "openai/gpt-oss-120b"
    agent_timeout_s:          float = 20.0
    agent_audit_file:         str   = "decisions.jsonl"
    agent_replay_file:        str   = ""

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg = cls()
        cfg.n_clients                = int(data.get('n_clients',                cfg.n_clients))
        cfg.n_rounds                 = int(data.get('n_rounds',                 cfg.n_rounds))
        cfg.weight_mode              = str(data.get('weight_mode',              cfg.weight_mode))
        fw = {k: v for k, v in data.get('fixed_weights', {}).items()
              if k.lstrip('-').isdigit()}
        cfg.fixed_weights            = {int(k): float(v) for k, v in fw.items()}
        cfg.high_latency_periodicity = int(data.get('high_latency_periodicity', cfg.high_latency_periodicity))
        cfg.high_latency_fraction    = float(data.get('high_latency_fraction',  cfg.high_latency_fraction))
        cfg.seed                     = int(data.get('seed',                     cfg.seed))
        cfg.local_epochs             = int(data.get('local_epochs',             cfg.local_epochs))
        cfg.batch_size               = int(data.get('batch_size',               cfg.batch_size))

        cfg.deadline_mode            = str(data.get('deadline_mode',            cfg.deadline_mode))
        cfg.round_deadline           = float(data.get('round_deadline',         cfg.round_deadline))
        cfg.deadline_percentile      = float(data.get('deadline_percentile',    cfg.deadline_percentile))
        cfg.comp_time_per_sample     = float(data.get('comp_time_per_sample',   cfg.comp_time_per_sample))
        cfg.enable_staleness         = bool(data.get('enable_staleness',        cfg.enable_staleness))
        cfg.staleness_gamma          = float(data.get('staleness_gamma',        cfg.staleness_gamma))
        cfg.drop_stragglers          = bool(data.get('drop_stragglers',         cfg.drop_stragglers))
        legacy_policy = "discard" if cfg.drop_stragglers else cfg.late_update_policy
        cfg.late_update_policy       = str(data.get('late_update_policy', legacy_policy))
        cfg.late_update_weight_factor = float(data.get('late_update_weight_factor', cfg.late_update_weight_factor))
        cfg.max_staleness            = int(data.get('max_staleness', cfg.max_staleness))
        cfg.selection_mode           = str(data.get('selection_mode', cfg.selection_mode))
        cfg.selection_info_weight    = float(data.get('selection_info_weight', cfg.selection_info_weight))
        cfg.selection_rare_weight    = float(data.get('selection_rare_weight', cfg.selection_rare_weight))
        cfg.selection_diversity_weight = float(data.get('selection_diversity_weight', cfg.selection_diversity_weight))
        cfg.selection_time_weight    = float(data.get('selection_time_weight', cfg.selection_time_weight))
        cfg.rare_class_threshold     = float(data.get('rare_class_threshold', cfg.rare_class_threshold))
        cfg.time_jitter_fraction     = float(data.get('time_jitter_fraction', cfg.time_jitter_fraction))
        cfg.agent_policy_mode        = str(data.get('agent_policy_mode', cfg.agent_policy_mode))
        cfg.agent_model_id           = str(data.get('agent_model_id', cfg.agent_model_id))
        cfg.agent_timeout_s          = float(data.get('agent_timeout_s', cfg.agent_timeout_s))
        cfg.agent_audit_file         = str(data.get('agent_audit_file', cfg.agent_audit_file))
        cfg.agent_replay_file        = str(data.get('agent_replay_file', cfg.agent_replay_file))

        if cfg.n_clients < 1 or cfg.n_rounds < 1:
            raise ValueError("n_clients e n_rounds devem ser maiores que zero")
        if cfg.local_epochs < 1 or cfg.batch_size < 1:
            raise ValueError("local_epochs e batch_size devem ser maiores que zero")
        if cfg.seed < 0:
            raise ValueError("seed deve ser nao negativa")
        if cfg.high_latency_periodicity < 0:
            raise ValueError("high_latency_periodicity deve ser nao negativa")
        if not np.isfinite(cfg.high_latency_fraction) or not 0 <= cfg.high_latency_fraction <= 1:
            raise ValueError("high_latency_fraction deve estar entre 0 e 1")

        if cfg.weight_mode not in ("random", "fixed"):
            raise ValueError(f"weight_mode deve ser 'random' ou 'fixed', recebido: {cfg.weight_mode!r}")
        if cfg.deadline_mode not in ("fixed", "percentile"):
            raise ValueError("deadline_mode deve ser 'fixed' ou 'percentile'")
        if not np.isfinite(cfg.round_deadline) or cfg.round_deadline <= 0:
            raise ValueError("round_deadline deve ser maior que zero")
        if not np.isfinite(cfg.deadline_percentile) or not 0 <= cfg.deadline_percentile <= 100:
            raise ValueError("deadline_percentile deve estar entre 0 e 100")
        if not np.isfinite(cfg.comp_time_per_sample) or cfg.comp_time_per_sample < 0:
            raise ValueError("comp_time_per_sample deve ser finito e nao negativo")
        if not np.isfinite(cfg.staleness_gamma) or cfg.staleness_gamma < 0:
            raise ValueError("staleness_gamma deve ser finito e nao negativo")
        if cfg.max_staleness < 0:
            raise ValueError("max_staleness deve ser nao negativo")
        if cfg.late_update_policy not in ("discard", "defer", "reduced_weight"):
            raise ValueError("late_update_policy deve ser 'discard', 'defer' ou 'reduced_weight'")
        if not np.isfinite(cfg.late_update_weight_factor) or not 0 < cfg.late_update_weight_factor <= 1:
            raise ValueError("late_update_weight_factor deve estar no intervalo (0, 1]")
        if cfg.selection_mode not in ("random", "fastest", "dijkstra", "latency", "multi_objective"):
            raise ValueError("selection_mode deve ser random, fastest, dijkstra ou multi_objective")
        if not np.isfinite(cfg.rare_class_threshold) or not 0 <= cfg.rare_class_threshold <= 1:
            raise ValueError("rare_class_threshold deve estar entre 0 e 1")
        if not np.isfinite(cfg.time_jitter_fraction) or not 0 <= cfg.time_jitter_fraction <= 1:
            raise ValueError("time_jitter_fraction deve estar entre 0 e 1")
        selection_weights = (
            cfg.selection_info_weight,
            cfg.selection_rare_weight,
            cfg.selection_diversity_weight,
            cfg.selection_time_weight,
        )
        if (not all(np.isfinite(weight) and weight >= 0 for weight in selection_weights)
                or sum(selection_weights) <= 0):
            raise ValueError("pesos de selecao devem ser nao negativos e ter soma positiva")
        if cfg.agent_policy_mode not in ("baseline", "shadow", "active", "replay"):
            raise ValueError(
                "agent_policy_mode deve ser baseline, shadow, active ou replay"
            )
        if not cfg.agent_model_id.strip():
            raise ValueError("agent_model_id nao pode ser vazio")
        if not np.isfinite(cfg.agent_timeout_s) or cfg.agent_timeout_s <= 0:
            raise ValueError("agent_timeout_s deve ser maior que zero")
        if not cfg.agent_audit_file.strip():
            raise ValueError("agent_audit_file nao pode ser vazio")
        if cfg.agent_policy_mode == "replay" and not cfg.agent_replay_file.strip():
            raise ValueError("agent_replay_file e obrigatorio no modo replay")
        return cfg

    def to_summary_lines(self) -> List[str]:
        mode_str = "Aleatorio" if self.weight_mode == "random" else "Fixo"
        dl_str   = f"Fixo ({self.round_deadline:.1f}s)" if self.deadline_mode == "fixed" else f"P{int(self.deadline_percentile)}"
        lines = [
            f"Clientes  : {self.n_clients}",
            f"Rounds    : {self.n_rounds}",
            f"Pesos     : {mode_str}",
            f"Deadline  : {dl_str}",
            f"Staleness : {'Ativo' if self.enable_staleness else 'Inativo'}",
        ]
        if self.high_latency_periodicity > 0:
            lines.append(f"Alta-lat. : a cada {self.high_latency_periodicity} rounds")
            lines.append(f"Fracao    : {int(self.high_latency_fraction * 100)}%")
        else:
            lines.append("Alta-lat. : desativada")
        lines.append(f"Agno      : {self.agent_policy_mode.upper()}")
        lines.append(f"Selecao   : {self.selection_mode}")
        lines.append(
            f"Atrasos   : {self.late_update_policy} "
            f"(fator={self.late_update_weight_factor:.2f}, max tau={self.max_staleness})"
        )
        if self.agent_policy_mode != "baseline":
            lines.append(f"LLM       : {self.agent_model_id}")
        lines.append(f"Seed      : {self.seed}")
        lines.append(f"Epocas loc: {self.local_epochs}")
        lines.append(f"Batch     : {self.batch_size}")
        return lines


# Caminho absoluto do diretório do script — usado em todos os paths de dados
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    train_targets = getattr(_cifar_trainset, "targets", [])
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

# ==============================================================================
#  DATASET CIFAR-10
# ==============================================================================

# Caminhos absolutos relativos ao diretório do script
DATA_DIR    = os.path.join(_SCRIPT_DIR, "data_cifar10")
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "fl_results_v15")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CLASSES = 10
CLASS_NAMES = [
    "aviao", "automovel", "passaro", "gato", "veado",
    "cachorro", "sapo", "cavalo", "navio", "caminhao"
]
CLASS_COLORS = [
    "#4FC3F7", "#EF5350", "#66BB6A", "#FFA726", "#AB47BC",
    "#26C6DA", "#D4E157", "#FF7043", "#42A5F5", "#8D6E63"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Dispositivo de computacao: {DEVICE}\n")

TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616))
])
TRANSFORM_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616))
])
TRANSFORM_VIS = transforms.ToTensor()

_cifar_trainset    = None
_cifar_testset     = None
_cifar_raw_dataset = None
_central_testloader: Optional[DataLoader] = None
_images_loaded     = False
_client_sample_images: Dict[int, Tuple[np.ndarray, str, int]] = {}


def load_cifar10():
    global _cifar_trainset, _cifar_testset, _cifar_raw_dataset
    global _central_testloader
    print("  [CIFAR-10] Carregando dataset...")
    batches_dir = os.path.join(DATA_DIR, "cifar-10-batches-py")
    already_extracted = os.path.exists(batches_dir) and os.path.exists(os.path.join(batches_dir, "data_batch_1"))
    should_download = not already_extracted

    try:
        _cifar_trainset = CIFAR10(root=DATA_DIR, train=True, download=should_download,
                                  transform=TRANSFORM_TRAIN)
        _cifar_testset  = CIFAR10(root=DATA_DIR, train=False, download=should_download,
                                  transform=TRANSFORM_TEST)
    except Exception:
        _cifar_trainset = CIFAR10(root=DATA_DIR, train=True, download=False,
                                  transform=TRANSFORM_TRAIN)
        _cifar_testset  = CIFAR10(root=DATA_DIR, train=False, download=False,
                                  transform=TRANSFORM_TEST)

    _cifar_raw_dataset = CIFAR10(root=DATA_DIR, train=True, download=False,
                                 transform=TRANSFORM_VIS)
    _central_testloader = DataLoader(
        _cifar_testset, batch_size=128, shuffle=False, num_workers=0
    )
    print(f"  [CIFAR-10] {len(_cifar_trainset)} treino | {len(_cifar_testset)} teste")


def partition_cifar10(n_clients: int, seed: int = 42) -> List[Subset]:
    """Partition CIFAR-10 without overlap, preserving a controllable label skew."""
    if _cifar_trainset is None:
        raise RuntimeError("CIFAR-10 de treino ainda nao foi carregado")
    labels = getattr(_cifar_trainset, "targets", None)
    if labels is None:
        raise RuntimeError("O dataset CIFAR-10 nao disponibilizou seus rotulos")
    indices_by_client = partition_indices(labels, n_clients, seed, dirichlet_alpha=0.4)
    return [Subset(_cifar_trainset, indices) for indices in indices_by_client]


def load_sample_images_for_map():
    global _client_sample_images, _images_loaded

    class_indices_raw = {c: [] for c in range(NUM_CLASSES)}
    for idx, (_, label) in enumerate(_cifar_raw_dataset):
        class_indices_raw[label].append(idx)
        if all(len(v) > 0 for v in class_indices_raw.values()):
            if min(len(v) for v in class_indices_raw.values()) >= 5:
                break

    for i, bs in enumerate(client_stations):
        dominant_class = i % NUM_CLASSES
        idxs = class_indices_raw[dominant_class]
        chosen_idx = idxs[i % len(idxs)]
        img_tensor, label = _cifar_raw_dataset[chosen_idx]
        img_arr = img_tensor.numpy().transpose(1, 2, 0)
        img_arr = (img_arr * 255).astype(np.uint8)
        _client_sample_images[bs.id] = (img_arr, CLASS_NAMES[dominant_class], dominant_class)

    _images_loaded = True
    print(f"  [Imagens] {len(_client_sample_images)} thumbnails carregados")


# ==============================================================================
#  MODELO CNN
# ==============================================================================

class CifarCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block2(self.conv_block1(x)))


def get_weights(model: nn.Module) -> NDArrays:
    return [v.cpu().detach().numpy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: NDArrays) -> None:
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


def train_image_local(model: nn.Module, loader: DataLoader,
                      epochs: int = 1) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    model.train()
    model.to(DEVICE)
    total_loss, total_corr, total_n = 0.0, 0, 0
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_corr += out.argmax(1).eq(labels).sum().item()
            total_n    += images.size(0)
    return total_loss / max(total_n, 1), total_corr / max(total_n, 1)


def eval_image_local(model: nn.Module,
                     loader: DataLoader) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to(DEVICE)
    total_loss, total_corr, total_n = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out  = model(images)
            total_loss += criterion(out, labels).item() * images.size(0)
            total_corr += out.argmax(1).eq(labels).sum().item()
            total_n    += images.size(0)
    return total_loss / max(total_n, 1), total_corr / max(total_n, 1)


def evaluate_global_weights(weights: NDArrays) -> Tuple[float, float]:
    """Avalia uma vez o modelo global no conjunto de teste centralizado."""
    if _central_testloader is None:
        raise RuntimeError("Dataset de teste centralizado ainda nao foi carregado")
    evaluation_model = CifarCNN()
    set_weights(evaluation_model, weights)
    return eval_image_local(evaluation_model, _central_testloader)


# ==============================================================================
#  CLIENTE FEDERADO — Flower NumPyClient
# ==============================================================================

class CifarFederatedClient(NumPyClient):
    """
    Cliente FL que treina localmente CifarCNN com uma particao CIFAR-10
    e reporta pesos e metricas ao servidor via Flower.
    """

    def __init__(self, cid: int, base_station: BaseStation,
                 partition: Subset, batch_size: int, label_entropy: float,
                 seed: int = 42):
        self.cid          = cid
        self.base_station = base_station
        self.seed         = seed
        self.label_entropy = float(label_entropy)
        self.model        = CifarCNN()
        self._loader_generator = torch.Generator().manual_seed(seed)
        self.trainloader  = DataLoader(
            partition, batch_size=batch_size, shuffle=True, num_workers=0,
            generator=self._loader_generator,
        )
        self.n_train = len(partition)

    def get_parameters(self, config) -> NDArrays:
        return get_weights(self.model)

    def _set_parameters(self, parameters: NDArrays):
        set_weights(self.model, parameters)

    def fit(self, parameters: NDArrays, config) -> Tuple[NDArrays, int, Dict]:
        round_num = int(config.get("server_round", 1)) if isinstance(config, dict) else 1
        round_seed = self.seed + round_num
        random.seed(round_seed)
        np.random.seed(round_seed)
        torch.manual_seed(round_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(round_seed)
        self._loader_generator.manual_seed(round_seed)
        initial_weights = [np.copy(w) for w in parameters]
        self._set_parameters(parameters)
        dist   = dijkstra_distances.get(self.base_station.network_switch, 0.0)
        epochs = int(config.get("local_epochs", 1)) if isinstance(config, dict) else 1
        loss, acc = train_image_local(self.model, self.trainloader, epochs=epochs)
        new_weights = get_weights(self.model)

        # Norma L2 da atualizacao: ||w_local - w_global||_2
        diff_sq = sum(np.sum((w_new - w_old) ** 2)
                      for w_new, w_old in zip(new_weights, initial_weights))
        update_norm = float(np.sqrt(diff_sq))
        print(f"    [C{self.cid:02d} | BS{self.base_station.id} | dist={dist:.1f}ms] "
              f"loss={loss:.4f}  acc={acc*100:.1f}%  ||Δw||={update_norm:.3f}")
        return new_weights, self.n_train, {
            "train_loss":  float(loss),
            "train_acc":   float(acc),
            "update_norm": float(update_norm),
            "entropy":     self.label_entropy,
        }

# ==============================================================================
#  HISTORICO FL
# ==============================================================================

@dataclass
class FLHistory:
    losses_distributed: List[Tuple[int, float]] = field(default_factory=list)
    fit_metrics:        Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    eval_metrics:       Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    num_rounds:  int = 0
    num_clients: int = 0

    def add_fit_metrics(self, round_num: int, metrics: Dict):
        for k, v in metrics.items():
            self.fit_metrics.setdefault(k, []).append((round_num, float(v)))

    def add_eval_metrics(self, round_num: int, metrics: Dict):
        for k, v in metrics.items():
            self.eval_metrics.setdefault(k, []).append((round_num, float(v)))

    def summary(self) -> str:
        lines = [f"FL Concluido - {self.num_rounds} rounds, {self.num_clients} clientes"]
        if self.losses_distributed:
            lines.append(f"  Loss final   : {self.losses_distributed[-1][1]:.4f}")
        if 'accuracy' in self.eval_metrics and self.eval_metrics['accuracy']:
            acc = self.eval_metrics['accuracy'][-1][1] * 100
            lines.append(f"  Acuracia final: {acc:.2f}%")
        return "\n".join(lines)


# ==============================================================================
#  AUXILIARES DE REDE / SELECAO
# ==============================================================================

def _build_round_dijkstra(cfg: ExperimentConfig) -> Dict:
    if cfg.weight_mode == "random":
        for u, v in topology_ref.edges():
            topology_ref.edges[u, v]['delay']     = round(random.uniform(1, 5), 2)
            topology_ref.edges[u, v]['bandwidth'] = random.randint(5, 20)
        return dict(nx.single_source_dijkstra_path_length(
            topology_ref, CENTRAL_SW, weight='delay'
        ))
    else:
        result = {}
        for sw in topology_ref.nodes():
            bs = sw.base_station
            if bs and bs.id in cfg.fixed_weights:
                result[sw] = cfg.fixed_weights[bs.id]
            else:
                result[sw] = dijkstra_distances.get(sw, float('inf'))
        return result


def _profile_distribution(profile: ClientProfile) -> np.ndarray:
    counts = np.asarray(profile.label_counts, dtype=float)
    return counts / max(float(counts.sum()), 1.0)


def _selection_score(cid: int, selected: List[int], estimated_times: Dict[int, float],
                     profiles: Dict[int, ClientProfile], cfg: ExperimentConfig) -> float:
    profile = profiles.get(cid)
    if profile is None:
        return -estimated_times.get(cid, 0.0)
    values = list(estimated_times.values())
    low, high = min(values, default=0.0), max(values, default=0.0)
    time_cost = ((estimated_times[cid] - low) / (high - low)) if high > low else 0.0
    diversity = 1.0 if not selected else min(
        jensen_shannon_divergence(
            _profile_distribution(profile), _profile_distribution(profiles[other])
        )
        / max(np.log(2.0), 1e-12) for other in selected)
    return (cfg.selection_info_weight * profile.information_score
            + cfg.selection_rare_weight * profile.rare_coverage
            + cfg.selection_diversity_weight * diversity
            - cfg.selection_time_weight * time_cost)


def _client_history_for_agent(cid: int) -> Dict[str, Any]:
    """Resume somente observacoes anteriores, sem enviar pesos ou dados brutos."""
    observed: List[Dict[str, Any]] = []
    for round_entry in fl_round_log:
        for detail in round_entry.get("clients_detail", []):
            if (int(detail.get("cid", -1)) == cid
                    and detail.get("status") in {
                        "on_time", "accepted_stale", "accepted_reduced",
                        "discarded_deadline", "discarded_staleness", "failed",
                    }):
                observed.append(detail)

    selection_state = _agent_selection_history.get(cid, {})
    accepted_states = {"on_time", "accepted_stale", "accepted_reduced"}
    accepted_count = sum(1 for detail in observed if detail.get("status") in accepted_states)
    late_count = sum(
        1 for detail in observed
        if detail.get("status") not in {"on_time", "failed"}
    )
    return {
        "times_selected": int(selection_state.get("times_selected", 0)),
        "last_selected_round": selection_state.get("last_selected_round"),
        "observed_updates": len(observed),
        "accepted_updates": accepted_count,
        "deadline_miss_rate": round(late_count / len(observed), 4) if observed else 0.0,
        "mean_staleness": round(float(np.mean([
            float(detail.get("staleness", 0.0)) for detail in observed
        ])), 4) if observed else 0.0,
        "mean_update_norm": round(float(np.mean([
            float(detail.get("update_norm", 0.0)) for detail in observed
        ])), 6) if observed else 0.0,
        "mean_alignment_with_aggregate": round(float(np.mean([
            float(detail.get("alignment_with_aggregate", 0.0)) for detail in observed
            if detail.get("alignment_with_aggregate") is not None
        ])), 6) if any(
            detail.get("alignment_with_aggregate") is not None for detail in observed
        ) else 0.0,
    }


def _select_clients_for_round(
        round_num: int, cfg: ExperimentConfig, all_stations: List[BaseStation],
        round_dijkstra: Dict, round_start_time: float = 0.0,
        excluded_cids: Optional[set[int]] = None) -> Tuple[List[BaseStation], bool]:
    """Invite a configured number of clients using a named selection baseline."""
    excluded_cids = excluded_cids or set()
    available_ids = [cid for cid in range(len(all_stations)) if cid not in excluded_cids]
    available_stations = [all_stations[cid] for cid in available_ids]
    n_clients = min(cfg.n_clients, len(available_stations))
    if n_clients == 0:
        return [], False
    sorted_asc = sorted(
        available_stations,
        key=lambda bs: round_dijkstra.get(bs.network_switch, float('inf')),
    )
    is_hl = (cfg.high_latency_periodicity > 0 and round_num % cfg.high_latency_periodicity == 0)

    # Estimativas comuns ao baseline e ao contexto enviado ao agente.
    model_size = get_model_size_megabits(global_image_model)
    estimated_times: Dict[int, float] = {}
    estimated_timings: Dict[int, ClientTimingResult] = {}
    for idx in available_ids:
        bs = all_stations[idx]
        timing = compute_client_timing(
            bs, idx, len(_partitions[idx]) if idx < len(_partitions) else 100,
            cfg.local_epochs, model_size, cfg, 0.0, round_start_time,
            client_profile=_client_profiles.get(idx))
        estimated_times[idx] = timing.t_total
        estimated_timings[idx] = timing

    dijkstra_costs = {
        cid: round_dijkstra.get(bs.network_switch, float("inf"))
        for cid, bs in enumerate(all_stations) if cid in available_ids
    }
    chosen = choose_client_ids(
        mode=cfg.selection_mode,
        candidate_ids=estimated_times,
        n_clients=n_clients,
        predicted_times=estimated_times,
        dijkstra_costs=dijkstra_costs,
        seed=cfg.seed,
        round_num=round_num,
        score_fn=lambda cid, already_selected: _selection_score(
            cid, already_selected, estimated_times, _client_profiles, cfg
        ),
    )
    selected = [all_stations[idx] for idx in chosen]
    if is_hl and selected:
        n_high = (
            min(n_clients, max(1, int(n_clients * cfg.high_latency_fraction)))
            if cfg.high_latency_fraction > 0 else 0
        )
        high_pool = sorted(sorted_asc, key=lambda bs: round_dijkstra.get(bs.network_switch, 0.0), reverse=True)[:n_high]
        retained = [bs for bs in selected if bs not in high_pool][:max(0, n_clients - n_high)]
        selected = retained + high_pool
        print(f"  [Round {round_num}] Alta-latencia: {n_high} clientes HL + {len(retained)} normais")

    # O Agno recebe todos os candidatos em uma unica chamada por rodada.
    # Em shadow, decide e audita, mas AgnoRoundPolicy devolve o fallback.
    if _agno_policy is not None:
        station_to_idx = {bs.id: idx for idx, bs in enumerate(all_stations)}
        baseline_ids = {station_to_idx[bs.id] for bs in selected}
        fallback = {
            cid: ("select" if cid in baseline_ids else "skip")
        for cid in available_ids
        }

        timing_values = list(estimated_times.values())
        if cfg.deadline_mode == "fixed":
            provisional_deadline = cfg.round_deadline
        else:
            provisional_deadline = (
                float(np.percentile(timing_values, cfg.deadline_percentile))
                if timing_values else 0.0
            )

        candidates = []
        for cid in available_ids:
            bs = all_stations[cid]
            timing = estimated_timings[cid]
            profile = _client_profiles.get(cid)
            counts = np.asarray(profile.label_counts if profile else [], dtype=float)
            distribution = (
                (counts / counts.sum()).round(4).tolist()
                if counts.size and counts.sum() > 0 else []
            )
            cpu_cores = bs.edge_servers[0].cpu if bs.edge_servers else 32
            history = _client_history_for_agent(cid)
            candidates.append({
                "cid": cid,
                "bs_id": bs.id,
                "infrastructure": {
                    "cpu_cores": cpu_cores,
                    "latency_ms": timing.latency_ms,
                    "bandwidth_mbps": timing.bandwidth_mbps,
                },
                "data": {
                    "n_samples": len(_partitions[cid]) if cid < len(_partitions) else 0,
                    "class_distribution": distribution,
                    "entropy": round(profile.entropy, 6) if profile else 0.0,
                    "js_divergence": round(profile.js_divergence, 6) if profile else 0.0,
                    "rare_coverage": round(profile.rare_coverage, 6) if profile else 0.0,
                    "information_score": round(profile.information_score, 6) if profile else 0.0,
                },
                "predicted_timing": {
                    "download_s": timing.t_down,
                    "compute_s": timing.t_comp,
                    "upload_s": timing.t_up,
                    "total_s": timing.t_total,
                    "deadline_ratio": round(
                        timing.t_total / provisional_deadline, 4
                    ) if provisional_deadline > 0 else None,
                },
                "history": history,
                "previous_contribution": {
                    "mean_update_norm": history["mean_update_norm"],
                    "mean_alignment_with_aggregate": history[
                        "mean_alignment_with_aggregate"
                    ],
                },
            })

        recent_rounds = [{
            "round": entry.get("round"),
            "val_loss": entry.get("val_loss"),
            "val_acc": entry.get("val_acc"),
            "clients_on_time": entry.get("clients_on_time"),
            "clients_late": entry.get("clients_late"),
        } for entry in fl_round_log[-3:]]

        policy_result = _agno_policy.decide(
            phase="pre_round",
            context={
                "run_id": _agno_run_id,
                "round": round_num,
                "required_clients": n_clients,
                "selection_baseline": cfg.selection_mode,
                "high_latency_round": is_hl,
                "deadline": {
                    "mode": cfg.deadline_mode,
                    "provisional_s": round(provisional_deadline, 3),
                    "percentile": cfg.deadline_percentile,
                },
                "recent_rounds": recent_rounds,
                "candidates": candidates,
            },
            allowed_actions={"select", "skip"},
            fallback=fallback,
            required_counts={"select": n_clients},
        )
        selected_ids = [
            cid for cid, action in policy_result.actions.items()
            if action == "select"
        ]
        # Defesa adicional: nunca deixa uma resposta externa quebrar a cardinalidade.
        if len(selected_ids) != n_clients:
            selected_ids = sorted(baseline_ids)
            print("  [Agno] Cardinalidade invalida; usando selecao baseline.")
        selected = [all_stations[cid] for cid in selected_ids]

        if cfg.agent_policy_mode == "shadow":
            proposed_ids = sorted(
                cid for cid, detail in policy_result.proposed_details.items()
                if detail.get("action") == "select"
            )
            if proposed_ids:
                print(f"  [Agno shadow] proposta={proposed_ids} | "
                      f"aplicado_baseline={sorted(baseline_ids)}")
            else:
                print("  [Agno shadow] sem proposta valida; baseline aplicado.")

    for cid, bs in enumerate(all_stations):
        if bs in selected:
            state = _agent_selection_history.setdefault(cid, {
                "times_selected": 0, "last_selected_round": None,
            })
            state["times_selected"] += 1
            state["last_selected_round"] = round_num
    return selected, is_hl


# ==============================================================================
#  STRATEGY — TopologyAwareFedAvg
# ==============================================================================

class TopologyAwareFedAvg(FlowerFedAvg):
    """Flower Strategy with virtual deadlines, delayed updates, and audited decisions."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        sorted_stations: List[BaseStation],
        fl_history: FLHistory,
        gui_callback=None,
        status_callback=None,
        recorder: Optional[RunArtifactRecorder] = None,
        **kwargs,
    ):
        self.cfg = cfg
        self.sorted_stations = sorted_stations
        self.fl_history = fl_history
        self.gui_callback = gui_callback
        self.status_callback = status_callback
        self.recorder = recorder
        self._round_dijkstra: Dict = {}
        self._last_is_hl = False
        self._last_fit_metrics: Dict = {}
        self._last_aggregated_params: Optional[Parameters] = None
        self._current_parameters: Optional[Parameters] = None
        self._virtual_time = 0.0
        self._round_start_time = 0.0
        self._round_deadline = 0.0
        self._client_timings: Dict[int, ClientTimingResult] = {}
        self._invited_by_round: Dict[int, set[int]] = {}
        self._model_version = 0
        self._pending_updates: List[PendingUpdate] = []

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=min(cfg.n_clients, len(sorted_stations)),
            min_evaluate_clients=0,
            min_available_clients=len(sorted_stations),
            **kwargs,
        )

    def _sorted_proxies(self, client_manager) -> List[ClientProxy]:
        return sorted(
            client_manager.all().values(),
            key=lambda proxy: int(proxy.cid) if str(proxy.cid).lstrip("-").isdigit() else 0,
        )

    def _proxy_index(self, proxy: ClientProxy) -> int:
        try:
            return int(proxy.cid) % len(self.sorted_stations)
        except (TypeError, ValueError):
            return 0

    def _record_state(self, round_num: int, cid: int, state: str, **details: Any) -> None:
        if self.recorder is not None:
            self.recorder.record_client_state(round_num, cid, state, **details)

    def _client_context(self, cid: int, timing: Optional[ClientTimingResult] = None,
                        metrics: Optional[Dict] = None, staleness_tau: int = 0) -> Dict[str, Any]:
        station = self.sorted_stations[cid]
        profile = _client_profiles.get(cid)
        counts = np.asarray(profile.label_counts if profile else [], dtype=float)
        distribution = (
            (counts / counts.sum()).round(6).tolist()
            if counts.size and counts.sum() > 0 else []
        )
        network_latency, bandwidth, _ = get_path_network_metrics(station)
        history = _client_history_for_agent(cid)
        result: Dict[str, Any] = {
            "cid": cid,
            "bs_id": station.id,
            "infrastructure": {
                "cpu_cores": station.edge_servers[0].cpu if station.edge_servers else 32,
                "latency_ms": round(float(network_latency), 4),
                "bandwidth_mbps": round(float(bandwidth), 4),
            },
            "data": {
                "n_samples": len(_partitions[cid]) if cid < len(_partitions) else 0,
                "class_distribution": distribution,
                "entropy": round(profile.entropy, 6) if profile else 0.0,
                "js_divergence": round(profile.js_divergence, 6) if profile else 0.0,
                "rare_coverage": round(profile.rare_coverage, 6) if profile else 0.0,
                "information_score": round(profile.information_score, 6) if profile else 0.0,
            },
            "history": history,
            "previous_contribution": {
                "mean_update_norm": history["mean_update_norm"],
                "mean_alignment_with_aggregate": history["mean_alignment_with_aggregate"],
            },
            "staleness_tau": staleness_tau,
        }
        if timing is not None:
            result["predicted_timing"] = {
                "download_s": timing.t_down,
                "compute_s": timing.t_comp,
                "upload_s": timing.t_up,
                "total_s": timing.t_total,
                "arrival_time": timing.arrival_time,
                "deadline": timing.deadline,
                "on_time": timing.on_time,
            }
        if metrics:
            result["current_update"] = {
                "train_loss": float(metrics.get("train_loss", 0.0)),
                "train_acc": float(metrics.get("train_acc", 0.0)),
                "update_norm": float(metrics.get("update_norm", 0.0)),
            }
        return result

    def configure_fit(self, server_round, parameters, client_manager):
        if self.status_callback:
            self.status_callback(
                f"Round {server_round}/{self.cfg.n_rounds} [{self.cfg.weight_mode}]..."
            )

        client_manager.wait_for(len(self.sorted_stations), timeout=86400)
        self._current_parameters = parameters
        self._round_start_time = self._virtual_time
        self._round_dijkstra = _build_round_dijkstra(self.cfg)

        pending_cids = {update.cid for update in self._pending_updates}
        if len(pending_cids) == len(self.sorted_stations) and self._pending_updates:
            # Advance the virtual clock with one client while preventing a second
            # deferred update from that client until its existing update closes.
            oldest = min(
                self._pending_updates,
                key=lambda update: (update.arrival_time, update.invited_round),
            )
            pending_cids.discard(oldest.cid)
        selected_stations, is_hl = _select_clients_for_round(
            server_round,
            self.cfg,
            self.sorted_stations,
            self._round_dijkstra,
            round_start_time=self._round_start_time,
            excluded_cids=pending_cids,
        )
        self._last_is_hl = is_hl
        station_to_idx = {station.id: idx for idx, station in enumerate(self.sorted_stations)}
        selected_indices = [station_to_idx[station.id] for station in selected_stations]
        self._invited_by_round[server_round] = set(selected_indices)

        model_size = get_model_size_megabits(global_image_model)
        tentative: Dict[int, ClientTimingResult] = {}
        for cid in selected_indices:
            station = self.sorted_stations[cid]
            profile = _client_profiles.get(cid)
            tentative[cid] = compute_client_timing(
                client_station=station,
                cid=cid,
                n_samples=len(_partitions[cid]),
                epochs=self.cfg.local_epochs,
                model_size_megabits=model_size,
                cfg=self.cfg,
                round_deadline=0.0,
                round_start_time=self._round_start_time,
                client_profile=profile,
            )

        self._round_deadline = (
            choose_round_deadline(
                [timing.t_total for timing in tentative.values()],
                self.cfg.deadline_mode,
                self.cfg.round_deadline,
                self.cfg.deadline_percentile,
            )
            if tentative else max(float(self.cfg.round_deadline), 0.001)
        )
        self._client_timings = {}
        for cid, timing in tentative.items():
            timing.deadline = self._round_deadline
            timing.on_time = deadline_status(
                timing.arrival_time, self._round_start_time + self._round_deadline
            )[0]
            timing.staleness = 0
            self._client_timings[cid] = timing
            self._record_state(
                server_round,
                cid,
                "invited",
                base_model_version=self._model_version,
                predicted_arrival_time=timing.arrival_time,
                predicted_duration_s=timing.t_total,
                deadline_s=self._round_deadline,
                on_time=timing.on_time,
            )

        proxies_by_cid = {
            self._proxy_index(proxy): proxy for proxy in self._sorted_proxies(client_manager)
        }
        config = {
            "local_epochs": self.cfg.local_epochs,
            "server_round": server_round,
            "base_model_version": self._model_version,
        }
        fit_ins = FitIns(parameters, config)
        return [
            (proxies_by_cid[cid], fit_ins)
            for cid in selected_indices
            if cid in proxies_by_cid
        ]

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Validation occurs once on the global test set in evaluate_and_record_round.
        return []

    def _decide_post_deadline(
        self,
        server_round: int,
        decision_group: str,
        candidates: List[Dict[str, Any]],
        fallback: Dict[int, str],
        allowed_actions: set[str],
        allowed_by_client: Optional[Dict[int, set[str]]] = None,
    ) -> Dict[int, str]:
        if not candidates:
            return {}
        if _agno_policy is None:
            return fallback
        result = _agno_policy.decide(
            phase="post_deadline",
            context={
                "run_id": _agno_run_id,
                "round": server_round,
                "decision_group": decision_group,
                "round_start_time": self._round_start_time,
                "round_deadline_s": self._round_deadline,
                "round_end_time": self._round_start_time + self._round_deadline,
                "current_model_version": self._model_version,
                "max_staleness": self.cfg.max_staleness,
                "updates": candidates,
            },
            allowed_actions=allowed_actions,
            fallback=fallback,
            client_allowed_actions=allowed_by_client,
        )
        return result.actions

    def aggregate_fit(self, server_round, results, failures):
        if self._current_parameters is None:
            raise RuntimeError("Strategy recebeu resultados sem parametros globais")

        round_end_time = self._round_start_time + self._round_deadline
        selected = self._invited_by_round.pop(server_round, set())
        received: Dict[int, tuple[ClientProxy, FitRes]] = {
            self._proxy_index(proxy): (proxy, fit_res) for proxy, fit_res in results
        }
        accepted: List[tuple[int, NDArrays, int, Dict, float, Dict]] = []
        clients_detail: List[Dict[str, Any]] = []
        for cid in sorted(selected - set(received)):
            self._record_state(server_round, cid, "failed", reason="cliente sem FitRes")
            clients_detail.append({
                "cid": cid,
                "bs_id": self.sorted_stations[cid].id,
                "on_time": False,
                "staleness": 0,
                "status": "failed",
            })

        arrived_pending: List[tuple[PendingUpdate, int]] = []
        still_pending: List[PendingUpdate] = []
        for update in self._pending_updates:
            tau = compute_staleness(self._model_version, update.base_version)
            if tau > self.cfg.max_staleness:
                self._record_state(
                    server_round, update.cid, "discarded_staleness",
                    invited_round=update.invited_round,
                    staleness_tau=tau,
                    max_staleness=self.cfg.max_staleness,
                )
                clients_detail.append({
                    "cid": update.cid,
                    "bs_id": self.sorted_stations[update.cid].id,
                    "on_time": False,
                    "staleness": tau,
                    "status": "discarded_staleness",
                    "invited_round": update.invited_round,
                })
            elif update.arrival_time <= round_end_time:
                arrived_pending.append((update, tau))
            else:
                still_pending.append(update)
        self._pending_updates = still_pending

        if arrived_pending:
            candidates = []
            fallback = {}
            for update, tau in arrived_pending:
                fallback[update.cid] = "accept_stale"
                candidate = self._client_context(
                    update.cid, metrics=update.metrics, staleness_tau=tau
                )
                candidate.update({
                    "invited_round": update.invited_round,
                    "base_model_version": update.base_version,
                    "arrival_time": update.arrival_time,
                    "lateness_seconds": max(0.0, update.arrival_time - (
                        update.round_start_time + update.round_deadline
                    )),
                    "allowed_actions": ["accept_stale", "reduce_weight", "discard"],
                })
                candidates.append(candidate)
            actions = self._decide_post_deadline(
                server_round, "arrived_pending", candidates, fallback,
                {"accept_stale", "reduce_weight", "discard"},
            )
            for update, tau in arrived_pending:
                action = actions.get(update.cid, "accept_stale")
                if action == "discard":
                    state = "discarded_deadline"
                    effective_weight = 0.0
                    clients_detail.append({
                        "cid": update.cid,
                        "bs_id": self.sorted_stations[update.cid].id,
                        "on_time": False,
                        "staleness": tau,
                        "status": state,
                        "arrival_time": update.arrival_time,
                        "effective_weight": 0.0,
                        "invited_round": update.invited_round,
                    })
                else:
                    factor = self.cfg.late_update_weight_factor if action == "reduce_weight" else 1.0
                    gamma = self.cfg.staleness_gamma if self.cfg.enable_staleness else 0.0
                    effective_weight = effective_sample_weight(
                        update.n_samples, tau, gamma,
                        factor if self.cfg.enable_staleness or factor < 1 else 1.0,
                    )
                    state = "accepted_reduced" if action == "reduce_weight" else "accepted_stale"
                    accepted.append((
                        update.cid, update.payload, update.n_samples, update.metrics,
                        effective_weight, {
                            "cid": update.cid,
                            "bs_id": self.sorted_stations[update.cid].id,
                            "t_total": update.t_total,
                            "arrival_time": update.arrival_time,
                            "deadline": update.round_deadline,
                            "on_time": False,
                            "staleness": tau,
                            "status": state,
                            "update_norm": float(update.metrics.get("update_norm", 0.0)),
                            "entropy": float(update.metrics.get("entropy", 0.0)),
                            "js_divergence": float(update.metrics.get("js_divergence", 0.0)),
                            "rare_coverage": float(update.metrics.get("rare_coverage", 0.0)),
                            "effective_weight": effective_weight,
                            "invited_round": update.invited_round,
                        },
                    ))
                self._record_state(
                    server_round, update.cid, state,
                    invited_round=update.invited_round,
                    base_model_version=update.base_version,
                    staleness_tau=tau,
                    effective_weight=effective_weight,
                    action=action,
                )

        late_current: List[tuple[int, NDArrays, int, Dict, ClientTimingResult]] = []
        for cid, (_proxy, fit_res) in received.items():
            timing = self._client_timings.get(cid)
            if timing is None:
                self._record_state(server_round, cid, "failed", reason="tempo previsto ausente")
                continue
            metrics = dict(fit_res.metrics)
            profile = _client_profiles.get(cid)
            metrics.setdefault("js_divergence", profile.js_divergence if profile else 0.0)
            metrics.setdefault("rare_coverage", profile.rare_coverage if profile else 0.0)
            if timing.on_time:
                weight = effective_sample_weight(fit_res.num_examples)
                detail = {
                    "cid": cid, "bs_id": self.sorted_stations[cid].id,
                    "loss": float(metrics.get("train_loss", 0.0)),
                    "acc": float(metrics.get("train_acc", 0.0)),
                    "t_down": timing.t_down, "t_comp": timing.t_comp,
                    "t_up": timing.t_up, "t_total": timing.t_total,
                    "arrival_time": timing.arrival_time, "deadline": self._round_deadline,
                    "on_time": True, "staleness": 0, "status": "on_time",
                    "bandwidth": timing.bandwidth_mbps, "latency_ms": timing.latency_ms,
                    "update_norm": float(metrics.get("update_norm", 0.0)),
                    "entropy": float(metrics.get("entropy", 0.0)),
                    "js_divergence": float(metrics.get("js_divergence", 0.0)),
                    "rare_coverage": float(metrics.get("rare_coverage", 0.0)),
                    "effective_weight": weight,
                }
                accepted.append((cid, parameters_to_ndarrays(fit_res.parameters),
                                 fit_res.num_examples, metrics, weight, detail))
                self._record_state(
                    server_round, cid, "on_time", arrival_time=timing.arrival_time,
                    deadline_s=self._round_deadline, effective_weight=weight,
                    base_model_version=self._model_version,
                )
            else:
                late_current.append((
                    cid, parameters_to_ndarrays(fit_res.parameters),
                    fit_res.num_examples, metrics, timing,
                ))

        if late_current:
            candidates = []
            fallback = {}
            allowed_by_client: Dict[int, set[str]] = {}
            pending_cids = {update.cid for update in self._pending_updates}
            for cid, _weights, _n, metrics, timing in late_current:
                allowed_for_client = {"defer", "discard", "reduce_weight"}
                if cid in pending_cids:
                    allowed_for_client = {"discard", "reduce_weight"}
                allowed_by_client[cid] = allowed_for_client
                action = {
                    "defer": "defer",
                    "discard": "discard",
                    "reduced_weight": "reduce_weight",
                }[self.cfg.late_update_policy]
                if action not in allowed_for_client:
                    action = "discard"
                fallback[cid] = action
                candidate = self._client_context(cid, timing, metrics)
                candidate.update({
                    "base_model_version": self._model_version,
                    "lateness_seconds": max(0.0, timing.arrival_time - (
                        self._round_start_time + self._round_deadline
                    )),
                    "allowed_actions": sorted(allowed_for_client),
                })
                candidates.append(candidate)
            actions = self._decide_post_deadline(
                server_round, "new_late_updates", candidates, fallback,
                {"defer", "discard", "reduce_weight"}, allowed_by_client,
            )
            for cid, weights, n_samples, metrics, timing in late_current:
                action = actions.get(cid, fallback[cid])
                profile = _client_profiles.get(cid)
                detail = {
                    "cid": cid, "bs_id": self.sorted_stations[cid].id,
                    "loss": float(metrics.get("train_loss", 0.0)),
                    "acc": float(metrics.get("train_acc", 0.0)),
                    "t_down": timing.t_down, "t_comp": timing.t_comp,
                    "t_up": timing.t_up, "t_total": timing.t_total,
                    "arrival_time": timing.arrival_time, "deadline": self._round_deadline,
                    "on_time": False, "staleness": 0,
                    "bandwidth": timing.bandwidth_mbps, "latency_ms": timing.latency_ms,
                    "update_norm": float(metrics.get("update_norm", 0.0)),
                    "entropy": float(metrics.get("entropy", 0.0)),
                    "js_divergence": profile.js_divergence if profile else 0.0,
                    "rare_coverage": profile.rare_coverage if profile else 0.0,
                    "action": action,
                }
                if action == "defer":
                    detail["status"] = "late_pending"
                    pending = PendingUpdate(
                        cid=cid,
                        base_version=self._model_version,
                        arrival_time=timing.arrival_time,
                        payload=weights,
                        n_samples=n_samples,
                        metrics=metrics,
                        invited_round=server_round,
                        round_start_time=self._round_start_time,
                        round_deadline=self._round_deadline,
                        t_total=timing.t_total,
                    )
                    self._pending_updates.append(pending)
                    self._record_state(
                        server_round, cid, "late_pending",
                        arrival_time=timing.arrival_time,
                        deadline_s=self._round_deadline,
                        base_model_version=self._model_version,
                        lateness_seconds=max(0.0, timing.arrival_time - round_end_time),
                        action=action,
                    )
                elif action == "discard":
                    detail["status"] = "discarded_deadline"
                    self._record_state(
                        server_round, cid, "discarded_deadline",
                        arrival_time=timing.arrival_time,
                        deadline_s=self._round_deadline,
                        action=action,
                    )
                else:
                    weight = effective_sample_weight(
                        n_samples, 0, self.cfg.staleness_gamma,
                        self.cfg.late_update_weight_factor,
                    )
                    detail.update({
                        "status": "accepted_reduced",
                        "effective_weight": weight,
                    })
                    accepted.append((cid, weights, n_samples, metrics, weight, detail))
                    self._record_state(
                        server_round, cid, "accepted_reduced",
                        arrival_time=timing.arrival_time,
                        deadline_s=self._round_deadline,
                        effective_weight=weight,
                        action=action,
                        base_model_version=self._model_version,
                    )
                clients_detail.append(detail)

        clients_detail.extend(item[5] for item in accepted if item[5] not in clients_detail)
        current_arrays = parameters_to_ndarrays(self._current_parameters)
        if accepted:
            weighted_updates = [(item[1], item[4]) for item in accepted]
            aggregated_arrays = weighted_average_updates(weighted_updates)
            for item in accepted:
                item[5]["alignment_with_aggregate"] = update_alignment(
                    current_arrays, item[1], aggregated_arrays
                )
            aggregated_parameters = ndarrays_to_parameters(aggregated_arrays)
            self._last_aggregated_params = aggregated_parameters
            self._current_parameters = aggregated_parameters
            self._model_version += 1
            current_arrays = aggregated_arrays
            total_effective_weight = sum(item[4] for item in accepted)
            total_samples = sum(item[2] for item in accepted)
            train_loss = sum(
                float(item[3].get("train_loss", 0.0)) * item[4] for item in accepted
            ) / total_effective_weight
            train_acc = sum(
                float(item[3].get("train_acc", 0.0)) * item[4] for item in accepted
            ) / total_effective_weight
            update_norm = sum(
                float(item[3].get("update_norm", 0.0)) * item[4] for item in accepted
            ) / total_effective_weight
            aggregated_result = aggregated_parameters
        else:
            train_loss = train_acc = update_norm = 0.0
            total_samples = 0
            aggregated_result = None

        timing_values = [self._client_timings[cid].t_total for cid in selected
                         if cid in self._client_timings]
        self._last_fit_metrics = {
            "n_clients": len(selected),
            "total_samples": total_samples,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "update_norm": update_norm,
            "deadline": self._round_deadline,
            "clients_on_time": sum(1 for cid in selected
                                   if cid in self._client_timings and self._client_timings[cid].on_time),
            "clients_late": sum(1 for cid in selected
                                if cid in self._client_timings and not self._client_timings[cid].on_time),
            "min_time": min(timing_values, default=0.0),
            "max_time": max(timing_values, default=0.0),
            "avg_time": float(np.mean(timing_values)) if timing_values else 0.0,
            "clients_detail": clients_detail,
            "model_version": self._model_version,
        }
        self._virtual_time = round_end_time
        self.fl_history.add_fit_metrics(server_round, {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "update_norm": update_norm,
        })
        self.evaluate_and_record_round(
            server_round, aggregated_result or self._current_parameters,
        )
        return aggregated_result, {}

    def evaluate_and_record_round(self, server_round: int,
                                  parameters: Parameters) -> Tuple[float, Dict]:
        """Evaluate the global model once and persist the round result."""
        val_loss, val_acc = evaluate_global_weights(parameters_to_ndarrays(parameters))
        self.fl_history.losses_distributed.append((server_round, val_loss))
        self.fl_history.add_eval_metrics(server_round, {"accuracy": val_acc})

        fit_metrics = self._last_fit_metrics
        round_entry = {
            "run_id": _agno_run_id,
            "round": server_round,
            "model_version": fit_metrics.get("model_version", self._model_version),
            "n_clients": fit_metrics.get("n_clients", 0),
            "weight_mode": self.cfg.weight_mode,
            "selection_mode": self.cfg.selection_mode,
            "is_hl_round": self._last_is_hl,
            "total_samples": fit_metrics.get("total_samples", 0),
            "train_loss": fit_metrics.get("train_loss", 0.0),
            "train_acc": fit_metrics.get("train_acc", 0.0),
            "update_norm": fit_metrics.get("update_norm", 0.0),
            "deadline": fit_metrics.get("deadline", 0.0),
            "clients_on_time": fit_metrics.get("clients_on_time", 0),
            "clients_late": fit_metrics.get("clients_late", 0),
            "min_time": fit_metrics.get("min_time", 0.0),
            "max_time": fit_metrics.get("max_time", 0.0),
            "avg_time": fit_metrics.get("avg_time", 0.0),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "evaluation_mode": "centralized",
            "clients_detail": fit_metrics.get("clients_detail", []),
        }
        fl_round_log.append(round_entry)
        if self.recorder is not None:
            self.recorder.record_round(round_entry)
        if self.gui_callback:
            self.gui_callback(server_round, round_entry)
        return float(val_loss), {"accuracy": float(val_acc)}

    def finalize(self) -> None:
        """Close all invited updates with a persisted terminal state."""
        if not self._pending_updates:
            return
        candidates = []
        fallback = {}
        for update in self._pending_updates:
            tau = compute_staleness(self._model_version, update.base_version)
            candidate = self._client_context(update.cid, metrics=update.metrics, staleness_tau=tau)
            candidate.update({
                "invited_round": update.invited_round,
                "arrival_time": update.arrival_time,
                "lateness_seconds": max(0.0, update.arrival_time - (
                    update.round_start_time + update.round_deadline
                )),
                "final_round": True,
                "allowed_actions": ["discard"],
            })
            candidates.append(candidate)
            fallback[update.cid] = "discard"
        actions = self._decide_post_deadline(
            self.cfg.n_rounds, "experiment_finalization", candidates,
            fallback, {"discard"},
        )
        for update in self._pending_updates:
            tau = compute_staleness(self._model_version, update.base_version)
            state = "discarded_staleness" if tau > self.cfg.max_staleness else "discarded_deadline"
            self._record_state(
                self.cfg.n_rounds, update.cid, state,
                invited_round=update.invited_round,
                base_model_version=update.base_version,
                staleness_tau=tau,
                arrival_time=update.arrival_time,
                action=actions.get(update.cid, "discard"),
                reason="fim_do_experimento",
            )
        self._pending_updates.clear()


# ==============================================================================
#  ESTADO GLOBAL FL
# ==============================================================================

global_image_model  = CifarCNN()
fl_model_ready      = False
fl_training_status  = "Nao treinado"
fl_history: Optional[FLHistory] = None

NUM_ROUNDS_FL  = 5
NUM_CLIENTS_FL = 5

active_config:    Optional[ExperimentConfig] = None
config_file_name: str = ""

fl_round_log: List[Dict] = []

# Estado da politica Agno para uma execucao completa do experimento.
_agno_policy: Optional[AgnoRoundPolicy] = None
_agno_run_id: Optional[str] = None
_agent_selection_history: Dict[int, Dict[str, Any]] = {}

# Estado preparado por _prepare_fl_state antes de run_simulation
_sorted_stations: List[BaseStation] = []
_partitions:      List[Subset]      = []
_client_profiles: Dict[int, ClientProfile] = {}


# ==============================================================================
#  PREPARACAO DO ESTADO FL e CLIENT_FN
# ==============================================================================

def _prepare_fl_state(cfg: ExperimentConfig):
    """Inicializa sorted_stations e particoes antes de run_simulation."""
    global _sorted_stations, _partitions, _client_profiles

    all_stations = [bs for bs in BaseStation.all() if bs != CENTRAL_BS]
    _sorted_stations = sorted(
        all_stations,
        key=lambda b: dijkstra_distances.get(b.network_switch, float('inf'))
    )
    _partitions = partition_cifar10(len(_sorted_stations), seed=cfg.seed)
    _client_profiles = build_client_profiles(_partitions, cfg)
    print(f"  [FL] {len(_sorted_stations)} clientes particionados "
          f"(Non-IID, seed={cfg.seed})")


def client_fn(context: Context) -> fl.client.Client:
    """
    Factory de clientes para o ClientApp do Flower.
    node_id e atribuido sequencialmente (0..N-1) pelo run_simulation,
    mapeando diretamente ao indice em _sorted_stations.
    """
    node_id = int(context.node_id)
    idx     = node_id % len(_sorted_stations)
    bs      = _sorted_stations[idx]
    partition = _partitions[idx]

    cfg = active_config if active_config else ExperimentConfig()
    profile = _client_profiles[idx]
    client = CifarFederatedClient(
        cid=idx, base_station=bs,
        partition=partition, batch_size=cfg.batch_size,
        label_entropy=profile.entropy, seed=cfg.seed + idx * 1009,
    )
    return client.to_client()


# ==============================================================================
#  LOOP PRINCIPAL DE TREINAMENTO FEDERADO
# ==============================================================================

def run_federated_training_image(status_callback=None):
    global global_image_model, fl_model_ready, fl_training_status
    global fl_history, fl_round_log, active_config
    global _agno_policy, _agno_run_id, _agent_selection_history

    cfg = active_config if active_config is not None else ExperimentConfig(
        n_clients=NUM_CLIENTS_FL,
        n_rounds=NUM_ROUNDS_FL,
        weight_mode="random",
    )
    active_config = cfg
    fl_model_ready = False
    fl_history = None
    _agno_policy = None
    fl_round_log = []
    _agent_selection_history = {}
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    global_image_model = CifarCNN()

    def _cb(message: str) -> None:
        global fl_training_status
        fl_training_status = message
        if status_callback:
            status_callback(message)

    recorder: Optional[RunArtifactRecorder] = None
    strategy: Optional[TopologyAwareFedAvg] = None
    run_history: Optional[FLHistory] = None
    try:
        _agno_run_id = f"fl-{uuid4().hex}"
        recorder = RunArtifactRecorder(
            Path(RESULTS_DIR) / "runs",
            _agno_run_id,
            asdict(cfg),
        )
        configured_audit = Path(cfg.agent_audit_file)
        audit_path = (
            configured_audit
            if configured_audit.is_absolute()
            else recorder.run_dir / configured_audit.name
        )
        replay_path = Path(cfg.agent_replay_file) if cfg.agent_replay_file else None
        if replay_path is not None and not replay_path.is_absolute():
            replay_path = Path(_SCRIPT_DIR) / replay_path
        effective_config = asdict(cfg)
        effective_config["agent_audit_file"] = str(audit_path)
        effective_config["agent_replay_file"] = str(replay_path) if replay_path else ""
        recorder.write_effective_config(effective_config)
        _agno_policy = AgnoRoundPolicy(
            mode=cfg.agent_policy_mode,
            model_id=cfg.agent_model_id,
            audit_path=str(audit_path),
            timeout_s=cfg.agent_timeout_s,
            seed=cfg.seed,
            replay_path=str(replay_path) if replay_path else None,
        )
        print(f"  [Agno] modo={cfg.agent_policy_mode.upper()} | "
              f"modelo={cfg.agent_model_id} | run_id={_agno_run_id}")
        try:
            window.after(0, _gui_clear_round_list)
        except Exception:
            pass

        _cb("Carregando CIFAR-10...")
        if _cifar_trainset is None:
            load_cifar10()
        if not _images_loaded:
            load_sample_images_for_map()

        _cb("Particionando dataset...")
        _prepare_fl_state(cfg)
        n_total = len(_sorted_stations)
        n_clients = min(cfg.n_clients, n_total)

        print("\n" + "=" * 64)
        print("  APRENDIZADO FEDERADO - CIFAR-10 (Flower Strategy)")
        print(f"  Strategy      : TopologyAwareFedAvg")
        print(f"  Modo delays   : {cfg.weight_mode.upper()}")
        print(f"  Clientes/round: {n_clients}  (pool={n_total})")
        print(f"  Rounds        : {cfg.n_rounds}")
        print(f"  Selecao       : {cfg.selection_mode}")
        print(f"  Politica tardia: {cfg.late_update_policy}")
        print(f"  Resultados    : {recorder.run_dir}")
        print("  PRIVACIDADE   : apenas pesos e metricas agregadas trafegam")
        print("=" * 64)

        history = FLHistory(num_rounds=cfg.n_rounds, num_clients=n_clients)
        run_history = history

        def _gui_round_callback(round_num: int, entry: Dict) -> None:
            try:
                window.after(0, _gui_append_round, entry)
            except Exception:
                pass

        strategy = TopologyAwareFedAvg(
            cfg=cfg,
            sorted_stations=_sorted_stations,
            fl_history=history,
            gui_callback=_gui_round_callback,
            status_callback=_cb,
            recorder=recorder,
            initial_parameters=ndarrays_to_parameters(get_weights(global_image_model)),
        )
        if not _HAS_RUN_SIMULATION:
            raise RuntimeError(
                "Flower run_simulation nao esta disponivel. Execute uv sync "
                "para instalar flwr[simulation] e Ray."
            )

        client_app = ClientApp(client_fn=client_fn)
        server_app = ServerApp(
            config=ServerConfig(num_rounds=cfg.n_rounds),
            strategy=strategy,
        )
        _cb(f"Iniciando Flower ({cfg.n_rounds} rounds, {n_clients} clientes/round)...")
        _flwr_run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=n_total,
            backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
        )
        strategy.finalize()

        if strategy._last_aggregated_params is not None:
            set_weights(
                global_image_model,
                parameters_to_ndarrays(strategy._last_aggregated_params),
            )
        fl_history = history
        fl_model_ready = True
        _cb("Modelo pronto")
        recorder.finish({
            "status": "completed",
            "n_rounds_completed": len(history.losses_distributed),
            "final_summary": history.summary(),
        })

        print("\n" + "=" * 64)
        print("  FL com imagens concluido!")
        print(history.summary())
        print(f"  Artefatos salvos em: {recorder.run_dir}")
        print("=" * 64 + "\n")
        try:
            window.after(0, refresh_map)
        except Exception:
            pass
    except Exception as exc:
        if strategy is not None:
            try:
                strategy.finalize()
            except Exception:
                pass
        if recorder is not None:
            recorder.finish({
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "n_rounds_completed": (
                    len(run_history.losses_distributed) if run_history is not None else 0
                ),
            })
        _cb(f"Erro: {str(exc)[:120]}")
        print(f"\nErro no treinamento FL: {exc}")
        import traceback
        traceback.print_exc()


# ==============================================================================
#  CONFIGURACAO VISUAL DO GRAFO
# ==============================================================================

positions:        Dict = {}
node_labels:      Dict = {}
base_sizes:       List = []
base_colors:      List = []
edge_colors_list: List = []
edge_widths_list: List = []


def setup_graph_style():
    global positions, node_labels, base_sizes, base_colors
    global edge_colors_list, edge_widths_list

    topo  = Topology.first()
    max_d = max(dijkstra_distances.values()) if dijkstra_distances else 1.0

    positions   = {}
    node_labels = {}
    base_sizes  = []
    base_colors = []

    for node in topo.nodes():
        positions[node]   = node.coordinates
        if node == CENTRAL_SW:
            node_labels[node] = f"*\nS{node.id}"
            base_sizes.append(1100)
            base_colors.append("#FFD700")
        else:
            node_labels[node] = f"C{node.id}"
            base_sizes.append(600)
            d     = dijkstra_distances.get(node, max_d)
            ratio = d / max_d
            r = int(30  + ratio * 80)
            g = int(100 - ratio * 60)
            b = int(220 - ratio * 80)
            base_colors.append(f"#{r:02x}{g:02x}{b:02x}")

    edge_colors_list = []
    edge_widths_list = []
    for (u, v) in topo.edges():
        delay     = topo.edges[u, v]['delay']
        bandwidth = topo.edges[u, v]['bandwidth']
        cost = delay / bandwidth
        if cost < 0.1:
            edge_colors_list.append('#32CD32')
            edge_widths_list.append(1.8)
        elif cost < 0.8:
            edge_colors_list.append('#FFA500')
            edge_widths_list.append(2.2)
        else:
            edge_colors_list.append('#FF4444')
            edge_widths_list.append(2.8)


setup_graph_style()

# ==============================================================================
#  GUI - JANELA PRINCIPAL
# ==============================================================================

window = tk.Tk()
window.title(
    "SimulatorFederatedLearning"
)
window.geometry("1480x920")
window.configure(bg="#1A1A2E")


def on_closing():
    window.quit()
    window.destroy()
    sys.exit(0)


window.protocol("WM_DELETE_WINDOW", on_closing)

style = ttk.Style()
try:
    style.theme_use("clam")
except Exception:
    pass
style.configure("TFrame",           background="#1A1A2E")
style.configure("TLabel",           background="#1A1A2E", foreground="#E0E0E0")
style.configure("TButton",          background="#2A2A5A", foreground="#E0E0E0")
style.configure("TCombobox",        fieldbackground="#2A2A5A", foreground="#E0E0E0")
style.configure("TSeparator",       background="#3A3A6A")
style.configure("TLabelframe",      background="#1A1A2E", foreground="#E0E0E0")
style.configure("TLabelframe.Label",background="#1A1A2E", foreground="#FFD700")

ctrl_frame = ttk.Frame(window, padding="6")
ctrl_frame.pack(fill=tk.X, side=tk.TOP)

fl_status_var = tk.StringVar(value=f"Status: {fl_training_status}")


def update_fl_status(text: str):
    fl_status_var.set(f"Status: {text}")
    window.update_idletasks()


def _apply_config(path: str, display_name: str):
    global active_config, config_file_name
    try:
        cfg = ExperimentConfig.from_json(path)
        active_config    = cfg
        config_file_name = display_name
        _refresh_config_panel()
        messagebox.showinfo("Configuracao carregada",
                            f"Fonte: {config_file_name}\n"
                            + "\n".join(cfg.to_summary_lines()))
    except Exception as e:
        messagebox.showerror("Erro ao carregar arquivo JSON", str(e))


def _load_from_url_dialog():
    url_win = tk.Toplevel(window)
    url_win.title("Carregar Parametros")
    url_win.geometry("500x160")
    url_win.configure(bg="#1A1A2E")
    url_win.resizable(False, False)
    url_win.grab_set()

    ttk.Label(url_win, text="URL do arquivo JSON:",
              foreground="#E0E0E0").pack(pady=(18, 4))
    url_entry = ttk.Entry(url_win, width=58)
    url_entry.pack(padx=20)
    url_entry.insert(0, "https://")

    def _fetch():
        url = url_entry.get().strip()
        if not url or url == "https://":
            return
        url_win.destroy()
        try:
            update_fl_status("Baixando JSON...")
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                             delete=False, encoding='utf-8') as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            name = url.rstrip("/").split("/")[-1] or "url_config.json"
            _apply_config(tmp_path, name)
            os.unlink(tmp_path)
            update_fl_status("Config URL carregada")
        except Exception as e:
            messagebox.showerror("Erro ao baixar JSON", str(e))
            update_fl_status("Erro no download")

    btn_f = ttk.Frame(url_win)
    btn_f.pack(pady=14)
    ttk.Button(btn_f, text="Carregar", command=_fetch).pack(side=tk.LEFT, padx=10)
    ttk.Button(btn_f, text="Cancelar", command=url_win.destroy).pack(side=tk.LEFT, padx=10)


def load_config_json():
    choice_win = tk.Toplevel(window)
    choice_win.title("Carregar Configuracao JSON")
    choice_win.geometry("380x170")
    choice_win.configure(bg="#1A1A2E")
    choice_win.resizable(False, False)
    choice_win.grab_set()

    ttk.Label(choice_win, text="Como deseja carregar o JSON?",
              font=("Arial", 11, "bold"), foreground="#FFD700",
              background="#1A1A2E").pack(pady=(20, 12))

    def _file():
        choice_win.destroy()
        path = filedialog.askopenfilename(
            title="Selecionar configuracao JSON",
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")]
        )
        if path:
            _apply_config(path, os.path.basename(path))

    def _url():
        choice_win.destroy()
        _load_from_url_dialog()

    btn_f = ttk.Frame(choice_win)
    btn_f.pack(pady=4)
    ttk.Button(btn_f, text="Arquivo Local",
               command=_file, width=18).pack(side=tk.LEFT, padx=10)
    ttk.Button(btn_f, text="URL / Link",
               command=_url, width=18).pack(side=tk.LEFT, padx=10)
    ttk.Button(choice_win, text="Cancelar",
               command=choice_win.destroy).pack(pady=8)


def _refresh_config_panel():
    if active_config is None:
        cfg_text_var.set("Nenhum arquivo de configuracao carregado.\nUsando valores padrao.")
    else:
        lines = [f"Fonte: {config_file_name}"] + active_config.to_summary_lines()
        cfg_text_var.set("\n".join(lines))


def start_fl_training():
    if active_config is None:
        defaults = ExperimentConfig()
        ok = messagebox.askyesno(
            "Sem configuracao JSON",
            f"Nenhum JSON carregado.\n"
            f"Usar padroes: {defaults.n_clients} clientes, "
            f"{defaults.n_rounds} rounds, modo '{defaults.weight_mode}'.\n\n"
            "Deseja continuar?"
        )
        if not ok:
            return

    if _cifar_trainset is None:
        messagebox.showinfo("Atencao",
                            "Carregue o dataset CIFAR-10 primeiro\n"
                            "(botao 'Carregar Dataset').")
        return
    fl_btn.config(state=tk.DISABLED)
    metrics_btn.config(state=tk.DISABLED)

    def _thread():
        run_federated_training_image(status_callback=update_fl_status)
        window.after(0, lambda: fl_btn.config(state=tk.NORMAL))
        window.after(0, lambda: metrics_btn.config(state=tk.NORMAL))

    threading.Thread(target=_thread, daemon=True).start()


def start_load_dataset():
    load_btn.config(state=tk.DISABLED)
    update_fl_status("Baixando CIFAR-10...")

    def _thread():
        load_cifar10()
        load_sample_images_for_map()
        update_fl_status("Dataset pronto")
        window.after(0, refresh_map)
        window.after(0, lambda: load_btn.config(state=tk.NORMAL))

    threading.Thread(target=_thread, daemon=True).start()


load_btn = ttk.Button(ctrl_frame, text="Carregar Dataset",
                      command=start_load_dataset)
load_btn.pack(side=tk.LEFT, padx=4)

fl_btn = ttk.Button(ctrl_frame, text="Treinar Modelo",
                    command=start_fl_training)
fl_btn.pack(side=tk.LEFT, padx=4)

ttk.Label(ctrl_frame, textvariable=fl_status_var,
          foreground="#9ECDFF").pack(side=tk.LEFT, padx=6)

ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

metrics_btn = ttk.Button(ctrl_frame, text="Ver metricas",
                         command=lambda: show_fl_metrics())
metrics_btn.pack(side=tk.LEFT, padx=4)

ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

json_btn = ttk.Button(ctrl_frame, text="Configurações",
                      command=load_config_json)
json_btn.pack(side=tk.LEFT, padx=4)

content_frame = ttk.Frame(window)
content_frame.pack(fill=tk.BOTH, expand=True)

map_frame = ttk.Frame(content_frame)
map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

side_frame = ttk.Frame(content_frame, width=380)
side_frame.pack(side=tk.RIGHT, fill=tk.Y)
side_frame.pack_propagate(False)

fig_map, ax_map = plt.subplots(figsize=(11, 7.5))
fig_map.patch.set_facecolor("#0F0F1A")
ax_map.set_facecolor("#0F0F1A")

canvas_map = FigureCanvasTkAgg(fig_map, master=map_frame)
canvas_map.draw()
canvas_map.get_tk_widget().pack(fill=tk.BOTH, expand=True)

cfg_panel = ttk.LabelFrame(side_frame, text="Configuracao Ativa", padding=6)
cfg_panel.pack(fill=tk.X, padx=8, pady=(8, 2))
cfg_text_var = tk.StringVar(
    value="Nenhum arquivo de configuracao carregado.\nUsando valores padrao."
)
ttk.Label(cfg_panel, textvariable=cfg_text_var, justify=tk.LEFT,
          foreground="#B0BEC5", font=("Consolas", 8)).pack(fill=tk.X)

rounds_panel = ttk.LabelFrame(side_frame, text="Rodadas de Treinamento", padding=6)
rounds_panel.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 4))

_rounds_list_frame = ttk.Frame(rounds_panel)
_rounds_list_frame.pack(fill=tk.BOTH, expand=True)

rounds_listbox = tk.Listbox(
    _rounds_list_frame,
    bg="#0F0F1A", fg="#9ECDFF",
    selectbackground="#2A4A8A", selectforeground="#FFFFFF",
    font=("Consolas", 9),
    activestyle="none",
    borderwidth=0, highlightthickness=0,
    relief=tk.FLAT,
)
rounds_scrollbar = ttk.Scrollbar(_rounds_list_frame, orient=tk.VERTICAL,
                                  command=rounds_listbox.yview)
rounds_listbox.configure(yscrollcommand=rounds_scrollbar.set)
rounds_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
rounds_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

details_panel = ttk.LabelFrame(side_frame, text="Detalhes da Rodada", padding=8)
details_panel.pack(fill=tk.X, padx=8, pady=(0, 4))

details_text_var = tk.StringVar(value="Clique em uma rodada para\nver os detalhes.")
details_label = ttk.Label(
    details_panel,
    textvariable=details_text_var,
    justify=tk.LEFT,
    wraplength=340,
    foreground="#CFD8DC",
    font=("Consolas", 9),
)
details_label.pack(fill=tk.X)

legend_frame = ttk.LabelFrame(side_frame, text="Legenda", padding=6)
legend_frame.pack(fill=tk.X, padx=8, pady=4)

legend_items = [
    ("*  Servidor Central", "#FFD700"),
    ("o  Cliente (perto)", "#3060DC"),
    ("o  Cliente (longe)",  "#5A2A8A"),
    ("-- Link rapido",      "#32CD32"),
    ("-- Link medio",       "#FFA500"),
    ("-- Link lento",       "#FF4444"),
]
for text, color in legend_items:
    lf = ttk.Frame(legend_frame)
    lf.pack(fill=tk.X, pady=1)
    tk.Label(lf, text="=", foreground=color,
             background="#1A1A2E", font=("Arial", 11)).pack(side=tk.LEFT)
    ttk.Label(lf, text=text, foreground="#CFD8DC",
              font=("Arial", 9)).pack(side=tk.LEFT, padx=4)

_selected_bs: Optional[BaseStation] = None


# ==============================================================================
#  DESENHO DO MAPA DE REDE
# ==============================================================================

def draw_network_map(highlight_bs: Optional[BaseStation] = None,
                     highlight_path: Optional[List] = None):
    topo = Topology.first()
    ax_map.cla()
    ax_map.set_facecolor("#0F0F1A")

    max_d = max(dijkstra_distances.values()) if dijkstra_distances else 1.0
    node_colors = []
    for node in topo.nodes():
        if node == CENTRAL_SW:
            node_colors.append("#FFD700")
        elif highlight_bs and node == highlight_bs.network_switch:
            node_colors.append("#00FFFF")
        else:
            d     = dijkstra_distances.get(node, max_d)
            ratio = d / max_d
            r = int(30  + ratio * 80)
            g = int(100 - ratio * 60)
            b = int(220 - ratio * 80)
            node_colors.append(f"#{r:02x}{g:02x}{b:02x}")

    nx.draw_networkx_edges(
        topo, ax=ax_map, pos=positions,
        edge_color=edge_colors_list,
        width=edge_widths_list,
        alpha=0.7
    )
    nx.draw_networkx_nodes(
        topo, ax=ax_map, pos=positions,
        node_color=node_colors,
        node_size=base_sizes,
        alpha=0.9
    )

    if highlight_path and len(highlight_path) >= 2:
        path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
        nx.draw_networkx_edges(
            topo, ax=ax_map, pos=positions,
            edgelist=path_edges,
            edge_color="#00FF88",
            width=4.0,
            style="solid",
            alpha=1.0
        )

    if _images_loaded and _client_sample_images:
        for bs in client_stations:
            if bs.id in _client_sample_images:
                img_arr, cls_name, cls_idx = _client_sample_images[bs.id]
                node = bs.network_switch
                if node in positions:
                    zoom = 0.48 if node != (highlight_bs.network_switch
                                            if highlight_bs else None) else 0.62
                    imagebox = OffsetImage(img_arr, zoom=zoom)
                    imagebox.image.axes = ax_map
                    border_color = ("#00FFFF" if highlight_bs and
                                    node == highlight_bs.network_switch
                                    else CLASS_COLORS[cls_idx])
                    ab = AnnotationBbox(
                        imagebox, positions[node],
                        frameon=True,
                        bboxprops=dict(edgecolor=border_color,
                                       linewidth=2, boxstyle="round,pad=0.1"),
                        pad=0.05,
                        xycoords='data'
                    )
                    ax_map.add_artist(ab)

    cx, cy = CENTRAL_SW.coordinates
    ax_map.scatter(cx, cy, c="#FFD700", s=800, zorder=10,
                   edgecolors="white", linewidths=2)
    ax_map.text(cx, cy + 0.25, f"* Servidor\nBS{CENTRAL_BS.id}",
                ha='center', va='bottom', fontsize=7, fontweight='bold',
                color="#FFD700")

    if not _images_loaded:
        for node in topo.nodes():
            if node != CENTRAL_SW:
                x, y = positions[node]
                ax_map.text(x, y, f"C{node.id}", ha='center', va='center',
                            fontsize=6, color='white', fontweight='bold')

    ax_map.set_title(
        "Rede Hexagonal 5x5 - SimulatorFederatedLearning com CIFAR-10\n"
        f"Servidor Central: BS{CENTRAL_BS.id} | "
        f"Clientes: {len(client_stations)} | "
        f"Strategy: TopologyAwareFedAvg",
        color="#E0E0E0", fontsize=11, pad=10
    )
    ax_map.axis('off')
    canvas_map.draw()


def draw_initial():
    draw_network_map()


# ==============================================================================
#  CALLBACKS DA GUI
# ==============================================================================

def _gui_clear_round_list():
    rounds_listbox.delete(0, tk.END)
    details_text_var.set("Clique em uma rodada para\nver os detalhes.")


def _gui_append_round(entry: Dict):
    r    = entry["round"]
    acc  = entry["val_acc"] * 100
    loss = entry["val_loss"]
    n    = entry["n_clients"]
    mode = entry.get("weight_mode", "?")[0].upper()
    hl   = " HL" if entry.get("is_hl_round") else ""
    line = f" R{r:>2} [{mode}]{hl}  Loss {loss:.4f}  Acc {acc:5.1f}%  C:{n}"
    rounds_listbox.insert(tk.END, line)
    if entry.get("is_hl_round"):
        color = "#3A1A0A"
    else:
        color = "#1A2A4A" if r % 2 == 0 else "#0F1A30"
    rounds_listbox.itemconfig(tk.END, background=color)
    rounds_listbox.selection_clear(0, tk.END)
    rounds_listbox.selection_set(tk.END)
    rounds_listbox.see(tk.END)
    _show_round_details(len(fl_round_log) - 1)


def _show_round_details(idx: int):
    if idx < 0 or idx >= len(fl_round_log):
        details_text_var.set("Nenhum dado disponivel.")
        return
    e        = fl_round_log[idx]
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
        prev   = fl_round_log[idx - 1]
        d_loss = vl - prev["val_loss"]
        d_acc  = va - prev["val_acc"] * 100
        sign_l = "+" if d_loss >= 0 else ""
        sign_a = "+" if d_acc  >= 0 else ""
        delta_loss_str = f"  ({sign_l}{d_loss:.4f})"
        delta_acc_str  = f"  ({sign_a}{d_acc:.2f}%)"

    total_rounds = (active_config.n_rounds if active_config else NUM_ROUNDS_FL)
    mode  = e.get("weight_mode", "?").upper()
    is_hl = e.get("is_hl_round", False)
    sep   = "=" * 33

    deadline_s = e.get("deadline", 0.0)
    on_time_c  = e.get("clients_on_time", n_cli)
    late_c     = e.get("clients_late", 0)
    avg_t      = e.get("avg_time", 0.0)
    up_norm    = e.get("update_norm", 0.0)

    lines = [
        f"+{sep}+",
        f"  Rodada  : {r} / {total_rounds}  [{mode}]{'  HL' if is_hl else ''}",
        f"  Clientes: {n_cli} (No prazo: {on_time_c} | Atrasados: {late_c})",
        f"  Deadline: {deadline_s:.2f}s | T.Med: {avg_t:.2f}s",
        f"  Amostras: {samples:,}",
        f"+{sep}+",
        f"  TREINO  (FedAvg ponderado)",
        f"  Loss    : {tl:.4f}",
        f"  Acc     : {ta:.2f}%",
        f"  ||Δw||  : {up_norm:.3f} (norma media)",
        f"+{sep}+",
        f"  VALIDACAO",
        f"  Loss    : {vl:.4f}{delta_loss_str}",
        f"  Acc     : {va:.2f}%{delta_acc_str}",
        f"+{sep}+",
        f"  CLIENTES (Tempo | Prazo | Inform.)",
    ]

    for cd in e.get("clients_detail", []):
        cid     = cd["cid"]
        bs_id   = cd["bs_id"]
        t_tot   = cd.get("t_total")
        on_tm   = cd.get("on_time")
        tau     = cd.get("staleness", 0)
        c_loss  = cd.get("loss")
        u_norm  = cd.get("update_norm")
        status  = cd.get("status")
        status_labels = {
            "on_time": "OK",
            "late_pending": "PENDENTE",
            "accepted_stale": f"STALE(tau={tau})",
            "accepted_reduced": "PESO REDUZIDO",
            "discarded_deadline": "DESCARTADO (prazo)",
            "discarded_staleness": f"DESCARTADO (tau={tau})",
            "failed": "FALHOU",
        }
        st_tag = status_labels.get(
            status,
            "LATE" if on_tm is False else "OK" if on_tm is True else "DESCONHECIDO",
        )
        time_label = f"{float(t_tot):4.1f}s" if t_tot is not None else "   -- "
        norm_label = f"{float(u_norm):4.2f}" if u_norm is not None else "  --"
        loss_label = f"{float(c_loss):4.2f}" if c_loss is not None else "  --"
        lines.append(
            f"  [C{cid:02d}|BS{bs_id:<2}] t={time_label} [{st_tag}]"
            f" ||dw||={norm_label} L={loss_label}"
        )

    lines.append(f"+{sep}+")
    details_text_var.set("\n".join(lines))


def _on_round_select(event):
    sel = rounds_listbox.curselection()
    if sel:
        _show_round_details(sel[0])


rounds_listbox.bind("<<ListboxSelect>>", _on_round_select)


def refresh_map():
    draw_network_map()


# ==============================================================================
#  JANELA DE METRICAS FL
# ==============================================================================

def show_fl_metrics():
    if fl_history is None or not fl_history.losses_distributed:
        messagebox.showinfo(
            "Metricas FL",
            "Nenhuma metrica disponivel.\n"
            "Execute o treinamento FL primeiro."
        )
        return

    met_win = tk.Toplevel(window)
    met_win.title("Metricas - SimulatorFederatedLearning CIFAR-10")
    met_win.geometry("1000x680")
    met_win.configure(bg="#1A1A2E")

    fig_m, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig_m.patch.set_facecolor("#1A1A2E")
    fig_m.suptitle(
        f"SimulatorFederatedLearning - CIFAR-10 | {fl_history.num_rounds} rounds | "
        f"{fl_history.num_clients} clientes | TopologyAwareFedAvg + Dijkstra",
        fontsize=12, fontweight='bold', color='#E0E0E0'
    )

    rounds_e = [r for r, _ in fl_history.losses_distributed]
    losses_e = [l for _, l in fl_history.losses_distributed]

    def _style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#0F0F1A")
        ax.set_title(title, color='#E0E0E0', fontsize=10)
        ax.set_xlabel(xlabel, color='#9E9E9E', fontsize=9)
        ax.set_ylabel(ylabel, color='#9E9E9E', fontsize=9)
        ax.tick_params(colors='#9E9E9E')
        ax.grid(True, alpha=0.2, color='#333355')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    axes[0, 0].plot(rounds_e, losses_e, 'o-', color='#4FC3F7',
                    linewidth=2, markersize=7, label="Val Loss")
    axes[0, 0].fill_between(rounds_e, losses_e, alpha=0.15, color='#4FC3F7')
    _style_ax(axes[0, 0], "Loss de Validacao por Round", "Round", "Loss")
    axes[0, 0].legend(facecolor="#1A1A2E", labelcolor="#E0E0E0")

    if 'accuracy' in fl_history.eval_metrics:
        acc_rounds = [r for r, _ in fl_history.eval_metrics['accuracy']]
        acc_vals   = [v * 100 for _, v in fl_history.eval_metrics['accuracy']]
        axes[0, 1].plot(acc_rounds, acc_vals, 's-', color='#66BB6A',
                        linewidth=2, markersize=7, label="Val Acc")
        axes[0, 1].fill_between(acc_rounds, acc_vals, alpha=0.15, color='#66BB6A')
        axes[0, 1].set_ylim(0, 100)
        _style_ax(axes[0, 1], "Acuracia de Validacao por Round", "Round", "Acuracia (%)")
        axes[0, 1].legend(facecolor="#1A1A2E", labelcolor="#E0E0E0")

    if 'train_loss' in fl_history.fit_metrics:
        tl_r = [r for r, _ in fl_history.fit_metrics['train_loss']]
        tl_v = [v for _, v in fl_history.fit_metrics['train_loss']]
        axes[1, 0].plot(tl_r, tl_v, 'D-', color='#FFA726',
                        linewidth=2, markersize=7, label="Train Loss")
        axes[1, 0].fill_between(tl_r, tl_v, alpha=0.15, color='#FFA726')
        _style_ax(axes[1, 0], "Train Loss (FedAvg Agregado)", "Round", "Loss")
        axes[1, 0].legend(facecolor="#1A1A2E", labelcolor="#E0E0E0")

    if 'train_acc' in fl_history.fit_metrics:
        ta_r = [r for r, _ in fl_history.fit_metrics['train_acc']]
        ta_v = [v * 100 for _, v in fl_history.fit_metrics['train_acc']]
        axes[1, 1].plot(ta_r, ta_v, '^-', color='#CE93D8',
                        linewidth=2, markersize=7, label="Train Acc")
        axes[1, 1].fill_between(ta_r, ta_v, alpha=0.15, color='#CE93D8')
        axes[1, 1].set_ylim(0, 100)
        _style_ax(axes[1, 1], "Train Accuracy (Agregada)", "Round", "Acuracia (%)")
        axes[1, 1].legend(facecolor="#1A1A2E", labelcolor="#E0E0E0")

    plt.tight_layout()
    fig_m.savefig(os.path.join(RESULTS_DIR, "fl_cifar10_metrics.png"),
                  dpi=120, bbox_inches="tight", facecolor=fig_m.get_facecolor())
    print(f"  Grafico salvo -> {RESULTS_DIR}/fl_cifar10_metrics.png")

    canvas_m = FigureCanvasTkAgg(fig_m, master=met_win)
    canvas_m.draw()
    canvas_m.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    sum_frame = ttk.LabelFrame(met_win, text="Resumo Final", padding=6)
    sum_frame.pack(fill=tk.X, padx=10, pady=6)

    final_r    = fl_history.losses_distributed[-1][0]
    final_loss = fl_history.losses_distributed[-1][1]
    final_acc  = (fl_history.eval_metrics['accuracy'][-1][1] * 100
                  if 'accuracy' in fl_history.eval_metrics else 0)

    ttk.Label(
        sum_frame,
        text=(f"Round {final_r}  |  Val Loss: {final_loss:.4f}  |  "
              f"Val Acc: {final_acc:.2f}%  |  "
              f"Clientes: {fl_history.num_clients}  |  "
              f"Strategy: TopologyAwareFedAvg + Dijkstra"),
        font=('Arial', 10, 'bold')
    ).pack()

    def _close():
        plt.close(fig_m)
        met_win.destroy()

    met_win.protocol("WM_DELETE_WINDOW", _close)


# ==============================================================================
#  INICIALIZACAO
# ==============================================================================

if __name__ == "__main__":
    draw_initial()

    print("  Interface carregada. Use os botoes para:")
    print("  1. 'Carregar CIFAR-10'          -> baixa dataset e exibe thumbnails nos nos")
    print("  2. 'Treinar Modelo'             -> executa rounds via Flower (ClientApp + ServerApp)")
    print("  3. 'Ver metricas FL'            -> graficos de convergencia")
    print("  4. 'Carregar configuracao JSON' -> personaliza o experimento\n")

    window.mainloop()
