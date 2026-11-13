# -*- coding: utf-8 -*-
"""
fl_engine — nucleo do simulador (sem interface grafica)
Topologia Edge Computing hexagonal com aprendizado federado via Flower nativo.
Use run_experiment.py (linha de comando) ou SimulatorFederatedLearning.py (GUI).

Arquitetura Flower:
  - ClientApp  → encapsula CifarFederatedClient (NumPyClient)
  - ServerApp  → configura rodadas e strategy
  - TopologyAwareFedAvg(FedAvg) → seleção por Dijkstra, alta-latência,
    coleta de métricas e callbacks opcionais para a interface
"""

import sys

import networkx as nx
import numpy as np
import random
import os
import json
import threading
import urllib.request
import tempfile
from pathlib import Path
from uuid import uuid4
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
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
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

# Modulos extraidos do motor (reexportados aqui para a GUI e os testes).
from experiment_config import (
    CLASS_COLORS,
    CLASS_NAMES,
    DATA_DIR,
    ExperimentConfig,
    NUM_CLASSES,
    RESULTS_DIR,
    _SCRIPT_DIR,
)
from edge_network import (
    BaseStation,
    CENTRAL_BS,
    CENTRAL_SERVER,
    CENTRAL_SW,
    ClientProfile,
    ClientTimingResult,
    EdgeServer,
    NetworkSwitch,
    PendingUpdate,
    ScenarioExporter,
    Topology,
    User,
    _dijkstra_lengths,
    betweenness,
    bs,
    build_client_profiles,
    central_server,
    central_switch,
    client_stations,
    compute_client_timing,
    coords,
    dijkstra_distances,
    generate_random_links,
    get_dijkstra_path,
    get_model_size_megabits,
    get_path_network_metrics,
    get_path_to_server,
    hexagonal_grid,
    i,
    map_coordinates,
    partially_connected_hexagonal_mesh,
    topology_ref,
)
from cifar_model import (
    CifarCNN,
    DEVICE,
    eval_image_local,
    get_weights,
    set_weights,
    train_image_local,
)


# ==============================================================================
#  DATASET CIFAR-10
# ==============================================================================

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

        proxies = self._sorted_proxies(client_manager)
        if len(proxies) < len(selected_indices):
            raise RuntimeError(
                f"Flower registrou {len(proxies)} clientes, mas {len(selected_indices)} "
                "foram selecionados"
            )
        instructions = []
        for proxy, cid in zip(proxies, selected_indices):
            config = {
                "local_epochs": self.cfg.local_epochs,
                "server_round": server_round,
                "base_model_version": self._model_version,
                "station_idx": cid,
            }
            instructions.append((proxy, FitIns(parameters, config)))
        return instructions

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
            int(fit_res.metrics["station_idx"]): (proxy, fit_res)
            for proxy, fit_res in results
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


class StationClient(NumPyClient):
    """Cliente generico: a estacao e escolhida pelo servidor via config['station_idx'].

    Os node_ids do Flower sao inteiros aleatorios, entao nao ha mapeamento
    estavel entre node_id e indice de estacao. O servidor indica o indice a
    cada rodada e o cliente devolve 'station_idx' nas metricas.
    """

    @staticmethod
    def _build(idx: int) -> CifarFederatedClient:
        idx = int(idx) % len(_sorted_stations)
        cfg = active_config if active_config else ExperimentConfig()
        profile = _client_profiles[idx]
        return CifarFederatedClient(
            cid=idx, base_station=_sorted_stations[idx],
            partition=_partitions[idx], batch_size=cfg.batch_size,
            label_entropy=profile.entropy, seed=cfg.seed + idx * 1009,
        )

    def get_parameters(self, config) -> NDArrays:
        return self._build(0).get_parameters(config)

    def fit(self, parameters: NDArrays, config) -> Tuple[NDArrays, int, Dict]:
        idx = int(config["station_idx"])
        weights, n_train, metrics = self._build(idx).fit(parameters, config)
        metrics["station_idx"] = idx
        return weights, n_train, metrics


def client_fn(context: Context) -> fl.client.Client:
    """Factory de clientes para o ClientApp do Flower."""
    return StationClient().to_client()


# ==============================================================================
#  LOOP PRINCIPAL DE TREINAMENTO FEDERADO
# ==============================================================================

def run_federated_training_image(status_callback=None, round_callback=None,
                                  on_start=None, on_finish=None,
                                  load_thumbnails=True):
    """Executa o experimento. Os callbacks sao opcionais (a GUI os usa)."""
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
        _agno_policy = AgnoRoundPolicy(
            mode=cfg.agent_policy_mode,
            model_id=cfg.agent_model_id,
            audit_path=str(audit_path),
            timeout_s=cfg.agent_timeout_s,
            seed=cfg.seed,
            replay_path=str(replay_path) if replay_path else None,
            provider=cfg.agent_provider,
        )
        # Registra os valores efetivos (apos precedencia do ambiente/.env).
        agent_settings = _agno_policy.settings
        effective_config = asdict(cfg)
        effective_config["agent_audit_file"] = str(audit_path)
        effective_config["agent_replay_file"] = str(replay_path) if replay_path else ""
        effective_config["agent_provider"] = agent_settings.provider
        effective_config["agent_model_id"] = agent_settings.model_id
        effective_config["agent_name"] = agent_settings.name
        effective_config["agent_strategy_instructions"] = agent_settings.strategy_instructions
        effective_config["agent_use_json_mode"] = agent_settings.use_json_mode
        recorder.write_effective_config(effective_config)
        print(f"  [Agno] modo={cfg.agent_policy_mode.upper()} | "
              f"modelo={agent_settings.formatted_model} | run_id={_agno_run_id}")
        if on_start is not None:
            on_start()

        _cb("Carregando CIFAR-10...")
        if _cifar_trainset is None:
            load_cifar10()
        if load_thumbnails and not _images_loaded:
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

        strategy = TopologyAwareFedAvg(
            cfg=cfg,
            sorted_stations=_sorted_stations,
            fl_history=history,
            gui_callback=round_callback,
            status_callback=_cb,
            recorder=recorder,
            initial_parameters=ndarrays_to_parameters(get_weights(global_image_model)),
        )
        if not _HAS_RUN_SIMULATION:
            raise RuntimeError(
                "Flower run_simulation nao esta disponivel. Execute uv sync "
                "para instalar flwr[simulation] e Ray."
            )

        # Os workers do Ray nao reimportam este modulo com o estado preparado;
        # serializar por valor leva as particoes e a topologia junto, como em __main__.
        from ray import cloudpickle
        for module_name in (__name__, "experiment_config", "edge_network", "cifar_model"):
            cloudpickle.register_pickle_by_value(sys.modules[module_name])

        client_app = ClientApp(client_fn=client_fn)

        def server_fn(context: Context) -> ServerAppComponents:
            return ServerAppComponents(
                strategy=strategy,
                config=ServerConfig(num_rounds=cfg.n_rounds),
            )

        server_app = ServerApp(server_fn=server_fn)
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
        if on_finish is not None:
            on_finish()
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
