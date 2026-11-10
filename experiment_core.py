"""Pure helpers for the federated-learning experiment.

The Tkinter application and Flower adapters call these functions, while tests can
exercise partitioning, selection, deadlines, and aggregation without starting a
GUI or downloading CIFAR-10.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


def partition_indices(
    labels: Sequence[int],
    n_clients: int,
    seed: int,
    dirichlet_alpha: float = 0.4,
) -> list[list[int]]:
    """Return a reproducible, disjoint, non-IID partition covering every item."""
    if n_clients < 1:
        raise ValueError("n_clients deve ser maior que zero")
    if seed < 0:
        raise ValueError("seed deve ser nao negativa")
    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha deve ser maior que zero")
    if len(labels) == 0:
        return [[] for _ in range(n_clients)]

    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels, dtype=np.int64)
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]

    for label in np.unique(labels_array):
        indices = np.flatnonzero(labels_array == label)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(n_clients, dirichlet_alpha))
        counts = rng.multinomial(len(indices), proportions)
        offset = 0
        for cid, count in enumerate(counts):
            client_indices[cid].extend(indices[offset:offset + count].tolist())
            offset += int(count)

    # Dirichlet draws can leave a client empty in small synthetic/test datasets.
    # Move one sample from the largest client so all clients can participate.
    for cid in range(n_clients):
        if client_indices[cid]:
            continue
        donor = max(range(n_clients), key=lambda i: len(client_indices[i]))
        if len(client_indices[donor]) <= 1:
            raise ValueError("nao ha amostras suficientes para todos os clientes")
        client_indices[cid].append(client_indices[donor].pop())

    for indices in client_indices:
        rng.shuffle(indices)
    return client_indices


def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Return Jensen-Shannon divergence in nats for sparse distributions."""
    p_array = np.asarray(p, dtype=float)
    q_array = np.asarray(q, dtype=float)
    if p_array.shape != q_array.shape or p_array.ndim != 1:
        raise ValueError("p e q devem ter o mesmo formato unidimensional")
    if np.any(p_array < 0) or np.any(q_array < 0):
        raise ValueError("distribuicoes nao podem conter valores negativos")

    p_array = p_array / max(float(p_array.sum()), 1e-12)
    q_array = q_array / max(float(q_array.sum()), 1e-12)
    return _jensen_shannon_from_distributions(p_array, q_array)


def _jensen_shannon_from_distributions(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = 0.5 * (p + q)
    mask_p, mask_q = p > 0, q > 0
    return float(
        0.5 * np.sum(p[mask_p] * np.log(p[mask_p] / midpoint[mask_p]))
        + 0.5 * np.sum(q[mask_q] * np.log(q[mask_q] / midpoint[mask_q]))
    )


def label_distribution_metrics(
    local_counts: Sequence[int],
    global_counts: Sequence[int],
    rare_class_threshold: float,
) -> dict[str, float]:
    """Compute auditable entropy, rare-label coverage, and Jensen-Shannon divergence."""
    local = np.asarray(local_counts, dtype=float)
    global_ = np.asarray(global_counts, dtype=float)
    if local.shape != global_.shape or local.ndim != 1:
        raise ValueError("local_counts e global_counts devem ter o mesmo formato")
    if np.any(local < 0) or np.any(global_ < 0):
        raise ValueError("contagens de classes nao podem ser negativas")
    if not 0 <= rare_class_threshold <= 1:
        raise ValueError("rare_class_threshold deve estar entre 0 e 1")

    local_dist = local / max(float(local.sum()), 1.0)
    global_dist = global_ / max(float(global_.sum()), 1.0)
    positive_local = local_dist > 0
    entropy = float(-np.sum(local_dist[positive_local] * np.log2(local_dist[positive_local])))
    rare_classes = global_dist <= rare_class_threshold
    rare_coverage = (
        float(np.count_nonzero((local > 0) & rare_classes) / np.count_nonzero(rare_classes))
        if np.any(rare_classes) else 0.0
    )

    js_divergence = _jensen_shannon_from_distributions(local_dist, global_dist)
    return {
        "entropy": entropy,
        "rare_coverage": rare_coverage,
        "js_divergence": js_divergence,
    }


def choose_client_ids(
    mode: str,
    candidate_ids: Iterable[int],
    n_clients: int,
    predicted_times: Mapping[int, float],
    dijkstra_costs: Mapping[int, float],
    seed: int,
    round_num: int,
    score_fn: Callable[[int, list[int]], float] | None = None,
) -> list[int]:
    """Choose clients using a named, repeatable experimental baseline."""
    ids = sorted(set(int(cid) for cid in candidate_ids))
    if n_clients < 0 or n_clients > len(ids):
        raise ValueError("n_clients fora do intervalo de candidatos")
    normalized_mode = "dijkstra" if mode == "latency" else mode
    if normalized_mode not in {"random", "fastest", "dijkstra", "multi_objective"}:
        raise ValueError(f"selection_mode desconhecido: {mode}")
    if n_clients == 0:
        return []

    if normalized_mode == "random":
        rng = np.random.default_rng(np.random.SeedSequence([seed, round_num]))
        return sorted(int(cid) for cid in rng.choice(ids, size=n_clients, replace=False))
    if normalized_mode == "fastest":
        return sorted(ids, key=lambda cid: (predicted_times.get(cid, float("inf")), cid))[:n_clients]
    if normalized_mode == "dijkstra":
        return sorted(ids, key=lambda cid: (dijkstra_costs.get(cid, float("inf")), cid))[:n_clients]

    if score_fn is None:
        raise ValueError("multi_objective exige score_fn")
    selected: list[int] = []
    remaining = set(ids)
    while remaining and len(selected) < n_clients:
        chosen = max(remaining, key=lambda cid: (score_fn(cid, selected), -cid))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def choose_round_deadline(
    times: Sequence[float], mode: str, fixed_seconds: float, percentile: float
) -> float:
    """Compute a positive virtual deadline from selected-client durations."""
    if mode == "fixed":
        if fixed_seconds <= 0:
            raise ValueError("round_deadline deve ser maior que zero")
        return round(float(fixed_seconds), 3)
    if mode != "percentile":
        raise ValueError("deadline_mode deve ser fixed ou percentile")
    if not 0 <= percentile <= 100:
        raise ValueError("deadline_percentile deve estar entre 0 e 100")
    values = np.asarray(times, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("times deve conter duracoes finitas e nao negativas")
    return max(round(float(np.percentile(values, percentile)), 3), 0.001)


def compute_staleness(current_model_version: int, base_model_version: int) -> int:
    tau = current_model_version - base_model_version
    if tau < 0:
        raise ValueError("base_model_version nao pode estar no futuro")
    return tau


def deadline_status(arrival_time: float, round_end_time: float) -> tuple[bool, float]:
    """Return whether an update arrived on time and its temporal lateness."""
    if not np.isfinite(arrival_time) or not np.isfinite(round_end_time):
        raise ValueError("arrival_time e round_end_time devem ser finitos")
    lateness = max(0.0, float(arrival_time - round_end_time))
    return lateness == 0.0, lateness


def estimate_virtual_duration(
    model_size_megabits: float,
    n_samples: int,
    epochs: int,
    comp_time_per_sample: float,
    bandwidth_mbps: float,
    latency_ms: float,
    cpu_cores: float = 32.0,
    compute_factor: float = 1.0,
    jitter_fraction: float = 0.0,
) -> dict[str, float]:
    """Estimate download, local compute, and upload durations in virtual seconds."""
    if model_size_megabits < 0 or n_samples < 0 or epochs < 0 or comp_time_per_sample < 0:
        raise ValueError("tamanho, amostras e epocas devem ser nao negativos")
    if bandwidth_mbps < 0 or latency_ms < 0 or cpu_cores <= 0 or compute_factor <= 0:
        raise ValueError("infraestrutura de cliente invalida")
    latency_seconds = latency_ms / 1000.0
    transfer_seconds = model_size_megabits / max(bandwidth_mbps, 0.1) + latency_seconds
    cpu_speed_factor = max((cpu_cores / 32.0) * compute_factor, 0.2)
    download = transfer_seconds
    compute = (n_samples * epochs * comp_time_per_sample) / cpu_speed_factor
    upload = transfer_seconds
    total = (download + compute + upload) * max(0.05, 1.0 + jitter_fraction)
    return {
        "download_s": float(download),
        "compute_s": float(compute),
        "upload_s": float(upload),
        "total_s": float(total),
    }


def effective_sample_weight(
    n_samples: int,
    staleness_tau: int = 0,
    gamma: float = 0.5,
    late_factor: float = 1.0,
) -> float:
    if n_samples <= 0 or staleness_tau < 0 or gamma < 0:
        raise ValueError("amostras, staleness e gamma invalidos")
    if not 0 < late_factor <= 1:
        raise ValueError("late_factor deve estar no intervalo (0, 1]")
    return float(n_samples) * late_factor / ((1 + staleness_tau) ** gamma)


def weighted_average_updates(
    updates: Sequence[tuple[Sequence[np.ndarray], float]],
) -> list[np.ndarray]:
    """FedAvg arrays with explicit effective sample weights."""
    if not updates:
        raise ValueError("updates nao pode estar vazio")
    total_weight = sum(float(weight) for _, weight in updates)
    if total_weight <= 0 or any(float(weight) <= 0 for _, weight in updates):
        raise ValueError("pesos efetivos devem ser positivos")
    layer_count = len(updates[0][0])
    if any(len(arrays) != layer_count for arrays, _ in updates):
        raise ValueError("atualizacoes possuem quantidades diferentes de camadas")

    averaged: list[np.ndarray] = []
    for layer_index in range(layer_count):
        reference_shape = np.asarray(updates[0][0][layer_index]).shape
        if any(np.asarray(arrays[layer_index]).shape != reference_shape for arrays, _ in updates):
            raise ValueError("camadas possuem formatos diferentes")
        averaged.append(sum(
            np.asarray(arrays[layer_index], dtype=np.float64) * (float(weight) / total_weight)
            for arrays, weight in updates
        ).astype(np.asarray(updates[0][0][layer_index]).dtype, copy=False))
    return averaged


def update_alignment(
    base_parameters: Sequence[np.ndarray],
    client_parameters: Sequence[np.ndarray],
    aggregate_parameters: Sequence[np.ndarray],
) -> float:
    """Cosine alignment of a local delta with the aggregated round delta."""
    if not (len(base_parameters) == len(client_parameters) == len(aggregate_parameters)):
        raise ValueError("quantidades diferentes de camadas")
    dot = 0.0
    client_norm_sq = 0.0
    aggregate_norm_sq = 0.0
    for base, client, aggregate in zip(base_parameters, client_parameters, aggregate_parameters):
        base_array = np.asarray(base, dtype=np.float64)
        client_delta = np.asarray(client, dtype=np.float64) - base_array
        aggregate_delta = np.asarray(aggregate, dtype=np.float64) - base_array
        if client_delta.shape != aggregate_delta.shape:
            raise ValueError("camadas possuem formatos diferentes")
        dot += float(np.vdot(client_delta.ravel(), aggregate_delta.ravel()))
        client_norm_sq += float(np.vdot(client_delta.ravel(), client_delta.ravel()))
        aggregate_norm_sq += float(np.vdot(aggregate_delta.ravel(), aggregate_delta.ravel()))
    denominator = (client_norm_sq * aggregate_norm_sq) ** 0.5
    return 0.0 if denominator == 0 else float(np.clip(dot / denominator, -1.0, 1.0))
