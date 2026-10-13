import numpy as np
import pytest

from experiment_core import (
    choose_client_ids,
    choose_round_deadline,
    compute_staleness,
    deadline_status,
    estimate_virtual_duration,
    effective_sample_weight,
    label_distribution_metrics,
    partition_indices,
    update_alignment,
    weighted_average_updates,
)


def test_partition_is_disjoint_complete_non_iid_and_reproducible():
    labels = [0] * 60 + [1] * 60 + [2] * 60
    first = partition_indices(labels, n_clients=4, seed=91, dirichlet_alpha=0.25)
    second = partition_indices(labels, n_clients=4, seed=91, dirichlet_alpha=0.25)

    flattened = [index for client in first for index in client]
    assert first == second
    assert len(flattened) == len(labels)
    assert sorted(flattened) == list(range(len(labels)))
    assert all(client for client in first)
    assert len({tuple(np.bincount(np.asarray(labels)[client], minlength=3)) for client in first}) > 1


def test_partition_rejects_invalid_client_count():
    with pytest.raises(ValueError, match="n_clients"):
        partition_indices([0, 1], n_clients=0, seed=1)


def test_information_metrics_are_computable_and_bounded():
    metrics = label_distribution_metrics(
        local_counts=[8, 2, 0],
        global_counts=[80, 10, 10],
        rare_class_threshold=0.15,
    )
    assert metrics["entropy"] > 0
    assert metrics["rare_coverage"] == pytest.approx(0.5)
    assert 0 <= metrics["js_divergence"] <= np.log(2)


def test_selection_baselines_choose_expected_candidates():
    times = {0: 5.0, 1: 1.0, 2: 3.0}
    distances = {0: 1.0, 1: 4.0, 2: 2.0}
    common = {
        "candidate_ids": times,
        "n_clients": 2,
        "predicted_times": times,
        "dijkstra_costs": distances,
        "seed": 42,
        "round_num": 1,
    }
    assert choose_client_ids("fastest", **common) == [1, 2]
    assert choose_client_ids("dijkstra", **common) == [0, 2]
    assert choose_client_ids("latency", **common) == [0, 2]
    assert choose_client_ids("random", **common) == choose_client_ids("random", **common)
    assert len(choose_client_ids("random", **common)) == 2
    assert choose_client_ids(
        "multi_objective", **common, score_fn=lambda cid, selected: float(cid)
    ) == [2, 1]


def test_deadline_and_staleness_are_separate_and_weight_is_bounded():
    assert choose_round_deadline([1.0, 2.0, 3.0], "percentile", 9.0, 50) == 2.0
    assert choose_round_deadline([], "fixed", 4.5, 80) == 4.5
    assert deadline_status(9.0, 10.0) == (True, 0.0)
    assert deadline_status(12.5, 10.0) == (False, 2.5)
    assert compute_staleness(5, 2) == 3
    assert effective_sample_weight(100, 3, gamma=0.5) == pytest.approx(50.0)
    assert effective_sample_weight(100, 3, gamma=0.5, late_factor=0.5) == pytest.approx(25.0)


def test_virtual_duration_models_transfer_compute_and_jitter():
    duration = estimate_virtual_duration(
        model_size_megabits=10,
        n_samples=100,
        epochs=2,
        comp_time_per_sample=0.01,
        bandwidth_mbps=5,
        latency_ms=100,
        cpu_cores=16,
        compute_factor=1,
        jitter_fraction=0,
    )
    assert duration["download_s"] == pytest.approx(2.1)
    assert duration["upload_s"] == pytest.approx(2.1)
    assert duration["compute_s"] == pytest.approx(4.0)
    assert duration["total_s"] == pytest.approx(8.2)


def test_weighted_fedavg_and_update_alignment():
    base = [np.array([0.0, 0.0])]
    client_a = [np.array([2.0, 0.0])]
    client_b = [np.array([0.0, 4.0])]
    aggregate = weighted_average_updates([(client_a, 1.0), (client_b, 3.0)])
    np.testing.assert_allclose(aggregate[0], np.array([0.5, 3.0]))
    alignment = update_alignment(base, client_a, [np.array([1.0, 0.0])])
    assert alignment == pytest.approx(1.0)
    assert update_alignment(base, client_a, [np.array([0.0, 1.0])]) == pytest.approx(0.0)


def test_weighted_fedavg_rejects_zero_weights():
    with pytest.raises(ValueError, match="positivos"):
        weighted_average_updates([([np.array([1.0])], 0.0)])
