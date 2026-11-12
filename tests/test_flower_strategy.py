"""Testa o envio de station_idx pela Strategy sem Ray, sem GUI e sem chamar o Flower.

Os node_ids do Flower sao inteiros aleatorios; o simulador nao pode derivar a
estacao do node_id. Estes testes usam clientes falsos com cids aleatorios.
"""
import random
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sim():
    if not (PROJECT_DIR / "data_cifar10" / "cifar-10-batches-py").exists():
        pytest.skip("CIFAR-10 local ausente")
    sys.path.insert(0, str(PROJECT_DIR))
    import fl_engine

    fl_engine.load_cifar10()
    return fl_engine


class FakeProxy:
    def __init__(self, cid):
        self.cid = str(cid)


class FakeClientManager:
    def __init__(self, cids):
        self._proxies = {str(cid): FakeProxy(cid) for cid in cids}

    def wait_for(self, num_clients, timeout=86400):
        return True

    def all(self):
        return self._proxies


def _strategy(sim, n_clients):
    cfg = sim.ExperimentConfig(
        n_clients=n_clients, n_rounds=1, agent_policy_mode="baseline"
    )
    sim.active_config = cfg
    sim._agno_policy = None
    sim._prepare_fl_state(cfg)
    strategy = sim.TopologyAwareFedAvg(
        cfg=cfg, sorted_stations=sim._sorted_stations, fl_history=sim.FLHistory()
    )
    params = sim.ndarrays_to_parameters(sim.get_weights(sim.CifarCNN()))
    return strategy, params


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_configure_fit_assigns_distinct_proxies_with_random_node_ids(sim, seed):
    strategy, params = _strategy(sim, n_clients=2)
    rng = random.Random(seed)
    cids = {rng.getrandbits(63) for _ in range(len(sim._sorted_stations))}
    assert len(cids) == len(sim._sorted_stations)

    instructions = strategy.configure_fit(1, params, FakeClientManager(cids))

    assert len(instructions) == 2
    assert len({proxy.cid for proxy, _ in instructions}) == 2
    invited = strategy._invited_by_round[1]
    assert {fit_ins.config["station_idx"] for _, fit_ins in instructions} == invited
    assert all(0 <= idx < len(sim._sorted_stations) for idx in invited)


def test_configure_fit_fails_clearly_without_enough_clients(sim):
    strategy, params = _strategy(sim, n_clients=2)
    manager = FakeClientManager([123])

    with pytest.raises(RuntimeError, match="Flower registrou"):
        strategy.configure_fit(1, params, manager)


def test_station_client_builds_the_requested_station(sim):
    _strategy(sim, n_clients=2)
    client = sim.StationClient._build(5)

    assert client.cid == 5
    assert client.base_station is sim._sorted_stations[5]
    assert client.n_train == len(sim._partitions[5])
