"""Testa o envio de station_idx pela Strategy sem Ray, sem GUI e sem chamar o Flower.

Os node_ids do Flower sao inteiros aleatorios; o simulador nao pode derivar a
estacao do node_id. Estes testes usam clientes falsos com cids aleatorios.
"""
import random
import sys
from pathlib import Path

import pytest
from flwr.common import Code, Status

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


# ---------------------------------------------------------------- aggregate_fit

def _constant_params(sim, value):
    arrays = [value * (sim.np.ones_like(w)) for w in sim.get_weights(sim.CifarCNN())]
    return sim.ndarrays_to_parameters(arrays)


def _fit_res(sim, station_idx, value, n_examples=100):
    return sim.FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=_constant_params(sim, value),
        num_examples=n_examples,
        metrics={
            "train_loss": 1.0, "train_acc": 0.5, "update_norm": 1.0,
            "entropy": 2.0, "station_idx": station_idx,
        },
    )


def _round(sim, late_policy="discard", late_positions=()):
    """Roda configure_fit e devolve (strategy, results) com cids de proxy aleatorios."""
    cfg = sim.ExperimentConfig(
        n_clients=2, n_rounds=1, agent_policy_mode="baseline",
        late_update_policy=late_policy, late_update_weight_factor=0.5,
    )
    sim.active_config = cfg
    sim._agno_policy = None
    sim._prepare_fl_state(cfg)
    strategy = sim.TopologyAwareFedAvg(
        cfg=cfg, sorted_stations=sim._sorted_stations, fl_history=sim.FLHistory()
    )
    params = sim.ndarrays_to_parameters(sim.get_weights(sim.CifarCNN()))
    rng = random.Random(7)
    cids = {rng.getrandbits(63) for _ in range(len(sim._sorted_stations))}
    instructions = strategy.configure_fit(1, params, FakeClientManager(cids))
    stations = sorted(strategy._invited_by_round[1])
    deadline_end = strategy._round_start_time + strategy._round_deadline
    for position, (_, fit_ins) in enumerate(instructions):
        # position indexa a ordem das instrucoes (e dos resultados nos testes).
        # O percentil do deadline atrasa naturalmente o mais lento; aqui o
        # prazo de cada cliente e fixado para isolar o que cada teste verifica.
        timing = strategy._client_timings[fit_ins.config["station_idx"]]
        if position in late_positions:
            timing.on_time = False
            timing.arrival_time = deadline_end + 5.0
        else:
            timing.on_time = True
            timing.arrival_time = deadline_end - 1.0
    return strategy, instructions, stations


def _mean_value(sim, params):
    return float(sim.np.mean(sim.parameters_to_ndarrays(params)[0]))


def test_aggregate_fit_uses_station_idx_and_averages_on_time_updates(sim):
    strategy, instructions, stations = _round(sim)
    results = [
        (proxy, _fit_res(sim, fit_ins.config["station_idx"], 1.0 if i == 0 else 3.0))
        for i, (proxy, fit_ins) in enumerate(instructions)
    ]

    aggregated, _ = strategy.aggregate_fit(1, results, [])

    # So passa se ambas as atualizacoes foram associadas as estacoes corretas.
    detail_cids = {d["cid"] for d in strategy._last_fit_metrics["clients_detail"]}
    assert detail_cids == set(stations)
    assert strategy._model_version == 1
    assert _mean_value(sim, aggregated) == pytest.approx(2.0)
    assert strategy.fl_history.losses_distributed[0][0] == 1


def test_aggregate_fit_records_client_without_result_as_failed(sim):
    strategy, instructions, stations = _round(sim)
    proxy, fit_ins = instructions[0]
    results = [(proxy, _fit_res(sim, fit_ins.config["station_idx"], 5.0))]

    aggregated, _ = strategy.aggregate_fit(1, results, [])

    status = {d["cid"]: d["status"] for d in strategy._last_fit_metrics["clients_detail"]}
    missing = [cid for cid in stations if cid != fit_ins.config["station_idx"]][0]
    assert status[missing] == "failed"
    assert _mean_value(sim, aggregated) == pytest.approx(5.0)


def test_late_update_is_excluded_with_discard_policy(sim):
    strategy, instructions, _ = _round(sim, late_policy="discard", late_positions=[1])
    results = [
        (proxy, _fit_res(sim, fit_ins.config["station_idx"], 1.0 if i == 0 else 3.0))
        for i, (proxy, fit_ins) in enumerate(instructions)
    ]

    aggregated, _ = strategy.aggregate_fit(1, results, [])

    assert _mean_value(sim, aggregated) == pytest.approx(1.0)
    assert strategy._last_fit_metrics["clients_late"] == 1


def test_late_update_gets_reduced_weight_with_reduced_weight_policy(sim):
    strategy, instructions, _ = _round(sim, late_policy="reduced_weight", late_positions=[1])
    results = [
        (proxy, _fit_res(sim, fit_ins.config["station_idx"], 1.0 if i == 0 else 3.0))
        for i, (proxy, fit_ins) in enumerate(instructions)
    ]

    aggregated, _ = strategy.aggregate_fit(1, results, [])

    # Com peso cheio daria 2.0; com peso reduzido fica entre 1.0 e 2.0.
    assert 1.0 < _mean_value(sim, aggregated) < 2.0
