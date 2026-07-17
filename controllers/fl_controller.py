# -*- coding: utf-8 -*-
"""
controllers/fl_controller.py
FLController — orquestra os rounds de Aprendizado Federado em thread separada.
"""
import random
import threading
from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np

from models.network.base_station import BaseStation
from models.network.network_switch import NetworkSwitch
from models.network.topology import Topology
from models.federated.experiment_config import ExperimentConfig
from models.federated.fl_history import FlHistory
from models.federated.cifar_cnn import CifarCNN, get_weights, set_weights
from models.federated.fl_client import ImageMigrationClient
from models.federated.fl_aggregator import fedavg_aggregate, weighted_average
from models.federated.dataset_manager import DatasetManager


class FLController:
    """
    Controla a execução do treinamento FL:
        - Carrega o dataset (em thread).
        - Particiona dados pelos clientes.
        - Executa os rounds FL com FedAvg.
        - Notifica a View a cada round.

    Callbacks injetados:
        on_status(msg):                Atualiza o label de status na View.
        on_round_complete(entry):      Adiciona entry à lista de rounds na View.
        on_training_done():            Habilita botões ao finalizar.
        on_map_refresh():              Atualiza o mapa de rede.
        on_round_list_clear():         Limpa a listbox de rounds.
        on_error(title, msg):          Exibe mensagem de erro.
    """

    def __init__(
        self,
        topology: Topology,
        central_sw: NetworkSwitch,
        central_bs: BaseStation,
        dijkstra_distances: Dict,
        client_stations: List[BaseStation],
        dataset_manager: DatasetManager,
        results_dir: str = "./fl_results_v15",
        on_status: Callable[[str], None] = None,
        on_round_complete: Callable[[Dict], None] = None,
        on_training_done: Callable[[], None] = None,
        on_map_refresh: Callable[[], None] = None,
        on_round_list_clear: Callable[[], None] = None,
        on_error: Callable[[str, str], None] = None,
        on_dataset_loaded: Callable[[int, int], None] = None,
    ) -> None:
        self.topology = topology
        self.central_sw = central_sw
        self.central_bs = central_bs
        self.dijkstra_distances = dijkstra_distances
        self.client_stations = client_stations
        self.dataset_manager = dataset_manager
        self.results_dir = results_dir

        # Callbacks para a View
        self._on_status = on_status or (lambda m: None)
        self._on_round_complete = on_round_complete or (lambda e: None)
        self._on_training_done = on_training_done or (lambda: None)
        self._on_map_refresh = on_map_refresh or (lambda: None)
        self._on_round_list_clear = on_round_list_clear or (lambda: None)
        self._on_error = on_error or (lambda t, m: None)
        self._on_dataset_loaded = on_dataset_loaded or (lambda n_train, n_img: None)

        # Estado interno
        self.global_model: CifarCNN = CifarCNN()
        self.fl_history: Optional[FlHistory] = None
        self.fl_round_log: List[Dict] = []
        self.fl_clients: List[ImageMigrationClient] = []
        self.is_model_ready: bool = False
        self.training_status: str = "Não treinado"

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def start_load_dataset(self) -> None:
        """Inicia o carregamento do dataset CIFAR-10 em thread separada."""
        threading.Thread(target=self._load_dataset_thread, daemon=True).start()

    def start_training(self, cfg: ExperimentConfig) -> None:
        """
        Inicia o treinamento FL em thread separada.

        Args:
            cfg: Configuração do experimento a usar.
        """
        threading.Thread(
            target=self._training_thread,
            args=(cfg,),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------

    def _load_dataset_thread(self) -> None:
        self.dataset_manager.load()
        self.dataset_manager.load_sample_images(self.client_stations)
        n_train = len(self.dataset_manager.trainset) if self.dataset_manager.trainset else 0
        n_imgs  = len(self.client_stations)
        self._on_status("Dataset pronto ✓")
        self._on_map_refresh()
        self._on_dataset_loaded(n_train, n_imgs)
        self._on_training_done()

    def _training_thread(self, cfg: ExperimentConfig) -> None:
        try:
            self._run_training(cfg)
        except Exception as exc:
            self._on_status(f"Erro: {str(exc)[:50]}")
            self._on_error("Erro no Treinamento FL", str(exc))
            import traceback
            traceback.print_exc()
        finally:
            self._on_training_done()

    # ------------------------------------------------------------------
    # Lógica principal de treinamento
    # ------------------------------------------------------------------

    def _run_training(self, cfg: ExperimentConfig) -> None:
        random.seed(cfg.seed)
        self.fl_round_log = []
        self._on_round_list_clear()

        # --- Dataset ---
        self._on_status("Carregando CIFAR-10...")
        if not self.dataset_manager.is_loaded:
            self.dataset_manager.load()
        if not self.dataset_manager.images_loaded:
            self.dataset_manager.load_sample_images(self.client_stations)

        all_stations: List[BaseStation] = [
            bs for bs in BaseStation.all() if bs != self.central_bs
        ]
        n_total = len(all_stations)

        # --- Particionamento ---
        self._on_status("Particionando dataset...")
        partitions = self.dataset_manager.partition(n_total)

        # --- Criação de clientes ---
        sorted_all = sorted(
            all_stations,
            key=lambda b: self.dijkstra_distances.get(b.network_switch, float("inf")),
        )
        all_clients_map: Dict[int, ImageMigrationClient] = {}
        for i, bs in enumerate(sorted_all):
            client = ImageMigrationClient(
                cid=i,
                base_station=bs,
                partition=partitions[i],
                testset=self.dataset_manager.testset,
                batch_size=cfg.batch_size,
            )
            all_clients_map[bs.id] = client

        n_clients = min(cfg.n_clients, n_total)

        print("\n" + "=" * 64)
        print("  APRENDIZADO FEDERADO — CIFAR-10")
        print(f"  Modo pesos    : {cfg.weight_mode.upper()}")
        print(f"  Clientes/round: {n_clients}  (pool={n_total})")
        print(f"  Rounds        : {cfg.n_rounds}")
        if cfg.high_latency_periodicity > 0:
            print(
                f"  Alta-lat.     : a cada {cfg.high_latency_periodicity} rounds "
                f"({int(cfg.high_latency_fraction * 100)}% HL)"
            )
        print("  PRIVACIDADE   : apenas pesos trafegam")
        print("=" * 64)

        # --- Loop de rounds ---
        history = FlHistory(num_rounds=cfg.n_rounds, num_clients=n_clients)
        global_weights = get_weights(self.global_model)

        for round_num in range(1, cfg.n_rounds + 1):
            self._on_status(f"Round {round_num}/{cfg.n_rounds} [{cfg.weight_mode}]...")

            round_dijkstra = self._build_round_dijkstra(cfg)

            selected_stations, is_hl = self._select_clients(
                round_num, cfg, all_stations, round_dijkstra
            )
            clients = [all_clients_map[bs.id] for bs in selected_stations]
            self.fl_clients = clients
            n_this = len(clients)

            print(
                f"\n  ── Round {round_num}/{cfg.n_rounds} "
                f"[{cfg.weight_mode}]{'  ⚠ HL' if is_hl else ''} ──"
            )

            # Treino local
            fit_results, fit_metrics_ag, clients_detail = [], [], []
            for client in clients:
                new_w, n, fit_m = client.fit(
                    [w.copy() for w in global_weights],
                    config={"local_epochs": cfg.local_epochs},
                )
                fit_results.append((n, new_w))
                fit_metrics_ag.append((n, fit_m))
                dist_ms = round_dijkstra.get(client.base_station.network_switch, 0.0)
                clients_detail.append({
                    "cid":     client.cid,
                    "bs_id":   client.base_station.id,
                    "dist_ms": dist_ms,
                    "loss":    fit_m.get("train_loss", 0.0),
                    "acc":     fit_m.get("train_acc",  0.0),
                })

            # Agregação FedAvg
            global_weights = fedavg_aggregate(fit_results)
            set_weights(self.global_model, global_weights)
            total_samples = sum(n for n, _ in fit_results)
            print(f"  Servidor: FedAvg → {n_this} clientes ({total_samples} amostras)")

            agg_fit = weighted_average(fit_metrics_ag)
            history.add_fit_metrics(round_num, agg_fit)

            # Avaliação
            eval_losses, eval_metrics_ag = [], []
            for client in clients:
                loss, n_val, eval_m = client.evaluate(
                    [w.copy() for w in global_weights], config={}
                )
                eval_losses.append((n_val, loss))
                eval_metrics_ag.append((n_val, eval_m))

            total_eval = sum(n for n, _ in eval_losses)
            agg_loss   = sum(n * l for n, l in eval_losses) / total_eval
            history.losses_distributed.append((round_num, agg_loss))

            agg_eval = weighted_average(eval_metrics_ag)
            history.add_eval_metrics(round_num, agg_eval)

            acc_pct = agg_eval.get("accuracy", 0.0) * 100
            print(f"  Round {round_num} — Loss: {agg_loss:.4f} | Ac.: {acc_pct:.2f}%")

            # Notifica a View
            entry = {
                "round":          round_num,
                "n_clients":      n_this,
                "weight_mode":    cfg.weight_mode,
                "is_hl_round":    is_hl,
                "total_samples":  total_samples,
                "train_loss":     agg_fit.get("train_loss", 0.0),
                "train_acc":      agg_fit.get("train_acc",  0.0),
                "val_loss":       agg_loss,
                "val_acc":        agg_eval.get("accuracy", 0.0),
                "clients_detail": clients_detail,
            }
            self.fl_round_log.append(entry)
            self._on_round_complete(entry)

        # Finalização
        self.fl_history = history
        self.is_model_ready = True
        self.training_status = "Modelo pronto ✓"
        self._on_status("Modelo pronto ✓")
        self._on_map_refresh()

        print("\n" + "=" * 64)
        print("  FL com imagens concluído!")
        print(history.summary())
        print("=" * 64 + "\n")

    # ------------------------------------------------------------------
    # Métodos auxiliares de roteamento e seleção
    # ------------------------------------------------------------------

    def _build_round_dijkstra(self, cfg: ExperimentConfig) -> Dict:
        """
        Recalcula distâncias Dijkstra para o round atual.
        Modo 'random': reatribui delays aleatoriamente.
        Modo 'fixed':  usa latências fixas da configuração.
        """
        if cfg.weight_mode == "random":
            import random as _random
            for u, v in self.topology.edges():
                self.topology.edges[u, v]["delay"]     = round(_random.uniform(1, 5), 2)
                self.topology.edges[u, v]["bandwidth"] = _random.randint(5, 20)
            return dict(
                nx.single_source_dijkstra_path_length(
                    self.topology, self.central_sw, weight="delay"
                )
            )
        else:
            result = {}
            for sw in self.topology.nodes():
                bs = sw.base_station
                if bs and bs.id in cfg.fixed_weights:
                    result[sw] = cfg.fixed_weights[bs.id]
                else:
                    result[sw] = self.dijkstra_distances.get(sw, float("inf"))
            return result

    def _select_clients(
        self,
        round_num: int,
        cfg: ExperimentConfig,
        all_stations: List[BaseStation],
        round_dijkstra: Dict,
    ):
        """
        Seleciona os clientes para o round atual.
        Em rounds de alta-latência, mistura clientes próximos e distantes.

        Returns:
            (lista_de_BSs, is_hl_round)
        """
        n_clients = min(cfg.n_clients, len(all_stations))
        sorted_asc = sorted(
            all_stations,
            key=lambda bs: round_dijkstra.get(bs.network_switch, float("inf")),
        )
        is_hl = (
            cfg.high_latency_periodicity > 0
            and round_num % cfg.high_latency_periodicity == 0
        )
        if is_hl:
            n_high   = max(1, int(n_clients * cfg.high_latency_fraction))
            n_normal = n_clients - n_high
            selected = sorted_asc[:n_normal] + sorted_asc[-n_high:]
            print(
                f"  [Round {round_num}] ⚠ Alta-latência: "
                f"{n_high} clientes HL + {n_normal} normais"
            )
        else:
            selected = sorted_asc[:n_clients]

        return selected, is_hl
