# -*- coding: utf-8 -*-
"""
controllers/app_controller.py
AppController — orquestra toda a aplicação: cria os models, inicializa a
topologia e conecta controllers e views.
"""
import os
import sys

from models.network.topology_builder import TopologyBuilder
from models.federated.dataset_manager import DatasetManager
from models.federated.experiment_config import ExperimentConfig

from .fl_controller import FLController
from .config_controller import ConfigController


class AppController:
    """
    Ponto central de coordenação da aplicação MVC.

    Responsabilidades:
        1. Construir a topologia de rede (TopologyBuilder).
        2. Instanciar DatasetManager e gerenciar resultados.
        3. Instanciar FLController e ConfigController.
        4. Inicializar a MainWindow (View) e injetar callbacks.
        5. Iniciar o loop principal Tkinter.

    Uso:
        app = AppController()
        app.start()
    """

    RESULTS_DIR = "./fl_results_v15"
    DATA_DIR    = "./data_cifar10"

    def __init__(self) -> None:
        # Limpeza de arquivo temporário gerado pelo Code Runner
        if os.path.exists("tempCodeRunnerFile.py"):
            os.remove("tempCodeRunnerFile.py")

        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)

        # --- Topologia ---
        builder = TopologyBuilder(x_size=5, y_size=5, seed=42)
        self._topo_result = builder.build()

        # --- Dataset ---
        self._dataset_manager = DatasetManager(data_dir=self.DATA_DIR)

        # --- Sub-controllers ---
        self._config_ctrl = ConfigController(
            on_config_loaded=self._on_config_loaded,
            on_error=self._on_error,
            on_status=self._on_status,
        )

        self._fl_ctrl = FLController(
            topology=self._topo_result.topology,
            central_sw=self._topo_result.central_sw,
            central_bs=self._topo_result.central_bs,
            dijkstra_distances=self._topo_result.dijkstra_distances,
            client_stations=self._topo_result.client_stations,
            dataset_manager=self._dataset_manager,
            results_dir=self.RESULTS_DIR,
            on_status=self._on_status,
            on_round_complete=self._on_round_complete,
            on_training_done=self._on_training_done,
            on_map_refresh=self._on_map_refresh,
            on_round_list_clear=self._on_round_list_clear,
            on_error=self._on_error,
            on_dataset_loaded=self._on_dataset_loaded,
        )

        # Referência à View (injetada em start())
        self._window = None

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Constrói a janela principal (View) e inicia o mainloop Tkinter.
        Importado aqui para evitar dependência circular no topo do módulo.
        """
        from views.main_window import MainWindow

        self._window = MainWindow(
            topo_result=self._topo_result,
            on_load_dataset=self._handle_load_dataset,
            on_start_training=self._handle_start_training,
            on_load_config_file=self._config_ctrl.load_from_file,
            on_load_config_url=self._config_ctrl.load_from_url,
            on_show_metrics=self._handle_show_metrics,
        )

        print("  Interface carregada. Use os botões para:")
        print("  1. 'Carregar CIFAR-10' → baixa dataset e exibe thumbnails nos nós")
        print("  2. 'Treinar FL (Dijkstra)' → executa rounds federados")
        print("  3. 'Ver Métricas FL' → gráficos de convergência\n")

        self._window.mainloop()

    # ------------------------------------------------------------------
    # Handlers de eventos da View
    # ------------------------------------------------------------------

    def _handle_load_dataset(self) -> None:
        """Delegado ao FLController para carregar dataset em thread."""
        self._fl_ctrl.start_load_dataset()

    def _handle_start_training(self) -> None:
        """
        Valida pré-condições e delega treinamento ao FLController.
        Exibe aviso se dataset não foi carregado.
        """
        if not self._dataset_manager.is_loaded:
            if self._window:
                self._window.show_warning(
                    "Atenção",
                    "Carregue o dataset CIFAR-10 primeiro\n"
                    "(botão 'Carregar CIFAR-10').",
                )
            return

        cfg = self._config_ctrl.get_config()

        # Avisa se está usando padrões
        if self._config_ctrl.active_config is None:
            defaults = ExperimentConfig()
            if self._window:
                ok = self._window.ask_yes_no(
                    "Sem configuração JSON",
                    f"Nenhum JSON carregado.\n"
                    f"Usar padrões: {defaults.n_clients} clientes, "
                    f"{defaults.n_rounds} rounds, modo '{defaults.weight_mode}'.\n\n"
                    "Deseja continuar? (clique 'Não' para carregar um JSON primeiro)",
                )
                if not ok:
                    return

        if self._window:
            self._window.set_training_buttons_state(enabled=False)

        self._fl_ctrl.start_training(cfg)

    def _handle_show_metrics(self) -> None:
        """Abre a janela de métricas se houver histórico disponível."""
        if self._window:
            history = self._fl_ctrl.fl_history
            if history is None or history.is_empty():
                self._window.show_info(
                    "Métricas FL",
                    "Nenhuma métrica disponível.\n"
                    "Execute o treinamento FL primeiro.",
                )
                return
            self._window.open_metrics_window(
                history=history,
                results_dir=self.RESULTS_DIR,
            )

    # ------------------------------------------------------------------
    # Callbacks dos sub-controllers → View
    # ------------------------------------------------------------------

    def _on_status(self, message: str) -> None:
        if self._window:
            self._window.update_status(message)

    def _on_config_loaded(self, cfg: ExperimentConfig, display_name: str) -> None:
        if self._window:
            self._window.update_config_panel(cfg, display_name)
            self._window.show_info(
                "Config carregada",
                f"Fonte: {display_name}\n" + "\n".join(cfg.to_summary_lines()),
            )

    def _on_round_complete(self, entry: dict) -> None:
        if self._window:
            self._window.append_round(entry)

    def _on_training_done(self) -> None:
        if self._window:
            self._window.set_training_buttons_state(enabled=True)

    def _on_map_refresh(self) -> None:
        if self._window:
            self._window.refresh_map()

    def _on_round_list_clear(self) -> None:
        if self._window:
            self._window.clear_round_list()

    def _on_dataset_loaded(self, n_train: int, n_imgs: int) -> None:
        if self._window:
            self._window.show_info(
                "CIFAR-10 Carregado com Sucesso!",
                f"Dataset carregado: {n_train:,} amostras de treino.\n"
                f"Thumbnails exibidos no mapa: {n_imgs} nos.\n\n"
                "Proximo passo:\n"
                "  Clique em '\u2699 Carregar Config JSON' para\n"
                "  selecionar o arquivo de configuracao do experimento.",
            )

    def _on_error(self, title: str, message: str) -> None:
        if self._window:
            self._window.show_error(title, message)
