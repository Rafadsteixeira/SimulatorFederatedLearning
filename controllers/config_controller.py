# -*- coding: utf-8 -*-
"""
controllers/config_controller.py
ConfigController — carrega, valida e distribui configurações de experimento JSON.
"""
import os
import tempfile
import urllib.request
from typing import Callable, Optional

from models.federated.experiment_config import ExperimentConfig


class ConfigController:
    """
    Controla o ciclo de vida da configuração do experimento:
        - Carregamento via arquivo local.
        - Carregamento via URL remota.
        - Notificação de observers quando a config muda.

    Callbacks injetados:
        on_config_loaded(cfg, display_name):  Chamado após carregamento bem-sucedido.
        on_error(title, message):             Chamado em caso de erro.
        on_status(message):                   Atualiza status na View.
    """

    def __init__(
        self,
        on_config_loaded: Callable[[ExperimentConfig, str], None] = None,
        on_error: Callable[[str, str], None] = None,
        on_status: Callable[[str], None] = None,
    ) -> None:
        self._on_config_loaded = on_config_loaded or (lambda c, n: None)
        self._on_error = on_error or (lambda t, m: None)
        self._on_status = on_status or (lambda m: None)

        self.active_config: Optional[ExperimentConfig] = None
        self.config_file_name: str = ""

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def load_from_file(self, path: str) -> None:
        """
        Carrega ExperimentConfig de um arquivo JSON local.

        Args:
            path: Caminho absoluto para o arquivo .json.
        """
        try:
            cfg = ExperimentConfig.from_json(path)
            display_name = os.path.basename(path)
            self._apply(cfg, display_name)
        except Exception as exc:
            self._on_error("Erro ao carregar JSON", str(exc))

    def load_from_url(self, url: str) -> None:
        """
        Faz download do JSON por URL e carrega a configuração.

        Args:
            url: URL pública do arquivo .json.
        """
        if not url or url in ("https://", "http://"):
            return
        try:
            self._on_status("Baixando JSON...")
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode("utf-8")

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

            display_name = url.rstrip("/").split("/")[-1] or "url_config.json"
            cfg = ExperimentConfig.from_json(tmp_path)
            os.unlink(tmp_path)

            self._apply(cfg, display_name)
            self._on_status("Config URL carregada ✓")
        except Exception as exc:
            self._on_error("Erro ao baixar JSON", str(exc))
            self._on_status("Erro no download")

    def reset(self) -> None:
        """Remove a configuração ativa e volta ao padrão."""
        self.active_config = None
        self.config_file_name = ""

    def get_config(self) -> ExperimentConfig:
        """
        Retorna a configuração ativa ou um ExperimentConfig padrão
        se nenhum JSON foi carregado.
        """
        return self.active_config if self.active_config is not None \
            else ExperimentConfig()

    # ------------------------------------------------------------------
    # Método interno
    # ------------------------------------------------------------------

    def _apply(self, cfg: ExperimentConfig, display_name: str) -> None:
        self.active_config = cfg
        self.config_file_name = display_name
        self._on_config_loaded(cfg, display_name)
