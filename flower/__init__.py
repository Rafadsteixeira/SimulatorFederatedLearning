"""Flower-aligned FL orchestration components for the simulator."""

from .baseline import FedAvgBaseline, build_mvp_baseline, run_local_fedavg_training
from .client import BaselineClient, create_client_app, create_client_factory
from .server import build_server_app, create_server_app, create_server_config
from .strategy import DelayAwareFedAvg, FedAvgStrategy, build_strategy

__all__ = [
    "BaselineClient",
    "DelayAwareFedAvg",
    "FedAvgBaseline",
    "FedAvgStrategy",
    "build_mvp_baseline",
    "build_server_app",
    "build_strategy",
    "create_client_app",
    "create_client_factory",
    "create_server_app",
    "create_server_config",
    "run_local_fedavg_training",
]
