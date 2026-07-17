# -*- coding: utf-8 -*-
"""
models/__init__.py
Exportações de alto nível da camada Model.
"""
from .network import (
    NetworkSwitch, BaseStation, EdgeServer, User, Topology, TopologyBuilder
)
from .federated import (
    ExperimentConfig, FlHistory, CifarCNN, get_weights, set_weights,
    ImageMigrationClient, fedavg_aggregate, weighted_average, DatasetManager
)

__all__ = [
    "NetworkSwitch", "BaseStation", "EdgeServer", "User",
    "Topology", "TopologyBuilder",
    "ExperimentConfig", "FlHistory",
    "CifarCNN", "get_weights", "set_weights",
    "ImageMigrationClient",
    "fedavg_aggregate", "weighted_average",
    "DatasetManager",
]
