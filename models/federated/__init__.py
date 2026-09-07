# -*- coding: utf-8 -*-
"""
models/federated/__init__.py
Exportações públicas do módulo de Aprendizado Federado.
"""
from .experiment_config import ExperimentConfig
from .fl_history import FLHistory
from .cifar_cnn import CifarCNN, get_weights, set_weights
from .fl_client import ImageMigrationClient
from .fl_aggregator import fedavg_aggregate, weighted_average
from .dataset_manager import DatasetManager

__all__ = [
    "ExperimentConfig",
    "FLHistory",
    "CifarCNN",
    "get_weights",
    "set_weights",
    "ImageMigrationClient",
    "fedavg_aggregate",
    "weighted_average",
    "DatasetManager",
]
