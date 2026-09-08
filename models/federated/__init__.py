# -*- coding: utf-8 -*-
"""Minimal federated-learning exports for the new Flower baseline."""

from .cifar_cnn import CifarCNN, get_weights, set_weights
from .fl_aggregator import fedavg_aggregate, weighted_average

__all__ = [
    "CifarCNN",
    "get_weights",
    "set_weights",
    "fedavg_aggregate",
    "weighted_average",
]
