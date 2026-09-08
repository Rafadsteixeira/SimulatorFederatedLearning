# -*- coding: utf-8 -*-
"""Minimal model package for the Flower-based baseline."""

from .federated import CifarCNN, fedavg_aggregate, get_weights, set_weights, weighted_average

__all__ = [
    "CifarCNN",
    "fedavg_aggregate",
    "get_weights",
    "set_weights",
    "weighted_average",
]
