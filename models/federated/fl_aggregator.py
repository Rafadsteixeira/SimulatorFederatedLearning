# -*- coding: utf-8 -*-
"""
models/federated/fl_aggregator.py
Funções de agregação federada: FedAvg e média ponderada de métricas.
"""
from typing import Dict, List, Tuple

import numpy as np
from flwr.common import NDArrays


def fedavg_aggregate(fit_results: List[Tuple[int, NDArrays]]) -> NDArrays:
    """
    Agrega pesos de múltiplos clientes via FedAvg (média ponderada por n_amostras).

    Args:
        fit_results: Lista de (n_amostras, pesos_numpy) por cliente.

    Returns:
        Lista de arrays NumPy com os pesos globais agregados,
        preservando o dtype original.
    """
    total = sum(n for n, _ in fit_results)
    n_layers = len(fit_results[0][1])
    orig_dtypes = [fit_results[0][1][layer].dtype for layer in range(n_layers)]

    aggregated = [
        np.zeros_like(fit_results[0][1][layer], dtype=np.float64)
        for layer in range(n_layers)
    ]

    for n_samples, weights in fit_results:
        factor = n_samples / total
        for layer in range(n_layers):
            aggregated[layer] += weights[layer].astype(np.float64) * factor

    return [aggregated[layer].astype(orig_dtypes[layer]) for layer in range(n_layers)]


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """
    Calcula a média ponderada de um dicionário de métricas.

    Args:
        metrics: Lista de (n_amostras, dict_de_metricas) por cliente.

    Returns:
        Dicionário com a média ponderada de cada métrica.
        Retorna {} se a lista estiver vazia.
    """
    total = sum(n for n, _ in metrics)
    if total == 0 or not metrics:
        return {}

    result: Dict = {}
    for key in metrics[0][1].keys():
        result[key] = sum(m[key] * n for n, m in metrics) / total

    return result
