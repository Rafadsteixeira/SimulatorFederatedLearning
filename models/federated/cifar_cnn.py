# -*- coding: utf-8 -*-
"""
models/federated/cifar_cnn.py
CifarCNN — rede convolucional para classificação CIFAR-10.
Funções auxiliares: get_weights / set_weights para interoperabilidade com Flower.
"""
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn

from flwr.common import NDArrays

NUM_CLASSES: int = 10


class CifarCNN(nn.Module):
    """
    CNN de duas camadas convolucionais para CIFAR-10 (32x32 RGB → 10 classes).

    Arquitetura:
        conv_block1: Conv 3→32 + BN + ReLU + Conv 32→32 + BN + ReLU + MaxPool + Dropout(0.25)
        conv_block2: Conv 32→64 + BN + ReLU + Conv 64→64 + BN + ReLU + MaxPool + Dropout(0.25)
        classifier:  Flatten → FC(64*8*8 → 512) + ReLU + Dropout(0.5) → FC(512 → 10)
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block2(self.conv_block1(x)))


# ------------------------------------------------------------------
# Funções auxiliares de serialização de pesos
# ------------------------------------------------------------------

def get_weights(model: nn.Module) -> NDArrays:
    """
    Extrai os pesos do modelo como lista de arrays NumPy.

    Args:
        model: Instância de nn.Module.

    Returns:
        Lista de np.ndarray com os pesos de cada camada.
    """
    return [v.cpu().detach().numpy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: NDArrays) -> None:
    """
    Carrega pesos NumPy no modelo.

    Args:
        model:   Instância de nn.Module a ser atualizada.
        weights: Lista de arrays NumPy na mesma ordem do state_dict.
    """
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)
