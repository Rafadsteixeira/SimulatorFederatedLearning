# -*- coding: utf-8 -*-
"""
models/federated/fl_client.py
ImageMigrationClient — cliente Flower que treina localmente com CIFAR-10.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import flwr as fl
from flwr.common import NDArrays

from .cifar_cnn import CifarCNN, get_weights, set_weights
from ..network.base_station import BaseStation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageMigrationClient(fl.client.NumPyClient):
    """
    Cliente FL que encapsula o treino e avaliação local com CIFAR-10.

    Args:
        cid (int):               Índice do cliente.
        base_station:            BaseStation associada ao cliente.
        partition (Subset):      Partição de treino CIFAR-10.
        testset:                 Dataset de teste global.
        batch_size (int):        Tamanho do batch.
    """

    def __init__(
        self,
        cid: int,
        base_station: BaseStation,
        partition: Subset,
        testset,
        batch_size: int = 32,
    ) -> None:
        self.cid = cid
        self.base_station = base_station
        self.model = CifarCNN()
        self.n_train = len(partition)

        self.trainloader = DataLoader(
            partition, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.testloader = DataLoader(
            testset, batch_size=64, shuffle=False, num_workers=0
        )

    # ------------------------------------------------------------------
    # Interface Flower (NumPyClient)
    # ------------------------------------------------------------------

    def get_parameters(self, config) -> NDArrays:
        return get_weights(self.model)

    def fit(
        self, parameters: NDArrays, config
    ) -> Tuple[NDArrays, int, Dict]:
        """
        Recebe pesos globais, treina localmente e retorna pesos atualizados.
        """
        set_weights(self.model, parameters)
        epochs = int(config.get("local_epochs", 1)) if isinstance(config, dict) else 1
        loss, acc = self._train(epochs)
        print(
            f"    [C{self.cid:02d} | BS{self.base_station.id}] "
            f"loss={loss:.4f}  acc={acc * 100:.1f}%"
        )
        return (
            get_weights(self.model),
            self.n_train,
            {"train_loss": float(loss), "train_acc": float(acc)},
        )

    def evaluate(
        self, parameters: NDArrays, config
    ) -> Tuple[float, int, Dict]:
        """Avalia os pesos globais no dataset de teste local."""
        set_weights(self.model, parameters)
        loss, acc = self._evaluate()
        return (
            float(loss),
            len(self.testloader.dataset),
            {"accuracy": float(acc), "val_loss": float(loss)},
        )

    # ------------------------------------------------------------------
    # Métodos internos de treino / avaliação
    # ------------------------------------------------------------------

    def _train(self, epochs: int = 1) -> Tuple[float, float]:
        """Treino local com SGD + Cross-Entropy."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        self.model.train()
        self.model.to(DEVICE)
        total_loss, total_corr, total_n = 0.0, 0, 0
        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(images)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)
                total_corr += out.argmax(1).eq(labels).sum().item()
                total_n += images.size(0)
        return total_loss / max(total_n, 1), total_corr / max(total_n, 1)

    def _evaluate(self) -> Tuple[float, float]:
        """Avaliação local sem gradiente."""
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        self.model.to(DEVICE)
        total_loss, total_corr, total_n = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                out = self.model(images)
                total_loss += criterion(out, labels).item() * images.size(0)
                total_corr += out.argmax(1).eq(labels).sum().item()
                total_n += images.size(0)
        return total_loss / max(total_n, 1), total_corr / max(total_n, 1)
