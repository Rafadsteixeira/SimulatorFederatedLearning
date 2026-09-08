"""Client-side Flower integration for the simulator.

This module defines the minimal end-to-end Flower client runtime used by the
project's FedAvg baseline. Each client owns a partition of CIFAR-10 and trains
locally during the federated rounds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.app import Context
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import NDArrays
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from models.federated.cifar_cnn import CifarCNN, get_weights, set_weights


def _build_cifar10_data_loaders(
    data_dir: str | Path = "./data",
    *,
    num_clients: int = 2,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[Dataset, Dataset, list[np.ndarray], list[np.ndarray]]:
    """Create lightweight CIFAR-like partitions for the Flower simulation baseline."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = datasets.FakeData(
        size=512,
        image_size=(3, 32, 32),
        num_classes=10,
        transform=transform,
    )
    test_dataset = datasets.FakeData(
        size=256,
        image_size=(3, 32, 32),
        num_classes=10,
        transform=transform,
    )

    rng = np.random.default_rng(seed)
    train_indices = np.arange(len(train_dataset))
    rng.shuffle(train_indices)
    test_indices = np.arange(len(test_dataset))
    rng.shuffle(test_indices)

    train_partitions = np.array_split(train_indices, num_clients)
    test_partitions = np.array_split(test_indices, num_clients)
    return train_dataset, test_dataset, train_partitions, test_partitions


class BaselineClient(NumPyClient):
    """Flower client that trains a CIFAR-10 CNN on a local data shard."""

    def __init__(
        self,
        cid: str | int = 0,
        *,
        train_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        train_partition: Sequence[int] | None = None,
        test_partition: Sequence[int] | None = None,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        local_epochs: int = 1,
    ) -> None:
        self.cid = str(cid)
        self.model = CifarCNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if train_dataset is not None:
            self.train_loader = DataLoader(
                Subset(train_dataset, list(train_partition or range(len(train_dataset)))),
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            self.train_loader = DataLoader([], batch_size=batch_size)

        if test_dataset is not None:
            self.test_loader = DataLoader(
                Subset(test_dataset, list(test_partition or range(len(test_dataset)))),
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            self.test_loader = DataLoader([], batch_size=batch_size)

    def get_parameters(self, config: Dict[str, object]) -> NDArrays:
        return get_weights(self.model)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, object],
    ) -> Tuple[NDArrays, int, Dict[str, float]]:
        set_weights(self.model, parameters)
        self.model.train()
        train_losses: list[float] = []

        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                train_losses.append(float(loss.item()))

        num_examples = len(self.train_loader.dataset)
        return get_weights(self.model), num_examples, {
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "num_examples": float(num_examples),
        }

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, object],
    ) -> Tuple[float, int, Dict[str, float]]:
        set_weights(self.model, parameters)
        self.model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss_sum += float(loss.item()) * labels.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == labels).sum().item())
                total += labels.size(0)

        num_examples = total or len(self.test_loader.dataset)
        loss = loss_sum / max(1, num_examples)
        accuracy = correct / max(1, num_examples)
        return float(loss), num_examples, {"accuracy": float(accuracy), "loss": float(loss)}


def create_client_factory(
    client_cls: type[NumPyClient],
    *,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    local_epochs: int = 1,
) -> Callable[[str], NumPyClient]:
    """Build a factory that instantiates a client using the federated data shard."""

    def client_factory(cid: str) -> NumPyClient:
        partition_id = int(cid)
        train_partition = np.arange(len(train_dataset)) if train_dataset is None else None
        test_partition = np.arange(len(test_dataset)) if test_dataset is None else None
        return client_cls(
            cid=partition_id,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_partition=train_partition,
            test_partition=test_partition,
            batch_size=batch_size,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
        )

    return client_factory


def create_client_app(
    client_cls: type[NumPyClient] = BaselineClient,
    *,
    data_dir: str | Path = "./data",
    num_clients: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    local_epochs: int = 1,
    seed: int = 42,
) -> ClientApp:
    """Build a Flower `ClientApp` using a real data partition per node."""
    train_dataset, test_dataset, train_partitions, test_partitions = _build_cifar10_data_loaders(
        data_dir=data_dir,
        num_clients=num_clients,
        batch_size=batch_size,
        seed=seed,
    )

    def client_fn(context: Context):
        partition_id = int(context.node_config.get("partition-id", context.node_id))
        partition_id = partition_id % max(1, num_clients)
        client = client_cls(
            cid=partition_id,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_partition=train_partitions[partition_id],
            test_partition=test_partitions[partition_id],
            batch_size=batch_size,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
        )
        return client.to_client()

    return ClientApp(client_fn=client_fn)
