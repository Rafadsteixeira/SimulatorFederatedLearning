"""Minimal Flower-based baseline for the simulator MVP.

This module exposes the clean server/client/strategy trio used by Flower while
keeping the project focused on a small but real FedAvg pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from models.federated.cifar_cnn import CifarCNN, get_weights, set_weights
from models.federated.fl_aggregator import fedavg_aggregate

from .client import BaselineClient, create_client_app
from .server import create_server_app, create_server_config
from .strategy import FedAvgStrategy, build_strategy


@dataclass
class FedAvgBaseline:
    """Container for the minimal Flower baseline used by the simulator MVP."""

    strategy: FedAvgStrategy
    server_config: Any
    server_app: Any
    client_app: Any

    @property
    def model_name(self) -> str:
        return self.strategy.__class__.__name__


def _build_federated_datasets(
    data_dir: str | Path = "./data",
    *,
    num_clients: int = 2,
    seed: int = 42,
) -> tuple[Dataset, Dataset, list[np.ndarray], list[np.ndarray]]:
    """Create lightweight partitions for a real local FL round loop."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    train_idx = np.arange(len(train_dataset))
    test_idx = np.arange(len(test_dataset))
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_dataset, test_dataset, np.array_split(train_idx, num_clients), np.array_split(test_idx, num_clients)


def run_local_fedavg_training(
    *,
    num_rounds: int = 3,
    num_clients: int = 2,
    batch_size: int = 32,
    local_epochs: int = 1,
    learning_rate: float = 0.01,
    data_dir: str | Path = "./data",
    seed: int = 42,
) -> list[dict[str, float]]:
    """Run a minimal local FedAvg loop without the experimental Ray backend."""
    train_dataset, test_dataset, train_partitions, test_partitions = _build_federated_datasets(
        data_dir=data_dir,
        num_clients=num_clients,
        seed=seed,
    )
    global_model = CifarCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    for round_index in range(1, num_rounds + 1):
        client_results: list[tuple[int, list[np.ndarray]]] = []

        for client_id in range(num_clients):
            client_model = CifarCNN()
            set_weights(client_model, get_weights(global_model))
            client_model.to(device)
            optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate)
            subset = Subset(train_dataset, train_partitions[client_id])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

            for _ in range(local_epochs):
                client_model.train()
                for images, labels in loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    logits = client_model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

            client_results.append((len(subset), get_weights(client_model)))

        aggregated = fedavg_aggregate(client_results)
        set_weights(global_model, aggregated)

        global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in DataLoader(
                Subset(test_dataset, test_partitions[0]),
                batch_size=batch_size,
                shuffle=False,
            ):
                images = images.to(device)
                labels = labels.to(device)
                logits = global_model(images)
                pred = logits.argmax(dim=1)
                correct += int((pred == labels).sum().item())
                total += labels.size(0)
        accuracy = correct / max(1, total)
        history.append({"round": float(round_index), "accuracy": float(accuracy)})

    return history


def build_mvp_baseline(
    *,
    num_rounds: int = 5,
    delay_mode: str = "random",
    fixed_delays: Optional[Dict[int, float]] = None,
    min_fit_clients: int = 2,
    min_available_clients: int = 2,
    num_clients: int = 2,
    batch_size: int = 32,
    local_epochs: int = 1,
    learning_rate: float = 0.01,
    data_dir: str | Path = "./data",
    seed: int = 42,
) -> FedAvgBaseline:
    """Create a minimal Flower FedAvg baseline for the simulator."""
    strategy = build_strategy(
        delay_mode=delay_mode,
        fixed_delays=fixed_delays,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
    )
    server_config = create_server_config(num_rounds=num_rounds)
    server_app = create_server_app(strategy=strategy, num_rounds=num_rounds)
    client_app = create_client_app(
        BaselineClient,
        data_dir=data_dir,
        num_clients=num_clients,
        batch_size=batch_size,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        seed=seed,
    )
    return FedAvgBaseline(
        strategy=strategy,
        server_config=server_config,
        server_app=server_app,
        client_app=client_app,
    )
