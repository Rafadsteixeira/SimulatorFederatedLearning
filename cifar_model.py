# -*- coding: utf-8 -*-
"""CNN do CIFAR-10 e funcoes de treino/avaliacao local."""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from flwr.common import NDArrays
from torch.utils.data import DataLoader
from typing import Tuple
from experiment_config import NUM_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Dispositivo de computacao: {DEVICE}\n")


# ==============================================================================
#  MODELO CNN
# ==============================================================================

class CifarCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block2(self.conv_block1(x)))


def get_weights(model: nn.Module) -> NDArrays:
    return [v.cpu().detach().numpy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: NDArrays) -> None:
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


def train_image_local(model: nn.Module, loader: DataLoader,
                      epochs: int = 1) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    model.train()
    model.to(DEVICE)
    total_loss, total_corr, total_n = 0.0, 0, 0
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_corr += out.argmax(1).eq(labels).sum().item()
            total_n    += images.size(0)
    return total_loss / max(total_n, 1), total_corr / max(total_n, 1)


def eval_image_local(model: nn.Module,
                     loader: DataLoader) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to(DEVICE)
    total_loss, total_corr, total_n = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out  = model(images)
            total_loss += criterion(out, labels).item() * images.size(0)
            total_corr += out.argmax(1).eq(labels).sum().item()
            total_n    += images.size(0)
    return total_loss / max(total_n, 1), total_corr / max(total_n, 1)
