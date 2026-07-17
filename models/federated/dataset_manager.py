# -*- coding: utf-8 -*-
"""
models/federated/dataset_manager.py
DatasetManager — carregamento, particionamento e thumbnails do CIFAR-10.
"""
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from ..network.base_station import BaseStation

# ---------------------------------------------------------------------------
# Constantes do domínio CIFAR-10
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 10

CLASS_NAMES: List[str] = [
    "avião", "automóvel", "pássaro", "gato", "veado",
    "cachorro", "sapo", "cavalo", "navio", "caminhão",
]

CLASS_COLORS: List[str] = [
    "#4FC3F7", "#EF5350", "#66BB6A", "#FFA726", "#AB47BC",
    "#26C6DA", "#D4E157", "#FF7043", "#42A5F5", "#8D6E63",
]

# Transformações padrão
TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])

TRANSFORM_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])

TRANSFORM_VIS = transforms.ToTensor()


class DatasetManager:
    """
    Gerencia o ciclo de vida do dataset CIFAR-10:
        - Download e carregamento dos splits de treino/teste.
        - Particionamento heterogêneo por cliente (dominância de classe).
        - Geração de thumbnails para o mapa de rede.

    Atributos:
        data_dir (str):         Diretório local de dados.
        trainset:               CIFAR10 com TRANSFORM_TRAIN.
        testset:                CIFAR10 com TRANSFORM_TEST.
        raw_dataset:            CIFAR10 com TRANSFORM_VIS (sem normalização).
        client_images (Dict):   {bs_id: (img_arr, class_name, class_idx)}.
        is_loaded (bool):       True após load() executado com sucesso.
        images_loaded (bool):   True após load_sample_images() executado.
    """

    def __init__(self, data_dir: str = "./data_cifar10") -> None:
        self.data_dir = data_dir
        self.trainset: Optional[CIFAR10] = None
        self.testset:  Optional[CIFAR10] = None
        self.raw_dataset: Optional[CIFAR10] = None
        self.client_images: Dict[int, Tuple[np.ndarray, str, int]] = {}
        self.is_loaded: bool = False
        self.images_loaded: bool = False

    # ------------------------------------------------------------------
    # Carregamento
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Faz download (se necessário) e carrega os datasets CIFAR-10.
        Após execução, is_loaded = True.
        """
        print("  [CIFAR-10] Carregando dataset...")
        self.trainset = CIFAR10(
            root=self.data_dir, train=True, download=True, transform=TRANSFORM_TRAIN
        )
        self.testset = CIFAR10(
            root=self.data_dir, train=False, download=True, transform=TRANSFORM_TEST
        )
        self.raw_dataset = CIFAR10(
            root=self.data_dir, train=True, download=False, transform=TRANSFORM_VIS
        )
        self.is_loaded = True
        print(
            f"  [CIFAR-10] {len(self.trainset)} treino | {len(self.testset)} teste"
        )

    # ------------------------------------------------------------------
    # Particionamento
    # ------------------------------------------------------------------

    def partition(self, n_clients: int, n_dom: int = 120, n_min: int = 30) -> List[Subset]:
        """
        Cria partições heterogêneas (distribuição não-IID) para N clientes.

        Cada cliente tem 2 classes dominantes (N_DOM amostras cada) e
        N_MIN amostras das demais classes.

        Args:
            n_clients: Número de clientes.
            n_dom:     Amostras das classes dominantes por cliente.
            n_min:     Amostras das demais classes por cliente.

        Returns:
            Lista de Subset (um por cliente).
        """
        assert self.is_loaded, "Chame load() antes de partition()."

        class_indices: Dict[int, List[int]] = {c: [] for c in range(NUM_CLASSES)}
        for idx, (_, label) in enumerate(self.trainset):
            class_indices[label].append(idx)

        client_subsets = []
        for cid in range(n_clients):
            dom_a = cid % NUM_CLASSES
            dom_b = (cid + 1) % NUM_CLASSES
            indices: List[int] = []

            for c in [dom_a, dom_b]:
                n = min(n_dom, len(class_indices[c]))
                indices.extend(random.sample(class_indices[c], n))

            for c in range(NUM_CLASSES):
                if c not in [dom_a, dom_b]:
                    n = min(n_min, len(class_indices[c]))
                    indices.extend(random.sample(class_indices[c], n))

            random.shuffle(indices)
            client_subsets.append(Subset(self.trainset, indices))

        return client_subsets

    # ------------------------------------------------------------------
    # Thumbnails
    # ------------------------------------------------------------------

    def load_sample_images(self, client_stations: List[BaseStation]) -> None:
        """
        Carrega um thumbnail CIFAR-10 representativo para cada BS cliente.
        Preenche self.client_images e marca images_loaded = True.

        Args:
            client_stations: Lista de BaseStations (clientes FL).
        """
        assert self.is_loaded, "Chame load() antes de load_sample_images()."

        # Indexa um exemplo por classe
        class_indices_raw: Dict[int, List[int]] = {c: [] for c in range(NUM_CLASSES)}
        for idx, (_, label) in enumerate(self.raw_dataset):
            class_indices_raw[label].append(idx)
            if all(len(v) > 0 for v in class_indices_raw.values()):
                if min(len(v) for v in class_indices_raw.values()) >= 5:
                    break

        for i, bs in enumerate(client_stations):
            dominant_class = i % NUM_CLASSES
            idxs = class_indices_raw[dominant_class]
            chosen_idx = idxs[i % len(idxs)]
            img_tensor, label = self.raw_dataset[chosen_idx]
            img_arr = img_tensor.numpy().transpose(1, 2, 0)
            img_arr = (img_arr * 255).astype(np.uint8)
            self.client_images[bs.id] = (img_arr, CLASS_NAMES[dominant_class], dominant_class)

        self.images_loaded = True
        print(f"  [Imagens] {len(self.client_images)} thumbnails carregados")
