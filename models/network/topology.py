# -*- coding: utf-8 -*-
"""
models/network/topology.py
Entidade Topology — grafo da rede baseado em networkx.Graph.
"""
import networkx as nx


class Topology(nx.Graph):
    """
    Grafo de rede que representa a topologia hexagonal.

    Herda de networkx.Graph e adiciona:
        - Singleton de instância (_instance).
        - Cálculo de atraso acumulado ao longo de um caminho.

    Cada aresta possui atributos:
        delay (float):     Atraso em milissegundos.
        bandwidth (int):   Largura de banda em Mbps.
    """

    _instance: "Topology | None" = None

    def __init__(self) -> None:
        super().__init__()
        Topology._instance = self

    # ------------------------------------------------------------------
    # Métodos de domínio
    # ------------------------------------------------------------------

    def calculate_path_delay(self, path: list) -> float:
        """
        Soma os delays das arestas em um caminho ordenado de nós.

        Args:
            path: Lista de nós (NetworkSwitch) formando o caminho.

        Returns:
            Atraso total em ms. Retorna 0.0 se o caminho for inválido.
        """
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            edge = self.edges.get((path[i], path[i + 1]))
            if edge:
                total += edge.get("delay", 0.0)
        return total

    # ------------------------------------------------------------------
    # Métodos de classe (singleton)
    # ------------------------------------------------------------------

    @classmethod
    def first(cls) -> "Topology | None":
        """Retorna a instância singleton atual."""
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destrói o singleton."""
        cls._instance = None
