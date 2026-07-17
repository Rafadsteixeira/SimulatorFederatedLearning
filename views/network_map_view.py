# -*- coding: utf-8 -*-
"""
views/network_map_view.py
NetworkMapView — canvas matplotlib embutido no Tkinter com o mapa de rede.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional

import networkx as nx
import tkinter as tk
from tkinter import ttk

if TYPE_CHECKING:
    from models.network.topology_builder import TopologyResult
    from models.network.base_station import BaseStation
    from models.federated.dataset_manager import DatasetManager


class NetworkMapView(ttk.Frame):
    """
    Frame que exibe o mapa da rede hexagonal usando matplotlib.

    Responsabilidades:
        - Renderizar nós (switches), arestas e servidor central.
        - Renderizar thumbnails CIFAR-10 sobre os nós dos clientes.
        - Destacar um nó selecionado e o caminho Dijkstra até o servidor.

    Args:
        parent:       Widget pai Tkinter.
        topo_result:  Resultado do TopologyBuilder.
        dataset_mgr:  DatasetManager (para thumbnails).
    """

    def __init__(
        self,
        parent,
        topo_result: "TopologyResult",
        dataset_mgr: "DatasetManager",
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)

        self._topo = topo_result
        self._dataset_mgr = dataset_mgr

        # Pré-calcula estilo do grafo
        self._positions:        Dict = {}
        self._node_labels:      Dict = {}
        self._base_sizes:       List = []
        self._edge_colors_list: List = []
        self._edge_widths_list: List = []
        self._setup_graph_style()

        # Lazy imports — evita erro do Pylance e conflito de backend
        import matplotlib.pyplot as plt                                   # noqa: PLC0415
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: PLC0415
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox      # noqa: PLC0415
        self._plt = plt
        self._OffsetImage = OffsetImage
        self._AnnotationBbox = AnnotationBbox

        # Figura matplotlib
        self._fig, self._ax = plt.subplots(figsize=(11, 7.5))
        self._fig.patch.set_facecolor("#0F0F1A")
        self._ax.set_facecolor("#0F0F1A")

        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def draw(
        self,
        highlight_bs: Optional["BaseStation"] = None,
        highlight_path: Optional[List] = None,
    ) -> None:
        """
        Redesenha o mapa completo.

        Args:
            highlight_bs:   BS a destacar em ciano (opcional).
            highlight_path: Caminho Dijkstra a desenhar em verde (opcional).
        """
        topo = self._topo.topology
        dijkstra = self._topo.dijkstra_distances
        central_sw = self._topo.central_sw
        central_bs = self._topo.central_bs
        client_stations = self._topo.client_stations

        self._ax.cla()
        self._ax.set_facecolor("#0F0F1A")

        max_d = max(dijkstra.values()) if dijkstra else 1.0

        # Cores dos nós
        node_colors = []
        for node in topo.nodes():
            if node == central_sw:
                node_colors.append("#FFD700")
            elif highlight_bs and node == highlight_bs.network_switch:
                node_colors.append("#00FFFF")
            else:
                d     = dijkstra.get(node, max_d)
                ratio = d / max_d
                r = int(30  + ratio * 80)
                g = int(100 - ratio * 60)
                b = int(220 - ratio * 80)
                node_colors.append(f"#{r:02x}{g:02x}{b:02x}")

        nx.draw_networkx_edges(
            topo, ax=self._ax, pos=self._positions,
            edge_color=self._edge_colors_list,
            width=self._edge_widths_list,
            alpha=0.7,
        )
        nx.draw_networkx_nodes(
            topo, ax=self._ax, pos=self._positions,
            node_color=node_colors,
            node_size=self._base_sizes,
            alpha=0.9,
        )

        # Caminho destacado
        if highlight_path and len(highlight_path) >= 2:
            path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
            nx.draw_networkx_edges(
                topo, ax=self._ax, pos=self._positions,
                edgelist=path_edges,
                edge_color="#00FF88",
                width=4.0,
                alpha=1.0,
            )

        # Thumbnails CIFAR-10
        from models.federated.dataset_manager import CLASS_COLORS
        if self._dataset_mgr.images_loaded and self._dataset_mgr.client_images:
            for bs in client_stations:
                if bs.id in self._dataset_mgr.client_images:
                    img_arr, cls_name, cls_idx = self._dataset_mgr.client_images[bs.id]
                    node = bs.network_switch
                    if node in self._positions:
                        zoom = 0.62 if (highlight_bs and node == highlight_bs.network_switch) else 0.48
                        imagebox = self._OffsetImage(img_arr, zoom=zoom)
                        imagebox.image.axes = self._ax
                        border_color = (
                            "#00FFFF"
                            if highlight_bs and node == highlight_bs.network_switch
                            else CLASS_COLORS[cls_idx]
                        )
                        ab = self._AnnotationBbox(
                            imagebox,
                            self._positions[node],
                            frameon=True,
                            bboxprops=dict(
                                edgecolor=border_color,
                                linewidth=2,
                                boxstyle="round,pad=0.1",
                            ),
                            pad=0.05,
                            xycoords="data",
                        )
                        self._ax.add_artist(ab)

        # Marcador do servidor central
        cx, cy = central_sw.coordinates
        self._ax.scatter(cx, cy, c="#FFD700", s=800, zorder=10,
                         edgecolors="white", linewidths=2)
        self._ax.text(
            cx, cy + 0.25,
            f"★ Servidor\nBS{central_bs.id}",
            ha="center", va="bottom",
            fontsize=7, fontweight="bold",
            color="#FFD700",
        )

        # Labels sem thumbnail
        if not self._dataset_mgr.images_loaded:
            for node in topo.nodes():
                if node != central_sw:
                    x, y = self._positions[node]
                    self._ax.text(
                        x, y, f"C{node.id}",
                        ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold",
                    )

        self._ax.set_title(
            f"Rede Hexagonal 5×5 — Digital Twin FL com CIFAR-10\n"
            f"Servidor Central: BS{central_bs.id} | "
            f"Clientes: {len(client_stations)} | "
            f"Rounds FL: {len(self._topo.client_stations)}",
            color="#E0E0E0", fontsize=11, pad=10,
        )
        self._ax.axis("off")
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Método privado: estilo do grafo
    # ------------------------------------------------------------------

    def _setup_graph_style(self) -> None:
        topo    = self._topo.topology
        dijkstra = self._topo.dijkstra_distances
        central_sw = self._topo.central_sw
        max_d = max(dijkstra.values()) if dijkstra else 1.0

        for node in topo.nodes():
            self._positions[node] = node.coordinates
            if node == central_sw:
                self._node_labels[node] = f"★\nS{node.id}"
                self._base_sizes.append(1100)
            else:
                self._node_labels[node] = f"C{node.id}"
                self._base_sizes.append(600)

        for (u, v) in topo.edges():
            delay     = topo.edges[u, v]["delay"]
            bandwidth = topo.edges[u, v]["bandwidth"]
            cost = delay / bandwidth
            if cost < 0.1:
                self._edge_colors_list.append("#32CD32")
                self._edge_widths_list.append(1.8)
            elif cost < 0.8:
                self._edge_colors_list.append("#FFA500")
                self._edge_widths_list.append(2.2)
            else:
                self._edge_colors_list.append("#FF4444")
                self._edge_widths_list.append(2.8)
