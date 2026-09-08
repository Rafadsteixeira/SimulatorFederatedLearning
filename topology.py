"""Hexagonal network topology generator for the Flower federated learning baseline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

HexCoord = Tuple[int, int]

HEX_DIRECTIONS: tuple[HexCoord, ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
)


def hex_distance(a: HexCoord, b: HexCoord) -> int:
    """Return the axial distance between two hex coordinates."""
    q1, r1 = a
    q2, r2 = b
    return max(abs(q1 - q2), abs(r1 - r2), abs((q1 + r1) - (q2 + r2)))


def axial_to_cartesian(cell: HexCoord) -> tuple[float, float]:
    """Convert axial coordinates to 2D Cartesian coordinates for drawing."""
    q, r = cell
    x = math.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return float(x), float(y)


def build_hexagonal_topology(
    radius: int = 2,
    num_base_stations: int = 6,
    grid_size: int = 5,
) -> tuple[nx.Graph, HexCoord, list[HexCoord], dict[HexCoord, tuple[float, float]], dict[HexCoord, list[HexCoord]]]:
    """Create a 5x5 hexagonal lattice with a central server and base stations on the outer ring.

    The default model is a 5x5 hexagonal grid, which is represented as a rhombus in axial
    coordinates but drawn as a hexagonal lattice when converted to Cartesian coordinates.
    """
    graph = nx.Graph()
    offset = grid_size // 2

    for q in range(grid_size):
        for r in range(grid_size):
            node = (q - offset, r - offset)
            graph.add_node(node)
            for dq, dr in HEX_DIRECTIONS:
                neighbor = (node[0] + dq, node[1] + dr)
                if neighbor in graph.nodes:
                    graph.add_edge(node, neighbor, weight=1)

    server = (0, 0)
    ring_nodes = [
        node
        for node in graph.nodes
        if max(abs(node[0]), abs(node[1]), abs(node[0] + node[1])) >= max(1, radius)
    ]
    ring_nodes.sort(key=lambda point: math.atan2(axial_to_cartesian(point)[1], axial_to_cartesian(point)[0]))

    if not ring_nodes:
        raise ValueError("No nodes found on the outer ring for the base stations.")

    base_stations: list[HexCoord] = []
    step = len(ring_nodes) / max(1, num_base_stations)
    for index in range(num_base_stations):
        offset_index = int(round(index * step)) % len(ring_nodes)
        base_stations.append(ring_nodes[offset_index])

    seen: set[HexCoord] = set()
    unique_base_stations: list[HexCoord] = []
    for node in base_stations:
        if node not in seen:
            seen.add(node)
            unique_base_stations.append(node)

    positions = {node: axial_to_cartesian(node) for node in graph.nodes}
    routes: dict[HexCoord, list[HexCoord]] = {}
    for station in unique_base_stations:
        routes[station] = nx.shortest_path(graph, source=server, target=station, weight="weight")

    return graph, server, unique_base_stations, positions, routes


def _station_accuracy_values(
    training_history: Sequence[dict[str, float]] | None,
    num_base_stations: int,
) -> list[float]:
    """Map training metrics to per-station colors when a trained Flower model history is available."""
    if not training_history:
        return [0.0 for _ in range(num_base_stations)]

    values: list[float] = []
    for idx in range(num_base_stations):
        if idx < len(training_history):
            values.append(float(training_history[idx].get("accuracy", 0.0)))
        else:
            values.append(float(training_history[-1].get("accuracy", 0.0)))
    return values


def draw_hexagonal_topology(
    ax,
    radius: int = 2,
    num_base_stations: int = 6,
    grid_size: int = 5,
    training_history: Sequence[dict[str, float]] | None = None,
    model_name: str = "Flower FedAvg",
) -> dict[str, object]:
    """Draw the hexagonal FL topology directly on an existing matplotlib axes."""
    graph, server, base_stations, positions, routes = build_hexagonal_topology(
        radius=radius,
        num_base_stations=num_base_stations,
        grid_size=grid_size,
    )

    station_metrics = _station_accuracy_values(training_history, len(base_stations))
    norm = plt.Normalize(vmin=min(station_metrics) if station_metrics else 0.0, vmax=max(station_metrics) if station_metrics else 1.0)
    cmap = plt.get_cmap("RdYlGn")

    nx.draw_networkx_edges(graph, pos=positions, ax=ax, width=1.2, edge_color="lightgray", alpha=0.9)

    for station in base_stations:
        path = routes[station]
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(
            graph,
            pos=positions,
            ax=ax,
            edgelist=path_edges,
            edge_color="tab:red",
            width=3.0,
            alpha=0.95,
        )

    nx.draw_networkx_nodes(
        graph,
        pos=positions,
        nodelist=[server],
        node_color="tab:blue",
        node_size=400,
        edgecolors="black",
        ax=ax,
    )

    for index, station in enumerate(base_stations, start=1):
        station_value = station_metrics[index - 1]
        node_color = cmap(norm(station_value))
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=[station],
            node_color=[node_color],
            node_size=350,
            edgecolors="black",
            ax=ax,
        )

    labels = {server: "Servidor"}
    for index, station in enumerate(base_stations, start=1):
        label = f"BS-{index}\nacc={station_metrics[index - 1]:.3f}"
        labels[station] = label
    nx.draw_networkx_labels(graph, pos=positions, labels=labels, font_size=9, ax=ax)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"Topologia Hexagonal do modelo FL treinado ({model_name})")

    return {
        "server": server,
        "base_stations": base_stations,
        "routes": routes,
        "training_history": list(training_history) if training_history else [],
        "station_metrics": station_metrics,
    }


def generate_topology_figure(
    radius: int = 2,
    num_base_stations: int = 6,
    output_path: str | Path | None = "topology_hexagonal.png",
    training_history: Sequence[dict[str, float]] | None = None,
    model_name: str = "Flower FedAvg",
    grid_size: int = 5,
) -> dict[str, object]:
    """Create a topology figure with routing plus trained-model metadata overlay.

    When output_path is None, the figure is rendered only in memory and no file is generated.
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    result = draw_hexagonal_topology(
        ax,
        radius=radius,
        num_base_stations=num_base_stations,
        grid_size=grid_size,
        training_history=training_history,
        model_name=model_name,
    )

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        result["output_path"] = str(output_file)
    else:
        result["output_path"] = None

    plt.close(fig)
    return result


def generate_trained_model_topology(
    training_history: Sequence[dict[str, float]],
    output_path: str | Path = "trained_fl_topology.png",
    radius: int = 3,
    num_base_stations: int = 6,
) -> dict[str, object]:
    """Generate a hex topology graph using the output of the Flower FedAvg training loop."""
    return generate_topology_figure(
        radius=radius,
        num_base_stations=num_base_stations,
        output_path=output_path,
        training_history=training_history,
        model_name="FedAvg",
    )


if __name__ == "__main__":
    generate_topology_figure(radius=3, num_base_stations=6, output_path="topology_hexagonal.png")
    print("Arquivo gerado: topology_hexagonal.png")
