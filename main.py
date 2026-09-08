# -*- coding: utf-8 -*-
"""Entry point for the Flower-based FL baseline with a small real training loop."""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flower import build_mvp_baseline, run_local_fedavg_training


def run_cli() -> None:
    """Build the baseline and run a local FedAvg training loop with per-round metrics."""
    baseline = build_mvp_baseline(
        num_rounds=2,
        delay_mode="random",
        min_fit_clients=2,
        min_available_clients=2,
        num_clients=2,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.01,
    )

    print("SimulatorFederatedLearning - Flower baseline")
    print(f"Strategy: {type(baseline.strategy).__name__}")
    print(f"ServerApp: {type(baseline.server_app).__name__}")
    print(f"ClientApp: {type(baseline.client_app).__name__}")
    print(f"ServerConfig rounds: {baseline.server_config.num_rounds}")
    print("Running a small local FedAvg loop with round metrics...")

    metrics = run_local_fedavg_training(
        num_rounds=2,
        num_clients=2,
        batch_size=32,
        local_epochs=1,
        learning_rate=0.01,
    )
    for row in metrics:
        print(f"round={int(row['round'])} accuracy={row['accuracy']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SimulatorFederatedLearning baseline")
    parser.add_argument("--ui", action="store_true", help="open the lightweight Tkinter dashboard")
    parser.add_argument("--graph", action="store_true", help="generate the hexagonal topology graph and save it as an image")
    parser.add_argument("--graph-trained", action="store_true", help="generate the topology on top of a trained Flower FedAvg model")
    parser.add_argument("--graph-output", default="topology_hexagonal.png", help="output path for the generated topology graph")
    args = parser.parse_args()

    if args.graph_trained:
        from topology import generate_trained_model_topology

        training_history = run_local_fedavg_training(
            num_rounds=3,
            num_clients=6,
            batch_size=32,
            local_epochs=1,
            learning_rate=0.01,
        )
        result = generate_trained_model_topology(
            training_history=training_history,
            output_path=args.graph_output,
            radius=3,
            num_base_stations=6,
        )
        print(f"Trained topology saved to: {result['output_path']}")
        print(f"Final accuracy: {training_history[-1]['accuracy']:.4f}")
        return

    if args.graph:
        from topology import generate_topology_figure

        result = generate_topology_figure(output_path=args.graph_output)
        print(f"Topology saved to: {result['output_path']}")
        for index, station in enumerate(result["base_stations"], start=1):
            route = result["routes"][station]
            print(f"BS-{index}: {route}")
        return

    if args.ui:
        try:
            from ui.dashboard import run_dashboard

            run_dashboard()
            return
        except Exception as exc:  # pragma: no cover - platform-dependent UI gating
            print(f"Unable to open the dashboard: {exc}")
            print("Falling back to the CLI training loop.")

    run_cli()


if __name__ == "__main__":
    main()


