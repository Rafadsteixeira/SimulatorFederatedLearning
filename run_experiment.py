# -*- coding: utf-8 -*-
"""Executa um experimento sem interface grafica.

Uso:
    python run_experiment.py config.json [--seeds 42 43 44]

Cada seed gera uma execucao independente em fl_results_v15/runs/<run_id>,
com config.json, decisions.jsonl, rounds.jsonl e summary.json.
"""

import argparse
import dataclasses
import json
import sys

import fl_engine as engine


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Executa um experimento FL sem GUI")
    parser.add_argument("config", help="arquivo JSON de configuracao")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="executa uma vez por seed (sobrescreve a seed do JSON)")
    args = parser.parse_args(argv)

    base = engine.ExperimentConfig.from_json(args.config)
    seeds = args.seeds or [base.seed]

    engine.load_cifar10()
    results = []
    for seed in seeds:
        cfg = dataclasses.replace(base, seed=seed)
        engine.active_config = cfg
        print(f"\n>>> Execucao seed={seed}")
        engine.run_federated_training_image(load_thumbnails=False)
        history = engine.fl_history
        losses = history.losses_distributed if history else []
        results.append({
            "seed": seed,
            "rodadas": len(losses),
            "loss": losses[-1][1] if losses else None,
            "status": engine.fl_training_status,
        })

    print("\n=== Resumo ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(r["loss"] is not None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
