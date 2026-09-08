"""Flower strategy layer used by the project.

This module follows the same abstraction used by Flower strategies:
- a server-side strategy object decides how to aggregate updates;
- the project can optionally plug this strategy into a Flower server app.

The simulator still keeps its UI/controller layer, but the aggregation logic is
implemented using a Flower-style strategy API instead of ad-hoc procedural code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from flwr.server.strategy import FedAvg

from models.federated.fl_aggregator import fedavg_aggregate, weighted_average


class FedAvgStrategy(FedAvg):
    """Minimal FedAvg strategy used as the simulator's Flower baseline."""

    def __init__(
        self,
        *,
        delay_mode: str = "random",
        fixed_delays: Optional[Dict[int, float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.delay_mode = delay_mode
        self.fixed_delays = fixed_delays or {}

    def aggregate_round(self, fit_results: List[Tuple[int, Sequence[Any]]]) -> Optional[List[Any]]:
        if not fit_results:
            return None
        return fedavg_aggregate(fit_results)

    def aggregate_metrics(self, metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        return weighted_average(metrics)


DelayAwareFedAvg = FedAvgStrategy


def build_strategy(
    *,
    delay_mode: str = "random",
    fixed_delays: Optional[Dict[int, float]] = None,
    **kwargs: Any,
) -> FedAvgStrategy:
    """Create the simulator's minimal Flower FedAvg strategy."""
    return FedAvgStrategy(
        delay_mode=delay_mode,
        fixed_delays=fixed_delays,
        **kwargs,
    )
