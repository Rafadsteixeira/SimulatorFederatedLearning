"""Server-side Flower integration for the simulator.

This follows Flower's current preferred mechanism:
- build a `server_fn` returning `ServerAppComponents`
- instantiate `ServerApp(server_fn=server_fn)`
"""

from __future__ import annotations

from typing import Any, Dict

from flwr.app import Context
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy
from flwr.serverapp import ServerApp


def create_server_config(num_rounds: int = 5, *, config: Dict[str, Any] | None = None) -> ServerConfig:
    """Create the server configuration using Flower's server config API."""
    cfg = config or {}
    return ServerConfig(num_rounds=num_rounds, **cfg)


def create_server_app(
    strategy: Strategy,
    *,
    num_rounds: int = 5,
    config: Dict[str, Any] | None = None,
) -> ServerApp:
    """Build a Flower server app using the modern `server_fn` mechanism."""
    server_config = create_server_config(num_rounds=num_rounds, config=config)

    def server_fn(context: Context) -> ServerAppComponents:
        return ServerAppComponents(strategy=strategy, config=server_config)

    return ServerApp(server_fn=server_fn)


def build_server_app(
    *,
    strategy: Strategy,
    num_rounds: int = 5,
    config: Dict[str, Any] | None = None,
) -> ServerApp:
    """Alias for the minimal server runtime used by the simulator baseline."""
    return create_server_app(strategy=strategy, num_rounds=num_rounds, config=config)
