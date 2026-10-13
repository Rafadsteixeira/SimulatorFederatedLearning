"""Per-run persistence for reproducible federated-learning experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Mapping


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"valor nao serializavel em JSON: {type(value).__name__}")


class RunArtifactRecorder:
    """Write effective configuration, round metrics, and client state events."""

    def __init__(self, root: str | Path, run_id: str, effective_config: Mapping[str, Any]):
        self.run_id = run_id
        self.run_dir = Path(root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._lock = threading.Lock()
        self.started_at_utc = datetime.now(timezone.utc).isoformat()
        self._write_json("config.json", {
            "run_id": run_id,
            "started_at_utc": self.started_at_utc,
            "effective_config": dict(effective_config),
        })

    def _write_json(self, filename: str, payload: Mapping[str, Any]) -> None:
        path = self.run_dir / filename
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )

    def write_effective_config(self, effective_config: Mapping[str, Any]) -> None:
        self._write_json("config.json", {
            "run_id": self.run_id,
            "started_at_utc": self.started_at_utc,
            "effective_config": dict(effective_config),
        })

    def append_jsonl(self, filename: str, payload: Mapping[str, Any]) -> None:
        event = dict(payload)
        event.setdefault("run_id", self.run_id)
        event.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        line = json.dumps(event, ensure_ascii=False, default=_json_default)
        with self._lock, (self.run_dir / filename).open("a", encoding="utf-8") as output:
            output.write(line + "\n")

    def record_client_state(self, round_num: int, cid: int, state: str, **details: Any) -> None:
        self.append_jsonl("client_events.jsonl", {
            "event_type": "client_update_state",
            "round": round_num,
            "cid": cid,
            "state": state,
            **details,
        })

    def record_round(self, round_entry: Mapping[str, Any]) -> None:
        self.append_jsonl("rounds.jsonl", {
            "event_type": "round_metrics",
            **dict(round_entry),
        })

    def finish(self, summary: Mapping[str, Any]) -> None:
        self._write_json("summary.json", {
            "run_id": self.run_id,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            **dict(summary),
        })
