import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from run_artifacts import RunArtifactRecorder


PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifact_dir():
    directory = PROJECT_DIR / ".pytest_cache" / f"run-artifact-test-{uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    yield directory
    for artifact in sorted(directory.rglob("*"), key=lambda path: len(path.parts), reverse=True):
        if artifact.is_file():
            artifact.unlink()
        elif artifact.is_dir():
            artifact.rmdir()
    directory.rmdir()


def test_run_persists_effective_config_client_events_round_metrics_and_summary(artifact_dir):
    recorder = RunArtifactRecorder(artifact_dir, "run-abc", {"seed": 7})
    recorder.record_client_state(1, 2, "invited", base_model_version=0)
    recorder.record_client_state(1, 2, "accepted_reduced", effective_weight=12.5)
    recorder.record_round({"round": 1, "val_acc": 0.5, "weights_preview": np.array([1, 2])})
    recorder.finish({"status": "completed"})

    run_dir = artifact_dir / "run-abc"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    client_events = [
        json.loads(line)
        for line in (run_dir / "client_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    round_event = json.loads((run_dir / "rounds.jsonl").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert config["run_id"] == "run-abc"
    assert config["effective_config"]["seed"] == 7
    assert all(event["run_id"] == "run-abc" for event in client_events)
    assert [event["state"] for event in client_events] == ["invited", "accepted_reduced"]
    assert round_event["val_acc"] == 0.5
    assert round_event["weights_preview"] == [1, 2]
    assert summary["status"] == "completed"
