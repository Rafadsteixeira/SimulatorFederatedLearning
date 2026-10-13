import json
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "filename",
    ["config_exemplo.json", "config_agno_shadow_teste.json"],
)
def test_json_configs_are_valid_and_have_safe_agno_defaults(filename):
    config = json.loads((PROJECT_DIR / filename).read_text(encoding="utf-8"))

    assert config["agent_policy_mode"] == "shadow"
    assert config["agent_model_id"] == "openai/gpt-oss-120b"
    assert config["agent_timeout_s"] > 0
    assert config["agent_audit_file"].endswith(".jsonl")
    assert config["agent_replay_file"] == ""
    assert config["n_clients"] > 0
    assert config["n_rounds"] > 0
    assert config["late_update_weight_factor"] > 0
    assert config["selection_mode"] in {
        "random", "fastest", "dijkstra", "multi_objective"
    }


def test_secret_file_is_not_part_of_documented_example():
    example = (PROJECT_DIR / ".env.example").read_text(encoding="utf-8")

    assert "GROQ_API_KEY=" in example
    assert "gsk_" not in example
