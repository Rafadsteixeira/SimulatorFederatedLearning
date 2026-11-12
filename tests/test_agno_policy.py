import json
from pathlib import Path
from uuid import uuid4

from agno_policy import AgnoRoundPolicy, ClientAction, RoundDecision
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifact_dir():
    directory = PROJECT_DIR / ".pytest_cache" / f"policy-test-{uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    yield directory
    for artifact in directory.iterdir():
        artifact.unlink()
    directory.rmdir()


class FakeResponse:
    def __init__(self, content, run_id="fake-run"):
        self.content = content
        self.run_id = run_id


class FakeAgent:
    def __init__(self, content):
        self.content = content

    def run(self, _prompt):
        return FakeResponse(self.content)


def valid_selection():
    return RoundDecision(
        phase="pre_round",
        actions=[
            ClientAction(
                cid=0,
                action="select",
                reason_codes=["FAST"],
                rationale="Menor tempo previsto.",
            ),
            ClientAction(
                cid=1,
                action="skip",
                reason_codes=["COST"],
                rationale="Maior custo temporal.",
            ),
        ],
        policy_summary="Prioriza o cliente mais rapido.",
    )


def run_decision(artifact_dir, mode, response):
    audit_path = artifact_dir / "decisions.jsonl"
    policy = AgnoRoundPolicy(
        mode=mode,
        model_id="fake-model",
        audit_path=str(audit_path),
        timeout_s=1,
        seed=42,
    )
    policy._agent = FakeAgent(response)
    result = policy.decide(
        phase="pre_round",
        context={"run_id": "test-run", "round": 1},
        allowed_actions={"select", "skip"},
        fallback={0: "skip", 1: "select"},
        required_counts={"select": 1},
    )
    event = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[-1])
    return result, event


def test_shadow_audits_proposal_but_applies_baseline(artifact_dir):
    result, event = run_decision(artifact_dir, "shadow", valid_selection())

    assert result.actions == {0: "skip", 1: "select"}
    assert result.source == "baseline"
    assert result.fallback_used is True
    assert result.proposed_details[0]["action"] == "select"
    assert event["error"] is None
    assert event["proposed"] == {"0": "select", "1": "skip"}
    assert event["applied"] == {"0": "skip", "1": "select"}
    assert event["context_sha256"]
    assert event["event_sha256"]


def test_active_applies_valid_complete_proposal(artifact_dir):
    result, event = run_decision(artifact_dir, "active", valid_selection())

    assert result.actions == {0: "select", 1: "skip"}
    assert result.source == "agno"
    assert result.fallback_used is False
    assert event["error"] is None


def test_active_rejects_wrong_selection_cardinality(artifact_dir):
    invalid = RoundDecision(
        phase="pre_round",
        actions=[
            ClientAction(cid=0, action="select"),
            ClientAction(cid=1, action="select"),
        ],
    )

    result, event = run_decision(artifact_dir, "active", invalid)

    assert result.actions == {0: "skip", 1: "select"}
    assert result.source == "baseline"
    assert result.fallback_used is True
    assert "exige 1 ocorrencias" in event["error"]


def test_audit_events_form_hash_chain(artifact_dir):
    audit_path = artifact_dir / "decisions.jsonl"
    policy = AgnoRoundPolicy("baseline", "fake-model", str(audit_path))
    kwargs = {
        "phase": "pre_round",
        "allowed_actions": {"select", "skip"},
        "fallback": {0: "select"},
        "required_counts": {"select": 1},
    }

    policy.decide(context={"run_id": "test", "round": 1}, **kwargs)
    policy.decide(context={"run_id": "test", "round": 2}, **kwargs)

    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["previous_event_sha256"] is None
    assert events[1]["previous_event_sha256"] == events[0]["event_sha256"]


def test_post_deadline_accepts_only_valid_actions(artifact_dir):
    response = RoundDecision(
        phase="post_deadline",
        actions=[ClientAction(cid=3, action="accept_stale", reason_codes=["INFORMATIVE"])],
    )
    policy = AgnoRoundPolicy("active", "fake-model", str(artifact_dir / "post.jsonl"))
    policy._agent = FakeAgent(response)
    result = policy.decide(
        phase="post_deadline",
        context={"run_id": "run-1", "round": 2, "decision_group": "arrived_pending"},
        allowed_actions={"accept_stale", "reduce_weight", "discard"},
        fallback={3: "discard"},
    )
    assert result.actions == {3: "accept_stale"}
    assert result.source == "agno"


def test_policy_enforces_per_client_action_allowlist(artifact_dir):
    proposal = RoundDecision(
        phase="post_deadline",
        actions=[ClientAction(cid=4, action="defer")],
    )
    policy = AgnoRoundPolicy("active", "fake-model", str(artifact_dir / "restricted.jsonl"))
    policy._agent = FakeAgent(proposal)
    result = policy.decide(
        phase="post_deadline",
        context={"round": 3},
        allowed_actions={"defer", "discard", "reduce_weight"},
        fallback={4: "discard"},
        client_allowed_actions={4: {"discard", "reduce_weight"}},
    )
    assert result.actions == {4: "discard"}
    assert result.fallback_used is True


def test_replay_reuses_recorded_decision_without_calling_model(artifact_dir):
    source_path = artifact_dir / "source.jsonl"
    source = AgnoRoundPolicy("active", "fake-model", str(source_path))
    source._agent = FakeAgent(valid_selection())
    source.decide(
        phase="pre_round",
        context={"run_id": "source-run", "round": 1},
        allowed_actions={"select", "skip"},
        fallback={0: "skip", 1: "select"},
        required_counts={"select": 1},
    )

    replay = AgnoRoundPolicy(
        "replay", "fake-model", str(artifact_dir / "replayed.jsonl"),
        replay_path=str(source_path),
    )
    result = replay.decide(
        phase="pre_round",
        context={"run_id": "replay-run", "round": 1},
        allowed_actions={"select", "skip"},
        fallback={0: "skip", 1: "select"},
        required_counts={"select": 1},
    )
    assert result.actions == {0: "select", 1: "skip"}
    assert result.source == "replay"


# --- Parametrizacao da criacao do agente -----------------------------------

from agno_policy import (  # noqa: E402
    AgentSettings,
    DEFAULT_STRATEGY_INSTRUCTIONS,
    PROTOCOL_INSTRUCTIONS,
    build_agent_model,
)


def test_settings_use_config_defaults_when_env_is_empty():
    settings = AgentSettings.from_env("groq", "openai/gpt-oss-120b", env={
        "AGENT_PROVIDER": "", "AGENT_MODEL_ID": "  ",
    })

    assert settings.formatted_model == "groq:openai/gpt-oss-120b"
    assert settings.name == "fl-round-policy"
    assert settings.debug_mode is False
    assert settings.debug_level == 1
    assert settings.markdown is False
    assert settings.use_json_mode is True
    assert settings.instructions == [*PROTOCOL_INSTRUCTIONS, DEFAULT_STRATEGY_INSTRUCTIONS]


def test_env_overrides_provider_model_and_agent_attributes():
    settings = AgentSettings.from_env("groq", "openai/gpt-oss-120b", env={
        "AGENT_PROVIDER": "Anthropic",
        "AGENT_MODEL_ID": "claude-opus-5-5",
        "AGENT_NAME": "Politica FL",
        "AGENT_INSTRUCTIONS": "Favoreca clientes com classes raras.",
        "AGENT_DEBUG": "true",
        "AGENT_DEBUG_LEVEL": "2",
        "AGENT_MARKDOWN": "false",
        "AGENT_USE_JSON_MODE": "false",
    })

    assert settings.formatted_model == "anthropic:claude-opus-5-5"
    assert settings.name == "Politica FL"
    assert settings.debug_mode is True
    assert settings.debug_level == 2
    assert settings.use_json_mode is False
    # As regras de protocolo continuam presentes; so o criterio muda.
    assert settings.instructions[:-1] == list(PROTOCOL_INSTRUCTIONS)
    assert settings.instructions[-1] == "Favoreca clientes com classes raras."


@pytest.mark.parametrize("env", [
    {"AGENT_DEBUG": "talvez"},
    {"AGENT_DEBUG_LEVEL": "3"},
])
def test_invalid_agent_env_values_are_rejected(env):
    with pytest.raises(ValueError):
        AgentSettings.from_env("groq", "openai/gpt-oss-120b", env=env)


def test_anthropic_model_omits_sampling_params():
    settings = AgentSettings(provider="anthropic", model_id="claude-opus-5-5")
    model = build_agent_model(settings, timeout_s=20.0, seed=42)

    assert type(model).__name__ == "Claude"
    assert model.id == "claude-opus-5-5"
    assert model.temperature is None
    assert model.timeout == 20.0
    assert model.client_params["max_retries"] == 2


def test_groq_model_keeps_deterministic_params():
    settings = AgentSettings(provider="groq", model_id="openai/gpt-oss-120b")
    model = build_agent_model(settings, timeout_s=20.0, seed=7)

    assert type(model).__name__ == "Groq"
    assert model.temperature == 0
    assert model.seed == 7
    assert model.max_retries == 2


def test_missing_provider_key_falls_back_to_baseline(artifact_dir, monkeypatch):
    # Impede que o .env real do desenvolvedor devolva a chave e dispare chamada paga.
    monkeypatch.setattr("agno_policy.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    policy = AgnoRoundPolicy(
        "active", "claude-opus-5-5", str(artifact_dir / "nokey.jsonl"),
        settings=AgentSettings(provider="anthropic", model_id="claude-opus-5-5"),
    )
    result = policy.decide(
        phase="pre_round",
        context={"run_id": "r", "round": 1},
        allowed_actions={"select", "skip"},
        fallback={0: "select"},
        required_counts={"select": 1},
    )
    event = json.loads((artifact_dir / "nokey.jsonl").read_text(encoding="utf-8"))

    assert result.fallback_used is True
    assert "ANTHROPIC_API_KEY" in event["error"]
    assert event["provider"] == "anthropic"
    assert event["model_id"] == "claude-opus-5-5"


def test_long_rationale_is_truncated_instead_of_rejected():
    from agno_policy import RoundDecision

    decision = RoundDecision.model_validate({
        "phase": "post_deadline",
        "actions": [{"cid": 1, "action": "accept_stale", "rationale": "x" * 900}],
        "policy_summary": "y" * 900,
    })

    assert len(decision.actions[0].rationale) == 300
    assert len(decision.policy_summary) == 500
