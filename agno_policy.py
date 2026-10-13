"""Camada opcional de politica Agno para o simulador de FL.

Este modulo nao conhece pesos nem dados de treinamento. Ele somente recebe
estatisticas agregadas, pede uma decisao estruturada ao LLM, valida seu formato
e grava um evento JSONL que permite reproduzir a execucao sem nova chamada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


Action = Literal["select", "skip", "defer", "accept_stale", "reduce_weight", "discard"]


class ClientAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cid: int
    action: Action
    reason_codes: List[str] = Field(default_factory=list, max_length=4)
    rationale: str = Field(default="", max_length=300)


class RoundDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: Literal["pre_round", "post_deadline"]
    actions: List[ClientAction] = Field(min_length=1)
    policy_summary: str = Field(default="", max_length=500)


@dataclass
class PolicyResult:
    actions: Dict[int, str]
    source: str
    fallback_used: bool
    policy_summary: str = ""
    proposed_details: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    raw_response: Any = None


class AgnoRoundPolicy:
    """Politica com tres modos: baseline, shadow e active.

    ``shadow`` chama o modelo e audita a decisao, mas devolve o fallback.
    ``active`` aplica apenas a parte validada da resposta do modelo.
    """

    VALID_MODES = {"baseline", "shadow", "active", "replay"}

    def __init__(self, mode: str, model_id: str, audit_path: str,
                 timeout_s: float = 20.0, seed: int = 42,
                 replay_path: Optional[str] = None) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode deve ser um de {sorted(self.VALID_MODES)}, recebido: {mode!r}"
            )
        if timeout_s <= 0:
            raise ValueError("timeout_s deve ser maior que zero")

        self.mode = mode
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.seed = seed
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.replay_path = Path(replay_path) if replay_path else None
        self._replay_events: List[Dict[str, Any]] = []
        self._replay_cursor = 0
        if self.mode == "replay":
            if self.replay_path is None or not self.replay_path.is_file():
                raise ValueError("modo replay exige agent_replay_file JSONL existente")
            self._replay_events = [
                json.loads(line)
                for line in self.replay_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        self._agent = None
        self._previous_event_hash: Optional[str] = None

        # A chave permanece no arquivo local e nunca e incorporada ao prompt/log.
        load_dotenv(Path(__file__).with_name(".env"))

    def _get_agent(self):
        if self._agent is not None:
            return self._agent
        try:
            from agno.agent import Agent
            from agno.models.groq import Groq
        except ImportError as exc:
            raise RuntimeError("Agno/Groq nao instalado. Execute uv sync para instalar as dependencias.") from exc
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY nao configurada no ambiente ou no arquivo .env"
            )
        self._agent = Agent(
            name="fl-round-policy",
            model=Groq(
                id=self.model_id,
                temperature=0,
                timeout=max(1, int(self.timeout_s)),
                max_retries=2,
                seed=self.seed,
            ),
            output_schema=RoundDecision,
            use_json_mode=True,
            instructions=[
                "Voce e uma politica de aprendizado federado. Responda somente pelo schema.",
                "Nunca invente clientes, metricas ou acoes fora da lista permitida.",
                "Retorne exatamente uma acao para cada cliente recebido, sem repetir IDs.",
                "Nao escolha pesos numericos: o simulador calcula pesos e limites.",
                "Priorize qualidade de validacao, deadline, diversidade de dados e equidade.",
            ],
            markdown=False,
            retries=1,
        )
        return self._agent

    @staticmethod
    def _parse_response(content: Any) -> RoundDecision:
        if isinstance(content, RoundDecision):
            return content
        if isinstance(content, BaseModel):
            content = content.model_dump() if hasattr(content, "model_dump") else content.dict()
        if isinstance(content, str):
            content = json.loads(content)
        if hasattr(RoundDecision, "model_validate"):
            return RoundDecision.model_validate(content)
        return RoundDecision.parse_obj(content)

    def _next_replay_response(self, phase: str, context: Mapping[str, Any]) -> Dict[str, Any]:
        wanted_round = context.get("round")
        wanted_group = context.get("decision_group")
        for index in range(self._replay_cursor, len(self._replay_events)):
            event = self._replay_events[index]
            event_context = event.get("context", {})
            if event.get("phase") != phase or event.get("round") != wanted_round:
                continue
            if event_context.get("decision_group") != wanted_group:
                continue
            self._replay_cursor = index + 1
            actions = event.get("proposed") or event.get("applied") or {}
            return {
                "phase": phase,
                "actions": [
                    {"cid": int(cid), "action": action}
                    for cid, action in actions.items()
                ],
                "policy_summary": event.get("policy_summary", "Reproducido do JSONL."),
            }
        raise ValueError(
            f"decisao ausente no replay: phase={phase}, round={wanted_round}, group={wanted_group}"
        )

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, Mapping):
            return {str(k): AgnoRoundPolicy._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [AgnoRoundPolicy._jsonable(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _audit(self, event: Dict[str, Any]) -> None:
        event = self._jsonable(event)
        event["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        event["context_sha256"] = sha256(
            json.dumps(event["context"], sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        event["previous_event_sha256"] = self._previous_event_hash
        canonical_event = json.dumps(
            event, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        event["event_sha256"] = sha256(canonical_event.encode("utf-8")).hexdigest()
        self._previous_event_hash = event["event_sha256"]
        with self.audit_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(event, ensure_ascii=False) + "\n")

    def decide(self, phase: str, context: Dict[str, Any],
               allowed_actions: Iterable[str], fallback: Mapping[int, str],
               required_counts: Optional[Mapping[str, int]] = None,
               client_allowed_actions: Optional[Mapping[int, Iterable[str]]] = None) -> PolicyResult:
        allowed = set(allowed_actions)
        fallback_dict = dict(fallback)
        required_counts_dict = dict(required_counts or {})
        client_allowed = {
            int(cid): set(actions) for cid, actions in (client_allowed_actions or {}).items()
        }
        if set(client_allowed) - set(fallback_dict):
            raise ValueError("client_allowed_actions referencia candidatos desconhecidos")
        if any(actions - allowed for actions in client_allowed.values()):
            raise ValueError("client_allowed_actions contem acoes fora de allowed_actions")
        if any(action not in allowed for action in fallback_dict.values()):
            raise ValueError("fallback contem acoes fora de allowed_actions")
        if any(fallback_dict[cid] not in actions for cid, actions in client_allowed.items()):
            raise ValueError("fallback viola acoes validas de um candidato")
        unknown_count_actions = set(required_counts_dict) - allowed
        if unknown_count_actions:
            raise ValueError(
                f"required_counts contem acoes nao permitidas: {sorted(unknown_count_actions)}"
            )
        prompt = json.dumps({
            "phase": phase,
            "allowed_actions": sorted(allowed),
            "client_allowed_actions": {
                str(cid): sorted(actions) for cid, actions in client_allowed.items()
            },
            "required_action_counts": required_counts_dict,
            "context": context,
        }, ensure_ascii=False, sort_keys=True)
        raw_response: Any = None
        error: Optional[str] = None
        proposed: Dict[int, str] = {}
        proposed_details: Dict[int, Dict[str, Any]] = {}
        summary = ""
        agno_run_id: Optional[str] = None
        started_at = perf_counter()

        if self.mode in {"shadow", "active"}:
            try:
                response = self._get_agent().run(prompt)
                agno_run_id = getattr(response, "run_id", None)
                raw_response = getattr(response, "content", response)
                parsed = self._parse_response(raw_response)
                if parsed.phase != phase:
                    raise ValueError("fase retornada pelo agente nao corresponde a chamada")
                summary = parsed.policy_summary
                seen_cids = set()
                for item in parsed.actions:
                    if item.cid in seen_cids:
                        raise ValueError(f"cliente duplicado na resposta: {item.cid}")
                    if item.cid not in fallback_dict:
                        raise ValueError(f"cliente desconhecido na resposta: {item.cid}")
                    if item.action not in client_allowed.get(item.cid, allowed):
                        raise ValueError(
                            f"acao {item.action!r} nao permitida para cliente {item.cid}"
                        )
                    seen_cids.add(item.cid)
                    proposed[item.cid] = item.action
                    proposed_details[item.cid] = {
                        "action": item.action,
                        "reason_codes": list(item.reason_codes),
                        "rationale": item.rationale,
                    }
                missing = set(fallback_dict) - seen_cids
                if missing:
                    raise ValueError(
                        f"resposta incompleta; clientes ausentes: {sorted(missing)}"
                    )
                for action, required_count in required_counts_dict.items():
                    actual_count = sum(1 for value in proposed.values() if value == action)
                    if actual_count != required_count:
                        raise ValueError(
                            f"acao {action!r} exige {required_count} ocorrencias; "
                            f"agente retornou {actual_count}"
                        )
            except Exception as exc:  # falha de rede/modelo nunca interrompe o experimento
                error = f"{type(exc).__name__}: {exc}"
                proposed = {}
                proposed_details = {}
        elif self.mode == "replay":
            try:
                raw_response = self._next_replay_response(phase, context)
                parsed = self._parse_response(raw_response)
                if parsed.phase != phase:
                    raise ValueError("fase retornada pelo replay nao corresponde a chamada")
                summary = parsed.policy_summary
                seen_cids = set()
                for item in parsed.actions:
                    if item.cid in seen_cids:
                        raise ValueError(f"cliente duplicado no replay: {item.cid}")
                    if item.cid not in fallback_dict:
                        raise ValueError(f"cliente desconhecido no replay: {item.cid}")
                    if item.action not in client_allowed.get(item.cid, allowed):
                        raise ValueError(f"acao {item.action!r} nao permitida no replay")
                    seen_cids.add(item.cid)
                    proposed[item.cid] = item.action
                    proposed_details[item.cid] = {
                        "action": item.action,
                        "reason_codes": list(item.reason_codes),
                        "rationale": item.rationale,
                    }
                if seen_cids != set(fallback_dict):
                    raise ValueError("decisao replay nao cobre todos os candidatos")
                for action, required_count in required_counts_dict.items():
                    actual_count = sum(1 for value in proposed.values() if value == action)
                    if actual_count != required_count:
                        raise ValueError(f"replay viola cardinalidade obrigatoria para {action!r}")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                proposed = {}
                proposed_details = {}

        elapsed_ms = round((perf_counter() - started_at) * 1000.0, 3)
        use_proposed = bool(proposed) and self.mode in {"active", "replay"}
        result = PolicyResult(
            actions=(proposed if use_proposed else fallback_dict),
            source=("replay" if self.mode == "replay" else "agno")
            if use_proposed else "baseline",
            fallback_used=not use_proposed,
            policy_summary=summary,
            proposed_details=proposed_details,
            raw_response=raw_response,
        )
        self._audit({
            "run_id": context.get("run_id"), "round": context.get("round"),
            "phase": phase, "mode": self.mode, "model_id": self.model_id,
            "seed": self.seed, "agno_run_id": agno_run_id,
            "elapsed_ms": elapsed_ms,
            "context": context, "fallback": fallback_dict,
            "required_action_counts": required_counts_dict,
            "client_allowed_actions": {
                str(cid): sorted(actions) for cid, actions in client_allowed.items()
            },
            "proposed": proposed,
            "proposed_details": proposed_details,
            "applied": result.actions, "source": result.source,
            "fallback_used": result.fallback_used, "policy_summary": summary,
            "raw_response": raw_response, "error": error,
        })
        return result
