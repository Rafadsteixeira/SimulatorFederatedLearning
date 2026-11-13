# -*- coding: utf-8 -*-
"""Configuracao do experimento, caminhos e constantes do CIFAR-10."""

import json
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Dict, List

# ==============================================================================
#  CONFIGURACAO DO EXPERIMENTO
# ==============================================================================

@dataclass
class ExperimentConfig:
    n_clients:                int   = 5
    n_rounds:                 int   = 5
    weight_mode:              str   = "random"
    fixed_weights:            Dict  = field(default_factory=dict)
    high_latency_periodicity: int   = 0
    high_latency_fraction:    float = 0.3
    seed:                     int   = 42
    local_epochs:             int   = 1
    batch_size:               int   = 32

    # Modelo de tempo, deadlines e staleness
    deadline_mode:            str   = "percentile"  # 'fixed' ou 'percentile'
    round_deadline:           float = 12.0          # segundos simulados (se deadline_mode == 'fixed')
    deadline_percentile:      float = 80.0          # percentil de corte (se deadline_mode == 'percentile')
    comp_time_per_sample:     float = 0.005         # tempo computacional base (s/amostra/epoca)
    enable_staleness:         bool  = True          # aplicar amortecimento de staleness na agregacao
    staleness_gamma:          float = 0.5           # expoente de penalizacao: 1 / (1 + tau)^gamma
    drop_stragglers:          bool  = False         # Compatibilidade: equivale a late_update_policy='discard'
    late_update_policy:       str   = "reduced_weight"  # discard, defer ou reduced_weight
    late_update_weight_factor: float = 0.5
    max_staleness:            int   = 2

    # Baselines de selecao e score multiobjetivo fixo.
    selection_mode:           str   = "multi_objective"
    selection_info_weight:    float = 0.45
    selection_rare_weight:    float = 0.25
    selection_diversity_weight: float = 0.15
    selection_time_weight:    float = 0.15
    rare_class_threshold:     float = 0.08
    time_jitter_fraction:     float = 0.0

    # Politica opcional baseada em LLM/Agno.
    # Em shadow o agente decide e audita, mas o baseline continua sendo aplicado.
    agent_policy_mode:        str   = "shadow"  # baseline, shadow, active ou replay
    # Provedor e modelo Agno ("provider:model_id"); AGENT_PROVIDER e
    # AGENT_MODEL_ID no ambiente/.env tem precedencia sobre estes valores.
    agent_provider:           str   = "groq"
    agent_model_id:           str   = "openai/gpt-oss-120b"
    agent_timeout_s:          float = 20.0
    agent_audit_file:         str   = "decisions.jsonl"
    agent_replay_file:        str   = ""

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg = cls()
        cfg.n_clients                = int(data.get('n_clients',                cfg.n_clients))
        cfg.n_rounds                 = int(data.get('n_rounds',                 cfg.n_rounds))
        cfg.weight_mode              = str(data.get('weight_mode',              cfg.weight_mode))
        fw = {k: v for k, v in data.get('fixed_weights', {}).items()
              if k.lstrip('-').isdigit()}
        cfg.fixed_weights            = {int(k): float(v) for k, v in fw.items()}
        cfg.high_latency_periodicity = int(data.get('high_latency_periodicity', cfg.high_latency_periodicity))
        cfg.high_latency_fraction    = float(data.get('high_latency_fraction',  cfg.high_latency_fraction))
        cfg.seed                     = int(data.get('seed',                     cfg.seed))
        cfg.local_epochs             = int(data.get('local_epochs',             cfg.local_epochs))
        cfg.batch_size               = int(data.get('batch_size',               cfg.batch_size))

        cfg.deadline_mode            = str(data.get('deadline_mode',            cfg.deadline_mode))
        cfg.round_deadline           = float(data.get('round_deadline',         cfg.round_deadline))
        cfg.deadline_percentile      = float(data.get('deadline_percentile',    cfg.deadline_percentile))
        cfg.comp_time_per_sample     = float(data.get('comp_time_per_sample',   cfg.comp_time_per_sample))
        cfg.enable_staleness         = bool(data.get('enable_staleness',        cfg.enable_staleness))
        cfg.staleness_gamma          = float(data.get('staleness_gamma',        cfg.staleness_gamma))
        cfg.drop_stragglers          = bool(data.get('drop_stragglers',         cfg.drop_stragglers))
        legacy_policy = "discard" if cfg.drop_stragglers else cfg.late_update_policy
        cfg.late_update_policy       = str(data.get('late_update_policy', legacy_policy))
        cfg.late_update_weight_factor = float(data.get('late_update_weight_factor', cfg.late_update_weight_factor))
        cfg.max_staleness            = int(data.get('max_staleness', cfg.max_staleness))
        cfg.selection_mode           = str(data.get('selection_mode', cfg.selection_mode))
        cfg.selection_info_weight    = float(data.get('selection_info_weight', cfg.selection_info_weight))
        cfg.selection_rare_weight    = float(data.get('selection_rare_weight', cfg.selection_rare_weight))
        cfg.selection_diversity_weight = float(data.get('selection_diversity_weight', cfg.selection_diversity_weight))
        cfg.selection_time_weight    = float(data.get('selection_time_weight', cfg.selection_time_weight))
        cfg.rare_class_threshold     = float(data.get('rare_class_threshold', cfg.rare_class_threshold))
        cfg.time_jitter_fraction     = float(data.get('time_jitter_fraction', cfg.time_jitter_fraction))
        cfg.agent_policy_mode        = str(data.get('agent_policy_mode', cfg.agent_policy_mode))
        cfg.agent_provider           = str(data.get('agent_provider', cfg.agent_provider)).strip().lower()
        cfg.agent_model_id           = str(data.get('agent_model_id', cfg.agent_model_id))
        cfg.agent_timeout_s          = float(data.get('agent_timeout_s', cfg.agent_timeout_s))
        cfg.agent_audit_file         = str(data.get('agent_audit_file', cfg.agent_audit_file))
        cfg.agent_replay_file        = str(data.get('agent_replay_file', cfg.agent_replay_file))

        if cfg.n_clients < 1 or cfg.n_rounds < 1:
            raise ValueError("n_clients e n_rounds devem ser maiores que zero")
        if cfg.local_epochs < 1 or cfg.batch_size < 1:
            raise ValueError("local_epochs e batch_size devem ser maiores que zero")
        if cfg.seed < 0:
            raise ValueError("seed deve ser nao negativa")
        if cfg.high_latency_periodicity < 0:
            raise ValueError("high_latency_periodicity deve ser nao negativa")
        if not np.isfinite(cfg.high_latency_fraction) or not 0 <= cfg.high_latency_fraction <= 1:
            raise ValueError("high_latency_fraction deve estar entre 0 e 1")

        if cfg.weight_mode not in ("random", "fixed"):
            raise ValueError(f"weight_mode deve ser 'random' ou 'fixed', recebido: {cfg.weight_mode!r}")
        if cfg.deadline_mode not in ("fixed", "percentile"):
            raise ValueError("deadline_mode deve ser 'fixed' ou 'percentile'")
        if not np.isfinite(cfg.round_deadline) or cfg.round_deadline <= 0:
            raise ValueError("round_deadline deve ser maior que zero")
        if not np.isfinite(cfg.deadline_percentile) or not 0 <= cfg.deadline_percentile <= 100:
            raise ValueError("deadline_percentile deve estar entre 0 e 100")
        if not np.isfinite(cfg.comp_time_per_sample) or cfg.comp_time_per_sample < 0:
            raise ValueError("comp_time_per_sample deve ser finito e nao negativo")
        if not np.isfinite(cfg.staleness_gamma) or cfg.staleness_gamma < 0:
            raise ValueError("staleness_gamma deve ser finito e nao negativo")
        if cfg.max_staleness < 0:
            raise ValueError("max_staleness deve ser nao negativo")
        if cfg.late_update_policy not in ("discard", "defer", "reduced_weight"):
            raise ValueError("late_update_policy deve ser 'discard', 'defer' ou 'reduced_weight'")
        if not np.isfinite(cfg.late_update_weight_factor) or not 0 < cfg.late_update_weight_factor <= 1:
            raise ValueError("late_update_weight_factor deve estar no intervalo (0, 1]")
        if cfg.selection_mode not in ("random", "fastest", "dijkstra", "latency", "multi_objective"):
            raise ValueError("selection_mode deve ser random, fastest, dijkstra ou multi_objective")
        if not np.isfinite(cfg.rare_class_threshold) or not 0 <= cfg.rare_class_threshold <= 1:
            raise ValueError("rare_class_threshold deve estar entre 0 e 1")
        if not np.isfinite(cfg.time_jitter_fraction) or not 0 <= cfg.time_jitter_fraction <= 1:
            raise ValueError("time_jitter_fraction deve estar entre 0 e 1")
        selection_weights = (
            cfg.selection_info_weight,
            cfg.selection_rare_weight,
            cfg.selection_diversity_weight,
            cfg.selection_time_weight,
        )
        if (not all(np.isfinite(weight) and weight >= 0 for weight in selection_weights)
                or sum(selection_weights) <= 0):
            raise ValueError("pesos de selecao devem ser nao negativos e ter soma positiva")
        if cfg.agent_policy_mode not in ("baseline", "shadow", "active", "replay"):
            raise ValueError(
                "agent_policy_mode deve ser baseline, shadow, active ou replay"
            )
        if not cfg.agent_provider.strip():
            raise ValueError("agent_provider nao pode ser vazio")
        if not cfg.agent_model_id.strip():
            raise ValueError("agent_model_id nao pode ser vazio")
        if not np.isfinite(cfg.agent_timeout_s) or cfg.agent_timeout_s <= 0:
            raise ValueError("agent_timeout_s deve ser maior que zero")
        if not cfg.agent_audit_file.strip():
            raise ValueError("agent_audit_file nao pode ser vazio")
        if cfg.agent_policy_mode == "replay" and not cfg.agent_replay_file.strip():
            raise ValueError("agent_replay_file e obrigatorio no modo replay")
        return cfg

    def to_summary_lines(self) -> List[str]:
        mode_str = "Aleatorio" if self.weight_mode == "random" else "Fixo"
        dl_str   = f"Fixo ({self.round_deadline:.1f}s)" if self.deadline_mode == "fixed" else f"P{int(self.deadline_percentile)}"
        lines = [
            f"Clientes  : {self.n_clients}",
            f"Rounds    : {self.n_rounds}",
            f"Pesos     : {mode_str}",
            f"Deadline  : {dl_str}",
            f"Staleness : {'Ativo' if self.enable_staleness else 'Inativo'}",
        ]
        if self.high_latency_periodicity > 0:
            lines.append(f"Alta-lat. : a cada {self.high_latency_periodicity} rounds")
            lines.append(f"Fracao    : {int(self.high_latency_fraction * 100)}%")
        else:
            lines.append("Alta-lat. : desativada")
        lines.append(f"Agno      : {self.agent_policy_mode.upper()}")
        lines.append(f"Selecao   : {self.selection_mode}")
        lines.append(
            f"Atrasos   : {self.late_update_policy} "
            f"(fator={self.late_update_weight_factor:.2f}, max tau={self.max_staleness})"
        )
        if self.agent_policy_mode != "baseline":
            lines.append(f"LLM       : {self.agent_provider}:{self.agent_model_id}")
        lines.append(f"Seed      : {self.seed}")
        lines.append(f"Epocas loc: {self.local_epochs}")
        lines.append(f"Batch     : {self.batch_size}")
        return lines


# Caminho absoluto do diretório do script — usado em todos os paths de dados
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Caminhos absolutos relativos ao diretório do script
DATA_DIR    = os.path.join(_SCRIPT_DIR, "data_cifar10")
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "fl_results_v15")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CLASSES = 10
CLASS_NAMES = [
    "aviao", "automovel", "passaro", "gato", "veado",
    "cachorro", "sapo", "cavalo", "navio", "caminhao"
]
CLASS_COLORS = [
    "#4FC3F7", "#EF5350", "#66BB6A", "#FFA726", "#AB47BC",
    "#26C6DA", "#D4E157", "#FF7043", "#42A5F5", "#8D6E63"
]
