"""Gate-only MoE experiments.

Universal gate replacement for HuggingFace MoE models.

The HF MoE block keeps its original routing logic (softmax, top-k, dispatch)
unchanged.  Only the gate/scorer module that produces expert logits is
replaced with a custom trainable network.

Replacement gates are **not shipped by this package** — they are provided
externally as a callable factory ``(model_dim, num_experts) -> nn.Module``,
compiled source (``compile_gate_from_string``), or a name resolved through a
caller-provided registry.  The package focuses on how gates are discovered,
replaced, integrated, initialized, verified, and trained.  See
``moe_gate_only.contract`` for the gate contract.

Usage
-----
>>> from transformers import AutoModelForCausalLM
>>> from moe_gate_only import find_moe_gates, install_gates, freeze_except_gates

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

>>> def linear_gate(model_dim: int, num_experts: int):
...     import torch.nn as nn
...     return nn.Linear(model_dim, num_experts, bias=False)

>>> sites = find_moe_gates(model)              # optional inspection
>>> installs = install_gates(model, linear_gate)  # or compile_gate_from_string / registry
>>> freeze_except_gates(model, installs)       # freeze base model

>>> # train normally
"""

from .contract import compile_gate_from_string
from .morphism import initialize_gate_from_projection, promoted_projection_dtype
from .universal import (
    GateCandidateReport,
    GateInstall,
    GateSite,
    assert_hf_native_model,
    count_parameters,
    find_moe_gates,
    freeze_except_gates,
    gate_trainable_parameters,
    explain_gate_candidates,
    get_gate_candidate_report,
    get_gate_logits,
    install_gates,
    restore_gates,
    set_teacher_student_weight,
    teacher_student_distillation_loss,
    trainable_parameter_names,
)
from .session import GateSessionSummary, MoEGateSession, SessionState
from .checkpoint import load_gate_checkpoint, save_gate_checkpoint
from .training import (
    GateTrainingResult,
    create_gate_optimizer,
    evaluate_language_model_loss,
    train_gate_step,
    train_gates,
)
from .nngpt_data import build_nngenprompt_dataloaders
from .metrics import (
    collect_gate_metrics,
    collect_teacher_student_metrics,
    reset_teacher_student_metrics,
)

__all__ = [
    "GateCandidateReport",
    "GateInstall",
    "GateSite",
    "GateSessionSummary",
    "GateTrainingResult",
    "MoEGateSession",
    "SessionState",
    "assert_hf_native_model",
    "build_nngenprompt_dataloaders",
    "compile_gate_from_string",
    "collect_gate_metrics",
    "collect_teacher_student_metrics",
    "count_parameters",
    "find_moe_gates",
    "freeze_except_gates",
    "gate_trainable_parameters",
    "explain_gate_candidates",
    "get_gate_candidate_report",
    "get_gate_logits",
    "initialize_gate_from_projection",
    "install_gates",
    "load_gate_checkpoint",
    "promoted_projection_dtype",
    "restore_gates",
    "reset_teacher_student_metrics",
    "set_teacher_student_weight",
    "save_gate_checkpoint",
    "create_gate_optimizer",
    "evaluate_language_model_loss",
    "train_gate_step",
    "train_gates",
    "trainable_parameter_names",
    "teacher_student_distillation_loss",
]
