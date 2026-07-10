"""Gate-only MoE experiments.

Universal gate replacement for HuggingFace MoE models.

The HF MoE block keeps its original routing logic (softmax, top-k, dispatch)
unchanged.  Only the gate/scorer module that produces expert logits is
replaced with a custom trainable network.

Usage
-----
>>> from transformers import AutoModelForCausalLM
>>> from moe_gate_only import find_moe_gates, install_gates, freeze_except_gates

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

>>> sites = find_moe_gates(model)              # optional inspection
>>> installs = install_gates(model, "mlp")     # or a callable/string
>>> freeze_except_gates(model, installs)       # freeze base model

>>> # train normally
"""

from .gates import GATE_FACTORIES, build_gate, compile_gate_from_string
from .universal import (
    GateCandidateReport,
    GateInstall,
    GateSite,
    assert_hf_native_model,
    count_parameters,
    find_moe_gates,
    freeze_except_gates,
    explain_gate_candidates,
    get_gate_candidate_report,
    get_gate_logits,
    install_gates,
    trainable_parameter_names,
)

__all__ = [
    "GATE_FACTORIES",
    "GateCandidateReport",
    "GateInstall",
    "GateSite",
    "assert_hf_native_model",
    "build_gate",
    "compile_gate_from_string",
    "count_parameters",
    "find_moe_gates",
    "freeze_except_gates",
    "explain_gate_candidates",
    "get_gate_candidate_report",
    "get_gate_logits",
    "install_gates",
    "trainable_parameter_names",
]
