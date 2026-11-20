"""Universal MoE gate discovery, replacement, and management API.

Backward-compatible facade over the focused implementation modules.  Public
names live in ``__all__``; only the private names still imported by the test
suite or bundled scripts are re-exported here.
"""

from __future__ import annotations

from .contract import compile_gate_from_string
from .contracts import _DeepSeekV2Gate, _DeepSeekV2TeacherStudentGate
from .datastructures import GateCandidateReport, GateFactory, GateInstall, GateSite
from .discovery import explain_gate_candidates, find_moe_gates, get_gate_candidate_report
from .hooks import _extract_logits_from_gate_output, get_gate_logits
from .install import (
    _new_gate_for_site,
    _resolve_top_k,
    assert_hf_native_model,
    count_parameters,
    freeze_except_gates,
    gate_trainable_parameters,
    install_gates,
    restore_gates,
    set_teacher_student_weight,
    teacher_student_distillation_loss,
    trainable_parameter_names,
)
from .morphism import initialize_gate_from_projection, promoted_projection_dtype

# Private names kept for the bundled tests/scripts; not part of the public API.
_COMPAT_PRIVATE = (
    _DeepSeekV2Gate,
    _DeepSeekV2TeacherStudentGate,
    _extract_logits_from_gate_output,
    _new_gate_for_site,
    _resolve_top_k,
)

__all__ = [
    "GateCandidateReport",
    "GateFactory",
    "GateInstall",
    "GateSite",
    "assert_hf_native_model",
    "compile_gate_from_string",
    "count_parameters",
    "explain_gate_candidates",
    "find_moe_gates",
    "freeze_except_gates",
    "gate_trainable_parameters",
    "get_gate_candidate_report",
    "get_gate_logits",
    "initialize_gate_from_projection",
    "install_gates",
    "promoted_projection_dtype",
    "restore_gates",
    "set_teacher_student_weight",
    "teacher_student_distillation_loss",
    "trainable_parameter_names",
]
