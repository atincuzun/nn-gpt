"""Universal MoE gate discovery, replacement, and management API.

Backward-compatible facade.  The implementation now lives in focused
submodules:

- :mod:`moe_gate_only.datastructures` — shared data structures
- :mod:`moe_gate_only.sniff` — model-dimension sniffing
- :mod:`moe_gate_only.tree` — module-tree / path helpers
- :mod:`moe_gate_only.contracts` — routing-contract wrappers and detection
- :mod:`moe_gate_only.contract` — the replacement-gate contract
- :mod:`moe_gate_only.morphism` — initialization services
- :mod:`moe_gate_only.hooks` — gate logit capture hooks
- :mod:`moe_gate_only.verify` — transactional verify-forward and rollback
- :mod:`moe_gate_only.discovery` — structural and dynamic gate discovery
- :mod:`moe_gate_only.install` — gate installation, freezing, and utilities

Importing from ``moe_gate_only.universal`` keeps working unchanged.
"""

from __future__ import annotations

from .contract import compile_gate_from_string
from .contracts import (
    _DeepSeekV2Gate,
    _DeepSeekV2TeacherStudentGate,
    _HashScoreFnGate,
    _LogitsOnlyGate,
    _POST_PROCESSING_CACHE,
    _ScoreFnTopKGate,
    _SigmoidTopKGate,
    _SoftmaxTopKGate,
    _SparseSigmoidGate,
    _TopKWeightsIndicesGate,
    _copy_gate_attrs,
    _detect_from_source,
    _detect_post_processing,
    _get_gate_input_dim,
    _looks_probability_like,
    _module_device_dtype,
    _trace_gate_contract,
)
from .datastructures import (
    GateCandidateReport,
    GateFactory,
    GateInstall,
    GateSite,
    _ForwardTrace,
    _ModuleTrace,
    _TensorObservation,
)
from .discovery import (
    _WEIGHT_ACCESS_CACHE,
    _classify_gate,
    _discover_num_experts,
    _dynamic_expert_dims,
    _enumerate_dynamic_candidates,
    _expert_bank_paths,
    _extract_expert_counts,
    _extract_expert_counts_from_config,
    _filter_moe_gates,
    _find_linear_in_module,
    _find_linear_owner_attr,
    _find_moe_gates_dynamic,
    _find_moe_gates_fallback,
    _find_moe_gates_primary,
    _has_parameter_weight,
    _infer_shape_from_linear,
    _infer_shape_from_parameter,
    _iter_tensors,
    _make_direct_site,
    _observe_tensor,
    _parent_accesses_weight_of_child,
    _path_distance,
    _paths_overlap,
    _probe_gate_site,
    _same_child_signature,
    _score_dynamic_site,
    _site_report,
    _trace_has_input_dim,
    _trace_has_output_dim,
    _trace_has_topk_tuple,
    _trace_model_forward,
    explain_gate_candidates,
    find_moe_gates,
    get_gate_candidate_report,
)
from .hooks import (
    _HOOK_HANDLES,
    _current_replacement_module,
    _extract_logits_from_gate_output,
    _install_gate_logit_hook,
    get_gate_logits,
)
from .install import (
    _infer_top_k,
    _install_gate_site,
    _install_teacher_student_site,
    _new_gate_for_site,
    _patch_output_recorder,
    _resolve_top_k,
    _wrap_gate_for_contract,
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
from .sniff import (
    _discover_hidden_size,
    _discover_vocab_size,
    _guess_hidden_size,
    _guess_vocab_size,
)
from .tree import (
    _assign_child,
    _extract_layer_index,
    _find_module_path,
    _find_parent_block,
    _module_path_map,
    _resolve_parent_attr,
    _resolve_path_component,
)
from .verify import (
    _forward_with_sample,
    _move_sample_to_device,
    _remove_install_hooks,
    _rollback,
    _verify_forward,
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
