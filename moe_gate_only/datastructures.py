from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

GateFactory = Callable[[int, int], nn.Module]


@dataclass
class GateSite:
    """A discovered MoE gate/scorer location inside a model."""

    layer_index: int
    block: nn.Module
    gate: nn.Module
    gate_attr: str
    model_dim: int
    num_experts: int
    pattern: str  # "linear", "wrapped_linear", "parameter_gate"
    path: str = ""
    score: float = 0.0
    evidence: Tuple[str, ...] = ()


@dataclass
class GateInstall:
    """Record of a single gate replacement."""

    site: GateSite
    old_gate: nn.Module
    new_gate: nn.Module
    owner: Optional[nn.Module] = None
    attr: Optional[str] = None
    old_child: Optional[nn.Module] = None
    hook_handles: List[Any] = field(default_factory=list)
    mode: str = "direct"
    teacher_gate: Optional[nn.Module] = None
    student_gate: Optional[nn.Module] = None
    teacher_requires_grad: Optional[Tuple[bool, ...]] = None


@dataclass
class GateCandidateReport:
    """Explanation for an accepted or rejected dynamic gate candidate."""

    path: str
    module_type: str
    pattern: str
    model_dim: Optional[int]
    num_experts: Optional[int]
    score: float
    accepted: bool
    reason: str
    evidence: Tuple[str, ...] = ()


@dataclass
class _TensorObservation:
    shape: Tuple[int, ...]
    dtype: torch.dtype
    is_floating_point: bool


@dataclass
class _ModuleTrace:
    path: str
    module_type: str
    input_tensors: List[_TensorObservation] = field(default_factory=list)
    output_tensors: List[_TensorObservation] = field(default_factory=list)


@dataclass
class _ForwardTrace:
    succeeded: bool
    traces: Dict[str, _ModuleTrace]
    error: Optional[str] = None
