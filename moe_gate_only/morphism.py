"""Initialization services for externally provided replacement gates.

The gate system does not ship concrete gate implementations.  A replacement
gate is provided externally and must satisfy the gate contract documented in
:mod:`moe_gate_only.contract`.  This module implements the initialization
strategies the system applies when a replacement gate should reproduce the
native router projection at step zero:

- **base copy**: gates exposing ``base = nn.Linear(model_dim, num_experts,
  bias=False)`` get the native router weights copied into ``base.weight`` and
  ``base.bias`` zeroed.
- **function-preserving delegation**: gates implementing
  ``initialize_from_projection(weight, bias=None) -> dict`` take full control
  of their own initialization and report ``morphism_metrics`` back.
- **promoted precision**: :func:`promoted_projection_dtype` chooses a safe
  compute dtype for factorized morphism initializers
  (float8 -> float16 -> float32 -> float64).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def promoted_projection_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the next practical floating dtype for factorized projections."""
    float8_dtypes = {
        getattr(torch, name)
        for name in (
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
        )
        if hasattr(torch, name)
    }
    if dtype in float8_dtypes:
        return torch.float16
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    if dtype == torch.float32:
        return torch.float64
    return dtype


def initialize_gate_from_projection(
    gate: nn.Module,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Initialize *gate* to reproduce a native router projection.

    Delegates to ``gate.initialize_from_projection(weight, bias=...)`` when the
    gate implements that protocol.  Otherwise requires ``gate.base`` to be an
    ``nn.Linear`` matching the native weight shape; the native weights are
    copied in and the base bias is zeroed.

    Returns a dict of morphism metrics describing the initialization.
    """
    if not isinstance(weight, torch.Tensor):
        raise ValueError("Original-weight initialization requires a native weight tensor")

    initializer = getattr(gate, "initialize_from_projection", None)
    if callable(initializer):
        return dict(
            initializer(weight, bias=bias if isinstance(bias, torch.Tensor) else None)
        )

    base = getattr(gate, "base", None)
    target = getattr(base, "weight", None)
    if not isinstance(target, torch.Tensor):
        raise ValueError(
            "initialize_gate_from_projection requires the gate to implement "
            "initialize_from_projection(weight, bias=None) or to expose "
            "base = nn.Linear(model_dim, num_experts, bias=False)"
        )
    if tuple(weight.shape) != tuple(target.shape):
        raise ValueError(
            f"Gate base projection shape {tuple(target.shape)} does not match "
            f"native router shape {tuple(weight.shape)}"
        )
    with torch.no_grad():
        target.copy_(weight.to(device=target.device, dtype=target.dtype))
        base_bias = getattr(base, "bias", None)
        if isinstance(base_bias, torch.Tensor):
            base_bias.zero_()
    return {
        "initialization": "copied_base",
        "source_dtype": str(weight.dtype),
        "parameter_dtype": str(target.dtype),
        "bit_exact_projection_expected": True,
        "native_bias": bias is not None,
    }
