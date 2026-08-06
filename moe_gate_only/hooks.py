from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn

from .datastructures import GateInstall

_HOOK_HANDLES: dict[int, list[Any]] = {}


def _extract_logits_from_gate_output(out: Any, num_experts: int) -> torch.Tensor:
    """Extract the router logits tensor from a gate output (tensor or tuple)."""
    if isinstance(out, torch.Tensor):
        return out

    if isinstance(out, (tuple, list)):
        candidates = [
            x for x in out
            if isinstance(x, torch.Tensor)
            and x.is_floating_point()
            and x.ndim >= 2
            and x.shape[-1] == num_experts
        ]
        if not candidates:
            raise RuntimeError("No expert-sized float tensor found in gate output")

        # Llama4-style sparse sigmoid returns (sparse_scores, logits).  Both
        # tensors are finite and expert-sized, so prefer the second item for
        # two-tensor contracts.
        if len(out) == 2 and len(candidates) == 2:
            return candidates[-1]

        # Common HF routers return (logits, weights, indices).  If the first
        # element is expert-sized, it is the raw router logits.
        first = out[0]
        if (
            isinstance(first, torch.Tensor)
            and first.is_floating_point()
            and first.ndim >= 2
            and first.shape[-1] == num_experts
        ):
            return first

        return candidates[-1]

    raise RuntimeError(f"Unexpected gate output type: {type(out).__name__}")


def _current_replacement_module(inst: GateInstall) -> nn.Module:
    if inst.owner is not None and inst.attr is not None:
        return getattr(inst.owner, inst.attr)
    return getattr(inst.site.block, inst.site.gate_attr)


def _install_gate_logit_hook(inst: GateInstall) -> None:
    """Register a forward hook that captures the gate's router logits.

    The logits are stored in ``inst.new_gate._gate_logits`` (a list).
    Gradients are NOT detached — aux-loss gradients flow through.
    """
    target = _current_replacement_module(inst)
    if not hasattr(target, "_gate_logits"):
        target._gate_logits: list[torch.Tensor] = []

    def hook_fn(mod, inp, out):
        try:
            logits = _extract_logits_from_gate_output(out, inst.site.num_experts)
        except RuntimeError:
            logits = getattr(mod, "_last_gate_logits", None)
            if not isinstance(logits, torch.Tensor):
                raise
        mod._gate_logits.clear()
        mod._gate_logits.append(logits)  # keep only the latest forward's graph

    handle = target.register_forward_hook(hook_fn)
    inst.hook_handles.append(handle)
    _HOOK_HANDLES.setdefault(id(inst), []).append(handle)


def get_gate_logits(model: nn.Module) -> list[torch.Tensor]:
    """Return all collected gate logits from the last forward pass.

    After a model forward, this returns the router logits from every replaced
    gate.  Use these to compute a custom auxiliary loss.
    """
    all_logits: list[torch.Tensor] = []
    for name, module in model.named_modules():
        if hasattr(module, "_gate_logits"):
            all_logits.extend(module._gate_logits)
            module._gate_logits.clear()
    return all_logits
