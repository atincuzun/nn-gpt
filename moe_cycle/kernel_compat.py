"""Device-aware dispatch for the LFM2-MoE causal convolution.

transformers swaps in the real ``causal_conv1d`` CUDA kernel as soon as the
package is importable, without checking the device of the input. A CPU forward
then fails with "Expected x.is_cuda() to be true", which breaks the CPU-only
runs, while the kernel is a large speed-up on the GPU.

The patch keeps the installed kernel for CUDA tensors and routes everything else
to the reference PyTorch implementation that transformers defines alongside it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _reference_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    """The pure-PyTorch implementation transformers ships for LFM2-MoE."""
    from transformers.activations import ACT2FN

    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def _reference_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    """The pure-PyTorch conv-state update transformers ships for LFM2-MoE."""
    from transformers.activations import ACT2FN

    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def ensure_device_aware_causal_conv1d() -> bool:
    """Make the LFM2-MoE convolution fall back to PyTorch off CUDA.

    Returns True when a patch was installed.
    """
    try:
        from transformers.models.lfm2_moe import modeling_lfm2_moe as module
    except ImportError:
        return False

    patched = False
    for name, reference in (
        ("causal_conv1d_fn", _reference_conv1d_fn),
        ("causal_conv1d_update", _reference_conv1d_update),
    ):
        current = getattr(module, name, None)
        if current is None or getattr(current, "_device_aware", False):
            continue

        def make_wrapper(kernel, torch_reference):
            def wrapper(*args, **kwargs):
                tensor = args[0] if args else kwargs.get("hidden_states")
                if isinstance(tensor, torch.Tensor) and tensor.device.type != "cuda":
                    return torch_reference(*args, **kwargs)
                return kernel(*args, **kwargs)

            wrapper._device_aware = True
            return wrapper

        setattr(module, name, make_wrapper(current, reference))
        patched = True
    return patched
