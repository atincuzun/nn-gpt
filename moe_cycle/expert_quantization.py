"""Quantization for fused Mixture-of-Experts expert stacks.

bitsandbytes replaces ``nn.Linear`` modules, but transformers v5 stores MoE
expert weights as fused 3-D ``nn.Parameter`` tensors (``gate_up_proj``,
``down_proj``) that hold the overwhelming majority of a MoE's parameters. A
plain ``load_in_4bit=True`` load therefore leaves those stacks in bf16 and peaks
at the full model size on the GPU (transformers#43032).

Two quantized representations are supported for the fused stacks:

``nf4``
    ``bitsandbytes.nn.parametrize.replace_parameter_4bit`` (bitsandbytes PR
    #1720). 4-bit, ~9% relative weight error.
``int8``
    Row-wise int8 through ``bitsandbytes.functional.int8_vectorwise_quant``
    with the same parametrization mechanism. 8-bit, ~0.8% relative weight
    error, half the memory of bf16 instead of a quarter.

The int8 path matters for this model: NF4's error is large enough to degrade
code generation (a broken generated model fails evaluation before it can be
scored), while int8 leaves the generator essentially unchanged.

The load itself still has to happen somewhere that can hold the bf16 stacks, so
the caller stages the load on the CPU and moves the model to the GPU afterwards.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

# Fused expert stacks use these attribute names across the transformers v5 MoE
# architectures (LFM2-MoE, Qwen3-MoE, OLMoE, Granite-MoE, ...).
_FUSED_EXPERT_PARAMS = ("gate_up_proj", "down_proj")

_INT8_MAX = 127.0


class _Int8StackParametrization(nn.Module):
    """Dequantizes a row-wise int8 expert stack when the parameter is accessed."""

    def __init__(self, scales: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype):
        super().__init__()
        self.register_buffer("scales", scales)
        self.shape = shape
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        rows = packed.reshape(-1, packed.shape[-1])
        values = rows.float() * self.scales[:, None].float() / _INT8_MAX
        return values.reshape(self.shape).to(self.dtype)


def find_fused_expert_modules(model: nn.Module) -> list[nn.Module]:
    """Return the modules holding a fused expert stack that is still in float."""
    modules: list[nn.Module] = []
    for module in model.modules():
        for name in _FUSED_EXPERT_PARAMS:
            param = getattr(module, name, None)
            if isinstance(param, nn.Parameter) and param.is_floating_point():
                modules.append(module)
                break
    return modules


def _quantize_int8(module: nn.Module, param_name: str) -> None:
    import bitsandbytes.functional as bnb_functional

    weight = getattr(module, param_name).data
    shape = tuple(weight.shape)
    # bnb's vector-wise int8 kernel is fp16-only; bf16 -> fp16 is lossless for
    # weights (fp16 has the wider mantissa).
    rows = weight.reshape(-1, weight.shape[-1]).to(torch.float16)
    packed, scales, _ = bnb_functional.int8_vectorwise_quant(rows)

    carrier = nn.Parameter(weight.new_empty(0), requires_grad=False)
    carrier.data = packed
    setattr(module, param_name, carrier)
    P.register_parametrization(
        module,
        param_name,
        _Int8StackParametrization(scales, shape, weight.dtype),
        unsafe=True,
    )


def _pack_int8_stack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise int8 pack of a fused expert stack -> (packed, scales).

    The rows of a fused stack are the rows of its experts, so the per-row
    scales are exactly per-expert-per-row scales.
    """
    import bitsandbytes.functional as bnb_functional

    rows = weight.reshape(-1, weight.shape[-1]).to(torch.float16)
    packed, scales, _ = bnb_functional.int8_vectorwise_quant(rows)
    return packed.view(weight.shape), scales.view(*weight.shape[:-1], 1)


def _install_per_expert_int8_forward(module: nn.Module) -> None:
    """Dequantize only the experts a token actually routed to.

    The parametrization form materializes the whole fused stack on every
    access, which is 15.5 GiB of bf16 per token for a model whose router
    selects four experts out of thirty-two. Dequantizing the routed expert
    rows alone moves an eighth of that.
    """
    import torch.nn.functional as F

    packed_up = module._packed_gate_up
    packed_down = module._packed_down
    scales_up = module._scales_gate_up
    scales_down = module._scales_down
    dtype = module._packed_dtype
    num_experts = module.num_experts

    def dequant(packed: torch.Tensor, scales: torch.Tensor, index: int) -> torch.Tensor:
        return (packed[index].float() * scales[index].float() / _INT8_MAX).to(dtype)

    def forward(hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(
                current_state, dequant(packed_up, scales_up, expert_idx)
            ).chunk(2, dim=-1)
            current_hidden_states = module.act_fn(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states, dequant(packed_down, scales_down, expert_idx)
            )
            current_hidden_states = current_hidden_states * top_k_weights[
                token_idx, top_k_pos, None
            ]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states

    module.forward = forward


def _quantize_int8_per_expert(module: nn.Module) -> None:
    """Replace a fused expert stack with packed int8 data and a sparse forward."""
    up_weight = module.gate_up_proj.data
    down_weight = module.down_proj.data
    packed_up, scales_up = _pack_int8_stack(up_weight)
    packed_down, scales_down = _pack_int8_stack(down_weight)

    module.register_buffer("_packed_gate_up", packed_up)
    module.register_buffer("_packed_down", packed_down)
    module.register_buffer("_scales_gate_up", scales_up)
    module.register_buffer("_scales_down", scales_down)
    module._packed_dtype = up_weight.dtype

    del module.gate_up_proj
    del module.down_proj
    _install_per_expert_int8_forward(module)


def quantize_fused_experts(
    model: nn.Module,
    device: str | torch.device = "cuda:0",
    quant_type: str = "nf4",
) -> int:
    """Quantize every fused expert stack of *model*, in place.

    Each stack is moved to *device* first so the quantization kernels run on the
    accelerator; the quantized stack stays there. Returns the number of stacks
    that were quantized.
    """
    if quant_type not in ("nf4", "int8", "int8_per_expert"):
        raise ValueError(f"unsupported expert quantization: {quant_type}")

    modules = find_fused_expert_modules(model)
    if quant_type == "int8_per_expert":
        for module in modules:
            module.to(device)
            _quantize_int8_per_expert(module)
        return len(modules)
    if quant_type == "int8":
        for module in modules:
            module.to(device)
            for name in _FUSED_EXPERT_PARAMS:
                param = getattr(module, name, None)
                if isinstance(param, nn.Parameter) and param.is_floating_point():
                    _quantize_int8(module, name)
        return len(modules)

    from bitsandbytes.nn.parametrize import replace_parameter_4bit

    for module in modules:
        module.to(device)
        for name in _FUSED_EXPERT_PARAMS:
            param = getattr(module, name, None)
            if isinstance(param, nn.Parameter) and param.is_floating_point():
                replace_parameter_4bit(module, name, quant_type="nf4")
    return len(modules)
