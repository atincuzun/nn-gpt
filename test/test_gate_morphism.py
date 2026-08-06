from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_gate_only.universal import (
    GateSite,
    _DeepSeekV2Gate,
    initialize_gate_from_projection,
)
from run_moe_gate_cycle import _verify_morphed_gate_projections


class ToyBaseGate(nn.Module):
    """An externally provided gate following the base-projection contract."""

    def __init__(self, model_dim: int, num_experts: int, *, residual: bool = False) -> None:
        super().__init__()
        self.base = nn.Linear(model_dim, num_experts, bias=False)
        self.residual = nn.Linear(model_dim, num_experts, bias=False) if residual else None
        if self.residual is not None:
            nn.init.zeros_(self.residual.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if self.residual is not None:
            output = output + self.residual(x)
        return output


class ProtocolGate(nn.Module):
    """An externally provided gate implementing the function-preserving protocol."""

    def __init__(self, model_dim: int, num_experts: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, model_dim))
        self.bias = nn.Parameter(torch.zeros(num_experts))
        self.morphism_metrics: dict | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def initialize_from_projection(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> dict:
        with torch.no_grad():
            self.weight.copy_(weight.to(dtype=self.weight.dtype))
            if bias is not None:
                self.bias.copy_(bias.to(dtype=self.bias.dtype))
            else:
                self.bias.zero_()
        self.morphism_metrics = {
            "initialization": "protocol_delegated",
            "runtime_relative_reconstruction_error": 0.0,
        }
        return dict(self.morphism_metrics)


def _native_site(weight: torch.Tensor) -> GateSite:
    native_gate = torch.nn.Module()
    native_gate.weight = torch.nn.Parameter(weight.clone())
    block = torch.nn.Module()
    block.gate = native_gate
    return GateSite(
        layer_index=0,
        block=block,
        gate=native_gate,
        gate_attr="gate",
        model_dim=weight.shape[1],
        num_experts=weight.shape[0],
        pattern="parameter_gate",
        path="layers.0.mlp.gate",
    )


def test_signed_pair_activations_recover_identity() -> None:
    values = torch.linspace(-10.0, 10.0, 1001)
    torch.testing.assert_close(
        F.gelu(values) - F.gelu(-values),
        values,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        F.silu(values) - F.silu(-values),
        values,
        rtol=1e-6,
        atol=1e-6,
    )


def test_morphism_verifier_preserves_native_input_dtype() -> None:
    class PromotedLinear(torch.nn.Module):
        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(weight.float())
            self.top_k = 2

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            assert inputs.dtype == torch.bfloat16
            return F.linear(inputs.float(), self.weight).to(inputs.dtype)

    weight = torch.eye(4, dtype=torch.bfloat16)
    inputs = torch.tensor(
        [[1.0, 0.75, 0.5, 0.25], [0.125, 0.25, 0.5, 1.0]],
        dtype=torch.bfloat16,
    )
    native_gate = torch.nn.Linear(4, 4, bias=False).to(dtype=torch.bfloat16)
    with torch.no_grad():
        native_gate.weight.copy_(weight)
    replacement = PromotedLinear(weight)
    install = SimpleNamespace(
        old_gate=native_gate,
        new_gate=replacement,
        site=SimpleNamespace(
            layer_index=0,
            gate=native_gate,
            block=torch.nn.Module(),
            num_experts=4,
        ),
    )

    metrics = _verify_morphed_gate_projections(
        [install],
        {id(native_gate): inputs},
    )

    assert metrics[0]["projection_max_abs_error"] == 0.0
    assert metrics[0]["routing_weights_equal"] is True


def test_morphism_verifier_requires_opt_in_for_approximation() -> None:
    class PerturbedLinear(torch.nn.Module):
        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(weight.float())
            self.top_k = 2

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            projected = F.linear(inputs.float(), self.weight).to(inputs.dtype)
            return projected + torch.tensor(0.25, dtype=inputs.dtype, device=inputs.device)

    weight = torch.eye(4, dtype=torch.bfloat16)
    inputs = torch.tensor(
        [[1.0, 0.75, 0.5, 0.25], [0.125, 0.25, 0.5, 1.0]],
        dtype=torch.bfloat16,
    )
    native_gate = torch.nn.Linear(4, 4, bias=False).to(dtype=torch.bfloat16)
    with torch.no_grad():
        native_gate.weight.copy_(weight)
    replacement = PerturbedLinear(weight)
    install = SimpleNamespace(
        old_gate=native_gate,
        new_gate=replacement,
        site=SimpleNamespace(
            layer_index=0,
            gate=native_gate,
            block=torch.nn.Module(),
            num_experts=4,
        ),
    )
    native_inputs = {id(native_gate): inputs}

    with pytest.raises(RuntimeError, match="failed per-layer routing equivalence"):
        _verify_morphed_gate_projections([install], native_inputs)

    metrics = _verify_morphed_gate_projections(
        [install],
        native_inputs,
        allow_approximate=True,
    )

    assert metrics[0]["strict_equivalent"] is False
    assert metrics[0]["approximate_accepted"] is True


def test_initializer_copies_base_projection_bitwise() -> None:
    generator = torch.Generator().manual_seed(13)
    weight = torch.randn(8, 24, generator=generator, dtype=torch.bfloat16)
    inputs = torch.randn(3, 5, 24, generator=generator, dtype=torch.bfloat16)
    gate = ToyBaseGate(24, 8, residual=True)

    metrics = initialize_gate_from_projection(gate, weight)
    expected = F.linear(inputs.float(), weight.float())
    actual = gate(inputs.float())

    assert torch.equal(actual, expected)
    assert torch.count_nonzero(gate.residual(inputs.float())) == 0
    assert metrics["initialization"] == "copied_base"
    assert metrics["bit_exact_projection_expected"] is True


def test_initializer_delegates_to_initialize_from_projection_protocol() -> None:
    weight = torch.randn(6, 14)
    bias = torch.randn(6)
    gate = ProtocolGate(14, 6)

    metrics = initialize_gate_from_projection(gate, weight, bias=bias)

    inputs = torch.randn(4, 14)
    torch.testing.assert_close(
        gate(inputs),
        F.linear(inputs, weight, bias),
        rtol=1e-6,
        atol=1e-6,
    )
    assert gate.morphism_metrics == metrics
    assert metrics["initialization"] == "protocol_delegated"


def test_initializer_rejects_gate_without_base_or_protocol() -> None:
    class BareGate(torch.nn.Module):
        def __init__(self, model_dim: int, num_experts: int) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(model_dim, num_experts, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    weight = torch.randn(6, 14)
    with pytest.raises(ValueError, match="initialize_gate_from_projection requires"):
        initialize_gate_from_projection(BareGate(14, 6), weight)


def test_initializer_rejects_shape_mismatch() -> None:
    gate = ToyBaseGate(14, 6)
    weight = torch.randn(8, 24)
    with pytest.raises(ValueError, match="does not match"):
        initialize_gate_from_projection(gate, weight)


def test_base_gate_preserves_deepseek_routing_bits() -> None:
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(8, 24, generator=generator, dtype=torch.bfloat16)
    hidden_states = torch.randn(2, 5, 24, generator=generator, dtype=torch.bfloat16)
    gate = ToyBaseGate(24, 8, residual=True)
    initialize_gate_from_projection(gate, weight)
    replacement = _DeepSeekV2Gate(gate, top_k=3)
    replacement.scoring_func = "softmax"
    replacement.topk_method = "greedy"
    replacement.norm_topk_prob = False
    replacement.routed_scaling_factor = 1.0
    replacement.alpha = 0.0

    native_logits = F.linear(hidden_states.reshape(-1, 24).float(), weight.float())
    native_scores = native_logits.softmax(dim=-1, dtype=torch.float32)
    native_weights, native_indices = torch.topk(
        native_scores, 3, dim=-1, sorted=False
    )
    indices, weights, auxiliary_loss = replacement(hidden_states)

    assert torch.equal(replacement._last_gate_logits, native_logits)
    assert torch.equal(indices, native_indices)
    assert torch.equal(weights, native_weights)
    assert auxiliary_loss is None


def test_projection_initializer_protocol_and_site() -> None:
    weight = torch.randn(6, 14)
    bias = torch.randn(6)
    site = _native_site(weight)
    site.gate.bias = torch.nn.Parameter(bias.clone())
    gate = ProtocolGate(14, 6)

    initialize_gate_from_projection(
        gate,
        site.gate.weight,
        bias=site.gate.bias,
    )

    inputs = torch.randn(4, 14)
    torch.testing.assert_close(
        gate(inputs),
        F.linear(inputs, weight, bias),
        rtol=1e-6,
        atol=1e-6,
    )
    assert gate.morphism_metrics is not None
    assert gate.morphism_metrics["runtime_relative_reconstruction_error"] == 0.0


def test_initialized_gate_parameters_can_leave_linear_solution() -> None:
    torch.manual_seed(11)
    weight = torch.randn(5, 12)
    gate = ToyBaseGate(12, 5, residual=True)
    initialize_gate_from_projection(gate, weight)
    inputs = torch.randn(8, 12)
    targets = torch.randn(8, 5)
    before = gate(inputs).detach().clone()

    optimizer = torch.optim.SGD(gate.parameters(), lr=0.05)
    loss = F.mse_loss(gate(inputs), targets)
    loss.backward()
    for parameter in gate.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.norm()) > 0
    optimizer.step()

    assert not torch.allclose(gate(inputs), before)
