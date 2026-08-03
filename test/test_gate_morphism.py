from __future__ import annotations

import torch
import torch.nn.functional as F

from moe_gate_only.gates import GATE_FACTORIES
from moe_gate_only.morphism import ExactResidualMlpGate, SvdSignedPairGate
from moe_gate_only.universal import (
    GateSite,
    _DeepSeekV2Gate,
    _initialize_from_original_projection,
)


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


def test_svd_signed_pair_gate_matches_native_projection_and_topk() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(8, 24, generator=generator, dtype=torch.bfloat16)
    inputs = torch.randn(3, 5, 24, generator=generator)

    for activation in ("gelu", "silu"):
        gate = SvdSignedPairGate(24, 8, activation=activation)
        metrics = gate.initialize_from_projection(weight)
        expected = F.linear(inputs.float(), weight.float())
        actual = gate(inputs.float())

        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
        assert torch.equal(actual.topk(2, dim=-1).indices, expected.topk(2, dim=-1).indices)
        assert metrics["rank"] == 8
        assert metrics["hidden_dim"] == 16
        assert metrics["svd_relative_reconstruction_error"] <= 1e-12


def test_exact_residual_gate_is_bit_identical_at_initialization() -> None:
    generator = torch.Generator().manual_seed(13)
    weight = torch.randn(8, 24, generator=generator, dtype=torch.bfloat16)
    inputs = torch.randn(3, 5, 24, generator=generator, dtype=torch.bfloat16)
    gate = ExactResidualMlpGate(24, 8)

    metrics = gate.initialize_from_projection(weight)
    expected = F.linear(inputs.float(), weight.float())
    actual = gate(inputs.float())

    assert torch.equal(actual, expected)
    assert torch.count_nonzero(gate.residual(inputs.float())) == 0
    assert metrics["bit_exact_projection_expected"] is True


def test_exact_residual_gate_preserves_deepseek_routing_bits() -> None:
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(8, 24, generator=generator, dtype=torch.bfloat16)
    hidden_states = torch.randn(2, 5, 24, generator=generator, dtype=torch.bfloat16)
    gate = ExactResidualMlpGate(24, 8)
    gate.initialize_from_projection(weight)
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


def test_projection_initializer_protocol_and_factory() -> None:
    weight = torch.randn(6, 14)
    bias = torch.randn(6)
    site = _native_site(weight)
    site.gate.bias = torch.nn.Parameter(bias.clone())
    gate = GATE_FACTORIES["svd_signed_pair_silu"](14, 6)

    _initialize_from_original_projection(gate, site)

    inputs = torch.randn(4, 14)
    torch.testing.assert_close(
        gate(inputs),
        F.linear(inputs, weight, bias),
        rtol=5e-5,
        atol=5e-5,
    )
    assert gate.morphism_metrics is not None
    assert gate.morphism_metrics["runtime_relative_reconstruction_error"] <= 5e-5


def test_morphed_gate_parameters_can_leave_linear_solution() -> None:
    torch.manual_seed(11)
    weight = torch.randn(5, 12)
    gate = SvdSignedPairGate(12, 5, activation="silu")
    gate.initialize_from_projection(weight)
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
