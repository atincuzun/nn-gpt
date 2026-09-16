from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from manual_complex_gate import LLMGeneratedGate
from moe_cycle.gate_source import _validate_gate_source
from moe_gate_only.contract import compile_gate_from_string
from moe_gate_only.contracts import _DeepSeekV2Gate
from moe_gate_only.morphism import initialize_gate_from_projection


@pytest.mark.parametrize("model_dim,num_experts", [(32, 4), (2048, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_initialization_shapes_and_reload(model_dim, num_experts, dtype):
    torch.manual_seed(42)
    gate = LLMGeneratedGate(model_dim, num_experts).to(dtype=dtype)
    weight = torch.randn(num_experts, model_dim, dtype=dtype)
    initialize_gate_from_projection(gate, weight)
    for shape in [(model_dim,), (6, model_dim), (2, 3, model_dim)]:
        x = torch.randn(*shape, dtype=dtype)
        for training in [True, False]:
            gate.train(training)
            output = gate(x)
            assert output.shape == (*shape[:-1], num_experts)
            assert torch.isfinite(output).all()
            assert torch.equal(output, F.linear(x, weight))
    restored = LLMGeneratedGate(model_dim, num_experts).to(dtype=dtype)
    restored.load_state_dict(gate.state_dict())
    assert torch.equal(restored(x), gate(x))


def test_manual_source_passes_cycle_validator_and_loader():
    source = (Path(__file__).resolve().parents[1] / "manual_complex_gate.py").read_text()
    _validate_gate_source(source, [(32, 4), (2048, 64)])
    gate = compile_gate_from_string(source)(32, 4)
    assert gate(torch.randn(2, 3, 32)).shape == (2, 3, 4)


def test_correction_and_hidden_layers_learn_without_changing_base():
    torch.manual_seed(42)
    gate = LLMGeneratedGate(32, 4)
    gate.base.requires_grad_(False)
    original_base = gate.base.weight.detach().clone()
    optimizer = torch.optim.SGD(gate.parameters(), lr=0.1)
    x = torch.randn(16, 32)
    target = gate.base(x).detach() + torch.randn(16, 4)

    F.mse_loss(gate(x), target).backward()
    assert gate.correction.weight.grad.abs().sum() > 0
    assert gate.value.weight.grad.count_nonzero() == 0
    optimizer.step()
    optimizer.zero_grad()

    F.mse_loss(gate(x), target).backward()
    for name, parameter in gate.named_parameters():
        if name.startswith("base."):
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert torch.equal(gate.base.weight, original_base)
    assert not torch.equal(gate(x), gate.base(x))
    assert torch.allclose(gate(x), torch.stack([gate(token) for token in x]), atol=1e-6)
    assert torch.equal(gate(x), gate(x))


def test_deepseek_wrapper_preserves_native_routing_at_initialization():
    torch.manual_seed(42)
    gate = LLMGeneratedGate(2048, 64)
    weight = torch.randn(64, 2048) * 0.02
    initialize_gate_from_projection(gate, weight)
    wrapped = _DeepSeekV2Gate(gate, top_k=6)
    wrapped.norm_topk_prob = True
    wrapped.alpha = 0.001
    wrapped.seq_aux = True
    x = torch.randn(2, 3, 2048, dtype=torch.bfloat16)
    logits = F.linear(x.reshape(-1, 2048).float(), weight)
    for training in [False, True]:
        wrapped.train(training)
        expected = wrapped._route_from_logits(x, logits)
        actual = wrapped(x)
        for result, reference in zip(actual, expected):
            if reference is None:
                assert result is None
            else:
                assert torch.equal(result, reference)
