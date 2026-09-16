import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from manual_fresh_gate import LLMGeneratedGate
from moe_cycle.cli import parse_args
from moe_cycle import cycle
from moe_cycle.gate_source import _validate_gate_source


def test_fresh_source_needs_explicit_mode():
    source = (Path(__file__).resolve().parents[1] / "manual_fresh_gate.py").read_text()
    with pytest.raises(ValueError, match="must define base"):
        _validate_gate_source(source, [(32, 4)])
    _validate_gate_source(source, [(32, 4), (2048, 64)], require_base=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fresh_gate_shapes_and_immediate_gradients(dtype):
    torch.manual_seed(42)
    gate = LLMGeneratedGate(2048, 64).to(dtype=dtype)
    assert not hasattr(gate, "base")
    assert gate.output.weight.count_nonzero() > 0
    x = torch.randn(2, 3, 2048, dtype=dtype)
    output = gate(x)
    assert output.shape == (2, 3, 64)
    assert torch.isfinite(output).all()
    assert torch.equal(output, gate(x))
    assert torch.equal(output.reshape(6, 64), gate(x.reshape(6, 2048)))
    output.float().square().mean().backward()
    for name, parameter in gate.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_random_init_cli_constraints(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py"])
    args = parse_args()
    assert not args.gate_random_init
    args.gate_random_init = True
    with pytest.raises(ValueError, match="requires --gate-source"):
        cycle._validate_args(args)
    args.gate_source = Path("manual_fresh_gate.py")
    args.gate_init_noise_scale = 0.01
    with pytest.raises(ValueError, match="requires --gate-init-noise-scale 0"):
        cycle._validate_args(args)
    args.gate_init_noise_scale = 0
    cycle._validate_args(args)


@pytest.mark.parametrize("random_init", [True, False])
def test_install_copy_and_equivalence_policy(monkeypatch, tmp_path, random_init):
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py"])
    args = parse_args()
    args.gate_random_init = random_init
    args.gate_source = Path("manual_fresh_gate.py")
    recorded = {}

    class Session:
        model = object()
        installs = []

        def replace_source(self, source, **kwargs):
            recorded.update(kwargs)

        def freeze_except_gates(self):
            recorded["frozen"] = True

    def perturb(*args):
        assert not random_init, "Fresh mode must not perturb or access a native base"
        return []

    monkeypatch.setattr(cycle, "_perturb_gate_weights", perturb)
    monkeypatch.setattr(cycle, "_model_logits", lambda *args: torch.ones(2))
    ctx = SimpleNamespace(
        args=args, session=Session(), gate_dir=tmp_path, gate_source="SOURCE",
        sample_input=None, native_logits=torch.zeros(2),
        native_repeatable=True, native_repeat_max_abs_difference=0,
    )
    if random_init:
        cycle._install_and_verify(ctx)
        assert recorded["frozen"]
    else:
        with pytest.raises(RuntimeError, match="do not reproduce native"):
            cycle._install_and_verify(ctx)
    assert recorded["initialize_from_original"] is (not random_init)
    assert recorded["verify"] is True
    metrics = json.loads((tmp_path / "step_zero_equivalence.json").read_text())
    assert metrics["initialize_from_original"] is (not random_init)
    assert metrics["equivalent"] is False


def test_fresh_mode_still_rejects_nonfinite_model_logits(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py", "--gate-random-init"])
    ctx = SimpleNamespace(
        args=parse_args(), gate_dir=tmp_path, gate_source="SOURCE", sample_input=None,
        session=SimpleNamespace(replace_source=lambda *args, **kwargs: None, model=None),
    )
    monkeypatch.setattr(cycle, "_model_logits", lambda *args: torch.tensor([float("nan")]))
    with pytest.raises(RuntimeError, match="non-finite"):
        cycle._install_and_verify(ctx)
