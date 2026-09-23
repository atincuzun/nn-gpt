"""Tests for --gate-carry-forward weight transfer, checkpoint lookup, and CLI rules."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from moe_cycle.cycle import _latest_trained_checkpoint, _validate_args
from moe_cycle.gate_prompt import gate_proposal_prompt, gate_sft_examples
from moe_cycle.morphism import carry_gate_weights, verify_parent_function_carry


class _Gate(nn.Module):
    def __init__(self, dim: int = 4, experts: int = 3, extra: bool = True) -> None:
        super().__init__()
        self.base = nn.Linear(dim, experts, bias=False)
        if extra:
            self.correction = nn.Linear(dim, experts, bias=False)
            nn.init.zeros_(self.correction.weight)


def _fake_install(gate: nn.Module, path: str, layer_index: int):
    return SimpleNamespace(
        site=SimpleNamespace(path=path, layer_index=layer_index),
        new_gate=gate,
    )


def _write_checkpoint(root: Path, by_path: dict[str, dict[str, torch.Tensor]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    sites = []
    gate_states = {}
    for index, (path, state) in enumerate(by_path.items()):
        key = f"{index:04d}:{path}"
        gate_states[key] = {name: value.detach().clone() for name, value in state.items()}
        sites.append({"key": key, "path": path, "layer_index": index})
    (root / "metadata.json").write_text(json.dumps({"sites": sites}), encoding="utf-8")
    torch.save({"gate_states": gate_states}, root / "gate_weights.pt")
    return root


def _args(**overrides) -> Namespace:
    defaults = dict(
        gate_random_init=False,
        gate_source=None,
        gate_init_noise_scale=0.0,
        gate_outer_search=True,
        gate_author="native",
        gate_carry_forward=None,
        gate_phase_b=False,
        gate_outer_sft=False,
        gate_sft_every=10,
        gate_sft_steps=30,
        gate_sft_rank=16,
        gate_min_pairs=1,
        repetition_penalty=1.0,
        generation_backend="pipeline",
        fixed_evaluation_prompts=False,
        gate_candidates=1,
        max_length=4096,
        batch_size=1,
        gradient_checkpointing=True,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_carry_gate_weights_copies_matching_parameters(tmp_path: Path) -> None:
    parent = _Gate(extra=True)
    with torch.no_grad():
        parent.base.weight.add_(torch.ones_like(parent.base.weight))
        parent.correction.weight.add_(2.0 * torch.ones_like(parent.correction.weight))
    checkpoint = _write_checkpoint(tmp_path / "ckpt", {"blocks.0.gate": parent.state_dict()})

    successor = _Gate(extra=True)
    install = _fake_install(successor, "blocks.0.gate", 0)
    report = carry_gate_weights([install], checkpoint)

    assert torch.equal(successor.base.weight, parent.base.weight)
    assert torch.equal(successor.correction.weight, parent.correction.weight)
    assert report["n_parameters_copied"] == 2
    assert report["n_parameters_fresh"] == 0
    assert report["sites"][0]["matched_parent_by"] == "path"


def test_carry_gate_weights_keeps_new_parameters_at_constructor_init(tmp_path: Path) -> None:
    parent = _Gate(extra=False)
    checkpoint = _write_checkpoint(tmp_path / "ckpt", {"blocks.0.gate": parent.state_dict()})

    successor = _Gate(extra=True)  # adds a zero-initialized correction branch
    install = _fake_install(successor, "blocks.0.gate", 0)
    report = carry_gate_weights([install], checkpoint)

    assert torch.equal(successor.base.weight, parent.base.weight)
    assert torch.equal(successor.correction.weight, torch.zeros_like(successor.correction.weight))
    assert report["sites"][0]["parameters_copied"] == ["base.weight"]
    assert report["sites"][0]["parameters_fresh"] == ["correction.weight"]


def test_carry_gate_weights_shape_mismatch_stays_fresh(tmp_path: Path) -> None:
    parent = _Gate(dim=4, experts=3, extra=True)
    checkpoint = _write_checkpoint(tmp_path / "ckpt", {"blocks.0.gate": parent.state_dict()})

    successor = _Gate(dim=4, experts=5, extra=True)  # base output dim differs
    install = _fake_install(successor, "blocks.0.gate", 0)
    report = carry_gate_weights([install], checkpoint)

    assert successor.base.weight.shape == (5, 4)
    assert report["sites"][0]["parameters_copied"] == []
    assert report["sites"][0]["parameters_fresh"] == ["base.weight", "correction.weight"]


def test_carry_gate_weights_unmatched_site_stays_fresh(tmp_path: Path) -> None:
    parent = _Gate(extra=True)
    checkpoint = _write_checkpoint(tmp_path / "ckpt", {"other.gate": parent.state_dict()})

    successor = _Gate(extra=True)
    install = _fake_install(successor, "blocks.0.gate", 7)
    report = carry_gate_weights([install], checkpoint)

    assert report["sites"][0]["matched_parent_by"] is None
    assert report["n_parameters_copied"] == 0
    assert report["n_parameters_fresh"] == 2


@pytest.mark.parametrize("shared_path", ["", "blocks.0.gate"])
def test_carry_gate_weights_duplicate_paths_fall_back_to_layer_index(
    tmp_path: Path, shared_path: str
) -> None:
    """Two sites recording one path must not share one layer's trained weights.

    Some discovery modes leave ``site.path`` empty or duplicated; a by-path
    lookup would overwrite one layer's checkpoint entry with the next and
    hand both layers the same trained weights. Layer-index matching must
    take over instead.
    """
    parent0, parent1 = _Gate(extra=False), _Gate(extra=False)
    with torch.no_grad():
        parent0.base.weight.add_(1.0)
        parent1.base.weight.add_(2.0)
    root = tmp_path / "ckpt"
    root.mkdir(parents=True)
    sites, gate_states = [], {}
    for index, parent in enumerate((parent0, parent1)):
        key = f"{index:04d}:{shared_path}"
        gate_states[key] = parent.state_dict()
        sites.append({"key": key, "path": shared_path, "layer_index": index})
    (root / "metadata.json").write_text(json.dumps({"sites": sites}), encoding="utf-8")
    torch.save({"gate_states": gate_states}, root / "gate_weights.pt")

    successor0, successor1 = _Gate(extra=False), _Gate(extra=False)
    installs = [
        _fake_install(successor0, shared_path, 0),
        _fake_install(successor1, shared_path, 1),
    ]
    report = carry_gate_weights(installs, root)

    assert torch.equal(successor0.base.weight, parent0.base.weight)
    assert torch.equal(successor1.base.weight, parent1.base.weight)
    assert [entry["matched_parent_by"] for entry in report["sites"]] == [
        "layer_index",
        "layer_index",
    ]


def test_carry_gate_weights_casts_dtype(tmp_path: Path) -> None:
    parent = _Gate(extra=False)
    checkpoint = _write_checkpoint(tmp_path / "ckpt", {"blocks.0.gate": parent.state_dict()})

    successor = _Gate(extra=False).to(torch.bfloat16)
    install = _fake_install(successor, "blocks.0.gate", 0)
    carry_gate_weights([install], checkpoint)

    assert successor.base.weight.dtype == torch.bfloat16
    assert torch.equal(
        successor.base.weight.float(),
        parent.base.weight.to(torch.bfloat16).float(),
    )


def test_latest_trained_checkpoint_picks_highest_epoch(tmp_path: Path) -> None:
    gate_root = tmp_path / "gates"
    for epoch in (0, 2):
        (gate_root / "gate_003" / "epochs" / f"A{epoch}" / "gate_post_train").mkdir(parents=True)
    (gate_root / "gate_003" / "epochs" / "A2" / "metrics.json").write_text("{}", encoding="utf-8")
    (gate_root / "gate_003" / "epochs" / "A5").mkdir(parents=True)  # ran out before saving

    checkpoint = _latest_trained_checkpoint(gate_root, 3)
    assert checkpoint == gate_root / "gate_003" / "epochs" / "A2" / "gate_post_train"
    assert _latest_trained_checkpoint(gate_root, 4) is None


def test_validate_args_carry_forward_requires_outer_search() -> None:
    with pytest.raises(ValueError, match="--gate-outer-search"):
        _validate_args(_args(gate_carry_forward="best", gate_outer_search=False))


def test_validate_args_carry_forward_conflicts_with_gate_author() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        _validate_args(_args(gate_carry_forward="best", gate_author="last"))
    # Matching selection modes (or the default) are accepted.
    _validate_args(_args(gate_carry_forward="best", gate_author="best"))
    _validate_args(_args(gate_carry_forward="best"))


def test_validate_args_carry_forward_rejects_init_noise() -> None:
    with pytest.raises(ValueError, match="gate-init-noise-scale"):
        _validate_args(_args(gate_carry_forward="last", gate_init_noise_scale=0.1))


def test_gate_prompt_inherit_mode_extends_reference() -> None:
    prompt = gate_proposal_prompt(
        [(4, 3)],
        reference_source="class LLMGeneratedGate:\n    pass",
        reference_accuracy=0.4,
        goal_accuracy=0.4,
        dataset="cifar-10",
        inherit_reference=True,
    )
    assert "extending it" in prompt
    assert "parameter name" in prompt
    assert "zero-initialize" in prompt
    assert "rather than copying" not in prompt
    assert "Current gate code:" in prompt

    plain = gate_proposal_prompt(
        [(4, 3)],
        reference_source="class LLMGeneratedGate:\n    pass",
        reference_accuracy=0.4,
        inherit_reference=False,
    )
    assert "rather than copying" in plain
    assert "Baseline gate code:" in plain


def test_gate_sft_examples_inherit_mode_matches() -> None:
    pair = {
        "pair_id": 1,
        "lower_source": "class A:\n    pass",
        "lower_accuracy": 0.3,
        "higher_source": "class B:\n    pass",
        "higher_accuracy": 0.4,
        "dataset": "cifar-10",
    }
    example = gate_sft_examples([pair], [(4, 3)], inherit_reference=True)[0]
    assert "rather than copying" not in example["messages"][1]["content"]
    assert "extending it" in example["messages"][1]["content"]


def test_verify_parent_function_carry() -> None:
    parent = torch.zeros(2, 5)
    exact = verify_parent_function_carry(parent, parent.clone())
    assert exact["equivalent"] and exact["bit_exact"]

    close = verify_parent_function_carry(parent, parent + 1e-6)
    assert close["equivalent"] and not close["bit_exact"]

    far = verify_parent_function_carry(parent, parent + 1.0)
    assert not far["equivalent"]
    assert far["max_abs_difference"] == pytest.approx(1.0)
