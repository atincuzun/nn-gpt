"""Regression tests for install-time mode hygiene and pair integrity.

Reproduces the 2026-09-23 run failure: after a candidate's gate training,
detached native gates keep stale eval flags while the model stays in train
mode; the next install copies the stale flag, and the verify forward dies in
DeepSeek's AddAuxiliaryLoss ("'NoneType' object has no attribute 'numel'").
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from moe_cycle.gate_pairs import build_gate_pairs
from moe_gate_only.verify import _verify_forward


class _MockRouter(nn.Module):
    """Mirrors the DeepSeek MoEGate contract: aux loss only in train mode."""

    def __init__(self, dim: int = 4, experts: int = 3):
        super().__init__()
        self.alpha = 0.001
        self.lin = nn.Linear(dim, experts, bias=False)

    def forward(self, x):
        logits = self.lin(x)
        topk_weight, topk_idx = torch.topk(logits, 2, dim=-1)
        aux = logits.sum() * self.alpha if self.training else None
        return topk_idx, topk_weight, aux


class _MockBlock(nn.Module):
    """Mirrors the MoE block's unguarded AddAuxiliaryLoss apply."""

    def __init__(self):
        super().__init__()
        self.gate = _MockRouter()

    def forward(self, x):
        _topk_idx, _topk_weight, aux = self.gate(x)
        if self.training:
            # Mirrors AddAuxiliaryLoss.forward's `loss.numel()` on None.
            _ = aux.numel()
        return torch.zeros_like(x)


class _MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _MockBlock()

    def forward(self, input_ids=None, **kwargs):
        return self.block(torch.randn(2, 4))


def test_mock_reproduces_the_train_block_eval_gate_crash():
    model = _MockModel()
    model.train()
    # Replacement attached AFTER the train sweep, carrying the stale
    # eval flag of a detached native gate.
    model.block.gate = _MockRouter().eval()
    assert model.block.training is True
    assert model.block.gate.training is False
    with torch.no_grad():
        try:
            model(torch.zeros(1, 4))
        except AttributeError as exc:
            assert "numel" in str(exc)
        else:
            raise AssertionError("mock failed to reproduce the aux-loss crash")


def test_verify_forward_succeeds_in_dirty_mode_state_and_restores_mode():
    model = _MockModel()
    model.train()
    model.block.gate = _MockRouter().eval()  # stale-flag replacement installed

    errors: list[BaseException] = []
    ok = _verify_forward(
        model, [SimpleNamespace()], sample_input=None, errors=errors,
    )

    assert ok is True
    assert errors == []
    assert model.training is True  # mode restored after the smoke pass


def _write_gate(root: Path, gate_id: int, structural_hash: str, accuracy: float):
    gate_dir = root / f"gate_{gate_id:03d}"
    gate_dir.mkdir(parents=True)
    (gate_dir / "summary.json").write_text(
        json.dumps({
            "gate_id": gate_id,
            "gate_code": f"# gate {gate_id}",
            "structural_hash": structural_hash,
            "task": "img-classification",
            "dataset": "cifar-10",
            "metric": "accuracy",
            "score": {"eligible": True, "accuracy": accuracy},
        }),
        encoding="utf-8",
    )


def test_build_gate_pairs_skips_same_architecture_reruns(tmp_path: Path):
    _write_gate(tmp_path, 0, "hash-a", 0.30)
    _write_gate(tmp_path, 1, "hash-a", 0.40)  # rerun of gate 0's architecture
    _write_gate(tmp_path, 2, "hash-b", 0.35)

    pairs = build_gate_pairs(tmp_path)

    pair_ids = {pair["pair_id"] for pair in pairs}
    # Gates 0 (0.30, hash-a) and 1 (0.40, hash-a) are reruns of one
    # architecture and must never pair; both pair with gate 2 (0.35, hash-b).
    assert pair_ids == {"000->002", "002->001"}
    for pair in pairs:
        assert pair["lower_source"] != pair["higher_source"]
