"""Tests for OOM recovery and failure diagnosability in the gate cycle.

Covers the fixes for run 20260922_222052, where a gate-training OOM discarded
a scoring candidate (gate 031, four epochs of rising accuracy) and the
fragmented reserve then rolled back the next installs as opaque "failed
verification" records.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

import ab.gpt.util.Const as Const
import ab.gpt.util.Tune as Tune
import moe_gate_only
from moe_cycle.cycle import (
    _record_failed_candidate,
    _release_cuda_memory,
    _run_epochs,
)
from moe_gate_only.verify import _verify_forward


class _FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True)

    def eval(self):
        return self

    def train(self):
        return self


class _BoomModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # A parameter is required: _verify_forward treats parameterless
        # modules as unverifiable and returns True without a forward pass.
        self.unused = nn.Linear(4, 4)

    def forward(self, *args, **kwargs):
        raise ValueError("kaboom")


class _OkModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x.float())


def test_verify_forward_reports_underlying_exception():
    errors: list[BaseException] = []
    ok = _verify_forward(
        _BoomModel(), [SimpleNamespace()], sample_input=None, errors=errors,
    )
    assert ok is False
    assert errors and "kaboom" in str(errors[-1])


def test_verify_forward_records_nothing_when_forward_works():
    errors: list[BaseException] = []
    ok = _verify_forward(
        _OkModel(), [SimpleNamespace()], sample_input=None, errors=errors,
    )
    assert ok is True
    assert errors == []


def test_release_cuda_memory_is_safe_without_cuda():
    _release_cuda_memory()


def test_record_failed_candidate_keeps_partial_epochs(tmp_path: Path):
    partial = {
        "epoch": 0,
        "cv_accuracy_mean": 0.4,
        "cv_accuracy_best": 0.5,
        "n_measured": 2,
        "n_attempted": 2,
        "n_trained": 2,
        "n_generated": 2,
        "mean_accuracy_delta": None,
        "train_loss": 0.5,
        "validation_loss": None,
        "active_gate_layers": [3],
        "cycle_success": True,
        "epoch_path": "/removed/run",
    }
    ctx = SimpleNamespace(
        args=Namespace(gate_carry_forward=None, gate_author="native"),
        gate_source="import torch\n",
        reference_gate_id=None,
        author_gate_id=None,
        carry_source_id=None,
        carry_forward_metrics=None,
        rematch_for=None,
        llm_version="base-x",
        seeded=False,
        gate_root=tmp_path,
        gate_epoch_metrics=[partial],
    )

    _record_failed_candidate(ctx, 7, RuntimeError("late pipeline failure"))

    record = json.loads((tmp_path / "gate_007" / "summary.json").read_text())
    assert "late pipeline failure" in record["failure"]
    assert record["score"]["eligible"] is False
    assert record["epochs"] == [partial]


def test_run_epochs_harvests_partial_measurements_on_oom(
    tmp_path: Path, monkeypatch,
):
    """A gate-training OOM stops the candidate but keeps its measured epochs."""
    nngpt = tmp_path / "nngpt"
    nngpt.mkdir()
    train_conf = tmp_path / "conf_train"
    train_conf.mkdir()
    (train_conf / "NN_gen.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Const, "nngpt_dir", nngpt)
    monkeypatch.setattr(Const, "conf_train_dir", train_conf)
    monkeypatch.setattr(Const, "synth_dir", lambda epoch_path: Path(epoch_path) / "synth")

    def fake_nn_gen(*args, **kwargs):
        return None

    def fake_evaluate_epoch(epoch, epoch_path, *args, **kwargs):
        (nngpt / "cycle_results.json").write_text(
            json.dumps({"success": True, "evaluation": {"models_trained": 2}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(Tune, "nn_gen", fake_nn_gen)
    monkeypatch.setattr(Tune, "_evaluate_epoch", fake_evaluate_epoch)
    monkeypatch.setattr(
        moe_gate_only, "build_nngenprompt_dataloaders",
        lambda *a, **k: (None, None, [{"input_ids": [1] * 8}, {"input_ids": [1] * 9}]),
    )
    monkeypatch.setattr(moe_gate_only, "collect_gate_metrics", lambda installs: {})

    training_calls = {"count": 0}

    def fake_train_gates(model, installs, loader, **kwargs):
        training_calls["count"] += 1
        if training_calls["count"] >= 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.0 GiB")
        return SimpleNamespace(steps=2, mean_train_loss=0.5), None

    monkeypatch.setattr(moe_gate_only, "train_gates", fake_train_gates)

    args = Namespace(
        epochs=3,
        seed=42,
        progressive_unfreeze_descending=False,
        gate_implementation="llm_generated",
        fixed_eval_hyperparameters=None,
        nn_train_epochs=1,
        conf_keys=["improve_classification_only"],
        dataset="cifar-10",
        test_nn=2,
        generation_max_new_tokens=64,
        nn_name_prefix="moe-gate-cycle",
        generation_max_input_length=None,
        train_prompt_config="NN_gen.json",
        test_prompt_config="NN_gen.json",
        max_length=128,
        max_prompts=8,
        batch_size=1,
        validation_fraction=0.1,
        gate_train_steps=2,
        gate_learning_rate=1e-4,
        gradient_checkpointing=False,
        validation_steps=1,
        gate_outer_search=True,
        gate_random_init=False,
        sft_nn_prefixes=["moe-gate-cycle"],
        generation_nn_prefixes=["moe-gate-cycle"],
    )
    session = SimpleNamespace(
        model=_FakeModel(),
        installs=[SimpleNamespace(site=SimpleNamespace(layer_index=3))],
        save=lambda *a, **k: None,
        gate_source="import torch\n",
    )
    ctx = SimpleNamespace(
        args=args,
        session=session,
        chat_bot=object(),
        tokenizer=object(),
        candidate_root=tmp_path / "cand",
        gate_root=tmp_path / "store",
        candidate_index=0,
        gate_epoch_metrics=[],
        previous_feedback_summary="",
        used_prompts=[],
        prompt_dict={"improve_classification_only": {"prompt": "p"}},
    )
    (tmp_path / "cand").mkdir()

    epoch_paths = _run_epochs(ctx)

    # Epoch 0 trained normally; epoch 1 OOM'd and the loop stopped there —
    # epoch 2 never ran.
    assert training_calls["count"] == 2
    assert [path.name for path in epoch_paths] == ["A0", "A1"]
    assert [metrics["epoch"] for metrics in ctx.gate_epoch_metrics] == [0, 1]
    assert ctx.gate_epoch_metrics[0]["train_loss"] == 0.5
    assert ctx.gate_epoch_metrics[1]["train_loss"] is None

    oom_metrics = json.loads(
        (epoch_paths[1] / "gate_training_metrics.json").read_text()
    )
    assert oom_metrics["oom_recovered"] is True
    assert oom_metrics["mean_train_loss"] is None
    assert (tmp_path / "store" / "gate_000" / "epoch_01" / "metrics.json").is_file()
