"""Tests for the simplified gate store layout and project archive."""

import json
from pathlib import Path
from types import SimpleNamespace

from moe_cycle.records import _write_gate_record
from moe_cycle.gate_store import (
    load_gate_summaries,
    write_gate_record,
)

SOURCE = (
    "import torch\n"
    "import torch.nn as nn\n\n"
    "class LLMGeneratedGate(nn.Module):\n"
    "    def __init__(self, model_dim, num_experts):\n"
    "        super().__init__()\n"
    "        self.base = nn.Linear(model_dim, num_experts, bias=False)\n"
    "    def forward(self, x):\n"
    "        return self.base(x)\n"
)


def _epoch(accuracy):
    return {
        "epoch": 0,
        "cv_accuracy_mean": accuracy,
        "cv_accuracy_best": accuracy,
        "n_measured": 1 if accuracy is not None else 0,
        "n_attempted": 1,
        "n_trained": 1 if accuracy is not None else 0,
        "n_generated": 1,
        "train_loss": 0.5,
        "validation_loss": 0.4,
        "active_gate_layers": [1],
        "cycle_success": True,
        "duration": 1000,
        "synth_nn": "epochs/A0/synth_nn",
    }


def _record(gate_id, accuracy):
    return {
        "gate_id": gate_id,
        "gate_code": SOURCE,
        "structural_hash": "a" * 64,
        "task": "img-classification",
        "dataset": "cifar-10",
        "metric": "accuracy",
        "score": {
            "objective": "mean_of_epoch_means_v1",
            "accuracy": accuracy,
            "eligible": accuracy is not None,
            "n_measured": 1 if accuracy is not None else 0,
            "n_attempted": 1,
        },
        "epochs": [_epoch(accuracy)],
    }


def test_write_gate_record_is_the_only_gate_json(tmp_path: Path):
    store = tmp_path / "store"
    gate = store / "gate_000"
    gate.mkdir(parents=True)
    (gate / "gate.py").write_text(SOURCE, encoding="utf-8")

    write_gate_record(store, 0, _record(0, 0.5), run_root=tmp_path / "run", model="m")

    record = json.loads((gate / "record.json").read_text(encoding="utf-8"))
    assert record["schema_version"] == 2
    assert record["nn"]["code"] == SOURCE
    assert record["nn"]["id"]
    assert record["stats"][0]["accuracy"] == 0.5
    assert record["stats"][0]["model"] == record["nn"]["name"]
    assert record["stats"][0]["train_stat"]["train_accuracy"] == 0.5
    assert not (gate / "summary.json").exists()


def test_load_gate_summaries_hydrates_code_from_gate_py(tmp_path: Path):
    store = tmp_path / "store"
    gate = store / "gate_010"
    gate.mkdir(parents=True)
    (gate / "gate.py").write_text(SOURCE, encoding="utf-8")
    write_gate_record(store, 10, _record(10, 0.4))

    loaded = load_gate_summaries(store)
    assert [item["gate_id"] for item in loaded] == [10]
    assert loaded[0]["gate_code"] == SOURCE.rstrip("\n")


def test_load_gate_summaries_orders_numerically(tmp_path: Path):
    store = tmp_path / "store"
    for gate_id in (2, 10):
        gate = store / f"gate_{gate_id:03d}"
        gate.mkdir(parents=True)
        (gate / "gate.py").write_text(SOURCE, encoding="utf-8")
        write_gate_record(store, gate_id, _record(gate_id, 0.4))

    assert [item["gate_id"] for item in load_gate_summaries(store)] == [2, 10]


def test_measured_gate_is_archived_with_lemur_shaped_record(tmp_path: Path):
    store = tmp_path / "store"
    gate = store / "gate_000"
    gate.mkdir(parents=True)
    (gate / "gate.py").write_text(SOURCE, encoding="utf-8")
    run = tmp_path / "out" / "run_001"
    ctx = SimpleNamespace(
        args=SimpleNamespace(
            model="test/model",
            gate_author="best",
            gate_archive_dir=tmp_path / "archive",
        ),
        candidate_index=0,
        candidate_root=gate,
        gate_root=store,
        run_root=run,
        gate_source=SOURCE,
        gate_epoch_metrics=[_epoch(0.42)],
        install_metrics={"step_zero": {"equivalent": True}},
        proposal_prompt="proposal text",
        reference_accuracy=0.3,
        goal_accuracy=0.5,
        created_at="2026-01-01T00:00:00+00:00",
        author_gate_id=None,
    )

    _write_gate_record(ctx)

    archived = tmp_path / "archive" / "test-model" / "run_001" / "gate_000"
    assert (archived / "gate.py").read_text(encoding="utf-8") == SOURCE
    record = json.loads((archived / "record.json").read_text(encoding="utf-8"))
    assert record["score"]["accuracy"] == 0.42
    assert record["install"]["step_zero"]["equivalent"] is True
    assert record["proposal"]["prompt"] == "proposal text"
    assert record["stats"][0]["accuracy"] == 0.42


def test_unmeasured_gate_is_not_archived(tmp_path: Path):
    store = tmp_path / "store"
    gate = store / "gate_000"
    gate.mkdir(parents=True)
    (gate / "gate.py").write_text(SOURCE, encoding="utf-8")
    ctx = SimpleNamespace(
        args=SimpleNamespace(
            model="test/model",
            gate_author="best",
            gate_archive_dir=tmp_path / "archive",
        ),
        candidate_index=0,
        candidate_root=gate,
        gate_root=store,
        run_root=tmp_path / "out" / "run_001",
        gate_source=SOURCE,
        gate_epoch_metrics=[_epoch(None)],
        created_at="2026-01-01T00:00:00+00:00",
    )

    _write_gate_record(ctx)

    assert (store / "gate_000" / "record.json").is_file()
    assert not (tmp_path / "archive").exists()
