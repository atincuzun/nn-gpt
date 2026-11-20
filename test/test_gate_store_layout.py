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
    "class Net(nn.Module):\n"
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
    assert record["stats"][0]["train_stat"]["train_accuracy"] is None
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


def test_running_gate_is_not_best_or_pairable(tmp_path: Path):
    """A gate killed mid-run must never win best_gate or form a Phase B pair.

    Incremental writes leave ``running: True`` on disk after each epoch.  If
    the process dies, that record stays forever.  Without the running filter,
    a gate scored on 1 of 4 epochs would beat a complete gate scored on all
    4, poison carry-forward, and corrupt Phase B pairs.
    """
    from moe_cycle.gate_pairs import build_gate_pairs
    from moe_cycle.gate_store import best_gate, write_gate_record

    store = tmp_path / "store"

    # A running gate with a lucky 1-epoch score.
    running_gate = store / "gate_000"
    running_gate.mkdir(parents=True)
    (running_gate / "gate.py").write_text(SOURCE, encoding="utf-8")
    write_gate_record(store, 0, {**_record(0, 0.9), "running": True})

    on_disk = json.loads((running_gate / "record.json").read_text(encoding="utf-8"))
    assert on_disk["running"] is True
    assert on_disk["status"] == "running"

    # A finished gate with a lower but complete score.
    finished_gate = store / "gate_001"
    finished_gate.mkdir(parents=True)
    (finished_gate / "gate.py").write_text(SOURCE, encoding="utf-8")
    write_gate_record(store, 1, _record(1, 0.5))

    # The running gate must be invisible to selection and pairing.
    best = best_gate(store)
    assert best is not None
    assert best["gate_id"] == 1
    assert best["score"]["accuracy"] == 0.5

    pairs = build_gate_pairs(store)
    running_ids = {
        gate_id
        for pair in pairs
        for gate_id in (pair["lower_gate_id"], pair["higher_gate_id"])
    }
    assert 0 not in running_ids


def test_create_gate_entry_uses_lemur_name_and_index(tmp_path: Path):
    from moe_cycle.gate_store import create_gate_entry, gate_dir, gate_library_name

    store = tmp_path / "store"
    path = create_gate_entry(
        store, 0, SOURCE, seed_name="01-additive-mlp-gelu", kind="seed",
    )
    assert path.name == gate_library_name(
        SOURCE, seed_name="01-additive-mlp-gelu", kind="seed",
    )
    assert path.name.startswith("gate-01-additive-mlp-gelu-")
    assert gate_dir(store, 0) == path
    assert (path / "gate.py").is_file()

    write_gate_record(store, 0, _record(0, 0.5))
    assert (path / "stats.json").is_file()

    second = create_gate_entry(
        store, 1, SOURCE, seed_name="01-additive-mlp-gelu", kind="seed",
    )
    assert second != path
    assert second.name.endswith("-g1")


def test_find_reusable_seed_requires_matching_protocol_and_weights(tmp_path: Path):
    from moe_cycle.gate_store import create_gate_entry, find_reusable_seed

    store = tmp_path / "store"
    path = create_gate_entry(store, 0, SOURCE, seed_name="01-additive-mlp-gelu", kind="seed")
    weights = path / "weights"
    weights.mkdir()
    (weights / "metadata.json").write_text("{}", encoding="utf-8")
    (weights / "gate_weights.pt").write_bytes(b"")
    write_gate_record(store, 0, {
        **_record(0, 0.5),
        "seeded": True,
        "llm_version": "model-v1",
        "protocol": {"fingerprint": "protocol-1"},
    })

    hit = find_reusable_seed(
        store, SOURCE, llm_version="model-v1", protocol={"fingerprint": "protocol-1"},
    )
    assert hit is not None and hit[1] == path
    assert find_reusable_seed(
        store, SOURCE, llm_version="model-v2", protocol={"fingerprint": "protocol-1"},
    ) is None
    assert find_reusable_seed(
        store, SOURCE, llm_version="model-v1", protocol={"fingerprint": "protocol-2"},
    ) is None
