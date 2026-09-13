import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from moe_cycle.cli import parse_args
from moe_cycle.gate_benchmark import (
    BENCHMARK_PREFIX,
    adapter_identity,
    benchmark_prefix,
    file_hash,
    write_json,
)
from moe_cycle.gate_prompt import (
    GATE_SYSTEM_PROMPT,
    gate_prompt_scope,
    gate_proposal_prompt,
    gate_sft_examples,
)
from moe_cycle.gate_proposer import tokenize_gate_examples
from moe_cycle.gate_store import GateStore, fingerprint, source_hash, summarize_scores
from moe_cycle.outer_loop import validate_outer_args


GATE_SOURCE = (
    "import torch\nimport torch.nn as nn\n\n"
    "class LLMGeneratedGate(nn.Module):\n"
    "    def __init__(self, model_dim: int, num_experts: int):\n"
    "        super().__init__()\n"
    "        self.base = nn.Linear(model_dim, num_experts, bias=False)\n\n"
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        return self.base(x)\n"
)

RESIDUAL_VARIANT = GATE_SOURCE.replace(
    "        return self.base(x)\n",
    "        return self.base(x) * 1.0\n",
)


def _outcomes(*pairs):
    return [
        {"candidate": f"B{index}", "status": status, "accuracy": accuracy}
        for index, (status, accuracy) in enumerate(pairs)
    ]


def _protocol():
    protocol = {"settings": {"version": 1}, "manifest_hash": "abc"}
    return {"protocol_id": fingerprint(protocol), "protocol": protocol}


def _record(protocol, *, trial_id, source=GATE_SOURCE, outcomes=None):
    outcomes = outcomes if outcomes is not None else _outcomes(("measured", 0.4))
    score = summarize_scores(outcomes, min_success_rate=0.5, min_measured=1)
    return {
        "trial_id": trial_id,
        "source": source,
        "protocol_id": protocol["protocol_id"],
        "protocol": protocol["protocol"],
        "outcomes": outcomes,
        "score": score,
    }


def test_summarize_scores_averages_only_measured():
    score = summarize_scores(
        _outcomes(("measured", 0.4), ("measured", 0.6), ("unavailable", None)),
        min_success_rate=0.5,
        min_measured=1,
    )
    assert score["mean_accuracy"] == pytest.approx(0.5)
    assert score["n_measured"] == 2
    assert score["success_rate"] == pytest.approx(2 / 3)
    assert score["incomplete"] is True
    assert score["eligible"] is False


def test_summarize_scores_excludes_invalid_from_mean():
    score = summarize_scores(
        _outcomes(("measured", 0.9), ("invalid", None), ("invalid", None)),
        min_success_rate=0.3,
        min_measured=1,
    )
    assert score["mean_accuracy"] == pytest.approx(0.9)
    assert score["success_rate"] == pytest.approx(1 / 3)
    assert score["incomplete"] is False
    assert score["eligible"] is True


def test_summarize_scores_rejects_out_of_range_accuracy():
    with pytest.raises(ValueError):
        summarize_scores(_outcomes(("measured", 1.5)), min_success_rate=0.5, min_measured=1)


def test_source_hash_ignores_formatting_but_not_semantics():
    reformatted = GATE_SOURCE.replace("model_dim: int", "model_dim:int")
    assert source_hash(GATE_SOURCE) == source_hash(reformatted)
    changed = GATE_SOURCE.replace("bias=False", "bias=True")
    assert source_hash(GATE_SOURCE) != source_hash(changed)


def test_store_roundtrip_and_protocol_isolation(tmp_path: Path):
    store = GateStore(tmp_path / "archive.sqlite3")
    protocol = _protocol()
    record = store.add(_record(protocol, trial_id="t1"))
    assert record["architecture_id"] == source_hash(GATE_SOURCE)
    assert [r["trial_id"] for r in store.records(protocol["protocol_id"])] == ["t1"]
    assert store.records("different-protocol") == []


def test_store_rejects_score_outcomes_mismatch(tmp_path: Path):
    store = GateStore(tmp_path / "archive.sqlite3")
    bad = _record(_protocol(), trial_id="t1")
    bad["score"] = dict(bad["score"], mean_accuracy=0.99)
    with pytest.raises(ValueError):
        store.add(bad)


def test_store_rejects_protocol_fingerprint_mismatch(tmp_path: Path):
    store = GateStore(tmp_path / "archive.sqlite3")
    bad = _record(_protocol(), trial_id="t1")
    bad["protocol_id"] = "not-the-fingerprint"
    with pytest.raises(ValueError):
        store.add(bad)


def test_subset_averages_repeats_and_disqualifies_ineligible(tmp_path: Path):
    store = GateStore(tmp_path / "archive.sqlite3")
    protocol = _protocol()
    for accuracy in (0.4, 0.6):
        store.add(
            _record(
                protocol,
                trial_id=f"gate-a-{accuracy}",
                outcomes=_outcomes(("measured", accuracy)),
            )
        )
    store.add(
        _record(
            protocol,
            trial_id="gate-b-ok",
            source=RESIDUAL_VARIANT,
            outcomes=_outcomes(("measured", 0.5)),
        )
    )
    store.add(
        _record(
            protocol,
            trial_id="gate-b-fail",
            source=RESIDUAL_VARIANT,
            outcomes=_outcomes(("unavailable", None)),
        )
    )
    selected = store.training_subset(
        protocol["protocol_id"], max_examples=5, top_fraction=1.0, seed=7
    )
    assert len(selected) == 1
    assert sorted(selected[0]["selection_trials"]) == ["gate-a-0.4", "gate-a-0.6"]
    assert selected[0]["selection_accuracy"] == pytest.approx(0.5)
    assert "gate-b" not in "".join(selected[0]["selection_trials"])


def test_subset_only_contains_eligible_architectures(tmp_path: Path):
    store = GateStore(tmp_path / "archive.sqlite3")
    protocol = _protocol()
    store.add(
        _record(
            protocol,
            trial_id="weak",
            outcomes=_outcomes(("measured", 0.3), ("unavailable", None)),
        )
    )
    assert (
        store.training_subset(
            protocol["protocol_id"], max_examples=4, top_fraction=1.0, seed=0
        )
        == []
    )


def test_gate_sft_examples_do_not_leak_target_or_score():
    record = {
        "trial_id": "t1",
        "architecture_id": "a1",
        "selection_trials": ["t1"],
        "selection_accuracy": 0.7,
        "source": GATE_SOURCE,
    }
    example = gate_sft_examples([record], shapes=[(8, 4)])[0]
    user_prompt = example["messages"][1]["content"]
    assistant = example["messages"][2]["content"]
    assert example["messages"][0]["content"] == GATE_SYSTEM_PROMPT
    # The user prompt states the generic contract and output format but must not
    # contain the target implementation or the quality signal that selected it.
    assert GATE_SOURCE.strip() not in user_prompt
    assert "return self.base(x)" not in user_prompt
    assert "0.7" not in user_prompt
    assert assistant == "<gate>\n" + GATE_SOURCE.strip() + "\n</gate>"
    assert "<gate>" in gate_proposal_prompt([(8, 4)])


def test_gate_prompt_scope_restores_previous_system_prompt():
    chatbot = SimpleNamespace(system_prompt="cv-system")
    with gate_prompt_scope(chatbot):
        assert chatbot.system_prompt == GATE_SYSTEM_PROMPT
    assert chatbot.system_prompt == "cv-system"


class _CharTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        return text + ("|assistant:" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=True, **_kwargs):
        return {"input_ids": [ord(char) for char in text]}


def _single_example():
    return gate_sft_examples(
        [
            {
                "trial_id": "t1",
                "architecture_id": "a1",
                "selection_trials": ["t1"],
                "selection_accuracy": 0.7,
                "source": GATE_SOURCE,
            }
        ],
        shapes=[(8, 4)],
    )[0]


def test_tokenize_gate_examples_masks_prompt_and_keeps_full_source():
    rows = tokenize_gate_examples(_CharTokenizer(), [_single_example()], max_length=100000)
    assert len(rows) == 1
    labels = rows[0]["labels"]
    assert -100 in labels and any(label != -100 for label in labels)
    target = "".join(chr(code) for code in labels if code != -100)
    match = re.search(r"<gate>\s*([\s\S]*?)\s*</gate>", target)
    assert match is not None
    assert source_hash(match.group(1)) == source_hash(GATE_SOURCE)


def test_tokenize_gate_examples_skips_oversized_source():
    assert tokenize_gate_examples(_CharTokenizer(), [_single_example()], max_length=50) == []


def test_write_json_is_atomic_and_readable(tmp_path: Path):
    target = tmp_path / "nested" / "result.json"
    write_json(target, {"value": 1})
    assert json.loads(target.read_text()) == {"value": 1}
    assert not target.with_suffix(".json.tmp").exists()


def test_adapter_identity_hashes_adapter_files_only(tmp_path: Path):
    assert adapter_identity(None) is None
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("ignore")
    identity = adapter_identity(tmp_path)
    assert identity == {"adapter_config.json": file_hash(tmp_path / "adapter_config.json")}


def _outer_args(**overrides):
    args = SimpleNamespace(
        gate_outer_sft=True,
        gate_source=None,
        gate_proposer_checkpoint=None,
        epochs=1,
        gate_train_steps=1,
        nn_train_epochs=1,
        test_nn=1,
        gate_benchmark_size=1,
        gate_min_measured=1,
        gate_proposer_steps=1,
        gate_proposer_rank=1,
        gate_proposer_max_length=8,
        gate_proposer_batch_size=1,
        gate_proposer_max_examples=2,
        gate_proposer_min_examples=1,
        gate_min_success_rate=0.5,
        gate_proposer_top_fraction=0.5,
        gate_proposer_learning_rate=1e-4,
        gate_benchmark_seeds=[1, 2],
        fixed_eval_hyperparameters={"batch": 1},
        gate_benchmark=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_validate_outer_args_accepts_valid_configuration():
    validate_outer_args(_outer_args())


@pytest.mark.parametrize(
    "overrides",
    [
        {"gate_source": Path("gate.py")},
        {"gate_min_success_rate": 1.5},
        {"gate_proposer_top_fraction": 0.0},
        {"fixed_eval_hyperparameters": None},
        {"gate_benchmark_seeds": [1, 1]},
        {"gate_proposer_min_examples": 5},
    ],
)
def test_validate_outer_args_rejects_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        validate_outer_args(_outer_args(**overrides))


def test_validate_outer_args_checkpoint_requires_outer_sft():
    with pytest.raises(ValueError):
        validate_outer_args(
            _outer_args(gate_outer_sft=False, gate_proposer_checkpoint=Path("adapter"))
        )


def test_benchmark_prefix_is_isolated_and_detects_collisions():
    args = SimpleNamespace(sft_nn_prefixes=["ga-", "moe-gate-cycle"],
                           generation_nn_prefixes=["ga-"])
    assert benchmark_prefix(args) == BENCHMARK_PREFIX
    colliding = SimpleNamespace(sft_nn_prefixes=["gate-"], generation_nn_prefixes=[])
    with pytest.raises(ValueError):
        benchmark_prefix(colliding)


def test_cli_exposes_outer_sft_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py"])
    args = parse_args()
    assert args.gate_outer_sft is False
    assert args.gate_min_success_rate == 0.5
    assert args.gate_proposer_checkpoint is None
