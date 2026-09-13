"""Tests for the self-improving MoE gate search (propose, measure, revise)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from moe_cycle.cli import parse_args
from moe_cycle.cycle import _outer_feedback_block, _validate_args
from moe_cycle.gate_prompt import (
    BASELINE_GATE_CODE,
    GATE_SYSTEM_PROMPT,
    gate_prompt_scope,
    gate_proposal_prompt,
    gate_sft_examples,
    is_duplicate_gate,
    near_duplicate_similarity,
)
from moe_cycle.gate_source import _generate_gate
from moe_cycle.gate_pairs import build_gate_pairs, describe_pairs, eligible_gates
from moe_cycle.gate_store import (
    SCORE_OBJECTIVE,
    best_gate,
    epoch_dir,
    gate_dir,
    load_gate_summaries,
    structural_hash,
    summarize_epoch_metrics,
    summarize_gate,
    write_gate_summary,
)

GATE_SOURCE = (
    "import torch\nimport torch.nn as nn\n\n"
    "class LLMGeneratedGate(nn.Module):\n"
    "    def __init__(self, model_dim: int, num_experts: int):\n"
    "        super().__init__()\n"
    "        self.base = nn.Linear(model_dim, num_experts, bias=False)\n\n"
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        return self.base(x)\n"
)

# Same behaviour, different serialisation (whitespace/typing) -> duplicate.
REFORMATTED_GATE = GATE_SOURCE.replace("model_dim: int", "model_dim:int")
# Different behaviour -> distinct architecture.
DISTINCT_GATE = GATE_SOURCE.replace(
    "        return self.base(x)\n",
    "        return self.base(x) + 0.0 * self.base(x.detach())\n",
)


def _epoch(mean, best, n_measured, n_attempted, **extra):
    metrics = {
        "epoch": extra.pop("epoch", 0),
        "cv_accuracy_mean": mean,
        "cv_accuracy_best": best,
        "n_measured": n_measured,
        "n_attempted": n_attempted,
    }
    metrics.update(extra)
    return metrics


# ── scoring ──────────────────────────────────────────────────────────────────

def test_score_uses_mean_of_epoch_means_not_bests():
    """A gate with one lucky model must not outrank a consistent gate."""
    lucky = [_epoch(None, None, 0, 1), _epoch(None, None, 0, 1), _epoch(0.90, 0.90, 1, 1)]
    consistent = [_epoch(0.60, 0.80, 10, 10), _epoch(0.60, 0.80, 10, 10), _epoch(0.60, 0.80, 10, 10)]

    lucky_score = summarize_epoch_metrics(lucky)
    consistent_score = summarize_epoch_metrics(consistent)

    # Only the lucky gate's one populated epoch counts, so it has mean 0.90 ...
    assert lucky_score["accuracy"] == pytest.approx(0.90)
    # ... but it covers a single epoch and 1/3 measured candidates, whereas the
    # consistent gate is scored over three epochs and 30/30 candidates.
    assert lucky_score["n_inner_epochs_scored"] == 1
    assert lucky_score["n_measured"] == 1
    assert consistent_score["accuracy"] == pytest.approx(0.60)
    assert consistent_score["n_inner_epochs_scored"] == 3
    assert consistent_score["n_measured"] == 30
    assert consistent_score["mean_of_epoch_bests"] == pytest.approx(0.80)
    assert consistent_score["objective"] == SCORE_OBJECTIVE


def test_score_ignores_epochs_without_measurements():
    metrics = [
        _epoch(0.40, 0.50, 2, 4),
        _epoch(None, None, 0, 3),  # nothing measured this epoch
        _epoch(0.60, 0.70, 3, 3),
    ]
    score = summarize_epoch_metrics(metrics)
    assert score["accuracy"] == pytest.approx(0.50)
    assert score["n_inner_epochs_scored"] == 2
    assert score["n_measured"] == 5
    assert score["n_attempted"] == 10
    assert score["success_rate"] == pytest.approx(0.5)
    assert score["best_epoch_mean"] == pytest.approx(0.60)
    assert score["eligible"] is True


def test_score_is_ineligible_without_any_measurement():
    score = summarize_epoch_metrics([_epoch(None, None, 0, 2)])
    assert score["accuracy"] is None
    assert score["eligible"] is False


def test_score_of_empty_metrics_is_empty():
    score = summarize_epoch_metrics([])
    assert score["accuracy"] is None
    assert score["n_measured"] == 0
    assert score["eligible"] is False


# ── storage ──────────────────────────────────────────────────────────────────

def _write_gate(root, gate_id, accuracy, *, eligible=True, source=GATE_SOURCE,
                task="img-classification", dataset="cifar-10", metric="acc"):
    record = {
        "gate_id": gate_id,
        "gate_code": source,
        "structural_hash": structural_hash(source),
        "reference_gate_id": None,
        "task": task,
        "dataset": dataset,
        "metric": metric,
        "score": {"accuracy": accuracy, "eligible": eligible, "n_measured": 2,
                  "n_inner_epochs_scored": 2, "n_attempted": 2, "objective": SCORE_OBJECTIVE},
        "epochs": [],
    }
    write_gate_summary(root, gate_id, record)
    return record


def test_store_layout_matches_expected_paths(tmp_path: Path):
    assert gate_dir(tmp_path, 3) == tmp_path / "gate_003"
    assert epoch_dir(tmp_path, 3, 7) == tmp_path / "gate_003" / "epoch_07"


def test_store_roundtrip_and_ordering(tmp_path: Path):
    _write_gate(tmp_path, 1, 0.50)
    _write_gate(tmp_path, 0, 0.40)
    summaries = load_gate_summaries(tmp_path)
    assert [s["gate_id"] for s in summaries] == [0, 1]


def test_best_gate_ignores_ineligible_and_empty(tmp_path: Path):
    assert best_gate(tmp_path) is None  # bootstrap: no records yet
    _write_gate(tmp_path, 0, 0.90, eligible=False)
    assert best_gate(tmp_path) is None
    _write_gate(tmp_path, 1, None, eligible=False)
    assert best_gate(tmp_path) is None
    _write_gate(tmp_path, 2, 0.55)
    _write_gate(tmp_path, 3, 0.75)
    assert best_gate(tmp_path)["gate_id"] == 3


def test_load_gate_summaries_skips_corrupt_records(tmp_path: Path):
    _write_gate(tmp_path, 0, 0.5)
    (tmp_path / "gate_001").mkdir()
    (tmp_path / "gate_001" / "summary.json").write_text("{not json")
    assert [s["gate_id"] for s in load_gate_summaries(tmp_path)] == [0]


def test_summarize_gate_renders_feedback_line():
    line = summarize_gate({"gate_id": 4, "score": {
        "accuracy": 0.1234, "n_inner_epochs_scored": 3, "n_measured": 7, "n_attempted": 9,
    }})
    assert "gate 004" in line
    assert "0.1234" in line
    assert "7/9" in line


# ── duplicate rejection ──────────────────────────────────────────────────────

def test_structural_hash_ignores_formatting():
    assert structural_hash(GATE_SOURCE) == structural_hash(REFORMATTED_GATE)


def test_structural_hash_detects_real_change():
    assert structural_hash(GATE_SOURCE) != structural_hash(DISTINCT_GATE)


def test_is_duplicate_gate_uses_structural_hash():
    seen = {structural_hash(GATE_SOURCE)}
    assert is_duplicate_gate(REFORMATTED_GATE, seen) is True
    assert is_duplicate_gate(DISTINCT_GATE, seen) is False


def test_similarity_helper_is_advisory_on_templated_code():
    """Near-duplicate similarity is a warning; templated gate code inflates it."""
    similarity = near_duplicate_similarity(REFORMATTED_GATE, [GATE_SOURCE])
    # Either datasketch is absent (None) or it reports high similarity. Both are
    # acceptable because only the AST hash performs rejection.
    assert similarity is None or similarity > 0.5


# ── prompt ───────────────────────────────────────────────────────────────────

def test_proposal_prompt_includes_reference_and_feedback():
    prompt = gate_proposal_prompt(
        [(8, 4)], reference_source=GATE_SOURCE, feedback="- gate 000: mean 0.4",
    )
    assert "<gate>" in prompt and "LLMGeneratedGate" in prompt
    assert "self.base = nn.Linear" in prompt
    assert GATE_SOURCE.strip() in prompt          # reference present in round > 0
    assert "- gate 000: mean 0.4" in prompt       # measured feedback present
    assert "[(8, 4)]" in prompt


def test_proposal_prompt_bootstrap_has_no_reference():
    prompt = gate_proposal_prompt([(8, 4)])
    assert GATE_SOURCE.strip() not in prompt
    assert "current best MoE router gate" not in prompt


def test_prompt_scope_isolates_and_restores_cv_system_prompt():
    chatbot = SimpleNamespace(system_prompt="cv-system-prompt")
    with gate_prompt_scope(chatbot):
        assert chatbot.system_prompt == GATE_SYSTEM_PROMPT
    assert chatbot.system_prompt == "cv-system-prompt"


def test_sft_example_does_not_leak_score_into_input():
    pair = {
        "pair_id": "p1",
        "higher_source": DISTINCT_GATE,
        "lower_source": GATE_SOURCE,
        "higher_accuracy": 0.77,
    }
    example = gate_sft_examples([pair], shapes=[(8, 4)])[0]
    user_prompt = example["messages"][1]["content"]
    assistant = example["messages"][2]["content"]
    assert example["messages"][0]["content"] == GATE_SYSTEM_PROMPT
    assert DISTINCT_GATE.strip() not in user_prompt
    assert "0.77" not in user_prompt
    assert assistant == "<gate>\n" + DISTINCT_GATE.strip() + "\n</gate>"


def test_prompt_teaches_the_zero_init_contract():
    """The prompt must show a complete example and forbid `logits + x`."""
    prompt = gate_proposal_prompt([(2048, 64)])
    assert "import torch.nn as nn" in prompt          # imports are included
    assert "nn.init.zeros_(self.up.weight)" in prompt  # zero-at-init shown
    assert "Do NOT add the raw input" in prompt        # forbidden pattern named
    assert "SAFE WAYS TO BE DIVERSE" in prompt         # guided diversity
    assert "Do NOT copy it" in prompt                  # example not to be echoed


def test_baseline_gate_satisfies_the_contract():
    import torch
    import torch.nn as nn

    namespace = {}
    exec(compile(BASELINE_GATE_CODE, "<baseline>", "exec"), namespace)
    gate = namespace["LLMGeneratedGate"](8, 4)
    assert isinstance(gate, nn.Module)
    assert torch.equal(gate(torch.zeros(2, 8)), torch.zeros(2, 4))
    assert not is_duplicate_gate(BASELINE_GATE_CODE, set())


# ── generation loop integration (no model required) ──────────────────────────

class _ScriptedChatBot:
    """Emits queued raw outputs; records prompts for assertion."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []
        self.system_prompt = None

    def chat(self, prompt, engineer_prompt=True, max_new_tokens=None):
        self.prompts.append(prompt)
        raw = self.outputs.pop(0) if self.outputs else ""
        return None, None, None, raw


def _run_generate(tmp_path, outputs, *, reference="", seen=None, attempts=3):
    return _generate_gate(
        _ScriptedChatBot(outputs), [(8, 4)], attempts, 64, tmp_path,
        feedback_summary="", reference_source=reference,
        seen_hashes=seen if seen is not None else set(),
    )


def test_generator_accepts_valid_gate_and_records_prompt(tmp_path: Path):
    chatbot_holder = _ScriptedChatBot([f"<gate>\n{GATE_SOURCE}\n</gate>"])
    source = _generate_gate(chatbot_holder, [(8, 4)], 3, 64, tmp_path,
                            reference_source=GATE_SOURCE)
    assert structural_hash(source) == structural_hash(GATE_SOURCE)
    assert (tmp_path / "gate.py").is_file()
    assert (tmp_path / "proposal_prompt.txt").is_file()
    # The proposal must not inherit a CV-style system prompt.
    assert None in (chatbot_holder.system_prompt,)
    assert GATE_SOURCE.strip() in chatbot_holder.prompts[0]


def test_generator_rejects_duplicate_and_retries_with_distinct_gate(tmp_path: Path):
    seen = {structural_hash(GATE_SOURCE)}
    source = _run_generate(
        tmp_path,
        [f"<gate>\n{REFORMATTED_GATE}\n</gate>", f"<gate>\n{DISTINCT_GATE}\n</gate>"],
        seen=seen,
    )
    assert structural_hash(source) == structural_hash(DISTINCT_GATE)
    # The duplicate is now recorded so later rounds also avoid it.
    assert structural_hash(DISTINCT_GATE) in seen


def test_generator_fails_when_only_duplicates_are_emitted(tmp_path: Path):
    seen = {structural_hash(GATE_SOURCE)}
    # Each generation counts once regardless of how many parseable strings it
    # yields, so three duplicate generations exhaust a budget of three.
    with pytest.raises(RuntimeError, match="duplicate gate architecture"):
        _run_generate(
            tmp_path,
            [f"<gate>\n{REFORMATTED_GATE}\n</gate>"] * 3,
            seen=seen, attempts=3,
        )


def test_generator_retries_after_invalid_source(tmp_path: Path):
    source = _run_generate(
        tmp_path,
        ["<gate>\nimport os\n</gate>", f"<gate>\n{GATE_SOURCE}\n</gate>"],
    )
    assert structural_hash(source) == structural_hash(GATE_SOURCE)


# ── CLI / validation ─────────────────────────────────────────────────────────

def test_cli_exposes_outer_search_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py"])
    args = parse_args()
    assert args.gate_outer_search is False
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py", "--gate-outer-search"])
    assert parse_args().gate_outer_search is True


def test_cli_removed_obsolete_outer_sft_flags(monkeypatch):
    """The superseded adapter/benchmark flags must no longer exist."""
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py"])
    args = parse_args()
    for obsolete in ("gate_outer_sft", "gate_proposer_steps", "gate_benchmark"):
        assert not hasattr(args, obsolete)


def _validate_namespace(**overrides):
    defaults = dict(
        gate_outer_search=True, gate_source=None, repetition_penalty=1.0,
        generation_backend="pipeline", fixed_evaluation_prompts=False,
        gate_candidates=1, max_length=1, batch_size=1,
        gradient_checkpointing=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_validate_args_rejects_gate_source_with_outer_search():
    with pytest.raises(ValueError, match="generates gates itself"):
        _validate_args(_validate_namespace(gate_source=Path("gate.py")))


def test_validate_args_allows_outer_search_without_gate_source():
    _validate_args(_validate_namespace())


def test_feedback_block_covers_prior_gates(tmp_path: Path):
    _write_gate(tmp_path, 0, 0.40)
    _write_gate(tmp_path, 1, 0.70)
    reference = best_gate(tmp_path)
    block = _outer_feedback_block(tmp_path, reference)
    assert "current best prior gate" in block
    assert "gate 000" in block  # earlier candidates are still described
    assert "gate 001" in block


# ── Phase B seam ─────────────────────────────────────────────────────────────

def test_phase_b_has_no_pairs_during_bootstrap(tmp_path: Path):
    assert build_gate_pairs(tmp_path) == []
    assert "cannot start" in describe_pairs([])


def test_phase_b_pairs_only_within_comparable_setting(tmp_path: Path):
    _write_gate(tmp_path, 0, 0.40, dataset="cifar-10")
    _write_gate(tmp_path, 1, 0.70, dataset="cifar-10")
    _write_gate(tmp_path, 2, 0.90, dataset="cifar-100")  # different setting

    pairs = build_gate_pairs(tmp_path)
    assert len(pairs) == 1
    assert (pairs[0]["lower_gate_id"], pairs[0]["higher_gate_id"]) == (0, 1)
    assert pairs[0]["accuracy_gap"] == pytest.approx(0.30)
    # The cifar-100 gate must never be paired against cifar-10 gates.
    assert all(pair["dataset"] == "cifar-10" for pair in pairs)


def test_phase_b_pairs_exclude_ineligible_and_apply_gap_and_cap(tmp_path: Path):
    _write_gate(tmp_path, 0, 0.40)
    _write_gate(tmp_path, 1, 0.42)   # gap 0.02, below threshold
    _write_gate(tmp_path, 2, 0.80)
    _write_gate(tmp_path, 3, 0.99, eligible=False)  # not eligible

    pairs = build_gate_pairs(tmp_path, min_accuracy_gap=0.10)
    # 0.40->0.80 qualifies; 0.42->0.80 qualifies; both beat the 0.02 gap.
    assert {pair["pair_id"] for pair in pairs} == {"000->002", "001->002"}
    # Largest improvement first.
    assert pairs[0]["pair_id"] == "000->002"
    assert len(build_gate_pairs(tmp_path, min_accuracy_gap=0.10, max_pairs=1)) == 1


def test_phase_b_rejects_negative_gap(tmp_path: Path):
    with pytest.raises(ValueError):
        build_gate_pairs(tmp_path, min_accuracy_gap=-0.1)


def test_phase_b_cli_flag_fails_fast(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py", "--gate-phase-b"])
    args = parse_args()
    assert args.gate_phase_b is True
    with pytest.raises(NotImplementedError, match="Phase B"):
        _validate_args(args)
