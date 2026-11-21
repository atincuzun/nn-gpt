"""Unit tests for the controlled MoE experiment harness (no model needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _CharTokenizer:
    """Token id == character code; enough to drive the stopping criterion."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids if int(i) > 0)


def _ids(text: str) -> list[int]:
    return [ord(c) for c in text]


def _batch(prompt: str, completions: list[str]) -> torch.Tensor:
    width = max(len(c) for c in completions)
    rows = [_ids(prompt) + _ids(c) + [0] * (width - len(c)) for c in completions]
    return torch.tensor(rows)


def test_stopping_is_per_row():
    from moe_cycle.generation_stop import _TagStoppingCriteria

    prompt = "<nn>reference</nn> "
    criterion = _TagStoppingCriteria(_CharTokenizer(), prompt_len=len(prompt))
    verdict = criterion(_batch(prompt, ["<hp>{}</hp><nn>x=1</nn>", "<hp>{}</hp><nn>x=1"]), None)
    assert verdict.tolist() == [True, False]


def test_stopping_remembers_an_opening_tag_scrolled_out_of_the_tail():
    from moe_cycle.generation_stop import _TagStoppingCriteria

    criterion = _TagStoppingCriteria(_CharTokenizer(), prompt_len=0, tail_tokens=16)
    body = "<nn>" + "a" * 40
    assert criterion(torch.tensor([_ids(body[:10])]), None).tolist() == [False]
    assert criterion(torch.tensor([_ids(body)]), None).tolist() == [False]
    assert criterion(torch.tensor([_ids(body + "</nn>")]), None).tolist() == [True]


def test_stopping_ignores_a_closing_tag_before_the_opening():
    from moe_cycle.generation_stop import _TagStoppingCriteria

    criterion = _TagStoppingCriteria(_CharTokenizer(), prompt_len=0)
    assert criterion(torch.tensor([_ids("</nn> then <nn>code")]), None).tolist() == [False]


def test_loop_detector_flags_exact_repetition_only():
    from moe_cycle.generation_stop import _ends_in_loop

    block = "        layers += [\n            nn.Conv2d(256, 256, 3, padding=1),\n        ]\n"
    assert _ends_in_loop("def f():\n" + block * 8)
    assert not _ends_in_loop("def f():\n" + block * 3)
    varied = "".join(f"        self.conv{i} = nn.Conv2d({i}, {i + 1}, 3)\n" for i in range(40))
    assert not _ends_in_loop(varied)


def test_loop_criterion_stops_only_the_looping_row():
    from moe_cycle.generation_stop import _LoopStoppingCriteria

    criterion = _LoopStoppingCriteria(_CharTokenizer(), prompt_len=0, every=1)
    looping = "x = 1\n" + "    nn.ReLU(inplace=True),\n" * 40
    healthy = "".join(f"line {i}: value {i * i}\n" for i in range(60))[: len(looping)]
    verdict = criterion(_batch("", [looping, healthy]), None)
    assert verdict.tolist() == [True, False]


def test_gate_sources_resolve():
    from moe_cycle.experiment import CONSTANT_GATE_CODE, NATIVE_GATE_CODE, _gate_source

    assert _gate_source("native") == (NATIVE_GATE_CODE, "native")
    assert _gate_source("constant") == (CONSTANT_GATE_CODE, "constant")
    source, label = _gate_source("catalog:12")
    assert label == "12-swiglu" and "class Net" in source
    assert _gate_source("catalog:05-multiplicative-modulation")[1] == "05-multiplicative-modulation"
    with pytest.raises(KeyError):
        _gate_source("catalog:does-not-exist")


def test_native_and_constant_gates_compute_what_they_claim():
    from moe_cycle.experiment import CONSTANT_GATE_CODE, NATIVE_GATE_CODE
    from moe_gate_only.contract import compile_gate_from_string

    x = torch.randn(5, 8)
    native = compile_gate_from_string(NATIVE_GATE_CODE, class_name="Net")(8, 4)
    assert torch.equal(native(x), native.base(x))
    constant = compile_gate_from_string(CONSTANT_GATE_CODE, class_name="Net")(8, 4)
    assert torch.count_nonzero(constant(x)) == 0


def test_outcome_summary_counts_failures_as_zero_yield():
    from moe_cycle.experiment import _summarize_outcomes

    summary = _summarize_outcomes([
        {"status": "measured", "accuracy": 0.6, "baseline_accuracy": 0.5},
        {"status": "measured", "accuracy": 0.4, "baseline_accuracy": 0.5},
        {"status": "failed", "accuracy": None, "baseline_accuracy": 0.5},
        {"status": "failed", "accuracy": None, "baseline_accuracy": None},
    ])
    assert summary["n_slots"] == 4 and summary["n_measured"] == 2
    assert summary["valid_rate"] == pytest.approx(0.5)
    assert summary["mean_accuracy"] == pytest.approx(0.5)
    assert summary["yield_accuracy"] == pytest.approx(0.25)
    assert summary["improved_rate"] == pytest.approx(0.25)


def test_paired_contrast_matches_seed_and_prompt():
    sys.path.insert(0, str(ROOT / "moe_experiments"))
    import analyze

    def row(arm, nn, acc, gen_seed=1001, round_=1):
        return {"arm": arm, "round": round_, "gen_seed": gen_seed, "source_nn": nn,
                "accuracy": acc, "code_sha1": None}

    rows = [
        row("a", "n1", 0.7), row("a", "n2", None), row("a", "n3", 0.5),
        row("b", "n1", 0.6), row("b", "n2", 0.4), row("b", "n3", 0.5),
        row("b", "n4", 0.9),  # unmatched prompt is ignored
    ]
    yield_diff = analyze._paired(rows, "a", "b", {1}, None, "yield")
    assert sum(sum(v) for v in yield_diff.values()) == pytest.approx(0.1 - 0.4 + 0.0)
    acc_diff = analyze._paired(rows, "a", "b", {1}, None, "acc")
    assert set(acc_diff) == {"n1", "n3"}
    stats = analyze._stats(yield_diff)
    assert stats["n"] == 3 and stats["ci_low"] <= stats["mean"] <= stats["ci_high"]
