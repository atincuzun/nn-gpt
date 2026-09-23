"""Tests for the self-improving MoE gate search (propose, measure, revise)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from moe_cycle.cli import parse_args
from moe_cycle.cycle import _validate_args
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
    "        self.base = nn.Linear(model_dim, num_experts, bias=False)\n"
    "        self.branch = nn.Linear(model_dim, num_experts, bias=False)\n"
    "        nn.init.zeros_(self.branch.weight)\n\n"
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        return self.base(x) + self.branch(x)\n"
)

# Same behaviour, different serialisation (whitespace/typing) -> duplicate.
REFORMATTED_GATE = GATE_SOURCE.replace("model_dim: int", "model_dim:int")
# Different structure -> distinct architecture (silent Tanh branch).
DISTINCT_GATE = GATE_SOURCE.replace(
    "        self.branch = nn.Linear(model_dim, num_experts, bias=False)\n"
    "        nn.init.zeros_(self.branch.weight)\n",
    "        self.branch = nn.Sequential(\n"
    "            nn.Linear(model_dim, 2 * model_dim),\n"
    "            nn.Tanh(),\n"
    "            nn.Linear(2 * model_dim, num_experts, bias=False),\n"
    "        )\n"
    "        nn.init.zeros_(self.branch[-1].weight)\n",
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

def _write_gate(root, gate_id, accuracy, *, eligible=True, source=None,
                task="img-classification", dataset="cifar-10", metric="acc"):
    if source is None:
        # Distinct architecture per gate: build_gate_pairs excludes
        # same-architecture reruns by structural hash, so a shared default
        # source would silently empty the pair set.
        source = GATE_SOURCE.replace(
            "class LLMGeneratedGate", f"class LLMGeneratedGateV{gate_id}"
        )
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

def test_proposal_prompt_includes_reference_pairing():
    """LEMUR-CV pairing: reference code + its score + the goal score; no tables."""
    prompt = gate_proposal_prompt(
        [(8, 4)], reference_source=GATE_SOURCE,
        reference_accuracy=0.45, goal_accuracy=0.52, dataset="cifar-10",
    )
    assert "<gate>" in prompt and "LLMGeneratedGate" in prompt
    assert "self.base = nn.Linear" in prompt
    assert GATE_SOURCE.strip() in prompt           # reference present
    assert "0.4500" in prompt                      # reference score
    assert "at least 0.5200" in prompt             # goal score, NN_gen style
    assert "cifar-10" in prompt
    assert "Measured results" not in prompt        # history table is gone
    assert "unavailable" not in prompt


def test_proposal_prompt_bootstrap_has_no_reference():
    prompt = gate_proposal_prompt([(8, 4)])
    assert GATE_SOURCE.strip() not in prompt
    assert "current best MoE router gate" not in prompt
    assert "at least" not in prompt                # no goal without a scored reference


def test_prompt_scope_isolates_and_restores_cv_system_prompt():
    chatbot = SimpleNamespace(system_prompt="cv-system-prompt")
    with gate_prompt_scope(chatbot):
        assert chatbot.system_prompt == GATE_SYSTEM_PROMPT
    assert chatbot.system_prompt == "cv-system-prompt"


def test_sft_example_mirrors_nn_gen_pairing():
    """NN_gen.json mirror: reference gate + its score in the prompt, the
    better gate's score stated as the goal, better code only as the target."""
    pair = {
        "pair_id": "p1",
        "lower_gate_id": 5,
        "higher_gate_id": 9,
        "higher_source": DISTINCT_GATE,
        "lower_source": GATE_SOURCE,
        "lower_accuracy": 0.4566,
        "higher_accuracy": 0.77,
        "dataset": "cifar-10",
    }
    example = gate_sft_examples([pair], shapes=[(8, 4)])[0]
    user_prompt = example["messages"][1]["content"]
    assistant = example["messages"][2]["content"]
    assert example["messages"][0]["content"] == GATE_SYSTEM_PROMPT
    # Reference gate: code + measured accuracy.
    assert GATE_SOURCE.strip() in user_prompt
    assert "0.4566" in user_prompt
    # Goal: the better score, stated like NN_gen's "increase to at least".
    assert "at least 0.7700" in user_prompt
    # The better architecture appears only as the target.
    assert DISTINCT_GATE.strip() not in user_prompt
    assert assistant == "<gate>\n" + DISTINCT_GATE.strip() + "\n</gate>"


def test_prompt_states_the_contract_and_leaves_the_architecture_open():
    """The prompt pins the runtime contract; the network design stays open."""
    prompt = gate_proposal_prompt([(2048, 64)])
    assert len(prompt) < 5000, "prompt over-specified for the proposal context"
    # The seam.
    assert "LLMGeneratedGate(nn.Module)" in prompt
    assert "__init__(self, model_dim: int, num_experts: int)" in prompt
    assert "self.base = nn.Linear(model_dim, num_experts, bias=False)" in prompt
    assert "Input shape:" in prompt and "Output shape:" in prompt
    assert '`(..., model_dim)`' in prompt and '`(..., num_experts)`' in prompt
    assert "finite raw router logits" in prompt
    # The host owns the routing that follows the logits.
    assert "Softmax, expert selection, top-k routing, auxiliary routing losses, and expert dispatch are handled externally" in prompt
    # Step zero must reproduce the copied native weight.
    assert "reproduce `self.base(x)` exactly" in prompt
    # Connectivity without degeneracy.
    assert "connected to the output computation" in prompt
    # The network is explicitly free.
    assert "Choose the architecture, internal representations, transformations, nonlinearities, parameterization, and composition yourself" in prompt
    assert "meaningful learnable capacity beyond the required native-weight interface" in prompt
    # Shapes are injected, and no concrete model is named.
    assert "[(2048, 64)]" in prompt
    assert "DeepSeek" not in prompt and "MoEGate" not in prompt
    # Output format.
    assert "<gate>" in prompt


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
        reference_source=reference,
        seen_hashes=seen if seen is not None else set(),
    )


def test_generator_accepts_valid_gate_and_records_prompt(tmp_path: Path):
    chatbot_holder = _ScriptedChatBot([f"<gate>\n{GATE_SOURCE}\n</gate>"])
    sources = _generate_gate(chatbot_holder, [(8, 4)], 3, 64, tmp_path,
                             reference_source=GATE_SOURCE)
    assert len(sources) == 1
    assert structural_hash(sources[0]) == structural_hash(GATE_SOURCE)
    assert (tmp_path / "gate.py").is_file()
    assert (tmp_path / "proposal_prompt.txt").is_file()
    # The proposal must not inherit a CV-style system prompt.
    assert None in (chatbot_holder.system_prompt,)
    assert GATE_SOURCE.strip() in chatbot_holder.prompts[0]


def test_generator_harvests_all_valid_gates_from_one_response(tmp_path: Path):
    """Nothing usable is discarded: multiple valid distinct gates in one
    response are all returned, order preserved."""
    raw = f"<gate>\n{GATE_SOURCE}\n</gate>\n<gate>\n{DISTINCT_GATE}\n</gate>"
    sources = _run_generate(tmp_path, [raw])
    assert [structural_hash(s) for s in sources] == [
        structural_hash(GATE_SOURCE), structural_hash(DISTINCT_GATE)]
    # First harvested gate is the round's artifact.
    assert structural_hash((tmp_path / "gate.py").read_text(encoding="utf-8"))         == structural_hash(GATE_SOURCE)


def test_generator_harvest_stops_after_first_dry_attempt(tmp_path: Path):
    """Once something valid is harvested, one dry attempt ends the round."""
    raw_good = f"<gate>\n{GATE_SOURCE}\n</gate>"
    sources = _run_generate(tmp_path, [raw_good, "no gate here", raw_good])
    assert len(sources) == 1
    assert len(chatbot_attempt_files := list(tmp_path.glob("generation_attempt_*.txt"))) == 2



def test_generator_rejects_duplicate_and_retries_with_distinct_gate(tmp_path: Path):
    seen = {structural_hash(GATE_SOURCE)}
    sources = _run_generate(
        tmp_path,
        [f"<gate>\n{REFORMATTED_GATE}\n</gate>", f"<gate>\n{DISTINCT_GATE}\n</gate>"],
        seen=seen,
    )
    assert [structural_hash(s) for s in sources] == [structural_hash(DISTINCT_GATE)]
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
    sources = _run_generate(
        tmp_path,
        ["<gate>\nimport os\n</gate>", f"<gate>\n{GATE_SOURCE}\n</gate>"],
    )
    assert structural_hash(sources[0]) == structural_hash(GATE_SOURCE)


# ── CLI / validation ─────────────────────────────────────────────────────────

def test_cli_exposes_random_init_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py"])
    assert parse_args().gate_random_init is False
    monkeypatch.setattr(
        sys, "argv", ["run_moe_gate_cycle.py", "--gate-random-init"]
    )
    assert parse_args().gate_random_init is True


def test_random_init_skips_the_native_weight_copy(monkeypatch, tmp_path):
    """--gate-random-init must not copy the native router into the gate."""
    import torch

    from moe_cycle import cycle as cycle_module

    recorded = {}

    class _FakeModel:
        config = SimpleNamespace(use_cache=False)

        def eval(self):
            return self

    class _FakeSession:
        installs = []
        gate_source = None
        model = _FakeModel()

        def replace_source(self, source, **kwargs):
            recorded["replace_kwargs"] = kwargs
            return []

        def freeze_except_gates(self):
            recorded["frozen"] = True

    class _FakeContext:
        args = SimpleNamespace(
            seed=1,
            layers=None,
            router_top_k=None,
            gate_source=None,
            gate_class="LLMGeneratedGate",
            gate_init_noise_scale=0.0,
            gate_random_init=True,
            load_in_8bit=False,
            gradient_checkpointing=False,
        )
        session = _FakeSession()
        gate_dir = tmp_path
        sample_input = None
        native_logits = torch.zeros(1)
        native_repeatable = True
        native_repeat_max_abs_difference = 0.0
        gate_source = "class LLMGeneratedGate: pass"

    monkeypatch.setattr(
        cycle_module, "_model_logits",
        lambda model, sample_input: torch.zeros(1),
    )
    monkeypatch.setattr(
        cycle_module, "_perturb_gate_weights", lambda installs, scale, seed: []
    )

    cycle_module._install_and_verify(_FakeContext())

    assert recorded["replace_kwargs"]["initialize_from_original"] is False


def test_cli_exposes_outer_search_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py"])
    args = parse_args()
    assert args.gate_outer_search is False
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py", "--gate-outer-search"])
    assert parse_args().gate_outer_search is True


def test_cli_removed_obsolete_outer_sft_flags(monkeypatch):
    """Superseded adapter/benchmark flags must no longer exist.

    ``gate_outer_sft`` is no longer in this list: it was dead-code scaffolding
    when this test was written, and is now the implemented Phase B switch.
    """
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py"])
    args = parse_args()
    for obsolete in ("gate_proposer_steps", "gate_benchmark"):
        assert not hasattr(args, obsolete)


def _validate_namespace(**overrides):
    defaults = dict(
        gate_outer_search=True, gate_source=None, repetition_penalty=1.0,
        gate_random_init=False, gate_author="native",
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


def test_validate_args_gate_author_modes_require_outer_search():
    with pytest.raises(ValueError, match="gate-author"):
        _validate_args(_validate_namespace(gate_outer_search=False, gate_author="last"))
    with pytest.raises(ValueError, match="gate-author"):
        _validate_args(_validate_namespace(gate_outer_search=False, gate_author="best"))
    _validate_args(_validate_namespace(gate_outer_search=False, gate_author="native"))
    _validate_args(_validate_namespace(gate_author="last"))
    _validate_args(_validate_namespace(gate_author="best"))


# ── gate author state (--gate-author) ────────────────────────────────────────

def test_cli_gate_author_defaults_to_native(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py"])
    args = parse_args()
    assert args.gate_author == "native"
    # The default conf keys must exist in the default test prompt config.
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "ab/gpt/conf/prompt/test/NN_gen.json")
        .read_text(encoding="utf-8")
    )
    assert set(args.conf_keys) <= set(config), args.conf_keys


def test_cli_gate_author_accepts_last_and_best(monkeypatch):
    for value in ("last", "best"):
        monkeypatch.setattr(
            sys, "argv",
            ["run_moe_gate_cycle.py", "--gate-outer-search", "--gate-author", value],
        )
        assert parse_args().gate_author == value


def test_cli_gate_author_rejects_unknown_choice(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["run_moe_gate_cycle.py", "--gate-author", "both"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_author_gate_id_selection(tmp_path: Path):
    from moe_cycle.cycle import _select_author_gate_id

    # native mode and round 0 never install an author gate
    assert _select_author_gate_id("native", 3, tmp_path) is None
    assert _select_author_gate_id("last", 0, tmp_path) is None
    assert _select_author_gate_id("best", 0, tmp_path) is None

    # last mode chains to the previous candidate
    assert _select_author_gate_id("last", 2, tmp_path) == 1

    # best mode picks the highest-scoring eligible gate, or falls back to native
    _write_gate(tmp_path, 0, 0.40)
    _write_gate(tmp_path, 1, 0.55)
    _write_gate(tmp_path, 2, 0.90, eligible=False)  # ineligible gates are ignored
    assert _select_author_gate_id("best", 3, tmp_path) == 1
    assert _select_author_gate_id("best", 1, tmp_path / "empty") is None

    with pytest.raises(ValueError, match="unknown gate author mode"):
        _select_author_gate_id("chain", 1, tmp_path)


def test_gate_record_stores_author_state(tmp_path: Path):
    from moe_cycle.cycle import _write_gate_record

    ctx = SimpleNamespace(
        args=SimpleNamespace(gate_author="best"),
        candidate_index=2,
        gate_source=DISTINCT_GATE,
        reference_gate_id=1,
        gate_epoch_metrics=[_epoch(0.42, 0.5, 2, 2)],
        gate_root=tmp_path,
        author_gate_id=1,
    )
    record = _write_gate_record(ctx)
    assert record["author_mode"] == "best"
    assert record["author_gate_id"] == 1
    stored = json.loads(
        (tmp_path / "gate_002" / "summary.json").read_text(encoding="utf-8")
    )
    assert stored["author_mode"] == "best"
    assert stored["author_gate_id"] == 1




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


# ── cold-start bootstrap seeds ────────────────────────────────────────────────

def test_seed_gates_satisfy_the_llm_proposal_validator():
    """Seeds must pass the SAME contract/degenerate checks as LLM proposals."""
    from moe_cycle.gate_seeds import SEED_GATES
    from moe_cycle.gate_source import _validate_gate_source

    hashes = set()
    for name, source in SEED_GATES:
        _validate_gate_source(source, [(2048, 64), (512, 8)])
        hashes.add(structural_hash(source))
    assert len(hashes) == len(SEED_GATES), "seed mechanisms must be distinct"


def test_baseline_reference_is_not_degenerate():
    """The round-0 reference must itself satisfy the enforced contract."""
    from moe_cycle.gate_prompt import BASELINE_GATE_CODE
    from moe_cycle.gate_seeds import SEED_GATES
    from moe_cycle.gate_source import _validate_gate_source

    _validate_gate_source(BASELINE_GATE_CODE, [(2048, 64)])
    seed_hashes = {structural_hash(source) for _name, source in SEED_GATES}
    assert structural_hash(BASELINE_GATE_CODE) not in seed_hashes


def test_seed_candidates_prepare_without_llm(tmp_path: Path, monkeypatch):
    """_prepare_seed_candidate writes a valid gate with no chatbot involved."""
    import torch

    from moe_cycle.cycle import _prepare_seed_candidate

    ctx = SimpleNamespace(
        gate_root=tmp_path, shapes=[(2048, 64)], gate_source=None,
        used_prompts=[], seeded=None,
    )
    monkeypatch.setattr(
        "moe_cycle.cycle._initial_feedback_summary", lambda source: ""
    )
    _prepare_seed_candidate(ctx, 1)
    assert (tmp_path / "gate_001" / "gate.py").is_file()
    assert ctx.seeded is True
    source = (tmp_path / "gate_001" / "gate.py").read_text(encoding="utf-8")
    namespace = {}
    exec(compile(source, "<seed>", "exec"), namespace)
    gate = namespace["LLMGeneratedGate"](2048, 64).float().eval()
    with torch.no_grad():
        out = gate(torch.randn(2, 2048))
    assert out.shape == (2, 64) and torch.isfinite(out).all()
