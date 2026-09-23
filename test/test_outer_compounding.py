"""End-to-end verification of the outer-loop compounding flow on a tiny model.

Uses a real (tiny) HF Mixtral model whose native router is a plain
``nn.Linear`` — the same shape as DeepSeek-V2-Lite's ``MoEGate`` — and drives
the actual cycle primitives: ``MoEGateSession.replace_source`` /
``load_weights`` / ``save``, ``carry_gate_weights``,
``verify_parent_function_carry``, and ``_run_sft_batch_under_best_gate``.

Verified claims, in the order the user stated them:
1. Architecture propagates: a successor written in inherit mode (keeps the
   parent's module names, adds a zero-initialized branch) contains every
   parent parameter, and the carry report copies exactly the parent's
   parameters while leaving only the new branch fresh.
2. Weights propagate: every copied parameter is tensor-equal to the parent's
   TRAINED checkpoint values (not the native copy, not constructor init).
3. Function propagates: the successor's step-zero model logits reproduce the
   trained parent's logits (the parent-equivalence gate accepts it), while a
   successor that rewrites the inherited path is rejected by the same check.
4. Proposer SFT runs under the best gate: ``_run_sft_batch_under_best_gate``
   installs the best store gate's architecture + trained weights, freezes the
   gate parameters, and restores the native routers afterwards; it falls back
   to native routing with no scored gate or a corrupt gate source.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import moe_cycle.cycle as cycle_module  # noqa: E402
from moe_cycle.cycle import _run_sft_batch_under_best_gate  # noqa: E402
from moe_cycle.gate_seeds import seed_gate_source  # noqa: E402
from moe_cycle.morphism import (  # noqa: E402
    _model_logits,
    carry_gate_weights,
    verify_parent_function_carry,
)
from moe_gate_only import MoEGateSession  # noqa: E402


def _tiny_mixtral():
    from transformers import MixtralConfig, MixtralForCausalLM

    cfg = MixtralConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        num_local_experts=4, num_experts_per_tok=2,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False,
    )
    return MixtralForCausalLM(cfg)


# The cold-start seed "additive-mlp-gelu" from moe_cycle.gate_seeds: the
# parent architecture whose trained weights the successor must inherit. The
# successor below is what the LLM returns in inherit mode: parent modules
# kept by name, one new zero-initialized branch appended to the computation.
PARENT_SOURCE = seed_gate_source(0)

SUCCESSOR_SOURCE = """import torch
import torch.nn as nn

class LLMGeneratedGate(nn.Module):
    def __init__(self, model_dim: int, num_experts: int):
        super().__init__()
        self.base = nn.Linear(model_dim, num_experts, bias=False)
        self.branch = nn.Sequential(
            nn.Linear(model_dim, 2 * model_dim),
            nn.GELU(),
            nn.Linear(2 * model_dim, num_experts, bias=False),
        )
        self.branch2 = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, num_experts, bias=False),
        )
        nn.init.zeros_(self.branch2[-1].weight)

    def forward(self, x):
        return self.base(x) + self.branch(x) + self.branch2(x)
"""

# A successor that keeps the parent's parameter names but rewrites the
# inherited computation: the weight carry succeeds, the function check must
# reject it.
REWRITTEN_SOURCE = """import torch
import torch.nn as nn

class LLMGeneratedGate(nn.Module):
    def __init__(self, model_dim: int, num_experts: int):
        super().__init__()
        self.base = nn.Linear(model_dim, num_experts, bias=False)
        self.branch = nn.Sequential(
            nn.Linear(model_dim, 2 * model_dim),
            nn.GELU(),
            nn.Linear(2 * model_dim, num_experts, bias=False),
        )
        nn.init.zeros_(self.branch[-1].weight)

    def forward(self, x):
        return 1.5 * self.base(x) + self.branch(x)
"""


def _sample_input() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {"input_ids": torch.randint(0, 32, (1, 8), generator=generator)}


def _train_gate_parameters(session: MoEGateSession) -> None:
    """Simulate the inner loop's gate-only training with deterministic noise.

    A uniform shift must be avoided: adding the same constant to every expert
    logit is (near-)invariant under softmax/top-k routing, so the model logits
    would barely move and the carry checks below would be vacuous.
    """
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for install in session.installs:
            for parameter in install.new_gate.parameters():
                parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)


def _build_trained_parent_checkpoint(tmp_path: Path):
    """Install the parent, train it, save a real cycle-format checkpoint.

    Returns (session, checkpoint_path, parent_trained_logits, native_weights,
    trained_states) with the session restored to native routers.
    """
    torch.manual_seed(42)
    session = MoEGateSession(_tiny_mixtral())
    session.model.eval()
    sample_input = _sample_input()

    sites = session.inspect()
    assert sites, "discovery found no gate sites in the tiny Mixtral model"
    native_weights = {
        site.layer_index: site.gate.weight.detach().clone() for site in sites
    }
    native_logits = _model_logits(session.model, sample_input)

    with session:
        session.replace_source(
            PARENT_SOURCE,
            class_name="LLMGeneratedGate",
            sample_input=sample_input,
            verify=True,
            # The cycle installs and saves with dynamic discovery off; the
            # checkpoint keys must match what _install_author_gate expects.
            dynamic_discovery=False,
            initialize_from_original=True,
        )
        # Step zero: the native-copy morphism must keep the model bit-identical.
        step_zero = _model_logits(session.model, sample_input)
        assert torch.equal(native_logits, step_zero), (
            "parent install did not reproduce native step-zero logits"
        )

        _train_gate_parameters(session)
        trained_states = {
            install.site.layer_index: {
                name: value.detach().clone()
                for name, value in install.new_gate.state_dict().items()
            }
            for install in session.installs
        }
        parent_trained_logits = _model_logits(session.model, sample_input)
        max_training_shift = float(
            (native_logits - parent_trained_logits).abs().max().item()
        )
        assert max_training_shift > 1e-4, (
            "training the gates did not measurably change the model function "
            f"(max shift {max_training_shift}); the carry checks would be vacuous"
        )
        checkpoint = session.save(tmp_path / "epochs" / "A0" / "gate_post_train")
    assert session.installs == []
    for site in sites:
        assert torch.equal(site.gate.weight, native_weights[site.layer_index]), (
            "native router was modified by the parent candidate lifecycle"
        )
    return SimpleNamespace(
        session=session,
        checkpoint=checkpoint,
        parent_trained_logits=parent_trained_logits,
        native_weights=native_weights,
        trained_states=trained_states,
        sample_input=sample_input,
    )


def test_carry_propagates_architecture_weights_and_function(tmp_path: Path) -> None:
    built = _build_trained_parent_checkpoint(tmp_path / "parent")
    session = built.session

    with session:
        session.replace_source(
            SUCCESSOR_SOURCE,
            class_name="LLMGeneratedGate",
            sample_input=built.sample_input,
            verify=False,
            initialize_from_original=True,
        )
        report = carry_gate_weights(session.installs, built.checkpoint)
        successor_logits = _model_logits(session.model, built.sample_input)

        for install in session.installs:
            layer = install.site.layer_index
            successor_state = install.new_gate.state_dict()
            parent_state = built.trained_states[layer]
            # 1. Architecture: every parent parameter exists in the successor.
            missing = set(parent_state) - set(successor_state)
            assert not missing, f"successor dropped parent parameters: {missing}"
            # 2. Weights: inherited parameters carry the parent's TRAINED values.
            for name, parent_value in parent_state.items():
                assert torch.equal(successor_state[name], parent_value), (
                    f"parameter {name} did not receive the parent's trained weights"
                )
            # The new branch is the only fresh capacity; its output layer is
            # zero-initialized so it contributes nothing at step zero.
            fresh = {name for name in successor_state if name not in parent_state}
            assert fresh == {
                "branch2.0.weight", "branch2.0.bias", "branch2.2.weight",
            }, f"unexpected fresh parameters: {fresh}"
            assert torch.count_nonzero(successor_state["branch2.2.weight"]).item() == 0
            site_report = next(
                entry for entry in report["sites"]
                if entry["path"] == install.site.path
                and entry["layer_index"] == layer
            )
            assert set(site_report["parameters_copied"]) == set(parent_state)
            assert set(site_report["parameters_fresh"]) == fresh

        # 3. Function: the successor reproduces the trained parent exactly.
        carry_check = verify_parent_function_carry(
            built.parent_trained_logits, successor_logits
        )
        assert carry_check["equivalent"], (
            "successor does not reproduce the trained parent's step-zero logits: "
            f"{carry_check}"
        )
    assert session.installs == []


def test_rewritten_successor_fails_the_parent_function_check(tmp_path: Path) -> None:
    built = _build_trained_parent_checkpoint(tmp_path / "parent")
    session = built.session

    with session:
        session.replace_source(
            REWRITTEN_SOURCE,
            class_name="LLMGeneratedGate",
            sample_input=built.sample_input,
            verify=False,
            initialize_from_original=True,
        )
        # The carry itself succeeds (names match) — the function gate is what
        # catches the rewrite, exactly as in cycle._install_and_verify.
        carry_gate_weights(session.installs, built.checkpoint)
        rewritten_logits = _model_logits(session.model, built.sample_input)
        carry_check = verify_parent_function_carry(
            built.parent_trained_logits, rewritten_logits
        )
        assert not carry_check["equivalent"], (
            "a rewritten successor passed the parent-equivalence gate"
        )
        # Softmax/top-k routing damps small router changes, so the logit shift
        # is modest — it only needs to be far beyond the rtol/atol used for
        # the equivalence call above.
        assert carry_check["max_abs_difference"] > 1e-4


def _write_best_gate_store(store: Path, checkpoint: Path) -> Path:
    """A persistent store with one scored best gate (the user's gate 2 at 0.42)."""
    gate_root = store / "gate_002"
    (gate_root / "epochs" / "A1").mkdir(parents=True)
    (gate_root / "gate.py").write_text(PARENT_SOURCE, encoding="utf-8")
    (checkpoint / "gate.py").write_text(PARENT_SOURCE, encoding="utf-8")
    checkpoint.replace(gate_root / "epochs" / "A1" / "gate_post_train")
    (gate_root / "summary.json").write_text(json.dumps({
        "gate_id": 2,
        "score": {"objective": "mean_of_epoch_means_v1", "accuracy": 0.42,
                  "eligible": True, "n_measured": 2, "n_attempted": 2},
        "task": "img-classification", "dataset": "cifar-10", "metric": "accuracy",
    }), encoding="utf-8")
    return store


def _sft_args(store: Path) -> Namespace:
    return Namespace(
        layers=None, router_top_k=None, gate_author="best",
        gate_carry_forward="best", gate_store=store, gate_min_pairs=1,
        gate_sft_mode="sft", gate_sft_every=10, gate_sft_steps=30,
        gate_sft_rank=16, model="tiny-mixtral",
    )


def test_sft_batch_runs_under_best_gate_architecture_and_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _build_trained_parent_checkpoint(tmp_path / "parent")
    store = _write_best_gate_store(tmp_path / "store", built.checkpoint)
    ctx = SimpleNamespace(
        gate_root=store, session=built.session, args=_sft_args(store),
        sample_input=built.sample_input,
    )

    captured: dict = {}

    def spy_phase_b_batch(ctx, batch_index):
        captured["installs"] = len(ctx.session.installs)
        captured["gate_source"] = ctx.session.gate_source
        captured["states"] = {
            install.site.layer_index: {
                name: value.detach().clone()
                for name, value in install.new_gate.state_dict().items()
            }
            for install in ctx.session.installs
        }
        captured["trainable"] = {
            install.site.layer_index: [
                name for name, value in install.new_gate.named_parameters()
                if value.requires_grad
            ]
            for install in ctx.session.installs
        }
        return "proposer-v-next"

    monkeypatch.setattr(cycle_module, "_run_phase_b_batch", spy_phase_b_batch)
    version = _run_sft_batch_under_best_gate(ctx, 1)

    assert version == "proposer-v-next"
    # Compare the gates: the SFT batch ran on the best gate's architecture.
    assert captured["installs"] == 2
    assert captured["gate_source"] == PARENT_SOURCE
    # Compare the weights: they are the best gate's TRAINED weights, not the
    # native router copy and not constructor initialization.
    for layer, state in captured["states"].items():
        for name, trained_value in built.trained_states[layer].items():
            assert torch.equal(state[name], trained_value), (
                f"SFT ran under layer {layer} {name} without the best gate's "
                "trained weights"
            )
        assert not torch.equal(state["base.weight"], built.native_weights[layer])
    # Gate parameters stay frozen; only the proposer LoRA would train.
    assert all(not names for names in captured["trainable"].values())
    # The session is restored: no gates remain installed and the native
    # routers are untouched.
    assert built.session.installs == []
    for site in built.session.inspect():
        assert torch.equal(site.gate.weight, built.native_weights[site.layer_index])


def test_sft_batch_falls_back_to_native_without_scored_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _build_trained_parent_checkpoint(tmp_path / "parent")
    store = tmp_path / "empty_store"
    store.mkdir()
    ctx = SimpleNamespace(
        gate_root=store, session=built.session, args=_sft_args(store),
        sample_input=built.sample_input,
    )

    def spy_phase_b_batch(ctx, batch_index):
        assert ctx.session.installs == [], "batch must run under native routing"
        return "proposer-v-native"

    monkeypatch.setattr(cycle_module, "_run_phase_b_batch", spy_phase_b_batch)
    assert _run_sft_batch_under_best_gate(ctx, 1) == "proposer-v-native"


def test_sft_batch_falls_back_when_best_gate_source_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _build_trained_parent_checkpoint(tmp_path / "parent")
    store = _write_best_gate_store(tmp_path / "store", built.checkpoint)
    (store / "gate_002" / "gate.py").write_text("class LLMGeneratedGate(:\n", encoding="utf-8")
    ctx = SimpleNamespace(
        gate_root=store, session=built.session, args=_sft_args(store),
        sample_input=built.sample_input,
    )

    def spy_phase_b_batch(ctx, batch_index):
        assert ctx.session.installs == [], "broken incumbent must not block SFT"
        return "proposer-v-native"

    monkeypatch.setattr(cycle_module, "_run_phase_b_batch", spy_phase_b_batch)
    assert _run_sft_batch_under_best_gate(ctx, 1) == "proposer-v-native"
