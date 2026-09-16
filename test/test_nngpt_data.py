import json
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from moe_cycle.cli import parse_args
from moe_cycle.cycle import _generation_prefixes, _nas_prefixes, _pinned_prompt_dict
from moe_cycle.feedback import _build_gate_feedback_summary
from moe_gate_only.nngpt_data import (
    _completion_only_dataset,
    _fit_completion_to_context,
    _runtime_prompt_config,
)


def test_fit_completion_trims_only_prompt_tokens() -> None:
    tokenized = {
        "input_ids": list(range(12)),
        "attention_mask": [1] * 12,
        "labels": [-100] * 8 + list(range(8, 12)),
        "response_length": 4,
    }

    fitted = _fit_completion_to_context(tokenized, context_length=8)

    assert fitted["fits_context"] is True
    assert fitted["input_ids"] == [0, 1, 6, 7, 8, 9, 10, 11]
    assert fitted["labels"] == [-100] * 4 + [8, 9, 10, 11]
    assert fitted["truncated_prompt_tokens"] == 4


def test_cycle_enables_gradient_checkpointing_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_moe_gate_cycle.py"])
    assert parse_args().gradient_checkpointing is True


def test_cycle_allows_gradient_checkpointing_opt_out(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_moe_gate_cycle.py", "--no-gradient-checkpointing"],
    )
    assert parse_args().gradient_checkpointing is False


def test_cycle_parses_fixed_evaluation_hyperparameters(monkeypatch) -> None:
    fixed_prm = {
        "batch": 64,
        "dropout": 0.2,
        "lr": 0.01,
        "momentum": 0.9,
        "transform": "norm_32_flip",
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_moe_gate_cycle.py",
            "--fixed-eval-hyperparameters",
            json.dumps(fixed_prm),
        ],
    )

    assert parse_args().fixed_eval_hyperparameters == fixed_prm


def test_cycle_rejects_non_object_fixed_evaluation_hyperparameters(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_moe_gate_cycle.py", "--fixed-eval-hyperparameters", '["lr"]'],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_fit_completion_rejects_response_larger_than_context() -> None:
    tokenized = {
        "input_ids": list(range(8)),
        "attention_mask": [1] * 8,
        "labels": [-100] + list(range(1, 8)),
        "response_length": 7,
    }

    fitted = _fit_completion_to_context(tokenized, context_length=6)

    assert fitted["fits_context"] is False


def test_completion_only_dataset_masks_prompt_tokens() -> None:
    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            text = "|".join(f"{m['role']}:{m['content']}" for m in messages)
            return text + ("|assistant:" if add_generation_prompt else "")

        def __call__(self, text, **_kwargs):
            return {"input_ids": list(range(1, len(text) + 1)), "attention_mask": [1] * len(text)}

    class Processor:
        tokenizer = Tokenizer()

        def get_raw_dataset(self, *_args):
            return pd.DataFrame([{
                "instruction": "question",
                "response": "answer",
                "text": "system:rules|user:question|assistant:answer",
            }])

        def _build_messages(self, user, assistant=None, system_prompt=None):
            messages = [{"role": "user", "content": user}]
            if assistant is not None:
                messages.append({"role": "assistant", "content": assistant})
            return messages

        def _apply_chat_template(self, messages, **kwargs):
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    dataset = _completion_only_dataset(
        Processor(), context_length=256, max_prompts=1, max_new_tokens=256,
        only_best_accuracy=False, seed=1,
    )
    labels = dataset[0]["labels"]
    assert -100 in labels
    assert any(label != -100 for label in labels)


def test_runtime_prompt_config_overrides_only_temporary_copy(tmp_path: Path) -> None:
    source = tmp_path / "NN_gen.json"
    original = {"gate": {"dataset": "old", "nn_prefixes": ["base"]}}
    source.write_text(json.dumps(original), encoding="utf-8")

    with _runtime_prompt_config(
        source,
        dataset_name="cifar-10",
        nn_prefixes=("ga-", "moe-gate-cycle"),
    ) as runtime_path:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        assert runtime["gate"]["dataset"] == "cifar-10"
        assert runtime["gate"]["nn_prefixes"] == ["ga-", "moe-gate-cycle"]

    assert json.loads(source.read_text(encoding="utf-8")) == original


def test_pinned_prompt_dict_overrides_only_a_copy() -> None:
    prompt_dict = {
        "gate": {
            "prompt": ["Model: {nn_code}"],
            "dataset": "old",
            "nn_prefixes": ["old"],
        }
    }
    rendered = _pinned_prompt_dict(
        prompt_dict,
        ["gate"],
        dataset="cifar-10",
        nn_prefixes=("ga-", "moe-gate-cycle"),
    )

    assert rendered["gate"]["dataset"] == "cifar-10"
    assert rendered["gate"]["nn_prefixes"] == ["ga-", "moe-gate-cycle"]
    assert prompt_dict["gate"]["dataset"] == "old"
    assert prompt_dict["gate"]["nn_prefixes"] == ["old"]


def test_generated_prefix_is_added_once() -> None:
    args = SimpleNamespace(
        sft_nn_prefixes=["ga-", "GenFractalNet", "moe-gate-cycle"],
        generation_nn_prefixes=["GenFractalNet", "moe-gate-cycle"],
        nn_name_prefix="moe-gate-cycle",
    )
    assert _nas_prefixes(args) == ("ga-", "GenFractalNet", "moe-gate-cycle")
    assert _generation_prefixes(args) == ("GenFractalNet", "moe-gate-cycle")


def test_feedback_summary_reports_candidate_accuracy_delta(tmp_path: Path) -> None:
    epoch_path = tmp_path / "A0"
    candidate = epoch_path / "synth_nn" / "B0"
    candidate.mkdir(parents=True)
    (candidate / "eval_summary.json").write_text(
        json.dumps([{"epoch": 1, "accuracy": 0.62}]),
        encoding="utf-8",
    )
    pd.Series({
        "accuracy": 0.50,
        "dataset": "cifar-10",
        "task": "img-classification",
        "nn": "GenFractalNet-source",
    }).to_pickle(candidate / "dataframe.df")

    feedback = _build_gate_feedback_summary(
        epoch_path,
        used_prompts=[],
        cycle_results={"evaluation": {"models_trained": 1}},
        session=SimpleNamespace(gate_source="class Gate: pass", gate_factory_name=None),
    )

    assert feedback["mean_accuracy"] == 0.62
    assert feedback["best_accuracy"] == 0.62
    assert feedback["mean_accuracy_delta"] == 0.12
    assert feedback["improved_candidates"] == 1
    assert "delta=+0.1200" in feedback["summary"]


def test_train_prompt_uses_paired_upstream_config_and_generated_prefixes(
    monkeypatch,
) -> None:
    module = importlib.import_module("ab.gpt.util.prompt.NNGenPrompt")
    captured = {}

    def fake_data(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame([{
            "nn": "ga-source",
            "nn_code": "class Net: pass",
            "accuracy": 0.40,
            "epoch": 1,
            "dataset": "cifar-10",
            "task": "img-classification",
            "metric": "acc",
            "metric_code": "def acc(): pass",
            "transform_code": "def transform(): pass",
            "prm": '{"batch": 64}',
            "accuracy_2": 0.60,
            "nn_code_2": "class Net: improved = True",
            "transform_code_2": "def transform(): return 'improved'",
            "prm_2": '{"batch": 128}',
        }])

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False):
            assert tokenize is False
            return "\n".join(
                f"{message['role']}: {message['content']}" for message in messages
            )

    monkeypatch.setattr(module.lemur, "data", fake_data)
    monkeypatch.setattr(module, "patch_join_nn_query", lambda: None)
    monkeypatch.setattr(module, "enrich_dataframe", lambda data: None)

    config = (
        Path(__file__).resolve().parents[1]
        / "ab/gpt/conf/prompt/train/NN_gen.json"
    )
    with _runtime_prompt_config(
        config,
        dataset_name="cifar-10",
        nn_prefixes=("ga-", "GenFractalNet", "moe-gate-cycle"),
    ) as runtime_config:
        processor = module.NNGenPrompt(4096, FakeTokenizer(), runtime_config)
        raw = processor.get_raw_dataset(False, 1)

    assert len(raw) == 1
    assert "class Net: improved = True" in raw.iloc[0]["response"]
    assert captured["dataset"] == "cifar-10"
    assert captured["nn_prefixes"] == ("ga-", "GenFractalNet", "moe-gate-cycle")
