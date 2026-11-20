"""Tests for the outer-loop SFT collator compatibility bridge.

trl 0.12.2 removed the DataCollatorForLanguageModeling keyword arguments
(pad_token_id, completion_only_loss) that upstream LoRA.py passes; the
outer-loop SFT bridges that with a padding collator plus a version-adaptive
context manager, feeding LoRA.train pre-tokenized rows.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch

import ab.gpt.util.LoRA as lora_module
from moe_cycle.phase_b import (
    _LegacyKwargsPaddingCollator,
    _legacy_trl_collator_compat,
)


# ── padding collator ─────────────────────────────────────────────────────────

def test_collator_accepts_the_exact_legacy_kwargs_lora_py_passes():
    # This is literally the call LoRA.py makes; it must not raise.
    collator = _LegacyKwargsPaddingCollator(
        pad_token_id=7,
        completion_only_loss=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    batch = collator([
        {"input_ids": [1, 2, 3], "labels": [-100, -100, 3]},
        {"input_ids": [4, 5], "labels": [-100, 5]},
    ])
    # Padded to a multiple of 8 with pad_token_id / -100 / zero-mask.
    assert batch["input_ids"].shape == (2, 8)
    assert batch["input_ids"][0, 3:].tolist() == [7] * 5
    assert batch["labels"][0, 3:].tolist() == [-100] * 5
    assert batch["labels"][1].tolist() == [-100, 5] + [-100] * 6
    assert batch["attention_mask"][1].tolist() == [1, 1] + [0] * 6


def test_collator_preserves_completion_only_labels_unchanged():
    collator = _LegacyKwargsPaddingCollator(pad_token_id=0)
    batch = collator([{"input_ids": [9, 8, 7], "labels": [-100, 8, 7]}])
    # The completion-only masking lives in the labels; padding never rewrites
    # supervised positions.
    assert batch["labels"][0].tolist() == [-100, 8, 7]


# ── version-adaptive context manager ─────────────────────────────────────────

def test_compat_patches_only_when_legacy_kwargs_are_missing():
    # The installed trl (0.12.2) lacks the kwargs: the swap must happen and
    # the original class must be restored afterwards.
    real = lora_module.DataCollatorForLanguageModeling
    with _legacy_trl_collator_compat():
        patched = lora_module.DataCollatorForLanguageModeling
        if patched is not real:  # legacy kwargs unsupported in this venv
            assert patched is _LegacyKwargsPaddingCollator
    assert lora_module.DataCollatorForLanguageModeling is real


def test_compat_is_a_noop_when_legacy_kwargs_are_supported(monkeypatch):
    class _OldTrlCollator:
        def __init__(self, pad_token_id=None, completion_only_loss=True,
                     pad_to_multiple_of=None, return_tensors="pt"):
            pass

    monkeypatch.setattr(
        lora_module, "DataCollatorForLanguageModeling", _OldTrlCollator,
    )
    with _legacy_trl_collator_compat():
        # A pipeline-side fix or older trl must keep its own collator.
        assert lora_module.DataCollatorForLanguageModeling is _OldTrlCollator


def test_compat_restores_the_original_even_on_error():
    real = lora_module.DataCollatorForLanguageModeling
    with pytest.raises(RuntimeError):
        with _legacy_trl_collator_compat():
            raise RuntimeError("boom")
    assert lora_module.DataCollatorForLanguageModeling is real


# ── end-to-end: the real LoRA.train through the bridge ───────────────────────

def _tiny_model_and_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        MixtralConfig,
        MixtralForCausalLM,
        PreTrainedTokenizerFast,
    )

    vocab = {
        "<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3, "gate": 4, "one": 5,
        "two": 6, "better": 7, "ASSISTANT:": 8, "code": 9, "new": 10,
    }
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="<unk>", pad_token="<pad>", bos_token="<s>", eos_token="</s>",
        chat_template="{% for m in messages %}{{ m['content'] }} {% endfor %}ASSISTANT:",
    )
    config = MixtralConfig(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        num_local_experts=4, num_experts_per_tok=2,
        pad_token_id=1, bos_token_id=2, eos_token_id=3, use_cache=False,
    )
    return MixtralForCausalLM(config), tokenizer


def test_train_proposer_sft_mode_runs_the_real_lora_train(tmp_path: Path):
    """The full SFT delegation survives the trl 0.12.2 collator removal.

    Mirrors train_proposer's SFT branch: pre-tokenized rows from
    build_pair_batches fed to the REAL ab.gpt.util.LoRA.LoRA.train under
    _legacy_trl_collator_compat. Before the bridge this exact call died
    with TypeError('pad_token_id').
    """
    from datasets import Dataset as HFDataset
    from trl import SFTConfig

    from ab.gpt.util.LoRA import LoRA, create_peft_config, find_all_linear_names
    from moe_cycle.phase_b import build_pair_batches

    model, tokenizer = _tiny_model_and_tokenizer()
    training_args = SFTConfig(
        output_dir=str(tmp_path / "sft"),
        max_steps=2,
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        bf16=False,
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        seed=42,
        use_cpu=True,  # a gate-cycle run may be using the only GPU
    )
    targets = sorted(set(find_all_linear_names(model)) - {"gate"})
    tuner = LoRA(
        model,
        tokenizer,
        training_args=training_args,
        peft_config=create_peft_config(targets),
        use_unsloth=False,
    )

    pairs = [
        {
            "pair_id": f"{i:03d}->{i + 1:03d}",
            "lower_gate_id": i,
            "higher_gate_id": i + 1,
            "lower_source": "gate one",
            "higher_source": "gate two better",
            "lower_accuracy": 0.30 + i * 0.01,
            "higher_accuracy": 0.40 + i * 0.01,
            "accuracy_gap": 0.10,
            "task": "img-classification",
            "dataset": "cifar-10",
            "metric": "accuracy",
        }
        for i in range(6)
    ]
    args = Namespace(gate_sft_steps=2, gate_sft_mode="sft", max_length=64)
    batches = build_pair_batches(pairs, tokenizer, [(16, 4)], include_lower=False)
    rows = [
        {"input_ids": batch.higher_ids, "labels": batch.higher_labels}
        for batch in batches
    ]
    assert rows, "pair tokenisation produced no rows"

    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in tuner.peft_model.named_parameters()
        if ".lora_" in name
    }
    with _legacy_trl_collator_compat():
        tuner.train(
            HFDataset.from_list(rows),
            tokenizer,
            str(tmp_path / "out"),
            train_on_completions_only=True,
        )
    changed = [
        name
        for name, parameter in tuner.peft_model.named_parameters()
        if ".lora_" in name
        and not torch.equal(parameter.detach().cpu(), before[name])
    ]
    assert changed, "training ran but no LoRA parameter moved"
