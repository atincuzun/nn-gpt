"""Regression coverage for pre-tokenized, completion-only proposer SFT."""

from __future__ import annotations

import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import PeftModel

from ab.gpt.util.llm.LoRA import LoRA
from moe_cycle.gate_pairs import build_gate_pairs
from moe_cycle.gate_store import write_gate_summary
from moe_cycle.phase_b import build_pair_batches, llm_version, wrap_proposer_with_lora
from moe_cycle.phase_b_runner import _run_phase_b_batch
from moe_gate_only import MoEGateSession

def _tiny_model_and_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        MixtralConfig,
        MixtralForCausalLM,
        PreTrainedTokenizerFast,
    )

    vocab = {
        "<unk>": 0, "<s>": 1, "</s>": 2, "gate": 3, "one": 4, "better": 5,
    }
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="<unk>", pad_token="</s>", bos_token="<s>", eos_token="</s>",
        model_input_names=["input_ids", "attention_mask"],
        chat_template="{% for m in messages %}{{ m['content'] }} {% endfor %}ASSISTANT:",
    )
    config = MixtralConfig(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        num_local_experts=4, num_experts_per_tok=2,
        pad_token_id=2, bos_token_id=1, eos_token_id=2, use_cache=False,
    )
    return MixtralForCausalLM(config), tokenizer


@pytest.mark.parametrize("gate_count", [2, 4])
def test_real_phase_b_sft_preserves_labels_and_persists_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate_count: int,
):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    # Accelerate probes optional DeepSpeed even without a DeepSpeed config;
    # under Python 3.12 -S the legacy package cannot import distutils.
    monkeypatch.setattr("accelerate.utils.other.is_deepspeed_available", lambda: False)
    monkeypatch.setenv("NNGPT_SEED_SFT_SPLIT", "1")
    torch.manual_seed(42)
    model, tokenizer = _tiny_model_and_tokenizer()
    pristine_model = deepcopy(model)
    session = MoEGateSession(model, tokenizer)
    routers = session.inspect()
    assert len(routers) == 2
    args = Namespace(
        model="tiny-mixtral", gate_sft_mode="sft", gate_sft_steps=2,
        gate_sft_lr=1e-3, gate_min_pairs=1, gate_store=tmp_path / "store", seed=42,
    )
    proposer = wrap_proposer_with_lora(session, args)
    tuner = session.proposer_tuner
    assert isinstance(tuner, LoRA)
    assert "q_proj" in tuner.peft_config.target_modules
    assert {"gate", "lm_head"}.isdisjoint(tuner.peft_config.target_modules)
    assert all(not p.requires_grad for p in proposer.parameters())
    tuner.training_args.output_dir = str(tmp_path / "trainer")
    tuner.training_args.per_device_train_batch_size = 2
    tuner.training_args.gradient_accumulation_steps = 1

    parent = llm_version(proposer, args)
    for i in range(gate_count):
        write_gate_summary(args.gate_store, i, {
            "gate_id": i, "llm_version": parent,
            "task": "img-classification", "dataset": "cifar-10", "metric": "accuracy",
            "score": {"eligible": True, "accuracy": 0.3 + i * 0.01},
            "gate_code": "gate " + "better " * (2 * i) + "one",
        })
    shapes = [(16, 4)]
    pairs = build_gate_pairs(args.gate_store)
    batches = build_pair_batches(pairs, tokenizer, shapes, include_lower=False)
    lengths = {len(batch.higher_ids) for batch in batches}
    assert len(pairs) == gate_count * (gate_count - 1) // 2
    args.max_length = max(lengths) - (2 if max(lengths) % 8 == 1 else 1)
    expected = {
        tuple(batch.higher_ids[-args.max_length:]): batch.higher_labels[-args.max_length:]
        for batch in batches
    }

    before = {
        name: parameter.detach().clone() for name, parameter in proposer.named_parameters()
    }
    train_rows = []

    def capture_training_batch(_module, _args, kwargs):
        assert all(".lora_" in n for n, p in proposer.named_parameters() if p.requires_grad)
        train_rows.extend(zip(
            kwargs["input_ids"].tolist(), kwargs["labels"].tolist(),
            kwargs["attention_mask"].tolist(),
        ))

    ctx = SimpleNamespace(
        args=args, session=session, tokenizer=tokenizer, shapes=shapes,
        gate_root=args.gate_store, llm_version=parent, run_root=tmp_path / "run",
    )
    with proposer.register_forward_pre_hook(capture_training_batch, with_kwargs=True):
        version = _run_phase_b_batch(ctx, 0)

    assert train_rows and any(0 in mask for _, _, mask in train_rows)
    for ids, labels, mask in train_rows:
        length = sum(mask)
        assert len(ids) % 8 == 0
        assert labels[:length] == expected[tuple(ids[:length])]
        assert labels[0] == -100 and labels[length - 1] == tokenizer.eos_token_id
        assert labels[length:] == [-100] * (len(ids) - length)
        assert ids[length:] == [tokenizer.pad_token_id] * (len(ids) - length)
    changed = {n for n, p in proposer.named_parameters() if not torch.equal(p.detach(), before[n])}
    assert changed and all(".lora_" in n for n in changed)
    assert [site.gate for site in session.inspect()] == [site.gate for site in routers]
    assert all(not p.requires_grad for p in proposer.parameters())
    assert not proposer.training and proposer.config.use_cache
    assert version == llm_version(proposer, args, merged_from=parent) != parent

    out = ctx.run_root / "proposer_adapters" / "batch_000"
    summary = json.loads((out / "training_summary.json").read_text())
    assert summary["steps"] == 2 and summary["pairs"] == len(pairs)
    assert (out / "tokenizer_config.json").is_file()
    persisted = args.gate_store / "proposer"
    lineage = json.loads((persisted / "lineage.json").read_text())
    assert lineage["parent_version"] == parent and lineage["version"] == version

    reloaded = PeftModel.from_pretrained(deepcopy(pristine_model), persisted / "adapter").eval()
    saved_weights = dict(reloaded.named_parameters())
    assert all(torch.equal(p, saved_weights[n]) for n, p in proposer.named_parameters() if ".lora_" in n)
    resumed = MoEGateSession(pristine_model, tokenizer)
    continued = wrap_proposer_with_lora(resumed, args, merge_from=persisted / "adapter")
    sample = tokenizer("gate one", return_tensors="pt")
    with torch.no_grad():
        torch.testing.assert_close(continued.eval()(**sample).logits, reloaded(**sample).logits)
    assert all(not p.requires_grad for p in continued.parameters())
