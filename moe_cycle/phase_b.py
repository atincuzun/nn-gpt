"""Outer-loop SFT: train the gate proposer itself on measured gate outcomes.

Reuses the ORIGINAL LoRA path machinery from ``ab/gpt/util/LoRA.py`` rather
than reimplementing it: adapter attachment goes through the same ``LoRA``
class (``prepare_model_for_kbit_training``, gradient checkpointing,
``get_peft_model``) with the same default config (``create_peft_config``:
DoRA r=32, alpha=64, dropout=0.1 over all linear modules), and SFT training
runs through the same ``LoRA.train`` (TRL SFTTrainer, completions-only
masking, validation split, per-call adapter saving).

Deliberate deviations, each required by the gate cycle:
- Router modules stay adapter-free: ``find_all_linear_names`` results exclude
  ``gate`` for the same reason LoRA.py itself excludes ``lm_head`` — the
  routing head is what candidate gates replace and must never be adapted.
- Cross-run continuation follows the Tune.py pattern (L1139-1142): the newest
  adapter persisted under ``--gate-store/proposer`` is merged into the base
  before a fresh adapter is attached, so proposer weights compound across runs.
- SFT examples are LEMUR-CV style: the lower-scoring gate's code + accuracy
  are the prompt context, the higher-scoring gate is the completion.
- DPO mode (opt-in, ``--gate-sft-mode dpo``) keeps a manual contrastive loop:
  it needs pairwise policy/reference log-probabilities that SFTTrainer does
  not expose.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Attention projections only. The MoE router ("gate") and expert MLPs must
# never receive adapters: routers are replaced by the candidate gates and
# adapters there would confound the measurement.
PROPOSER_LORA_TARGETS = (
    "in_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "out_proj",
)


def wrap_proposer_with_lora(session: Any, args: Any, merge_from: Path | None = None) -> Any:
    """Attach the proposer LoRA using the original LoRA path machinery.

    When ``merge_from`` points at a previously saved proposer adapter, it is
    merged into the base model first (Tune.py L1139-1142 pattern) so weights
    compound across runs; a fresh adapter is then attached for this run.
    Must run before the ChatBot is constructed. Returns the PEFT-wrapped
    model, which becomes ``session.model``; the tuner is stashed on
    ``session.proposer_tuner`` for the SFT batches.
    """
    from ab.gpt.util.LoRA import LoRA as ProposerTuner, create_peft_config, find_all_linear_names
    from trl import SFTConfig

    model = session.model
    if merge_from is not None:
        from peft import PeftModel

        print(f"[GATE SEARCH] outer-loop SFT: merging persisted proposer adapter {merge_from}")
        model = PeftModel.from_pretrained(model, merge_from, is_trainable=True).merge_and_unload()

    # Same target discovery as Tune.py's default, minus the routing head:
    # "gate" is excluded for the same reason LoRA.py excludes "lm_head".
    targets = sorted(set(find_all_linear_names(model)) - {"gate"})
    if not targets:
        raise RuntimeError("No adapter-able linear modules found for the proposer")
    print(f"[GATE SEARCH] outer-loop SFT: {len(targets)} adapter target names (routers excluded)")
    peft_config = create_peft_config(targets)
    training_args = SFTConfig(
        output_dir="proposer_sft_tmp",
        max_steps=args.gate_sft_steps,
        learning_rate=args.gate_sft_lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        save_strategy="no",
        logging_steps=5,
        report_to=[],
        seed=args.seed,
    )
    tuner = ProposerTuner(
        model,
        session.tokenizer,
        training_args=training_args,
        peft_config=peft_config,
        use_unsloth=False,
    )
    for parameter in tuner.peft_model.parameters():
        parameter.requires_grad_(False)
    session.model = tuner.peft_model
    session.proposer_tuner = tuner
    return tuner.peft_model


def llm_version(model: Any, args: Any, merged_from: str | None = None) -> str:
    """Fingerprint of the proposer state that authored/measured gates.

    Covers the base model identity, the lineage it continues from, and the
    current adapter weights, so every training batch produces a new version
    even though the backbone is frozen. Without an adapter it reduces to the
    model identity plus merge lineage.
    """
    digest = hashlib.sha256()
    digest.update(str(getattr(args, "model", "")).encode("utf-8"))
    if merged_from:
        digest.update(f"continued-from:{merged_from}".encode("utf-8"))
    if getattr(model, "peft_config", None) is not None:
        for name, parameter in sorted(model.named_parameters()):
            if ".lora_" in name:
                digest.update(name.encode("utf-8"))
                digest.update(parameter.detach().float().cpu().numpy().tobytes())
        return f"lora-{digest.hexdigest()[:16]}"
    return f"base-{digest.hexdigest()[:16]}"


@dataclass
class PairBatch:
    """Tokenised (lower, higher) pair with completion labels masked to -100."""

    lower_ids: list[int]
    higher_ids: list[int]
    lower_labels: list[int]
    higher_labels: list[int]


def _tokenise_pair(
    pair: dict[str, Any], tokenizer: Any, shapes: Any, *, include_lower: bool = True
) -> PairBatch:
    from .gate_prompt import gate_sft_examples

    example = gate_sft_examples([pair], shapes)[0]
    messages = example["messages"]
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True,
    )
    completion = messages[-1]["content"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    def build(completion_ids: list[int]) -> tuple[list[int], list[int]]:
        ids = list(prompt_ids) + list(completion_ids)
        labels = [-100] * len(prompt_ids) + list(completion_ids)
        return ids, labels

    # The lower side is only needed for the DPO contrast; SFT imitates the
    # winner alone (the loser text is embedded in the shared prompt), so
    # tokenising it there would be dead work.
    if include_lower:
        lower_ids, lower_labels = build(
            tokenizer(pair["lower_source"], add_special_tokens=False)["input_ids"]
        )
    else:
        lower_ids, lower_labels = [], []
    higher_ids, higher_labels = build(
        tokenizer(completion + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    )
    return PairBatch(lower_ids, higher_ids, lower_labels, higher_labels)


def build_pair_batches(
    pairs: list[dict[str, Any]], tokenizer: Any, shapes: Any, *,
    include_lower: bool = True,
) -> list[PairBatch]:
    return [
        _tokenise_pair(pair, tokenizer, shapes, include_lower=include_lower)
        for pair in pairs
    ]


def _completion_logprob_tensor(model: Any, input_ids: list[int], labels: list[int]):
    """Sum (as a tensor) of token log-probabilities on unmasked labels."""
    import torch

    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], device=device)
    logits = model(input_ids=ids).logits[0, :-1].float()
    targets = torch.tensor([labels], device=device)[0, 1:]
    mask = targets != -100
    if not mask.any():
        return torch.zeros((), device=device, requires_grad=True)
    log_probs = torch.log_softmax(logits[mask], dim=-1)
    picked = targets[mask]
    return log_probs.gather(1, picked.unsqueeze(1)).sum()


def train_proposer(
    session: Any,
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    args: Any,
    shapes: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """One outer-loop SFT batch on the accumulated gate pairs.

    SFT mode (default) delegates entirely to ``ab.gpt.util.LoRA.LoRA.train``:
    TRL SFTTrainer, completions-only masking, validation split, adapter and
    tokenizer saving — the same machinery the CV pipeline tunes with. DPO mode
    keeps the manual contrastive loop (policy vs adapter-disabled reference).
    Only LoRA parameters ever train; the backbone stays frozen.
    """
    import torch
    import torch.nn.functional as F

    model = session.model
    batches = build_pair_batches(
        pairs, tokenizer, shapes, include_lower=(args.gate_sft_mode == "dpo"),
    )
    if not batches:
        raise RuntimeError("Outer-loop SFT received no tokenisable gate pairs")

    lora_parameters = [
        parameter for name, parameter in model.named_parameters() if ".lora_" in name
    ]
    if not lora_parameters:
        raise RuntimeError("Proposer model has no LoRA parameters to train")
    for parameter in lora_parameters:
        parameter.requires_grad_(True)

    if args.gate_sft_mode == "dpo":
        optimizer = torch.optim.AdamW(lora_parameters, lr=args.gate_sft_lr)
        model.train()
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        beta = 0.1
        losses: list[float] = []
        preferences_correct = 0
        for step in range(1, args.gate_sft_steps + 1):
            batch = batches[(step - 1) % len(batches)]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), model.disable_adapter():
                ref_higher = _completion_logprob_tensor(
                    model, batch.higher_ids, batch.higher_labels
                ).item()
                ref_lower = _completion_logprob_tensor(
                    model, batch.lower_ids, batch.lower_labels
                ).item()
            policy_higher = _completion_logprob_tensor(
                model, batch.higher_ids, batch.higher_labels
            )
            policy_lower = _completion_logprob_tensor(
                model, batch.lower_ids, batch.lower_labels
            )
            margin = beta * (
                (policy_higher - ref_higher) - (policy_lower - ref_lower)
            )
            loss = -F.logsigmoid(margin)
            if policy_higher.item() > policy_lower.item():
                preferences_correct += 1
            losses.append(float(loss.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_parameters, 1.0)
            optimizer.step()
            if step % 5 == 0 or step == args.gate_sft_steps:
                recent = sum(losses[-5:]) / len(losses[-5:])
                print(
                    f"[GATE SEARCH] outer-loop SFT (dpo) step {step}/{args.gate_sft_steps} "
                    f"loss={recent:.6f} pair_accuracy={preferences_correct / step:.3f}"
                )
        summary = {
            "mode": "dpo",
            "steps": args.gate_sft_steps,
            "pairs": len(batches),
            "mean_loss": sum(losses) / len(losses),
            "pair_accuracy": preferences_correct / args.gate_sft_steps,
        }
    else:
        tuner = getattr(session, "proposer_tuner", None)
        if tuner is None:
            raise RuntimeError(
                "Outer-loop SFT needs session.proposer_tuner (attach it with "
                "wrap_proposer_with_lora before training)"
            )
        from datasets import Dataset as HFDataset

        from .gate_prompt import gate_sft_examples

        rows = []
        for pair in pairs:
            example = gate_sft_examples([pair], shapes)[0]
            prompt_text = tokenizer.apply_chat_template(
                example["messages"][:-1], tokenize=False, add_generation_prompt=True,
            )
            rows.append({
                "prompt": prompt_text,
                "completion": example["messages"][-1]["content"] + tokenizer.eos_token,
            })
        # Delegates to ab.gpt.util.LoRA.LoRA.train: SFTTrainer handles the
        # scheduler, gradient accumulation, the completions-only collator,
        # the validation split, and adapter/tokenizer saving.
        tuner.train(
            HFDataset.from_list(rows),
            tokenizer,
            str(Path(output_dir)),
            train_on_completions_only=True,
        )
        summary = {
            "mode": "sft",
            "steps": args.gate_sft_steps,
            "pairs": len(rows),
            "trainer": "ab.gpt.util.LoRA.LoRA.train (TRL SFTTrainer)",
        }

    for parameter in lora_parameters:
        parameter.requires_grad_(False)
    model.eval()
    model.config.use_cache = True

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    summary["output_dir"] = str(output_dir)
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[GATE SEARCH] outer-loop SFT: adapter saved {output_dir}")
    return summary
