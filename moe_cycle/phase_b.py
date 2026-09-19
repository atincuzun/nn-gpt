"""Phase B: train the gate proposer itself on measured gate outcomes.

Phase A improves gate proposals in context only.  Phase B wraps the frozen
backbone with a LoRA adapter (attention projections only, never the MoE
routers) and, every batch of candidates, fits it to the accumulated
``gate_pairs`` so the LLM authors better gate code directly.

Two loss modes on each (lower, higher) pair:

- ``dpo``: preference loss with the adapter-disabled pass as the reference
  policy.  Losers teach the direction to avoid; the frozen backbone makes an
  explicit reference copy unnecessary.
- ``sft``: plain causal-LM loss on the higher-scoring gate source.

The adapter is never merged into the quantised backbone; it stays active for
generation.  Each gate record stores the proposer version (adapter state
fingerprint) it was measured under, so scores stay comparable within a
version and the incumbent is re-measured (king-of-the-hill) after each batch.
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


def wrap_proposer_with_lora(session: Any, args: Any) -> Any:
    """Attach a trainable LoRA adapter to the backbone.

    Must run before the ChatBot is constructed so generation goes through the
    adapter. Returns the PEFT-wrapped model, which becomes ``session.model``.
    Adapter parameters are created trainable but frozen immediately; Phase B
    training (and only Phase B training) unfreezes them.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    targets = {
        name.rsplit(".", 1)[-1]
        for name, _ in session.model.named_modules()
        if name.rsplit(".", 1)[-1] in PROPOSER_LORA_TARGETS
    }
    if not targets:
        raise RuntimeError("No attention projection modules found for the proposer LoRA")
    print(f"[PHASE B] proposer LoRA targets: {sorted(targets)}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.gate_sft_rank,
        lora_alpha=2 * args.gate_sft_rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=sorted(targets),
    )
    peft_model = get_peft_model(session.model, peft_config)
    for parameter in peft_model.parameters():
        parameter.requires_grad_(False)
    session.model = peft_model
    return peft_model


def llm_version(model: Any, args: Any) -> str:
    """Fingerprint of the proposer state that authored/measured gates.

    Covers the base model identity plus the current adapter weights, so every
    Phase B training batch produces a new version even though the backbone is
    frozen. Without an adapter (Phase A) it reduces to the model identity.
    """
    digest = hashlib.sha256()
    digest.update(str(getattr(args, "model", "")).encode("utf-8"))
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


def _tokenise_pair(pair: dict[str, Any], tokenizer: Any, shapes: Any) -> PairBatch:
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

    lower_ids, lower_labels = build(
        tokenizer(pair["lower_source"], add_special_tokens=False)["input_ids"]
    )
    higher_ids, higher_labels = build(
        tokenizer(completion + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    )
    return PairBatch(lower_ids, higher_ids, lower_labels, higher_labels)


def build_pair_batches(
    pairs: list[dict[str, Any]], tokenizer: Any, shapes: Any
) -> list[PairBatch]:
    return [_tokenise_pair(pair, tokenizer, shapes) for pair in pairs]


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
    model: Any,
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    args: Any,
    shapes: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """One Phase B training batch on the accumulated gate pairs.

    Trains only the LoRA parameters; the backbone stays frozen. The reference
    policy for DPO is the adapter-disabled pass over the same backbone. Saves
    the adapter under ``output_dir`` and returns a summary dict.
    """
    import torch
    import torch.nn.functional as F

    batches = build_pair_batches(pairs, tokenizer, shapes)
    if not batches:
        raise RuntimeError("Phase B received no tokenisable gate pairs")

    lora_parameters = [
        parameter for name, parameter in model.named_parameters() if ".lora_" in name
    ]
    if not lora_parameters:
        raise RuntimeError("Proposer model has no LoRA parameters to train")
    for parameter in lora_parameters:
        parameter.requires_grad_(True)
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
        if args.gate_sft_mode == "dpo":
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
        else:
            policy_higher = _completion_logprob_tensor(
                model, batch.higher_ids, batch.higher_labels
            )
            n_completion = sum(label != -100 for label in batch.higher_labels[1:])
            loss = -policy_higher / max(n_completion, 1)
        losses.append(float(loss.item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters, 1.0)
        optimizer.step()
        if step % 5 == 0 or step == args.gate_sft_steps:
            recent = sum(losses[-5:]) / len(losses[-5:])
            if args.gate_sft_mode == "dpo":
                print(
                    f"[PHASE B] step {step}/{args.gate_sft_steps} loss={recent:.6f} "
                    f"pair_accuracy={preferences_correct / step:.3f}"
                )
            else:
                print(f"[PHASE B] step {step}/{args.gate_sft_steps} loss={recent:.6f}")

    if hasattr(model, "disable_input_require_grads"):
        model.disable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    for parameter in lora_parameters:
        parameter.requires_grad_(False)
    model.eval()
    model.config.use_cache = True

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    summary = {
        "mode": args.gate_sft_mode,
        "steps": args.gate_sft_steps,
        "pairs": len(batches),
        "mean_loss": sum(losses) / len(losses),
        "pair_accuracy": (
            preferences_correct / args.gate_sft_steps if args.gate_sft_mode == "dpo" else None
        ),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[PHASE B] adapter saved: {output_dir}")
    return summary
