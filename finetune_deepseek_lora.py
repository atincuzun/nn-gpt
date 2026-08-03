"""Fine-tune a supported causal language model on NNGenPrompt with LoRA."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

from moe_gate_only import build_nngenprompt_dataloaders
from moe_gate_only.training import evaluate_language_model_loss, model_input_device, move_batch_to_device


ATTENTION_TARGETS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-config", type=Path, default=Path("ab/gpt/conf/prompt/train/NN_gen.json"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-prompts", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-steps", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def discover_targets(model: nn.Module) -> list[str]:
    targets = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in ATTENTION_TARGETS
    }
    if not targets:
        raise RuntimeError("No supported attention or hybrid projection modules were found")
    return sorted(targets)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.gradient_accumulation < 1:
        raise ValueError("steps and gradient-accumulation must be positive")
    seed_all(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    targets = discover_targets(model)
    print(f"LoRA target modules: {targets}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=targets,
    )
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_loader, validation_loader, dataset = build_nngenprompt_dataloaders(
        tokenizer,
        args.prompt_config,
        context_length=args.max_length,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    device = model_input_device(model)
    iterator = iter(train_loader)
    losses: list[float] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = move_batch_to_device(batch, device)
            loss = model(**batch).loss / args.gradient_accumulation
            loss.backward()
            accumulated_loss += float(loss.detach().item())
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(accumulated_loss)
        print(f"lora_step={step:04d} loss={accumulated_loss:.6f}")

    validation_loss = None
    if validation_loader is not None:
        validation_loss = evaluate_language_model_loss(
            model,
            validation_loader,
            max_steps=args.validation_steps,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    torch.save(optimizer.state_dict(), args.output / "optimizer.pt")
    metrics = {
        "completed_at": datetime.now().isoformat(),
        "base_model": args.model,
        "dataset_examples": len(dataset),
        "steps": args.steps,
        "gradient_accumulation": args.gradient_accumulation,
        "mean_train_loss": sum(losses) / len(losses),
        "final_train_loss": losses[-1],
        "validation_loss": validation_loss,
        "target_modules": targets,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    (args.output / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved NNGPT LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
