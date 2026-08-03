"""Compare a DeepSeek LoRA adapter with its base model on held-out prompts."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftConfig, PeftModel

from moe_gate_only import build_nngenprompt_dataloaders
from moe_gate_only.training import evaluate_language_model_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="Override the adapter's recorded base model")
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=Path("ab/gpt/conf/prompt/train/NN_gen.json"),
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-prompts", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 709 else math.inf


def main() -> None:
    args = parse_args()
    if not args.adapter.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {args.adapter}")
    if args.validation_steps < 0:
        raise ValueError("validation-steps cannot be negative")
    seed_all(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    peft_config = PeftConfig.from_pretrained(
        args.adapter,
        local_files_only=args.local_files_only,
    )
    base_model_name = args.model or peft_config.base_model_name_or_path
    tokenizer_source = args.adapter if (args.adapter / "tokenizer_config.json").exists() else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
        is_trainable=False,
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False

    _, validation_loader, dataset = build_nngenprompt_dataloaders(
        tokenizer,
        args.prompt_config,
        context_length=args.max_length,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if validation_loader is None:
        raise RuntimeError("No validation examples were produced; increase max-prompts or validation-fraction")
    max_steps = args.validation_steps or None

    with model.disable_adapter():
        base_loss = evaluate_language_model_loss(model, validation_loader, max_steps=max_steps)
    adapter_loss = evaluate_language_model_loss(model, validation_loader, max_steps=max_steps)
    absolute_change = adapter_loss - base_loss
    relative_improvement = (base_loss - adapter_loss) / base_loss * 100 if base_loss else 0.0

    result = {
        "completed_at": datetime.now().isoformat(),
        "base_model": base_model_name,
        "adapter": str(args.adapter.resolve()),
        "prompt_config": str(args.prompt_config),
        "dataset_examples": len(dataset),
        "validation_examples": len(validation_loader.dataset),
        "validation_steps": max_steps,
        "seed": args.seed,
        "base": {"loss": base_loss, "perplexity": perplexity(base_loss)},
        "finetuned": {"loss": adapter_loss, "perplexity": perplexity(adapter_loss)},
        "loss_change": absolute_change,
        "loss_improvement_percent": relative_improvement,
        "winner": "finetuned" if adapter_loss < base_loss else "base" if base_loss < adapter_loss else "tie",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
