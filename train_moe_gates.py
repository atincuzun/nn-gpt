"""Standalone gate-only training for a Hugging Face MoE causal LM.

This script intentionally has no dependency on the NNGPT Tune pipeline. It
demonstrates the complete model load, router discovery, replacement, freezing,
training, checkpointing, and restoration lifecycle.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from moe_gate_only import (
    MoEGateSession,
    build_nngenprompt_dataloaders,
    train_gates,
)


DEFAULT_TEXTS = (
    "Write a PyTorch function that implements scaled dot-product attention.",
    "Implement a residual multilayer perceptron module using torch.nn.Module.",
    "Create a training loop with AdamW, gradient clipping, and validation loss.",
    "Define a convolutional classifier for ten image categories in PyTorch.",
)


class CausalTextDataset(Dataset):
    def __init__(self, texts: list[str]) -> None:
        if not texts:
            raise ValueError("At least one training text is required")
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        return self.texts[index]


class CausalTextCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    parser.add_argument(
        "--gate",
        choices=("linear", "low_rank", "mlp", "residual_mlp", "fourier"),
        default="low_rank",
    )
    parser.add_argument("--gate-source", type=Path, help="Python file defining LLMGeneratedGate")
    parser.add_argument("--gate-class", default="LLMGeneratedGate")
    parser.add_argument("--checkpoint", type=Path, help="Existing gate checkpoint to resume")
    parser.add_argument(
        "--data-source",
        choices=("nngenprompt", "text"),
        default="nngenprompt",
        help="Use NNGPT's formatted LEMUR data or a plain-text fallback",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=Path("ab/gpt/conf/prompt/train/NN_gen.json"),
    )
    parser.add_argument("--train-text", type=Path, help="UTF-8 file with one example per non-empty line")
    parser.add_argument("--output", type=Path, default=Path("out/moe_gate_standalone"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--only-best-accuracy", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map or 'none'")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_texts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_TEXTS)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def print_sites(sites: list[Any]) -> None:
    print(f"Discovered {len(sites)} MoE gate sites")
    for site in sites:
        print(
            f"  layer={site.layer_index:<3} path={site.path or '<unknown>'} "
            f"pattern={site.pattern} shape={site.model_dim}->{site.num_experts}"
        )


def main() -> None:
    args = parse_args()
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype_from_name(args.dtype),
        "local_files_only": args.local_files_only,
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map

    print(f"Loading {args.model}")
    session = MoEGateSession.from_pretrained(
        args.model,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"local_files_only": args.local_files_only},
    )
    tokenizer = session.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.gradient_checkpointing:
        session.model.gradient_checkpointing_enable()
    if hasattr(session.model, "config"):
        session.model.config.use_cache = False

    if args.data_source == "nngenprompt":
        print(f"Building formatted NN generation data from {args.prompt_config}")
        dataloader, validation_loader, dataset = build_nngenprompt_dataloaders(
            tokenizer,
            args.prompt_config,
            context_length=args.max_length,
            max_prompts=args.max_prompts,
            max_new_tokens=args.max_new_tokens,
            only_best_accuracy=args.only_best_accuracy,
            batch_size=args.batch_size,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        print(f"NNGenPrompt produced {len(dataset)} training examples")
        sample_batch = next(iter(dataloader))
    else:
        texts = load_texts(args.train_text)
        collator = CausalTextCollator(tokenizer, args.max_length)
        dataloader = DataLoader(
            CausalTextDataset(texts),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        validation_loader = None
        sample_batch = collator([texts[0]])
    discovery_input = {
        "input_ids": sample_batch["input_ids"],
        "attention_mask": sample_batch["attention_mask"],
    }

    sites = session.inspect()
    print_sites(sites)
    if args.inspect_only:
        return

    with session:
        replace_kwargs = {
            "layers": args.layers,
            "sample_input": discovery_input,
            "verify": True,
            "top_k": args.top_k,
            "allow_remote_code": True,
            "dynamic_discovery": False,
        }
        gate_source = args.gate_source
        if gate_source is None and args.checkpoint is not None:
            checkpoint_source = args.checkpoint / "gate.py"
            if checkpoint_source.is_file():
                gate_source = checkpoint_source
        if gate_source is not None:
            source = gate_source.read_text(encoding="utf-8")
            session.replace_source(source, class_name=args.gate_class, **replace_kwargs)
        else:
            session.replace(args.gate, **replace_kwargs)

        session.freeze_except_gates()
        if args.gradient_checkpointing:
            # Reentrant checkpointing needs one grad-bearing input when all
            # upstream model parameters (including embeddings) are frozen.
            session.model.enable_input_require_grads()
        if args.checkpoint is not None:
            session.load_weights(args.checkpoint)
            print(f"Loaded replacement-gate weights from {args.checkpoint}")
        summary = session.summary()
        print(
            f"Installed {summary.installed_gates} gates; trainable parameters: "
            f"{summary.trainable_parameters:,}/{summary.total_parameters:,}"
        )

        def report_step(step: int, loss: float) -> None:
            print(f"step={step:04d} loss={loss:.6f}")

        result, optimizer = train_gates(
            session.model,
            session.installs,
            dataloader,
            steps=args.steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            validation_loader=validation_loader,
            validation_steps=args.validation_steps,
            on_step=report_step,
        )
        checkpoint = session.save(args.output, optimizer=optimizer)
        print(f"Mean training loss: {result.mean_train_loss:.6f}")
        if result.validation_loss is not None:
            print(f"Validation loss: {result.validation_loss:.6f}")
        print(f"Saved replacement gates to {checkpoint}")

    if args.gradient_checkpointing:
        session.model.disable_input_require_grads()
    print("Original gates and parameter trainability restored")


if __name__ == "__main__":
    main()
