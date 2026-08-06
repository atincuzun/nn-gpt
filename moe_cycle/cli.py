"""CLI argument parsing for the MoE gate-training NAS cycle."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-8B-A1B")
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Optional PEFT LoRA adapter to merge into the frozen teacher model",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of pipeline generation/evaluation/training epochs")
    parser.add_argument("--test-nn", type=int, default=10)
    parser.add_argument("--nn-train-epochs", type=int, default=3)
    parser.add_argument("--gate-train-steps", type=int, default=50)
    parser.add_argument("--gate-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--gate-implementation",
        choices=("llm_generated",),
        default="llm_generated",
        help="Gate source mechanism: llm_generated asks the model to write the gate. "
        "An externally supplied gate file takes precedence when --gate-source is given",
    )
    parser.add_argument(
        "--gate-source",
        type=Path,
        help="Python file defining the replacement gate class (external gate; "
        "overrides LLM generation)",
    )
    parser.add_argument(
        "--gate-class",
        default="LLMGeneratedGate",
        help="Class name to load from --gate-source",
    )
    parser.add_argument(
        "--gate-mode",
        choices=("teacher-student", "direct"),
        default="direct",
        help="Train a random shadow student (DeepSeek-V2 routers only) or the direct "
        "replacement initialized from the native router",
    )
    parser.add_argument(
        "--gate-init-noise-scale",
        type=float,
        default=0.0,
        help="Gaussian initialization noise as a fraction of each native gate weight std",
    )
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--distillation-temperature", type=float, default=1.0)
    parser.add_argument("--student-weight-start", type=float, default=0.0)
    parser.add_argument("--student-weight-step", type=float, default=0.1)
    parser.add_argument(
        "--handoff-mode",
        choices=("guarded", "never", "fixed"),
        default="guarded",
        help="Increase student routing control only after imitation succeeds, never, or every epoch",
    )
    parser.add_argument("--handoff-min-topk-overlap", type=float, default=0.95)
    parser.add_argument("--handoff-max-kl", type=float, default=0.1)
    parser.add_argument("--handoff-max-validation-loss-increase", type=float, default=0.05)
    parser.add_argument("--gate-generation-attempts", type=int, default=3)
    parser.add_argument("--gate-max-new-tokens", type=int, default=1024)
    parser.add_argument("--generation-max-new-tokens", type=int, default=16384)
    parser.add_argument(
        "--generation-max-input-length",
        type=int,
        help="Optional input-only prompt limit; unset by default for non-Unsloth generation",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompts", type=int, default=4096)
    parser.add_argument(
        "--train-prompt-config",
        default="NN_gate.json",
        help="Feedback-aware prompt config under ab/gpt/conf/prompt/train",
    )
    parser.add_argument(
        "--test-prompt-config",
        default="NN_gate.json",
        help="Feedback-aware prompt config under ab/gpt/conf/prompt/test",
    )
    parser.add_argument("--dataset", default="cifar-10")
    parser.add_argument(
        "--sft-nn-prefixes",
        nargs="+",
        default=["ga-", "GenFractalNet"],
        help="LEMUR model prefixes used for paired gate-training examples. "
        "The current --nn-name-prefix is added automatically.",
    )
    parser.add_argument(
        "--generation-nn-prefixes",
        nargs="+",
        default=["GenFractalNet"],
        help="One-record-per-model LEMUR prefixes used as comparable CV generation seeds. "
        "The current --nn-name-prefix is added automatically.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Deprecated compatibility option; ignored by the upstream ChatBot",
    )
    parser.add_argument("--router-top-k", type=int)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument(
        "--progressive-unfreeze-descending",
        action="store_true",
        help="Train the highest selected layer first, then add one earlier layer per epoch",
    )
    parser.add_argument(
        "--conf-keys",
        nargs="+",
        default=["improve_classification_gate_feedback"],
    )
    parser.add_argument("--nn-name-prefix", default="moe-gate-cycle")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Quantize the frozen base model with bitsandbytes (saves ~50%% VRAM). "
        "The gate starts randomly (int8 router weights cannot seed the copy), "
        "step-zero equivalence is recorded but not enforced, and teacher-student "
        "mode is unavailable",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generation-backend",
        choices=("pipeline", "direct"),
        default="pipeline",
        help="Deprecated compatibility option; upstream ChatBot selects its backend",
    )
    parser.add_argument(
        "--fixed-evaluation-prompts",
        action="store_true",
        help="Deprecated compatibility option; fixed prompt reuse is disabled",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--no-gate-feedback",
        action="store_true",
        help="Omit cycle feedback text from generation and training prompts "
        "(feedback is still built and saved to disk). Useful for isolating "
        "the effect of explicit text feedback vs implicit LEMUR DB feedback.",
    )
    return parser.parse_args()
