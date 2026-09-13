"""CLI argument parsing for the MoE gate-training NAS cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ab.gpt.util.Const import DEFAULT_NN_PREFIXES


def _json_object(value: str) -> dict[str, object]:
    """Parse one command-line JSON object with an argparse-friendly error."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"expected a JSON object, got {type(parsed).__name__}"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-8B-A1B")
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Optional PEFT LoRA adapter to merge into the frozen model",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of pipeline generation/evaluation/training epochs")
    parser.add_argument("--test-nn", type=int, default=10)
    parser.add_argument("--nn-train-epochs", type=int, default=3)
    parser.add_argument(
        "--fixed-eval-hyperparameters",
        "--eval-prm-json",
        dest="fixed_eval_hyperparameters",
        type=_json_object,
        metavar="JSON",
        help="JSON object applied last to every generated NN evaluation, overriding "
        "values from generated hp.txt and the source dataframe. Evaluation epoch is "
        "controlled separately by --nn-train-epochs.",
    )
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
        choices=("direct",),
        default="direct",
        help="Deprecated compatibility option; the current cycle always trains the direct replacement gate",
    )
    parser.add_argument(
        "--gate-init-noise-scale",
        type=float,
        default=0.0,
        help="Gaussian initialization noise as a fraction of each native gate weight std",
    )
    parser.add_argument("--gate-generation-attempts", type=int, default=3)
    parser.add_argument(
        "--gate-candidates",
        type=int,
        default=1,
        help="Independent outer-loop gate architectures to generate and evaluate",
    )
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
        default=list(DEFAULT_NN_PREFIXES),
        help="LEMUR model prefixes used for paired gate-training examples. "
        "Defaults to the full LEMUR family census (see ab.gpt.util.Const). "
        "The current --nn-name-prefix is added automatically.",
    )
    parser.add_argument(
        "--generation-nn-prefixes",
        nargs="+",
        default=list(DEFAULT_NN_PREFIXES),
        help="LEMUR prefixes used as CV generation seeds (nn_gen samples one "
        "record per model). Defaults to the full LEMUR family census "
        "(see ab.gpt.util.Const). The current --nn-name-prefix is added "
        "automatically.",
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
        help="Quantize the frozen backbone weights (plain nn.Linear modules) "
        "with bitsandbytes int8 (~50%% VRAM saving). Router parameters are not "
        "quantized, so the replacement gates are still seeded by an exact "
        "copy of the native router.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Quantize the frozen backbone with bitsandbytes NF4/QLoRA-style "
        "4-bit (~75%% VRAM saving vs bf16). Same caveats as --load-in-8bit; "
        "the two options are mutually exclusive",
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
    checkpointing = parser.add_mutually_exclusive_group()
    checkpointing.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Checkpoint decoder layers while training gates to reduce activation VRAM. "
        "Enabled by default; this flag is retained for command compatibility.",
    )
    checkpointing.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable activation checkpointing. This can OOM on long gate-training sequences.",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--no-gate-feedback",
        action="store_true",
        help="Omit cycle feedback text from generation and training prompts "
        "(feedback is still built and saved to disk). Useful for isolating "
        "the effect of explicit text feedback vs implicit LEMUR DB feedback.",
    )
    parser.add_argument(
        "--gate-outer-search",
        action="store_true",
        help="Self-improving gate search: propose a gate, replace the routers, run "
        "the full inner loop, measure CV accuracy, then propose the next gate from "
        "the best prior gate plus measured feedback. Gates are compared on the mean "
        "of per-epoch mean CV accuracies.",
    )
    parser.add_argument(
        "--gate-phase-b",
        action="store_true",
        help="Reserved Phase B switch: train the LLM on accumulated (lower -> higher) "
        "gate pairs so it authors better gate code directly. The training mechanism "
        "is undecided, so this currently fails fast instead of silently doing nothing.",
    )
    return parser.parse_args()
