"""CLI argument parsing for the MoE gate-training NAS cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Comprehensive LEMUR family census (nn-gpt/db/ab.nn.db): every prefix below has
# img-classification accuracy rows in the stat table. Prefixes are applied as
# case-insensitive SQL "nn LIKE 'prefix%'" OR-chains, so 'alt' covers
# alt-nn1..6/alt-1nn, 'rl-' covers rl-bb-*/rl-init*, 'llr' covers llr2/llr3,
# 'ast-' covers ast-dimension/ast-activation, 'MoE' covers MoE4Own/MoE4/MoEv*,
# and 'ResNet'/'UNet' also match RESNETLSTM/ResNetTransformer/UNet2D via LIKE.
# Defined here (not in the shared Const.py) so the MoE cycle can train on the
# full corpus without changing the upstream default corpus.
GATE_CYCLE_NN_PREFIXES = (
    # LLM-generated experiment families (bulk of the corpus)
    'ga-', 'GenFractalNet', 'unq', 'rag', 'alt', 'rl-', 'llr', 'del',
    'ast-', 'MoE', 'l1', 'l2', 'l3', 'moe-gate-cycle',
    # Classical / torchvision families
    'AlexNet', 'AirNet', 'AirNext', 'BagNet', 'BayesianNet', 'ComplexNet',
    'ConvNeXt', 'DPN', 'DarkNet', 'DenseNet', 'Diffuser', 'EfficientNet',
    'FractalNet', 'GoogLeNet', 'ICNet', 'InceptionV3', 'LSTM', 'MaxVit',
    'MNASNet', 'MobileNet', 'RegNet', 'ResNet', 'RNN', 'ShuffleNet',
    'SqueezeNet', 'SwinTransformer', 'TitanV', 'UNet', 'VGG',
    'VisionTransformer',
)


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
    parser.add_argument(
        "--gate-random-init",
        action="store_true",
        help="Use --gate-source constructor initialization without copying native "
        "router weights or requiring a base projection. Requires zero initialization noise.",
    )
    parser.add_argument(
        "--gate-skip-step-zero-verify",
        action="store_true",
        help="Install the proposed gate even when its step-zero model logits diverge "
        "from the native model instead of raising the max_abs_difference RuntimeError. "
        "The measured difference is still recorded in step_zero_equivalence.json; the "
        "non-finite-logits guard stays active.",
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
        default="NN_gen.json",
        help="Paired SFT prompt config under ab/gpt/conf/prompt/train; the "
        "upstream NN_gen.json is the default so gate training sees the same "
        "distribution as the LoRA pipeline",
    )
    parser.add_argument(
        "--test-prompt-config",
        default="NN_gen.json",
        help="Generation prompt config under ab/gpt/conf/prompt/test; the "
        "upstream NN_gen.json is the default",
    )
    parser.add_argument("--dataset", default="cifar-10")
    parser.add_argument(
        "--sft-nn-prefixes",
        nargs="+",
        default=list(GATE_CYCLE_NN_PREFIXES),
        help="LEMUR model prefixes used for paired gate-training examples. "
        "Defaults to the full LEMUR family census (GATE_CYCLE_NN_PREFIXES). "
        "The current --nn-name-prefix is added automatically.",
    )
    parser.add_argument(
        "--generation-nn-prefixes",
        nargs="+",
        default=list(GATE_CYCLE_NN_PREFIXES),
        help="LEMUR prefixes used as CV generation seeds (nn_gen samples one "
        "record per model). Defaults to the full LEMUR family census "
        "(GATE_CYCLE_NN_PREFIXES). The current --nn-name-prefix is added "
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
        default=["improve_classification_only"],
        help="Keys selected from the TEST prompt config for CV generation; the "
        "train config is consumed whole by the gate-training data builder",
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
        "--gate-outer-search",
        action="store_true",
        help="Self-improving gate search: propose a gate, replace the routers, run "
        "the full inner loop, measure CV accuracy, then propose the next gate from "
        "the best prior gate plus measured feedback. Gates are compared on the mean "
        "of per-epoch mean CV accuracies.",
    )
    parser.add_argument(
        "--gate-author",
        choices=("native", "last", "best"),
        default="native",
        help="Router state the LLM authors the next gate under (requires "
        "--gate-outer-search). 'native' restores the original routers before each "
        "proposal (default, comparable baseline). 'last' installs the previous "
        "candidate's trained weights while proposing. 'best' installs the "
        "best-scoring trained gate so far. Scores stay comparable in every mode "
        "because each candidate still trains from a fresh native-copy "
        "initialization.",
    )
    parser.add_argument(
        "--gate-phase-b",
        action="store_true",
        help="Reserved Phase B switch: train the LLM on accumulated (lower -> higher) "
        "gate pairs so it authors better gate code directly. The training mechanism "
        "is undecided, so this currently fails fast instead of silently doing nothing.",
    )
    parser.add_argument(
        "--gate-store",
        type=Path,
        help="Persistent gate store directory shared across runs. Default places the "
        "store inside this run's output directory, so nothing survives the run; a "
        "shared directory lets later runs continue gate ids, dedup history, best-gate "
        "reference, and Phase B pairs from earlier runs.",
    )
    parser.add_argument(
        "--gate-outer-sft",
        action="store_true",
        help="Phase B proposer training: every --gate-sft-every candidates, fine-tune "
        "a LoRA adapter on the accumulated (lower -> higher) gate pairs so the LLM "
        "authors better gates directly. Requires --gate-outer-search. The adapter "
        "stays active (unmerged) for the rest of the run; each gate record stores the "
        "proposer version it was measured under.",
    )
    parser.add_argument("--gate-sft-every", type=int, default=10)
    parser.add_argument("--gate-sft-steps", type=int, default=30)
    parser.add_argument("--gate-sft-lr", type=float, default=1e-4)
    parser.add_argument("--gate-sft-mode", choices=("sft", "dpo"), default="sft",
                        help="Phase B loss: sft imitates the higher-scoring gate "
                        "code (default); dpo additionally contrasts it against "
                        "the lower-scoring gate of each pair")
    parser.add_argument("--gate-sft-rank", type=int, default=16)
    parser.add_argument(
        "--gate-min-pairs",
        type=int,
        default=1,
        help="Minimum comparable gate pairs required before a Phase B training batch runs "
        "(default 1: train whenever any (worse -> better) pair exists, matching the LoRA "
        "path which SFTs on corpus data regardless of current-cycle success)",
    )
    parser.add_argument(
        "--gate-fresh-proposer",
        action="store_true",
        help="Ignore any proposer adapter persisted under --gate-store and start "
        "outer-loop SFT from the base model. Without this flag the Tune.py-style "
        "merge-and-continue is applied automatically when a persisted adapter exists.",
    )
    parser.add_argument(
        "--gate-no-rematch",
        action="store_true",
        help="Skip the incumbent re-measure that normally follows each Phase B batch. "
        "The re-match re-runs the current best gate under the updated proposer so "
        "cross-version score comparisons stay fair (king-of-the-hill).",
    )
    return parser.parse_args()
