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


def build_parser() -> argparse.ArgumentParser:
    """The cycle's argument parser; ``moe_cycle.experiment`` extends it."""
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
        "--gate-eval-seed",
        type=int,
        default=None,
        help="Seed the RNG before each gate's CV evaluation so the same "
        "candidate index is measured under the same training randomness in "
        "every gate (paired comparison across architectures)",
    )
    parser.add_argument(
        "--gate-source",
        type=Path,
        help="Python file defining the replacement gate class (external gate; "
        "overrides LLM generation)",
    )
    parser.add_argument(
        "--gate-class",
        default="Net",
        help="Class name to load from --gate-source",
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
        "The measured difference is still recorded in record.json under install; the "
        "non-finite-logits guard stays active.",
    )
    parser.add_argument("--gate-generation-attempts", type=int, default=100)
    parser.add_argument(
        "--gate-catalog-candidates",
        "--gate-seed-candidates",
        dest="gate_catalog_candidates",
        type=int,
        default=0,
        help="Evaluate the first N built-in catalog gates as initial data instead "
        "of LLM proposals. The catalog is a source of gate code, not a separate "
        "data kind; records and selection treat it like any other gate. Use 50 "
        "to materialize the full catalog; 0 goes straight to LLM proposals.",
    )
    parser.add_argument(
        "--gate-reuse-policy",
        "--gate-seed-policy",
        dest="gate_reuse_policy",
        choices=("reuse", "evaluate"),
        default="reuse",
        help="With a persistent gate store, reuse a measured gate whose code hash, "
        "proposer version and protocol fingerprint match. 'evaluate' forces a "
        "fresh measurement (used for catalog materialization and ablation).",
    )
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
    parser.add_argument(
        "--gate-prompt-key",
        default="reference_best",
        help="Gate proposal prompt variant, read from "
        "ab/gpt/conf/prompt/<split>/Gate_proposal.json. 'reference_best' uses "
        "the best prior gate as the baseline; 'pair' uses the lower-scoring "
        "member of the best measured pair as the baseline and the higher "
        "member's accuracy as the goal (the framing Phase B trains on)",
    )
    parser.add_argument(
        "--gate-prompt-split",
        default="test",
        help="Prompt-config folder the gate prompt variant is read from",
    )
    parser.add_argument(
        "--gate-expert-quant",
        choices=("none", "nf4", "int8", "int8_per_expert"),
        default="none",
        help="Quantize the fused MoE expert stacks, which bitsandbytes' "
        "nn.Linear walker cannot reach. 'nf4' is 4-bit (~9%% weight error), "
        "'int8' is row-wise 8-bit (~0.8%% error) and keeps code generation "
        "intact at twice the memory, 'int8_per_expert' is the same 8-bit "
        "encoding but dequantizes only the routed experts, which is several "
        "times faster to generate with. 'none' leaves the experts in bf16",
    )
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
        "--gate-carry-forward",
        choices=("best", "last"),
        help="Self-improving architecture mode (requires --gate-outer-search): the "
        "selected trained gate ('best' scoring so far, or the 'last' candidate) stays "
        "installed as the LLM's router architecture while it authors the next gate, "
        "and its trained weights seed the successor's training (parameters matched by "
        "name; parameters the parent never had keep their constructor initialization). "
        "'best' is elitist: the incumbent only changes when a successor scores higher.",
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
        "--gate-archive-dir",
        type=Path,
        help="Project-level archive for measured gate code and LEMUR-shaped records. "
        "Copies (never moves) gate.py and record.json per measured gate. Default: "
        "nn-gpt/gate_archive.",
    )
    parser.add_argument(
        "--gate-outer-sft",
        action="store_true",
        help="Outer-loop proposer training: every --gate-sft-every candidates, fine-tune "
        "a LoRA adapter on the accumulated (lower -> higher) gate pairs so the LLM "
        "authors better gates directly. Requires --gate-outer-search. Each batch runs "
        "with the best trained gate installed (architecture + carried weights), so the "
        "proposer trains in the routing state it will propose from. The adapter "
        "stays active (unmerged) for the rest of the run; each gate record stores the "
        "proposer version it was measured under."
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
    parser.add_argument(
        "--measure-after-training",
        action="store_true",
        help="Train the gate first and measure CV accuracy afterwards in every inner "
        "epoch. Without it each epoch measures before training, so with --epochs 1 "
        "the score is taken while every function-preserving gate is still an exact "
        "copy of the native router and cannot reflect the architecture.",
    )
    parser.add_argument(
        "--generation-batch",
        type=int,
        default=1,
        help="CV prompts generated together in one batched generate() call. Each "
        "row stops at its own closing </nn> tag.",
    )
    parser.add_argument(
        "--pinned-seed-prompts",
        action="store_true",
        help="Draw CV generation seeds from the pinned list named by NNGPT_SEED_FILE "
        "(upstream pin_generation_seeds) instead of the legacy alphabetical draw, so "
        "every gate is measured on the same seed networks.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
