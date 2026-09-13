"""MoE-local orchestration for database -> gate SFT -> proposal -> measured trial."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from .gate_benchmark import evaluate_candidate, file_hash, prepare_benchmark, write_json
from .gate_proposer import GateProposer
from .gate_store import GateStore, fingerprint


def validate_outer_args(args) -> None:
    if not args.gate_outer_sft:
        if args.gate_proposer_checkpoint is not None:
            raise ValueError("--gate-proposer-checkpoint requires --gate-outer-sft")
        return
    if args.gate_source is not None:
        raise ValueError("--gate-outer-sft generates gates; do not supply --gate-source")
    for name in ("epochs", "gate_train_steps", "nn_train_epochs", "test_nn", "gate_benchmark_size",
                 "gate_min_measured", "gate_proposer_steps", "gate_proposer_rank",
                 "gate_proposer_max_length", "gate_proposer_batch_size", "gate_proposer_max_examples",
                 "gate_proposer_min_examples"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.gate_min_success_rate <= 1:
        raise ValueError("--gate-min-success-rate must be in [0, 1]")
    if not 0 < args.gate_proposer_top_fraction <= 1:
        raise ValueError("--gate-proposer-top-fraction must be in (0, 1]")
    if not args.gate_proposer_learning_rate > 0:
        raise ValueError("--gate-proposer-learning-rate must be positive")
    if args.gate_proposer_min_examples > args.gate_proposer_max_examples:
        raise ValueError("Proposer minimum examples exceeds maximum examples")
    if len(set(args.gate_benchmark_seeds)) != len(args.gate_benchmark_seeds):
        raise ValueError("Gate benchmark seeds must be distinct")
    if args.fixed_eval_hyperparameters is None:
        raise ValueError("--gate-outer-sft requires --fixed-eval-hyperparameters for comparable CV budgets")
    if args.gate_proposer_checkpoint is not None and args.gate_benchmark is None:
        raise ValueError("Resuming a proposer requires its --gate-benchmark manifest")


class GateOuterLoop:
    def __init__(self, ctx):
        self.manifest, self.protocol = prepare_benchmark(ctx)
        self.protocol_id = fingerprint(self.protocol)
        self.store = GateStore(ctx.args.gate_db or ctx.run_root / "gate_archive.sqlite3")
        self.proposer = GateProposer(ctx.args, ctx.run_root, self.protocol_id)
        self.trials = []
        ctx.gate_benchmark = self.manifest
        print("[GATE OUTER] Fixed training rows and post-training comparison prompts enabled; "
              "candidate-specific training feedback omitted for architecture comparability.")

    def record(self, ctx, epoch_paths):
        score, outcomes, checkpoint = evaluate_candidate(ctx, self.manifest)
        training = [json.loads((p / "gate_training_metrics.json").read_text()) for p in epoch_paths]
        record = self.store.add({
            "trial_id": uuid.uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
            "run_root": str(ctx.run_root), "candidate_index": ctx.candidate_index,
            "source": ctx.gate_source, "class_name": "LLMGeneratedGate",
            "protocol_id": self.protocol_id, "protocol": self.protocol,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_hash(checkpoint / "gate_weights.pt"),
            "proposer_checkpoint": str(self.proposer.checkpoint.resolve()) if self.proposer.checkpoint else None,
            "training": training, "score": score, "outcomes": outcomes,
        })
        write_json(ctx.candidate_root / "gate_trial.json", record)
        self.trials.append(record["trial_id"])
        # Best is descriptive/persisted, not a separate evolutionary controller.
        best = self.store.training_subset(self.protocol_id, max_examples=1, top_fraction=1e-9, seed=0)
        write_json(ctx.run_root / "best_gate.json", {
            "protocol_id": self.protocol_id, "database": str(self.store.path.resolve()),
            "best": best[0] if best else None,
        })
        print(f"[GATE OUTER] candidate={ctx.candidate_index} mean_accuracy={score['mean_accuracy']} "
              f"success_rate={score['success_rate']:.3f} eligible={score['eligible']}")

    def summary(self):
        return {"protocol_id": self.protocol_id, "database": str(self.store.path.resolve()),
                "trials": self.trials,
                "proposer_checkpoint": str(self.proposer.checkpoint.resolve()) if self.proposer.checkpoint else None}
