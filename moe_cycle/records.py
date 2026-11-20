"""Canonical gate records, failures, and run summaries for the MoE cycle."""

from __future__ import annotations

import json
from pathlib import Path

from .context import RunContext, _generation_prefixes, _nas_prefixes
from .gate_store import (
    archive_gate,
    load_gate_summaries,
    protocol_metadata,
    structural_hash,
    summarize_epoch_metrics,
    write_gate_record,
)


def _harvest_epoch_metrics(
    epoch: int,
    epoch_path: Path,
    gate_feedback: dict,
    cycle_results: dict,
    models_trained: int,
    train_loss: float | None,
    validation_loss: float | None,
    active_layers: set[int],
    duration_ns: int = 0,
) -> dict:
    """Collect one inner epoch's measurements for the outer gate score.

    ``cv_accuracy_mean`` averages the successfully measured CV candidates, which
    is robust to a single lucky model.  ``cv_accuracy_best`` is retained for
    reference.  Failures are counted, never scored as zero.
    """
    accuracies = [
        outcome["accuracy"]
        for outcome in gate_feedback.get("candidate_outcomes", [])
        if outcome.get("status") == "measured" and outcome.get("accuracy") is not None
    ]
    n_generated = int(gate_feedback.get("n_generated", 0))
    mean = sum(accuracies) / len(accuracies) if accuracies else None
    return {
        "epoch": epoch,
        "cv_accuracy_mean": mean,
        "cv_accuracy_best": max(accuracies) if accuracies else None,
        "n_measured": len(accuracies),
        "n_attempted": n_generated,
        "n_trained": int(models_trained),
        "n_generated": n_generated,
        "mean_accuracy_delta": gate_feedback.get("mean_accuracy_delta"),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "active_gate_layers": sorted(active_layers),
        "cycle_success": bool(cycle_results.get("success")),
        "duration": duration_ns,
        "synth_nn": f"epochs/A{epoch}/synth_nn",
    }
def _write_gate_record(ctx: RunContext, *, finalize: bool = True) -> dict:
    """Persist one gate candidate, incrementally and at completion.

    ``finalize=False`` writes a ``running`` record after each inner epoch so a
    killed run still leaves usable measurements. The final write sets the
    terminal status and archives measured gates.
    """

    score = summarize_epoch_metrics(ctx.gate_epoch_metrics)
    gate_source = getattr(ctx, "gate_source", None)
    record = {
        "gate_id": ctx.candidate_index,
        "gate_code": gate_source,
        "structural_hash": structural_hash(gate_source) if gate_source else None,
        "reference_gate_id": getattr(ctx, "reference_gate_id", None),
        "author_mode": (
            getattr(ctx.args, "gate_carry_forward", None)
            or getattr(ctx.args, "gate_author", None)
        ),
        "author_gate_id": getattr(ctx, "author_gate_id", None),
        "carry_source_gate_id": getattr(ctx, "carry_source_id", None),
        "carry_forward": getattr(ctx, "carry_forward_metrics", None),
        "rematch_for": getattr(ctx, "rematch_for", None),
        "llm_version": getattr(ctx, "llm_version", "base"),
        "seeded": getattr(ctx, "seeded", False),
        "seed_name": getattr(ctx, "seed_name", None),
        "kind": getattr(ctx, "gate_kind", "llm"),
        "reused_from": getattr(ctx, "reused_from", None),
        "created_at": getattr(ctx, "created_at", None),
        "running": not finalize,
        "install": getattr(ctx, "install_metrics", None),
        "proposal": {
            "prompt": getattr(ctx, "proposal_prompt", None),
            "reference_gate_id": getattr(ctx, "reference_gate_id", None),
            "reference_accuracy": getattr(ctx, "reference_accuracy", None),
            "goal_accuracy": getattr(ctx, "goal_accuracy", None),
            "attempts": len(getattr(ctx, "proposal_attempts", [])),
        },
        "protocol": protocol_metadata(ctx.args),
        # gate_pairs.build_gate_pairs groups comparable records on exactly
        # these three keys; without them no Phase B pairs can ever form.
        "task": "img-classification",
        "dataset": getattr(ctx.args, "dataset", "cifar-10"),
        "metric": "accuracy",
        "score": score,
        "epochs": ctx.gate_epoch_metrics,
    }
    write_gate_record(
        ctx.gate_root,
        ctx.candidate_index,
        record,
        run_root=getattr(ctx, "run_root", None),
        model=getattr(ctx.args, "model", None),
    )
    if finalize:
        archive_gate(
            record,
            candidate_root=getattr(ctx, "candidate_root", None),
            gate_id=ctx.candidate_index,
            run_root=getattr(ctx, "run_root", None),
            model=getattr(ctx.args, "model", None),
            archive_dir=getattr(ctx.args, "gate_archive_dir", None),
        )
    return record
def _record_failed_candidate(ctx: RunContext, gate_id: int, exc: Exception) -> None:
    """Persist a failed candidate so the run continues and the store keeps the lesson."""

    reason = f"{type(exc).__name__}: {exc}"
    print(f"[GATE SEARCH] gate {gate_id:03d} FAILED, continuing: {reason}")
    record = {
        "gate_id": gate_id,
        "gate_code": ctx.gate_source,
        "structural_hash": structural_hash(ctx.gate_source) if ctx.gate_source else None,
        "reference_gate_id": ctx.reference_gate_id,
        "author_mode": (
            getattr(ctx.args, "gate_carry_forward", None)
            or getattr(ctx.args, "gate_author", None)
        ),
        "author_gate_id": ctx.author_gate_id,
        "carry_source_gate_id": getattr(ctx, "carry_source_id", None),
        "carry_forward": getattr(ctx, "carry_forward_metrics", None),
        "rematch_for": getattr(ctx, "rematch_for", None),
        "llm_version": getattr(ctx, "llm_version", "base"),
        "seeded": getattr(ctx, "seeded", False),
        "seed_name": getattr(ctx, "seed_name", None),
        "kind": getattr(ctx, "gate_kind", "llm"),
        "reused_from": getattr(ctx, "reused_from", None),
        "created_at": getattr(ctx, "created_at", None),
        "running": False,
        "install": getattr(ctx, "install_metrics", None),
        "proposal": {
            "prompt": getattr(ctx, "proposal_prompt", None),
            "reference_gate_id": getattr(ctx, "reference_gate_id", None),
            "reference_accuracy": getattr(ctx, "reference_accuracy", None),
            "goal_accuracy": getattr(ctx, "goal_accuracy", None),
            "attempts": len(getattr(ctx, "proposal_attempts", [])),
        },
        "protocol": protocol_metadata(ctx.args),
        "score": {
            "objective": "mean_of_epoch_means_v1",
            "accuracy": None,
            "eligible": False,
            "n_measured": 0,
            "n_attempted": 0,
        },
        # Partial measurements survive the failure: the candidate stays
        # ineligible (it never completed the protocol) but its epochs remain
        # analysable instead of being discarded with the record.
        "epochs": ctx.gate_epoch_metrics,
        "failure": reason[:2000],
    }
    write_gate_record(
        ctx.gate_root,
        gate_id,
        record,
        run_root=getattr(ctx, "run_root", None),
        model=getattr(ctx.args, "model", None),
    )
def _write_final_summary(ctx: RunContext) -> None:
    """Write the run-level summary from the canonical gate records."""
    from ab.gpt.util.Const import nngpt_dir

    args = ctx.args
    summary = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "progressive_unfreeze_descending": args.progressive_unfreeze_descending,
        "generation_backend": "upstream_default",
        "fixed_evaluation_prompts": False,
        "train_prompt_config": args.train_prompt_config,
        "test_prompt_config": args.test_prompt_config,
        "dataset": args.dataset,
        "fixed_evaluation_hyperparameters": args.fixed_eval_hyperparameters,
        "nn_train_epochs": args.nn_train_epochs,
        "training_nn_prefixes": list(_nas_prefixes(args)),
        "generation_nn_prefixes": list(_generation_prefixes(args)),
        "gate_source": (
            str((ctx.candidate_root or ctx.gate_dir) / "gate.py")
            if ctx.gate_source is not None else None
        ),
        "gate_shapes": ctx.shapes,
        "replaced_gates": len(
            ctx.sites
            if args.layers is None
            else [s for s in ctx.sites if s.layer_index in args.layers]
        ),
        "epochs": args.epochs,
        "gate_candidates": args.gate_candidates,
        "gate_author": args.gate_author,
        "gate_carry_forward": getattr(args, "gate_carry_forward", None),
        "gate_outer_search": ctx.outer_summary,
        "run_root": str(ctx.run_root),
    }
    for record in load_gate_summaries(ctx.gate_root):
        provenance = record.get("provenance") or {}
        if provenance.get("run_root") not in (None, str(ctx.run_root)):
            continue
        gate_id = record.get("gate_id")
        if not isinstance(gate_id, int):
            continue
        summary[f"candidate_{gate_id:03d}"] = {
            "gate_source": (record.get("artifacts") or {}).get("source"),
            "status": record.get("status"),
            "score": record.get("score"),
            "protocol": record.get("protocol"),
            "epochs": record.get("epochs"),
        }

    (nngpt_dir / "one_cycle_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(f"MoE gate cycle completed: {ctx.run_root}")
