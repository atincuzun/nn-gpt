"""Outer gate-architecture loop: propose, replace, train, measure, record.

Phase A (default) improves gate generation purely in context: each proposal is
conditioned on the best gate measured so far, plus the run's recent results.
No weights are trained.  Phase B (``--gate-outer-sft``) reuses the same seam to
add LLM training on accumulated gate pairs from the on-disk store.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from .gate_prompt import (
    BASELINE_GATE_CODE,
    gate_proposal_prompt,
    gate_prompt_scope,
    gate_sft_examples,
    is_duplicate_gate,
    near_duplicate_similarity,
)
from .gate_store import (
    best_gate,
    fingerprint,
    gate_dir,
    load_gate_summaries,
    structural_hash,
    summarize_gate,
    write_json,
)


def validate_outer_args(args) -> None:
    if args.gate_candidates < 1:
        raise ValueError("--gate-candidates must be at least 1")
    if args.inner_epochs < 1:
        raise ValueError("--inner-epochs must be at least 1")
    if not 0 <= args.gate_near_dup_warn <= 1:
        raise ValueError("--gate-near-dup-warn must be in [0, 1]")
    if args.gate_outer_sft:
        if args.gate_source is not None:
            raise ValueError("--gate-outer-sft generates gates; do not supply --gate-source")
        if args.gate_proposer_steps < 1:
            raise ValueError("--gate-proposer-steps must be positive")


def protocol_fingerprint(ctx) -> str:
    """Identity of the comparison protocol; gates are only comparable within it."""
    args = ctx.args
    return fingerprint({
        "model": args.model,
        "shapes": [list(shape) for shape in ctx.shapes],
        "layers": args.layers,
        "router_top_k": args.router_top_k,
        "inner_epochs": args.inner_epochs,
        "gate_train_steps": args.gate_train_steps,
        "gate_learning_rate": args.gate_learning_rate,
        "dataset": args.dataset,
        "nn_train_epochs": args.nn_train_epochs,
        "fixed_eval_hyperparameters": args.fixed_eval_hyperparameters,
    })


class GateOuterLoop:
    """Owns the on-disk gate store and the single-reference feedback loop."""

    def __init__(self, ctx):
        self.root = ctx.gate_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.protocol_id = protocol_fingerprint(ctx)
        self.seen_hashes = {
            summary["source_hash"]
            for summary in load_gate_summaries(self.root)
            if summary.get("source_hash")
        }

    # ── proposal ────────────────────────────────────────────────────────────

    def reference(self) -> dict:
        """Best prior gate, or the baseline bootstrap reference on round 0."""
        incumbent = best_gate(self.root)
        if incumbent is None:
            return {
                "gate_id": None,
                "source": BASELINE_GATE_CODE,
                "source_hash": structural_hash(BASELINE_GATE_CODE),
                "score": {"accuracy": None},
                "is_baseline": True,
            }
        return incumbent

    def feedback(self, limit: int = 5) -> str:
        summaries = [s for s in load_gate_summaries(self.root) if s.get("score")]
        summaries.sort(key=lambda s: (s["score"].get("accuracy") is not None, s["score"].get("accuracy") or 0.0),
                       reverse=True)
        return "\n".join(summarize_gate(summary) for summary in summaries[:limit])

    def proposal_prompt(self, shapes) -> str:
        reference = self.reference()
        return gate_proposal_prompt(
            shapes,
            reference_source=reference["source"],
            feedback=self.feedback(),
        )

    @contextmanager
    def proposal_chat(self, ctx):
        """Chatbot scoped to the gate-only system prompt, with feedback injected.

        In Phase A this is exactly ``ctx.chat_bot``; the context only pins the
        system prompt so a CV system prompt cannot leak in.  Phase B swaps in
        the trained proposer model here.
        """
        with gate_prompt_scope(ctx.chat_bot):
            yield ctx.chat_bot

    # ── duplicate rejection ─────────────────────────────────────────────────

    def reject_duplicate(self, source: str) -> str | None:
        """Return a rejection reason, or None when the gate is acceptable.

        Exact structural duplicates are rejected; near-duplicates are only
        reported, because the mandatory base/forward skeleton inflates
