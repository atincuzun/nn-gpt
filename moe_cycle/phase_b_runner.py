"""Phase B proposer training orchestration for the MoE gate cycle."""

from __future__ import annotations

import json
from pathlib import Path

from .candidates import _install_author_gate
from .context import RunContext, _ts
from .gate_store import latest_trained_checkpoint as _latest_trained_checkpoint


def _run_phase_b_batch(ctx: RunContext, batch_index: int) -> str | None:
    """Train the proposer on accumulated pairs; return the new version tag.

    Returns None when there is not enough comparable data yet (nothing is
    trained, the version does not change).  Pairs measured under the current
    proposer version are preferred; earlier-version pairs are only used when
    the current version alone cannot fill --gate-min-pairs, because scores
    across versions carry generator drift on top of gate quality.
    """
    from .gate_pairs import build_gate_pairs
    from .gate_store import load_gate_summaries
    from .phase_b import llm_version as version_of
    from .phase_b import train_proposer

    all_pairs = build_gate_pairs(ctx.gate_root)
    if not all_pairs:
        print(f"[PHASE B] skipping batch {batch_index}: no comparable gate pairs yet")
        return None
    version_by_id = {
        summary["gate_id"]: summary.get("llm_version")
        for summary in load_gate_summaries(ctx.gate_root)
    }
    current_version_pairs = [
        pair for pair in all_pairs
        if version_by_id.get(pair["lower_gate_id"]) == ctx.llm_version
        and version_by_id.get(pair["higher_gate_id"]) == ctx.llm_version
    ]
    if len(current_version_pairs) >= getattr(ctx.args, "gate_min_pairs", 2):
        pairs = current_version_pairs
        version_scope = "current-version"
    else:
        pairs = all_pairs
        version_scope = "cross-version fallback"
    if len(pairs) < getattr(ctx.args, "gate_min_pairs", 2):
        print(
            f"[PHASE B] skipping batch {batch_index}: {len(pairs)} comparable "
            f"pair(s) < --gate-min-pairs {getattr(ctx.args, 'gate_min_pairs', 2)}"
        )
        return None
    # SFT batch receipt: WHICH pairs trained the proposer and under which
    # version scope — the first question of any post-run analysis of the
    # self-improvement loop ("what did the LoRA actually learn from?").
    pair_ids = ", ".join(pair["pair_id"] for pair in pairs[:8]) + (
        f" (+{len(pairs) - 8} more)" if len(pairs) > 8 else ""
    )
    print(
        f"[{_ts()}] [CYCLE] proposer SFT batch {batch_index}: {len(pairs)} pair(s) "
        f"[{pair_ids}] ({version_scope}), parent proposer {ctx.llm_version}"
    )
    adapter_dir = ctx.run_root / "proposer_adapters" / f"batch_{batch_index:03d}"
    parent_version = ctx.llm_version
    training_summary = train_proposer(
        ctx.session,
        ctx.tokenizer,
        pairs,
        ctx.args,
        ctx.shapes,
        adapter_dir,
    )
    new_version = version_of(ctx.session.model, ctx.args, merged_from=parent_version)
    print(
        f"[{_ts()}] [CYCLE] proposer SFT batch {batch_index} TRAINED: "
        f"steps={training_summary.get('steps')}, pairs={training_summary.get('pairs')}, "
        f"mode={training_summary.get('mode')}, adapter={training_summary.get('output_dir')}"
    )
    if ctx.args.gate_store is not None:
        # Persist the newest adapter so the next run continues from it
        # (Tune.py-style weight compounding across runs).
        persisted = Path(ctx.args.gate_store).resolve() / "proposer"
        (persisted / "adapter").mkdir(parents=True, exist_ok=True)
        ctx.session.model.save_pretrained(persisted / "adapter")
        (persisted / "lineage.json").write_text(
            json.dumps({
                "version": new_version,
                "parent_version": parent_version,
                "batch": batch_index,
                "mode": ctx.args.gate_sft_mode,
                "pairs": len(pairs),
                "source_run": str(ctx.run_root),
            }, indent=2),
            encoding="utf-8",
        )
        print(
            f"[GATE SEARCH] outer-loop SFT: persisted proposer adapter → {persisted / 'adapter'}"
        )
    print(f"[GATE SEARCH] outer-loop SFT: proposer version {parent_version} -> {new_version}")
    return new_version


def _run_sft_batch_under_best_gate(ctx: RunContext, batch_index: int) -> str | None:
    """Run one proposer SFT batch with the best inner-loop gate installed.

    The SFT loss is teacher-forced on gate-code text, but its forward pass is
    still routed by the currently installed gates. The batch therefore runs
    with the best gate candidate's architecture and trained weights active —
    the same carried routing state the proposer will author proposals from —
    instead of the native routers the session restores after every candidate.
    Gate parameters themselves stay frozen; only the proposer LoRA trains
    (full-model adaptation, unlike the gate-only training of the inner loop).

    Without a scored incumbent, without a trained checkpoint, or when
    installing/training under the carried gate fails, the batch falls back to
    native routing so a broken incumbent can never block proposer training.
    """
    from .gate_store import best_gate

    incumbent = best_gate(ctx.gate_root)
    if incumbent is not None and _latest_trained_checkpoint(
        ctx.gate_root, incumbent["gate_id"]
    ) is not None:
        try:
            with ctx.session:
                _install_author_gate(
                    ctx, incumbent["gate_id"], mode="sft-carry",
                    purpose=f"proposer SFT batch {batch_index}",
                )
                for install in ctx.session.installs:
                    for parameter in install.new_gate.parameters():
                        parameter.requires_grad_(False)
                return _run_phase_b_batch(ctx, batch_index)
        except Exception as exc:
            print(
                "[GATE SEARCH] proposer SFT batch under trained gate "
                f"{incumbent['gate_id']:03d} failed ({type(exc).__name__}: {exc}); "
                "falling back to native routing"
            )
    return _run_phase_b_batch(ctx, batch_index)
