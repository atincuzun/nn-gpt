"""Outer gate search orchestration for the MoE gate cycle."""

from __future__ import annotations

import shutil
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from .candidates import (
    _install_and_verify,
    _install_author_gate,
    _prepare_gate_candidate,
    _prepare_rematch_candidate,
    _prepare_seed_candidate,
    _select_author_gate_id,
)
from .context import (
    RunContext,
    _prepare_model_for_generation,
    _release_cuda_memory,
    _ts,
)
from .epochs import _run_epochs
from .gate_store import find_reusable_seed, protocol_metadata, structural_hash
from .morphism import _model_logits
from .phase_b_runner import _run_sft_batch_under_best_gate
from .records import _record_failed_candidate, _write_gate_record


def _propose_gate(ctx: RunContext, gate_id: int, seen_hashes: set[str],
                  source: str | None = None) -> None:
    """Propose the next gate under the currently installed routing.

    With ``--gate-author native`` (and for round 0) that is the native router.
    With ``last``/``best`` the caller has already installed the corresponding
    trained gate, so the proposal is authored by the improved composite.

    The prompt is one LEMUR-CV pairing: the best prior gate's code with its
    measured accuracy as the reference, its score as the goal. Round 0 (or an
    empty store) bootstraps from the baseline contract without scores.
    """
    from .gate_prompt import BASELINE_GATE_CODE
    from .gate_store import best_gate

    reference = best_gate(ctx.gate_root)
    reference_source = ""
    reference_accuracy = None
    goal_accuracy = None
    ctx.reference_gate_id = None
    if reference is not None:
        reference_source = reference.get("gate_code") or ""
        ctx.reference_gate_id = reference.get("gate_id")
        score = (reference.get("score") or {}).get("accuracy")
        if isinstance(score, float):
            reference_accuracy = score
            goal_accuracy = score
    elif gate_id > 0:
        # No eligible prior gate yet: fall back to the baseline contract so the
        # model still has an explicit reference to diverge from.
        reference_source = BASELINE_GATE_CODE
    ctx.reference_accuracy = reference_accuracy
    ctx.goal_accuracy = goal_accuracy

    _prepare_gate_candidate(
        ctx,
        gate_id,
        reference_source=reference_source,
        reference_accuracy=reference_accuracy,
        goal_accuracy=goal_accuracy,
        dataset=ctx.args.dataset,
        seen_hashes=seen_hashes,
        source=source,
    )
    print(
        f"[GATE SEARCH] proposed gate {gate_id:03d} "
        f"(reference={ctx.reference_gate_id if ctx.reference_gate_id is not None else 'baseline'}, "
        f"distinct architectures so far={len(seen_hashes)})"
    )


def _run_candidate_once(
    ctx: RunContext, gate_id: int, seen_hashes: set[str], *,
    rematch_for: int | None = None, seed: bool = False,
) -> list[Path]:
    """One outer candidate: author selection, proposal, install, inner loop.

    ``rematch_for`` re-runs a prior gate's code without a new proposal
    (king-of-the-hill re-measure after a Phase B batch). Failures never
    propagate: the candidate is recorded as failed and an empty list returns
    so the search continues.
    """
    args = ctx.args
    ctx.rematch_for = rematch_for
    # Reset per-candidate state: a failed proposal must not leave the previous
    # candidate's code/id in the failed record.
    ctx.gate_source = None
    ctx.reference_gate_id = None
    ctx.reference_accuracy = None
    ctx.goal_accuracy = None
    ctx.gate_epoch_metrics = []
    ctx.seeded = False
    ctx.seed_name = None
    ctx.gate_kind = "llm"
    ctx.reused_from = None
    ctx.carry_source_id = None
    ctx.parent_logits = None
    ctx.carry_forward_metrics = None
    ctx.install_metrics = None
    ctx.proposal_prompt = None
    ctx.proposal_attempts = []
    ctx.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    carry_mode = getattr(args, "gate_carry_forward", None)
    author_mode = carry_mode or args.gate_author
    author_gate_id = (
        None if rematch_for is not None
        else _select_author_gate_id(author_mode, gate_id, ctx.gate_root)
    )
    ctx.author_gate_id = author_gate_id
    if author_mode == "best" and gate_id > 0 and author_gate_id is None:
        print("[GATE AUTHOR] no eligible best gate yet; proposing under native routing")
    pooled_source = None
    if not seed and rematch_for is None and ctx.gate_pool:
        pooled_source = ctx.gate_pool.pop(0)
        print(
            f"[GATE SEARCH] gate {gate_id:03d} taken from proposal pool "
            f"({len(ctx.gate_pool)} still waiting)"
        )
    try:
        with ctx.session:
            if rematch_for is not None:
                _prepare_rematch_candidate(ctx, gate_id, rematch_for)
                if carry_mode is not None:
                    # Function-carry verification needs the parent's logits:
                    # reinstall the re-measured gate with its trained weights,
                    # capture, and restore before the successor is installed.
                    try:
                        _install_author_gate(ctx, rematch_for, mode=carry_mode)
                    except FileNotFoundError:
                        pass
                    else:
                        _prepare_model_for_generation(ctx)
                        ctx.parent_logits = _model_logits(
                            ctx.session.model, ctx.sample_input
                        )
                        ctx.session.restore()
            elif seed:
                # Cold-start bootstrap: built-in compliant gates fill the store
                # with scored references before the proposer is trusted.
                _prepare_seed_candidate(ctx, gate_id)
            else:
                if author_gate_id is not None:
                    try:
                        _install_author_gate(ctx, author_gate_id, mode=author_mode)
                    except FileNotFoundError as exc:
                        print(
                            f"[GATE AUTHOR] trained gate {author_gate_id:03d} unavailable "
                            f"({exc}); proposing under native routing instead"
                        )
                        author_gate_id = None
                        ctx.author_gate_id = None
                    else:
                        _prepare_model_for_generation(ctx)
                        ctx.parent_logits = _model_logits(
                            ctx.session.model, ctx.sample_input
                        )
                        _propose_gate(ctx, gate_id, seen_hashes, source=pooled_source)
                        ctx.session.restore()
                if author_gate_id is None:
                    # Round 0 and native mode: the proposal is authored with the
                    # native router active, outside the session.
                    _propose_gate(ctx, gate_id, seen_hashes, source=pooled_source)
            if carry_mode is not None:
                # Weight-carry source: the incumbent for fresh proposals, or the
                # re-measured gate itself for a rematch. Seeds bootstrap from the
                # native copy like every first-generation gate.
                if rematch_for is not None:
                    ctx.carry_source_id = rematch_for
                elif not seed:
                    ctx.carry_source_id = author_gate_id
            # One consolidated header per candidate: kind, routing it was
            # authored under, weight-carry source, and the proposer version —
            # everything a post-run log analysis needs on a single line.
            candidate_kind = (
                f"rematch of gate_{rematch_for:03d}" if rematch_for is not None
                else "seed" if seed
                else "pool" if pooled_source is not None
                else "llm-proposal"
            )
            print(
                f"[{_ts()}] [CYCLE] gate_{gate_id:03d} START: {candidate_kind}, "
                f"author={'native' if author_gate_id is None else f'gate_{author_gate_id:03d}'}, "
                f"carry={'none' if ctx.carry_source_id is None else f'gate_{ctx.carry_source_id:03d}'}, "
                f"proposer={ctx.llm_version}"
            )
            _install_and_verify(ctx)
            epoch_paths = []
            reusable = None
            if seed and getattr(args, "gate_seed_policy", "reuse") == "reuse":
                reusable = find_reusable_seed(
                    ctx.gate_root, ctx.gate_source,
                    llm_version=ctx.llm_version,
                    protocol=protocol_metadata(args),
                )
            if reusable is not None:
                cached, cached_path = reusable
                weights = cached_path / "weights"
                ctx.session.load_weights(weights)
                ctx.gate_epoch_metrics = [
                    dict(epoch) for epoch in cached.get("epochs") or []
                ]
                ctx.reused_from = cached.get("gate_id")
                if ctx.candidate_root != cached_path:
                    shutil.copytree(weights, ctx.candidate_root / "weights", dirs_exist_ok=True)
                print(
                    f"[GATE SEARCH] gate {gate_id:03d} reuses measured seed "
                    f"gate_{ctx.reused_from:03d} ({cached_path.name})"
                )
            else:
                epoch_paths = _run_epochs(ctx)
        record = _write_gate_record(ctx)
        score = record["score"]
        accuracy = score.get("accuracy")
        print(
            f"[GATE SEARCH] gate {gate_id:03d} finished: "
            f"mean CV accuracy "
            f"{f'{accuracy:.4f}' if isinstance(accuracy, float) else 'unavailable'} "
            f"({score.get('n_measured', 0)}/{score.get('n_attempted', 0)} measured)"
            + (f" [rematch of {rematch_for:03d}]" if rematch_for is not None else "")
        )
        return epoch_paths
    except Exception as exc:
        _record_failed_candidate(ctx, gate_id, exc)
        return []
    finally:
        # Whether the candidate succeeded or was rejected, hand fragmented
        # CUDA reserve back to the driver so the next install-verification
        # forward is not doomed by this candidate's peak usage.
        _release_cuda_memory()


def _run_outer_search(ctx: RunContext, args: Namespace) -> list[list[Path]]:
    """Self-improving gate search: propose, replace, train inner loop, measure, repeat.

    Bad candidates are rejected and recorded, never fatal. With
    ``--gate-outer-sft`` the proposer itself is fine-tuned every
    ``--gate-sft-every`` candidates — under the best trained gate's routing —
    followed by a king-of-the-hill re-measure of the incumbent under the new
    proposer version.
    """
    from .gate_store import best_gate, load_gate_summaries, next_gate_id, summarize_gate

    candidate_epochs: list[list[Path]] = []
    # A persistent store carries dedup history across runs; a fresh store only
    # knows the baseline contract, so the first proposal cannot echo it.
    seen_hashes = {
        summary["structural_hash"]
        for summary in load_gate_summaries(ctx.gate_root)
        if summary.get("structural_hash")
    }
    from .gate_prompt import BASELINE_GATE_CODE

    seen_hashes.add(structural_hash(BASELINE_GATE_CODE))
    # Seed architectures are also off-limits for verbatim LLM re-emission.
    from .gate_seeds import SEED_GATES, seed_gate_source

    for index in range(len(SEED_GATES)):
        seen_hashes.add(structural_hash(seed_gate_source(index)))
    sft_batch = 0
    candidates_run = 0

    while candidates_run < args.gate_candidates:
        # Fresh id each iteration: rematch candidates also consume store ids,
        # so a precomputed counter would collide with them.
        candidate_epochs.append(
            _run_candidate_once(
                ctx, next_gate_id(ctx.gate_root), seen_hashes,
                seed=candidates_run < getattr(args, "gate_seed_candidates", 0),
            )
        )
        candidates_run += 1
        # Fire after the batch completes, not before the next one starts:
        # with candidates == sft_every the "before" placement never triggers.
        if (
            getattr(args, "gate_outer_sft", False)
            and candidates_run % getattr(args, "gate_sft_every", 10) == 0
        ):
            sft_batch += 1
            try:
                new_version = _run_sft_batch_under_best_gate(ctx, sft_batch)
            except Exception as exc:
                # The proposer SFT has never run end-to-end on this hardware;
                # a broken batch must not take the whole search down with it.
                _release_cuda_memory()
                print(
                    f"[GATE SEARCH] proposer SFT batch {sft_batch} failed "
                    f"({type(exc).__name__}: {exc}); continuing with proposer "
                    f"{ctx.llm_version}"
                )
                new_version = None
            if new_version is not None:
                ctx.llm_version = new_version
                if not getattr(args, "gate_no_rematch", False):
                    incumbent = best_gate(ctx.gate_root)
                    if incumbent is not None:
                        candidate_epochs.append(
                            _run_candidate_once(
                                ctx, next_gate_id(ctx.gate_root),
                                seen_hashes, rematch_for=incumbent["gate_id"],
                            )
                        )

        record_probe = best_gate(ctx.gate_root)
        ctx.outer_summary = {
            "gate_root": str(ctx.gate_root),
            "gate_candidates": args.gate_candidates,
            "llm_version": ctx.llm_version,
            "best": (summarize_gate(record_probe) if record_probe else None),
            "phase_b_ready": _phase_b_status(ctx.gate_root),
        }
    return candidate_epochs


def _phase_b_status(gate_root: Path) -> str:
    """How close Phase A is to having usable Phase B training pairs."""
    from .gate_pairs import build_gate_pairs, describe_pairs

    return describe_pairs(build_gate_pairs(gate_root))
