"""Gate candidate preparation and installation for the MoE cycle."""

from __future__ import annotations

import json
import shutil

from typing import Any

from .context import (
    RunContext,
    _initial_feedback_summary,
    _prepare_model_for_generation,
    _ts,
)
from .gate_seeds import seed_gate_name, seed_gate_source
from .gate_source import _generate_gate, _validate_gate_source
from .gate_store import (
    best_gate,
    create_gate_entry,
    gate_dir,
    latest_trained_checkpoint,
)
from .morphism import (
    _model_logits,
    _perturb_gate_weights,
    _seed_all,
    carry_gate_weights,
    verify_parent_function_carry,
)


def _prepare_gate_candidate(ctx: RunContext, candidate_index: int, *,
                            reference_source: str = "",
                            reference_accuracy: float | None = None,
                            goal_accuracy: float | None = None,
                            dataset: str | None = None,
                            proposal_chat: Any = None,
                            seen_hashes: set[str] | None = None,
                            source: str | None = None) -> None:
    """Prepare one gate architecture while the native router is active.

    ``source`` comes from the proposal pool (harvested in an earlier round);
    without it a proposal round runs, the first harvested gate is used here
    and the remainder join the pool for later candidates.
    """
    args = ctx.args
    ctx.candidate_index = candidate_index
    ctx.candidate_root = gate_dir(ctx.gate_root, candidate_index)
    ctx.gate_dir = ctx.candidate_root
    ctx.candidate_root.mkdir(parents=True, exist_ok=True)
    ctx.proposal_prompt = None
    ctx.proposal_attempts = []
    if args.gate_source is not None:
        if not args.gate_source.is_file():
            raise FileNotFoundError(f"Gate source file does not exist: {args.gate_source}")
        source = args.gate_source.read_text(encoding="utf-8")
        _validate_gate_source(
            source, ctx.shapes, class_name=args.gate_class,
            require_base=not args.gate_random_init,
        )
        ctx.proposal_prompt = f"[external gate source] {args.gate_source}"
    elif source is None:
        sources = _generate_gate(
            proposal_chat or ctx.chat_bot, ctx.shapes, args.gate_generation_attempts,
            args.gate_max_new_tokens, ctx.candidate_root,
            reference_source=reference_source,
            reference_accuracy=reference_accuracy,
            goal_accuracy=goal_accuracy,
            dataset=dataset,
            seen_hashes=seen_hashes,
            inherit_reference=getattr(args, "gate_carry_forward", None) is not None,
            write_artifacts=False,
            prompt_out=ctx.proposal_attempts,
        )
        source = sources[0]
        ctx.proposal_prompt = (
            ctx.proposal_attempts[-1] if ctx.proposal_attempts else None
        )
        if len(sources) > 1:
            ctx.gate_pool.extend(sources[1:])
            print(
                f"[GATE SEARCH] {len(sources) - 1} extra valid gate(s) -> "
                f"proposal pool ({len(ctx.gate_pool)} waiting)"
            )
    else:
        ctx.proposal_prompt = "[pooled valid gate from an earlier proposal round]"
    final_root = create_gate_entry(ctx.gate_root, candidate_index, source, kind="llm")
    if final_root != ctx.candidate_root:
        final_root.mkdir(parents=True, exist_ok=True)
        for child in list(ctx.candidate_root.iterdir()):
            shutil.move(str(child), final_root / child.name)
        ctx.candidate_root.rmdir()
    ctx.candidate_root = final_root
    ctx.gate_dir = final_root
    (ctx.candidate_root / "gate.py").write_text(source.rstrip() + "\n", encoding="utf-8")
    ctx.gate_source = source
    ctx.seed_name = None
    ctx.gate_kind = "llm"
    ctx.previous_feedback_summary = _initial_feedback_summary(source)
    ctx.used_prompts.clear()


def _select_author_gate_id(mode: str, gate_id: int, gate_root) -> int | None:
    """Which trained gate (if any) authors round ``gate_id``'s proposal.

    ``native`` (and round 0) propose with the original routers. ``last`` chains
    to the previous candidate. ``best`` installs the highest-scoring eligible
    gate, falling back to native while the store has no eligible gate yet.
    """
    if mode == "native" or gate_id == 0:
        return None
    if mode == "last":
        return gate_id - 1
    if mode == "best":
        reference = best_gate(gate_root)
        return reference.get("gate_id") if reference is not None else None
    raise ValueError(f"unknown gate author mode {mode!r}")


def _install_author_gate(ctx: RunContext, author_gate_id: int, *,
                         mode: str | None = None,
                         purpose: str | None = None) -> None:
    """Install a previously trained gate so the LLM works under its routing.

    Used while the LLM authors the next proposal and while proposer SFT
    batches run. No equivalence checks: trained weights are not bit-exact
    with the native router by design. Must run inside ``with ctx.session:``
    with no gates currently installed.
    """
    args = ctx.args
    author_root = gate_dir(ctx.gate_root, author_gate_id)
    checkpoint = latest_trained_checkpoint(ctx.gate_root, author_gate_id)
    if checkpoint is None:
        raise FileNotFoundError(
            f"Author gate checkpoint not found under {author_root / 'weights'}"
        )
    session = ctx.session
    metadata_path = checkpoint / "metadata.json"
    class_name = getattr(args, "gate_class", "Net")
    if metadata_path.is_file():
        try:
            class_name = json.loads(metadata_path.read_text(encoding="utf-8")).get(
                "gate_class_name", class_name,
            )
        except (OSError, ValueError):
            pass
    session.replace_source(
        (author_root / "gate.py").read_text(encoding="utf-8"),
        class_name=class_name,
        layers=args.layers,
        sample_input=ctx.sample_input,
        verify=False,
        top_k=args.router_top_k,
        allow_remote_code=True,
        dynamic_discovery=False,
        initialize_from_original=True,
    )
    session.load_weights(checkpoint)
    if purpose is None:
        print(
            f"[GATE AUTHOR] gate {mode or args.gate_author} mode: next proposal is authored "
            f"under trained gate {author_gate_id:03d} ({checkpoint})"
        )
    else:
        print(
            f"[GATE AUTHOR] gate {mode or args.gate_author} mode: trained gate "
            f"{author_gate_id:03d} installed for {purpose} ({checkpoint})"
        )


def _install_and_verify(ctx: RunContext) -> None:
    """Install replacement gates, verify step-zero equivalence, and freeze.

    Must run inside ``with ctx.session:`` (owned by ``main``) so the installed
    gates survive for ``_run_epochs``; the session context restores them when
    the whole cycle finishes.
    """
    import torch

    args = ctx.args
    session = ctx.session

    # A previous candidate's gate training leaves the model in train mode
    # with use_cache disabled and stale flags on detached native gates; every
    # install must start from the clean generation state.
    _prepare_model_for_generation(ctx)
    _seed_all(args.seed)
    replace_kwargs = {
        "layers": args.layers,
        "sample_input": ctx.sample_input,
        "verify": True,
        "top_k": args.router_top_k,
        "allow_remote_code": True,
        "dynamic_discovery": False,
        "initialize_from_original": not args.gate_random_init,
    }
    session.replace_source(
        ctx.gate_source,
        class_name=(
            args.gate_class
            if args.gate_source is not None
            else "Net"
        ),
        **replace_kwargs,
    )
    carried = False
    carry_source_id = getattr(ctx, "carry_source_id", None)
    if getattr(args, "gate_carry_forward", None) is not None and carry_source_id is not None:
        checkpoint = latest_trained_checkpoint(ctx.gate_root, carry_source_id)
        if checkpoint is None:
            print(
                f"[GATE CARRY] gate {carry_source_id:03d} has no trained checkpoint "
                "yet; installing from the native copy"
            )
        else:
            ctx.carry_forward_metrics = carry_gate_weights(session.installs, checkpoint)
            ctx.carry_forward_metrics.update({
                "mode": args.gate_carry_forward,
                "source_gate_id": carry_source_id,
                "target_gate_id": ctx.candidate_index,
            })
            carried = True
            print(
                f"[GATE CARRY] trained weights of gate {carry_source_id:03d} -> "
                f"gate {ctx.candidate_index:03d} initialization "
                f"({ctx.carry_forward_metrics['n_parameters_copied']} parameters copied, "
                f"{ctx.carry_forward_metrics['n_parameters_fresh']} fresh)"
            )
    initialization_metrics = [] if args.gate_random_init else _perturb_gate_weights(
        session.installs, args.gate_init_noise_scale, args.seed
    )
    replacement_logits = _model_logits(session.model, ctx.sample_input)
    if not torch.isfinite(replacement_logits).all():
        raise RuntimeError("Replacement gates produced non-finite model logits")
    max_abs_diff = float((ctx.native_logits - replacement_logits).abs().max().item())
    equivalent = bool(torch.allclose(ctx.native_logits, replacement_logits, rtol=1e-5, atol=1e-5))
    parent_check = None
    parent_logits = getattr(ctx, "parent_logits", None)
    if carried and parent_logits is not None:
        parent_check = verify_parent_function_carry(parent_logits, replacement_logits)
    equivalence = {
        "equivalent": equivalent,
        "max_abs_logit_difference": max_abs_diff,
        "rtol": 1e-5,
        "atol": 1e-5,
        "gate_init_noise_scale": args.gate_init_noise_scale,
        "initialize_from_original": not args.gate_random_init,
        "load_in_8bit": args.load_in_8bit,
        "native_forward_bit_repeatable": ctx.native_repeatable,
        "native_repeat_max_abs_logit_difference": ctx.native_repeat_max_abs_difference,
        "carried_trained_weights": carried,
        "carry_source_gate_id": carry_source_id,
        "parent_function_equivalent": parent_check["equivalent"] if parent_check else None,
        "parent_logit_bit_exact": parent_check["bit_exact"] if parent_check else None,
        "parent_logit_max_abs_difference": (
            parent_check["max_abs_difference"] if parent_check else None
        ),
    }
    ctx.install_metrics = {
        "step_zero": equivalence,
        "initialization": initialization_metrics,
        "carry_forward": getattr(ctx, "carry_forward_metrics", None),
    }
    # Compact install receipt: the full metrics live in record.json; this
    # line makes the install auditable straight from the log (native-copy
    # vs carried, divergence size).
    print(
        f"[{_ts()}] [CYCLE] gate_{getattr(ctx, 'candidate_index', 0):03d} INSTALLED: "
        f"init={'gate_%03d-weights' % carry_source_id if carried else 'native-copy'}, "
        f"step-zero max_abs_diff={max_abs_diff:.3g}, equivalent={equivalent}, "
        f"parent-check={'n/a' if parent_check is None else parent_check['equivalent']}"
    )
    if parent_check is not None and not parent_check["equivalent"]:
        # Reject-and-continue: the candidate is recorded as failed by the
        # caller, the incumbent stays the self.
        raise RuntimeError(
            "Function-preserving carry failed: the successor gate does not "
            "reproduce the parent gate's step-zero logits "
            f"(max_abs_difference={parent_check['max_abs_difference']:.6g}); "
            "keep the inherited modules unchanged and put new capacity in "
            "zero-initialized branches"
        )
    skip_step_zero_verify = getattr(args, "gate_skip_step_zero_verify", False)
    if not equivalent and skip_step_zero_verify:
        print(
            "[GATE VERIFY] step-zero mismatch accepted (--gate-skip-step-zero-verify): "
            f"max_abs_difference={max_abs_diff}"
        )
    if carried and not equivalent:
        print(
            "[GATE CARRY] step-zero logits diverge from native as expected after "
            f"weight carry: max_abs_difference={max_abs_diff}"
        )
    if (
        not args.gate_random_init
        and args.gate_init_noise_scale == 0
        and not equivalent
        and not skip_step_zero_verify
        and not carried
    ):
        raise RuntimeError(
            "Replacement gates do not reproduce native step-zero model logits: "
            f"max_abs_difference={max_abs_diff}"
        )
    session.freeze_except_gates()


def _prepare_seed_candidate(ctx: RunContext, gate_id: int) -> None:
    """Cold-start bootstrap: install a built-in compliant seed gate.

    Seeds bypass the LLM proposal entirely so the store accumulates scored,
    valid, creative references even when the proposer cannot produce one.
    Seeds are validated with the same rules as LLM proposals and fail loudly
    if they ever stop complying.
    """
    ctx.candidate_index = gate_id
    source = seed_gate_source(gate_id)
    _validate_gate_source(source, ctx.shapes)
    ctx.candidate_root = create_gate_entry(
        ctx.gate_root, gate_id, source,
        seed_name=seed_gate_name(gate_id), kind="seed",
    )
    ctx.gate_dir = ctx.candidate_root
    ctx.proposal_prompt = (
        f"[cold-start bootstrap] built-in seed gate '{seed_gate_name(gate_id)}' "
        "installed without an LLM proposal"
    )
    ctx.proposal_attempts = []
    ctx.gate_source = source
    ctx.seeded = True
    ctx.seed_name = seed_gate_name(gate_id)
    ctx.gate_kind = "seed"
    ctx.previous_feedback_summary = _initial_feedback_summary(source)
    ctx.used_prompts.clear()
    print(
        f"[GATE SEARCH] gate {gate_id:03d} seeded (cold-start bootstrap): "
        f"{seed_gate_name(gate_id)}"
    )


def _prepare_rematch_candidate(ctx: RunContext, gate_id: int, source_gate_id: int) -> None:
    """Re-install a prior gate's code verbatim for a re-measure.

    Duplicate-structure rejection is intentionally bypassed: the rematch
    exists precisely to re-score the incumbent under the current proposer.
    """
    ctx.candidate_index = gate_id
    source_file = gate_dir(ctx.gate_root, source_gate_id) / "gate.py"
    source = source_file.read_text(encoding="utf-8")
    ctx.candidate_root = create_gate_entry(ctx.gate_root, gate_id, source, kind="rematch")
    ctx.gate_dir = ctx.candidate_root
    ctx.proposal_prompt = (
        f"[rematch] re-measuring gate {source_gate_id:03d} under proposer "
        f"{ctx.llm_version}"
    )
    ctx.proposal_attempts = []
    ctx.gate_source = source
    ctx.seed_name = None
    ctx.gate_kind = "rematch"
    ctx.reference_gate_id = source_gate_id
    ctx.previous_feedback_summary = _initial_feedback_summary(source)
    ctx.used_prompts.clear()
    print(
        f"[GATE SEARCH] rematch: re-measuring gate {source_gate_id:03d} under "
        f"proposer {ctx.llm_version} as gate {gate_id:03d}"
    )
