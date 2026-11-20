"""Entry point for the iterative MoE gate-training NAS cycle.

The cycle itself lives in focused modules: ``setup`` validates and initializes
the run, ``search`` orchestrates outer candidates, ``candidates`` prepares and
installs gates, ``epochs`` runs the inner loop, and ``records`` persists the
results. This module only wires them together.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .candidates import _install_and_verify, _prepare_gate_candidate
from .cli import parse_args
from .epochs import _run_epochs
from .feedback import candidate_feedback
from .records import _write_final_summary, _write_gate_record
from .search import _run_outer_search
from .setup import _setup_run, _validate_args


def main() -> None:
    from .eval_safety import ensure_moe_eval_safety
    from .generation_dtype import ensure_generation_dtype_policy

    ensure_generation_dtype_policy()
    # Keep malformed LLM-generated hp payloads from aborting the shared
    # evaluator without changing Tune.py or Eval.py themselves.
    ensure_moe_eval_safety()
    args = parse_args()
    _validate_args(args)
    ctx = _setup_run(args)
    if args.gate_outer_search:
        _run_outer_search(ctx, args)
    else:
        outer_feedback = ""
        for candidate_index in range(args.gate_candidates):
            if outer_feedback:
                # Pre-dates the keyword-only signature; accumulated results only
                # reach later candidates through the log and the best-gate
                # reference prompt.
                print(outer_feedback)
            ctx.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            ctx.proposal_attempts = []
            ctx.install_metrics = None
            _prepare_gate_candidate(ctx, candidate_index)
            # The context restores native routers after every candidate, so no
            # candidate inherits another candidate's trained gate parameters.
            with ctx.session:
                _install_and_verify(ctx)
                _run_epochs(ctx)
            _write_gate_record(ctx)
            outer_feedback += "\n" + candidate_feedback(
                ctx.gate_epoch_metrics, candidate_index,
            )
    _write_final_summary(ctx)
