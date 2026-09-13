"""Compatibility shim for the original MoE gate experiment.

This module previously contained a standalone gate experiment loop.  It could
not run: ``_DEFAULT_GATE_CODE = _BASELINE_GATE_CODE`` referenced an undefined
name, raising ``NameError`` at import, and the gate factory initialised routers
randomly instead of copying the native projection.

The supported implementation now lives in the ``moe_cycle`` package and is
driven by ``run_moe_gate_cycle.py --gate-outer-search``.  The only loop that
belongs in the shared pipeline is the CV generation/evaluation one; gate search
is orchestrated from ``moe_cycle`` so this file no longer duplicates it.

Kept as a shim so any external import of the old symbol names still resolves.
"""

from __future__ import annotations

from moe_cycle.cycle import main as gate_experiment_main
from moe_cycle.gate_prompt import BASELINE_GATE_CODE as _DEFAULT_GATE_CODE
from moe_cycle.gate_source import _generate_gate
from moe_cycle.gate_store import (
    best_gate,
    load_gate_summaries,
    summarize_epoch_metrics,
    summarize_gate,
)
from moe_gate_only import (
    MoEGateSession,
    find_moe_gates,
    freeze_except_gates,
    install_gates,
)

__all__ = [
    "_DEFAULT_GATE_CODE",
    "_generate_gate",
    "best_gate",
    "find_moe_gates",
    "gate_experiment_main",
    "install_gates",
    "freeze_except_gates",
    "load_gate_summaries",
    "MoEGateSession",
    "summarize_epoch_metrics",
    "summarize_gate",
]
