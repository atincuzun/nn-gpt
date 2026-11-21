"""Entry point for one controlled MoE router experiment arm.

Importing ``run_moe_gate_cycle`` first applies the same process setup as the
gate cycle: NNGPT_DIR_OVERRIDE from ``--output`` before any ``ab.gpt`` import,
the thinking-stripping extractors, and the generation dtype policy. The arm
itself is implemented in ``moe_cycle.experiment``; see its docstring, and
``moe_experiments/dispatch.py`` for the experiment plan that drives it.
"""

import run_moe_gate_cycle  # noqa: F401  (process setup side effects)

from moe_cycle.experiment import main

if __name__ == "__main__":
    main()
