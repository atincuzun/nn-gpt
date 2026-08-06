"""Iterative MoE gate-training NAS cycle mirroring Tune.py's tune() suite.

Each epoch:
  1. Generate NN architectures (nn_gen) using the gate-replaced LLM
  2. Evaluate generated NNs through Tune._evaluate_epoch
  3. Build NNGenPrompt data from real evaluated LEMUR results
  4. Train only generated MoE gates on language-model and router-distillation loss

Module layout
-------------
- ``cli``: argument parsing
- ``gate_source``: LLM gate source extraction, validation, and generation
- ``feedback``: gate-outcome feedback summaries for training prompts
- ``morphism``: step-zero morphism verification and initialization metrics
- ``cycle``: the orchestration phases (setup, install/verify, epochs, summary)
"""

from .cycle import main
from .morphism import _verify_morphed_gate_projections

__all__ = ["main", "_verify_morphed_gate_projections"]
