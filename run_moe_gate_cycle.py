"""Thin entry point for the MoE gate-training NAS cycle.

Implementation lives in the ``moe_cycle`` package; this module exists to keep
``python run_moe_gate_cycle.py`` and legacy imports working unchanged.

The in-pipeline extractors (``ab.gpt.util.Util``) run on raw chat text and get
contaminated by the reasoning model's ``<think>`` blocks: the anchor search can
start inside the thinking region and leak it into new_nn.py/hp.txt/tr.py.
Before anything else is imported, the Util extraction functions are swapped
for the thinking-stripping implementations in ``parse_nn_generation``.
Downstream contract checks (train_setup/learn/supported_hyperparameters) are
untouched and still run in Eval/verify_nn_code.
"""

import argparse as _argparse
import os as _os
from pathlib import Path as _Path

# ab.gpt.util.Const captures NNGPT_DIR_OVERRIDE at import time, so the flag
# must be exported BEFORE any ab.gpt import happens.
_pre_parser = _argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--output")
_known_args, _ = _pre_parser.parse_known_args()
if _known_args.output:
    _os.environ["NNGPT_DIR_OVERRIDE"] = str(
        _Path(_known_args.output).resolve() / "nngpt"
    )

import parse_nn_generation as _nn_parser

from ab.gpt.util import Util as _Util


def _extract_code(txt: str):
    if not isinstance(txt, str):
        return None
    code, truncated = _nn_parser.extract_nn_code(txt)
    if code is None:
        return None
    if truncated:
        print("[EXTRACT] NN block TRUNCATED (missing </nn>); code may be incomplete")
    return code


def _extract_hyperparam(txt: str):
    if not isinstance(txt, str):
        return None
    hp, truncated = _nn_parser.extract_hyperparam(txt)
    if hp is not None and truncated:
        print("[EXTRACT] <hp> block TRUNCATED (missing </hp>)")
    return hp


def _extract_transform(txt: str):
    if not isinstance(txt, str):
        return None
    tr, truncated = _nn_parser.extract_transform(txt)
    if tr is None:
        return None
    if truncated:
        print("[EXTRACT] <tr> block TRUNCATED (missing </tr>)")
    return tr


_Util.extract_code = _extract_code
_Util.extract_hyperparam = _extract_hyperparam
_Util.extract_transform = _extract_transform

# Scope-bounded fix: generation-head dtype inference must honor the ACTIVE
# bnb quantization mode (8-bit configs carry a float32 4-bit default that
# otherwise poisons lm_head/config dtype and crashes MoE conv blocks).
from moe_cycle.generation_dtype import ensure_generation_dtype_policy

ensure_generation_dtype_policy()

from moe_cycle.cycle import main
from moe_cycle.morphism import _verify_morphed_gate_projections

__all__ = ["main", "_verify_morphed_gate_projections"]

if __name__ == "__main__":
    main()
