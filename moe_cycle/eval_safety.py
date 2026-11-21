"""MoE-cycle-scoped safety patch for generated evaluation parameters.

The shared evaluator assumes that every truthy ``hp.txt`` JSON payload is a
mapping.  An LLM can instead emit valid JSON of another type (for example a
list of parameter names), which makes the evaluator fail before it can isolate
the bad generated model.  This module intentionally leaves the shared Tune and
Eval sources untouched: the MoE entrypoint installs a wrapper around Tune's
evaluation function at runtime.
"""

from __future__ import annotations

import functools
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


def _quarantine_destination(hp_path: Path) -> Path:
    """Return a non-conflicting sidecar path for a malformed ``hp.txt``."""
    base = hp_path.with_name(f"{hp_path.name}.moe-invalid")
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}.{suffix}")
        suffix += 1
    return candidate


def quarantine_invalid_hyperparameters(models_dir: str | Path) -> list[Path]:
    """Hide non-mapping ``hp.txt`` files from the shared evaluator.

    Invalid files are renamed, not deleted, so the exact LLM output remains
    available for diagnosis.  With ``hp.txt`` absent, the evaluator follows its
    existing command-line/default path and can still merge a valid ``prm`` from
    ``dataframe.df``.
    """
    quarantined: list[Path] = []
    root = Path(models_dir)
    if not root.is_dir():
        return quarantined

    for hp_path in sorted(root.glob("B*/hp.txt")):
        try:
            payload = json.loads(hp_path.read_text(encoding="utf-8"))
        except Exception as exc:
            reason = f"invalid JSON ({exc})"
        else:
            if isinstance(payload, dict):
                continue
            reason = f"expected a JSON object, got {type(payload).__name__}"

        destination = _quarantine_destination(hp_path)
        hp_path.replace(destination)
        quarantined.append(destination)
        print(
            f"[MOE EVAL SAFETY] {hp_path.parent.name}: ignoring malformed "
            f"{hp_path.name}: {reason}; preserved as {destination.name}",
            flush=True,
        )

    return quarantined


def _wrap_evaluate_epoch(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap Tune's evaluator with MoE-only generated-artifact validation."""
    if getattr(original, "_moe_eval_safety", False):
        return original

    @functools.wraps(original)
    def guarded_evaluate_epoch(
        epoch,
        out_path,
        nn_name_prefix,
        nn_train_epochs,
        trans_mode,
        classification_mode=False,
        custom_synth_dir=None,
        trainer_model=None,
        aug_mode=False,
        **kwargs,
    ):
        if not (trans_mode or classification_mode or aug_mode):
            if custom_synth_dir is not None:
                models_dir = Path(custom_synth_dir)
            else:
                from ab.gpt.util.Const import synth_dir

                models_dir = synth_dir(out_path)
            quarantine_invalid_hyperparameters(models_dir)

        return original(
            epoch,
            out_path,
            nn_name_prefix,
            nn_train_epochs,
            trans_mode,
            classification_mode=classification_mode,
            custom_synth_dir=custom_synth_dir,
            trainer_model=trainer_model,
            aug_mode=aug_mode,
            **kwargs,
        )

    guarded_evaluate_epoch._moe_eval_safety = True  # type: ignore[attr-defined]
    guarded_evaluate_epoch._moe_original = original  # type: ignore[attr-defined]
    return guarded_evaluate_epoch


@contextmanager
def _moe_cv_eval_environment(fixed_transform: str | None = None):
    """Retain the fork's parameter precedence and scoring in MoE.

    ``NNGPT_MOE_EVAL_CUDA`` optionally exposes a GPU to the evaluator
    subprocess only, so a CPU-resident proposer can still train the
    generated CV models on a GPU without ChatBot trying to place the LLM
    there.  The value ``cpu`` keeps the evaluator off the GPU entirely, which
    is what a shared card needs: the evaluator's training run then cannot
    collide with the LLM's weights.

    The evaluation runs in ``fixed`` prm mode so the transform is the one the
    run declares.  In the default ``seed`` mode the seed record's own
    transform wins, which measured at 256 square while the protocol stored
    with every gate said 32: the recorded protocol did not describe what was
    measured, and the training cost 64 times more than the protocol implied.
    """
    overrides = {"NNGPT_EVAL_PRM_MODE": "fixed", "NNGPT_OOM_RETRY": "0"}
    if fixed_transform:
        overrides["NNGPT_FIXED_EVAL_TRANSFORM"] = fixed_transform
    eval_cuda = os.environ.get("NNGPT_MOE_EVAL_CUDA", "").strip()
    if eval_cuda:
        overrides["CUDA_VISIBLE_DEVICES"] = (
            "" if eval_cuda.lower() == "cpu" else eval_cuda
        )
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def ensure_moe_eval_safety() -> None:
    """Install the safety wrapper only in a process running the MoE cycle."""
    import ab.gpt.util.tune.Tune as tune_module

    tune_module._evaluate_epoch = _wrap_evaluate_epoch(tune_module._evaluate_epoch)
