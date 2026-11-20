"""Shared runtime context and helpers for the MoE gate cycle."""

from __future__ import annotations

import copy
from argparse import Namespace
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    """Everything produced by ``_setup_run`` and consumed by the cycle phases."""

    args: Namespace
    run_root: Path
    session: Any
    tokenizer: Any
    chat_bot: Any
    sites: list[Any]
    shapes: list[tuple[int, int]]
    gate_dir: Path
    gate_source: str | None
    sample_input: Any
    prompt_dict: dict[str, Any]
    native_logits: Any
    native_gate_inputs: dict[int, Any]
    native_repeatable: bool
    native_repeat_max_abs_difference: float
    previous_feedback_summary: str
    proposal_prompt: str | None = None
    proposal_attempts: list = field(default_factory=list)
    install_metrics: dict | None = None
    created_at: str | None = None
    used_prompts: list = field(default_factory=list)
    candidate_index: int = 0
    candidate_root: Path | None = None
    gate_root: Path | None = None
    gate_epoch_metrics: list = field(default_factory=list)
    reference_gate_id: int | None = None
    reference_accuracy: float | None = None
    goal_accuracy: float | None = None
    outer_summary: dict | None = None
    author_gate_id: int | None = None
    llm_version: str = "base"
    rematch_for: int | None = None
    gate_pool: list = field(default_factory=list)
    seeded: bool = False
    seed_name: str | None = None
    gate_kind: str = "llm"
    reused_from: int | None = None
    carry_source_id: int | None = None
    parent_logits: Any = None
    carry_forward_metrics: dict | None = None


def _log_cuda_memory(label: str) -> dict[str, float] | None:
    """Print current-process CUDA memory so weight and activation use are visible."""
    import torch

    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    gib = 1024 ** 3
    status = {
        "allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "device_free_gib": free_bytes / gib,
        "device_total_gib": total_bytes / gib,
    }
    print(
        f"[VRAM] {label}: allocated={status['allocated_gib']:.2f} GiB, "
        f"reserved={status['reserved_gib']:.2f} GiB, "
        f"peak={status['peak_allocated_gib']:.2f} GiB, "
        f"device_free={status['device_free_gib']:.2f}/"
        f"{status['device_total_gib']:.2f} GiB"
    )
    return status


def _ts() -> str:
    """Wall-clock tag for milestone prints.

    nohup logs carry no timestamps on plain print() output, which made
    post-run forensics ("which gate ran when, for how long") needless file
    archaeology; every [CYCLE] milestone line below is anchored in time.
    """
    return datetime.now().strftime("%H:%M:%S")


def _release_cuda_memory() -> None:
    """Return fragmented CUDA reserve to the driver between candidates.

    A candidate that OOMs leaves GiBs reserved-but-unallocated; the next
    candidate's install-verification forward then inherits the shortage and
    fails for no architectural reason (run 20260922_222052: gates 032/033
    were rolled back this way right after gate 031's training OOM).
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _nas_prefixes(args: Namespace) -> tuple[str, ...]:
    """Return stable LEMUR prefixes including this cycle's generated models."""
    values = [*args.sft_nn_prefixes, args.nn_name_prefix]
    return tuple(dict.fromkeys(value for value in values if value))


def _generation_prefixes(args: Namespace) -> tuple[str, ...]:
    """Return comparable generation seeds plus this cycle's successful models."""
    values = [*args.generation_nn_prefixes, args.nn_name_prefix]
    return tuple(dict.fromkeys(value for value in values if value))


def _pinned_prompt_dict(
    prompt_dict: dict[str, Any],
    conf_keys: tuple[str, ...] | list[str],
    *,
    dataset: str,
    nn_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    """Pin the shared CV prompt config to the run's dataset and seed prefixes
    without modifying the shared nn_gen function.

    Gate feedback deliberately never enters CV generation or gate-training
    prompts: candidates are compared on CV accuracy, so a candidate's own gate
    code/score must never leak into the prompt it is scored or trained on.
    """
    rendered = copy.deepcopy(prompt_dict)
    for key in conf_keys:
        if key not in rendered:
            raise KeyError(f"Prompt key {key!r} is missing from the test prompt config")
        key_config = rendered[key]
        key_config["dataset"] = dataset
        key_config["nn_prefixes"] = list(nn_prefixes)
    return rendered


def _initial_feedback_summary(gate_source: str | None) -> str:
    gate_code = gate_source or "[installed gate]"
    if len(gate_code) > 2000:
        gate_code = gate_code[:2000] + "\n# ... (truncated)"
    return (
        "No previous CV generation cycle has been evaluated. Establish a valid "
        "NAS baseline and prioritize executable LEMUR candidates.\n"
        f"The current MoE router gate is:\n<gate>\n{gate_code}\n</gate>"
    )


def _prepare_model_for_generation(ctx: RunContext) -> None:
    """Put the model into the state generation runs under (eval, KV cache, no
    activation checkpointing), both for CV generation and gate proposals."""
    session = ctx.session
    session.model.eval()
    session.model.config.use_cache = True
    if ctx.args.gradient_checkpointing and hasattr(
        session.model, "gradient_checkpointing_disable",
    ):
        session.model.gradient_checkpointing_disable()
