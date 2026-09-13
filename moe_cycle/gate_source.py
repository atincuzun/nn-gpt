"""LLM-driven gate source extraction, validation, and generation."""

from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path
from typing import Any

from .gate_prompt import (
    gate_prompt_scope,
    gate_proposal_prompt,
    is_duplicate_gate,
)
from .gate_store import structural_hash


def _gate_candidates(raw: str):
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw or "", flags=re.IGNORECASE).strip()
    for match in re.finditer(r"<gate\b[^>]*>([\s\S]*?)</gate\s*>", cleaned, flags=re.IGNORECASE):
        yield textwrap.dedent(match.group(1)).strip()
    for match in re.finditer(r"```(?:python|py)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE):
        yield textwrap.dedent(match.group(1)).strip()
    starts = [
        pos for pos in (
            cleaned.find("import torch"),
            cleaned.find("from torch"),
            cleaned.find("class LLMGeneratedGate"),
        ) if pos >= 0
    ]
    if starts:
        snippet = textwrap.dedent(cleaned[min(starts):]).strip()
        # Generation can truncate mid-line. Yield progressively shorter
        # prefixes so a syntactically complete class body still parses.
        lines = snippet.splitlines()
        for cut in range(len(lines), 0, -1):
            yield "\n".join(lines[:cut]).strip()


def _validate_gate_source(
    source: str,
    shapes: list[tuple[int, int]],
    class_name: str = "LLMGeneratedGate",
) -> None:
    import torch
    import torch.nn as nn

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] != "torch" for alias in node.names):
                raise ValueError("Generated gate may import only torch")
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] != "torch":
            raise ValueError("Generated gate may import only torch")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {
                "bernoulli", "dropout", "multinomial", "normal",
                "rand", "rand_like", "randn", "randn_like",
            }:
                raise ValueError("Generated gate forward must be deterministic")
    namespace: dict[str, Any] = {"__name__": "_generated_gate"}
    exec(compile(tree, "<generated-gate>", "exec"), namespace)
    gate_cls = namespace.get(class_name)
    if not isinstance(gate_cls, type) or not issubclass(gate_cls, nn.Module):
        raise TypeError(f"Generated source must define nn.Module class {class_name}")
    for model_dim, num_experts in shapes:
        gate = gate_cls(model_dim, num_experts).float().eval()
        if any(isinstance(module, (nn.Dropout, nn.modules.batchnorm._BatchNorm)) for module in gate.modules()):
            raise ValueError("Generated gates must be deterministic; dropout/batchnorm are unsupported")
        base = getattr(gate, "base", None)
        if not isinstance(base, nn.Linear) or tuple(base.weight.shape) != (num_experts, model_dim):
            raise ValueError(
                "Generated gate must define base = nn.Linear(model_dim, num_experts, bias=False)"
            )
        for input_shape in ((2, model_dim), (2, 3, model_dim)):
            sample = torch.randn(*input_shape)
            with torch.no_grad():
                output = gate(sample)
            expected = input_shape[:-1] + (num_experts,)
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected:
                raise ValueError(f"Expected gate output {expected}, got {getattr(output, 'shape', None)}")
            if not torch.isfinite(output).all():
                raise ValueError("Generated gate returned non-finite logits")


def _generate_gate(
    chat_bot: Any, shapes: list[tuple[int, int]], attempts: int,
    max_new_tokens: int, artifact_dir: Path, feedback_summary: str = "",
    reference_source: str = "",
    seen_hashes: set[str] | None = None,
    max_attempts_per_duplicate: int = 3,
) -> str:
    """Ask the LLM for a replacement gate source.

    Its base weight is copied from the native router, so the replaced model stays
    bit-identical at step zero before training begins.  The prompt carries the
    current best gate as a reference plus measured feedback from earlier
    candidates, mirroring how the CV pipeline conditions generation on one
    reference artefact.
    """
    prompt = gate_proposal_prompt(
        shapes, reference_source=reference_source, feedback=feedback_summary,
    )
    seen = seen_hashes if seen_hashes is not None else set()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "proposal_prompt.txt").write_text(prompt, encoding="utf-8")
    previous_error = ""
    duplicate_attempts = 0
    for attempt in range(1, attempts + 1):
        current_prompt = prompt
        if previous_error:
            current_prompt += f"\nThe previous proposal failed validation: {previous_error}\nReturn a corrected implementation."
        with gate_prompt_scope(chat_bot):
            _, _, _, raw = chat_bot.chat(
                current_prompt, engineer_prompt=False, max_new_tokens=max_new_tokens,
            )
        (artifact_dir / f"generation_attempt_{attempt}.txt").write_text(raw, encoding="utf-8")
        errors: list[str] = []
        # A single generation can yield several parseable strings.  Duplicate
        # attempts are counted per generation, not per string, otherwise one
        # templated response would exhaust the retry budget immediately.
        saw_valid_candidate = False
        for candidate in _gate_candidates(raw):
            try:
                _validate_gate_source(candidate, shapes)
            except Exception as exc:
                errors.append(str(exc))
                continue
            saw_valid_candidate = True
            if is_duplicate_gate(candidate, seen):
                errors.append("structurally identical to an earlier gate candidate")
                continue
            seen.add(structural_hash(candidate))
            (artifact_dir / "gate.py").write_text(candidate.rstrip() + "\n", encoding="utf-8")
            return candidate
        if saw_valid_candidate:
            # Every structurally valid proposal this round was already seen.
            duplicate_attempts += 1
            if duplicate_attempts >= max_attempts_per_duplicate:
                raise RuntimeError(
                    "LLM kept re-emitting a duplicate gate architecture; "
                    "cannot build a distinct candidate"
                )
        previous_error = "; ".join(errors) or "No usable <gate> source was found"
    raise RuntimeError(f"LLM did not generate a valid gate after {attempts} attempts: {previous_error}")
