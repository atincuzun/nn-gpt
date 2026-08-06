"""LLM-driven gate source extraction, validation, and generation."""

from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path
from typing import Any


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
                raise ValueError("Generated student gate forward must be deterministic")
    namespace: dict[str, Any] = {"__name__": "_generated_gate"}
    exec(compile(tree, "<generated-gate>", "exec"), namespace)
    gate_cls = namespace.get(class_name)
    if not isinstance(gate_cls, type) or not issubclass(gate_cls, nn.Module):
        raise TypeError(f"Generated source must define nn.Module class {class_name}")
    for model_dim, num_experts in shapes:
        gate = gate_cls(model_dim, num_experts).float().eval()
        if any(isinstance(module, (nn.Dropout, nn.modules.batchnorm._BatchNorm)) for module in gate.modules()):
            raise ValueError("Generated student gates must be deterministic; dropout/batchnorm are unsupported")
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
    max_new_tokens: int, artifact_dir: Path, *, random_student: bool,
) -> str:
    initialization = (
        "The gate is a randomly initialized student trained beside a frozen native router. "
        "Do not assume its base weight is copied from the native router."
        if random_student
        else "Its base weight will be copied from the native router."
    )
    residual_requirement = (
        "Any residual branch may use normal random initialization."
        if random_student
        else "Initialize any residual branch's final projection to zero for native step-zero routing."
    )
    prompt = f"""
You are writing a tiny MoE router scorer, NOT a full neural network. Do NOT emit <nn>, <hp>, or <tr> blocks, datasets, training loops, or markdown.
Write one complete Python module defining exactly one class named LLMGeneratedGate.
Requirements:
- Import only torch and torch.nn.
- Inherit torch.nn.Module.
- Constructor: __init__(self, model_dim: int, num_experts: int).
- Forward: forward(self, x), accepting (..., model_dim) and returning finite raw logits (..., num_experts).
- Define self.base = nn.Linear(model_dim, num_experts, bias=False). {initialization}
- Return self.base(x), optionally with a small trainable residual branch.
- {residual_requirement}
- Do not apply softmax, top-k, expert dispatch, or auxiliary losses.
- Forward must be deterministic: do not sample random values or use dropout/batch normalization.
- Do not hard-code dimensions, move devices inside forward, or return tuples.
- Keep the gate small, differentiable, and numerically stable.
- The class must be COMPLETE and syntactically valid: close every string literal and parenthesis, and finish the class body before the closing tag. Never stop after the class header.
- Router shapes: {shapes!r}.
Expected format (example):
<gate>
import torch
import torch.nn as nn

class LLMGeneratedGate(nn.Module):
    def __init__(self, model_dim: int, num_experts: int):
        super().__init__()
        self.base = nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)
</gate>
Output only the complete source between <gate> and </gate>, without markdown or explanation.
""".strip()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    previous_error = ""
    for attempt in range(1, attempts + 1):
        current_prompt = prompt
        if previous_error:
            current_prompt += f"\nThe previous proposal failed validation: {previous_error}\nReturn a corrected implementation."
        _, _, _, raw = chat_bot.chat(
            current_prompt, engineer_prompt=False, max_new_tokens=max_new_tokens,
        )
        (artifact_dir / f"generation_attempt_{attempt}.txt").write_text(raw, encoding="utf-8")
        errors: list[str] = []
        for candidate in _gate_candidates(raw):
            try:
                _validate_gate_source(candidate, shapes)
                (artifact_dir / "gate.py").write_text(candidate.rstrip() + "\n", encoding="utf-8")
                return candidate
            except Exception as exc:
                errors.append(str(exc))
        previous_error = "; ".join(errors) or "No usable <gate> source was found"
    raise RuntimeError(f"LLM did not generate a valid gate after {attempts} attempts: {previous_error}")
