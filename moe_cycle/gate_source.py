"""LLM-driven gate source extraction, validation, and generation."""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path
from typing import Any

from .code_normalization import (
    strip_xml_tags,
    unescape_embedded_escapes,
    unescape_single_line_code,
)
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
            cleaned.find("class Net"),
        ) if pos >= 0
    ]
    if starts:
        snippet = textwrap.dedent(cleaned[min(starts):]).strip()
        # Generation can truncate mid-line. Yield progressively shorter
        # prefixes so a syntactically complete class body still parses.
        lines = snippet.splitlines()
        for cut in range(len(lines), 0, -1):
            yield "\n".join(lines[:cut]).strip()


def _gate_candidate_variants(candidate: str):
    """Yield a candidate and its recovered forms.

    A model that returns the module on one line with escaped newlines (and
    with its XML framing still attached) produces a payload that never
    parses, so the validator would reject it even though the code is fine.
    The embedded-escape form covers the payload that mixes real line breaks
    with literal ``\\n``; every variant is parsed before it is accepted, so a
    wrong rewrite is skipped rather than used.
    """
    yield candidate
    for recovered in (
        unescape_single_line_code(candidate),
        strip_xml_tags(candidate),
        unescape_embedded_escapes(candidate),
    ):
        if recovered != candidate:
            yield recovered
    for recovered in (
        unescape_single_line_code(strip_xml_tags(candidate)),
        unescape_embedded_escapes(strip_xml_tags(candidate)),
        unescape_embedded_escapes(unescape_single_line_code(candidate)),
    ):
        if recovered != candidate:
            yield recovered


_LEMUR_HP = "\n\ndef supported_hyperparameters():\n    # Hyperparameters accepted by this routing gate.\n    return {\"lr\", \"momentum\"}\n"

_LEMUR_METHODS = """    def train_setup(self, prm):
        # Build the optimizer for gate-specific routing training.
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=float(prm.get("lr", 1e-3)),
            momentum=float(prm.get("momentum", 0.9)),
        )
        self.criterion = nn.MSELoss()
        self.device = next(self.parameters()).device

    def learn(self, train_data):
        # Train on (hidden_states, target_logits) routing pairs.
        self.train()
        total_loss = 0.0
        total = 0
        for hidden_states, target_logits in train_data:
            hidden_states = hidden_states.to(self.device)
            target_logits = target_logits.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self(hidden_states), target_logits)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach()) * len(hidden_states)
            total += len(hidden_states)
        return 0.0, total_loss / max(total, 1)

"""


def _normalize_gate_class(source: str, class_name: str = "Net") -> str:
    """Rename the single generated nn.Module class to the host class name.

    Older proposer adapters were trained with a different class name; the
    architecture and forward are unchanged, so the harness renames it before
    validation and stores only the normalized source.
    """
    tree = ast.parse(source)
    if any(
        isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body
    ):
        return source
    module_classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(base, ast.Attribute) and base.attr == "Module")
            or (isinstance(base, ast.Name) and base.id == "Module")
            for base in node.bases
        )
    ]
    if len(module_classes) != 1:
        return source
    old = module_classes[0].name
    return re.sub(rf"\b{re.escape(old)}\b", class_name, source)


def _ensure_lemur_interface(source: str) -> str:
    """Add the standard LEMUR code contract without changing the gate forward."""
    tree = ast.parse(source)
    if not any(
        isinstance(node, ast.ClassDef) and node.name == "Net" for node in tree.body
    ):
        return source
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Net")
    has_train = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "train_setup"
        for node in cls.body
    )
    has_learn = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "learn"
        for node in cls.body
    )
    has_hp = any(
        isinstance(node, ast.FunctionDef) and node.name == "supported_hyperparameters"
        for node in tree.body
    )
    lines = source.splitlines(keepends=True)
    if not (has_train and has_learn):
        forward = next(
            (node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward"),
            None,
        )
        insert_at = (forward.lineno - 1) if forward is not None else cls.end_lineno
        while insert_at > 0 and not lines[insert_at - 1].strip() and forward is not None:
            insert_at -= 1
        lines.insert(insert_at, _LEMUR_METHODS)
        source = "".join(lines)
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Net")
    if not has_hp:
        lines = source.splitlines(keepends=True)
        class_line = cls.lineno - 1
        lines.insert(class_line, _LEMUR_HP)
        source = "".join(lines)
    return source


def _validate_gate_source(
    source: str,
    shapes: list[tuple[int, int]],
    class_name: str = "Net",
    *,
    require_base: bool = True,
) -> None:
    import torch
    import torch.nn as nn

    tree = ast.parse(source)
    try:
        from ab.nn.util.ArchUID import is_model_code as _lemur_is_model_code
    except ImportError:
        _lemur_is_model_code = None
    if _lemur_is_model_code is not None:
        if not _lemur_is_model_code(source):
            raise ValueError(
                "Generated gate must satisfy the LEMUR model-code contract "
                "(class Net with train_setup)"
            )
    else:
        if not any(
            isinstance(node, ast.ClassDef) and node.name == "Net"
            for node in tree.body
        ) or not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "train_setup"
            for node in ast.walk(tree)
        ):
            raise ValueError(
                "Generated gate must satisfy the LEMUR model-code contract "
                "(class Net with train_setup)"
            )
    if not any(
        isinstance(node, ast.FunctionDef)
        and node.name == "supported_hyperparameters"
        for node in tree.body
    ):
        raise ValueError(
            "Generated gate must define module-level supported_hyperparameters()"
        )
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
    # The gate runs inside the frozen model, on the model's device. A forward
    # that moves its input elsewhere installs cleanly and then dies in the
    # step-zero check, which wastes an install instead of a proposal.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "forward"):
            continue
        for inner in ast.walk(node):
            if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)):
                continue
            if inner.func.attr in {"cpu", "cuda"}:
                raise ValueError(
                    "Generated gate forward must not move tensors between devices"
                )
            if inner.func.attr == "to" and inner.args:
                target = inner.args[0]
                if isinstance(target, ast.Constant) and isinstance(target.value, str):
                    raise ValueError(
                        "Generated gate forward must not move tensors between devices"
                    )
    namespace: dict[str, Any] = {"__name__": "_generated_gate"}
    exec(compile(tree, "<generated-gate>", "exec"), namespace)
    gate_cls = namespace.get(class_name)
    if not isinstance(gate_cls, type) or not issubclass(gate_cls, nn.Module):
        raise TypeError(f"Generated source must define nn.Module class {class_name}")
    supported = namespace.get("supported_hyperparameters")
    if not callable(supported) or not isinstance(supported(), set):
        raise TypeError("supported_hyperparameters() must return a set")
    if not callable(getattr(gate_cls, "train_setup", None)) or not callable(
        getattr(gate_cls, "learn", None)
    ):
        raise TypeError("Gate class must define train_setup() and learn()")
    for model_dim, num_experts in shapes:
        gate = gate_cls(model_dim, num_experts).float().eval()
        if any(isinstance(module, (nn.Dropout, nn.modules.batchnorm._BatchNorm)) for module in gate.modules()):
            raise ValueError("Generated gates must be deterministic; dropout/batchnorm are unsupported")
        base = getattr(gate, "base", None)
        if require_base and (
            not isinstance(base, nn.Linear)
            or tuple(base.weight.shape) != (num_experts, model_dim)
        ):
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
            flat = output.detach().float()
            if flat.min().item() >= 0.0 and torch.allclose(
                flat.sum(dim=-1),
                torch.ones_like(flat.sum(dim=-1)),
                atol=1e-3,
            ):
                raise ValueError(
                    "Generated gate must return raw router logits; the host applies "
                    "softmax/sigmoid, so returning probabilities is invalid"
                )
        if require_base:
            # Reject degenerate proposals: the architecture must add trainable
            # parameters that participate in forward. Connectivity (grad is not
            # None), not magnitude: a silent branch (zero-initialised last
            # layer) has zero-valued but connected gradients and is the
            # intended pattern, while a bare linear copy of the base adds none.
            probe = gate_cls(model_dim, num_experts).float()
            output = probe(torch.randn(2, model_dim))
            if not isinstance(output, torch.Tensor) or not output.requires_grad:
                raise ValueError("Gate output does not depend on any trainable parameter")
            output.sum().backward()
            connected = any(
                parameter.grad is not None
                for name, parameter in probe.named_parameters()
                if not name.startswith("base.")
            )
            if not connected:
                raise ValueError(
                    "Gate must add trainable parameters beyond self.base "
                    "(e.g. a residual branch); a plain linear copy is not a new architecture"
                )


def _generate_gate(
    chat_bot: Any, shapes: list[tuple[int, int]], attempts: int,
    max_new_tokens: int, artifact_dir: Path, *,
    reference_source: str = "",
    reference_accuracy: float | None = None,
    goal_accuracy: float | None = None,
    dataset: str | None = None,
    seen_hashes: set[str] | None = None,
    max_attempts_per_duplicate: int = 3,
    inherit_reference: bool = False,
    write_artifacts: bool = True,
    prompt_out: list[str] | None = None,
    prompt_override: str | None = None,
    variant_key: str = "reference_best",
    variant_split: str = "test",
) -> list[str]:
    """Ask the LLM for replacement gate sources; HARVEST every valid distinct one.

    Valid proposals are the scarce resource, so nothing usable is discarded:
    all valid, structurally distinct gates across the attempts are returned
    (order preserved). The first is meant for the current candidate; the rest
    feed the outer loop's proposal pool.

    Stop rule: while nothing valid has been found, every attempt is used
    (retry-until-valid). Once at least one gate is harvested, the first dry
    attempt (no new valid gate) ends the round — productive streaks are
    drained, polishing is not chased.

    Its base weight is copied from the native router, so the replaced model
    stays bit-identical at step zero before training begins.
    """
    prompt = prompt_override if prompt_override is not None else gate_proposal_prompt(
        shapes,
        reference_source=reference_source,
        reference_accuracy=reference_accuracy,
        goal_accuracy=goal_accuracy,
        dataset=dataset,
        inherit_reference=inherit_reference,
        variant_key=variant_key,
        variant_split=variant_split,
    )
    seen = seen_hashes if seen_hashes is not None else set()
    if prompt_out is not None:
        prompt_out.append(prompt)
    if write_artifacts:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "proposal_prompt.txt").write_text(prompt, encoding="utf-8")
    previous_error = ""
    duplicate_attempts = 0
    harvested: list[str] = []
    for attempt in range(1, attempts + 1):
        current_prompt = prompt
        if previous_error:
            current_prompt += f"\nThe previous proposal failed validation: {previous_error}\nReturn a corrected implementation."
        with gate_prompt_scope(chat_bot):
            _, _, _, raw = chat_bot.chat(
                current_prompt, engineer_prompt=False, max_new_tokens=max_new_tokens,
            )
        if write_artifacts:
            (artifact_dir / f"generation_attempt_{attempt}.txt").write_text(raw, encoding="utf-8")
        errors: list[str] = []
        new_this_attempt = 0
        # A single generation can yield several parseable strings; every valid
        # distinct one is harvested, not just the first.
        for candidate in _gate_candidates(raw):
            for source in _gate_candidate_variants(candidate):
                try:
                    source = _normalize_gate_class(source)
                    source = _ensure_lemur_interface(source)
                    _validate_gate_source(source, shapes)
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                if is_duplicate_gate(source, seen):
                    errors.append("structurally identical to an earlier gate candidate")
                    continue
                seen.add(structural_hash(source))
                harvested.append(source)
                new_this_attempt += 1
        if harvested and new_this_attempt == 0:
            break  # productive streak drained; stop polishing
        if not harvested:
            # Duplicate-stall guard only applies while nothing valid exists.
            if any("structurally identical" in e for e in errors):
                duplicate_attempts += 1
                if duplicate_attempts >= max_attempts_per_duplicate:
                    raise RuntimeError(
                        "LLM kept re-emitting a duplicate gate architecture; "
                        "cannot build a distinct candidate"
                    )
        previous_error = "; ".join(errors) or "No usable <gate> source was found"
    if not harvested:
        raise RuntimeError(f"LLM did not generate a valid gate after {attempts} attempts: {previous_error}")
    if write_artifacts:
        (artifact_dir / "gate.py").write_text(harvested[0].rstrip() + "\n", encoding="utf-8")
    if len(harvested) > 1:
        print(
            f"[GATE SEARCH] proposal round harvested {len(harvested)} valid "
            "distinct gates (pool)"
        )
    return harvested
