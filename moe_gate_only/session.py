from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

import torch.nn as nn

from .gates import compile_gate_from_string
from .universal import (
    GateInstall,
    GateSite,
    count_parameters,
    find_moe_gates,
    freeze_except_gates,
    gate_trainable_parameters,
    install_gates,
    restore_gates,
    set_teacher_student_weight,
    trainable_parameter_names,
)


class SessionState(str, Enum):
    READY = "ready"
    REPLACED = "replaced"
    FROZEN = "frozen"
    RESTORED = "restored"


@dataclass(frozen=True)
class GateSessionSummary:
    state: str
    gate_sites: int
    installed_gates: int
    trainable_parameters: int
    total_parameters: int
    trainable_names: tuple[str, ...]


class MoEGateSession:
    """Transactional lifecycle manager for gate-only MoE experiments.

    The model is not wrapped or copied. Discovery and replacement delegate to
    ``moe_gate_only.universal`` so the functional and object APIs have exactly
    the same routing behavior.
    """

    def __init__(self, model: nn.Module, tokenizer: Any = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sites: list[GateSite] = []
        self.installs: list[GateInstall] = []
        self.state = SessionState.READY
        self._requires_grad: dict[str, bool] | None = None
        self._committed = False
        self.gate_source: str | None = None
        self.gate_class_name: str | None = None
        self.gate_factory_name: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        tokenizer_name_or_path: Optional[str] = None,
        trust_remote_code: bool = True,
        model_kwargs: Optional[dict[str, Any]] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> "MoEGateSession":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_options = dict(model_kwargs or {})
        tokenizer_options = dict(tokenizer_kwargs or {})
        model_options.setdefault("trust_remote_code", trust_remote_code)
        tokenizer_options.setdefault("trust_remote_code", trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path or model_name_or_path,
            **tokenizer_options,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_options)
        return cls(model, tokenizer)

    def inspect(self, sample_input: Any = None, *, allow_generic: bool = True) -> list[GateSite]:
        self.sites = find_moe_gates(
            self.model,
            sample_input=sample_input,
            allow_generic=allow_generic,
        )
        return list(self.sites)

    def replace(
        self,
        gate: str | Any,
        *,
        layers: Optional[Iterable[int]] = None,
        sample_input: Any = None,
        verify: bool = True,
        top_k: Optional[int] = None,
        allow_remote_code: bool = False,
        require_hf_native: bool = False,
        allow_generic: bool = True,
        dynamic_discovery: bool = True,
        initialize_from_original: bool = False,
        teacher_student: bool = False,
        student_weight: float = 0.0,
        distillation_temperature: float = 1.0,
    ) -> list[GateInstall]:
        if self.installs:
            raise RuntimeError("Gates are already installed; restore them before replacing again")
        self._snapshot_trainability()
        try:
            self.installs = install_gates(
                self.model,
                gate,
                layers=layers,
                sample_input=sample_input,
                verify=verify,
                top_k=top_k,
                allow_remote_code=allow_remote_code,
                require_hf_native=require_hf_native,
                allow_generic=allow_generic,
                dynamic_discovery=dynamic_discovery,
                initialize_from_original=initialize_from_original,
                teacher_student=teacher_student,
                student_weight=student_weight,
                distillation_temperature=distillation_temperature,
            )
        except Exception:
            self._restore_trainability()
            raise
        self.sites = [install.site for install in self.installs]
        self.gate_factory_name = gate if isinstance(gate, str) else None
        self.state = SessionState.REPLACED
        return list(self.installs)

    def replace_source(
        self,
        source: str,
        *,
        class_name: str = "LLMGeneratedGate",
        **replace_kwargs: Any,
    ) -> list[GateInstall]:
        gate_class = compile_gate_from_string(source, class_name=class_name)
        if not issubclass(gate_class, nn.Module):
            raise TypeError(f"{class_name} must inherit from torch.nn.Module")
        installs = self.replace(gate_class, **replace_kwargs)
        self.gate_source = source
        self.gate_class_name = class_name
        self.gate_factory_name = None
        return installs

    def freeze_except_gates(self) -> None:
        if not self.installs:
            raise RuntimeError("Replace gates before freezing the model")
        freeze_except_gates(self.model, self.installs)
        self.state = SessionState.FROZEN

    def gate_parameters(self) -> list[nn.Parameter]:
        return gate_trainable_parameters(self.installs)

    def set_student_weight(self, value: float) -> None:
        if not self.installs:
            raise RuntimeError("Install teacher-student gates before changing their weight")
        set_teacher_student_weight(self.installs, value)

    def student_weight(self) -> float | None:
        weights = [
            float(install.new_gate.student_weight)
            for install in self.installs
            if hasattr(install.new_gate, "student_weight")
        ]
        if not weights:
            return None
        if any(weight != weights[0] for weight in weights[1:]):
            raise RuntimeError("Installed teacher-student gates have inconsistent weights")
        return weights[0]

    def summary(self) -> GateSessionSummary:
        trainable, total = count_parameters(self.model)
        return GateSessionSummary(
            state=self.state.value,
            gate_sites=len(self.sites),
            installed_gates=len(self.installs),
            trainable_parameters=trainable,
            total_parameters=total,
            trainable_names=tuple(trainable_parameter_names(self.model)),
        )

    def save(self, path: str | Path, *, optimizer: Any = None) -> Path:
        from .checkpoint import save_gate_checkpoint

        return save_gate_checkpoint(self, path, optimizer=optimizer)

    def load_weights(self, path: str | Path, *, optimizer: Any = None) -> dict[str, Any]:
        from .checkpoint import load_gate_checkpoint

        return load_gate_checkpoint(self, path, optimizer=optimizer)

    def commit(self) -> None:
        """Keep replacements installed when leaving a context manager."""
        if not self.installs:
            raise RuntimeError("There are no installed gates to commit")
        self._committed = True

    def restore(self) -> None:
        if self.installs:
            restore_gates(self.installs)
            self.installs.clear()
        self._restore_trainability()
        self._committed = False
        self.state = SessionState.RESTORED

    def _snapshot_trainability(self) -> None:
        if self._requires_grad is None:
            self._requires_grad = {
                name: parameter.requires_grad
                for name, parameter in self.model.named_parameters()
            }

    def _restore_trainability(self) -> None:
        if self._requires_grad is None:
            return
        for name, parameter in self.model.named_parameters():
            if name in self._requires_grad:
                parameter.requires_grad_(self._requires_grad[name])
        self._requires_grad = None

    def __enter__(self) -> "MoEGateSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if not self._committed:
            self.restore()
        return False
