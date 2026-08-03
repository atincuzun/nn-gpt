from __future__ import annotations

import math
from typing import Callable, Dict

import torch
import torch.nn as nn

from .morphism import ExactResidualMlpGate, SvdSignedPairGate


class LinearGate(nn.Module):
    """Single trainable linear expert scorer."""

    def __init__(self, model_dim: int, num_experts: int, *, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(model_dim, num_experts, bias=bias)
        nn.init.normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LowRankGate(nn.Module):
    """Low-rank gate useful when only a small routing adapter should train."""

    def __init__(self, model_dim: int, num_experts: int, *, rank: int | None = None) -> None:
        super().__init__()
        rank = rank or max(4, min(128, model_dim // 8, num_experts * 2))
        self.down = nn.Linear(model_dim, rank, bias=False)
        self.up = nn.Linear(rank, num_experts, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(model_dim))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(torch.tanh(self.down(x)))


class MlpGate(nn.Module):
    """Two-layer generated gate with nonlinear token-dependent scoring."""

    def __init__(self, model_dim: int, num_experts: int, *, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(32, min(model_dim, num_experts * 8))
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts, bias=False),
        )
        nn.init.normal_(self.net[0].weight, std=0.02)
        nn.init.normal_(self.net[2].weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMlpGate(nn.Module):
    """Residual-style gate: linear base plus a small trainable residual policy."""

    def __init__(self, model_dim: int, num_experts: int, *, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(32, min(model_dim, num_experts * 8))
        self.base = nn.Linear(model_dim, num_experts, bias=False)
        self.residual = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts, bias=False),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.base.weight, std=0.02)
        nn.init.normal_(self.residual[1].weight, std=0.02)
        nn.init.zeros_(self.residual[3].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.residual_scale * self.residual(x)


class FourierFeatureGate(nn.Module):
    """Random Fourier feature gate with trainable readout."""

    def __init__(self, model_dim: int, num_experts: int, *, features: int | None = None) -> None:
        super().__init__()
        features = features or max(16, min(256, num_experts * 8))
        random_basis = torch.randn(model_dim, features) / math.sqrt(model_dim)
        self.register_buffer("basis", random_basis)
        self.readout = nn.Linear(features * 2, num_experts, bias=False)
        nn.init.normal_(self.readout.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = x.to(self.basis.dtype) @ self.basis
        features = torch.cat((torch.sin(projected), torch.cos(projected)), dim=-1)
        return self.readout(features.to(self.readout.weight.dtype)).to(x.dtype)


GateFactory = Callable[[int, int], nn.Module]


GATE_FACTORIES: Dict[str, GateFactory] = {
    "exact_residual_mlp": lambda model_dim, num_experts: ExactResidualMlpGate(
        model_dim, num_experts
    ),
    "linear": lambda model_dim, num_experts: LinearGate(model_dim, num_experts),
    "low_rank": lambda model_dim, num_experts: LowRankGate(model_dim, num_experts),
    "mlp": lambda model_dim, num_experts: MlpGate(model_dim, num_experts),
    "residual_mlp": lambda model_dim, num_experts: ResidualMlpGate(model_dim, num_experts),
    "fourier": lambda model_dim, num_experts: FourierFeatureGate(model_dim, num_experts),
    "svd_signed_pair_gelu": lambda model_dim, num_experts: SvdSignedPairGate(
        model_dim, num_experts, activation="gelu"
    ),
    "svd_signed_pair_silu": lambda model_dim, num_experts: SvdSignedPairGate(
        model_dim, num_experts, activation="silu"
    ),
}


def build_gate(name: str, model_dim: int, num_experts: int) -> nn.Module:
    try:
        return GATE_FACTORIES[name](model_dim, num_experts)
    except KeyError as exc:
        valid = ", ".join(sorted(GATE_FACTORIES))
        raise ValueError(f"Unknown gate {name!r}. Valid gates: {valid}") from exc


def compile_gate_from_string(code_string: str, class_name: str = "LLMGeneratedGate") -> type:
    """Compile a PyTorch ``nn.Module`` class from a Python code string.

    Parameters
    ----------
    code_string:
        Raw Python source. Must contain a class definition of ``class_name``.
    class_name:
        Name of the class to extract from the compiled module.

    Returns
    -------
    type
        The compiled class (not an instance).  Call it with
        ``(model_dim, num_experts)`` to instantiate a gate.

    Example
    -------
    >>> code = '''
    ... class LLMGeneratedGate(nn.Module):
    ...     def __init__(self, model_dim, num_experts):
    ...         super().__init__()
    ...         self.proj = nn.Linear(model_dim, num_experts)
    ...     def forward(self, x):
    ...         return self.proj(x)
    ... '''
    >>> Gate = compile_gate_from_string(code)
    >>> gate = Gate(64, 8)
    """
    namespace: dict = {}
    exec(compile(code_string, "<llm_generated>", "exec"), namespace)
    gate_class = namespace.get(class_name)
    if gate_class is None:
        raise ValueError(f"Code string must define a class named {class_name!r}")
    return gate_class
