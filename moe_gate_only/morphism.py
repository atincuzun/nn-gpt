from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ExactResidualMlpGate(nn.Module):
    """Bit-preserving native projection plus a zero-initialized MLP residual."""

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(32, 2 * num_experts)
        self.model_dim = int(model_dim)
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.base = nn.Linear(model_dim, num_experts, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(model_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts, bias=False),
        )
        nn.init.kaiming_uniform_(self.residual[0].weight, a=5**0.5)
        nn.init.zeros_(self.residual[2].weight)
        self.morphism_metrics: dict[str, Any] | None = None

    def initialize_from_projection(
        self,
        weight: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        expected = (self.num_experts, self.model_dim)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"Native router shape {tuple(weight.shape)} does not match {expected}"
            )
        if bias is not None:
            raise ValueError("ExactResidualMlpGate currently requires a bias-free native router")
        with torch.no_grad():
            self.base.weight.copy_(
                weight.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
            )
            self.residual[2].weight.zero_()
        self.morphism_metrics = {
            "initialization": "exact_copied_linear_plus_zero_residual",
            "hidden_dim": self.hidden_dim,
            "source_dtype": str(weight.dtype),
            "parameter_dtype": str(self.base.weight.dtype),
            "bit_exact_projection_expected": True,
            "native_bias": False,
        }
        return dict(self.morphism_metrics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.residual(x)


class SvdSignedPairGate(nn.Module):
    """Nonlinear gate initialized to reproduce a linear router projection.

    For activations satisfying ``activation(z) - activation(-z) == z``, the
    signed-pair construction below represents the native projection exactly in
    real arithmetic while leaving two independent linear layers trainable.
    """

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        *,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        rank = min(model_dim, num_experts)
        self.model_dim = int(model_dim)
        self.num_experts = int(num_experts)
        self.rank = int(rank)
        self.activation_name = activation
        self.input_projection = nn.Linear(model_dim, 2 * rank, bias=False)
        self.output_projection = nn.Linear(2 * rank, num_experts, bias=True)
        if activation == "gelu":
            self.activation = nn.GELU(approximate="none")
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported signed-pair activation: {activation!r}")
        self.morphism_metrics: dict[str, Any] | None = None

    def initialize_from_projection(
        self,
        weight: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Factor ``weight`` and initialize this gate to reproduce ``weight @ x``."""
        expected = (self.num_experts, self.model_dim)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"Native router shape {tuple(weight.shape)} does not match {expected}"
            )
        if not weight.is_floating_point():
            raise TypeError("Native router weight must be floating point")

        source = weight.detach().to(device="cpu", dtype=torch.float64)
        left_vectors, singular_values, right_vectors = torch.linalg.svd(
            source,
            full_matrices=False,
        )
        singular_roots = singular_values.sqrt()
        left = left_vectors * singular_roots.unsqueeze(0)
        right = singular_roots.unsqueeze(1) * right_vectors
        reconstruction = left @ right
        difference = reconstruction - source
        source_norm = float(source.norm().item())
        relative_error = float(difference.norm().item() / max(source_norm, 1e-30))
        max_abs_error = float(difference.abs().max().item())
        tolerance = max(10.0 * self.rank * torch.finfo(torch.float64).eps, 1e-12)
        if relative_error > tolerance:
            raise RuntimeError(
                "SVD router morphism did not reconstruct the native projection: "
                f"relative_error={relative_error} tolerance={tolerance}"
            )

        input_weight = torch.cat((right, -right), dim=0)
        output_weight = torch.cat((left, -left), dim=1)
        with torch.no_grad():
            self.input_projection.weight.copy_(
                input_weight.to(
                    device=self.input_projection.weight.device,
                    dtype=self.input_projection.weight.dtype,
                )
            )
            self.output_projection.weight.copy_(
                output_weight.to(
                    device=self.output_projection.weight.device,
                    dtype=self.output_projection.weight.dtype,
                )
            )
            if bias is None:
                self.output_projection.bias.zero_()
            else:
                if tuple(bias.shape) != (self.num_experts,):
                    raise ValueError(
                        f"Native router bias shape {tuple(bias.shape)} does not match "
                        f"{(self.num_experts,)}"
                    )
                self.output_projection.bias.copy_(
                    bias.to(
                        device=self.output_projection.bias.device,
                        dtype=self.output_projection.bias.dtype,
                    )
                )

        runtime_left = (
            self.output_projection.weight[:, :self.rank].detach().to(
                device="cpu", dtype=torch.float64
            )
        )
        runtime_right = (
            self.input_projection.weight[:self.rank].detach().to(
                device="cpu", dtype=torch.float64
            )
        )
        runtime_difference = runtime_left @ runtime_right - source
        runtime_relative_error = float(
            runtime_difference.norm().item() / max(source_norm, 1e-30)
        )
        runtime_max_abs_error = float(runtime_difference.abs().max().item())

        self.morphism_metrics = {
            "initialization": "svd_signed_pair",
            "activation": self.activation_name,
            "rank": self.rank,
            "hidden_dim": 2 * self.rank,
            "source_dtype": str(weight.dtype),
            "parameter_dtype": str(self.input_projection.weight.dtype),
            "svd_relative_reconstruction_error": relative_error,
            "svd_max_abs_reconstruction_error": max_abs_error,
            "runtime_relative_reconstruction_error": runtime_relative_error,
            "runtime_max_abs_reconstruction_error": runtime_max_abs_error,
            "native_bias": bias is not None,
            "bit_exact_projection_expected": False,
        }
        return dict(self.morphism_metrics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_projection(self.activation(self.input_projection(x)))

    def extra_repr(self) -> str:
        return (
            f"model_dim={self.model_dim}, num_experts={self.num_experts}, "
            f"rank={self.rank}, activation={self.activation_name!r}"
        )
