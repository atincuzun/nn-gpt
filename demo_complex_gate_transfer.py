"""Demonstrate complex gate patterns that do and do not preserve router bits."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_gate_only.morphism import initialize_gate_from_projection


class NativeDeepSeekGate(nn.Module):
    """Minimal simulation of DeepSeek's bias-free router projection."""

    def __init__(self, model_dim: int, num_experts: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, model_dim, dtype=torch.bfloat16)
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight.float())


class GatedResidualBlock(nn.Module):
    """LayerNorm + nonlinear gated MLP + residual connection."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.expand = nn.Linear(hidden_dim, hidden_dim * 2)
        self.contract = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.expand(self.norm(x)).chunk(2, dim=-1)
        update = self.contract(F.silu(left) * torch.sigmoid(right))
        return x + update


class ComplexResidualGate(nn.Module):
    """Copied native path plus an arbitrary zero-output correction network."""

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        hidden_dim: int = 128,
        blocks: int = 3,
    ) -> None:
        super().__init__()
        # This path MUST remain unchanged for bit-exact native behavior.
        self.base = nn.Linear(model_dim, num_experts, bias=False)

        # These operations are allowed because they live only in the parallel
        # correction path: normalization, parallel projections, nonlinear
        # gating, multiplication, residual blocks, and another normalization.
        self.input_norm = nn.LayerNorm(model_dim)
        self.left_projection = nn.Linear(model_dim, hidden_dim)
        self.right_projection = nn.Linear(model_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            GatedResidualBlock(hidden_dim) for _ in range(blocks)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.residual_out = nn.Linear(hidden_dim, num_experts, bias=True)

        # Only the final residual projection is zero. The deep branch therefore
        # contributes exactly zero at startup, while it can wake up in training.
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.base.weight.dtype)
        native_logits = self.base(x)

        normalized = self.input_norm(x)
        left = F.silu(self.left_projection(normalized))
        right = torch.sigmoid(self.right_projection(normalized))
        correction = left * right
        for block in self.blocks:
            correction = block(correction)
        correction = self.residual_out(self.output_norm(correction))

        # Calculate the unchanged native path directly, then add exact zeros.
        return native_logits + correction


class NonExactSerialGate(nn.Module):
    """Counterexample: normalization changes the copied path itself."""

    def __init__(self, model_dim: int, num_experts: int) -> None:
        super().__init__()
        self.base = nn.Linear(model_dim, num_experts, bias=False)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.base.weight.dtype)
        return self.base(self.norm(x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dim", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    native = NativeDeepSeekGate(args.model_dim, args.num_experts).eval()
    complex_gate = ComplexResidualGate(
        args.model_dim, args.num_experts, args.hidden_dim, args.blocks
    ).eval()
    non_exact_gate = NonExactSerialGate(args.model_dim, args.num_experts).eval()

    # This copies native.weight into complex_gate.base.weight.
    metrics = initialize_gate_from_projection(complex_gate, native.weight)
    initialize_gate_from_projection(non_exact_gate, native.weight)
    inputs = torch.randn(args.tokens, args.model_dim, dtype=torch.bfloat16)

    with torch.no_grad():
        native_logits = native(inputs)
        complex_logits = complex_gate(inputs)
        non_exact_logits = non_exact_gate(inputs)
        native_topk = native_logits.topk(args.top_k, dim=-1).indices
        complex_topk = complex_logits.topk(args.top_k, dim=-1).indices

    logits_bit_exact = torch.equal(native_logits, complex_logits)
    topk_bit_exact = torch.equal(native_topk, complex_topk)
    max_abs_error = (native_logits - complex_logits).abs().max().item()
    serial_bit_exact = torch.equal(native_logits, non_exact_logits)

    print(f"transfer: {metrics['initialization']}")
    print(f"complex correction blocks: {args.blocks}")
    print(f"logits bit exact: {logits_bit_exact}")
    print(f"top-k bit exact: {topk_bit_exact}")
    print(f"max absolute error: {max_abs_error}")
    print(f"LayerNorm on copied serial path bit exact: {serial_bit_exact}")

    # Show the real training behavior of a zero-final correction network.
    complex_gate.train()
    complex_gate.base.weight.requires_grad_(False)
    trainable = [parameter for parameter in complex_gate.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    target = torch.randn_like(complex_logits)
    for step in range(1, 3):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(complex_gate(inputs), target)
        loss.backward()
        inner_gradient = complex_gate.left_projection.weight.grad
        output_gradient = complex_gate.residual_out.weight.grad
        print(
            f"step {step} gradients: inner={inner_gradient.norm().item():.8f}, "
            f"output={output_gradient.norm().item():.8f}"
        )
        optimizer.step()

    if not logits_bit_exact or not topk_bit_exact:
        raise SystemExit("FAIL: transferred gate is not bit exact")
    if serial_bit_exact:
        raise SystemExit("FAIL: counterexample unexpectedly stayed bit exact")
    print("PASS")


if __name__ == "__main__":
    main()
