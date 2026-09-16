"""Fresh-weight nonlinear router; use --gate-source with --gate-random-init."""

import torch
import torch.nn as nn


class LLMGeneratedGate(nn.Module):
    def __init__(self, model_dim: int, num_experts: int):
        super().__init__()
        width = max(16, model_dim // 16)
        self.input_norm = nn.LayerNorm(model_dim)
        self.value = nn.Linear(model_dim, width)
        self.modulation = nn.Linear(model_dim, width)
        self.feature_norm = nn.LayerNorm(width)
        self.refine = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.activation = nn.SiLU()
        self.output = nn.Linear(width, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(x)
        features = self.value(normalized) * self.activation(self.modulation(normalized))
        features = features + self.refine(self.feature_norm(features))
        return self.output(self.activation(features))
