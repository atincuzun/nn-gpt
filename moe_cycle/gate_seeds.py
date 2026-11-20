"""Cold-start bootstrap seeds: built-in compliant gate architectures.

The proposer could not escape plain-linear proposals even with 100
error-feedback retries (run 20260920_015152: 100/100 attempts linear).
These seeds are installed directly for the first outer candidates so the
store accumulates scored, valid, reference architectures before the
proposer is trusted.

Every seed satisfies the same validator as LLM proposals: ``self.base`` plus
extra capacity that is silent (zero output) at initialization, deterministic
forward, torch-only imports, connected gradients. The 50 mechanisms span
activation MLPs, bottlenecks, gated linear units, multi-branch mixtures,
feature gating, low-rank/factorized projections, local convolutions and
pooling, normalization, polynomial/Fourier features, grouped experts, and
per-expert scale/shift.

The catalog is ordered round-robin across mechanism families, so any prefix
(including the default first three) is architecturally diverse. Selection is
deterministic: seed index i uses ``SEED_GATES[i % len(SEED_GATES)]``.
"""

from __future__ import annotations

import textwrap


def _src(text: str) -> str:
    """Dedent an inline source block and normalise the trailing newline."""
    return textwrap.dedent(text).strip("\n") + "\n"


SEED_GATES: tuple[tuple[str, str], ...] = (
    (
        "01-additive-mlp-gelu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch = nn.Sequential(
                        nn.Linear(model_dim, 2 * model_dim),
                        nn.GELU(),
                        nn.Linear(2 * model_dim, num_experts, bias=False),
                    )
                    nn.init.zeros_(self.branch[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch(x)
"""),
    ),
    (
        "02-bottleneck-quarter-gelu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    hidden = max(1, model_dim // 4)
                    self.down = nn.Linear(model_dim, hidden)
                    self.act = nn.GELU()
                    self.up = nn.Linear(hidden, num_experts, bias=False)
                    nn.init.zeros_(self.up.weight)

                def forward(self, x):
                    return self.base(x) + self.up(self.act(self.down(x)))
"""),
    ),
    (
        "03-geglu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.gate = nn.Linear(model_dim, 2 * model_dim)
                    self.value = nn.Linear(model_dim, 2 * model_dim)
                    self.act = nn.GELU()
                    self.out = nn.Linear(2 * model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.out.weight)

                def forward(self, x):
                    return self.base(x) + self.out(self.act(self.gate(x)) * self.value(x))
"""),
    ),
    (
        "04-two-branch-sum",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch_a = nn.Sequential(nn.Linear(model_dim, 2 * model_dim), nn.GELU(), nn.Linear(2 * model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_a[-1].weight)
                    self.branch_b = nn.Sequential(nn.Linear(model_dim, 2 * model_dim), nn.SiLU(), nn.Linear(2 * model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_b[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch_a(x) + self.branch_b(x)
"""),
    ),
    (
        "05-multiplicative-modulation",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.modulation = nn.Sequential(
                        nn.Linear(model_dim, num_experts),
                        nn.Tanh(),
                    )
                    nn.init.zeros_(self.modulation[0].weight)
                    nn.init.zeros_(self.modulation[0].bias)

                def forward(self, x):
                    return self.base(x) * (1.0 + self.modulation(x))
"""),
    ),
    (
        "06-low-rank-gelu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.down = nn.Linear(model_dim, max(1, model_dim // 4))
                    self.act = nn.GELU()
                    self.up = nn.Linear(max(1, model_dim // 4), num_experts, bias=False)
                    nn.init.zeros_(self.up.weight)

                def forward(self, x):
                    return self.base(x) + self.up(self.act(self.down(x)))
"""),
    ),
    (
        "07-conv1d-kernel3",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
                    self.act = nn.Tanh()
                    self.out = nn.Linear(model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.out.weight)

                def forward(self, x):
                    shape = x.shape
                    h = self.conv(x.reshape(-1, 1, shape[-1])).reshape(shape)
                    return self.base(x) + self.out(self.act(h))
"""),
    ),
    (
        "08-layernorm-mlp",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.norm = nn.LayerNorm(model_dim)
                    self.branch = nn.Sequential(
                        nn.Linear(model_dim, 2 * model_dim),
                        nn.GELU(),
                        nn.Linear(2 * model_dim, num_experts, bias=False),
                    )
                    nn.init.zeros_(self.branch[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch(self.norm(x))
"""),
    ),
    (
        "09-grouped-experts",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.group_dim = model_dim // 2
                    self.left = nn.Linear(self.group_dim, num_experts // 2, bias=False)
                    self.right = nn.Linear(model_dim - self.group_dim, num_experts // 2, bias=False)
                    nn.init.zeros_(self.left.weight)
                    nn.init.zeros_(self.right.weight)

                def forward(self, x):
                    left = self.left(x[..., :self.group_dim])
                    right = self.right(x[..., self.group_dim:])
                    return self.base(x) + torch.cat([left, right], dim=-1)
"""),
    ),
    (
        "10-additive-mlp-silu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch = nn.Sequential(
                        nn.Linear(model_dim, 2 * model_dim),
                        nn.SiLU(),
                        nn.Linear(2 * model_dim, num_experts, bias=False),
                    )
                    nn.init.zeros_(self.branch[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch(x)
"""),
    ),
    (
        "11-bottleneck-half-silu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    hidden = max(1, model_dim // 2)
                    self.down = nn.Linear(model_dim, hidden)
                    self.act = nn.SiLU()
                    self.up = nn.Linear(hidden, num_experts, bias=False)
                    nn.init.zeros_(self.up.weight)

                def forward(self, x):
                    return self.base(x) + self.up(self.act(self.down(x)))
"""),
    ),
    (
        "12-swiglu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.gate = nn.Linear(model_dim, 2 * model_dim)
                    self.value = nn.Linear(model_dim, 2 * model_dim)
                    self.act = nn.SiLU()
                    self.out = nn.Linear(2 * model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.out.weight)

                def forward(self, x):
                    return self.base(x) + self.out(self.act(self.gate(x)) * self.value(x))
"""),
    ),
    (
        "13-three-branch-sum",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch_a = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_a[-1].weight)
                    self.branch_b = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_b[-1].weight)
                    self.branch_c = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Tanh(), nn.Linear(model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_c[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch_a(x) + self.branch_b(x) + self.branch_c(x)
"""),
    ),
    (
        "14-sigmoid-gated-branch",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.gate = nn.Linear(model_dim, num_experts)
                    self.branch = nn.Linear(model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.branch.weight)

                def forward(self, x):
                    return self.base(x) + torch.sigmoid(self.gate(x)) * self.branch(x)
"""),
    ),
    (
        "15-low-rank-silu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.down = nn.Linear(model_dim, max(1, model_dim // 2))
                    self.act = nn.SiLU()
                    self.up = nn.Linear(max(1, model_dim // 2), num_experts, bias=False)
                    nn.init.zeros_(self.up.weight)

                def forward(self, x):
                    return self.base(x) + self.up(self.act(self.down(x)))
"""),
    ),
    (
        "16-conv1d-kernel5",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
                    self.act = nn.Tanh()
                    self.out = nn.Linear(model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.out.weight)

                def forward(self, x):
                    shape = x.shape
                    h = self.conv(x.reshape(-1, 1, shape[-1])).reshape(shape)
                    return self.base(x) + self.out(self.act(h))
"""),
    ),
    (
        "17-rmsnorm-mlp",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch = nn.Sequential(
                        nn.Linear(model_dim, 2 * model_dim),
                        nn.GELU(),
                        nn.Linear(2 * model_dim, num_experts, bias=False),
                    )
                    nn.init.zeros_(self.branch[-1].weight)

                def forward(self, x):
                    h = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                    return self.base(x) + self.branch(h)
"""),
    ),
    (
        "18-multi-head-concat",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.heads = nn.ModuleList([
                        nn.Linear(model_dim, num_experts // 4, bias=False) for _ in range(4)
                    ])
                    for head in self.heads:
                        nn.init.zeros_(head.weight)

                def forward(self, x):
                    outputs = [head(x) for head in self.heads]
                    return self.base(x) + torch.cat(outputs, dim=-1)
"""),
    ),
    (
        "19-additive-mlp-relu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch = nn.Sequential(
                        nn.Linear(model_dim, 2 * model_dim),
                        nn.ReLU(),
                        nn.Linear(2 * model_dim, num_experts, bias=False),
                    )
                    nn.init.zeros_(self.branch[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch(x)
"""),
    ),
    (
        "20-bottleneck-expand-tanh",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    hidden = 4 * model_dim
                    self.down = nn.Linear(model_dim, hidden)
                    self.act = nn.Tanh()
                    self.up = nn.Linear(hidden, num_experts, bias=False)
                    nn.init.zeros_(self.up.weight)

                def forward(self, x):
                    return self.base(x) + self.up(self.act(self.down(x)))
"""),
    ),
    (
        "21-reglu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.gate = nn.Linear(model_dim, 2 * model_dim)
                    self.value = nn.Linear(model_dim, 2 * model_dim)
                    self.act = nn.ReLU()
                    self.out = nn.Linear(2 * model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.out.weight)

                def forward(self, x):
                    return self.base(x) + self.out(self.act(self.gate(x)) * self.value(x))
"""),
    ),
    (
        "22-four-branch-sum",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch_a = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_a[-1].weight)
                    self.branch_b = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_b[-1].weight)
                    self.branch_c = nn.Sequential(nn.Linear(model_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_c[-1].weight)
                    self.branch_d = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Tanh(), nn.Linear(model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_d[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch_a(x) + self.branch_b(x) + self.branch_c(x) + self.branch_d(x)
"""),
    ),
    (
        "23-tanh-gated-branch",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.gate = nn.Linear(model_dim, num_experts)
                    self.branch = nn.Linear(model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.branch.weight)

                def forward(self, x):
                    return self.base(x) + torch.tanh(self.gate(x)) * self.branch(x)
"""),
    ),
    (
        "24-two-low-rank-sum",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.down_a = nn.Linear(model_dim, 2 * num_experts)
                    self.down_b = nn.Linear(model_dim, num_experts)
                    self.act = nn.GELU()
                    self.up_a = nn.Linear(2 * num_experts, num_experts, bias=False)
                    self.up_b = nn.Linear(num_experts, num_experts, bias=False)
                    nn.init.zeros_(self.up_a.weight)
                    nn.init.zeros_(self.up_b.weight)

                def forward(self, x):
                    return self.base(x) + self.up_a(self.act(self.down_a(x))) + self.up_b(self.act(self.down_b(x)))
"""),
    ),
    (
        "25-conv1d-multichannel",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.num_experts = num_experts
                    self.conv = nn.Conv1d(1, 4, kernel_size=3, padding=1)
                    self.act = nn.GELU()
                    self.out = nn.Linear(4 * model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.out.weight)

                def forward(self, x):
                    shape = x.shape
                    h = self.conv(x.reshape(-1, 1, shape[-1])).reshape(-1, 4 * shape[-1])
                    out = self.out(self.act(h)).reshape(*shape[:-1], self.num_experts)
                    return self.base(x) + out
"""),
    ),
    (
        "26-polynomial-branch",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.proj = nn.Linear(model_dim, 2 * model_dim)
                    self.out = nn.Linear(2 * model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.out.weight)

                def forward(self, x):
                    h = self.proj(x)
                    return self.base(x) + self.out(h + h * h)
"""),
    ),
    (
        "27-two-level-gate",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.first = nn.Linear(model_dim, num_experts // 2)
                    self.second = nn.Linear(model_dim, num_experts // 2)
                    self.out = nn.Linear(num_experts // 2, num_experts, bias=False)
                    nn.init.zeros_(self.out.weight)

                def forward(self, x):
                    h = self.first(x)
                    g = torch.sigmoid(self.second(x))
                    return self.base(x) + self.out(h * g)
"""),
    ),
    (
        "28-additive-mlp-tanh",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch = nn.Sequential(
                        nn.Linear(model_dim, 2 * model_dim),
                        nn.Tanh(),
                        nn.Linear(2 * model_dim, num_experts, bias=False),
                    )
                    nn.init.zeros_(self.branch[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch(x)
"""),
    ),
    (
        "29-two-bottleneck-stack",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.down = nn.Linear(model_dim, model_dim // 2)
                    self.act = nn.GELU()
                    self.mid = nn.Linear(model_dim // 2, model_dim // 4)
                    self.up = nn.Linear(model_dim // 4, num_experts, bias=False)
                    nn.init.zeros_(self.up.weight)

                def forward(self, x):
                    h = self.act(self.down(x))
                    h = self.act(self.mid(h))
                    return self.base(x) + self.up(h)
"""),
    ),
    (
        "30-bilinear-glu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.gate = nn.Linear(model_dim, 2 * model_dim)
                    self.value = nn.Linear(model_dim, 2 * model_dim)
                    self.out = nn.Linear(2 * model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.out.weight)

                def forward(self, x):
                    return self.base(x) + self.out(self.gate(x) * self.value(x))
"""),
    ),
    (
        "31-two-branch-softmax-mix",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch_a = nn.Sequential(nn.Linear(model_dim, 2 * model_dim), nn.GELU(), nn.Linear(2 * model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_a[-1].weight)
                    self.branch_b = nn.Sequential(nn.Linear(model_dim, 2 * model_dim), nn.SiLU(), nn.Linear(2 * model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_b[-1].weight)
                    self.mix = nn.Linear(model_dim, 2)

                def forward(self, x):
                    weights = torch.softmax(self.mix(x), dim=-1)
                    return self.base(x) + weights[..., 0:1] * self.branch_a(x) + weights[..., 1:2] * self.branch_b(x)
"""),
    ),
    (
        "32-film-scale-shift",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.scale = nn.Linear(model_dim, num_experts)
                    self.shift = nn.Linear(model_dim, num_experts)
                    self.branch = nn.Linear(model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.branch.weight)
                    nn.init.zeros_(self.shift.weight)
                    nn.init.zeros_(self.shift.bias)

                def forward(self, x):
                    return self.base(x) + self.scale(x) * self.branch(x) + self.shift(x)
"""),
    ),
    (
        "33-low-rank-diagonal",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.down = nn.Linear(model_dim, model_dim // 2)
                    self.diag = nn.Linear(model_dim // 2, model_dim // 2, bias=False)
                    self.act = nn.GELU()
                    self.up = nn.Linear(model_dim // 2, num_experts, bias=False)
                    nn.init.zeros_(self.up.weight)

                def forward(self, x):
                    h = self.act(self.down(x))
                    return self.base(x) + self.up(h * torch.sigmoid(self.diag(h)))
"""),
    ),
    (
        "34-avgpool-branch",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
                    self.act = nn.GELU()
                    self.out = nn.Linear(model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.out.weight)

                def forward(self, x):
                    shape = x.shape
                    h = self.pool(x.reshape(-1, 1, shape[-1])).reshape(shape)
                    return self.base(x) + self.out(self.act(h))
"""),
    ),
    (
        "35-fourier-features",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.freq = nn.Linear(model_dim, 2 * model_dim)
                    self.out = nn.Linear(4 * model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.out.weight)

                def forward(self, x):
                    h = self.freq(x)
                    features = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)
                    return self.base(x) + self.out(features)
"""),
    ),
    (
        "36-per-expert-scale",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.scale = nn.Linear(model_dim, num_experts)
                    nn.init.zeros_(self.scale.weight)
                    nn.init.zeros_(self.scale.bias)

                def forward(self, x):
                    return self.base(x) * (1.0 + self.scale(x))
"""),
    ),
    (
        "37-additive-mlp-elu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch = nn.Sequential(
                        nn.Linear(model_dim, 2 * model_dim),
                        nn.ELU(),
                        nn.Linear(2 * model_dim, num_experts, bias=False),
                    )
                    nn.init.zeros_(self.branch[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch(x)
"""),
    ),
    (
        "38-bottleneck-layernorm",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.norm = nn.LayerNorm(model_dim)
                    self.down = nn.Linear(model_dim, model_dim // 2)
                    self.act = nn.SiLU()
                    self.up = nn.Linear(model_dim // 2, num_experts, bias=False)
                    nn.init.zeros_(self.up.weight)

                def forward(self, x):
                    return self.base(x) + self.up(self.act(self.down(self.norm(x))))
"""),
    ),
    (
        "39-glu-hidden",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    hidden = 2 * num_experts
                    self.proj = nn.Linear(model_dim, 2 * hidden)
                    self.out = nn.Linear(hidden, num_experts, bias=False)
                    nn.init.zeros_(self.out.weight)

                def forward(self, x):
                    gate, value = self.proj(x).chunk(2, dim=-1)
                    return self.base(x) + self.out(gate * torch.sigmoid(value))
"""),
    ),
    (
        "40-three-branch-softmax-mix",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch_a = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_a[-1].weight)
                    self.branch_b = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_b[-1].weight)
                    self.branch_c = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Tanh(), nn.Linear(model_dim, num_experts, bias=False))
                    nn.init.zeros_(self.branch_c[-1].weight)
                    self.mix = nn.Linear(model_dim, 3)

                def forward(self, x):
                    weights = torch.softmax(self.mix(x), dim=-1)
                    return self.base(x) + weights[..., 0:1] * self.branch_a(x) + weights[..., 1:2] * self.branch_b(x) + weights[..., 2:3] * self.branch_c(x)
"""),
    ),
    (
        "41-highway-tanh",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.gate = nn.Linear(model_dim, num_experts)
                    self.branch = nn.Linear(model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.branch.weight)

                def forward(self, x):
                    return self.base(x) + torch.sigmoid(self.gate(x)) * torch.tanh(self.branch(x))
"""),
    ),
    (
        "42-grouped-low-rank",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    groups = 2
                    per_group = num_experts // groups
                    hidden = max(1, model_dim // 4)
                    self.downs = nn.ModuleList([nn.Linear(model_dim, hidden) for _ in range(groups)])
                    self.ups = nn.ModuleList([nn.Linear(hidden, per_group, bias=False) for _ in range(groups)])
                    self.act = nn.GELU()
                    for up in self.ups:
                        nn.init.zeros_(up.weight)

                def forward(self, x):
                    parts = [up(self.act(down(x))) for down, up in zip(self.downs, self.ups)]
                    return self.base(x) + torch.cat(parts, dim=-1)
"""),
    ),
    (
        "43-chebyshev-features",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.proj = nn.Linear(model_dim, 2 * model_dim)
                    self.out = nn.Linear(4 * model_dim, num_experts, bias=False)
                    nn.init.zeros_(self.out.weight)

                def forward(self, x):
                    h = self.proj(x)
                    features = torch.cat([h, 2 * h * h - 1], dim=-1)
                    return self.base(x) + self.out(features)
"""),
    ),
    (
        "44-per-expert-shift",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.shift = nn.Sequential(
                        nn.Linear(model_dim, 2 * num_experts),
                        nn.Tanh(),
                        nn.Linear(2 * num_experts, num_experts, bias=False),
                    )
                    nn.init.zeros_(self.shift[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.shift(x)
"""),
    ),
    (
        "45-additive-mlp-selu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch = nn.Sequential(
                        nn.Linear(model_dim, 2 * model_dim),
                        nn.SELU(),
                        nn.Linear(2 * model_dim, num_experts, bias=False),
                    )
                    nn.init.zeros_(self.branch[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch(x)
"""),
    ),
    (
        "46-tensor-factorized",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.u = nn.Linear(model_dim, model_dim // 4, bias=False)
                    self.v = nn.Linear(model_dim // 4, num_experts, bias=False)
                    self.scale = nn.Parameter(torch.zeros(1))
                    nn.init.zeros_(self.v.weight)

                def forward(self, x):
                    return self.base(x) + self.scale * self.v(torch.tanh(self.u(x)))
"""),
    ),
    (
        "47-additive-mlp-mish",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch = nn.Sequential(
                        nn.Linear(model_dim, 2 * model_dim),
                        nn.Mish(),
                        nn.Linear(2 * model_dim, num_experts, bias=False),
                    )
                    nn.init.zeros_(self.branch[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch(x)
"""),
    ),
    (
        "48-additive-mlp-hardswish",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch = nn.Sequential(
                        nn.Linear(model_dim, 2 * model_dim),
                        nn.Hardswish(),
                        nn.Linear(2 * model_dim, num_experts, bias=False),
                    )
                    nn.init.zeros_(self.branch[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch(x)
"""),
    ),
    (
        "49-additive-mlp-leakyrelu",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch = nn.Sequential(
                        nn.Linear(model_dim, 2 * model_dim),
                        nn.LeakyReLU(0.1),
                        nn.Linear(2 * model_dim, num_experts, bias=False),
                    )
                    nn.init.zeros_(self.branch[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch(x)
"""),
    ),
    (
        "50-additive-mlp-softplus",
        _src("""
            import torch
            import torch.nn as nn

            class LLMGeneratedGate(nn.Module):
                def __init__(self, model_dim: int, num_experts: int):
                    super().__init__()
                    self.base = nn.Linear(model_dim, num_experts, bias=False)
                    self.branch = nn.Sequential(
                        nn.Linear(model_dim, 2 * model_dim),
                        nn.Softplus(),
                        nn.Linear(2 * model_dim, num_experts, bias=False),
                    )
                    nn.init.zeros_(self.branch[-1].weight)

                def forward(self, x):
                    return self.base(x) + self.branch(x)
"""),
    ),
)


def seed_gate_source(index: int) -> str:
    return SEED_GATES[index % len(SEED_GATES)][1]


def seed_gate_name(index: int) -> str:
    return SEED_GATES[index % len(SEED_GATES)][0]
