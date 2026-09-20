"""Cold-start bootstrap seeds: built-in compliant gate architectures.

The proposer (V2-Lite 4-bit) could not escape plain-linear proposals even with
100 error-feedback retries (run 20260920_015152: 100/100 attempts linear).
These seeds are installed directly for the first outer candidates so the store
accumulates scored, valid, creative references; later LLM proposals are
conditioned on them and deduplicated against them.

Every seed satisfies the same validator as LLM proposals: ``self.base`` plus
extra capacity that is silent (zero output) at initialization, deterministic
forward, torch-only imports, connected gradients. Each seed implements a
DIFFERENT mechanism (additive MLP, gated mixture, multiplicative modulation)
so the search starts from genuine architectural diversity.
"""

SEED_GATES: tuple[tuple[str, str], ...] = (
    (
        "additive-mlp-gelu",
        """import torch
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
""",
    ),
    (
        "gated-two-branch",
        """import torch
import torch.nn as nn

class LLMGeneratedGate(nn.Module):
    def __init__(self, model_dim: int, num_experts: int):
        super().__init__()
        self.base = nn.Linear(model_dim, num_experts, bias=False)
        hidden = 2 * num_experts
        self.branch_a = nn.Sequential(
            nn.Linear(model_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_experts, bias=False),
        )
        self.branch_b = nn.Sequential(
            nn.Linear(model_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_experts, bias=False),
        )
        self.mix = nn.Linear(model_dim, 2, bias=True)
        nn.init.zeros_(self.branch_a[-1].weight)
        nn.init.zeros_(self.branch_b[-1].weight)

    def forward(self, x):
        weights = torch.softmax(self.mix(x), dim=-1)
        return self.base(x) + weights[..., 0:1] * self.branch_a(x) + weights[..., 1:2] * self.branch_b(x)
""",
    ),
    (
        "multiplicative-modulation",
        """import torch
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
""",
    ),
)


def seed_gate_source(index: int) -> str:
    return SEED_GATES[index % len(SEED_GATES)][1]


def seed_gate_name(index: int) -> str:
    return SEED_GATES[index % len(SEED_GATES)][0]
