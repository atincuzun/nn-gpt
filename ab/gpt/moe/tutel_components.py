import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLinearGate(nn.Module):
    """Minimal custom Tutel gate: linear logits with optional temperature."""

    def __init__(self, model_dim, num_global_experts, k=1, temperature=1.0, fp32_gate=False, **options):
        super().__init__()
        dtype = torch.float32 if fp32_gate else None
        self.wg = nn.Linear(model_dim, num_global_experts, bias=False, dtype=dtype)
        self.top_k = min(num_global_experts, int(k))
        self.temperature = float(max(1e-6, temperature))

        self.gate_noise = float(options.get("gate_noise", 0.0))
        self.capacity_factor = float(options.get("capacity_factor", 1.0))

    def forward(self, x):
        wg = self.wg.float() if self.wg.weight.dtype != x.dtype else self.wg
        return wg(x.to(dtype=wg.weight.dtype)) / self.temperature


class CustomExpertMLP(nn.Module):
    """Simple custom expert network compatible with Tutel expert API."""

    def __init__(self, model_dim, num_experts_per_device, sharded_count, hidden_size_per_expert=256, **_):
        super().__init__()
        if sharded_count != 1:
            raise ValueError("CustomExpertMLP demo supports sharded_count=1 only.")

        self.num_experts = int(num_experts_per_device)
        self.model_dim = int(model_dim)
        self.hidden = int(hidden_size_per_expert)

        self.w1 = nn.Parameter(torch.empty(self.num_experts, self.hidden, self.model_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_experts, self.hidden))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, self.hidden, self.model_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_experts, self.model_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x, _ctx):
        y = torch.einsum("etm,ehm->eth", x, self.w1) + self.b1.unsqueeze(1)
        y = F.silu(y)
        y = torch.einsum("eth,ehm->etm", y, self.w2) + self.b2.unsqueeze(1)
        return y
