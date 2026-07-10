from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import torch


GateFn = Callable[[torch.Tensor, int, int], torch.Tensor]
"""Signature: (x: [*, model_dim], model_dim, num_experts) -> logits: [*, num_experts]"""


class RouterPattern(Enum):
    """Known MoE router return-signature patterns."""

    SOFTMAX_TOPK = 1
    RAW_LOGITS = 2
    SPARSE_DISPATCH = 3
    SWITCH_CAPACITY = 4
    NLLB_CAPACITY = 5
    SIGMOID_SPARSIFIED = 6
    PER_EXPERT_SCALING = 7
    COHERE_TOPK = 8
    SCORING_FN_TOPK = 9
    AFMOE_SIGMOID_BIAS = 10
    MINIMAX_SIGMOID_BIAS = 11
    SPARSEMIXER = 12
    INLINE_LINEAR = 13
    ERNIE_TOPK = 14


@dataclass
class MoEModelInfo:
    """Auto-detected structural information about an MoE model."""

    model_type: str
    router_attr: str
    moe_block_attr: str
    pattern: RouterPattern
    routing_in_moe_block: bool
    conditional_moe: bool
    decoder_attr: str
    num_experts: int = 0
    top_k: int = 0
    router_cls: Optional[type] = None
    moe_block_cls: Optional[type] = None
