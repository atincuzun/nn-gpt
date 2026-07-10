from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import GateFn, MoEModelInfo, RouterPattern


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def _maybe_normalize_topk(
    weights: torch.Tensor, normalize: bool, eps: float = 1e-20
) -> torch.Tensor:
    if normalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)
    return weights


class PatternAdapter(abc.ABC):
    """Produces a replacement forward for a router module.

    Subclasses implement one pattern group.  Optional post_install() hook
    lets adapters patch the MoE block when routing logic lives there.
    """

    @abc.abstractmethod
    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        top_k: int = 2,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """Return a callable compatible with the original router's forward.

        The returned callable will be bound via types.MethodType and
        must take (self, x).
        """
        ...

    def post_install(
        self,
        layer: nn.Module,
        custom_gate: GateFn,
        info: MoEModelInfo,
    ) -> None:
        """Optional hook called after the router is replaced on *layer*."""
        return


# -- Pattern 1: (logits, scores, indices)  Softmax-topK ------------------


class SoftmaxTopKAdapter(PatternAdapter):
    """Softmax + top-K routing, returns the 3-tuple the MoE block expects.

    Applicable: mixtral, qwen2_moe, qwen3_moe, qwen3_5_moe, olmoe,
    ernie4_5_moe, qwen3_next, dbrx.
    """

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        top_k: int = 2,
        **kwargs: Any,
    ) -> Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        def forward(
            router: nn.Module, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_flat = _flatten_tokens(x)
            logits = custom_gate(x_flat, model_dim, num_experts)
            weights = F.softmax(logits, dim=-1, dtype=torch.float)
            scores, indices = torch.topk(weights, top_k, dim=-1)
            normalize = bool(getattr(router, "norm_topk_prob", False))
            if router.__class__.__name__.lower().startswith("mixtral"):
                normalize = True
            scores = _maybe_normalize_topk(scores, normalize)
            return logits, scores.to(x.dtype), indices

        return forward


# -- Pattern 2: raw logits  (routing in MoE block) -----------------------


class RawLogitsAdapter(PatternAdapter):
    """Router returns raw logits; the MoE block owns the routing logic.

    Applicable: deepseek_v3, exaone_moe, glm4_moe, glm4_moe_lite,
    glm_moe_dsa, nemotron_h, mistral4, solar_open, hunyuan_v1_moe.
    """

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> Callable[..., torch.Tensor]:

        def forward(router: nn.Module, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            return custom_gate(_flatten_tokens(x), model_dim, num_experts)

        return forward


# -- Pattern 3: sparse dispatch  (jetmoe / granite) ----------------------


class SparseDispatchAdapter(PatternAdapter):
    """Sparse-dispatch routing based on sigmoid-gates + top-K.

    The router returns a 5-tuple used by the MoE block for sparse
    scatter/gather dispatch.

    Applicable: jetmoe, granitemoe, granitemoehybrid, granitemoeshared.
    """

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        top_k: int = 2,
        **kwargs: Any,
    ) -> Callable[..., Tuple[torch.Tensor, ...]]:

        def forward(
            router: nn.Module, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            x_flat = _flatten_tokens(x)
            logits = custom_gate(x_flat, model_dim, num_experts)
            top_logits, top_indices = torch.topk(logits.float(), top_k, dim=-1)
            top_gates = torch.softmax(top_logits, dim=-1).type_as(x_flat)

            num_tokens = x_flat.shape[0]
            flat_indices = top_indices.reshape(-1)
            flat_gates = top_gates.reshape(-1)

            index_sorted = flat_indices.argsort(stable=True)
            batch_index = index_sorted.div(top_k, rounding_mode="trunc")
            batch_gates = flat_gates[index_sorted]

            expert_size = torch.zeros(num_experts, device=x.device, dtype=torch.long)
            for i in range(num_experts):
                expert_size[i] = (flat_indices == i).sum().long()

            return index_sorted, batch_index, batch_gates, expert_size.tolist(), logits

        return forward


# -- Pattern 4: switch / top-1 with capacity ------------------------------


class SwitchCapacityAdapter(PatternAdapter):
    """Top-1 routing with expert capacity, used by SwitchTransformers.

    Returns (router_probs, expert_index, router_logits).
    """

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        def forward(
            router: nn.Module, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_flat = _flatten_tokens(x)
            router_logits = custom_gate(x_flat, model_dim, num_experts)
            router_probs_full = F.softmax(router_logits, dim=-1, dtype=torch.float).to(x.dtype)
            routing_weights, expert_index = torch.max(router_probs_full, dim=-1, keepdim=True)
            selected_experts = F.one_hot(expert_index, num_classes=num_experts)
            expert_capacity = getattr(router, "expert_capacity", None)
            if expert_capacity is not None:
                token_priority = torch.cumsum(selected_experts, dim=-2)
                selected_experts = selected_experts * (token_priority <= expert_capacity)
            return routing_weights, selected_experts, routing_weights

        return forward


# -- Pattern 5: NLLB-MoE / top-2 with capacity ---------------------------


class NLLBCapacityAdapter(PatternAdapter):
    """Top-2 routing with capacity, used by NLLB-MoE.

    Returns (top_1_mask, router_probs, router_logits).
    """

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        def forward(
            router: nn.Module, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_flat = _flatten_tokens(x)
            router_logits = custom_gate(x_flat, model_dim, num_experts)
            router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
            expert_index = torch.argmax(router_probs, dim=-1)
            top_1_mask = F.one_hot(expert_index, num_classes=num_experts)
            return top_1_mask, router_probs.to(x.dtype), router_logits

        return forward


# -- Pattern 6: sigmoid-sparsified (llama4) -------------------------------


class SigmoidSparsifiedAdapter(PatternAdapter):
    """Sigmoid-based sparsified routing.

    Returns (router_scores, router_logits).
    Applicable: llama4 (scout, Maverick).
    """

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        top_k: int = 1,
        **kwargs: Any,
    ) -> Callable[..., Tuple[torch.Tensor, torch.Tensor]]:

        def forward(
            router: nn.Module, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            x_flat = _flatten_tokens(x)
            router_logits = custom_gate(x_flat, model_dim, num_experts)
            _, router_indices = torch.topk(router_logits, top_k, dim=-1)
            sparse_logits = torch.full_like(router_logits, float("-inf"))
            sparse_logits.scatter_(1, router_indices, router_logits.gather(1, router_indices))
            router_scores = torch.sigmoid(sparse_logits.float()).to(router_logits.dtype)
            return router_scores, router_logits

        return forward


# -- Pattern 7: per-expert scaling (gemma4) -------------------------------


class PerExpertScalingAdapter(PatternAdapter):
    """Softmax + top-K with per-expert scaling factors.

    Returns (router_probabilities, top_k_weights, top_k_index).
    Applicable: gemma4.
    """

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        top_k: int = 2,
        **kwargs: Any,
    ) -> Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        def forward(
            router: nn.Module, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_flat = _flatten_tokens(x)
            logits = custom_gate(x_flat, model_dim, num_experts)
            router_probabilities = F.softmax(logits, dim=-1, dtype=torch.float)
            top_k_weights, top_k_index = torch.topk(
                router_probabilities, top_k, dim=-1
            )
            top_k_weights = _maybe_normalize_topk(top_k_weights, True)
            per_expert_scale = getattr(router, "per_expert_scale", None)
            if per_expert_scale is not None:
                top_k_weights = top_k_weights * per_expert_scale[top_k_index]
            return router_probabilities, top_k_weights.to(x.dtype), top_k_index

        return forward


# -- Pattern 8: Cohere top-K (softmax or sigmoid) -------------------------


class CohereTopKAdapter(PatternAdapter):
    """Softmax or sigmoid top-K routing.

    use_sigmoid=True for sigmoid-based routing (default: False -> softmax).
    Returns (router_logits, router_scores, selected_experts).
    Applicable: cohere2_moe.
    """

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        top_k: int = 2,
        use_sigmoid: bool = False,
        **kwargs: Any,
    ) -> Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        def forward(
            router: nn.Module, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_flat = _flatten_tokens(x)
            router_logits = custom_gate(x_flat, model_dim, num_experts)
            top_scores, selected_experts = torch.topk(router_logits, top_k, dim=-1)
            selection_fn = getattr(router, "expert_selection_fn", None)
            if selection_fn == "sigmoid" or (selection_fn is None and use_sigmoid):
                router_scores = torch.sigmoid(top_scores)
                if bool(getattr(router, "norm_topk_prob", False)):
                    router_scores = _maybe_normalize_topk(router_scores, True)
            else:
                router_scores = F.softmax(top_scores, dim=-1, dtype=torch.float)
            return router_logits, router_scores.to(x.dtype), selected_experts

        return forward


# -- Pattern 9: scoring_fn top-K (deepseek_v4) ----------------------------


class ScoringFnTopKAdapter(PatternAdapter):
    """Top-K after an arbitrary scoring function.

    scoring_fn: Callable (logits, dim) -> scores. Default: F.softmax.
    Returns (logits, weights, indices).
    Applicable: deepseek_v4.
    """

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        top_k: int = 2,
        scoring_fn: Optional[Callable[..., torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        _scoring_fn = scoring_fn or (lambda logits, dim=-1: F.softmax(logits, dim=dim))

        def forward(
            router: nn.Module, x: torch.Tensor, *args: Any, **kwargs: Any
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_flat = _flatten_tokens(x)
            logits = custom_gate(x_flat, model_dim, num_experts)
            score_fn = getattr(router, "score_fn", None)
            if score_fn is not None:
                scores = score_fn(logits)
            else:
                scores = _scoring_fn(logits, dim=-1)
            bias = getattr(router, "e_score_correction_bias", None)
            scores_for_choice = scores + bias if bias is not None else scores
            indices = torch.topk(scores_for_choice, top_k, dim=-1, sorted=False).indices
            weights = scores.gather(1, indices)
            weights = _maybe_normalize_topk(weights, True)
            scale = getattr(router, "routed_scaling_factor", None)
            if scale is not None:
                weights = weights * scale
            return logits, weights.to(x.dtype), indices

        return forward


# -- Pattern 10: afmoe sigmoid + bias -------------------------------------


class AfmoeSigmoidBiasAdapter(PatternAdapter):
    """Sigmoid + top-K routing with expert bias.

    Note: expert_bias is not handled by this adapter.  Fold bias into
    custom_gate_fn if needed.

    Returns (router_logits, top_scores, selected_experts).
    Applicable: afmoe.
    """

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        top_k: int = 2,
        **kwargs: Any,
    ) -> Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        def forward(
            router: nn.Module, x: torch.Tensor, expert_bias: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_flat = _flatten_tokens(x)
            router_logits = custom_gate(x_flat, model_dim, num_experts)
            router_scores = torch.sigmoid(router_logits)
            scores_for_choice = router_scores + expert_bias if expert_bias is not None else router_scores
            selected_experts = torch.topk(scores_for_choice, top_k, dim=-1).indices
            top_scores = router_scores.gather(dim=1, index=selected_experts)
            top_scores = _maybe_normalize_topk(top_scores, True)
            route_scale = getattr(router, "route_scale", None)
            if route_scale is not None:
                top_scores = top_scores * route_scale
            return router_logits, top_scores, selected_experts

        return forward


# -- Pattern 11: minimax sigmoid + bias -----------------------------------


class MinimaxSigmoidBiasAdapter(PatternAdapter):
    """Sigmoid + top-K routing with e_score_correction_bias.

    Note: e_score_correction_bias is not handled.  Fold bias into
    custom_gate_fn.

    Returns (router_logits, router_scores, top_k_index).
    Applicable: minimax_m2.
    """

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        top_k: int = 2,
        **kwargs: Any,
    ) -> Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        def forward(
            router: nn.Module, x: torch.Tensor, e_score_correction_bias: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_flat = _flatten_tokens(x)
            router_logits = custom_gate(x_flat, model_dim, num_experts)
            router_scores = torch.sigmoid(router_logits)
            scores_for_choice = (
                router_scores + e_score_correction_bias
                if e_score_correction_bias is not None
                else router_scores
            )
            top_k_index = torch.topk(scores_for_choice, top_k, dim=-1, sorted=False).indices
            top_k_weights = router_scores.gather(1, top_k_index)
            top_k_weights = _maybe_normalize_topk(top_k_weights, True)
            return router_logits, top_k_weights, top_k_index

        return forward


# -- Pattern 12: sparsemixer (phimoe) -------------------------------------


class SparsemixerAdapter(PatternAdapter):
    """Softmax top-K (sparsemixer variant used by phimoe).

    Returns (router_logits, routing_weights, selected_experts).
    Applicable: phimoe.
    """

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        top_k: int = 2,
        **kwargs: Any,
    ) -> Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        def forward(
            router: nn.Module, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_flat = _flatten_tokens(x)
            router_logits = custom_gate(x_flat, model_dim, num_experts)
            scores = F.softmax(router_logits, dim=-1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(scores, top_k, dim=-1)
            return router_logits, routing_weights.to(x.dtype), selected_experts

        return forward


# -- Pattern 14: Ernie top-K ----------------------------------------------


class ErnieTopKAdapter(PatternAdapter):
    """Ernie-MoE routing returns (router_logits, selected_experts, weights)."""

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        top_k: int = 2,
        **kwargs: Any,
    ) -> Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        def forward(
            router: nn.Module, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_flat = _flatten_tokens(x)
            router_logits = custom_gate(x_flat, model_dim, num_experts)
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
            scorer = getattr(router, "moe_statics", None)
            choice_scores = scorer(routing_weights) if scorer is not None else routing_weights
            _, selected_experts = torch.topk(choice_scores, top_k, dim=-1)
            routing_weights = routing_weights.gather(dim=-1, index=selected_experts)
            norm_min = float(getattr(router, "norm_min", 1e-20))
            routing_weights = routing_weights / torch.clamp(
                routing_weights.sum(dim=-1, keepdim=True), min=norm_min
            )
            return router_logits, selected_experts, routing_weights.to(x.dtype)

        return forward

        return forward


# -- Pattern 13: inline nn.Linear (no dedicated router) -------------------


class InlineLinearAdapter(PatternAdapter):
    """No dedicated router class - the gate is an nn.Linear used inline.

    Applicable: lfm2_moe.
    """

    def create_forward(
        self,
        custom_gate: GateFn,
        model_dim: int,
        num_experts: int,
        *,
        original_router: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> Callable[..., torch.Tensor]:

        def forward(router: nn.Module, x: torch.Tensor) -> torch.Tensor:
            return custom_gate(x, model_dim, num_experts)

        return forward


# -- Adapter registry ------------------------------------------------------


ADAPTERS: Dict[RouterPattern, PatternAdapter] = {
    RouterPattern.SOFTMAX_TOPK: SoftmaxTopKAdapter(),
    RouterPattern.RAW_LOGITS: RawLogitsAdapter(),
    RouterPattern.SPARSE_DISPATCH: SparseDispatchAdapter(),
    RouterPattern.SWITCH_CAPACITY: SwitchCapacityAdapter(),
    RouterPattern.NLLB_CAPACITY: NLLBCapacityAdapter(),
    RouterPattern.SIGMOID_SPARSIFIED: SigmoidSparsifiedAdapter(),
    RouterPattern.PER_EXPERT_SCALING: PerExpertScalingAdapter(),
    RouterPattern.COHERE_TOPK: CohereTopKAdapter(),
    RouterPattern.SCORING_FN_TOPK: ScoringFnTopKAdapter(),
    RouterPattern.AFMOE_SIGMOID_BIAS: AfmoeSigmoidBiasAdapter(),
    RouterPattern.MINIMAX_SIGMOID_BIAS: MinimaxSigmoidBiasAdapter(),
    RouterPattern.SPARSEMIXER: SparsemixerAdapter(),
    RouterPattern.INLINE_LINEAR: InlineLinearAdapter(),
    RouterPattern.ERNIE_TOPK: ErnieTopKAdapter(),
}
