from __future__ import annotations

import logging
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .adapters import ADAPTERS
from .registry import (
    _get_layers,
    _first_moe_layer,
    _iter_config_candidates,
    _split_paths,
    _try_resolve_path,
    detect_model_info,
)
from .types import GateFn, MoEModelInfo, RouterPattern

logger = logging.getLogger(__name__)


# -- Helper: enumerate all MoE blocks ------------------------------------


def list_moe_blocks(model: PreTrainedModel) -> List[Tuple[int, nn.Module, nn.Module]]:
    """Return [(layer_index, decoder_layer, moe_block), ...] for every MoE layer."""
    info = detect_model_info(model)
    layers = _get_layers(model, info.decoder_attr)
    blocks: List[Tuple[int, nn.Module, nn.Module]] = []
    for idx, layer in enumerate(layers):
        moe_block = _try_resolve_path(layer, info.moe_block_attr)
        if moe_block is not None and _find_routers(layer, moe_block, info.router_attr):
            blocks.append((idx, layer, moe_block))
    return blocks


# -- replace_router -------------------------------------------------------


class _InlineLinearRouter(nn.Module):
    """Wrapper that replaces an inline nn.Linear gate."""

    def __init__(self, custom_gate_fn: GateFn, model_dim: int, num_experts: int) -> None:
        super().__init__()
        self.custom_gate_fn = custom_gate_fn
        self.model_dim = model_dim
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_gate_fn(x, self.model_dim, self.num_experts)

    def extra_repr(self) -> str:
        return f"model_dim={self.model_dim}, num_experts={self.num_experts}"


def _get_model_dim(model: PreTrainedModel, info: MoEModelInfo) -> int:
    """Infer the model's hidden dimension from config or router weights."""
    cfg = model.config
    for attr in ("hidden_size", "d_model", "n_embd", "dim", "hidden_dim"):
        val = getattr(cfg, attr, None)
        if val is not None:
            return int(val)
    try:
        _, moe_block = _first_moe_layer(model, info)
    except (StopIteration, ValueError):
        raise ValueError(
            f"Cannot determine model_dim for {info.model_type}. "
            "Pass model_dim=N in kwargs to replace_router()."
        ) from None

    router = None
    for router_attr in _split_paths(info.router_attr):
        router = _try_resolve_path(moe_block, router_attr)
        if router is not None:
            break
    if router is None:
        router = getattr(model, info.router_attr, None)
    if router is not None:
        if isinstance(router, nn.Linear):
            return int(router.in_features)
        for attr in ("layer", "proj", "gate", "classifier", "wg"):
            child = getattr(router, attr, None)
            if isinstance(child, nn.Linear):
                return int(child.in_features)
        for param in router.parameters():
            return param.shape[-1]

    raise ValueError(
        f"Cannot determine model_dim for {info.model_type}. "
        "Pass model_dim=N in kwargs to replace_router()."
    )


def _resolve_parent(obj: nn.Module, path: str) -> Tuple[Any, str]:
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise AttributeError("empty path has no parent")
    parent: Any = obj
    for part in parts[:-1]:
        if isinstance(parent, (list, tuple, nn.ModuleList)) and part.lstrip("-").isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _set_child(obj: nn.Module, path: str, value: nn.Module) -> None:
    parent, attr = _resolve_parent(obj, path)
    if isinstance(parent, nn.ModuleList) and attr.lstrip("-").isdigit():
        parent[int(attr)] = value
    else:
        setattr(parent, attr, value)


def _find_routers(
    layer: nn.Module, moe_block: nn.Module, router_attr: str
) -> List[Tuple[nn.Module, str, nn.Module]]:
    """Return [(router, router_path, owner), ...] for router_attr paths."""
    routers: List[Tuple[nn.Module, str, nn.Module]] = []
    for path in _split_paths(router_attr):
        router = _try_resolve_path(moe_block, path)
        owner = moe_block
        if router is None and moe_block is not layer:
            router = _try_resolve_path(layer, path)
            owner = layer
        if isinstance(router, nn.Module):
            routers.append((router, path, owner))
    return routers


def _read_int_attr(obj: Any, attrs: Tuple[str, ...]) -> Optional[int]:
    for attr in attrs:
        val = getattr(obj, attr, None)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            val = next((item for item in val if item is not None), None)
        try:
            return int(val)
        except (TypeError, ValueError):
            continue
    return None


def _runtime_num_experts(router: nn.Module, moe_block: nn.Module, fallback: int) -> int:
    found = _read_int_attr(router, ("num_experts", "n_routed_experts", "num_local_experts", "out_features"))
    if found is not None:
        return found
    found = _read_int_attr(moe_block, ("num_experts", "n_routed_experts", "num_local_experts"))
    if found is not None:
        return found
    if isinstance(router, nn.Linear):
        return int(router.out_features)
    for attr in ("layer", "proj", "gate", "classifier", "wg"):
        child = getattr(router, attr, None)
        if isinstance(child, nn.Linear):
            return int(child.out_features)
    weight = getattr(router, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim >= 2:
        return int(weight.shape[0])
    return fallback


def _runtime_top_k(router: nn.Module, moe_block: nn.Module, fallback: int, pattern: RouterPattern) -> int:
    found = _read_int_attr(router, ("top_k", "moe_top_k", "moe_topk", "moe_k"))
    if found is not None:
        return found
    found = _read_int_attr(moe_block, ("top_k", "moe_top_k", "moe_topk", "moe_k"))
    if found is not None:
        return found
    if fallback:
        return fallback
    if pattern == RouterPattern.SWITCH_CAPACITY:
        return 1
    if pattern in (RouterPattern.NLLB_CAPACITY, RouterPattern.SPARSEMIXER):
        return 2
    return 2


def _runtime_model_dim(
    model: PreTrainedModel,
    info: MoEModelInfo,
    router: nn.Module,
    moe_block: nn.Module,
    override: Optional[int],
) -> int:
    if override is not None:
        return int(override)
    found = _read_int_attr(router, ("hidden_dim", "hidden_size", "input_size", "in_features"))
    if found is not None:
        return found
    found = _read_int_attr(moe_block, ("hidden_dim", "hidden_size", "input_size"))
    if found is not None:
        return found
    if isinstance(router, nn.Linear):
        return int(router.in_features)
    for attr in ("layer", "proj", "gate", "classifier", "wg"):
        child = getattr(router, attr, None)
        if isinstance(child, nn.Linear):
            return int(child.in_features)
    weight = getattr(router, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim >= 2:
        return int(weight.shape[-1])
    return _get_model_dim(model, info)


def replace_router(
    model: PreTrainedModel,
    custom_gate_fn: GateFn,
    **kwargs: Any,
) -> int:
    """Replace every MoE router in model with a forward using custom_gate_fn.

    Parameters
    ----------
    model:
        A loaded HuggingFace PreTrainedModel with MoE layers.
    custom_gate_fn:
        Callable with signature (x, model_dim, num_experts) -> logits.
    **kwargs:
        Forwarded to the appropriate PatternAdapter.create_forward().
        Common options: top_k, model_dim, num_experts, use_sigmoid (Cohere),
        scoring_fn (DeepSeek-V4).

    Returns
    -------
    int
        Number of routers replaced.
    """
    info = detect_model_info(model)
    adapter = ADAPTERS[info.pattern]

    num_experts_override = kwargs.pop("num_experts", None)
    model_dim_override = kwargs.pop("model_dim", None)
    top_k_override = kwargs.pop("top_k", None)

    layers = _get_layers(model, info.decoder_attr)
    replaced = 0

    for layer_idx, layer in enumerate(layers):
        moe_block = _try_resolve_path(layer, info.moe_block_attr)
        if moe_block is None:
            continue
        routers = _find_routers(layer, moe_block, info.router_attr)
        if not routers:
            continue

        for router, router_path, router_owner in routers:
            num_experts = (
                int(num_experts_override)
                if num_experts_override is not None
                else _runtime_num_experts(router, moe_block, info.num_experts)
            )
            if num_experts == 0:
                raise ValueError(
                    f"Cannot determine num_experts for model_type={info.model_type!r}. "
                    "Pass num_experts=N in kwargs or set it in the config."
                )

            top_k = (
                int(top_k_override)
                if top_k_override is not None
                else _runtime_top_k(router, moe_block, info.top_k, info.pattern)
            )
            if top_k < 1 or top_k > num_experts:
                raise ValueError(
                    f"Invalid top_k={top_k} for num_experts={num_experts} "
                    f"in layer {layer_idx} ({info.model_type})."
                )

            model_dim = _runtime_model_dim(
                model, info, router, moe_block, model_dim_override
            )
            adapter_kwargs: Dict[str, Any] = {"top_k": top_k, **kwargs}

            if info.pattern == RouterPattern.INLINE_LINEAR:
                old_gate = router
                if not isinstance(old_gate, nn.Linear):
                    logger.warning(
                        "Layer %d: expected nn.Linear at %s, got %s",
                        layer_idx, router_path, type(old_gate).__name__,
                    )
                    continue
                new_gate = _InlineLinearRouter(custom_gate_fn, model_dim, num_experts)
                _set_child(router_owner, router_path, new_gate)
            else:
                new_forward = adapter.create_forward(
                    custom_gate_fn,
                    model_dim,
                    num_experts,
                    original_router=router,
                    **adapter_kwargs,
                )
                router.forward = types.MethodType(new_forward, router)

            replaced += 1

        adapter.post_install(layer, custom_gate_fn, info)

    logger.info(
        "Replaced %d routers using %s",
        replaced, info.pattern.name,
    )
    return replaced


# -- modify_config --------------------------------------------------------


def modify_config(model: PreTrainedModel, **config_changes: Any) -> None:
    """Update exposed configuration parameters and propagate to router instances.

    Parameters
    ----------
    **config_changes:
        Attribute-value pairs to set on model.config
        (e.g. top_k=4, capacity_factor=1.5).
    """
    cfg = model.config

    for key, value in config_changes.items():
        changed = False
        for candidate in _iter_config_candidates(cfg):
            if hasattr(candidate, key):
                setattr(candidate, key, value)
                logger.info("Config %s.%s -> %s", type(candidate).__name__, key, value)
                changed = True
        if not changed:
            logger.warning("Config has no attribute %r -- skipping", key)

    _ROUTER_ATTR_MAP: Dict[str, str] = {
        "num_experts": "num_experts",
        "num_local_experts": "num_experts",
        "num_routed_experts": "num_experts",
        "n_routed_experts": "num_experts",
        "moe_num_experts": "num_experts",
        "top_k": "top_k",
        "num_experts_per_tok": "top_k",
        "num_experts_per_token": "top_k",
        "top_k_experts": "top_k",
        "capacity_factor": "capacity_factor",
        "eval_capacity_factor": "eval_capacity_factor",
        "min_capacity": "min_capacity",
        "moe_top_k": "top_k",
        "moe_topk": "top_k",
        "moe_k": "top_k",
    }

    resolved: Dict[str, Any] = {}
    for cfg_attr, router_attr in _ROUTER_ATTR_MAP.items():
        if cfg_attr in config_changes:
            resolved[router_attr] = config_changes[cfg_attr]
        elif router_attr not in resolved:
            for candidate in _iter_config_candidates(cfg):
                val = getattr(candidate, cfg_attr, None)
                if val is not None:
                    if isinstance(val, (list, tuple)):
                        val = next((item for item in val if item is not None), None)
                    resolved[router_attr] = val
                    break

    if not resolved:
        return

    try:
        info = detect_model_info(model)
    except (ValueError, AttributeError):
        logger.warning("Could not detect model info; config changes applied to config only.")
        return

    for attr, value in resolved.items():
        _set_attr_on_routers(model, info, attr, value)


def _set_attr_on_routers(
    model: PreTrainedModel, info: MoEModelInfo, attr: str, value: Any
) -> None:
    """Set attr = value on every router instance in model."""
    try:
        layers = _get_layers(model, info.decoder_attr)
    except (ValueError, TypeError, AttributeError):
        return
    for layer in layers:
        moe_block = _try_resolve_path(layer, info.moe_block_attr)
        if moe_block is None:
            continue
        for router, _, _ in _find_routers(layer, moe_block, info.router_attr):
            if hasattr(router, attr):
                old = getattr(router, attr)
                setattr(router, attr, value)
                logger.debug("  %s.%s: %s -> %s", type(router).__name__, attr, old, value)
        # Also try setting on the MoE block (some models store top_k
        # on the block rather than the router, e.g. lfm2_moe).
        if hasattr(moe_block, attr):
            old = getattr(moe_block, attr)
            setattr(moe_block, attr, value)
            logger.debug("  %s.%s: %s -> %s", type(moe_block).__name__, attr, old, value)
