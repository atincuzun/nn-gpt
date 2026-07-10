from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from .types import MoEModelInfo, RouterPattern

logger = logging.getLogger(__name__)


# -- Config-resolver helpers ----------------------------------------------


def _iter_config_candidates(cfg: PretrainedConfig) -> Iterable[Any]:
    """Yield config objects that may hold MoE fields."""
    seen: set[int] = set()
    stack: List[Any] = [cfg]
    for attr in ("text_config", "ffn_config"):
        nested = getattr(cfg, attr, None)
        if nested is not None:
            stack.append(nested)

    while stack:
        item = stack.pop(0)
        if id(item) in seen:
            continue
        seen.add(id(item))
        yield item


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            converted = _to_int(item)
            if converted is not None:
                return converted
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_num_experts(cfg: PretrainedConfig) -> int:
    for candidate in _iter_config_candidates(cfg):
        for key in (
            "num_experts",
            "num_local_experts",
            "num_routed_experts",
            "n_routed_experts",
            "moe_num_experts",
            "expert_count",
        ):
            val = _to_int(getattr(candidate, key, None))
            if val is not None:
                return val
    return 0


def _resolve_top_k(cfg: PretrainedConfig) -> int:
    for candidate in _iter_config_candidates(cfg):
        for key in (
            "top_k",
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k_experts",
            "top_k_experts_per_token",
            "moe_top_k",
            "moe_topk",
            "moe_k",
            "topk",
        ):
            val = _to_int(getattr(candidate, key, None))
            if val is not None:
                return val
    return 0


def _split_paths(path: str) -> List[str]:
    return [part.strip() for part in path.split(",") if part.strip()]


def _resolve_path(obj: Any, path: str) -> Any:
    """Resolve dotted attributes with optional integer list indexes."""
    if path == "":
        return obj
    current = obj
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(current, (list, tuple, nn.ModuleList)) and part.lstrip("-").isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part, None)
        if current is None:
            raise AttributeError(f"Could not resolve {path!r}")
    return current


def _try_resolve_path(obj: Any, path: str) -> Optional[Any]:
    try:
        return _resolve_path(obj, path)
    except (AttributeError, IndexError, TypeError):
        return None


# -- Known-model table ----------------------------------------------------


def _known_models() -> Dict[str, MoEModelInfo]:
    rows: Dict[str, MoEModelInfo] = {}

    def add(
        model_type: str,
        router_attr: str,
        moe_block_attr: str,
        pattern: RouterPattern,
        routing_in_moe_block: bool,
        conditional_moe: bool,
        decoder_attr: str,
    ) -> None:
        rows[model_type] = MoEModelInfo(
            model_type=model_type,
            router_attr=router_attr,
            moe_block_attr=moe_block_attr,
            pattern=pattern,
            routing_in_moe_block=routing_in_moe_block,
            conditional_moe=conditional_moe,
            decoder_attr=decoder_attr,
        )

    # Pattern 1 - softmax-topK
    add("mixtral", "gate", "mlp", RouterPattern.SOFTMAX_TOPK, False, True, "model.layers")
    add("qwen2_moe", "gate", "mlp", RouterPattern.SOFTMAX_TOPK, False, True, "model.layers")
    add("qwen3_moe", "gate", "mlp", RouterPattern.SOFTMAX_TOPK, False, True, "model.layers")
    add("qwen3_5_moe", "gate", "mlp", RouterPattern.SOFTMAX_TOPK, False, True, "model.layers")
    add("qwen3_5_moe_text", "gate", "mlp", RouterPattern.SOFTMAX_TOPK, False, True, "model.layers")
    add("qwen3_next", "gate", "mlp", RouterPattern.SOFTMAX_TOPK, False, True, "model.layers")
    add("olmoe", "gate", "mlp", RouterPattern.SOFTMAX_TOPK, False, False, "model.layers")
    add("ernie4_5_moe", "gate", "mlp", RouterPattern.ERNIE_TOPK, False, True, "model.layers")
    add(
        "ernie4_5_vl_moe_text",
        "text_moe.gate,vision_moe.gate",
        "mlp",
        RouterPattern.ERNIE_TOPK,
        False,
        True,
        "model.layers",
    )
    add(
        "ernie4_5_vl_moe",
        "text_moe.gate,vision_moe.gate",
        "mlp",
        RouterPattern.ERNIE_TOPK,
        False,
        True,
        "model.language_model.layers,language_model.layers,model.layers",
    )

    # Pattern 2 - raw logits
    add("dbrx", "router", "ffn", RouterPattern.RAW_LOGITS, True, False, "transformer.blocks")
    add("deepseek_v3", "gate", "mlp", RouterPattern.RAW_LOGITS, True, True, "model.layers")
    add("exaone_moe", "gate", "mlp", RouterPattern.RAW_LOGITS, True, True, "model.layers")
    add("glm4_moe", "gate", "mlp", RouterPattern.RAW_LOGITS, True, True, "model.layers")
    add("glm4_moe_lite", "gate", "mlp", RouterPattern.RAW_LOGITS, True, True, "model.layers")
    add("glm4v_moe_text", "gate", "mlp", RouterPattern.RAW_LOGITS, True, True, "model.layers")
    add("glm4v_moe", "gate", "mlp", RouterPattern.RAW_LOGITS, True, True, "model.language_model.layers,language_model.layers,model.layers")
    add("glm_moe_dsa", "gate", "mlp", RouterPattern.RAW_LOGITS, True, True, "model.layers")
    add("nemotron_h", "gate", "mlp", RouterPattern.RAW_LOGITS, True, True, "model.layers")
    add("mistral4", "gate", "mlp", RouterPattern.RAW_LOGITS, True, True, "model.layers")
    add("solar_open", "gate", "mlp", RouterPattern.RAW_LOGITS, True, True, "model.layers")
    add("hunyuan_v1_moe", "gate", "mlp", RouterPattern.RAW_LOGITS, True, True, "model.layers")

    # Pattern 3 - sparse dispatch
    add("jetmoe", "router", "mlp", RouterPattern.SPARSE_DISPATCH, False, False, "model.layers")
    add("granitemoe", "router", "block_sparse_moe", RouterPattern.SPARSE_DISPATCH, False, False, "model.layers")
    add("granitemoehybrid", "router", "block_sparse_moe", RouterPattern.SPARSE_DISPATCH, False, False, "model.layers")
    add("granitemoeshared", "router", "block_sparse_moe", RouterPattern.SPARSE_DISPATCH, False, False, "model.layers")

    # Pattern 4 - switch capacity
    add("switch_transformers", "router", "layer.-1.mlp", RouterPattern.SWITCH_CAPACITY, False, False, "encoder.block,decoder.block,model.encoder.block,model.decoder.block")

    # Pattern 5 - NLLB capacity
    add("nllb_moe", "router", "ffn", RouterPattern.NLLB_CAPACITY, False, False, "model.encoder.layers,encoder.layers,model.decoder.layers,decoder.layers")
    add("nllb-moe", "router", "ffn", RouterPattern.NLLB_CAPACITY, False, False, "model.encoder.layers,encoder.layers,model.decoder.layers,decoder.layers")

    # Pattern 6 - sigmoid sparsified
    add("llama4", "router", "feed_forward", RouterPattern.SIGMOID_SPARSIFIED, False, True, "model.layers")
    add("llama4_text", "router", "feed_forward", RouterPattern.SIGMOID_SPARSIFIED, False, True, "model.layers")
    add("llama4", "router", "feed_forward", RouterPattern.SIGMOID_SPARSIFIED, False, True, "model.language_model.layers,language_model.layers,model.layers")

    # Pattern 7 - per-expert scaling (router on decoder layer directly)
    add("gemma4", "router", "", RouterPattern.PER_EXPERT_SCALING, False, True, "model.layers")
    add("gemma4_text", "router", "", RouterPattern.PER_EXPERT_SCALING, False, True, "model.layers")
    add("gemma4", "router", "", RouterPattern.PER_EXPERT_SCALING, False, True, "model.language_model.layers,language_model.layers,model.layers")

    # Pattern 8 - Cohere top-K
    add("cohere2_moe", "gate", "mlp", RouterPattern.COHERE_TOPK, False, True, "model.layers")

    # Pattern 9 - scoring_fn top-K
    add("deepseek_v4", "gate", "mlp", RouterPattern.SCORING_FN_TOPK, False, True, "model.layers")

    # Pattern 10 - afmoe
    add("afmoe", "router", "mlp", RouterPattern.AFMOE_SIGMOID_BIAS, False, True, "model.layers")

    # Pattern 11 - minimax
    add("minimax_m2", "gate", "mlp", RouterPattern.MINIMAX_SIGMOID_BIAS, False, True, "model.layers")

    # Pattern 12 - sparsemixer
    add("phimoe", "router", "mlp", RouterPattern.SPARSEMIXER, False, False, "model.layers")

    # lfm2_moe - sigmoid routing with expert bias, routing in MoE block
    # gate is nn.Linear, routing logic in Lfm2MoeSparseMoeBlock.route_tokens_to_experts
    add("lfm2_moe", "gate", "feed_forward", RouterPattern.INLINE_LINEAR, True, True, "model.layers")

    return rows


KNOWN_MODELS: Dict[str, MoEModelInfo] = _known_models()


# -- Auto-detection -------------------------------------------------------


def detect_model_info(model: PreTrainedModel) -> MoEModelInfo:
    """Auto-detect MoE structure for an in-memory HuggingFace model."""
    cfg = model.config
    model_type = getattr(cfg, "model_type", "unknown")

    # 1. Static lookup
    if model_type in KNOWN_MODELS:
        info = KNOWN_MODELS[model_type]
        result = MoEModelInfo(
            model_type=model_type,
            router_attr=info.router_attr,
            moe_block_attr=info.moe_block_attr,
            pattern=info.pattern,
            routing_in_moe_block=info.routing_in_moe_block,
            conditional_moe=info.conditional_moe,
            decoder_attr=info.decoder_attr,
            num_experts=_resolve_num_experts(cfg),
            top_k=_resolve_top_k(cfg),
        )
        _fill_classes_from_sample(model, result)
        return result

    # 2. Dynamic fallback
    logger.info("Unknown model_type=%r -- attempting dynamic detection ...", model_type)
    result = _detect_dynamic(model, model_type)
    result.num_experts = _resolve_num_experts(cfg)
    result.top_k = _resolve_top_k(cfg)
    _fill_classes_from_sample(model, result)
    return result


def _detect_dynamic(model: PreTrainedModel, model_type: str) -> MoEModelInfo:
    decoder_attr = _find_decoder_attr(model)
    layers = _get_layers(model, decoder_attr) if decoder_attr else []

    for layer in layers:
        router, router_attr, moe_attr, _ = _find_moe_components(layer)
        if router is not None:
            pattern = _infer_pattern(router)
            return MoEModelInfo(
                model_type=model_type,
                router_attr=router_attr,
                moe_block_attr=moe_attr,
                pattern=pattern,
                routing_in_moe_block=(pattern == RouterPattern.RAW_LOGITS),
                conditional_moe=_has_conditional_moe(layers, moe_attr),
                decoder_attr=decoder_attr,
            )

    raise ValueError(
        f"Could not locate any MoE layer in model of type {model_type!r}. "
        "Try passing structural hints via the static registry."
    )


def _find_decoder_attr(model: PreTrainedModel) -> str:
    """Return the dot-separated attribute path to the decoder block list."""
    for candidate in ("model", "transformer", "encoder", "decoder"):
        root = getattr(model, candidate, None)
        if root is None:
            continue
        for sub in ("layers", "blocks", "block"):
            if hasattr(root, sub):
                return f"{candidate}.{sub}"
    return "model.layers"


def _get_layers(model: PreTrainedModel, decoder_attr: str) -> List[nn.Module]:
    """Resolve and return the list of decoder layers."""
    layers: List[nn.Module] = []
    errors: List[str] = []
    for path in _split_paths(decoder_attr):
        try:
            obj = _resolve_path(model, path)
        except (AttributeError, IndexError, TypeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        if isinstance(obj, nn.ModuleList):
            layers.extend(list(obj))
        elif isinstance(obj, (list, tuple)):
            layers.extend(list(obj))
        else:
            errors.append(f"{path}: expected list/module-list, got {type(obj)}")
    if layers:
        return layers
    detail = "; ".join(errors) if errors else decoder_attr
    raise TypeError(f"Could not resolve any layer list for {type(model).__name__}: {detail}")


def _find_moe_components(
    layer: nn.Module,
) -> Tuple[Optional[nn.Module], str, str, Optional[nn.Module]]:
    """Search a single decoder layer for MoE router and block.

    Returns (router, router_attr, moe_block_attr, moe_block) or (None, "", "", None).
    """
    for moe_attr in ("block_sparse_moe", "feed_forward", "ffn", "moe", "mlp"):
        moe_block = getattr(layer, moe_attr, None)
        if moe_block is None:
            continue
        for router_attr in ("gate", "router"):
            router = getattr(moe_block, router_attr, None)
            if router is not None and isinstance(router, nn.Module):
                return router, router_attr, moe_attr, moe_block
    # Fallback: router may live directly on the layer (Gemma4-style).
    for router_attr in ("gate", "router"):
        router = getattr(layer, router_attr, None)
        if router is not None and isinstance(router, nn.Module):
            return router, router_attr, "", layer
    return None, "", "", None


def _infer_pattern(router: nn.Module) -> RouterPattern:
    """Infer the return pattern from a router module's forward method."""
    try:
        sig = inspect.signature(router.forward)
        ann = sig.return_annotation
        if ann is not inspect.Parameter.empty:
            mapped = _match_return_annotation(ann)
            if mapped is not None:
                return mapped
        source = inspect.getsource(router.forward)
        mapped = _infer_pattern_from_source(source)
        if mapped is not None:
            return mapped
    except (ValueError, TypeError, OSError):
        pass

    return RouterPattern.RAW_LOGITS


def _match_return_annotation(ann: Any) -> Optional[RouterPattern]:
    """Heuristic matching of type annotations to known patterns."""
    key: str
    if isinstance(ann, str):
        key = ann.lower().replace(" ", "")
    elif hasattr(ann, "__str__"):
        key = str(ann).lower().replace(" ", "")
    else:
        return None

    if key.startswith("tuple["):
        inner = key[6:-1] if key.endswith("]") else key[5:]
        count = inner.count(",") + 1 if inner else 0
        if count == 2:
            return RouterPattern.SIGMOID_SPARSIFIED
        if count >= 3:
            return RouterPattern.SOFTMAX_TOPK
    return None


def _infer_pattern_from_source(source: str) -> Optional[RouterPattern]:
    """Heuristic keyword-based pattern detection from router source."""
    src = source.lower()
    has_softmax = "softmax" in src
    has_topk = "topk" in src
    has_sigmoid = "sigmoid" in src
    has_argmax = "argmax" in src

    if has_softmax and has_topk:
        return RouterPattern.SOFTMAX_TOPK
    if "returnrouter_scores,router_logits" in src.replace(" ", ""):
        return RouterPattern.SIGMOID_SPARSIFIED
    if has_sigmoid and has_topk:
        return RouterPattern.COHERE_TOPK
    if has_sigmoid and not has_topk:
        return RouterPattern.SIGMOID_SPARSIFIED
    if has_softmax and has_argmax:
        return RouterPattern.SWITCH_CAPACITY
    return None


def _has_conditional_moe(layers: List[nn.Module], moe_attr: str) -> bool:
    """True iff only a subset of decoder layers carry moe_attr."""
    if moe_attr == "":
        return False
    moe_count = sum(1 for layer in layers if _try_resolve_path(layer, moe_attr) is not None)
    return 0 < moe_count < len(layers)


def _fill_classes_from_sample(model: PreTrainedModel, info: MoEModelInfo) -> None:
    """Populate router_cls and moe_block_cls from a sample layer."""
    try:
        _, moe_block = _first_moe_layer(model, info)
    except (StopIteration, ValueError, AttributeError):
        return
    info.moe_block_cls = type(moe_block)
    router = None
    for router_attr in _split_paths(info.router_attr):
        router = _try_resolve_path(moe_block, router_attr) or _try_resolve_path(layer, router_attr)
        if router is not None:
            break
    if router is not None:
        info.router_cls = type(router)


def _first_moe_layer(
    model: PreTrainedModel, info: MoEModelInfo
) -> Tuple[nn.Module, nn.Module]:
    """Return (layer, moe_block) for the first MoE decoder layer."""
    for layer in _get_layers(model, info.decoder_attr):
        moe_block = _try_resolve_path(layer, info.moe_block_attr)
        if moe_block is not None and any(
            _try_resolve_path(moe_block, router_attr) is not None
            or _try_resolve_path(layer, router_attr) is not None
            for router_attr in _split_paths(info.router_attr)
        ):
            return layer, moe_block
    raise ValueError(f"No MoE block found in model of type {info.model_type!r}")
