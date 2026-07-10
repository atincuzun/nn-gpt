"""moe_surgery - Load any HuggingFace MoE model and replace/modify the router."""

from .adapters import ADAPTERS, PatternAdapter
from .registry import KNOWN_MODELS, detect_model_info
from .surgery import list_moe_blocks, modify_config, replace_router
from .types import GateFn, MoEModelInfo, RouterPattern

__all__ = [
    "ADAPTERS",
    "GateFn",
    "KNOWN_MODELS",
    "MoEModelInfo",
    "PatternAdapter",
    "RouterPattern",
    "detect_model_info",
    "list_moe_blocks",
    "modify_config",
    "replace_router",
]
