"""MoE-cycle-scoped fix for generation-head dtype inference under bnb quantization.

``ab.gpt.util.GenerationDType.infer_generation_head_dtype`` reads
``bnb_4bit_compute_dtype`` unconditionally. ``BitsAndBytesConfig(load_in_8bit=True)``
materializes that attribute with its default (``float32``) even when running in
8-bit mode, so ChatBot casts ``lm_head`` to fp32 and rewrites
``model.config.dtype = float32`` — LFM2.5-MoE's conv blocks then crash with
"expected scalar type Float but found BFloat16".

The upstream file is shared by RL/Tune flows, so instead of editing it we
patch the symbol that Chatbot.py actually uses at import time, honoring only
the compute dtype of the ACTIVE quantization mode.
"""

from __future__ import annotations

from typing import Any, Optional


def infer_generation_head_dtype_for_cycle(model: Any, fallback: Optional[Any] = None):
    """Active-mode-aware variant of GenerationDType.infer_generation_head_dtype."""
    import torch

    from ab.gpt.util.GenerationDType import normalize_torch_dtype, _value_from

    config = getattr(model, "config", None)
    quantization_configs = []
    for owner in (model, config):
        quant_config = _value_from(owner, "quantization_config")
        if quant_config is not None and quant_config not in quantization_configs:
            quantization_configs.append(quant_config)

    def flag(quant_config: Any, name: str) -> bool:
        value = _value_from(quant_config, name)
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    for quant_config in quantization_configs:
        load_in_4bit = flag(quant_config, "load_in_4bit")
        load_in_8bit = flag(quant_config, "load_in_8bit")
        candidates = (
            ("bnb_4bit_compute_dtype", load_in_4bit),
            ("bnb_8bit_compute_dtype", load_in_8bit),
            ("compute_dtype", load_in_4bit or load_in_8bit),
        )
        for attr_name, active in candidates:
            if not active:
                continue
            dtype = normalize_torch_dtype(_value_from(quant_config, attr_name))
            if dtype is not None:
                return dtype

    for owner in (config, model):
        for attr_name in ("dtype", "torch_dtype"):
            dtype = normalize_torch_dtype(_value_from(owner, attr_name))
            if dtype is not None:
                return dtype

    return normalize_torch_dtype(fallback)


def ensure_generation_dtype_policy() -> None:
    """Point Chatbot's dtype inference at the active-mode-aware variant.

    Idempotent; records the applied state on the function so repeated calls
    (entry point + main()) are cheap no-ops.
    """
    import ab.gpt.util.Chatbot as chatbot_module

    if getattr(chatbot_module.infer_generation_head_dtype, "_moe_cycle_policy", False):
        return

    patched = infer_generation_head_dtype_for_cycle
    patched._moe_cycle_policy = True  # type: ignore[attr-defined]
    chatbot_module.infer_generation_head_dtype = patched
