from __future__ import annotations

import torch
import torch.nn as nn


def _guess_hidden_size(model: nn.Module) -> int | None:
    """Best-effort guess at hidden_size from model parameters."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for key in ("hidden_size", "d_model", "n_embd", "dim"):
            v = getattr(cfg, key, None)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
    return None


def _guess_vocab_size(model: nn.Module) -> int | None:
    """Best-effort guess at vocab_size from model config."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for key in ("vocab_size",):
            v = getattr(cfg, key, None)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
    return None


def _discover_hidden_size(model: nn.Module) -> int | None:
    """Discover hidden_size from parameters (no config)."""
    # Try embed_tokens
    embed = (getattr(model, "embed_tokens", None)
             or getattr(getattr(model, "model", None), "embed_tokens", None))
    if embed is not None and hasattr(embed, "weight"):
        return int(embed.weight.shape[-1])

    # Try first nn.Linear
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            return int(mod.in_features)

    return None


def _discover_vocab_size(model: nn.Module) -> int | None:
    """Discover vocab_size from parameters (no config)."""
    # Try lm_head
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Linear):
        return int(lm_head.out_features)

    # Try embed_tokens
    embed = (getattr(model, "embed_tokens", None)
             or getattr(getattr(model, "model", None), "embed_tokens", None))
    if embed is not None:
        return int(embed.weight.shape[0])

    return None
