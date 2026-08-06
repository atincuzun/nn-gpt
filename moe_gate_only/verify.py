from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, List, Optional

import torch
import torch.nn as nn

from .datastructures import GateInstall
from .hooks import _current_replacement_module, _HOOK_HANDLES
from .sniff import _discover_vocab_size
from .tree import _assign_child

logger = logging.getLogger(__name__)


def _verify_forward(model: nn.Module, installs: List[GateInstall],
                    sample_input: Optional[Any] = None) -> bool:
    """Run a dummy forward to verify the model still works after replacement.

    Returns ``True`` if the forward succeeds or if the model is on meta
    device (where forward is impossible).
    """
    if not installs:
        return True

    try:
        if next(model.parameters()).device.type == "meta":
            return True
    except StopIteration:
        return True

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    with torch.no_grad():
        if sample_input is not None:
            try:
                sample = _move_sample_to_device(sample_input, device)
                _forward_with_sample(model, sample)
                return True
            except Exception as exc:
                logger.debug("Gate verification with sample_input failed", exc_info=exc)
                return False

        vocab = _discover_vocab_size(model) or 1000
        ids = torch.randint(3, max(vocab, 4), (1, 4), device=device)

        # Try causal-LM forward
        try:
            model(ids)
            return True
        except Exception as exc:
            logger.debug("Gate verification with positional input_ids failed", exc_info=exc)
        try:
            model(input_ids=ids)
            return True
        except Exception as exc:
            logger.debug("Gate verification with keyword input_ids failed", exc_info=exc)
        # Try seq2seq forward
        try:
            model(input_ids=ids, decoder_input_ids=ids)
            return True
        except Exception as exc:
            logger.debug("Gate verification with seq2seq input_ids failed", exc_info=exc)
    return False


def _move_sample_to_device(sample_input: Any, device: torch.device) -> Any:
    if isinstance(sample_input, torch.Tensor):
        return sample_input.to(device)
    if isinstance(sample_input, Mapping):
        return {k: _move_sample_to_device(v, device) for k, v in sample_input.items()}
    if isinstance(sample_input, tuple):
        return tuple(_move_sample_to_device(v, device) for v in sample_input)
    if isinstance(sample_input, list):
        return [_move_sample_to_device(v, device) for v in sample_input]
    return sample_input


def _forward_with_sample(model: nn.Module, sample_input: Any) -> Any:
    if isinstance(sample_input, Mapping):
        return model(**sample_input)
    if isinstance(sample_input, tuple):
        return model(*sample_input)
    if isinstance(sample_input, list):
        return model(*sample_input)
    return model(sample_input)


def _remove_install_hooks(inst: GateInstall) -> None:
    for handle in list(inst.hook_handles):
        try:
            handle.remove()
        except Exception:
            pass
    inst.hook_handles.clear()
    _HOOK_HANDLES.pop(id(inst), None)

    modules: list[nn.Module] = []
    try:
        modules.append(_current_replacement_module(inst))
    except Exception:
        pass
    try:
        block_gate = getattr(inst.site.block, inst.site.gate_attr)
        if all(block_gate is not module for module in modules):
            modules.append(block_gate)
    except Exception:
        pass

    for module in modules:
        if hasattr(module, "_gate_logits"):
            try:
                module._gate_logits.clear()
            except Exception:
                pass
            try:
                delattr(module, "_gate_logits")
            except Exception:
                pass


def _rollback(installs: List[GateInstall]) -> None:
    """Restore original gates."""
    for inst in reversed(installs):
        _remove_install_hooks(inst)
        try:
            owner = inst.owner or inst.site.block
            attr = inst.attr or inst.site.gate_attr
            old_child = inst.old_child or inst.old_gate
            _assign_child(owner, attr, old_child)
            if inst.teacher_requires_grad is not None:
                for parameter, requires_grad in zip(
                    inst.old_gate.parameters(), inst.teacher_requires_grad
                ):
                    parameter.requires_grad_(requires_grad)
        except (AttributeError, TypeError):
            pass
