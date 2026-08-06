from __future__ import annotations

from typing import Any, Optional, Tuple

import torch.nn as nn


def _module_path_map(model: nn.Module) -> dict[int, str]:
    return {id(module): path for path, module in model.named_modules()}


def _resolve_parent_attr(model: nn.Module, path: str) -> Optional[Tuple[nn.Module, str]]:
    if not path:
        return None
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        try:
            parent = _resolve_path_component(parent, part)
        except (AttributeError, IndexError, TypeError):
            return None
    return parent, parts[-1]


def _resolve_path_component(obj: Any, part: str) -> Any:
    """Resolve one component of a dotted path, handling integer indexes."""
    if part.lstrip("-").isdigit():
        return obj[int(part)]
    return getattr(obj, part)


def _assign_child(owner: nn.Module, attr: str, child: nn.Module) -> None:
    """Assign a child module, including numeric ModuleList/Sequential entries."""
    if isinstance(owner, (nn.ModuleList, nn.Sequential)) and attr.lstrip("-").isdigit():
        owner[int(attr)] = child
    elif isinstance(owner, nn.ModuleDict):
        owner[attr] = child
    else:
        setattr(owner, attr, child)


def _extract_layer_index(module_name: str) -> int:
    """Best-effort extraction of layer index from a dotted module name."""
    parts = module_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
        if part == "block" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def _find_module_path(model: nn.Module, target: nn.Module) -> str | None:
    """Find the full dotted path of a module by identity."""
    for name, mod in model.named_modules():
        if mod is target:
            return name
    return None


def _find_parent_block(model: nn.Module, gate_path: str,
                       gate_module: nn.Module, pattern: str) -> dict:
    """Find the MoE block and the attribute name that points to *gate_module*.

    Walks up the module tree from *gate_module* via ``named_modules()``
    to find the parent that holds the gate as a direct child attribute.
    """
    if pattern == "linear" or pattern == "parameter_gate":
        # The parent is the MoE block; the gate is a direct child attr
        parts = gate_path.split(".")
        gate_local = parts[-1] if parts else "gate"
        # Walk to parent
        parent_obj = model
        for p in parts[:-1]:
            try:
                parent_obj = _resolve_path_component(parent_obj, p)
            except (AttributeError, IndexError, TypeError):
                return {"block": model, "gate_attr": gate_local}
        return {"block": parent_obj, "gate_attr": gate_local}

    if pattern == "wrapped_linear":
        # The gate module IS the wrapper.  Find which parent it belongs to.
        parts = gate_path.split(".")
        gate_local = parts[-1] if parts else "gate"
        parent_obj = model
        for p in parts[:-1]:
            try:
                parent_obj = _resolve_path_component(parent_obj, p)
            except (AttributeError, IndexError, TypeError):
                return {"block": model, "gate_attr": gate_local}
        return {"block": parent_obj, "gate_attr": gate_local}

    return {"block": model, "gate_attr": "gate"}
