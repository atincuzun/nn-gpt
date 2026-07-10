from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch

from ab.gpt.moe.config import build_moe_recipe_record


def _jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _iter_model_wrappers(model) -> Iterator[Any]:
    queue = [model]
    seen = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current

        for attr_name in ("module", "model", "base_model"):
            child = getattr(current, attr_name, None)
            if child is not None:
                queue.append(child)

        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            try:
                child = get_base_model()
            except TypeError:
                child = None
            if child is not None:
                queue.append(child)


def _resolve_tutel_adapter(model):
    for holder in _iter_model_wrappers(model):
        adapter = getattr(holder, "tutel_adapter", None)
        if adapter is not None:
            return adapter
    return None


def _resolve_adapter_checkpoint_path(checkpoint_path: Path, filename: str) -> Optional[Path]:
    checkpoint_path = Path(checkpoint_path)
    candidates = []
    if checkpoint_path.is_dir():
        candidates.extend([
            checkpoint_path / filename,
            checkpoint_path / "moe" / filename,
        ])
    else:
        candidates.extend([
            checkpoint_path,
            checkpoint_path.parent / filename,
            checkpoint_path.parent / "moe" / filename,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def persist_moe_artifacts(
    out_path: Path,
    moe_runtime: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not moe_runtime or not moe_runtime.get("enabled", False):
        return None

    moe_dir = out_path / "moe"
    moe_dir.mkdir(parents=True, exist_ok=True)

    record = build_moe_recipe_record(moe_runtime, extra=extra)
    metadata_filename = moe_runtime.get("metadata_filename", "moe_edit_metadata.json")
    with open(moe_dir / metadata_filename, "w") as fh:
        json.dump(_jsonable(record), fh, indent=2, default=str)

    return record


def save_tutel_adapter_checkpoint(model, output_path: Path) -> bool:
    adapter = _resolve_tutel_adapter(model)
    if adapter is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": adapter.state_dict(),
        "class_name": adapter.__class__.__name__,
    }
    torch.save(payload, output_path)
    return True


def load_tutel_adapter_checkpoint(
    model,
    checkpoint_path: Path,
    filename: str = "tutel_adapter.pt",
    strict: bool = True,
) -> bool:
    adapter = _resolve_tutel_adapter(model)
    if adapter is None:
        return False

    resolved_path = _resolve_adapter_checkpoint_path(checkpoint_path, filename)
    if resolved_path is None:
        return False

    map_location = "cpu"
    for parameter in adapter.parameters():
        map_location = parameter.device
        break

    payload = torch.load(resolved_path, map_location=map_location)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    adapter.load_state_dict(state_dict, strict=strict)
    return True
