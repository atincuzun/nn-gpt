from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .session import MoEGateSession


FORMAT_VERSION = 1


def _site_key(index: int, path: str) -> str:
    return f"{index:04d}:{path or '<unknown>'}"


def save_gate_checkpoint(
    session: "MoEGateSession",
    path: str | Path,
    *,
    optimizer: Any = None,
) -> Path:
    if not session.installs:
        raise RuntimeError("Install gates before saving a gate checkpoint")
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    gate_states: dict[str, dict[str, torch.Tensor]] = {}
    sites: list[dict[str, Any]] = []
    teacher_student = all(install.student_gate is not None for install in session.installs)
    for index, install in enumerate(session.installs):
        key = _site_key(index, install.site.path)
        saved_gate = install.student_gate if install.student_gate is not None else install.new_gate
        gate_states[key] = {
            name: value.detach().cpu()
            for name, value in saved_gate.state_dict().items()
        }
        sites.append({
            "key": key,
            "path": install.site.path,
            "layer_index": install.site.layer_index,
            "model_dim": install.site.model_dim,
            "num_experts": install.site.num_experts,
            "pattern": install.site.pattern,
        })
    config = getattr(session.model, "config", None)
    metadata = {
        "format_version": FORMAT_VERSION,
        "model_name_or_path": getattr(config, "_name_or_path", None),
        "gate_class_name": session.gate_class_name,
        "gate_factory_name": session.gate_factory_name,
        "mode": "teacher_student" if teacher_student else "direct",
        "student_weight": session.student_weight() if teacher_student else None,
        "distillation_temperature": (
            float(session.installs[0].new_gate.distillation_temperature)
            if teacher_student else None
        ),
        "top_k": (
            int(session.installs[0].new_gate.top_k) if teacher_student else None
        ),
        "sites": sites,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    torch.save({"gate_states": gate_states}, output / "gate_weights.pt")
    if session.gate_source is not None:
        (output / "gate.py").write_text(session.gate_source.rstrip() + "\n", encoding="utf-8")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), output / "optimizer.pt")
    return output


def load_gate_checkpoint(
    session: "MoEGateSession",
    path: str | Path,
    *,
    optimizer: Any = None,
) -> dict[str, Any]:
    if not session.installs:
        raise RuntimeError("Install the matching gate architecture before loading weights")
    source = Path(path)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported gate checkpoint format: {metadata.get('format_version')}")
    payload = torch.load(source / "gate_weights.pt", map_location="cpu", weights_only=True)
    states = payload["gate_states"]
    current_mode = (
        "teacher_student"
        if all(install.student_gate is not None for install in session.installs)
        else "direct"
    )
    saved_mode = metadata.get("mode", "direct")
    if saved_mode != current_mode:
        raise ValueError(
            f"Checkpoint mode {saved_mode!r} does not match installed mode {current_mode!r}"
        )
    saved_factory = metadata.get("gate_factory_name")
    if saved_factory is not None and saved_factory != session.gate_factory_name:
        raise ValueError(
            f"Checkpoint gate factory {saved_factory!r} does not match installed "
            f"factory {session.gate_factory_name!r}"
        )
    if current_mode == "teacher_student":
        expected_temperature = float(session.installs[0].new_gate.distillation_temperature)
        expected_top_k = int(session.installs[0].new_gate.top_k)
        if float(metadata.get("distillation_temperature", expected_temperature)) != expected_temperature:
            raise ValueError("Checkpoint distillation temperature does not match the session")
        if int(metadata.get("top_k", expected_top_k)) != expected_top_k:
            raise ValueError("Checkpoint router top_k does not match the session")
    expected = [_site_key(index, install.site.path) for index, install in enumerate(session.installs)]
    if set(states) != set(expected):
        raise ValueError("Checkpoint gate sites do not match the currently installed gates")
    for key, install in zip(expected, session.installs):
        saved_gate = install.student_gate if install.student_gate is not None else install.new_gate
        saved_gate.load_state_dict(states[key], strict=True)
    if metadata.get("mode") == "teacher_student":
        session.set_student_weight(float(metadata.get("student_weight", 0.0)))
    optimizer_path = source / "optimizer.pt"
    if optimizer is not None and optimizer_path.is_file():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
    gate_source = source / "gate.py"
    if gate_source.is_file():
        session.gate_source = gate_source.read_text(encoding="utf-8")
        session.gate_class_name = metadata.get("gate_class_name")
    return metadata
