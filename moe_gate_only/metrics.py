from __future__ import annotations

import math
from typing import Any, Iterable

import torch

from .universal import GateInstall


def _parameter_norms(parameters: Iterable[torch.nn.Parameter]) -> tuple[float, float]:
    parameters = list(parameters)
    weight_sq = sum(float(parameter.detach().float().pow(2).sum()) for parameter in parameters)
    grad_sq = sum(
        float(parameter.grad.detach().float().pow(2).sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    return math.sqrt(weight_sq), math.sqrt(grad_sq)


def collect_gate_metrics(installs: Iterable[GateInstall]) -> dict[str, Any]:
    """Collect routing and parameter diagnostics from installed gates."""
    layers: dict[str, Any] = {}
    for index, install in enumerate(installs):
        module = install.new_gate
        indices = getattr(module, "_last_topk_idx", None)
        weights = getattr(module, "_last_topk_weight", None)
        aux_loss = getattr(module, "_last_aux_loss", None)
        parameters = (
            install.student_gate.parameters()
            if install.student_gate is not None
            else module.parameters()
        )
        weight_norm, gradient_norm = _parameter_norms(parameters)
        layer: dict[str, Any] = {
            "path": install.site.path,
            "layer_index": install.site.layer_index,
            "model_dim": install.site.model_dim,
            "num_experts": install.site.num_experts,
            "gate_weight_norm": weight_norm,
            "gate_gradient_norm": gradient_norm,
            "auxiliary_loss": float(aux_loss.float().item()) if isinstance(aux_loss, torch.Tensor) else None,
        }
        if isinstance(indices, torch.Tensor) and indices.numel() > 0:
            counts = torch.bincount(
                indices.reshape(-1).cpu(), minlength=install.site.num_experts
            ).float()
            fractions = counts / counts.sum().clamp_min(1)
            nonzero = fractions[fractions > 0]
            layer.update({
                "routed_assignments": int(counts.sum().item()),
                "expert_counts": [int(value) for value in counts.tolist()],
                "expert_fractions": fractions.tolist(),
                "unused_experts": int((counts == 0).sum().item()),
                "routing_entropy": float(-(nonzero * nonzero.log()).sum().item()),
                "expert_concentration": float(fractions.max().item()),
            })
        if isinstance(weights, torch.Tensor) and weights.numel() > 0:
            layer["mean_selected_weight"] = float(weights.float().mean().item())
        layers[str(install.site.layer_index if install.site.layer_index >= 0 else index)] = layer
    return {"layers": layers}


def collect_teacher_student_metrics(installs: Iterable[GateInstall]) -> dict[str, Any]:
    """Compare frozen teacher and random student routing on the latest forward."""
    layers: dict[str, Any] = {}
    overlaps: list[float] = []
    divergences: list[float] = []
    for index, install in enumerate(installs):
        if install.student_gate is None or install.teacher_gate is None:
            continue
        module = install.new_gate
        teacher_indices = getattr(module, "_last_teacher_topk_idx", None)
        student_indices = getattr(module, "_last_student_topk_idx", None)
        teacher_logits = getattr(module, "_last_teacher_logits", None)
        student_logits = getattr(module, "_last_student_logits", None)
        distillation_value = getattr(module, "_last_distillation_value", None)
        metric_tokens = int(getattr(module, "_metric_tokens", 0))
        metric_topk_total = int(getattr(module, "_metric_topk_total", 0))
        teacher_norm, teacher_grad_norm = _parameter_norms(install.teacher_gate.parameters())
        student_norm, student_grad_norm = _parameter_norms(install.student_gate.parameters())
        layer: dict[str, Any] = {
            "path": install.site.path,
            "layer_index": install.site.layer_index,
            "student_weight": float(module.student_weight),
            "teacher_weight_norm": teacher_norm,
            "teacher_gradient_norm": teacher_grad_norm,
            "student_weight_norm": student_norm,
            "student_gradient_norm": student_grad_norm,
            "observed_tokens": metric_tokens,
            "distillation_kl": (
                float(getattr(module, "_metric_kl_weighted_sum", 0.0)) / metric_tokens
                if metric_tokens > 0
                else distillation_value
            ),
        }
        if metric_topk_total > 0:
            overlap = float(getattr(module, "_metric_topk_matches", 0.0)) / metric_topk_total
            layer["topk_overlap"] = overlap
            overlaps.append(overlap)
        elif (
            isinstance(teacher_indices, torch.Tensor)
            and isinstance(student_indices, torch.Tensor)
            and teacher_indices.shape == student_indices.shape
            and teacher_indices.numel() > 0
        ):
            matches = (
                teacher_indices.unsqueeze(-1) == student_indices.unsqueeze(-2)
            ).any(dim=-1)
            overlap = float(matches.float().mean().item())
            layer["topk_overlap"] = overlap
            overlaps.append(overlap)
        if (
            isinstance(teacher_logits, torch.Tensor)
            and isinstance(student_logits, torch.Tensor)
            and teacher_logits.shape == student_logits.shape
        ):
            centered_teacher = teacher_logits.float() - teacher_logits.float().mean(
                dim=-1, keepdim=True
            )
            centered_student = student_logits.float() - student_logits.float().mean(
                dim=-1, keepdim=True
            )
            layer["centered_logit_mse"] = float(
                (centered_teacher - centered_student).pow(2).mean().item()
            )
        if layer["distillation_kl"] is not None:
            divergences.append(float(layer["distillation_kl"]))
        key = str(install.site.layer_index if install.site.layer_index >= 0 else index)
        layers[key] = layer

    aggregate = {
        "layers_observed": len(layers),
        "mean_topk_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
        "minimum_topk_overlap": min(overlaps) if overlaps else None,
        "mean_distillation_kl": (
            sum(divergences) / len(divergences) if divergences else None
        ),
        "maximum_distillation_kl": max(divergences) if divergences else None,
    }
    return {"mode": "teacher_student", "aggregate": aggregate, "layers": layers}


def reset_teacher_student_metrics(installs: Iterable[GateInstall]) -> None:
    """Start a fresh teacher/student routing calibration window."""
    for install in installs:
        reset = getattr(install.new_gate, "reset_teacher_student_metrics", None)
        if callable(reset):
            reset()
