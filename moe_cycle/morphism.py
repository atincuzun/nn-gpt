"""Morphism verification and initialization-metric helpers for the gate cycle."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dtype(name: str) -> Any:
    import torch
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _model_logits(model: Any, inputs: dict[str, Any]):
    import torch

    embeddings = model.get_input_embeddings()
    device = next(embeddings.parameters()).device
    model_inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**model_inputs)
    return output.logits.detach().float().cpu()


def _capture_native_gate_inputs(
    model: Any,
    sites: list[Any],
    inputs: dict[str, Any],
) -> tuple[Any, dict[int, Any]]:
    captured: dict[int, Any] = {}
    handles = []

    def make_hook(gate_id: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            if gate_id not in captured and args and hasattr(args[0], "detach"):
                captured[gate_id] = args[0].detach().cpu()
        return hook

    for site in sites:
        handles.append(site.gate.register_forward_pre_hook(make_hook(id(site.gate))))
    try:
        logits = _model_logits(model, inputs)
    finally:
        for handle in handles:
            handle.remove()
    return logits, captured


def _verify_morphed_gate_projections(
    installs: list[Any],
    native_inputs: dict[int, Any],
    *,
    allow_approximate: bool = False,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from moe_gate_only.universal import _resolve_top_k

    metrics: list[dict[str, Any]] = []
    for install in installs:
        hidden_states = native_inputs.get(id(install.old_gate))
        native_weight = getattr(install.old_gate, "weight", None)
        if hidden_states is None or not isinstance(native_weight, torch.Tensor):
            raise RuntimeError(
                f"Could not verify native projection at layer {install.site.layer_index}"
            )
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        native_input = hidden_states.to(device=native_weight.device)
        with torch.no_grad():
            native_route = install.old_gate(native_input)
            replacement_route = install.new_gate(native_input)
        if isinstance(native_route, torch.Tensor):
            expected_native = native_route.reshape(-1, native_route.shape[-1]).detach()
        else:
            expected_native = F.linear(
                flat.float(), native_weight.detach().float().cpu()
            )
        generated_gate = getattr(install.new_gate, "gate", install.new_gate)
        parameter = next(generated_gate.parameters())
        recorded_logits = getattr(install.new_gate, "_last_gate_logits", None)
        if isinstance(recorded_logits, torch.Tensor):
            # Contract wrappers such as DeepSeek-V2 perform their own input
            # promotion before invoking the generated gate.  Reuse the logits
            # captured by the replacement-route call above instead of
            # bypassing that model-specific dtype contract.
            actual_native = recorded_logits.detach()
        else:
            with torch.no_grad():
                actual_native = generated_gate(
                    flat.to(device=parameter.device)
                ).detach()
        expected = expected_native.float().cpu()
        actual = actual_native.float().cpu()
        projection_finite = bool(torch.isfinite(actual).all())
        if not projection_finite:
            raise RuntimeError(
                "Morphed gate returned non-finite projection values at layer "
                f"{install.site.layer_index}"
            )
        difference = (actual - expected).abs()
        installed_top_k = getattr(install.new_gate, "top_k", None)
        top_k = int(
            installed_top_k
            if installed_top_k is not None
            else _resolve_top_k(
                install.site.gate,
                install.site.block,
                install.site.num_experts,
                None,
            )
        )
        expected_indices = expected.topk(top_k, dim=-1, sorted=False).indices
        actual_indices = actual.topk(top_k, dim=-1, sorted=False).indices
        topk_equal = bool(torch.equal(actual_indices, expected_indices))
        allclose = bool(torch.allclose(actual, expected, rtol=5e-5, atol=5e-5))
        if isinstance(native_route, torch.Tensor) and isinstance(replacement_route, torch.Tensor):
            route_tokens = getattr(install.site.block, "route_tokens_to_experts", None)
            if callable(route_tokens):
                native_indices, native_weights = route_tokens(native_route)
                replacement_indices, replacement_weights = route_tokens(replacement_route)
            else:
                native_indices = native_route.topk(top_k, dim=-1, sorted=False).indices
                replacement_indices = replacement_route.topk(
                    top_k, dim=-1, sorted=False
                ).indices
                native_weights = native_route
                replacement_weights = replacement_route
        elif (
            isinstance(native_route, (tuple, list))
            and len(native_route) >= 2
            and isinstance(replacement_route, (tuple, list))
            and len(replacement_route) >= 2
        ):
            native_indices, native_weights = native_route[:2]
            replacement_indices, replacement_weights = replacement_route[:2]
        else:
            raise RuntimeError("Native and replacement gates returned incompatible contracts")
        routing_indices_equal = bool(torch.equal(native_indices, replacement_indices))
        routing_weight_difference = (
            native_weights.detach().float().cpu()
            - replacement_weights.detach().float().cpu()
        ).abs()
        routing_weights_equal = bool(torch.equal(native_weights, replacement_weights))
        routing_weights_allclose = bool(
            torch.allclose(native_weights, replacement_weights, rtol=5e-5, atol=5e-5)
        )
        bit_exact_expected = bool(
            getattr(generated_gate, "morphism_metrics", {}).get(
                "bit_exact_projection_expected", False
            )
        )
        routing_failed = (
            not routing_indices_equal
            or not routing_weights_allclose
            or (
                bit_exact_expected
                and (
                    not torch.equal(actual, expected)
                    or not routing_weights_equal
                )
            )
        )
        strict_equivalent = allclose and topk_equal and not routing_failed
        layer_metrics = {
            "captured_tokens": int(flat.shape[0]),
            "projection_finite": projection_finite,
            "projection_allclose": allclose,
            "projection_max_abs_error": float(difference.max().item()),
            "projection_mean_abs_error": float(difference.mean().item()),
            "top_k": top_k,
            "topk_indices_equal": topk_equal,
            "routing_indices_equal": routing_indices_equal,
            "routing_weights_equal": routing_weights_equal,
            "routing_weights_allclose": routing_weights_allclose,
            "routing_weights_max_abs_error": float(routing_weight_difference.max().item()),
            "bit_exact_projection_expected": bit_exact_expected,
            "strict_equivalent": strict_equivalent,
            "approximate_accepted": bool(allow_approximate and not strict_equivalent),
        }
        if not strict_equivalent and not allow_approximate:
            raise RuntimeError(
                "Function-preserving gate failed per-layer routing equivalence at layer "
                f"{install.site.layer_index}: {layer_metrics}"
            )
        metrics.append(layer_metrics)
    return metrics


def _perturb_gate_weights(installs: list[Any], noise_scale: float, seed: int) -> list[dict[str, Any]]:
    import torch

    if noise_scale < 0:
        raise ValueError("gate-init-noise-scale must be non-negative")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    metrics: list[dict[str, Any]] = []
    for index, install in enumerate(installs):
        generated_gate = getattr(install.new_gate, "gate", install.new_gate)
        base = getattr(generated_gate, "base", None)
        weight = getattr(base, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise ValueError(f"Replacement gate {index} has no base projection weight")

        weight_std = float(weight.detach().float().std().item())
        absolute_noise_std = weight_std * noise_scale
        noise = torch.randn(
            weight.shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ) * absolute_noise_std
        with torch.no_grad():
            weight.add_(noise.to(device=weight.device, dtype=weight.dtype))
        metrics.append({
            "layer_index": install.site.layer_index,
            "weight_std": weight_std,
            "noise_scale": noise_scale,
            "absolute_noise_std": absolute_noise_std,
            "noise_l2_norm": float(noise.norm().item()),
        })
    return metrics


def _morphism_initialization_metrics(installs: list[Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for install in installs:
        generated_gate = getattr(install.new_gate, "gate", install.new_gate)
        gate_metrics = getattr(generated_gate, "morphism_metrics", None)
        if not isinstance(gate_metrics, dict):
            raise RuntimeError(
                f"Replacement gate at layer {install.site.layer_index} has no morphism metrics"
            )
        metrics.append({
            "layer_index": install.site.layer_index,
            "path": install.site.path,
            **gate_metrics,
        })
    return metrics


def verify_parent_function_carry(
    parent_logits: Any,
    replacement_logits: Any,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Check that a successor gate reproduces its parent's step-zero function.

    With inherited modules copied by name and every new branch zero-initialized,
    the successor must emit exactly the parent's model logits before training.
    Divergence means the successor rewrote the inherited path (or the weight
    carry lost parameters) and the function-preserving carry is broken.
    """
    import torch

    parent_logits = parent_logits.detach().float().cpu()
    replacement_logits = replacement_logits.detach().float().cpu()
    difference = (parent_logits - replacement_logits).abs()
    return {
        "equivalent": bool(torch.allclose(parent_logits, replacement_logits, rtol=rtol, atol=atol)),
        "bit_exact": bool(torch.equal(parent_logits, replacement_logits)),
        "max_abs_difference": float(difference.max().item()),
        "rtol": rtol,
        "atol": atol,
    }


def carry_gate_weights(installs: list[Any], checkpoint_dir: Any) -> dict[str, Any]:
    """Seed freshly installed gates with a parent gate's trained weights.

    Parameters are matched by name: same-name, same-shape tensors are
    overwritten with the parent's trained values (cast to the successor's
    dtype/device), while parameters the parent never had keep their constructor
    initialization. This is the ``--gate-carry-forward`` transfer: the successor
    continues training where the parent stopped instead of restarting from the
    native router copy.
    """
    import torch

    source = Path(checkpoint_dir)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    payload = torch.load(source / "gate_weights.pt", map_location="cpu", weights_only=True)
    states = payload["gate_states"]
    by_path: dict[str, dict[str, Any]] = {}
    by_layer: dict[int, dict[str, Any]] = {}
    for site in metadata.get("sites", []):
        saved = states.get(site.get("key"))
        if saved is None:
            continue
        path = site.get("path")
        if path:
            if path in by_path and by_path[path] is not saved:
                # Several sites share one path (some discovery modes record ""
                # or reuse a name). Path matching could hand one layer's
                # trained weights to another, so only layer_index is trusted.
                by_path[path] = None
            elif path not in by_path:
                by_path[path] = saved
        if site.get("layer_index") is not None:
            by_layer[int(site["layer_index"])] = saved

    site_reports: list[dict[str, Any]] = []
    total_copied = 0
    total_fresh = 0
    for install in installs:
        saved = by_path.get(install.site.path)
        matched_by = "path" if saved is not None else None
        if saved is None:
            saved = by_layer.get(install.site.layer_index)
            matched_by = "layer_index" if saved is not None else None
        gate = install.new_gate
        merged: dict[str, Any] = {}
        copied: list[str] = []
        fresh: list[str] = []
        for name, value in gate.state_dict().items():
            saved_value = saved.get(name) if saved else None
            if isinstance(saved_value, torch.Tensor) and saved_value.shape == value.shape:
                merged[name] = saved_value.to(device=value.device, dtype=value.dtype)
                copied.append(name)
            else:
                merged[name] = value
                fresh.append(name)
        gate.load_state_dict(merged, strict=True)
        total_copied += len(copied)
        total_fresh += len(fresh)
        site_reports.append({
            "layer_index": install.site.layer_index,
            "path": install.site.path,
            "matched_parent_by": matched_by,
            "parameters_copied": copied,
            "parameters_fresh": fresh,
        })
    return {
        "checkpoint": str(source),
        "sites": site_reports,
        "n_parameters_copied": total_copied,
        "n_parameters_fresh": total_fresh,
    }
