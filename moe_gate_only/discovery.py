from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .datastructures import (
    GateCandidateReport,
    GateFactory,
    GateInstall,
    GateSite,
    _ForwardTrace,
    _ModuleTrace,
    _TensorObservation,
)
from .sniff import (
    _discover_hidden_size,
    _discover_vocab_size,
    _guess_hidden_size,
    _guess_vocab_size,
)
from .tree import _extract_layer_index, _resolve_parent_attr
from .verify import _forward_with_sample, _move_sample_to_device, _rollback, _verify_forward

_LAST_CANDIDATE_REPORT: List[GateCandidateReport] = []


# ---------------------------------------------------------------------------
# Discovery: find all MoE gates in any HF model
# ---------------------------------------------------------------------------
# Primary: structural classification by parameter shape + module type.
# Fallback: forward-pass with hooks for truly unknown models.
# Zero hardcoded config field names in the primary path.


def _discover_num_experts(model: nn.Module,
                          sample_input: Any) -> set[int]:
    """Run one forward pass with hooks to discover expert counts.

    Hooks every module, collects output tensors, and finds the dimension
    that appears consistently as the last dimension of ``nn.Linear``-like
    module outputs across layers.  That dimension is ``num_experts``.
    """
    device = next(model.parameters()).device
    output_dims: dict[int, int] = {}  # last_dim -> count of modules

    hooks = []

    def make_hook(mod_name):
        def hook(_, inp, out):
            if isinstance(out, torch.Tensor) and out.ndim >= 2:
                last = out.shape[-1]
                output_dims[last] = output_dims.get(last, 0) + 1
        return hook

    for name, mod in model.named_modules():
        if list(mod.children()):
            continue  # only leaf modules
        h = mod.register_forward_hook(make_hook(name))
        hooks.append(h)

    try:
        ctx = torch.no_grad()
        with ctx:
            _ = _forward_with_sample(model, sample_input)
    except Exception:
        pass

    for h in hooks:
        h.remove()

    # num_experts is the dimension that appears N times where N ≈ num_layers
    if not output_dims:
        return set()

    # Get hidden_size and vocab_size to exclude them
    hidden = _guess_hidden_size(model)
    vocab = _guess_vocab_size(model)

    candidates = set()
    for dim, count in output_dims.items():
        if dim in (1, 2, 3):
            continue  # too small to be expert count
        if dim == hidden or dim == vocab:
            continue
        candidates.add(dim)

    return candidates


def _extract_expert_counts(model: nn.Module) -> set[int]:
    """Discover num_experts purely from model parameters — zero config parsing.

    Strategy: scan all 2-D parameter tensors.  Find the dimension that
    appears consistently across layers, excluding hidden_size, vocab_size,
    and intermediate_size (discovered from the embedding / LM head).
    """
    # Discover hidden_size and vocab_size from parameters
    hidden_size = _discover_hidden_size(model)
    vocab_size = _discover_vocab_size(model)

    # Count occurrences, but only from gate/scorer modules (by class name)
    dim_counts: dict[int, int] = {}
    for _, module in model.named_modules():
        cls_name = type(module).__name__.lower()
        # Gate modules typically have these in their class name
        if not any(kw in cls_name for kw in
                   ("router", "gate", "topk", "gating")):
            continue
        for p in module.parameters(recurse=False):
            if p.ndim != 2:
                continue
            if hidden_size is not None and int(p.shape[-1]) != hidden_size:
                continue
            s0 = int(p.shape[0])
            if s0 in (1, 2, 3):
                continue
            if hidden_size is not None and s0 == hidden_size:
                continue
            if vocab_size is not None and s0 == vocab_size:
                continue
            dim_counts[s0] = dim_counts.get(s0, 0) + 1

    if not dim_counts:
        # Fallback to config scan
        return _extract_expert_counts_from_config(model)

    # Pick the dimension that appears most often (the gate weight's first dim)
    best_dim = max(dim_counts, key=dim_counts.get)
    return {best_dim}


def _extract_expert_counts_from_config(model: nn.Module) -> set[int]:
    """Fallback: scan model config for num_experts fields."""
    counts: set[int] = set()
    cfg = getattr(model, "config", None)
    if cfg is None:
        return counts
    for nc in ([cfg] + [getattr(cfg, a, None)
                        for a in ("text_config", "ffn_config")
                        if hasattr(cfg, a)]):
        if nc is None:
            continue
        for key in ("num_experts", "num_local_experts", "n_routed_experts",
                    "num_routed_experts", "moe_num_experts", "expert_count"):
            val = getattr(nc, key, None)
            if val is not None:
                try:
                    counts.add(int(val))
                except (TypeError, ValueError):
                    pass
    return counts


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def _observe_tensor(tensor: torch.Tensor) -> _TensorObservation:
    return _TensorObservation(
        shape=tuple(int(dim) for dim in tensor.shape),
        dtype=tensor.dtype,
        is_floating_point=tensor.is_floating_point(),
    )


def _trace_model_forward(model: nn.Module, sample_input: Any) -> _ForwardTrace:
    """Run a real forward and collect tensor shape evidence for every module."""
    traces: Dict[str, _ModuleTrace] = {}
    hooks = []

    def make_hook(path: str, module: nn.Module):
        traces[path] = _ModuleTrace(path=path, module_type=type(module).__name__)

        def hook(_, inputs, output):
            trace = traces[path]
            trace.input_tensors.extend(_observe_tensor(t) for t in _iter_tensors(inputs))
            trace.output_tensors.extend(_observe_tensor(t) for t in _iter_tensors(output))

        return hook

    for path, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(path, module)))

    try:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        with torch.no_grad():
            _forward_with_sample(model, _move_sample_to_device(sample_input, device))
        return _ForwardTrace(succeeded=True, traces=traces)
    except Exception as exc:
        return _ForwardTrace(succeeded=False, traces=traces, error=f"{type(exc).__name__}: {exc}")
    finally:
        for handle in hooks:
            handle.remove()


def _trace_has_output_dim(trace: Optional[_ModuleTrace], dim: int) -> bool:
    if trace is None:
        return False
    return any(
        obs.is_floating_point and len(obs.shape) >= 2 and obs.shape[-1] == dim
        for obs in trace.output_tensors
    )


def _trace_has_input_dim(trace: Optional[_ModuleTrace], dim: int) -> bool:
    if trace is None:
        return False
    return any(len(obs.shape) >= 1 and obs.shape[-1] == dim for obs in trace.input_tensors)


def _trace_has_topk_tuple(trace: Optional[_ModuleTrace], num_experts: int) -> bool:
    if trace is None:
        return False
    for obs in trace.output_tensors:
        if len(obs.shape) < 1:
            continue
        last = obs.shape[-1]
        if 0 < last < num_experts:
            return True
    return False


def _dynamic_expert_dims(model: nn.Module, trace: Optional[_ForwardTrace]) -> set[int]:
    """Infer likely expert-count dimensions from runtime and parameter evidence."""
    hidden_size = _discover_hidden_size(model)
    vocab_size = _discover_vocab_size(model)
    param_counts: dict[int, int] = {}
    output_counts: dict[int, int] = {}

    for _, module in model.named_modules():
        for param in module.parameters(recurse=False):
            if param.ndim != 2:
                continue
            out_dim = int(param.shape[0])
            in_dim = int(param.shape[-1])
            if out_dim <= 1:
                continue
            if hidden_size is not None and out_dim == hidden_size:
                continue
            if vocab_size is not None and out_dim == vocab_size:
                continue
            if hidden_size is not None and in_dim != hidden_size:
                continue
            param_counts[out_dim] = param_counts.get(out_dim, 0) + 1

    if trace is not None:
        for module_trace in trace.traces.values():
            for obs in module_trace.output_tensors:
                if not obs.is_floating_point or len(obs.shape) < 2:
                    continue
                dim = obs.shape[-1]
                if dim <= 1:
                    continue
                if hidden_size is not None and dim == hidden_size:
                    continue
                if vocab_size is not None and dim == vocab_size:
                    continue
                output_counts[dim] = output_counts.get(dim, 0) + 1

    dims: set[int] = set()
    for dim in set(param_counts) | set(output_counts):
        score = 2 * param_counts.get(dim, 0) + 3 * output_counts.get(dim, 0)
        if score >= 2:
            dims.add(dim)
    return dims


def _same_child_signature(children: list[nn.Module]) -> bool:
    if len(children) < 2:
        return False
    first = children[0]
    first_shapes = tuple(tuple(param.shape) for param in first.parameters())
    if not first_shapes:
        return False
    first_type = type(first)
    for child in children[1:]:
        if type(child) is not first_type:
            return False
        shapes = tuple(tuple(param.shape) for param in child.parameters())
        if shapes != first_shapes:
            return False
    return True


def _expert_bank_paths(model: nn.Module, expert_dims: set[int]) -> dict[int, list[str]]:
    """Find expert banks structurally: same-shaped ModuleLists or 3D expert tensors."""
    banks: dict[int, list[str]] = {dim: [] for dim in expert_dims}
    for path, module in model.named_modules():
        if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            children = list(module.children())
            dim = len(children)
            if dim in expert_dims and _same_child_signature(children):
                banks.setdefault(dim, []).append(path)
        for param in module.parameters(recurse=False):
            if param.ndim >= 3:
                dim = int(param.shape[0])
                if dim in expert_dims:
                    banks.setdefault(dim, []).append(path)
    return {dim: paths for dim, paths in banks.items() if paths}


def _path_distance(a: str, b: str) -> int:
    a_parts = a.split(".") if a else []
    b_parts = b.split(".") if b else []
    common = 0
    for left, right in zip(a_parts, b_parts):
        if left != right:
            break
        common += 1
    return (len(a_parts) - common) + (len(b_parts) - common)


def _find_linear_in_module(mod: nn.Module, num_experts: int | None = None) -> Optional[nn.Linear]:
    """Find an nn.Linear child inside *mod* with ``out_features == num_experts``.

    Scans ALL child modules (not a hardcoded attribute list).  If
    *num_experts* is ``None``, returns the first nn.Linear found.
    """
    if isinstance(mod, nn.Linear):
        if num_experts is None or mod.out_features == num_experts:
            return mod
        return None
    for child in mod.children():
        result = _find_linear_in_module(child, num_experts)
        if result is not None:
            return result
    return None


def _find_linear_owner_attr(
    mod: nn.Module,
    num_experts: int | None = None,
) -> Optional[Tuple[nn.Module, str, nn.Linear]]:
    """Find the owner and child name for a nested expert-sized ``nn.Linear``."""
    for child_name, child in mod.named_children():
        if isinstance(child, nn.Linear):
            if num_experts is None or child.out_features == num_experts:
                return mod, child_name, child
        found = _find_linear_owner_attr(child, num_experts)
        if found is not None:
            return found
    return None


def _has_parameter_weight(mod: nn.Module, num_experts: int | None = None) -> bool:
    """True if *mod* has a directly-owned 2-D ``nn.Parameter`` named ``weight``
    with first dimension matching *num_experts* (if given)."""
    weight = getattr(mod, "weight", None)
    if not isinstance(weight, nn.Parameter) or weight.ndim != 2:
        return False
    if num_experts is not None and weight.shape[0] != num_experts:
        return False
    return True


def _infer_shape_from_linear(linear: nn.Linear) -> Tuple[int, int]:
    return int(linear.in_features), int(linear.out_features)


def _infer_shape_from_parameter(param: nn.Parameter) -> Tuple[int, int]:
    return int(param.shape[-1]), int(param.shape[0])


_WEIGHT_ACCESS_CACHE: dict[int, bool] = {}


def _parent_accesses_weight_of_child(gate: nn.Module, child: nn.Module) -> bool:
    """Check if the parent module's forward method accesses .weight on child."""
    cls = type(gate)
    key = id(cls)
    if key in _WEIGHT_ACCESS_CACHE:
        return _WEIGHT_ACCESS_CACHE[key]
    try:
        src = __import__("inspect").getsource(cls.forward).lower()
        result = ".weight" in src
    except (OSError, TypeError):
        result = False
    _WEIGHT_ACCESS_CACHE[key] = result
    return result


def _classify_gate(gate: nn.Module, num_experts: int) -> Optional[Tuple[str, int, int]]:
    """Classify a gate module purely by structure."""
    # Bare nn.Linear (not a subclass with overridden forward)
    if isinstance(gate, nn.Linear) and type(gate).forward is nn.Linear.forward:
        if gate.out_features == num_experts:
            return "linear", *_infer_shape_from_linear(gate)
    # nn.Linear subclass with custom forward → parameter_gate
    if isinstance(gate, nn.Linear) and gate.out_features == num_experts:
        return "parameter_gate", *_infer_shape_from_linear(gate)
    # nn.Module wrapping an nn.Linear with expert-sized output
    inner = _find_linear_in_module(gate, num_experts)
    if inner is not None:
        # If parent accesses .weight on the child, treat as parameter_gate
        if _parent_accesses_weight_of_child(gate, inner):
            return "parameter_gate", *_infer_shape_from_linear(inner)
        return "wrapped_linear", *_infer_shape_from_linear(inner)
    # nn.Parameter weight with expert-sized shape
    if _has_parameter_weight(gate, num_experts):
        return "parameter_gate", *_infer_shape_from_parameter(gate.weight)
    return None


def find_moe_gates(
    model: nn.Module,
    sample_input: Optional[Any] = None,
    *,
    allow_generic: bool = True,
    validate_candidates: bool = True,
    top_k: Optional[int] = None,
) -> List[GateSite]:
    """Walk the model and discover every MoE gate/scorer module.

    Uses two strategies:

    1. **Primary**: scan modules whose class name or path indicates MoE.
       Falls back to name-based attribute scanning.
    2. **Fallback**: dynamic weight-shape scanning + forward-pass discovery
       for truly unknown models.
    """
    global _LAST_CANDIDATE_REPORT

    if sample_input is not None:
        sites, reports = _find_moe_gates_dynamic(
            model,
            sample_input,
            top_k=top_k,
            validate=validate_candidates,
        )
        _LAST_CANDIDATE_REPORT = reports
        sites.sort(key=lambda s: s.layer_index)
        return sites

    # Fallback-only path for cases without usable runtime evidence.
    sites = _find_moe_gates_primary(model)
    if not sites and allow_generic:
        dynamic_counts = (
            _discover_num_experts(model, sample_input)
            if sample_input is not None else set()
        )
        sites = _filter_moe_gates(
            model,
            _find_moe_gates_fallback(model, dynamic_counts or None),
        )
    sites.sort(key=lambda s: s.layer_index)
    return sites


def _find_moe_gates_primary(model: nn.Module) -> List[GateSite]:
    """Primary discovery: scan modules in MoE-like paths with MoE-like class names."""
    num_experts_set = _extract_expert_counts(model)
    if not num_experts_set:
        return []

    # Attribute names that HF commonly uses for gates
    gate_attr_names = ("gate", "router", "classifier")
    sites: List[GateSite] = []
    seen_ids: set[int] = set()

    for name, module in model.named_modules():
        block_cls = type(module).__name__.lower()
        # Must be in a MoE-related block
        if not any(kw in block_cls for kw in
                   ("moe", "sparse", "ffn", "feed_forward", "mlp",
                    "decoderlayer", "block_sparse")):
            continue

        for attr in gate_attr_names:
            gate = getattr(module, attr, None)
            if gate is None or id(gate) in seen_ids:
                continue

            for ne in num_experts_set:
                classified = _classify_gate(gate, ne)
                if classified is None:
                    continue
                pattern, model_dim, ne_found = classified
                seen_ids.add(id(gate))
                sites.append(
                    GateSite(
                        layer_index=_extract_layer_index(name),
                        block=module,
                        gate=gate,
                        gate_attr=attr,
                        model_dim=model_dim,
                        num_experts=ne_found,
                        pattern=pattern,
                    )
                )
                break  # one classification per gate

    sites.sort(key=lambda s: s.layer_index)
    return sites


def _find_moe_gates_fallback(
    model: nn.Module,
    num_experts_set: Optional[set[int]] = None,
) -> List[GateSite]:
    """Fallback: dynamic structural discovery for unknown models."""
    if num_experts_set is None:
        num_experts_set = _extract_expert_counts(model)
    if not num_experts_set:
        return []

    sites: List[GateSite] = []
    seen_ids: set[int] = set()

    for name, module in model.named_modules():
        mid = id(module)
        if mid in seen_ids:
            continue
        for ne in num_experts_set:
            classified = _classify_gate(module, ne)
            if classified is None:
                continue
            pattern, model_dim, ne_found = classified
            seen_ids.add(mid)
            parent = _find_parent_block(model, name, module, pattern)
            sites.append(
                GateSite(
                    layer_index=_extract_layer_index(name),
                    block=parent["block"],
                    gate=module,
                    gate_attr=parent["gate_attr"],
                    model_dim=model_dim,
                    num_experts=ne_found,
                    pattern=pattern,
                )
            )
            break

    sites.sort(key=lambda s: s.layer_index)
    return sites


def _filter_moe_gates(model: nn.Module,
                      candidates: List[GateSite]) -> List[GateSite]:
    """Remove gate candidates that are not in MoE blocks."""
    filtered: List[GateSite] = []
    for s in candidates:
        # Find the full path by locating which module has our gate as a child
        gate_path = None
        for name, mod in model.named_modules():
            for child_name, child in mod.named_children():
                if child is s.gate:
                    gate_path = f"{name}.{child_name}" if name else child_name
                    break
            if gate_path is not None:
                break

        if gate_path is None:
            continue

        path_lower = gate_path.lower()
        block_cls = type(s.block).__name__.lower()
        gate_cls = type(s.gate).__name__.lower()

        is_moe_block = any(kw in block_cls for kw in
                           ("moe", "sparse", "sparsemoe", "mixtral", "jetmoe"))
        is_moe_gate = any(kw in gate_cls for kw in
                          ("router", "gate", "topk", "moe"))
        is_moe_path = any(kw in path_lower for kw in
                          ("mlp", "ffn", "feed_forward", "moe",
                           "sparse_moe", "block_sparse_moe"))

        if is_moe_block or is_moe_gate or is_moe_path:
            filtered.append(s)

    return filtered


def _site_report(
    site: GateSite,
    *,
    accepted: bool,
    reason: str,
) -> GateCandidateReport:
    return GateCandidateReport(
        path=site.path,
        module_type=type(site.gate).__name__,
        pattern=site.pattern,
        model_dim=site.model_dim,
        num_experts=site.num_experts,
        score=site.score,
        accepted=accepted,
        reason=reason,
        evidence=site.evidence,
    )


def _paths_overlap(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def _make_direct_site(
    model: nn.Module,
    path: str,
    module: nn.Module,
    model_dim: int,
    num_experts: int,
    pattern: str,
) -> Optional[GateSite]:
    resolved = _resolve_parent_attr(model, path)
    if resolved is None:
        return None
    owner, attr = resolved
    return GateSite(
        layer_index=_extract_layer_index(path),
        block=owner,
        gate=module,
        gate_attr=attr,
        model_dim=model_dim,
        num_experts=num_experts,
        pattern=pattern,
        path=path,
    )


def _score_dynamic_site(
    site: GateSite,
    trace: Optional[_ForwardTrace],
    expert_dims: set[int],
    expert_banks: dict[int, list[str]],
) -> GateSite:
    module_trace = trace.traces.get(site.path) if trace is not None else None
    evidence: list[str] = []
    score = 0.0

    if site.num_experts in expert_dims:
        score += 2.0
        evidence.append("expert_dim_inferred")
    if _trace_has_input_dim(module_trace, site.model_dim):
        score += 2.0
        evidence.append("runtime_input_matches_hidden")
    if _trace_has_output_dim(module_trace, site.num_experts):
        score += 8.0
        evidence.append("runtime_output_matches_experts")
    if _trace_has_topk_tuple(module_trace, site.num_experts):
        score += 3.0
        evidence.append("runtime_output_contains_topk_shape")
    if module_trace is not None and module_trace.output_tensors:
        score += 1.0
        evidence.append("module_executed")
    if site.pattern == "linear":
        score += 1.0
        evidence.append("lowest_level_linear")
    if site.pattern == "parameter_gate" and _trace_has_output_dim(module_trace, site.num_experts):
        score += 1.0
        evidence.append("whole_module_returns_expert_scores")

    bank_paths = expert_banks.get(site.num_experts, [])
    if bank_paths:
        nearest = min(_path_distance(site.path, bank_path) for bank_path in bank_paths)
        if nearest <= 3:
            score += 8.0
            evidence.append("near_expert_bank")
        elif nearest <= 5:
            score += 1.0
            evidence.append("weakly_near_expert_bank")

    site.score = score
    site.evidence = tuple(evidence)
    return site


def _enumerate_dynamic_candidates(
    model: nn.Module,
    trace: Optional[_ForwardTrace],
) -> List[GateSite]:
    """Enumerate scorer candidates without gate/router attr or MoE class names."""
    expert_dims = _dynamic_expert_dims(model, trace)
    if not expert_dims:
        return []
    expert_banks = _expert_bank_paths(model, expert_dims)

    sites: list[GateSite] = []
    seen: set[tuple[str, str]] = set()

    def add(site: Optional[GateSite]) -> None:
        if site is None:
            return
        key = (site.path, site.pattern)
        if key in seen:
            return
        seen.add(key)
        sites.append(_score_dynamic_site(site, trace, expert_dims, expert_banks))

    for path, module in model.named_modules():
        if not path:
            continue

        if isinstance(module, nn.Linear) and int(module.out_features) in expert_dims:
            pattern = "linear" if type(module).forward is nn.Linear.forward else "parameter_gate"
            add(_make_direct_site(
                model,
                path,
                module,
                int(module.in_features),
                int(module.out_features),
                pattern,
            ))

        weight = getattr(module, "weight", None)
        if (
            not isinstance(module, nn.Linear)
            and isinstance(weight, nn.Parameter)
            and weight.ndim == 2
            and int(weight.shape[0]) in expert_dims
        ):
            add(_make_direct_site(
                model,
                path,
                module,
                int(weight.shape[-1]),
                int(weight.shape[0]),
                "parameter_gate",
            ))

        if isinstance(module, (nn.Linear, nn.ModuleList, nn.ModuleDict, nn.Sequential)):
            continue
        inner = _find_linear_in_module(module)
        if inner is not None and int(inner.out_features) in expert_dims:
            add(_make_direct_site(
                model,
                path,
                module,
                int(inner.in_features),
                int(inner.out_features),
                "parameter_gate",
            ))

    sites.sort(key=lambda s: (s.score, len(s.path.split("."))), reverse=True)
    return sites


def _probe_gate_site(
    model: nn.Module,
    site: GateSite,
    factory: GateFactory,
    sample_input: Any,
    top_k: Optional[int],
) -> tuple[bool, str]:
    # Imported lazily to avoid a discovery <-> install module-level cycle.
    from .install import _install_gate_site, _new_gate_for_site

    install: Optional[GateInstall] = None
    try:
        install = _install_gate_site(site, _new_gate_for_site(factory, site), top_k)
        if _verify_forward(model, [install], sample_input=sample_input):
            return True, "validated_by_transactional_forward"
        return False, "forward_verification_failed"
    except Exception as exc:
        return False, f"probe_failed: {type(exc).__name__}: {exc}"
    finally:
        if install is not None:
            _rollback([install])


def _find_moe_gates_dynamic(
    model: nn.Module,
    sample_input: Any,
    *,
    factory: Optional[GateFactory] = None,
    top_k: Optional[int] = None,
    validate: bool = True,
) -> tuple[List[GateSite], List[GateCandidateReport]]:
    trace = _trace_model_forward(model, sample_input)
    if not trace.succeeded:
        return [], [GateCandidateReport(
            path="<model>",
            module_type=type(model).__name__,
            pattern="runtime_trace",
            model_dim=None,
            num_experts=None,
            score=0.0,
            accepted=False,
            reason=f"sample_input_forward_failed: {trace.error}",
        )]

    candidates = _enumerate_dynamic_candidates(model, trace)
    reports: list[GateCandidateReport] = []
    accepted: list[GateSite] = []
    probe_factory = factory or (lambda d, e: nn.Linear(d, e, bias=False))

    for site in candidates:
        if site.score < 5.0:
            reports.append(_site_report(site, accepted=False, reason="insufficient_dynamic_evidence"))
            continue
        if "near_expert_bank" not in site.evidence:
            reports.append(_site_report(site, accepted=False, reason="not_near_structural_expert_bank"))
            continue
        if any(_paths_overlap(site.path, existing.path) for existing in accepted):
            reports.append(_site_report(site, accepted=False, reason="overlaps_higher_ranked_candidate"))
            continue
        if validate:
            ok, reason = _probe_gate_site(model, site, probe_factory, sample_input, top_k)
        else:
            ok, reason = True, "accepted_without_probe"
        reports.append(_site_report(site, accepted=ok, reason=reason))
        if ok:
            accepted.append(site)

    accepted.sort(key=lambda s: s.layer_index)
    return accepted, reports


def get_gate_candidate_report() -> List[GateCandidateReport]:
    """Return the report from the most recent dynamic discovery/install call."""
    return list(_LAST_CANDIDATE_REPORT)


def explain_gate_candidates(
    model: nn.Module,
    sample_input: Any,
    *,
    validate: bool = True,
    top_k: Optional[int] = None,
) -> List[GateCandidateReport]:
    """Trace, enumerate, and optionally probe dynamic gate candidates."""
    global _LAST_CANDIDATE_REPORT
    _, reports = _find_moe_gates_dynamic(
        model,
        sample_input,
        top_k=top_k,
        validate=validate,
    )
    _LAST_CANDIDATE_REPORT = reports
    return reports
