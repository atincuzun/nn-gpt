from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

GateFactory = Callable[[int, int], nn.Module]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GateSite:
    """A discovered MoE gate/scorer location inside a model."""

    layer_index: int
    block: nn.Module
    gate: nn.Module
    gate_attr: str
    model_dim: int
    num_experts: int
    pattern: str  # "linear", "wrapped_linear", "parameter_gate"
    path: str = ""
    score: float = 0.0
    evidence: Tuple[str, ...] = ()


@dataclass
class GateInstall:
    """Record of a single gate replacement."""

    site: GateSite
    old_gate: nn.Module
    new_gate: nn.Module
    owner: Optional[nn.Module] = None
    attr: Optional[str] = None
    old_child: Optional[nn.Module] = None
    hook_handles: List[Any] = field(default_factory=list)
    mode: str = "direct"
    teacher_gate: Optional[nn.Module] = None
    student_gate: Optional[nn.Module] = None
    teacher_requires_grad: Optional[Tuple[bool, ...]] = None


@dataclass
class GateCandidateReport:
    """Explanation for an accepted or rejected dynamic gate candidate."""

    path: str
    module_type: str
    pattern: str
    model_dim: Optional[int]
    num_experts: Optional[int]
    score: float
    accepted: bool
    reason: str
    evidence: Tuple[str, ...] = ()


@dataclass
class _TensorObservation:
    shape: Tuple[int, ...]
    dtype: torch.dtype
    is_floating_point: bool


@dataclass
class _ModuleTrace:
    path: str
    module_type: str
    input_tensors: List[_TensorObservation] = field(default_factory=list)
    output_tensors: List[_TensorObservation] = field(default_factory=list)


@dataclass
class _ForwardTrace:
    succeeded: bool
    traces: Dict[str, _ModuleTrace]
    error: Optional[str] = None


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


def _module_path_map(model: nn.Module) -> Dict[int, str]:
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


# ---------------------------------------------------------------------------
# Wrappers for Pattern C (nn.Parameter + F.linear) gates
# ---------------------------------------------------------------------------


def _copy_gate_attrs(old_gate: nn.Module, wrapper: nn.Module) -> None:
    """Copy ALL scalar/array attributes and buffers from *old_gate* to *wrapper*.

    Copies every instance attribute from ``old_gate.__dict__`` that is NOT a
    submodule, parameter, buffer, or private member.  This replaces the
    hardcoded attribute list — any config attribute the original gate exposes
    (``top_k``, ``num_experts``, ``e_score_correction_bias``, custom fields,
    etc.) is automatically preserved.
    """
    # Register all buffers from the old gate on the wrapper
    for name, buf in old_gate.named_buffers():
        if "." in name:
            continue  # dots not allowed in buffer names
        if not hasattr(wrapper, name):
            try:
                wrapper.register_buffer(name, buf.clone())
            except (KeyError, RuntimeError):
                pass

    # Copy all non-module, non-parameter, non-buffer attributes
    import torch.nn as _nn
    _exclude = {*old_gate._parameters, *old_gate._modules, *old_gate._buffers}
    for attr, value in old_gate.__dict__.items():
        if attr.startswith("_"):
            continue
        if attr in _exclude:
            continue
        if isinstance(value, (_nn.Module, _nn.Parameter)):
            continue
        if isinstance(value, torch.Tensor):
            continue  # tensors are handled as buffers above
        # Don't overwrite the wrapper's own attributes
        if hasattr(wrapper, attr) and attr not in ("training",):
            continue
        try:
            setattr(wrapper, attr, value)
        except (AttributeError, TypeError):
            pass


class _LogitsOnlyGate(nn.Module):
    """Wraps a new gate to return only logits, matching raw-logits gates."""

    def __init__(self, new_gate: nn.Module):
        super().__init__()
        self.gate = new_gate

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.gate(x)


class _SoftmaxTopKGate(nn.Module):
    """Wraps a new gate for models that expect (logits, scores, indices)."""

    def __init__(
        self,
        new_gate: nn.Module,
        top_k: int,
        normalize: bool = True,
        order: str = "logits_weights_indices",
    ):
        super().__init__()
        self.gate = new_gate
        self.top_k = top_k
        self.normalize = normalize
        self.order = order

    def forward(self, x: torch.Tensor, *args, **kwargs):
        logits = self.gate(x)
        scores = F.softmax(logits, dim=-1, dtype=torch.float)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        if self.normalize:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        weights = weights.to(x.dtype)
        if self.order == "logits_indices_weights":
            return logits, indices, weights
        return logits, weights, indices


class _TopKWeightsIndicesGate(nn.Module):
    """Wraps a new gate for routers that return only (weights, indices)."""

    def __init__(self, new_gate: nn.Module, top_k: int):
        super().__init__()
        self.gate = new_gate
        self.top_k = top_k
        self._last_gate_logits: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, *args, **kwargs):
        logits = self.gate(x)
        self._last_gate_logits = logits
        scores = F.softmax(logits, dim=-1, dtype=torch.float)
        choice_scores = scores
        bias = getattr(self, "e_score_correction_bias", None)
        if isinstance(bias, torch.Tensor) and bias.shape[-1] == logits.shape[-1]:
            choice_scores = choice_scores + bias.to(device=scores.device, dtype=scores.dtype)
        _, indices = torch.topk(choice_scores, self.top_k, dim=-1, sorted=False)
        weights = scores.gather(dim=-1, index=indices)
        scaling = getattr(self, "routed_scaling_factor", 1.0)
        weights = weights * scaling
        return weights.to(x.dtype), indices


class _DeepSeekV2Gate(nn.Module):
    """Preserve the remote-code DeepSeek-V2 ``MoEGate`` return contract."""

    def __init__(self, new_gate: nn.Module, top_k: int):
        super().__init__()
        self.gate = new_gate
        self.top_k = top_k
        self._last_topk_idx: Optional[torch.Tensor] = None
        self._last_topk_weight: Optional[torch.Tensor] = None
        self._last_aux_loss: Optional[torch.Tensor] = None
        self._last_gate_logits: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, gate_dtype = _module_device_dtype(self.gate)
        gate_input = flat.to(dtype=gate_dtype) if gate_dtype is not None else flat
        logits = self.gate(gate_input).float()
        return self._route_from_logits(hidden_states, logits)

    def _route_from_logits(self, hidden_states: torch.Tensor, logits: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        self._last_gate_logits = logits
        scoring_func = getattr(self, "scoring_func", "softmax")
        if scoring_func != "softmax":
            raise NotImplementedError(f"Unsupported DeepSeek-V2 scoring function: {scoring_func}")
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        topk_method = getattr(self, "topk_method", "greedy")
        if topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1, sorted=False)
        elif topk_method == "group_limited_greedy":
            n_group = int(getattr(self, "n_group"))
            topk_group = int(getattr(self, "topk_group"))
            group_scores = scores.view(flat.shape[0], n_group, -1).max(dim=-1).values
            group_idx = torch.topk(group_scores, topk_group, dim=-1, sorted=False).indices
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            group_mask.scatter_(1, group_idx, True)
            score_mask = group_mask.unsqueeze(-1).expand(
                flat.shape[0], n_group, scores.shape[-1] // n_group
            ).reshape(flat.shape[0], -1)
            choice_scores = scores.masked_fill(~score_mask, 0.0)
            topk_weight, topk_idx = torch.topk(
                choice_scores, self.top_k, dim=-1, sorted=False
            )
        else:
            raise NotImplementedError(f"Unsupported DeepSeek-V2 top-k method: {topk_method}")

        if self.top_k > 1 and bool(getattr(self, "norm_topk_prob", False)):
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            topk_weight = topk_weight * float(getattr(self, "routed_scaling_factor", 1.0))

        alpha = float(getattr(self, "alpha", 0.0))
        aux_loss = None
        if self.training and alpha > 0.0:
            topk_idx_by_batch = topk_idx.view(batch_size, -1)
            if bool(getattr(self, "seq_aux", False)):
                scores_by_sequence = scores.view(batch_size, sequence_length, -1)
                counts = torch.zeros(
                    batch_size,
                    scores.shape[-1],
                    device=hidden_states.device,
                )
                counts.scatter_add_(
                    1,
                    topk_idx_by_batch,
                    torch.ones(
                        batch_size,
                        sequence_length * self.top_k,
                        device=hidden_states.device,
                    ),
                ).div_(sequence_length * self.top_k / scores.shape[-1])
                aux_loss = (counts * scores_by_sequence.mean(dim=1)).sum(dim=1).mean() * alpha
            else:
                assignment = F.one_hot(
                    topk_idx_by_batch.reshape(-1), num_classes=scores.shape[-1]
                ).float().mean(0)
                aux_loss = (scores.mean(0) * assignment * scores.shape[-1]).sum() * alpha
        self._last_topk_idx = topk_idx.detach()
        self._last_topk_weight = topk_weight.detach()
        self._last_aux_loss = aux_loss.detach() if aux_loss is not None else None
        return topk_idx, topk_weight, aux_loss


class _DeepSeekV2TeacherStudentGate(_DeepSeekV2Gate):
    """Train a random student while the frozen native gate controls startup."""

    def __init__(
        self,
        teacher_gate: nn.Module,
        student_gate: nn.Module,
        top_k: int,
        *,
        student_weight: float = 0.0,
        distillation_temperature: float = 1.0,
    ) -> None:
        super().__init__(student_gate, top_k)
        if not 0.0 <= student_weight <= 1.0:
            raise ValueError("student_weight must be between 0 and 1")
        if distillation_temperature <= 0:
            raise ValueError("distillation_temperature must be positive")
        self.teacher = teacher_gate
        student_device, _ = _module_device_dtype(student_gate)
        self.register_buffer(
            "_student_weight_tensor",
            torch.tensor(
                float(student_weight),
                dtype=torch.float32,
                device=student_device or torch.device("cpu"),
            ),
            persistent=False,
        )
        self.distillation_temperature = float(distillation_temperature)
        self._last_teacher_logits: Optional[torch.Tensor] = None
        self._last_student_logits: Optional[torch.Tensor] = None
        self._last_teacher_topk_idx: Optional[torch.Tensor] = None
        self._last_student_topk_idx: Optional[torch.Tensor] = None
        self._last_distillation_loss: Optional[torch.Tensor] = None
        self._last_distillation_value: Optional[float] = None
        self.reset_teacher_student_metrics()

    @property
    def student(self) -> nn.Module:
        return self.gate

    @property
    def student_weight(self) -> float:
        return float(self._student_weight_tensor.item())

    def set_student_weight(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("student_weight must be between 0 and 1")
        self._student_weight_tensor.fill_(float(value))

    def reset_teacher_student_metrics(self) -> None:
        self._metric_tokens = 0
        self._metric_topk_matches = 0.0
        self._metric_topk_total = 0
        self._metric_kl_weighted_sum = 0.0

    def _teacher_logits(self, flat: torch.Tensor) -> torch.Tensor:
        weight = getattr(self.teacher, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise TypeError("DeepSeek teacher gate must expose a weight tensor")
        return F.linear(flat.float(), weight.detach().float())

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        teacher_logits = self._teacher_logits(flat)
        _, student_dtype = _module_device_dtype(self.student)
        student_input = flat.to(dtype=student_dtype) if student_dtype is not None else flat
        student_logits = self.student(student_input).float()
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                "Student router logits do not match teacher shape: "
                f"{tuple(student_logits.shape)} != {tuple(teacher_logits.shape)}"
            )
        if not torch.isfinite(student_logits).all():
            raise ValueError("Student router produced non-finite logits")

        temperature = self.distillation_temperature
        teacher_probabilities = (teacher_logits / temperature).softmax(dim=-1).detach()
        student_log_probabilities = (student_logits / temperature).log_softmax(dim=-1)
        self._last_distillation_loss = F.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="batchmean",
        ) * (temperature * temperature)
        self._last_distillation_value = float(self._last_distillation_loss.detach().item())

        with torch.no_grad():
            teacher_scores = teacher_logits.softmax(dim=-1, dtype=torch.float32)
            student_scores = student_logits.softmax(dim=-1, dtype=torch.float32)
            self._last_teacher_topk_idx = torch.topk(
                teacher_scores, self.top_k, dim=-1, sorted=False
            ).indices
            self._last_student_topk_idx = torch.topk(
                student_scores, self.top_k, dim=-1, sorted=False
            ).indices
            self._last_teacher_logits = teacher_logits.detach()
            self._last_student_logits = student_logits.detach()
            matches = (
                self._last_teacher_topk_idx.unsqueeze(-1)
                == self._last_student_topk_idx.unsqueeze(-2)
            ).any(dim=-1)
            tokens = int(teacher_logits.shape[0])
            self._metric_tokens += tokens
            self._metric_topk_matches += float(matches.sum().item())
            self._metric_topk_total += int(matches.numel())
            self._metric_kl_weighted_sum += self._last_distillation_value * tokens

        if self.student_weight == 0.0:
            output = self.teacher(hidden_states, *args, **kwargs)
            if not isinstance(output, (tuple, list)) or len(output) < 3:
                raise TypeError("DeepSeek teacher gate returned an unexpected routing contract")
            self._last_gate_logits = teacher_logits
            self._last_topk_idx = output[0].detach()
            self._last_topk_weight = output[1].detach()
            aux_loss = output[2]
            self._last_aux_loss = aux_loss.detach() if isinstance(aux_loss, torch.Tensor) else None
            return output

        combined_logits = torch.lerp(teacher_logits, student_logits, self.student_weight)
        return self._route_from_logits(hidden_states, combined_logits)


class _ScoreFnTopKGate(nn.Module):
    """Wraps routers that score logits with a custom score_fn before top-k."""

    def __init__(self, new_gate: nn.Module, top_k: int):
        super().__init__()
        self.gate = new_gate
        self.top_k = top_k

    def _score(self, logits: torch.Tensor) -> torch.Tensor:
        score_fn = getattr(self, "score_fn", None)
        if callable(score_fn):
            return score_fn(logits)
        return F.softmax(logits, dim=-1, dtype=torch.float)

    def _flatten(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_dim = int(getattr(self, "hidden_dim", hidden_states.shape[-1]))
        return hidden_states.reshape(-1, hidden_dim)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        flat = self._flatten(hidden_states)
        logits = self.gate(flat)
        scores = self._score(logits)
        choice_scores = scores
        bias = getattr(self, "e_score_correction_bias", None)
        if isinstance(bias, torch.Tensor) and bias.shape[-1] == scores.shape[-1]:
            choice_scores = choice_scores + bias.to(device=scores.device, dtype=scores.dtype)
        indices = torch.topk(choice_scores, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        scaling = getattr(self, "routed_scaling_factor", 1.0)
        return logits, weights.to(logits.dtype) * scaling, indices


class _HashScoreFnGate(_ScoreFnTopKGate):
    """Wraps hash-MoE routers that select experts from input_ids."""

    def forward(self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None, *args, **kwargs):
        if input_ids is None:
            input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Hash router replacement requires input_ids")

        flat = self._flatten(hidden_states)
        logits = self.gate(flat)
        scores = self._score(logits)
        tid2eid = getattr(self, "tid2eid", None)
        if not isinstance(tid2eid, torch.Tensor):
            raise RuntimeError("Hash router replacement is missing tid2eid buffer")
        indices = tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        scaling = getattr(self, "routed_scaling_factor", 1.0)
        return logits, weights.to(logits.dtype) * scaling, indices


class _SigmoidTopKGate(nn.Module):
    """Wraps a new gate for models that expect (logits, scores, indices) with sigmoid."""

    def __init__(self, new_gate: nn.Module, top_k: int):
        super().__init__()
        self.gate = new_gate
        self.top_k = top_k

    def forward(self, x: torch.Tensor, *args, **kwargs):
        logits = self.gate(x)
        scores = torch.sigmoid(logits)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        return logits, weights.to(x.dtype), indices


class _SparseSigmoidGate(nn.Module):
    """Wraps a new gate for Llama4-style sparse sigmoid routing.

    Returns ``(sparse_scores, logits)`` where scores are top-k logits
    scattered into ``-inf`` then sigmoid-activated — matching the
    ``Llama4Router`` contract.
    """

    def __init__(self, new_gate: nn.Module, top_k: int):
        super().__init__()
        self.gate = new_gate
        self.top_k = top_k

    def forward(self, x: torch.Tensor, *args, **kwargs):
        logits = self.gate(x)
        top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)
        sparse = torch.full_like(logits, float("-inf"))
        sparse.scatter_(dim=-1, index=top_idx, src=top_vals)
        scores = torch.sigmoid(sparse.float()).to(logits.dtype)
        return scores, logits


_POST_PROCESSING_CACHE: dict[int, str] = {}


def _detect_post_processing(gate: nn.Module) -> str:
    """Detect what post-processing the original gate does after the linear
    projection.

    Tries, in order:
    1. **Runtime**: run the gate on a dummy input and observe the output
       structure (no source code needed).
    2. **Source**: inspect the source code for keywords.

    Returns ``"logits_only"``, ``"softmax_topk"``, ``"logits_indices_weights"``,
    ``"sigmoid_topk"``, ``"sparse_sigmoid"``, or ``"topk_weights_indices"``.
    """
    cls = type(gate)
    key = id(cls)
    cached = _POST_PROCESSING_CACHE.get(key)
    if cached is not None:
        return cached

    # DeepSeek-V2 remote-code routers return (indices, weights, aux_loss) and
    # optionally perform group-limited expert selection.
    if (
        hasattr(gate, "n_routed_experts")
        and hasattr(gate, "topk_method")
        and hasattr(gate, "scoring_func")
        and hasattr(gate, "seq_aux")
    ):
        result = "deepseek_v2"
        _POST_PROCESSING_CACHE[key] = result
        return result

    # DeepSeek-V4-style routers expose the routing activation as a callable
    # score_fn plus a routed_scaling_factor.  Hash-MoE variants also carry a
    # tid2eid buffer and require input_ids, so they need a distinct wrapper.
    if hasattr(gate, "score_fn") and hasattr(gate, "routed_scaling_factor"):
        result = "hash_scorefn_topk" if isinstance(getattr(gate, "tid2eid", None), torch.Tensor) else "scorefn_topk"
        _POST_PROCESSING_CACHE[key] = result
        return result

    # 1. Try runtime detection
    model_dim = _get_gate_input_dim(gate)
    if model_dim is not None:
        runtime = _trace_gate_contract(gate, model_dim)
        if runtime == "sparse_sigmoid":
            _POST_PROCESSING_CACHE[key] = runtime
            return runtime
        # For softmax_topk, we still check source to distinguish softmax vs sigmoid
        if runtime != "logits_only":
            # Try source to narrow softmax vs sigmoid
            src_result = _detect_from_source(gate)
            if src_result:
                _POST_PROCESSING_CACHE[key] = src_result
                return src_result
            _POST_PROCESSING_CACHE[key] = runtime
            return runtime
        # runtime said logits_only, but source might know better
        src_result = _detect_from_source(gate)
        result = src_result or runtime
        _POST_PROCESSING_CACHE[key] = result
        return result

    # 2. Fallback to source inspection
    result = _detect_from_source(gate) or "logits_only"
    _POST_PROCESSING_CACHE[key] = result
    return result


def _get_gate_input_dim(gate: nn.Module) -> int | None:
    """Get the expected input dimension of a gate module from its parameters."""
    for p in gate.parameters():
        if p.ndim >= 2:
            return int(p.shape[-1])
    if isinstance(gate, nn.Linear):
        return int(gate.in_features)
    for child in gate.children():
        if isinstance(child, nn.Linear):
            return int(child.in_features)
    return None


def _detect_from_source(gate: nn.Module) -> str | None:
    """Fallback: inspect source code for routing keywords."""
    try:
        src = __import__("inspect").getsource(type(gate).forward).lower()
    except (OSError, TypeError):
        return None

    has_softmax = "softmax" in src
    has_sigmoid = "sigmoid" in src
    has_topk = "topk" in src
    has_scatter = "scatter" in src

    if has_sigmoid and has_topk and has_scatter:
        return "sparse_sigmoid"
    if has_sigmoid and has_topk:
        return "sigmoid_topk"
    if has_softmax and has_topk:
        return "softmax_topk"
    if has_topk:
        return "softmax_topk"
    if has_sigmoid and not has_topk:
        return "sigmoid_topk"
    return None


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def assert_hf_native_model(
    model: nn.Module,
    *,
    allow_remote_code: bool = False,
    require_hf_native: bool = False,
) -> None:
    """Reject unsafe HF remote-code modules unless explicitly allowed.

    ``transformers_modules.*`` classes are created by HuggingFace remote-code
    loading.  Gate surgery can still work on them, but it is not HF-native and
    should be an explicit choice.  ``require_hf_native`` additionally rejects
    non-``transformers.models.*`` root model classes.
    """
    root = type(model).__module__
    if require_hf_native and not root.startswith("transformers.models."):
        raise RuntimeError(f"Root model is not HF-native: {root}")

    if allow_remote_code:
        return

    bad: list[tuple[str, str, str]] = []
    for name, mod in model.named_modules():
        path = type(mod).__module__
        if path.startswith("transformers_modules."):
            bad.append((name, type(mod).__name__, path))

    if bad:
        preview = ", ".join(
            f"{name or '<root>'}:{cls}({path})"
            for name, cls, path in bad[:20]
        )
        raise RuntimeError(
            "Remote-code modules found. Pass allow_remote_code=True to "
            f"install_gates() if this is intentional: {preview}"
        )


def _module_device_dtype(module: nn.Module) -> Tuple[Optional[torch.device], Optional[torch.dtype]]:
    for param in module.parameters(recurse=True):
        return param.device, param.dtype
    for buf in module.buffers(recurse=True):
        return buf.device, buf.dtype
    return None, None


def _wrap_gate_for_contract(
    site: GateSite,
    new_gate: nn.Module,
    top_k: Optional[int],
) -> nn.Module:
    post = _detect_post_processing(site.gate)
    if post == "softmax_topk":
        return _SoftmaxTopKGate(
            new_gate,
            top_k=_resolve_top_k(site.gate, site.block, site.num_experts, top_k),
        )
    if post == "sigmoid_topk":
        return _SigmoidTopKGate(
            new_gate,
            top_k=_resolve_top_k(site.gate, site.block, site.num_experts, top_k),
        )
    if post == "sparse_sigmoid":
        return _SparseSigmoidGate(
            new_gate,
            top_k=_resolve_top_k(site.gate, site.block, site.num_experts, top_k),
        )
    if post == "topk_weights_indices":
        return _TopKWeightsIndicesGate(
            new_gate,
            top_k=_resolve_top_k(site.gate, site.block, site.num_experts, top_k),
        )
    if post == "deepseek_v2":
        return _DeepSeekV2Gate(
            new_gate,
            top_k=_resolve_top_k(site.gate, site.block, site.num_experts, top_k),
        )
    if post == "logits_indices_weights":
        return _SoftmaxTopKGate(
            new_gate,
            top_k=_resolve_top_k(site.gate, site.block, site.num_experts, top_k),
            order="logits_indices_weights",
        )
    if post == "scorefn_topk":
        return _ScoreFnTopKGate(
            new_gate,
            top_k=_resolve_top_k(site.gate, site.block, site.num_experts, top_k),
        )
    if post == "hash_scorefn_topk":
        return _HashScoreFnGate(
            new_gate,
            top_k=_resolve_top_k(site.gate, site.block, site.num_experts, top_k),
        )
    return _LogitsOnlyGate(new_gate)


def _new_gate_for_site(factory: GateFactory, site: GateSite) -> nn.Module:
    new_gate = factory(site.model_dim, site.num_experts)
    device, dtype = _module_device_dtype(site.gate)
    if device is not None:
        new_gate = new_gate.to(device=device)
    # DeepSeek-V2 computes its native router projection in float32 even when
    # router weights and hidden states are bfloat16. Preserve that behavior.
    if (
        dtype is not None
        and dtype.is_floating_point
        and _detect_post_processing(site.gate) != "deepseek_v2"
    ):
        new_gate = new_gate.to(dtype=dtype)
    return new_gate


def _initialize_from_original_projection(new_gate: nn.Module, site: GateSite) -> None:
    """Initialize a replacement gate to reproduce the native router projection."""
    source = getattr(site.gate, "weight", None)
    if not isinstance(source, torch.Tensor):
        raise ValueError("Original-weight initialization requires a native weight tensor")

    initializer = getattr(new_gate, "initialize_from_projection", None)
    if callable(initializer):
        bias = getattr(site.gate, "bias", None)
        initializer(source, bias=bias if isinstance(bias, torch.Tensor) else None)
        return

    base = getattr(new_gate, "base", None)
    target = getattr(base, "weight", None)
    if not isinstance(target, torch.Tensor):
        raise ValueError(
            "Original-weight initialization requires initialize_from_projection(weight) "
            "or base = nn.Linear(model_dim, num_experts, bias=False)"
        )
    if source.shape != target.shape:
        raise ValueError(
            f"Generated base projection shape {tuple(target.shape)} does not match "
            f"native router shape {tuple(source.shape)}"
        )
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))
        bias = getattr(base, "bias", None)
        if isinstance(bias, torch.Tensor):
            bias.zero_()


def _install_gate_site(site: GateSite, new_gate: nn.Module, top_k: Optional[int]) -> GateInstall:
    owner: nn.Module = site.block
    attr = site.gate_attr
    old_child: nn.Module = site.gate
    installed_gate: nn.Module = new_gate

    if site.pattern == "linear":
        _assign_child(site.block, site.gate_attr, new_gate)

    elif site.pattern == "wrapped_linear":
        found = _find_linear_owner_attr(site.gate, site.num_experts)
        if found is not None:
            owner, attr, old_child = found
            _assign_child(owner, attr, new_gate)
        else:
            wrapped = _wrap_gate_for_contract(site, new_gate, top_k)
            _copy_gate_attrs(site.gate, wrapped)
            wrapped.train(site.gate.training)
            _assign_child(site.block, site.gate_attr, wrapped)
            installed_gate = wrapped

    elif site.pattern == "parameter_gate":
        wrapped = _wrap_gate_for_contract(site, new_gate, top_k)
        _copy_gate_attrs(site.gate, wrapped)
        wrapped.train(site.gate.training)
        _assign_child(site.block, site.gate_attr, wrapped)
        installed_gate = wrapped

    else:
        raise RuntimeError(f"Unsupported gate pattern: {site.pattern}")

    return GateInstall(
        site=site,
        old_gate=site.gate,
        new_gate=installed_gate,
        owner=owner,
        attr=attr,
        old_child=old_child,
    )


def _install_teacher_student_site(
    site: GateSite,
    student_gate: nn.Module,
    top_k: Optional[int],
    student_weight: float,
    distillation_temperature: float,
) -> GateInstall:
    if site.pattern != "parameter_gate" or _detect_post_processing(site.gate) != "deepseek_v2":
        raise ValueError(
            "Teacher-student mode currently supports DeepSeek-V2 parameter gates only"
        )
    teacher_requires_grad = tuple(
        parameter.requires_grad for parameter in site.gate.parameters()
    )
    try:
        wrapper = _DeepSeekV2TeacherStudentGate(
            site.gate,
            student_gate,
            top_k=_resolve_top_k(site.gate, site.block, site.num_experts, top_k),
            student_weight=student_weight,
            distillation_temperature=distillation_temperature,
        )
        _copy_gate_attrs(site.gate, wrapper)
        wrapper.train(site.gate.training)
        for parameter in site.gate.parameters():
            parameter.requires_grad_(False)
        _assign_child(site.block, site.gate_attr, wrapper)
    except Exception:
        for parameter, requires_grad in zip(site.gate.parameters(), teacher_requires_grad):
            parameter.requires_grad_(requires_grad)
        raise
    return GateInstall(
        site=site,
        old_gate=site.gate,
        new_gate=wrapper,
        owner=site.block,
        attr=site.gate_attr,
        old_child=site.gate,
        mode="teacher_student",
        teacher_gate=site.gate,
        student_gate=student_gate,
        teacher_requires_grad=teacher_requires_grad,
    )


def install_gates(
    model: nn.Module,
    gate: str | GateFactory,
    *,
    layers: Optional[Iterable[int]] = None,
    gate_registry: Optional[dict[str, GateFactory]] = None,
    verify: bool = True,
    sample_input: Optional[Any] = None,
    top_k: Optional[int] = None,
    allow_remote_code: bool = False,
    require_hf_native: bool = False,
    allow_generic: bool = True,
    dynamic_discovery: bool = True,
    initialize_from_original: bool = False,
    teacher_student: bool = False,
    student_weight: float = 0.0,
    distillation_temperature: float = 1.0,
) -> List[GateInstall]:
    """Replace MoE gate/scorer modules across any HF MoE model.

    Parameters
    ----------
    model:
        A loaded HuggingFace (or compatible) model with MoE layers.
    gate:
        Either a gate name (looked up in ``gate_registry``) or a callable
        ``(model_dim, num_experts) -> nn.Module``.
    layers:
        Optional set/sequence of layer indices to replace.  ``None`` means all.
    gate_registry:
        Mapping of gate name -> factory.  Defaults to :data:`GATE_FACTORIES`
        from ``gates.py`` if available.
    verify:
        Run a forward pass after replacement.  On failure, all replacements
        are rolled back and ``RuntimeError`` is raised.
    sample_input:
        Optional tensor, tuple/list, or kwargs dict used for dynamic discovery
        and verify-forward.
    top_k:
        Optional explicit router ``top_k`` for gates whose original contract
        cannot expose it dynamically.
    allow_remote_code:
        Allow models containing ``transformers_modules.*`` remote-code classes.
    require_hf_native:
        Require the root model class to come from ``transformers.models.*``.
    allow_generic:
        Allow heuristic fallback discovery if primary structural discovery
        finds no gates.
    dynamic_discovery:
        Use runtime candidate discovery when ``sample_input`` is provided.
        Disable this to use structural discovery while still retaining the
        sample for transactional post-install forward verification.
    initialize_from_original:
        Initialize each replacement from the native router projection through
        ``initialize_from_projection(weight)`` or ``new_gate.base``.
    teacher_student:
        Keep each native DeepSeek-V2 router frozen as a teacher and install the
        generated gate as a random trainable student.

    Returns
    -------
    List of :class:`GateInstall` describing each replacement.
    """
    assert_hf_native_model(
        model,
        allow_remote_code=allow_remote_code,
        require_hf_native=require_hf_native,
    )

    if gate_registry is None:
        try:
            from .gates import GATE_FACTORIES
            gate_registry = GATE_FACTORIES
        except ImportError:
            gate_registry = {}

    factory: GateFactory
    if isinstance(gate, str):
        if gate not in gate_registry:
            valid = ", ".join(sorted(gate_registry))
            raise ValueError(f"Unknown gate {gate!r}. Available: {valid}")
        factory = gate_registry[gate]
    else:
        factory = gate

    if teacher_student and initialize_from_original:
        raise ValueError(
            "Teacher-student mode requires a random student; do not initialize it from the teacher"
        )

    selected_layers = set(layers) if layers is not None else None
    global _LAST_CANDIDATE_REPORT
    if sample_input is not None and dynamic_discovery:
        sites, reports = _find_moe_gates_dynamic(
            model,
            sample_input,
            factory=factory,
            top_k=top_k,
            validate=True,
        )
        _LAST_CANDIDATE_REPORT = reports
        if not sites:
            raise ValueError(
                "No dynamically validated MoE gate candidates found. "
                "Call get_gate_candidate_report() for accepted/rejected reasons, "
                "or call install_gates() without sample_input to use heuristic fallback."
            )
    else:
        sites = find_moe_gates(
            model,
            sample_input=None,
            allow_generic=allow_generic,
        )

    installs: List[GateInstall] = []
    try:
        for site in sites:
            if selected_layers is not None and site.layer_index not in selected_layers:
                continue

            new_gate = _new_gate_for_site(factory, site)
            if teacher_student:
                install = _install_teacher_student_site(
                    site,
                    new_gate,
                    top_k,
                    student_weight,
                    distillation_temperature,
                )
            elif initialize_from_original:
                _initialize_from_original_projection(new_gate, site)
                install = _install_gate_site(site, new_gate, top_k)
            else:
                install = _install_gate_site(site, new_gate, top_k)
            installs.append(install)
            logger.info(
                "Replaced gate at layer %d (%s.%s, pattern=%s, %d->%d)",
                site.layer_index, type(site.block).__name__, site.gate_attr,
                site.pattern, site.model_dim, site.num_experts,
            )

        if not installs:
            raise ValueError("No MoE gates found in the model")

        if verify and not _verify_forward(model, installs, sample_input=sample_input):
            raise RuntimeError(
                "Gate installation failed verification and was rolled back"
            )

        # Register hooks only after verification so verify-forward does not
        # pollute the first call to get_gate_logits().
        for inst in installs:
            _install_gate_logit_hook(inst)

        # Patch _can_record_outputs so HF aux-loss collection works.
        _patch_output_recorder(model, installs)

    except Exception:
        _rollback(installs)
        raise

    return installs


def _patch_output_recorder(model: nn.Module, installs: List[GateInstall]) -> None:
    """Update ``_can_record_outputs`` so HF aux-loss collection sees our wrappers."""
    if not installs:
        return

    try:
        from transformers.utils.output_capturing import OutputRecorder
    except ImportError:
        return

    orig_classes: tuple[type, ...] = tuple({type(inst.old_gate) for inst in installs})
    wrapper_types: tuple[type, ...] = tuple(
        {type(getattr(inst.site.block, inst.site.gate_attr)) for inst in installs}
    )
    all_classes = orig_classes + wrapper_types

    for attr in ("model", "transformer"):
        inner = getattr(model, attr, None)
        if inner is not None:
            recorder = getattr(inner, "_can_record_outputs", None)
            if isinstance(recorder, dict) and recorder:
                for key in list(recorder.keys()):
                    rec = recorder[key]
                    if hasattr(rec, "target_class"):
                        recorder[key] = OutputRecorder(
                            target_class=all_classes,
                            index=getattr(rec, "index", 0),
                        )
                return


def _infer_top_k(
    gate: nn.Module,
    block: nn.Module,
    num_experts: Optional[int] = None,
) -> int:
    """Try to read top_k from gate or parent block, with runtime detection."""
    # Runtime: if gate returns (logits, scores, indices), the index shape is (T, K)
    model_dim = _get_gate_input_dim(gate)
    if model_dim is not None:
        device, dtype = _module_device_dtype(gate)
        device = device or torch.device("cpu")
        rand_kwargs: dict[str, Any] = {"device": device}
        if dtype is not None and dtype.is_floating_point:
            rand_kwargs["dtype"] = dtype
        try:
            with torch.no_grad():
                out = gate(torch.randn(2, model_dim, **rand_kwargs))
            if isinstance(out, (tuple, list)):
                for x in out:
                    if (
                        isinstance(x, torch.Tensor)
                        and not x.is_floating_point()
                        and x.ndim >= 1
                    ):
                        return int(x.shape[-1])
                if num_experts is not None:
                    for x in out:
                        if (
                            isinstance(x, torch.Tensor)
                            and x.is_floating_point()
                            and x.ndim >= 2
                            and x.shape[-1] == num_experts
                        ):
                            detached = x.detach()
                            if detached.numel() == 0 or not torch.isfinite(detached).all():
                                continue
                            counts = (detached > 0).sum(dim=-1)
                            counts = counts[(counts > 0) & (counts < num_experts)]
                            if counts.numel() > 0:
                                return int(counts.max().item())
        except Exception:
            pass

    # Fallback: scan attributes
    for obj in (gate, block):
        for attr in (
            "top_k", "topk", "num_experts_per_tok", "num_experts_per_token",
            "num_selected_experts", "router_top_k", "routing_top_k",
            "moe_top_k", "moe_topk", "moe_k", "k",
        ):
            val = getattr(obj, attr, None)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
    raise ValueError(
        "Could not infer top_k from gate or block. "
        "Pass top_k=N to install_gates()."
    )


def _resolve_top_k(
    gate: nn.Module,
    block: nn.Module,
    num_experts: int,
    top_k: Optional[int],
) -> int:
    value = int(top_k) if top_k is not None else _infer_top_k(gate, block, num_experts)
    if value < 1:
        raise ValueError(f"top_k must be >= 1, got {value}")
    if value > num_experts:
        raise ValueError(f"top_k={value} exceeds num_experts={num_experts}")
    return value


# ---------------------------------------------------------------------------
# Freeze / utility
# ---------------------------------------------------------------------------


def freeze_except_gates(
    model: nn.Module,
    installs: List[GateInstall],
) -> None:
    """Freeze all parameters except those belonging to replacement gates."""
    trainable_gate_ids: set[int] = set()
    for inst in installs:
        parameters = (
            inst.student_gate.parameters()
            if inst.student_gate is not None
            else inst.new_gate.parameters()
        )
        for param in parameters:
            trainable_gate_ids.add(id(param))

    for param in model.parameters():
        param.requires_grad_(id(param) in trainable_gate_ids)


def gate_trainable_parameters(installs: Iterable[GateInstall]) -> List[nn.Parameter]:
    """Return unique trainable gate parameters, excluding frozen teachers."""
    seen: set[int] = set()
    parameters: List[nn.Parameter] = []
    for install in installs:
        candidates = (
            install.student_gate.parameters()
            if install.student_gate is not None
            else install.new_gate.parameters()
        )
        for parameter in candidates:
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters


def teacher_student_distillation_loss(installs: Iterable[GateInstall]) -> torch.Tensor:
    """Average the current forward pass's teacher-to-student router KL losses."""
    losses: List[torch.Tensor] = []
    for install in installs:
        loss = getattr(install.new_gate, "_last_distillation_loss", None)
        if isinstance(loss, torch.Tensor):
            if torch.is_grad_enabled() and not loss.requires_grad:
                raise RuntimeError(
                    "Router distillation loss has no gradient; use non-reentrant checkpointing"
                )
            losses.append(loss)
            install.new_gate._last_distillation_loss = None
    if not losses:
        raise RuntimeError("No teacher-student distillation losses were captured")
    reduction_device = losses[0].device
    return torch.stack([
        loss if loss.device == reduction_device else loss.to(reduction_device)
        for loss in losses
    ]).mean()


def set_teacher_student_weight(installs: Iterable[GateInstall], value: float) -> None:
    """Set one routing handoff weight across all teacher-student gates."""
    found = False
    for install in installs:
        setter = getattr(install.new_gate, "set_student_weight", None)
        if callable(setter):
            setter(value)
            found = True
    if not found:
        raise RuntimeError("No teacher-student gates are installed")


def restore_gates(installs: Iterable[GateInstall]) -> None:
    """Restore gate replacements created by :func:`install_gates`.

    This is the public counterpart to the installer's transactional rollback.
    It removes logit hooks and correctly restores indexed module-container
    children as well as ordinary module attributes.
    """
    _rollback(list(installs))


def trainable_parameter_names(model: nn.Module) -> List[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ---------------------------------------------------------------------------
# Forward hooks for gate logit collection (aux loss)
# ---------------------------------------------------------------------------

_HOOK_HANDLES: dict[int, list[Any]] = {}


def _extract_logits_from_gate_output(out: Any, num_experts: int) -> torch.Tensor:
    """Extract the router logits tensor from a gate output (tensor or tuple)."""
    if isinstance(out, torch.Tensor):
        return out

    if isinstance(out, (tuple, list)):
        candidates = [
            x for x in out
            if isinstance(x, torch.Tensor)
            and x.is_floating_point()
            and x.ndim >= 2
            and x.shape[-1] == num_experts
        ]
        if not candidates:
            raise RuntimeError("No expert-sized float tensor found in gate output")

        # Llama4-style sparse sigmoid returns (sparse_scores, logits).  Both
        # tensors are finite and expert-sized, so prefer the second item for
        # two-tensor contracts.
        if len(out) == 2 and len(candidates) == 2:
            return candidates[-1]

        # Common HF routers return (logits, weights, indices).  If the first
        # element is expert-sized, it is the raw router logits.
        first = out[0]
        if (
            isinstance(first, torch.Tensor)
            and first.is_floating_point()
            and first.ndim >= 2
            and first.shape[-1] == num_experts
        ):
            return first

        return candidates[-1]

    raise RuntimeError(f"Unexpected gate output type: {type(out).__name__}")


def _current_replacement_module(inst: GateInstall) -> nn.Module:
    if inst.owner is not None and inst.attr is not None:
        return getattr(inst.owner, inst.attr)
    return getattr(inst.site.block, inst.site.gate_attr)


def _install_gate_logit_hook(inst: GateInstall) -> None:
    """Register a forward hook that captures the gate's router logits.

    The logits are stored in ``inst.new_gate._gate_logits`` (a list).
    Gradients are NOT detached — aux-loss gradients flow through.
    """
    target = _current_replacement_module(inst)
    if not hasattr(target, "_gate_logits"):
        target._gate_logits: list[torch.Tensor] = []

    def hook_fn(mod, inp, out):
        try:
            logits = _extract_logits_from_gate_output(out, inst.site.num_experts)
        except RuntimeError:
            logits = getattr(mod, "_last_gate_logits", None)
            if not isinstance(logits, torch.Tensor):
                raise
        mod._gate_logits.clear()
        mod._gate_logits.append(logits)  # keep only the latest forward's graph

    handle = target.register_forward_hook(hook_fn)
    inst.hook_handles.append(handle)
    _HOOK_HANDLES.setdefault(id(inst), []).append(handle)


def get_gate_logits(model: nn.Module) -> list[torch.Tensor]:
    """Return all collected gate logits from the last forward pass.

    After a model forward, this returns the router logits from every replaced
    gate.  Use these to compute a custom auxiliary loss.
    """
    all_logits: list[torch.Tensor] = []
    for name, module in model.named_modules():
        if hasattr(module, "_gate_logits"):
            all_logits.extend(module._gate_logits)
            module._gate_logits.clear()
    return all_logits


# ---------------------------------------------------------------------------
# Runtime contract detection (replaces inspect.getsource)
# ---------------------------------------------------------------------------


def _looks_probability_like(tensor: torch.Tensor) -> bool:
    try:
        detached = tensor.detach()
        if detached.numel() == 0 or not torch.isfinite(detached).all():
            return False
        return bool(detached.min().item() >= -1e-6 and detached.max().item() <= 1.0 + 1e-6)
    except Exception:
        return False


def _trace_gate_contract(gate: nn.Module, model_dim: int) -> str:
    """Run the gate on a dummy input and observe the output structure.

    Returns one of ``"logits_only"``, ``"softmax_topk"``, ``"sparse_sigmoid"``.
    Does NOT use source code inspection — purely runtime observation.
    """
    device, dtype = _module_device_dtype(gate)
    device = device or torch.device("cpu")
    rand_kwargs: dict[str, Any] = {"device": device}
    if dtype is not None and dtype.is_floating_point:
        rand_kwargs["dtype"] = dtype

    x = torch.randn(2, model_dim, **rand_kwargs)
    try:
        with torch.no_grad():
            out = gate(x)
    except Exception:
        return "logits_only"

    if isinstance(out, torch.Tensor):
        return "logits_only"
    if isinstance(out, (tuple, list)):
        n = len(out)
        if n == 2:
            first, second = out
            if (
                isinstance(first, torch.Tensor)
                and isinstance(second, torch.Tensor)
                and first.is_floating_point()
                and not second.is_floating_point()
                and first.ndim >= 1
                and second.shape == first.shape
            ):
                return "topk_weights_indices"
            if (
                isinstance(first, torch.Tensor)
                and isinstance(second, torch.Tensor)
                and first.is_floating_point()
                and second.is_floating_point()
                and first.ndim >= 2
                and second.shape == first.shape
                and (_looks_probability_like(first) or not _looks_probability_like(second))
            ):
                return "sparse_sigmoid"
            return "logits_only"
        if n >= 3:
            first_three = out[:3]
            if all(isinstance(x, torch.Tensor) for x in first_three):
                a, b, c = first_three
                if a.is_floating_point() and not b.is_floating_point() and c.is_floating_point():
                    return "logits_indices_weights"
                if a.is_floating_point() and b.is_floating_point() and not c.is_floating_point():
                    return "softmax_topk"
            return "softmax_topk"
    return "logits_only"


# ---------------------------------------------------------------------------
# Verify-forward with rollback
# ---------------------------------------------------------------------------


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
