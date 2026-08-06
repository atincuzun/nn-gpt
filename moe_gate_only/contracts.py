from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _module_device_dtype(module: nn.Module) -> Tuple[Optional[torch.device], Optional[torch.dtype]]:
    for param in module.parameters(recurse=True):
        return param.device, param.dtype
    for buf in module.buffers(recurse=True):
        return buf.device, buf.dtype
    return None, None


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
