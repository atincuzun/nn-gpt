from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from . import discovery
from .contracts import (
    _DeepSeekV2Gate,
    _DeepSeekV2TeacherStudentGate,
    _HashScoreFnGate,
    _LogitsOnlyGate,
    _ScoreFnTopKGate,
    _SigmoidTopKGate,
    _SoftmaxTopKGate,
    _SparseSigmoidGate,
    _TopKWeightsIndicesGate,
    _copy_gate_attrs,
    _detect_post_processing,
    _get_gate_input_dim,
    _module_device_dtype,
)
from .datastructures import GateFactory, GateInstall, GateSite
from .discovery import _find_linear_owner_attr
from .hooks import _install_gate_logit_hook
from .morphism import initialize_gate_from_projection
from .tree import _assign_child
from .verify import _rollback, _verify_forward

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


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
            # The block is attached to the live model, so its training flag is
            # current; the native gate may be detached with a stale flag.
            wrapped.train(site.block.training)
            _assign_child(site.block, site.gate_attr, wrapped)
            installed_gate = wrapped

    elif site.pattern == "parameter_gate":
        wrapped = _wrap_gate_for_contract(site, new_gate, top_k)
        _copy_gate_attrs(site.gate, wrapped)
        wrapped.train(site.block.training)
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
        Optional mapping of gate name -> factory used when ``gate`` is a
        string.  A registry must be provided for string-based selection;
        ``None`` with a string ``gate`` is an error.
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
        Initialize each replacement from the native router projection via
        :func:`moe_gate_only.morphism.initialize_gate_from_projection` — either
        the gate's ``initialize_from_projection(weight, bias)`` protocol or a
        ``base`` weight copy.
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

    factory: GateFactory
    if isinstance(gate, str):
        if gate_registry is None:
            raise ValueError(
                f"Gate name {gate!r} requires a gate_registry mapping names to "
                "factories. Pass a callable factory (model_dim, num_experts) -> "
                "nn.Module, or provide a registry (e.g. from an external gate "
                "package) for string-based selection."
            )
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
    if sample_input is not None and dynamic_discovery:
        sites, reports = discovery._find_moe_gates_dynamic(
            model,
            sample_input,
            factory=factory,
            top_k=top_k,
            validate=True,
        )
        discovery._LAST_CANDIDATE_REPORT = reports
        if not sites:
            raise ValueError(
                "No dynamically validated MoE gate candidates found. "
                "Call get_gate_candidate_report() for accepted/rejected reasons, "
                "or call install_gates() without sample_input to use heuristic fallback."
            )
    else:
        sites = discovery.find_moe_gates(
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
                source = getattr(site.gate, "weight", None)
                bias = getattr(site.gate, "bias", None)
                initialize_gate_from_projection(
                    new_gate,
                    source,
                    bias=bias if isinstance(bias, torch.Tensor) else None,
                )
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

        if verify:
            verify_errors: List[BaseException] = []
            if not _verify_forward(
                model, installs, sample_input=sample_input, errors=verify_errors,
            ):
                detail = ""
                if verify_errors:
                    cause = verify_errors[-1]
                    detail = f" ({type(cause).__name__}: {cause})"
                raise RuntimeError(
                    "Gate installation failed verification and was rolled back"
                    + detail
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
