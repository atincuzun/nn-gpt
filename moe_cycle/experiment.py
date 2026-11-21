"""Controlled MoE router experiments: one experiment arm per process.

The gate cycle answers "which gate scored best", but its scores could not tell
architectures apart: every gate starts as an exact copy of the native router,
so a score taken before training measures the native model. This runner
measures a single, fully specified arm instead:

* ``--gate-spec`` fixes the router: ``none`` (untouched native module),
  ``native`` (a trainable exact copy -- the control for gate training),
  ``catalog:<name>`` (a built-in gate architecture), ``constant`` (routing by
  expert bias only -- a deliberately broken router) or ``file:<path>``;
  ``--router-noise`` perturbs the copied native projection (dose-response).
* ``--train`` fixes what learns each round: ``none``, ``gate``, ``lora`` (the
  LoRA path of the NNGPT loop) or ``lora+gate``.
* Every round is ``train -> measure``: generate ``--test-nn`` CV models from
  the same pinned seed prompts with the same sampling seeds as every other arm,
  train each on the dataset, and record per-prompt outcomes. ``--measure-initial``
  adds a measurement before any training.
* After every phase a ``RoutingProbe`` records how far routing and the output
  distribution moved from the native model, so a null result can be told
  apart from "the router never changed".

The arm's own evaluated models are written to LEMUR under its unique
``--nn-name-prefix`` and enter its later training rounds, exactly as in the
generate -> evaluate -> fine-tune loop, without leaking into other arms.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NATIVE_GATE_CODE = '''import torch
import torch.nn as nn


class Net(nn.Module):
    """The native router as a trainable module: one bias-free projection."""

    def __init__(self, model_dim: int, num_experts: int):
        super().__init__()
        self.base = nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, x):
        return self.base(x)
'''

CONSTANT_GATE_CODE = '''import torch
import torch.nn as nn


class Net(nn.Module):
    """Broken router: constant logits, so experts are chosen by bias alone."""

    def __init__(self, model_dim: int, num_experts: int):
        super().__init__()
        self.base = nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, x):
        return self.base(x) * 0.0
'''

TRAIN_MODES = ("none", "gate", "lora", "lora+gate")
RESULT_FILE = "experiment_result.json"


class GenerationOOM(BaseException):
    """Batched generation ran out of memory.

    A BaseException on purpose: ChatBot.chat_batch catches ``Exception`` and
    silently falls back to one-prompt-at-a-time generation, which would make
    some arms sample differently from others without any record of it.
    """


def _guard_batched_generation(chat_bot) -> None:
    original = getattr(chat_bot, "_direct_generate_batch", None)
    if original is None or getattr(original, "_moe_oom_guard", False):
        return

    def guarded(*args, **kwargs):
        import torch

        try:
            return original(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as exc:
            raise GenerationOOM(str(exc)) from exc

    guarded._moe_oom_guard = True
    chat_bot._direct_generate_batch = guarded


def build_experiment_parser():
    from .cli import build_parser

    parser = build_parser()
    parser.description = __doc__
    group = parser.add_argument_group("experiment arm")
    group.add_argument("--arm", required=True, help="Arm identifier recorded in the result")
    group.add_argument("--stage", default="", help="Stage label recorded in the result")
    group.add_argument(
        "--gate-spec",
        default="none",
        help="none | native | catalog:<name-or-index> | constant | file:<path>",
    )
    group.add_argument(
        "--router-noise",
        type=float,
        default=0.0,
        help="Gaussian noise added to the copied native projection, as a fraction "
        "of its weight std (requires a gate with a 'base' projection)",
    )
    group.add_argument("--train", choices=TRAIN_MODES, default="none")
    group.add_argument(
        "--measure-initial",
        action="store_true",
        help="Also measure before the first training round",
    )
    group.add_argument(
        "--gen-seed",
        type=int,
        default=1000,
        help="Base sampling seed for CV generation; round r uses gen_seed + r in "
        "every arm, so arms share their random numbers",
    )
    group.add_argument("--grad-accum", type=int, default=4,
                       help="Micro-batches per optimizer step")
    group.add_argument("--lora-lr", type=float, default=1e-4)
    group.add_argument("--lora-rank", type=int, default=16)
    group.add_argument("--lora-alpha", type=int, default=32)
    group.add_argument("--lora-dropout", type=float, default=0.05)
    group.add_argument("--weight-decay", type=float, default=0.0)
    group.add_argument("--probe-sequences", type=int, default=12)
    group.add_argument("--probe-max-tokens", type=int, default=1024)
    group.add_argument(
        "--no-save-weights",
        action="store_true",
        help="Skip writing the final gate/LoRA weights",
    )
    return parser


def _validate(args) -> None:
    if args.train in ("gate", "lora+gate") and args.gate_spec == "none":
        raise ValueError("--train gate needs a replacement gate (--gate-spec)")
    if args.router_noise and args.gate_spec == "none":
        raise ValueError("--router-noise perturbs an installed gate (--gate-spec)")
    if args.epochs < 0:
        raise ValueError("--epochs (training rounds) must be >= 0")
    if args.epochs == 0 and not args.measure_initial:
        raise ValueError("nothing to do: --epochs 0 without --measure-initial")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")


def _gate_source(spec: str) -> tuple[str, str]:
    """(source, label) for a gate spec."""
    from .gate_catalog import GATE_CATALOG

    if spec == "native":
        return NATIVE_GATE_CODE, "native"
    if spec == "constant":
        return CONSTANT_GATE_CODE, "constant"
    if spec.startswith("file:"):
        path = Path(spec[len("file:"):])
        return path.read_text(encoding="utf-8"), f"file:{path.name}"
    if spec.startswith("catalog:"):
        key = spec[len("catalog:"):]
        for index, (name, source) in enumerate(GATE_CATALOG):
            if key in (name, str(index), f"{index:02d}") or name.startswith(key):
                return source, name
        raise KeyError(f"no catalog gate matches {key!r}")
    raise ValueError(f"unknown --gate-spec {spec!r}")


def _router_module_ids(sites) -> set[int]:
    ids: set[int] = set()
    for site in sites:
        for module in site.gate.modules():
            ids.add(id(module))
    return ids


def _inject_lora(model, sites, args) -> dict[str, Any]:
    """LoRA on every nn.Linear except the routers and the LM head."""
    import torch.nn as nn
    from peft import LoraConfig, inject_adapter_in_model

    router_ids = _router_module_ids(sites)
    targets = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and id(module) not in router_ids
        and not name.endswith("lm_head")
    ]
    if not targets:
        raise RuntimeError("No nn.Linear modules found for LoRA")
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
    )
    inject_adapter_in_model(config, model, adapter_name="default")
    lora_parameters = [
        parameter for name, parameter in model.named_parameters() if "lora_" in name
    ]
    return {
        "target_modules": len(targets),
        "parameters": int(sum(parameter.numel() for parameter in lora_parameters)),
    }


def _lora_parameters(model) -> list:
    return [parameter for name, parameter in model.named_parameters() if "lora_" in name]


def _set_trainable(ctx, train_mode: str) -> dict[str, int]:
    model = ctx.session.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gate_params = []
    if train_mode in ("gate", "lora+gate"):
        from moe_gate_only.universal import gate_trainable_parameters

        for install in ctx.session.installs:
            for parameter in install.new_gate.parameters():
                parameter.requires_grad_(True)
        gate_params = gate_trainable_parameters(ctx.session.installs)
    lora_params = []
    if train_mode in ("lora", "lora+gate"):
        lora_params = _lora_parameters(model)
        for parameter in lora_params:
            parameter.requires_grad_(True)
    ctx.experiment_gate_params = gate_params
    ctx.experiment_lora_params = lora_params
    return {
        "gate": int(sum(p.numel() for p in gate_params)),
        "lora": int(sum(p.numel() for p in lora_params)),
    }


def _install(ctx, args) -> dict[str, Any]:
    """Install the arm's router and verify what it computes at step zero."""
    import torch

    from .morphism import _model_logits, _perturb_gate_weights, _seed_all

    info: dict[str, Any] = {"gate_spec": args.gate_spec, "router_noise": args.router_noise}
    if args.gate_spec == "none":
        info["installed_layers"] = []
        return info
    source, label = _gate_source(args.gate_spec)
    function_preserving = args.gate_spec != "constant"
    _seed_all(args.seed)
    ctx.session.model.eval()
    ctx.session.replace_source(
        source,
        class_name="Net",
        layers=args.layers,
        sample_input=ctx.sample_input,
        verify=function_preserving and not args.router_noise,
        top_k=args.router_top_k,
        allow_remote_code=True,
        dynamic_discovery=False,
        initialize_from_original=True,
    )
    ctx.gate_source = source
    if args.router_noise:
        info["noise"] = _perturb_gate_weights(ctx.session.installs, args.router_noise, args.seed)
    logits = _model_logits(ctx.session.model, ctx.sample_input)
    if not torch.isfinite(logits).all():
        raise RuntimeError("installed gates produced non-finite logits")
    max_abs = float((ctx.native_logits - logits).abs().max().item())
    info.update({
        "gate_label": label,
        "gate_code": source,
        "installed_layers": sorted(install.site.layer_index for install in ctx.session.installs),
        "step_zero_max_abs_logit_difference": max_abs,
        "step_zero_equivalent": bool(
            torch.allclose(ctx.native_logits, logits, rtol=1e-5, atol=1e-5)
        ),
        "gate_parameters": int(sum(
            p.numel() for install in ctx.session.installs for p in install.new_gate.parameters()
        )),
    })
    if function_preserving and not args.router_noise and not info["step_zero_equivalent"]:
        raise RuntimeError(
            f"{label} does not reproduce the native logits at step zero "
            f"(max_abs_difference={max_abs})"
        )
    return info


def _step_zero(ctx, args) -> dict[str, Any]:
    """Whole-model check after every change (adapter + gates) before training.

    Arms that should start as the native model (zero-initialized LoRA B,
    function-preserving gates, no noise) must reproduce its logits exactly;
    otherwise their round-0 behavior would differ for a reason that is not
    the experimental factor.
    """
    import torch

    from .morphism import _model_logits

    logits = _model_logits(ctx.session.model, ctx.sample_input)
    max_abs = float((ctx.native_logits - logits).abs().max().item())
    expected_native = args.gate_spec != "constant" and not args.router_noise
    equivalent = bool(torch.allclose(ctx.native_logits, logits, rtol=1e-5, atol=1e-5))
    if expected_native and not equivalent:
        raise RuntimeError(
            "arm does not start as the native model "
            f"(max_abs_logit_difference={max_abs})"
        )
    return {
        "max_abs_logit_difference": max_abs,
        "equivalent_to_native": equivalent,
        "expected_native": expected_native,
    }


def _summarize_outcomes(outcomes: list[dict]) -> dict[str, Any]:
    n_slots = len(outcomes)
    measured = [o for o in outcomes if o.get("status") == "measured" and o.get("accuracy") is not None]
    accuracies = [float(o["accuracy"]) for o in measured]
    improved = [
        o for o in measured
        if o.get("baseline_accuracy") is not None and o["accuracy"] > o["baseline_accuracy"]
    ]
    mean = sum(accuracies) / len(accuracies) if accuracies else None
    return {
        "n_slots": n_slots,
        "n_measured": len(measured),
        "valid_rate": len(measured) / n_slots if n_slots else None,
        "mean_accuracy": mean,
        "yield_accuracy": sum(accuracies) / n_slots if n_slots else None,
        "best_accuracy": max(accuracies) if accuracies else None,
        "improved_rate": len(improved) / n_slots if n_slots else None,
    }


def _measure(ctx, args, round_index: int, root: Path) -> dict[str, Any]:
    from .epochs import _measure_cv_epoch
    from .morphism import _seed_all

    import shutil

    import torch

    from .context import _release_cuda_memory

    epoch_path = root / "rounds" / f"R{round_index}"
    started = time.monotonic()
    requested_batch = args.generation_batch
    while True:
        shutil.rmtree(epoch_path, ignore_errors=True)
        epoch_path.mkdir(parents=True, exist_ok=True)
        _seed_all(args.gen_seed + round_index)
        try:
            measured = _measure_cv_epoch(ctx, round_index, epoch_path)
            break
        except (GenerationOOM, torch.cuda.OutOfMemoryError):
            # A smaller batch changes how sampling noise is shared across
            # prompts, so the batch actually used is recorded with the round.
            if args.generation_batch <= 1:
                raise
            _release_cuda_memory()
            args.generation_batch = max(1, args.generation_batch // 2)
            print(f"[EXP] generation OOM; retrying round {round_index} with "
                  f"--generation-batch {args.generation_batch}", flush=True)
    used_batch = args.generation_batch
    args.generation_batch = requested_batch
    outcomes = measured["gate_feedback"].get("candidate_outcomes", [])
    slots = [
        {
            "slot": o.get("candidate"),
            "source_nn": o.get("source_nn"),
            "baseline_accuracy": o.get("baseline_accuracy"),
            "accuracy": o.get("accuracy"),
            "status": o.get("status"),
            "failure": o.get("failure"),
        }
        for o in outcomes
    ]
    import hashlib

    for slot in slots:
        code_file = epoch_path / "synth_nn" / str(slot["slot"]) / "new_nn.py"
        slot["code_sha1"] = (
            hashlib.sha1(code_file.read_bytes()).hexdigest()[:16]
            if code_file.is_file() else None
        )
    summary = _summarize_outcomes(slots)
    summary["seconds"] = round(time.monotonic() - started, 1)
    summary["generation_batch"] = used_batch
    return {"summary": summary, "slots": slots, "path": str(epoch_path)}


def _training_loaders(ctx, args, round_index: int):
    from ab.gpt.util.Const import conf_train_dir
    from moe_gate_only import build_nngenprompt_dataloaders

    from .context import _nas_prefixes
    from .morphism import _seed_all

    os.environ["NNGPT_SFT_SAMPLE_SEED"] = str(args.seed + round_index)
    _seed_all(args.seed + 1000 + round_index)
    return build_nngenprompt_dataloaders(
        ctx.tokenizer,
        conf_train_dir / args.train_prompt_config,
        context_length=args.max_length,
        max_prompts=args.max_prompts,
        max_new_tokens=args.generation_max_new_tokens,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed + round_index,
        dataset_name=args.dataset,
        nn_prefixes=_nas_prefixes(args),
    )


def _train_round(ctx, args, round_index: int) -> dict[str, Any]:
    """``--gate-train-steps`` optimizer steps on the NN-generation SFT data."""
    import torch

    from moe_gate_only import evaluate_language_model_loss
    from moe_gate_only.training import model_input_device, move_batch_to_device

    from .context import _log_cuda_memory, _release_cuda_memory
    from .morphism import _seed_all

    model = ctx.session.model
    gate_params = ctx.experiment_gate_params
    lora_params = ctx.experiment_lora_params
    groups = []
    if gate_params:
        groups.append({"params": gate_params, "lr": args.gate_learning_rate})
    if lora_params:
        groups.append({"params": lora_params, "lr": args.lora_lr})
    if not groups:
        raise RuntimeError("training round without trainable parameters")
    trainable = [p for group in groups for p in group["params"]]

    started = time.monotonic()
    train_loader, val_loader, train_dataset = _training_loaders(ctx, args, round_index)
    _seed_all(args.seed + 2000 + round_index)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    model.train()
    model.config.use_cache = False
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    device = model_input_device(model)
    iterator = iter(train_loader)
    step_losses: list[float] = []
    skipped_micro = 0
    for step in range(1, args.gate_train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            try:
                loss = model(**move_batch_to_device(batch, device)).loss
                (loss / args.grad_accum).backward()
                micro_losses.append(float(loss.detach().item()))
            except torch.cuda.OutOfMemoryError:
                skipped_micro += 1
                optimizer.zero_grad(set_to_none=True)
                micro_losses = []
                _release_cuda_memory()
                break
        if not micro_losses:
            continue
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        step_losses.append(sum(micro_losses) / len(micro_losses))
        if step % 10 == 0 or step == args.gate_train_steps:
            window = step_losses[-10:]
            print(
                f"[EXP] round={round_index} step={step:04d}/{args.gate_train_steps} "
                f"loss={sum(window) / len(window):.4f}",
                flush=True,
            )
    vram = _log_cuda_memory(f"round={round_index} after training")
    validation_loss = None
    if val_loader is not None:
        try:
            validation_loss = evaluate_language_model_loss(
                model, val_loader, max_steps=args.validation_steps
            )
        except torch.cuda.OutOfMemoryError:
            _release_cuda_memory()
    if args.gradient_checkpointing:
        if hasattr(model, "disable_input_require_grads"):
            model.disable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    _release_cuda_memory()
    head = step_losses[:10]
    tail = step_losses[-10:]
    return {
        "examples": len(train_dataset),
        "steps": len(step_losses),
        "grad_accum": args.grad_accum,
        "skipped_micro_batches_oom": skipped_micro,
        "loss_first10": sum(head) / len(head) if head else None,
        "loss_last10": sum(tail) / len(tail) if tail else None,
        "loss_curve": [round(value, 5) for value in step_losses],
        "validation_loss": validation_loss,
        "vram": vram,
        "seconds": round(time.monotonic() - started, 1),
    }


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _save_weights(ctx, args, root: Path) -> dict[str, str]:
    import torch

    saved: dict[str, str] = {}
    if ctx.session.installs and args.train in ("gate", "lora+gate"):
        ctx.session.save(root / "final_gate")
        saved["gate"] = str(root / "final_gate")
    if args.train in ("lora", "lora+gate"):
        state = {
            name: parameter.detach().cpu()
            for name, parameter in ctx.session.model.named_parameters()
            if "lora_" in name
        }
        torch.save(state, root / "final_lora.pt")
        saved["lora"] = str(root / "final_lora.pt")
    return saved


def main() -> None:
    from .eval_safety import ensure_moe_eval_safety
    from .generation_dtype import ensure_generation_dtype_policy
    from .kernel_compat import ensure_device_aware_causal_conv1d

    ensure_generation_dtype_policy()
    ensure_device_aware_causal_conv1d()
    ensure_moe_eval_safety()
    args = build_experiment_parser().parse_args()
    args.gate_outer_search = False
    args.gate_candidates = 1
    _validate(args)
    if args.output is None:
        raise ValueError("--output is required for an experiment arm")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.gate_store is None:
        args.gate_store = root / "gate_store"

    from .context import _nas_prefixes, _release_cuda_memory, _ts
    from .routing_probe import RoutingProbe
    from .setup import _setup_run

    result: dict[str, Any] = {
        "arm": args.arm,
        "stage": args.stage,
        "complete": False,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {key: (str(value) if isinstance(value, Path) else value)
                   for key, value in vars(args).items()},
        "seed_file": os.environ.get("NNGPT_SEED_FILE"),
        "rounds": [],
    }
    result_path = root / RESULT_FILE
    _write(result_path, result)

    ctx = _setup_run(args)
    ctx.candidate_index = 0
    ctx.candidate_root = root
    _guard_batched_generation(ctx.chat_bot)
    top_k = int(getattr(ctx.session.model.config, "num_experts_per_tok", 0) or args.router_top_k or 4)

    # Probe rows come from the training distribution's validation split.
    _, probe_loader, probe_train_rows = _training_loaders(ctx, args, -1)
    probe_rows = probe_loader.dataset if probe_loader is not None else probe_train_rows
    probe = RoutingProbe.from_dataset(
        probe_rows, count=args.probe_sequences,
        max_tokens=args.probe_max_tokens, top_k=top_k,
    )
    result["probe"] = probe.capture_reference(ctx.session.model, ctx.sites)
    print(f"[EXP] {args.arm}: native routing captured {result['probe']}", flush=True)

    if args.train in ("lora", "lora+gate"):
        result["lora"] = _inject_lora(ctx.session.model, ctx.sites, args)
        print(f"[EXP] {args.arm}: LoRA injected {result['lora']}", flush=True)
    result["install"] = _install(ctx, args)
    result["step_zero"] = _step_zero(ctx, args)
    result["trainable_parameters"] = _set_trainable(ctx, args.train)
    result["training_prefixes"] = list(_nas_prefixes(args))
    replaced = result["install"].get("installed_layers", [])
    print(
        f"[EXP] {args.arm}: gate={args.gate_spec} noise={args.router_noise} "
        f"train={args.train} layers={replaced} trainable={result['trainable_parameters']} "
        f"step0_diff={result['install'].get('step_zero_max_abs_logit_difference')}",
        flush=True,
    )
    _write(result_path, result)

    def diagnostics() -> dict[str, Any]:
        return probe.compare(ctx.session.model, ctx.sites, ctx.session.installs, replaced)

    total_steps = 0
    rounds = range(0 if args.measure_initial else 1, args.epochs + 1)
    for round_index in rounds:
        record: dict[str, Any] = {"round": round_index}
        if round_index > 0 and args.train != "none":
            print(f"\n[{_ts()}] [EXP] {args.arm}: training round {round_index}", flush=True)
            record["training"] = _train_round(ctx, args, round_index)
            total_steps += record["training"]["steps"]
        record["optimizer_steps_total"] = total_steps
        record["diagnostics"] = diagnostics()
        _release_cuda_memory()
        print(f"[{_ts()}] [EXP] {args.arm}: measuring round {round_index}", flush=True)
        record["measurement"] = _measure(ctx, args, round_index, root)
        _release_cuda_memory()
        summary = record["measurement"]["summary"]
        diag = record["diagnostics"]
        print(
            f"[{_ts()}] [EXP] {args.arm} R{round_index}: valid={summary['n_measured']}/"
            f"{summary['n_slots']} mean={summary['mean_accuracy']} yield={summary['yield_accuracy']} "
            f"flip_all={diag['flip_any_all_layers']} flip_replaced={diag['flip_any_replaced_layers']} "
            f"kl={diag['kl_native_to_current']} nll_delta={diag['probe_nll_delta']}",
            flush=True,
        )
        result["rounds"].append(record)
        _write(result_path, result)

    if not args.no_save_weights:
        try:
            result["weights"] = _save_weights(ctx, args, root)
        except Exception as exc:  # the measurements are the result; weights are a convenience
            result["weights"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[EXP] {args.arm}: saving final weights failed: {exc}", flush=True)
    result["complete"] = True
    result["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write(result_path, result)
    print(f"[EXP] {args.arm}: complete -> {result_path}", flush=True)
