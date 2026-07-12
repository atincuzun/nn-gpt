"""Iterative MoE gate-training NAS cycle mirroring Tune.py's tune() suite.

Each epoch:
  1. Generate NN architectures (nn_gen) using the gate-replaced LLM
  2. Evaluate generated NNs (NNEval, called internally by nn_gen)
  3. Build NNGenPrompt data from real evaluated LEMUR results
  4. Train only generated MoE gates on language-model and router-distillation loss
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import shutil
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of pipeline generation/evaluation/training epochs")
    parser.add_argument("--test-nn", type=int, default=2)
    parser.add_argument("--nn-train-epochs", type=int, default=1)
    parser.add_argument("--gate-train-steps", type=int, default=2)
    parser.add_argument("--gate-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--gate-mode",
        choices=("teacher-student", "direct"),
        default="teacher-student",
        help="Train a random shadow student or use the legacy direct replacement",
    )
    parser.add_argument(
        "--gate-init-noise-scale",
        type=float,
        default=0.0,
        help="Gaussian initialization noise as a fraction of each native gate weight std",
    )
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--distillation-temperature", type=float, default=1.0)
    parser.add_argument("--student-weight-start", type=float, default=0.0)
    parser.add_argument("--student-weight-step", type=float, default=0.1)
    parser.add_argument(
        "--handoff-mode",
        choices=("guarded", "never", "fixed"),
        default="guarded",
        help="Increase student routing control only after imitation succeeds, never, or every epoch",
    )
    parser.add_argument("--handoff-min-topk-overlap", type=float, default=0.95)
    parser.add_argument("--handoff-max-kl", type=float, default=0.1)
    parser.add_argument("--handoff-max-validation-loss-increase", type=float, default=0.05)
    parser.add_argument("--gate-generation-attempts", type=int, default=3)
    parser.add_argument("--gate-max-new-tokens", type=int, default=1024)
    parser.add_argument("--generation-max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--validation-steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--router-top-k", type=int)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--conf-keys", nargs="+", default=["improve_classification_only"])
    parser.add_argument("--nn-name-prefix", default="moe-gate-cycle")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


# ── Gate source extraction ─────────────────────────────────────────────────────

def _gate_candidates(raw: str):
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw or "", flags=re.IGNORECASE).strip()
    for match in re.finditer(r"<gate\b[^>]*>([\s\S]*?)</gate\s*>", cleaned, flags=re.IGNORECASE):
        yield textwrap.dedent(match.group(1)).strip()
    for match in re.finditer(r"```(?:python|py)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE):
        yield textwrap.dedent(match.group(1)).strip()
    starts = [
        pos for pos in (
            cleaned.find("import torch"),
            cleaned.find("from torch"),
            cleaned.find("class LLMGeneratedGate"),
        ) if pos >= 0
    ]
    if starts:
        yield textwrap.dedent(cleaned[min(starts):]).strip()


# ── Gate validation ─────────────────────────────────────────────────────────────

def _validate_gate_source(source: str, shapes: list[tuple[int, int]]) -> None:
    import torch
    import torch.nn as nn

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] != "torch" for alias in node.names):
                raise ValueError("Generated gate may import only torch")
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] != "torch":
            raise ValueError("Generated gate may import only torch")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {
                "bernoulli", "dropout", "multinomial", "normal",
                "rand", "rand_like", "randn", "randn_like",
            }:
                raise ValueError("Generated student gate forward must be deterministic")
    namespace: dict[str, Any] = {"__name__": "_generated_gate"}
    exec(compile(tree, "<generated-gate>", "exec"), namespace)
    gate_cls = namespace.get("LLMGeneratedGate")
    if not isinstance(gate_cls, type) or not issubclass(gate_cls, nn.Module):
        raise TypeError("Generated source must define nn.Module class LLMGeneratedGate")
    for model_dim, num_experts in shapes:
        gate = gate_cls(model_dim, num_experts).float().eval()
        if any(isinstance(module, (nn.Dropout, nn.modules.batchnorm._BatchNorm)) for module in gate.modules()):
            raise ValueError("Generated student gates must be deterministic; dropout/batchnorm are unsupported")
        base = getattr(gate, "base", None)
        if not isinstance(base, nn.Linear) or tuple(base.weight.shape) != (num_experts, model_dim):
            raise ValueError(
                "Generated gate must define base = nn.Linear(model_dim, num_experts, bias=False)"
            )
        for input_shape in ((2, model_dim), (2, 3, model_dim)):
            sample = torch.randn(*input_shape)
            with torch.no_grad():
                output = gate(sample)
            expected = input_shape[:-1] + (num_experts,)
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected:
                raise ValueError(f"Expected gate output {expected}, got {getattr(output, 'shape', None)}")
            if not torch.isfinite(output).all():
                raise ValueError("Generated gate returned non-finite logits")


# ── LLM-driven gate generation ──────────────────────────────────────────────────

def _generate_gate(
    chat_bot: Any, shapes: list[tuple[int, int]], attempts: int,
    max_new_tokens: int, artifact_dir: Path, *, random_student: bool,
) -> str:
    initialization = (
        "The gate is a randomly initialized student trained beside a frozen native router. "
        "Do not assume its base weight is copied from the native router."
        if random_student
        else "Its base weight will be copied from the native router."
    )
    residual_requirement = (
        "Any residual branch may use normal random initialization."
        if random_student
        else "Initialize any residual branch's final projection to zero for native step-zero routing."
    )
    prompt = f"""
Write one complete Python module defining exactly one MoE gate class named LLMGeneratedGate.
Requirements:
- Import only torch and torch.nn.
- Inherit torch.nn.Module.
- Constructor: __init__(self, model_dim: int, num_experts: int).
- Forward: forward(self, x), accepting (..., model_dim) and returning finite raw logits (..., num_experts).
- Define self.base = nn.Linear(model_dim, num_experts, bias=False). {initialization}
- Return self.base(x), optionally with a small trainable residual branch.
- {residual_requirement}
- Do not apply softmax, top-k, expert dispatch, or auxiliary losses.
- Forward must be deterministic: do not sample random values or use dropout/batch normalization.
- Do not hard-code dimensions, move devices inside forward, or return tuples.
- Keep the gate small, differentiable, and numerically stable.
- Router shapes: {shapes!r}.
Output only complete source between <gate> and </gate>, without markdown or explanation.
""".strip()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    previous_error = ""
    for attempt in range(1, attempts + 1):
        current_prompt = prompt
        if previous_error:
            current_prompt += f"\nThe previous proposal failed validation: {previous_error}\nReturn a corrected implementation."
        _, _, _, raw = chat_bot.chat(
            current_prompt, engineer_prompt=False, max_new_tokens=max_new_tokens,
        )
        (artifact_dir / f"generation_attempt_{attempt}.txt").write_text(raw, encoding="utf-8")
        errors: list[str] = []
        for candidate in _gate_candidates(raw):
            try:
                _validate_gate_source(candidate, shapes)
                (artifact_dir / "gate.py").write_text(candidate.rstrip() + "\n", encoding="utf-8")
                return candidate
            except Exception as exc:
                errors.append(str(exc))
        previous_error = "; ".join(errors) or "No usable <gate> source was found"
    raise RuntimeError(f"LLM did not generate a valid gate after {attempts} attempts: {previous_error}")


# ── Helpers ─────────────────────────────────────────────────────────────────────

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


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.output or Path("out") / f"moe_gate_cycle_{timestamp}").resolve()
    os.environ["NNGPT_FORCE_DIRECT_GENERATE"] = "1"
    os.environ["NNGPT_DIR_OVERRIDE"] = str(run_root / "nngpt")

    if args.gate_mode == "teacher-student" and args.gate_init_noise_scale != 0:
        raise ValueError("Teacher-student mode keeps the teacher unchanged; use gate-init-noise-scale=0")
    if args.gate_mode == "teacher-student" and args.student_weight_start != 0:
        raise ValueError("Teacher-student startup must use student-weight-start=0")
    if not 0.0 <= args.student_weight_start <= 1.0:
        raise ValueError("student-weight-start must be between 0 and 1")
    if not 0.0 <= args.student_weight_step <= 1.0:
        raise ValueError("student-weight-step must be between 0 and 1")

    import torch
    import numpy as np

    from ab.gpt.util.Chatbot import ChatBot
    from ab.gpt.util.Const import conf_test_dir, conf_train_dir, epoch_dir, nngpt_dir
    from ab.gpt.util.Tune import nn_gen
    from moe_gate_only import (
        MoEGateSession,
        build_nngenprompt_dataloaders,
        collect_gate_metrics,
        collect_teacher_student_metrics,
        evaluate_language_model_loss,
        reset_teacher_student_metrics,
        teacher_student_distillation_loss,
        train_gates,
    )

    # --------------- setup ---------------

    _seed_all(args.seed)
    nngpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(epoch_dir(), ignore_errors=True)
    gate_dir = nngpt_dir / "gates" / "candidate_000"

    # --------------- load model ---------------

    model_kwargs: dict[str, Any] = {
        "torch_dtype": _dtype(args.dtype),
        "local_files_only": args.local_files_only,
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map

    session = MoEGateSession.from_pretrained(
        args.model,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"local_files_only": args.local_files_only},
    )
    tokenizer = session.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------- discover sites ---------------

    sites = session.inspect()
    if not sites:
        raise RuntimeError("No MoE gate sites discovered")
    shapes = sorted({(site.model_dim, site.num_experts) for site in sites})

    # --------------- generate gate source (one-shot, via LLM) ---------------

    chat_bot = ChatBot(
        session.model, tokenizer,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
    )
    gate_source = _generate_gate(
        chat_bot, shapes, args.gate_generation_attempts,
        args.gate_max_new_tokens, gate_dir,
        random_student=args.gate_mode == "teacher-student",
    )
    # --------------- install gates & freeze ---------------

    sample_input = tokenizer(
        "Generate an improved PyTorch neural network architecture.",
        return_tensors="pt",
    )
    prompt_dict = json.loads((conf_test_dir / "NN_gen.json").read_text(encoding="utf-8"))
    session.model.eval()
    native_logits = _model_logits(session.model, sample_input)

    with session:
        _seed_all(args.seed)
        session.replace_source(
            gate_source, class_name="LLMGeneratedGate",
            layers=args.layers, sample_input=sample_input, verify=True,
            top_k=args.router_top_k, allow_remote_code=True,
            dynamic_discovery=False,
            initialize_from_original=args.gate_mode == "direct",
            teacher_student=args.gate_mode == "teacher-student",
            student_weight=args.student_weight_start,
            distillation_temperature=args.distillation_temperature,
        )
        initialization_metrics = (
            _perturb_gate_weights(session.installs, args.gate_init_noise_scale, args.seed)
            if args.gate_mode == "direct"
            else [{
                "layer_index": install.site.layer_index,
                "student_parameter_count": sum(
                    parameter.numel() for parameter in install.student_gate.parameters()
                ),
                "initialization": "random_shadow_student",
            } for install in session.installs]
        )
        (gate_dir / "initialization_metrics.json").write_text(
            json.dumps(initialization_metrics, indent=2), encoding="utf-8",
        )
        replacement_logits = _model_logits(session.model, sample_input)
        max_abs_diff = float((native_logits - replacement_logits).abs().max().item())
        equivalent = bool(torch.allclose(native_logits, replacement_logits, rtol=1e-5, atol=1e-5))
        equivalence = {
            "equivalent": equivalent,
            "max_abs_logit_difference": max_abs_diff,
            "rtol": 1e-5,
            "atol": 1e-5,
            "gate_init_noise_scale": args.gate_init_noise_scale,
            "mode": args.gate_mode,
            "student_weight": session.student_weight(),
        }
        (gate_dir / "step_zero_equivalence.json").write_text(
            json.dumps(equivalence, indent=2), encoding="utf-8",
        )
        if args.gate_mode == "teacher-student" and not equivalent:
            raise RuntimeError(
                "Teacher-routed shadow mode changed native model logits: "
                f"max_abs_difference={max_abs_diff}"
            )
        if args.gate_mode == "direct" and args.gate_init_noise_scale == 0 and not equivalent:
            raise RuntimeError(
                "Replacement gates do not reproduce native step-zero model logits: "
                f"max_abs_difference={max_abs_diff}"
            )
        session.freeze_except_gates()

        for epoch in range(args.epochs):
            print(f"\n{'='*60}\n  EPOCH {epoch} / {args.epochs}\n{'='*60}\n")
            _seed_all(args.seed + epoch)
            epoch_path = epoch_dir(epoch)
            epoch_path.mkdir(parents=True, exist_ok=True)
            active_student_weight = session.student_weight()
            (epoch_path / "routing_mode.json").write_text(
                json.dumps({
                    "gate_mode": args.gate_mode,
                    "student_weight": active_student_weight,
                }, indent=2),
                encoding="utf-8",
            )

            # ── 1. Generate NN architectures + evaluate (like nn_gen in tune()) ──

            session.save(epoch_path / "gate_pre_generation")
            session.model.eval()
            session.model.config.use_cache = True
            if args.gradient_checkpointing and hasattr(session.model, "gradient_checkpointing_disable"):
                session.model.gradient_checkpointing_disable()

            nn_gen(
                epoch,
                epoch_path,
                chat_bot,
                tuple(args.conf_keys),
                args.nn_train_epochs,
                prompt_dict,
                args.test_nn,
                args.generation_max_new_tokens,
                True,
                args.nn_name_prefix,
                args.max_length,
                1,
            )

            # Preserve cycle results (nn_gen calls NNEval.main internally)
            cycle_src = nngpt_dir / "cycle_results.json"
            if cycle_src.is_file():
                shutil.copy2(cycle_src, epoch_path / "cycle_results.json")
                shutil.copy2(cycle_src, nngpt_dir / f"cycle_results_A{epoch}.json")
                cycle_results = json.loads(cycle_src.read_text(encoding="utf-8"))
                if not cycle_results.get("success") or int(
                    cycle_results.get("evaluation", {}).get("models_trained", 0)
                ) < 1:
                    raise RuntimeError(
                        f"A{epoch} produced no successfully trained CV models; "
                        "gate training was not started"
                    )
            else:
                raise RuntimeError(f"A{epoch} did not produce cycle_results.json")

            # Save gate checkpoint + routing stats
            session.save(epoch_path / "gate")
            (epoch_path / "routing_stats.json").write_text(
                json.dumps(collect_gate_metrics(session.installs), indent=2),
                encoding="utf-8",
            )
            if args.gate_mode == "teacher-student":
                (epoch_path / "teacher_student_stats.json").write_text(
                    json.dumps(collect_teacher_student_metrics(session.installs), indent=2),
                    encoding="utf-8",
                )

            # ── 2. Build training data from real evaluated LEMUR results ──

            train_loader, val_loader, train_dataset = build_nngenprompt_dataloaders(
                tokenizer,
                conf_train_dir / "NN_gen.json",
                context_length=args.max_length,
                max_prompts=args.max_prompts,
                max_new_tokens=args.generation_max_new_tokens,
                batch_size=args.batch_size,
                validation_fraction=args.validation_fraction,
                seed=args.seed + epoch,
            )

            # ── 3. Train gates on formatted NN-generation examples ──

            session.model.train()
            session.model.config.use_cache = False
            if args.gradient_checkpointing:
                if hasattr(session.model, "gradient_checkpointing_enable"):
                    if args.gate_mode == "teacher-student":
                        try:
                            session.model.gradient_checkpointing_enable(
                                gradient_checkpointing_kwargs={"use_reentrant": False}
                            )
                        except TypeError as exc:
                            raise RuntimeError(
                                "Teacher-student training requires non-reentrant "
                                "gradient checkpointing"
                            ) from exc
                    else:
                        session.model.gradient_checkpointing_enable()
                if (
                    args.gate_mode == "direct"
                    and hasattr(session.model, "enable_input_require_grads")
                ):
                    session.model.enable_input_require_grads()

            def distillation_loss(_model: Any):
                return args.distillation_weight * teacher_student_distillation_loss(
                    session.installs
                )

            def log_step(step: int, loss: float) -> None:
                message = f"epoch={epoch} gate_step={step:04d} total_loss={loss:.6f}"
                if args.gate_mode == "teacher-student":
                    metrics = collect_teacher_student_metrics(session.installs)["aggregate"]
                    message += (
                        f" distill_kl={metrics['mean_distillation_kl']}"
                        f" topk_overlap={metrics['mean_topk_overlap']}"
                    )
                print(message)

            if args.gate_mode == "teacher-student":
                reset_teacher_student_metrics(session.installs)
            result, optimizer = train_gates(
                session.model,
                session.installs,
                train_loader,
                steps=args.gate_train_steps,
                learning_rate=args.gate_learning_rate,
                validation_loader=None,
                validation_steps=None,
                auxiliary_loss_fn=(
                    distillation_loss if args.gate_mode == "teacher-student" else None
                ),
                on_step=log_step,
            )

            validation_loss = None
            if args.gate_mode == "teacher-student":
                reset_teacher_student_metrics(session.installs)
            if val_loader is not None:
                validation_loss = evaluate_language_model_loss(
                    session.model,
                    val_loader,
                    max_steps=args.validation_steps,
                )

            if args.gradient_checkpointing:
                if (
                    args.gate_mode == "direct"
                    and hasattr(session.model, "disable_input_require_grads")
                ):
                    session.model.disable_input_require_grads()
                if hasattr(session.model, "gradient_checkpointing_disable"):
                    session.model.gradient_checkpointing_disable()

            # Save post-training checkpoint + metrics
            session.save(epoch_path / "gate_post_train", optimizer=optimizer)
            training_metrics = {
                "epoch": epoch,
                "examples": len(train_dataset),
                "steps": result.steps,
                "mean_train_loss": result.mean_train_loss,
                "validation_loss": validation_loss,
            }
            if args.gate_mode == "teacher-student":
                latest_student_stats = collect_teacher_student_metrics(session.installs)
                training_metrics.update({
                    "mean_distillation_kl": latest_student_stats["aggregate"][
                        "mean_distillation_kl"
                    ],
                    "mean_topk_overlap": latest_student_stats["aggregate"][
                        "mean_topk_overlap"
                    ],
                    "minimum_topk_overlap": latest_student_stats["aggregate"][
                        "minimum_topk_overlap"
                    ],
                })
            (epoch_path / "gate_training_metrics.json").write_text(
                json.dumps(training_metrics, indent=2), encoding="utf-8",
            )
            (epoch_path / "routing_stats_post_train.json").write_text(
                json.dumps(collect_gate_metrics(session.installs), indent=2),
                encoding="utf-8",
            )

            if args.gate_mode == "teacher-student":
                student_stats = collect_teacher_student_metrics(session.installs)
                (epoch_path / "teacher_student_stats_post_train.json").write_text(
                    json.dumps(student_stats, indent=2), encoding="utf-8",
                )
                aggregate = student_stats["aggregate"]
                minimum_overlap = aggregate["minimum_topk_overlap"]
                maximum_kl = aggregate["maximum_distillation_kl"]
                approved = args.handoff_mode == "fixed" or (
                    args.handoff_mode == "guarded"
                    and validation_loss is not None
                    and math.isfinite(validation_loss)
                    and minimum_overlap is not None
                    and maximum_kl is not None
                    and minimum_overlap >= args.handoff_min_topk_overlap
                    and maximum_kl <= args.handoff_max_kl
                )
                previous_weight = float(session.student_weight() or 0.0)
                next_weight = previous_weight
                candidate_validation_loss = None
                rejection_reason = None
                if approved:
                    next_weight = min(1.0, previous_weight + args.student_weight_step)
                    session.set_student_weight(next_weight)
                    try:
                        handoff_logits = _model_logits(session.model.eval(), sample_input)
                        if not torch.isfinite(handoff_logits).all():
                            raise RuntimeError("handoff produced non-finite model logits")
                        if val_loader is not None:
                            candidate_validation_loss = evaluate_language_model_loss(
                                session.model,
                                val_loader,
                                max_steps=args.validation_steps,
                            )
                            if (
                                validation_loss is not None
                                and (
                                    not math.isfinite(candidate_validation_loss)
                                    or candidate_validation_loss
                                    > validation_loss
                                    + args.handoff_max_validation_loss_increase
                                )
                            ):
                                rejection_reason = "validation_loss_increase"
                                approved = False
                    except Exception:
                        session.set_student_weight(previous_weight)
                        next_weight = previous_weight
                        raise
                    if not approved:
                        session.set_student_weight(previous_weight)
                        next_weight = previous_weight
                decision = {
                    "mode": args.handoff_mode,
                    "approved": approved,
                    "minimum_topk_overlap": minimum_overlap,
                    "required_minimum_topk_overlap": args.handoff_min_topk_overlap,
                    "maximum_distillation_kl": maximum_kl,
                    "required_maximum_distillation_kl": args.handoff_max_kl,
                    "previous_student_weight": previous_weight,
                    "next_student_weight": next_weight,
                    "teacher_validation_loss": validation_loss,
                    "candidate_validation_loss": candidate_validation_loss,
                    "maximum_validation_loss_increase": (
                        args.handoff_max_validation_loss_increase
                    ),
                    "rejection_reason": rejection_reason,
                }
                (epoch_path / "handoff_decision.json").write_text(
                    json.dumps(decision, indent=2), encoding="utf-8",
                )
                session.save(epoch_path / "gate_post_handoff", optimizer=optimizer)

            print(
                f"[epoch={epoch}] train_loss={result.mean_train_loss:.6f} "
                f"val_loss={validation_loss}"
            )

    # --------------- final summary ---------------

    summary = {
        "model": args.model,
        "gate_mode": args.gate_mode,
        "gate_source": str(gate_dir / "gate.py"),
        "gate_shapes": shapes,
        "replaced_gates": len(sites if args.layers is None else [s for s in sites if s.layer_index in args.layers]),
        "epochs": args.epochs,
        "run_root": str(run_root),
    }
    for epoch in range(args.epochs):
        epoch_path = epoch_dir(epoch)
        training_path = epoch_path / "gate_training_metrics.json"
        cycle_path = epoch_path / "cycle_results.json"
        entry: dict[str, Any] = {"epoch": epoch}
        if training_path.is_file():
            entry["gate_training"] = json.loads(training_path.read_text(encoding="utf-8"))
        if cycle_path.is_file():
            entry["cv_results"] = json.loads(cycle_path.read_text(encoding="utf-8"))
        summary[f"A{epoch}"] = entry

    (nngpt_dir / "one_cycle_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(f"MoE gate cycle completed: {run_root}")


if __name__ == "__main__":
    main()
