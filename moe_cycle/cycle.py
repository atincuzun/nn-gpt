"""Iterative MoE gate-training NAS cycle mirroring Tune.py's tune() suite.

Each epoch:
  1. Generate NN architectures (nn_gen) using the gate-replaced LLM
  2. Evaluate generated NNs through Tune._evaluate_epoch
  3. Build NNGenPrompt data from real evaluated LEMUR results
  4. Train only generated MoE gates on language-model and router-distillation loss
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
from argparse import Namespace
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .cli import parse_args
from .feedback import _build_gate_feedback_summary
from .gate_source import _generate_gate, _validate_gate_source
from .morphism import (
    _capture_native_gate_inputs,
    _dtype,
    _model_logits,
    _perturb_gate_weights,
    _seed_all,
)


@dataclass
class RunContext:
    """Everything produced by ``_setup_run`` and consumed by the cycle phases."""

    args: Namespace
    run_root: Path
    session: Any
    tokenizer: Any
    chat_bot: Any
    sites: list[Any]
    shapes: list[tuple[int, int]]
    gate_dir: Path
    gate_source: str | None
    sample_input: Any
    prompt_dict: dict[str, Any]
    native_logits: Any
    native_gate_inputs: dict[int, Any]
    native_repeatable: bool
    native_repeat_max_abs_difference: float
    previous_feedback_summary: str
    used_prompts: list = field(default_factory=list)


def _nas_prefixes(args: Namespace) -> tuple[str, ...]:
    """Return stable LEMUR prefixes including this cycle's generated models."""
    values = [*args.sft_nn_prefixes, args.nn_name_prefix]
    return tuple(dict.fromkeys(value for value in values if value))


def _generation_prefixes(args: Namespace) -> tuple[str, ...]:
    """Return comparable generation seeds plus this cycle's successful models."""
    values = [*args.generation_nn_prefixes, args.nn_name_prefix]
    return tuple(dict.fromkeys(value for value in values if value))


def _prompt_dict_with_feedback(
    prompt_dict: dict[str, Any],
    conf_keys: tuple[str, ...] | list[str],
    feedback_summary: str,
    *,
    dataset: str,
    nn_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    """Render MoE-cycle feedback without modifying the shared nn_gen function."""
    rendered = copy.deepcopy(prompt_dict)
    escaped_feedback = feedback_summary.replace("{", "{{").replace("}", "}}")
    for key in conf_keys:
        if key not in rendered:
            raise KeyError(f"Prompt key {key!r} is missing from the test prompt config")
        key_config = rendered[key]
        key_config["dataset"] = dataset
        key_config["nn_prefixes"] = list(nn_prefixes)
        key_config["prompt"] = [
            line.replace("{gate_summary}", escaped_feedback)
            for line in key_config["prompt"]
        ]
    return rendered


def _initial_feedback_summary(gate_source: str | None) -> str:
    gate_code = gate_source or "[installed gate]"
    if len(gate_code) > 2000:
        gate_code = gate_code[:2000] + "\n# ... (truncated)"
    return (
        "No previous CV generation cycle has been evaluated. Establish a valid "
        "NAS baseline and prioritize executable LEMUR candidates.\n"
        f"The current MoE router gate is:\n<gate>\n{gate_code}\n</gate>"
    )


def _validate_args(args: Namespace) -> None:
    """Reject unsupported or contradictory CLI combinations early."""
    if args.repetition_penalty != 1.0:
        print("[WARN] --repetition-penalty is ignored by the upstream ChatBot")
    if args.generation_backend != "pipeline":
        print("[WARN] --generation-backend is ignored; upstream ChatBot selects its backend")
    if args.fixed_evaluation_prompts:
        print("[WARN] --fixed-evaluation-prompts is ignored; fixed prompt reuse is disabled")

    if args.gate_mode == "teacher-student" and args.gate_init_noise_scale != 0:
        raise ValueError("Teacher-student mode keeps the teacher unchanged; use gate-init-noise-scale=0")
    if args.gate_mode == "teacher-student" and args.student_weight_start != 0:
        raise ValueError("Teacher-student startup must use student-weight-start=0")
    if not 0.0 <= args.student_weight_start <= 1.0:
        raise ValueError("student-weight-start must be between 0 and 1")
    if not 0.0 <= args.student_weight_step <= 1.0:
        raise ValueError("student-weight-step must be between 0 and 1")
    if args.gate_mode == "teacher-student" and args.load_in_8bit:
        raise ValueError(
            "Teacher-student mode is unsupported with --load-in-8bit: the teacher "
            "router stores int8 weights, so distillation logits would be garbage"
        )


def _setup_run(args: Namespace) -> RunContext:
    """Load the model, discover gate sites, generate gate source, capture native behavior."""
    import torch

    from ab.gpt.util.Chatbot import ChatBot
    from ab.gpt.util.Const import conf_test_dir, epoch_dir, nngpt_dir
    from moe_gate_only import MoEGateSession

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.output or Path("out") / f"moe_gate_cycle_{timestamp}").resolve()
    os.environ["NNGPT_DIR_OVERRIDE"] = str(run_root / "nngpt")

    _seed_all(args.seed)
    nngpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(epoch_dir(), ignore_errors=True)
    gate_dir = nngpt_dir / "gates" / "candidate_000"

    # --------------- load model ---------------

    model_kwargs: dict[str, Any] = {
        "dtype": _dtype(args.dtype),
        "local_files_only": args.local_files_only,
    }
    if args.load_in_8bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--load-in-8bit requires the bitsandbytes package "
                "(pip install bitsandbytes)"
            ) from exc
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
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
    if args.adapter is not None:
        if not args.adapter.is_dir():
            raise FileNotFoundError(f"LoRA adapter directory does not exist: {args.adapter}")
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(
            args.adapter,
            local_files_only=args.local_files_only,
        )
        expected_base = str(peft_config.base_model_name_or_path)
        print(f"Merging LoRA adapter {args.adapter} trained from {expected_base}")
        adapted_model = PeftModel.from_pretrained(
            session.model,
            args.adapter,
            is_trainable=False,
            local_files_only=args.local_files_only,
        )
        session.model = adapted_model.merge_and_unload(safe_merge=True)

    # --------------- discover sites ---------------

    sites = session.inspect()
    if not sites:
        raise RuntimeError("No MoE gate sites discovered")
    shapes = sorted({(site.model_dim, site.num_experts) for site in sites})

    # --------------- obtain gate source (LLM-generated or external file) ---------------

    chat_bot = ChatBot(
        session.model, tokenizer,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
    )
    gate_dir.mkdir(parents=True, exist_ok=True)
    if args.gate_source is not None:
        if not args.gate_source.is_file():
            raise FileNotFoundError(f"Gate source file does not exist: {args.gate_source}")
        gate_source = args.gate_source.read_text(encoding="utf-8")
        _validate_gate_source(gate_source, shapes, class_name=args.gate_class)
        (gate_dir / "gate.py").write_text(gate_source.rstrip() + "\n", encoding="utf-8")
        print(f"Using externally supplied gate from {args.gate_source} "
              f"(class {args.gate_class}, {len(gate_source)} chars)")
    else:
        gate_source = _generate_gate(
            chat_bot, shapes, args.gate_generation_attempts,
            args.gate_max_new_tokens, gate_dir,
            random_student=args.gate_mode == "teacher-student" or args.load_in_8bit,
        )

    # --------------- capture native behavior ---------------

    sample_input = tokenizer(
        "Generate an improved PyTorch neural network architecture.",
        return_tensors="pt",
    )
    test_prompt_path = conf_test_dir / args.test_prompt_config
    if not test_prompt_path.is_file():
        raise FileNotFoundError(f"Test prompt config does not exist: {test_prompt_path}")
    prompt_dict = json.loads(test_prompt_path.read_text(encoding="utf-8"))
    session.model.eval()
    native_logits, native_gate_inputs = _capture_native_gate_inputs(
        session.model,
        sites,
        sample_input,
    )
    native_repeat_logits = _model_logits(session.model, sample_input)
    native_repeat_difference = (native_logits - native_repeat_logits).abs()
    native_repeatable = bool(torch.equal(native_logits, native_repeat_logits))
    native_repeat_max_abs_difference = float(native_repeat_difference.max().item())
    if not native_repeatable:
        raise RuntimeError(
            "Native model is not bit-repeatable before gate replacement: "
            f"max_abs_difference={native_repeat_max_abs_difference}"
        )

    return RunContext(
        args=args,
        run_root=run_root,
        session=session,
        tokenizer=tokenizer,
        chat_bot=chat_bot,
        sites=sites,
        shapes=shapes,
        gate_dir=gate_dir,
        gate_source=gate_source,
        sample_input=sample_input,
        prompt_dict=prompt_dict,
        native_logits=native_logits,
        native_gate_inputs=native_gate_inputs,
        native_repeatable=native_repeatable,
        native_repeat_max_abs_difference=native_repeat_max_abs_difference,
        previous_feedback_summary=_initial_feedback_summary(gate_source) if not args.no_gate_feedback else "",
    )


def _install_and_verify(ctx: RunContext) -> None:
    """Install replacement gates, verify step-zero equivalence, and freeze.

    Must run inside ``with ctx.session:`` (owned by ``main``) so the installed
    gates survive for ``_run_epochs``; the session context restores them when
    the whole cycle finishes.
    """
    import torch

    args = ctx.args
    session = ctx.session
    gate_dir = ctx.gate_dir

    _seed_all(args.seed)
    replace_kwargs = {
        "layers": args.layers,
        "sample_input": ctx.sample_input,
        "verify": True,
        "top_k": args.router_top_k,
        "allow_remote_code": True,
            "dynamic_discovery": False,
            "initialize_from_original": (
                args.gate_mode == "direct" and not args.load_in_8bit
            ),
        "teacher_student": args.gate_mode == "teacher-student",
        "student_weight": args.student_weight_start,
        "distillation_temperature": args.distillation_temperature,
    }
    session.replace_source(
        ctx.gate_source,
        class_name=(
            args.gate_class
            if args.gate_source is not None
            else "LLMGeneratedGate"
        ),
        **replace_kwargs,
    )
    if args.gate_mode == "direct":
        initialization_metrics = _perturb_gate_weights(
            session.installs, args.gate_init_noise_scale, args.seed
        )
    else:
        initialization_metrics = [{
            "layer_index": install.site.layer_index,
            "student_parameter_count": sum(
                parameter.numel() for parameter in install.student_gate.parameters()
            ),
            "initialization": "random_shadow_student",
        } for install in session.installs]
    (gate_dir / "initialization_metrics.json").write_text(
        json.dumps(initialization_metrics, indent=2), encoding="utf-8",
    )
    replacement_logits = _model_logits(session.model, ctx.sample_input)
    if not torch.isfinite(replacement_logits).all():
        raise RuntimeError("Replacement gates produced non-finite model logits")
    max_abs_diff = float((ctx.native_logits - replacement_logits).abs().max().item())
    equivalent = bool(torch.allclose(ctx.native_logits, replacement_logits, rtol=1e-5, atol=1e-5))
    equivalence = {
        "equivalent": equivalent,
        "max_abs_logit_difference": max_abs_diff,
        "rtol": 1e-5,
        "atol": 1e-5,
        "gate_init_noise_scale": args.gate_init_noise_scale,
        "mode": args.gate_mode,
        "load_in_8bit": args.load_in_8bit,
        "student_weight": session.student_weight(),
        "native_forward_bit_repeatable": ctx.native_repeatable,
        "native_repeat_max_abs_logit_difference": ctx.native_repeat_max_abs_difference,
    }
    (gate_dir / "step_zero_equivalence.json").write_text(
        json.dumps(equivalence, indent=2), encoding="utf-8",
    )
    if args.load_in_8bit:
        print(
            "[WARN] 8-bit quantized base: step-zero equivalence is not enforced "
            f"(measured equivalent={equivalent}, max_abs_diff={max_abs_diff:.6g})"
        )
    elif args.gate_mode == "teacher-student" and not equivalent:
        raise RuntimeError(
            "Teacher-routed shadow mode changed native model logits: "
            f"max_abs_difference={max_abs_diff}"
        )
    if (
        args.gate_mode == "direct"
        and args.gate_init_noise_scale == 0
        and not equivalent
    ):
        raise RuntimeError(
            "Replacement gates do not reproduce native step-zero model logits: "
            f"max_abs_difference={max_abs_diff}"
        )
    session.freeze_except_gates()


def _run_epochs(ctx: RunContext) -> list[Path]:
    """Run the generation/evaluation/gate-training cycle for each epoch."""
    import torch

    from ab.gpt.util.Const import conf_train_dir, epoch_dir, nngpt_dir, synth_dir
    from ab.gpt.util.Tune import _evaluate_epoch, nn_gen
    from moe_gate_only import (
        build_nngenprompt_dataloaders,
        collect_gate_metrics,
        collect_teacher_student_metrics,
        evaluate_language_model_loss,
        reset_teacher_student_metrics,
        teacher_student_distillation_loss,
        train_gates,
    )

    args = ctx.args
    session = ctx.session
    chat_bot = ctx.chat_bot
    tokenizer = ctx.tokenizer
    epoch_paths: list[Path] = []
    training_prefixes = _nas_prefixes(args)
    generation_prefixes = _generation_prefixes(args)
    train_prompt_path = conf_train_dir / args.train_prompt_config
    if not train_prompt_path.is_file():
        raise FileNotFoundError(f"Train prompt config does not exist: {train_prompt_path}")

    for epoch in range(args.epochs):
        print(f"\n{'='*60}\n  EPOCH {epoch} / {args.epochs}\n{'='*60}\n")
        if args.no_gate_feedback:
            print("[FEEDBACK] Gate feedback text omitted from prompts (--no-gate-feedback)")
        _seed_all(args.seed + epoch)
        epoch_path = epoch_dir(epoch)
        epoch_path.mkdir(parents=True, exist_ok=True)
        epoch_paths.append(epoch_path)
        if args.progressive_unfreeze_descending:
            descending_layers = sorted(
                {install.site.layer_index for install in session.installs},
                reverse=True,
            )
            active_layers = set(descending_layers[:epoch + 1])
            for install in session.installs:
                trainable = install.site.layer_index in active_layers
                for parameter in install.new_gate.parameters():
                    parameter.requires_grad_(trainable)
            if not active_layers:
                raise RuntimeError("Progressive gate training selected no active layers")
        else:
            active_layers = {
                install.site.layer_index for install in session.installs
            }
        (epoch_path / "active_gate_layers.json").write_text(
            json.dumps({
                "epoch": epoch,
                "progressive_unfreeze_descending": args.progressive_unfreeze_descending,
                "active_layers": sorted(active_layers),
            }, indent=2),
            encoding="utf-8",
        )
        active_student_weight = session.student_weight()
        (epoch_path / "routing_mode.json").write_text(
            json.dumps({
                "gate_mode": args.gate_mode,
                "gate_implementation": args.gate_implementation,
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

        gen_feedback = ctx.previous_feedback_summary if not args.no_gate_feedback else ""
        generation_prompt_dict = _prompt_dict_with_feedback(
            ctx.prompt_dict,
            args.conf_keys,
            gen_feedback,
            dataset=args.dataset,
            nn_prefixes=generation_prefixes,
        )
        (epoch_path / "generation_prompt_feedback.txt").write_text(
            ctx.previous_feedback_summary.rstrip() + "\n",
            encoding="utf-8",
        )
        nn_gen(
            epoch,
            epoch_path,
            chat_bot,
            tuple(args.conf_keys),
            args.nn_train_epochs,
            generation_prompt_dict,
            args.test_nn,
            args.generation_max_new_tokens,
            True,
            args.nn_name_prefix,
            args.generation_max_input_length,
            1,
        )
        _evaluate_epoch(
            epoch,
            epoch_path,
            args.nn_name_prefix,
            args.nn_train_epochs,
            False,
            custom_synth_dir=synth_dir(epoch_path),
        )

        # Preserve cycle results written by NNEval.
        cycle_src = nngpt_dir / "cycle_results.json"
        if cycle_src.is_file():
            shutil.copy2(cycle_src, epoch_path / "cycle_results.json")
            shutil.copy2(cycle_src, nngpt_dir / f"cycle_results_A{epoch}.json")
            cycle_results = json.loads(cycle_src.read_text(encoding="utf-8"))
            cycle_models_trained = int(
                cycle_results.get("evaluation", {}).get("models_trained", 0)
            )
            cycle_cv_success = bool(cycle_results.get("success")) and cycle_models_trained > 0
            if not cycle_cv_success:
                print(
                    f"[epoch={epoch}] No CV candidate trained successfully; "
                    "continuing gate training from existing valid LEMUR examples."
                )
        else:
            raise RuntimeError(f"A{epoch} did not produce cycle_results.json")

        # Preserve gate outcome feedback for the training prompts
        used_prompts = ctx.used_prompts
        gate_feedback = _build_gate_feedback_summary(
            epoch_path, used_prompts, cycle_results, session
        )
        (epoch_path / "nas_feedback.json").write_text(
            json.dumps(gate_feedback, indent=2),
            encoding="utf-8",
        )
        (epoch_path / "training_prompt_feedback.txt").write_text(
            gate_feedback["summary"].rstrip() + "\n",
            encoding="utf-8",
        )
        ctx.previous_feedback_summary = gate_feedback["summary"]

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

        try:
            train_gate_summary = gate_feedback["summary"] if not args.no_gate_feedback else ""
            train_loader, val_loader, train_dataset = build_nngenprompt_dataloaders(
                tokenizer,
                train_prompt_path,
                context_length=args.max_length,
                max_prompts=args.max_prompts,
                max_new_tokens=args.generation_max_new_tokens,
                batch_size=args.batch_size,
                validation_fraction=args.validation_fraction,
                seed=args.seed + epoch,
                gate_summary=train_gate_summary,
                dataset_name=args.dataset,
                nn_prefixes=training_prefixes,
            )
        except ValueError as exc:
            if "NNGenPrompt produced no usable examples" not in str(exc):
                raise
            print(
                f"[epoch={epoch}] Gate training skipped: {exc}. "
                "Continuing to the next generation cycle."
            )
            session.save(epoch_path / "gate_post_train")
            (epoch_path / "gate_training_metrics.json").write_text(
                json.dumps({
                    "epoch": epoch,
                    "active_gate_layers": sorted(active_layers),
                    "training_skipped": True,
                    "skip_reason": str(exc),
                    "current_cycle_cv_success": cycle_cv_success,
                    "current_cycle_models_trained": cycle_models_trained,
                    "prompt_feedback_injected": True,
                    "train_prompt_config": args.train_prompt_config,
                    "training_nn_prefixes": list(training_prefixes),
                    "cycle_feedback": {
                        key: gate_feedback[key]
                        for key in (
                            "n_generated",
                            "n_trained",
                            "n_measured",
                            "mean_accuracy",
                            "std_accuracy",
                            "best_accuracy",
                            "mean_accuracy_delta",
                            "improved_candidates",
                            "datasets",
                            "tasks",
                        )
                    },
                }, indent=2),
                encoding="utf-8",
            )
            continue

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
            "active_gate_layers": sorted(active_layers),
            "examples": len(train_dataset),
            "steps": result.steps,
            "mean_train_loss": result.mean_train_loss,
            "validation_loss": validation_loss,
            "current_cycle_cv_success": cycle_cv_success,
            "current_cycle_models_trained": cycle_models_trained,
            "gate_feedback_enabled": not args.no_gate_feedback,
            "prompt_feedback_injected": not args.no_gate_feedback,
            "train_prompt_config": args.train_prompt_config,
            "test_prompt_config": args.test_prompt_config,
            "training_nn_prefixes": list(training_prefixes),
            "generation_nn_prefixes": list(generation_prefixes),
            "cycle_feedback": {
                "n_generated": gate_feedback["n_generated"],
                "n_trained": gate_feedback["n_trained"],
                "n_measured": gate_feedback["n_measured"],
                "mean_accuracy": gate_feedback["mean_accuracy"],
                "std_accuracy": gate_feedback["std_accuracy"],
                "best_accuracy": gate_feedback["best_accuracy"],
                "mean_accuracy_delta": gate_feedback["mean_accuracy_delta"],
                "improved_candidates": gate_feedback["improved_candidates"],
                "datasets": gate_feedback["datasets"],
                "tasks": gate_feedback["tasks"],
            },
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
                    handoff_logits = _model_logits(session.model.eval(), ctx.sample_input)
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

    return epoch_paths


def _write_final_summary(ctx: RunContext, epoch_paths: list[Path]) -> None:
    """Write the aggregated per-epoch summary next to the run artifacts."""
    from ab.gpt.util.Const import nngpt_dir

    args = ctx.args
    summary = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "gate_mode": args.gate_mode,
        "gate_implementation": args.gate_implementation,
        "progressive_unfreeze_descending": args.progressive_unfreeze_descending,
        "generation_backend": "upstream_default",
        "fixed_evaluation_prompts": False,
        "gate_outcome_prompt_feedback": True,
        "train_prompt_config": args.train_prompt_config,
        "test_prompt_config": args.test_prompt_config,
        "dataset": args.dataset,
        "training_nn_prefixes": list(_nas_prefixes(args)),
        "generation_nn_prefixes": list(_generation_prefixes(args)),
        "gate_source": str(ctx.gate_dir / "gate.py") if ctx.gate_source is not None else None,
        "gate_shapes": ctx.shapes,
        "replaced_gates": len(
            ctx.sites
            if args.layers is None
            else [s for s in ctx.sites if s.layer_index in args.layers]
        ),
        "epochs": args.epochs,
        "run_root": str(ctx.run_root),
    }
    for epoch, epoch_path in enumerate(epoch_paths):
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
    print(f"MoE gate cycle completed: {ctx.run_root}")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    ctx = _setup_run(args)
    with ctx.session:
        _install_and_verify(ctx)
        epoch_paths = _run_epochs(ctx)
    _write_final_summary(ctx, epoch_paths)
