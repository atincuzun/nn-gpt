"""Inner CV generation, evaluation, and gate-training epochs."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .context import (
    RunContext,
    _generation_prefixes,
    _log_cuda_memory,
    _nas_prefixes,
    _pinned_prompt_dict,
    _prepare_model_for_generation,
    _release_cuda_memory,
    _ts,
)
from .code_normalization import unescape_single_line_code
from .eval_safety import _moe_cv_eval_environment
from .feedback import _build_gate_feedback_summary, feedback_record
from .morphism import _seed_all
from .records import _harvest_epoch_metrics, _write_gate_record


def _normalize_escaped_generated_code(epoch_path: Path) -> int:
    """Unescape single-line ``new_nn.py`` payloads that still compile.

    Only files with no real newline are candidates, and the recovered text
    must parse before it replaces the model output, so legitimate backslashes
    in normal code are never rewritten.
    """
    fixed = 0
    for path in sorted((Path(epoch_path) / "synth_nn").glob("B*/new_nn.py")):
        original = path.read_text(encoding="utf-8")
        recovered = unescape_single_line_code(original)
        if recovered == original:
            continue
        try:
            compile(recovered, str(path), "exec")
        except SyntaxError:
            continue
        path.write_text(recovered, encoding="utf-8")
        fixed += 1
        print(
            f"[MOE CODE] {path.parent.name}: expanded escaped newlines in {path.name}"
        )
    return fixed


def _run_epochs(ctx: RunContext) -> list[Path]:
    """Run the generation/evaluation/gate-training cycle for each epoch."""
    import torch

    from ab.gpt.util.Const import conf_train_dir, nngpt_dir, synth_dir
    from ab.gpt.util.tune.Tune import _evaluate_epoch, nn_gen
    from moe_gate_only import (
        build_nngenprompt_dataloaders,
        evaluate_language_model_loss,
        train_gates,
    )

    args = ctx.args
    outer_mode = args.gate_outer_search
    session = ctx.session
    chat_bot = ctx.chat_bot
    tokenizer = ctx.tokenizer
    epoch_paths: list[Path] = []
    ctx.gate_epoch_metrics = []
    training_prefixes = _nas_prefixes(args)
    generation_prefixes = _generation_prefixes(args)
    train_prompt_path = conf_train_dir / args.train_prompt_config
    if not train_prompt_path.is_file():
        raise FileNotFoundError(f"Train prompt config does not exist: {train_prompt_path}")

    for epoch in range(args.epochs):
        print(
            f"\n{'='*60}\n  [{_ts()}] gate_{ctx.candidate_index:03d} "
            f"EPOCH {epoch} / {args.epochs}\n{'='*60}\n"
        )
        _seed_all(args.seed + epoch)
        if ctx.candidate_root is None:
            raise RuntimeError("Gate candidate was not prepared")
        epoch_started = time.monotonic()
        epoch_path = ctx.candidate_root / "epochs" / f"A{epoch}"
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

        # ── 1. Generate NN architectures + evaluate (like nn_gen in tune()) ──

        _prepare_model_for_generation(ctx)

        # Candidates are compared on CV accuracy, so a candidate's own gate
        # code/score must never leak into the prompt it is being scored on;
        # gate feedback therefore lives only in the outer proposal prompt and
        # the on-disk records.
        generation_prompt_dict = _pinned_prompt_dict(
            ctx.prompt_dict,
            args.conf_keys,
            dataset=args.dataset,
            nn_prefixes=generation_prefixes,
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
            legacy_prompt_selection=True,
        )
        _normalize_escaped_generated_code(epoch_path)
        with _moe_cv_eval_environment():
            _evaluate_epoch(
                epoch,
                epoch_path,
                args.nn_name_prefix,
                args.nn_train_epochs,
                False,
                custom_synth_dir=synth_dir(epoch_path),
                prm_json=args.fixed_eval_hyperparameters,
                batch=64,  # Fork fallback; seed and explicit overrides still win.
            )

        # Preserve cycle results written by NNEval.
        cycle_src = nngpt_dir / "cycle_results.json"
        if cycle_src.is_file():
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
        # Per-epoch CV receipt: without this line the measured accuracies
        # live only in the record and every post-run question
        # ("did this gate measure anything?") means file digging.
        _mean = gate_feedback["mean_accuracy"]
        _best = gate_feedback["best_accuracy"]
        print(
            f"[{_ts()}] [CYCLE] gate_{ctx.candidate_index:03d} epoch {epoch} CV: "
            f"generated={gate_feedback['n_generated']} trained={gate_feedback['n_trained']} "
            f"measured={gate_feedback['n_measured']}, "
            f"mean={_mean:.4f} best={_best:.4f}" if _mean is not None else
            f"[{_ts()}] [CYCLE] gate_{ctx.candidate_index:03d} epoch {epoch} CV: "
            f"generated={gate_feedback['n_generated']} trained={gate_feedback['n_trained']} "
            f"measured=0, mean=n/a best=n/a"
        )
        ctx.previous_feedback_summary = gate_feedback["summary"]

        # ── 2. Build training data from real evaluated LEMUR results ──

        try:
            train_loader, val_loader, train_dataset = build_nngenprompt_dataloaders(
                tokenizer,
                train_prompt_path,
                context_length=args.max_length,
                max_prompts=args.max_prompts,
                max_new_tokens=args.generation_max_new_tokens,
                batch_size=args.batch_size,
                validation_fraction=args.validation_fraction,
                seed=args.seed + epoch,
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
            epoch_metrics = _harvest_epoch_metrics(
                epoch, epoch_path, gate_feedback, cycle_results,
                cycle_models_trained, None, None, active_layers,
                duration_ns=int((time.monotonic() - epoch_started) * 1e9),
            )
            epoch_metrics.update({
                "training_skipped": True,
                "skip_reason": str(exc),
                "training": {
                    "epoch": epoch,
                    "active_gate_layers": sorted(active_layers),
                    "training_skipped": True,
                    "skip_reason": str(exc),
                    "current_cycle_cv_success": cycle_cv_success,
                    "current_cycle_models_trained": cycle_models_trained,
                    "prompt_feedback_injected": True,
                    "train_prompt_config": args.train_prompt_config,
                    "training_nn_prefixes": list(training_prefixes),
                },
                "feedback": feedback_record(gate_feedback),
                "cv": cycle_results,
            })
            ctx.gate_epoch_metrics.append(epoch_metrics)
            _write_gate_record(ctx, finalize=False)
            continue

        train_lengths = [len(row["input_ids"]) for row in train_dataset]
        train_length_stats = {
            "minimum": min(train_lengths),
            "maximum": max(train_lengths),
            "mean": sum(train_lengths) / len(train_lengths),
        }
        print(
            f"[epoch={epoch}] Gate-training sequences: examples={len(train_dataset)} "
            f"min={train_length_stats['minimum']} max={train_length_stats['maximum']} "
            f"mean={train_length_stats['mean']:.1f}; "
            f"gradient_checkpointing={args.gradient_checkpointing}"
        )

        # ── 3. Train gates on formatted NN-generation examples ──

        if outer_mode:
            # CV generation/evaluation consumes RNG; reset before gate training so
            # each candidate's gate update starts from the same sampling state.
            _seed_all(args.seed + epoch)
        session.model.train()
        session.model.config.use_cache = False
        if args.gradient_checkpointing:
            if hasattr(session.model, "gradient_checkpointing_enable"):
                session.model.gradient_checkpointing_enable()
            if hasattr(session.model, "enable_input_require_grads"):
                session.model.enable_input_require_grads()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pre_training_vram = _log_cuda_memory(
            f"epoch={epoch} before gate training"
        )

        def log_step(step: int, loss: float) -> None:
            print(f"epoch={epoch} gate_step={step:04d} total_loss={loss:.6f}")

        oom_recovered = False
        try:
            result, _ = train_gates(
                session.model,
                session.installs,
                train_loader,
                steps=args.gate_train_steps,
                learning_rate=args.gate_learning_rate,
                validation_loader=None,
                validation_steps=None,
                auxiliary_loss_fn=None,
                on_step=log_step,
            )
        except torch.cuda.OutOfMemoryError:
            _log_cuda_memory(f"epoch={epoch} gate-training OOM")
            print(
                "[VRAM] Gate-training OOM: 4-bit quantization reduces frozen "
                "weights but not attention/MLP activations. Harvesting the "
                "epochs measured so far and stopping this candidate early "
                "instead of discarding its score; reduce --max-length or "
                "--batch-size to avoid a recurrence."
            )
            _release_cuda_memory()
            oom_recovered = True
            result = None
        post_training_vram = _log_cuda_memory(
            f"epoch={epoch} after gate training"
        )

        validation_loss = None
        if val_loader is not None and not oom_recovered:
            try:
                validation_loss = evaluate_language_model_loss(
                    session.model,
                    val_loader,
                    max_steps=args.validation_steps,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"[epoch={epoch}] validation-loss evaluation OOM; skipping it")
                _release_cuda_memory()
                validation_loss = None

        if args.gradient_checkpointing:
            if hasattr(session.model, "disable_input_require_grads"):
                session.model.disable_input_require_grads()
            if hasattr(session.model, "gradient_checkpointing_disable"):
                session.model.gradient_checkpointing_disable()

        # Save the single latest trained checkpoint + metrics.
        _save_latest_weights(ctx)
        training_metrics = {
            "epoch": epoch,
            "active_gate_layers": sorted(active_layers),
            "examples": len(train_dataset),
            "sequence_lengths": train_length_stats,
            "gradient_checkpointing": args.gradient_checkpointing,
            "vram_before_training": pre_training_vram,
            "vram_after_training": post_training_vram,
            "steps": result.steps if result is not None else 0,
            "mean_train_loss": result.mean_train_loss if result is not None else None,
            "validation_loss": validation_loss,
            "oom_recovered": oom_recovered,
            "current_cycle_cv_success": cycle_cv_success,
            "current_cycle_models_trained": cycle_models_trained,
            "outer_gate_search": outer_mode,
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
        # ── Record this inner epoch for the outer gate score ──
        # Placed after training so the gate's own training loss is included.
        epoch_metrics = _harvest_epoch_metrics(
            epoch, epoch_path, gate_feedback, cycle_results,
            cycle_models_trained,
            result.mean_train_loss if result is not None else None,
            validation_loss,
            active_layers,
            duration_ns=int((time.monotonic() - epoch_started) * 1e9),
        )
        epoch_metrics.update({
            "training": training_metrics,
            "feedback": feedback_record(gate_feedback),
            "cv": cycle_results,
        })
        ctx.gate_epoch_metrics.append(epoch_metrics)
        _write_gate_record(ctx, finalize=False)

        if oom_recovered:
            # The candidate keeps the epochs it already measured; a truncated
            # protocol beats discarding a working gate (its score averages the
            # epochs that completed).
            print(
                f"[{_ts()}] [CYCLE] gate_{ctx.candidate_index:03d} epoch {epoch} "
                f"gate-training OOM recovered: stopping candidate early with "
                f"{len(ctx.gate_epoch_metrics)} harvested epoch(s)"
            )
            break
        print(
            f"[{_ts()}] [CYCLE] gate_{ctx.candidate_index:03d} epoch {epoch} "
            f"gate-training: train_loss={result.mean_train_loss:.6f} "
            f"val_loss={validation_loss}"
        )

    return epoch_paths
def _save_latest_weights(ctx: RunContext) -> Path | None:
    """Atomically replace ``<gate>/weights`` with the current trained gate.

    The previous checkpoint is renamed aside before the new one moves in, so
    a failure at any point leaves either the old or the new weights on disk
    — never neither.
    """
    from .gate_store import WEIGHTS_DIR

    if ctx.candidate_root is None:
        return None
    target = ctx.candidate_root / WEIGHTS_DIR
    temporary = ctx.candidate_root / f".{WEIGHTS_DIR}.tmp"
    backup = ctx.candidate_root / f".{WEIGHTS_DIR}.bak"
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        ctx.session.save(temporary)
        if not temporary.exists():
            return None
        if target.exists():
            target.rename(backup)
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if not target.exists() and backup.exists():
            backup.rename(target)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)
    return target
