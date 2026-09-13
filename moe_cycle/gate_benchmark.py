"""Frozen gate-selection inputs and post-training scoring, using shared NNEval.

This module does not patch shared generation, data preparation, or evaluation.
The selection set is adaptive search data, not an unbiased final test set.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .gate_store import canonical_json, fingerprint, summarize_scores


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Evaluation writes gate-selection CV models to the shared LEMUR database under
# this prefix. It must never collide with any configured training/generation
# prefix, or benchmark models would leak back into later candidate prompts and
# break matched comparisons.
BENCHMARK_PREFIX = "gate-selection"


def benchmark_prefix(args) -> str:
    prefixes = [*args.sft_nn_prefixes, *args.generation_nn_prefixes]
    collisions = [p for p in prefixes if p and BENCHMARK_PREFIX.startswith(p)]
    if collisions:
        raise ValueError(
            f"Gate benchmark prefix {BENCHMARK_PREFIX!r} collides with configured "
            f"prefixes {collisions}; choose an isolated prefix or adjust the corpus"
        )
    return BENCHMARK_PREFIX


def adapter_identity(path) -> dict | None:
    if path is None:
        return None
    path = Path(path)
    return {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix in {".json", ".bin", ".safetensors"}}


def protocol_settings(ctx) -> dict:
    args = ctx.args
    from ab.gpt.util.Const import conf_train_dir
    return {
        "version": 1, "model": args.model,
        "model_revision": getattr(ctx.session.model.config, "_commit_hash", None),
        "base_adapter": adapter_identity(args.adapter),
        "shapes": [list(s) for s in ctx.shapes],
        "layers": args.layers, "router_top_k": args.router_top_k,
        "dtype": args.dtype, "load_in_4bit": args.load_in_4bit, "load_in_8bit": args.load_in_8bit,
        "epochs": args.epochs, "gate_train_steps": args.gate_train_steps,
        "gate_learning_rate": args.gate_learning_rate, "batch_size": args.batch_size,
        "max_length": args.max_length, "max_prompts": args.max_prompts,
        "validation_fraction": args.validation_fraction, "validation_steps": args.validation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "progressive_unfreeze_descending": args.progressive_unfreeze_descending,
        "gate_init_noise_scale": args.gate_init_noise_scale, "seed": args.seed,
        "dataset": args.dataset, "nn_train_epochs": args.nn_train_epochs,
        "fixed_eval_hyperparameters": args.fixed_eval_hyperparameters,
        "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
        "generation_max_new_tokens": args.generation_max_new_tokens,
        "generation_max_input_length": args.generation_max_input_length,
        "conf_keys": args.conf_keys, "test_prompt": ctx.prompt_dict,
        "train_prompt": json.loads((conf_train_dir / args.train_prompt_config).read_text()),
        "training_prefixes": args.sft_nn_prefixes,
        "generation_prefixes": args.generation_nn_prefixes,
        "benchmark_size": args.gate_benchmark_size, "benchmark_seeds": args.gate_benchmark_seeds,
        "min_success_rate": args.gate_min_success_rate, "min_measured": args.gate_min_measured,
        "training_data_policy": "frozen_completion_rows_no_candidate_feedback",
        "comparison_prompt_policy": "frozen_no_candidate_feedback",
        "evaluation_seed_policy": "matched_generation_seeds_upstream_cv_randomness",
    }


def _snapshot_prompts(ctx) -> list[dict]:
    import ab.nn.api as lemur

    args = ctx.args
    prompts = []
    # Explicit prefixes exclude new search outputs; no mutating shared queries.
    for key in args.conf_keys:
        config = ctx.prompt_dict[key]
        if config.get("num_joint_nns", 1) != 1:
            raise ValueError("Gate benchmark currently requires single-reference CV prompts")
        data = lemur.data(
            only_best_accuracy=True, task=config["task"], dataset=args.dataset,
            nn_prefixes=tuple(args.generation_nn_prefixes),
        )
        if data.empty:
            raise ValueError("No source models available for gate benchmark")
        rows = data.groupby("nn").sample(n=1, random_state=args.seed)
        rows = rows.sample(frac=1, random_state=args.seed).head(args.gate_benchmark_size)
        for _, row in rows.iterrows():
            values = {item["para"]: row[item["value"]] for item in config["input_list"]}
            values.update(config.get("static_values", {}))
            values.update(gate_summary="", epoch=args.nn_train_epochs)
            prompt = "\n".join(config["prompt"]).format(**values)
            system = "\n".join(config.get("system", []))
            if args.generation_max_input_length:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
                size = len(ctx.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
                if size > args.generation_max_input_length:
                    raise ValueError("Frozen benchmark prompt exceeds generation input limit")
            prompts.append({
                "key": key, "system": system, "prompt": prompt,
                "metadata": {"nn": str(row["nn"]), "dataset": args.dataset,
                             "task": str(row["task"]), "metric": str(row["metric"]),
                             "epoch": args.nn_train_epochs},
            })
    if not prompts:
        raise ValueError("Gate benchmark has no prompts")
    return prompts


def prepare_benchmark(ctx) -> tuple[dict, dict]:
    """Save/reuse an immutable manifest including the actual tokenized corpus."""
    from ab.gpt.util.Const import conf_train_dir
    from moe_gate_only import build_nngenprompt_dataloaders

    args = ctx.args
    path = args.gate_benchmark or ctx.run_root / "gate_benchmark.json"
    settings = protocol_settings(ctx)
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["settings"] != settings:
            raise ValueError("Gate benchmark settings differ; use the original settings or a new manifest")
    else:
        _, val_loader, train_data = build_nngenprompt_dataloaders(
            ctx.tokenizer, conf_train_dir / args.train_prompt_config,
            context_length=args.max_length, max_prompts=args.max_prompts,
            max_new_tokens=args.generation_max_new_tokens, batch_size=args.batch_size,
            validation_fraction=args.validation_fraction, seed=args.seed,
            gate_summary="", dataset_name=args.dataset,
            nn_prefixes=tuple(args.sft_nn_prefixes),
        )
        manifest = {
            "settings": settings, "prompts": _snapshot_prompts(ctx),
            "train_rows": list(train_data),
            "validation_rows": list(val_loader.dataset) if val_loader is not None else [],
        }
        manifest["content_hash"] = fingerprint(manifest)
        write_json(path, manifest)
    payload = {key: value for key, value in manifest.items() if key != "content_hash"}
    if fingerprint(payload) != manifest["content_hash"]:
        raise ValueError("Gate benchmark content hash mismatch")
    protocol = {"settings": settings, "manifest_hash": manifest["content_hash"]}
    write_json(ctx.run_root / "gate_protocol.json", {"protocol_id": fingerprint(protocol), "protocol": protocol,
                                                 "manifest": str(path.resolve())})
    return manifest, protocol


def frozen_loaders(manifest: dict, tokenizer, batch_size: int, seed: int):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForTokenClassification

    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)
    train = manifest["train_rows"]
    validation = manifest["validation_rows"]
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collator,
                        generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(validation, batch_size=batch_size, collate_fn=collator) if validation else None
    return loader, val_loader, train


def read_outcomes(root: Path, count: int) -> list[dict]:
    outcomes = []
    for index in range(count):
        directory = root / "synth_nn" / f"B{index}"
        summary = directory / "eval_summary.json"
        outcome = {"candidate": f"B{index}", "status": "unavailable", "accuracy": None}
        if summary.is_file():
            try:
                entries = json.loads(summary.read_text())
                value = float(entries[-1].get("accuracy", entries[-1].get("acc")))
                if math.isfinite(value) and 0 <= value <= 1:
                    outcome.update(status="measured", accuracy=value)
            except (ValueError, TypeError, IndexError, KeyError):
                pass
        else:
            errors = [directory / name for name in ("error.txt", "eval_verification_failed.txt")]
            text = "\n".join(p.read_text(errors="replace") for p in errors if p.is_file())
            outcome["error"] = text[:2000] or "No per-candidate evaluation result"
            # Be conservative: system/GPU/worker failures are not bad architecture labels.
            infrastructure = any(s in text.lower() for s in (
                "out of memory", "cuda", "timeout", "timed out", "worker", "connection", "no space",
            ))
            if not infrastructure and ("gate_benchmark_invalid" in text or
                    (directory / "eval_verification_failed.txt").exists() or
                    any(s in text for s in ("NameError", "AttributeError", "SyntaxError", "TypeError",
                                           "size mismatch", "shapes cannot be multiplied", "lacks the required"))):
                outcome["status"] = "invalid"
        outcomes.append(outcome)
    return outcomes


def evaluate_candidate(ctx, manifest: dict) -> tuple[dict, list[dict], Path]:
    """Score the still-installed POST-training gates with proposer adapter absent."""
    import pandas as pd
    from ab.gpt.util.Tune import _evaluate_epoch
    from .morphism import _seed_all

    args = ctx.args
    root = ctx.candidate_root / "benchmark"
    if root.exists():
        raise FileExistsError(f"Refusing to reuse gate score artifacts: {root}")
    checkpoint = ctx.session.save(root / "gate_scored")
    ctx.session.model.eval()
    ctx.session.model.config.use_cache = True
    if hasattr(ctx.session.model, "gradient_checkpointing_disable"):
        ctx.session.model.gradient_checkpointing_disable()
    old_system = ctx.chat_bot.system_prompt
    count = 0
    try:
        for seed in args.gate_benchmark_seeds:
            for prompt_index, item in enumerate(manifest["prompts"]):
                directory = root / "synth_nn" / f"B{count}"
                directory.mkdir(parents=True)
                _seed_all(seed + prompt_index)
                ctx.chat_bot.system_prompt = item["system"]
                _, _, _, raw = ctx.chat_bot.chat(item["prompt"], engineer_prompt=False,
                                               max_new_tokens=args.generation_max_new_tokens)
                (directory / "full_output.txt").write_text(raw, encoding="utf-8")
                pd.Series(item["metadata"]).to_pickle(directory / "dataframe.df")
                write_json(directory / "benchmark_input.json", {"seed": seed + prompt_index,
                                                               "prompt_index": prompt_index})
                # NNEval/Postprocess consumes the original completion, as in nn_gen.
                from parse_nn_generation import extract_nn_code
                if not extract_nn_code(raw)[0]:
                    (directory / "error.txt").write_text("gate_benchmark_invalid: no NN code generated")
                count += 1
    finally:
        ctx.chat_bot.system_prompt = old_system
    _seed_all(args.seed)
    # Scoring must not clobber the candidate's real CV results: the shared
    # evaluator rewrites the global nngpt_dir/cycle_results.json.
    from ab.gpt.util.Const import nngpt_dir
    cycle_file = nngpt_dir / "cycle_results.json"
    saved_cycle = cycle_file.read_bytes() if cycle_file.is_file() else None
    try:
        _evaluate_epoch(
            0, root, benchmark_prefix(args), args.nn_train_epochs,
            False, custom_synth_dir=root / "synth_nn", prm_json=args.fixed_eval_hyperparameters,
        )
    finally:
        if saved_cycle is None:
            cycle_file.unlink(missing_ok=True)
        else:
            cycle_file.write_bytes(saved_cycle)
    outcomes = read_outcomes(root, count)
    score = summarize_scores(outcomes, min_success_rate=args.gate_min_success_rate,
                             min_measured=args.gate_min_measured)
    write_json(root / "score.json", {"score": score, "outcomes": outcomes,
                                    "checkpoint": str(checkpoint.resolve())})
    return score, outcomes, checkpoint
