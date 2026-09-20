"""Iterative MoE gate-training NAS cycle mirroring Tune.py's tune() suite.

Each epoch:
  1. Generate NN architectures (nn_gen) using the gate-replaced LLM
  2. Evaluate generated NNs through Tune._evaluate_epoch
  3. Build NNGenPrompt data from real evaluated LEMUR results
  4. Train only the generated MoE gates on language-model loss
"""

from __future__ import annotations

import copy
import json
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
from .gate_store import structural_hash
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
    candidate_index: int = 0
    candidate_root: Path | None = None
    gate_root: Path | None = None
    gate_epoch_metrics: list = field(default_factory=list)
    reference_gate_id: int | None = None
    outer_summary: dict | None = None
    author_gate_id: int | None = None
    llm_version: str = "base"
    rematch_for: int | None = None


def _log_cuda_memory(label: str) -> dict[str, float] | None:
    """Print current-process CUDA memory so weight and activation use are visible."""
    import torch

    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    gib = 1024 ** 3
    status = {
        "allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "device_free_gib": free_bytes / gib,
        "device_total_gib": total_bytes / gib,
    }
    print(
        f"[VRAM] {label}: allocated={status['allocated_gib']:.2f} GiB, "
        f"reserved={status['reserved_gib']:.2f} GiB, "
        f"peak={status['peak_allocated_gib']:.2f} GiB, "
        f"device_free={status['device_free_gib']:.2f}/"
        f"{status['device_total_gib']:.2f} GiB"
    )
    return status


def _nas_prefixes(args: Namespace) -> tuple[str, ...]:
    """Return stable LEMUR prefixes including this cycle's generated models."""
    values = [*args.sft_nn_prefixes, args.nn_name_prefix]
    return tuple(dict.fromkeys(value for value in values if value))


def _generation_prefixes(args: Namespace) -> tuple[str, ...]:
    """Return comparable generation seeds plus this cycle's successful models."""
    values = [*args.generation_nn_prefixes, args.nn_name_prefix]
    return tuple(dict.fromkeys(value for value in values if value))


def _pinned_prompt_dict(
    prompt_dict: dict[str, Any],
    conf_keys: tuple[str, ...] | list[str],
    *,
    dataset: str,
    nn_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    """Pin the shared CV prompt config to the run's dataset and seed prefixes
    without modifying the shared nn_gen function.

    Gate feedback deliberately never enters CV generation or gate-training
    prompts: candidates are compared on CV accuracy, so a candidate's own gate
    code/score must never leak into the prompt it is scored or trained on.
    """
    rendered = copy.deepcopy(prompt_dict)
    for key in conf_keys:
        if key not in rendered:
            raise KeyError(f"Prompt key {key!r} is missing from the test prompt config")
        key_config = rendered[key]
        key_config["dataset"] = dataset
        key_config["nn_prefixes"] = list(nn_prefixes)
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
    if args.gate_random_init:
        if args.gate_source is None:
            raise ValueError("--gate-random-init requires --gate-source")
        if args.gate_init_noise_scale != 0:
            raise ValueError("--gate-random-init requires --gate-init-noise-scale 0")
    if args.gate_outer_search and args.gate_source is not None:
        raise ValueError(
            "--gate-outer-search generates gates itself; do not pass --gate-source"
        )
    if args.gate_author != "native" and not args.gate_outer_search:
        raise ValueError(
            "--gate-author last/best installs a trained gate while proposing and "
            "only makes sense with --gate-outer-search"
        )
    if getattr(args, "gate_phase_b", False):
        raise NotImplementedError(
            "--gate-phase-b is a reserved switch. Phase B is implemented as "
            "--gate-outer-sft (batched LoRA DPO/SFT on accumulated gate pairs); "
            "use that instead."
        )
    if getattr(args, "gate_outer_sft", False) and not args.gate_outer_search:
        raise ValueError(
            "--gate-outer-sft trains on outer-search records; requires --gate-outer-search"
        )
    if getattr(args, "gate_outer_sft", False):
        for name in ("gate_sft_every", "gate_sft_steps", "gate_sft_rank", "gate_min_pairs"):
            if getattr(args, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.repetition_penalty != 1.0:
        print("[WARN] --repetition-penalty is ignored by the upstream ChatBot")
    if args.generation_backend != "pipeline":
        print("[WARN] --generation-backend is ignored; upstream ChatBot selects its backend")
    if args.fixed_evaluation_prompts:
        print("[WARN] --fixed-evaluation-prompts is ignored; fixed prompt reuse is disabled")
    if args.gate_candidates < 1:
        raise ValueError("--gate-candidates must be at least 1")
    if args.max_length < 1:
        raise ValueError("--max-length must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if (
        not args.gradient_checkpointing
        and args.max_length > 1024
        and args.batch_size == 1
    ):
        print(
            "[WARN] Gate training has gradient checkpointing disabled with "
            f"--max-length={args.max_length}. Activation memory, not 4-bit "
            "weights, may exceed a 24-GiB GPU; use the default checkpointing "
            "or lower --max-length to 1024."
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
    # Derive the gates root from the RUNTIME nngpt_dir.  The module-level
    # nngpt_gate_dir in ab.gpt.util.Const is computed before the --output
    # override is applied, so using it would place gate records outside the
    # selected run directory.  --gate-store overrides it with a directory
    # shared across runs so the search accumulates instead of restarting.
    gate_root = (
        args.gate_store.resolve()
        if args.gate_store is not None
        else nngpt_dir / "gates"
    )
    gate_root.mkdir(parents=True, exist_ok=True)
    if args.gate_store is not None and any(gate_root.glob("gate_*")):
        print(
            f"[GATE SEARCH] persistent store: continuing from existing records in {gate_root}"
        )
    print(f"[GATE SEARCH] gate root: {gate_root}")

    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("--load-in-8bit and --load-in-4bit are mutually exclusive")
    if args.load_in_8bit or args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--load-in-8bit/--load-in-4bit require the bitsandbytes package "
                "(pip install bitsandbytes)"
            ) from exc
        from transformers import BitsAndBytesConfig

        # Routers must stay unquantized: exact weight-copy seeding of the
        # replacement gates (and step-zero bit-exact equivalence) depends on
        # reading the native router's true float weights.
        # NOTE: once any explicit skip list is supplied, transformers drops its
        # automatic output-head protection, so lm_head must be listed here too.
        keep_unquantized = ["gate", "lm_head"]

    model_kwargs: dict[str, Any] = {
        # ``torch_dtype`` is the Transformers ``from_pretrained`` API name.
        # In particular, DeepSeek-V2's remote-code constructor does not accept
        # the newer/internal ``dtype`` spelling and forwards unknown kwargs to
        # ``DeepseekV2ForCausalLM.__init__``.
        "torch_dtype": _dtype(args.dtype),
        "local_files_only": args.local_files_only,
    }
    if args.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=list(keep_unquantized),
        )
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_dtype(args.dtype),
            llm_int8_skip_modules=list(keep_unquantized),
        )
    if args.device_map.lower() != "none":
        device_map = args.device_map
        if (
            device_map == "auto"
            and (args.load_in_8bit or args.load_in_4bit)
            and torch.cuda.device_count() == 1
        ):
            # Accelerate's automatic mapper reserves enough headroom to push
            # the final DeepSeek layers to CPU on a 24-GiB card.  The complete
            # quantized model fits on this GPU, and gate replacement/training
            # requires real (non-meta) router tensors on every layer.
            device_map = "cuda:0"
        model_kwargs["device_map"] = device_map

    session = MoEGateSession.from_pretrained(
        args.model,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"local_files_only": args.local_files_only},
    )
    _log_cuda_memory("model loaded")
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

    # Phase B: wrap the backbone with the proposer LoRA BEFORE the ChatBot is
    # constructed, so generation (proposals and CV NN code) flows through the
    # adapter for the rest of the run. Adapter parameters stay frozen except
    # during Phase B training batches.
    merge_from = None
    merged_from_version = None
    if getattr(args, "gate_outer_sft", False):
        # Tune.py L1139-1142 pattern: continue from the newest persisted
        # proposer adapter so outer-loop weights compound across runs.
        if args.gate_store is not None and not args.gate_fresh_proposer:
            persisted = args.gate_store.resolve() / "proposer"
            if (persisted / "adapter" / "adapter_config.json").is_file():
                merge_from = persisted / "adapter"
                lineage_file = persisted / "lineage.json"
                if lineage_file.is_file():
                    try:
                        merged_from_version = json.loads(
                            lineage_file.read_text(encoding="utf-8")
                        ).get("version")
                    except (OSError, ValueError):
                        merged_from_version = None
                print(
                    f"[GATE SEARCH] outer-loop SFT: continuing proposer from {merge_from}"
                )
        from .phase_b import wrap_proposer_with_lora

        wrap_proposer_with_lora(session, args, merge_from=merge_from)
    from .phase_b import llm_version

    proposer_version = llm_version(session.model, args, merged_from=merged_from_version)

    chat_bot = ChatBot(
        session.model, tokenizer,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
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
        gate_dir=gate_root / "gate_000" / "gate_source",
        gate_source=None,
        sample_input=sample_input,
        prompt_dict=prompt_dict,
        native_logits=native_logits,
        native_gate_inputs=native_gate_inputs,
        native_repeatable=native_repeatable,
        native_repeat_max_abs_difference=native_repeat_max_abs_difference,
        previous_feedback_summary="",
        gate_root=gate_root,
        llm_version=proposer_version,
    )


def _prepare_gate_candidate(ctx: RunContext, candidate_index: int, *,
                            reference_source: str = "",
                            reference_accuracy: float | None = None,
                            goal_accuracy: float | None = None,
                            dataset: str | None = None,
                            proposal_chat: Any = None,
                            seen_hashes: set[str] | None = None) -> None:
    """Generate/load one gate architecture while the native router is active.

    The prompt carries one reference gate with its measured accuracy and the
    goal accuracy (LEMUR-CV pairing). With no scored prior gate it bootstraps
    from the baseline contract.
    """
    from .gate_store import gate_dir

    args = ctx.args
    ctx.candidate_index = candidate_index
    ctx.candidate_root = gate_dir(ctx.gate_root, candidate_index)
    ctx.gate_dir = ctx.candidate_root / "gate_source"
    ctx.gate_dir.mkdir(parents=True, exist_ok=True)
    if args.gate_source is not None:
        if not args.gate_source.is_file():
            raise FileNotFoundError(f"Gate source file does not exist: {args.gate_source}")
        source = args.gate_source.read_text(encoding="utf-8")
        _validate_gate_source(
            source, ctx.shapes, class_name=args.gate_class,
            require_base=not args.gate_random_init,
        )
    else:
        source = _generate_gate(
            proposal_chat or ctx.chat_bot, ctx.shapes, args.gate_generation_attempts,
            args.gate_max_new_tokens, ctx.gate_dir,
            reference_source=reference_source,
            reference_accuracy=reference_accuracy,
            goal_accuracy=goal_accuracy,
            dataset=dataset,
            seen_hashes=seen_hashes,
        )
    (ctx.candidate_root / "gate.py").write_text(source.rstrip() + "\n", encoding="utf-8")
    ctx.gate_source = source
    ctx.previous_feedback_summary = _initial_feedback_summary(source)
    ctx.used_prompts.clear()


def _prepare_model_for_generation(ctx: RunContext) -> None:
    """Put the model into the state generation runs under (eval, KV cache, no
    activation checkpointing), both for CV generation and gate proposals."""
    session = ctx.session
    session.model.eval()
    session.model.config.use_cache = True
    if ctx.args.gradient_checkpointing and hasattr(session.model, "gradient_checkpointing_disable"):
        session.model.gradient_checkpointing_disable()


def _select_author_gate_id(mode: str, gate_id: int, gate_root) -> int | None:
    """Which trained gate (if any) authors round ``gate_id``'s proposal.

    ``native`` (and round 0) propose with the original routers. ``last`` chains
    to the previous candidate. ``best`` installs the highest-scoring eligible
    gate, falling back to native while the store has no eligible gate yet.
    """
    if mode == "native" or gate_id == 0:
        return None
    if mode == "last":
        return gate_id - 1
    if mode == "best":
        from .gate_store import best_gate

        reference = best_gate(gate_root)
        return reference.get("gate_id") if reference is not None else None
    raise ValueError(f"unknown gate author mode {mode!r}")


def _install_author_gate(ctx: RunContext, author_gate_id: int) -> None:
    """Install a previously trained gate so the LLM authors the next proposal
    with that routing active.

    No equivalence checks: trained weights are not bit-exact with the native
    router by design. Must run inside ``with ctx.session:`` with no gates
    currently installed.
    """
    args = ctx.args
    from .gate_store import gate_dir as store_gate_dir

    author_root = store_gate_dir(ctx.gate_root, author_gate_id)
    checkpoint = author_root / "epochs" / f"A{args.epochs - 1}" / "gate_post_train"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Author gate checkpoint not found: {checkpoint}")
    session = ctx.session
    session.replace_source(
        (author_root / "gate.py").read_text(encoding="utf-8"),
        class_name="LLMGeneratedGate",
        layers=args.layers,
        sample_input=ctx.sample_input,
        verify=False,
        top_k=args.router_top_k,
        allow_remote_code=True,
        dynamic_discovery=False,
        initialize_from_original=True,
    )
    session.load_weights(checkpoint)
    print(
        f"[GATE AUTHOR] gate {args.gate_author} mode: next proposal is authored "
        f"under trained gate {author_gate_id:03d} ({checkpoint})"
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
        "initialize_from_original": not args.gate_random_init,
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
    initialization_metrics = [] if args.gate_random_init else _perturb_gate_weights(
        session.installs, args.gate_init_noise_scale, args.seed
    )
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
        "initialize_from_original": not args.gate_random_init,
        "load_in_8bit": args.load_in_8bit,
        "native_forward_bit_repeatable": ctx.native_repeatable,
        "native_repeat_max_abs_logit_difference": ctx.native_repeat_max_abs_difference,
    }
    (gate_dir / "step_zero_equivalence.json").write_text(
        json.dumps(equivalence, indent=2), encoding="utf-8",
    )
    skip_step_zero_verify = getattr(args, "gate_skip_step_zero_verify", False)
    if not equivalent and skip_step_zero_verify:
        print(
            "[GATE VERIFY] step-zero mismatch accepted (--gate-skip-step-zero-verify): "
            f"max_abs_difference={max_abs_diff}"
        )
    if (
        not args.gate_random_init
        and args.gate_init_noise_scale == 0
        and not equivalent
        and not skip_step_zero_verify
    ):
        raise RuntimeError(
            "Replacement gates do not reproduce native step-zero model logits: "
            f"max_abs_difference={max_abs_diff}"
        )
    session.freeze_except_gates()


def _run_epochs(ctx: RunContext) -> list[Path]:
    """Run the generation/evaluation/gate-training cycle for each epoch."""
    import torch

    from ab.gpt.util.Const import conf_train_dir, nngpt_dir, synth_dir
    from ab.gpt.util.Tune import _evaluate_epoch, nn_gen
    from moe_gate_only import (
        build_nngenprompt_dataloaders,
        collect_gate_metrics,
        evaluate_language_model_loss,
        train_gates,
    )

    from .gate_store import epoch_dir, write_json

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
        print(f"\n{'='*60}\n  EPOCH {epoch} / {args.epochs}\n{'='*60}\n")
        _seed_all(args.seed + epoch)
        if ctx.candidate_root is None:
            raise RuntimeError("Gate candidate was not prepared")
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
        (epoch_path / "active_gate_layers.json").write_text(
            json.dumps({
                "epoch": epoch,
                "progressive_unfreeze_descending": args.progressive_unfreeze_descending,
                "active_layers": sorted(active_layers),
            }, indent=2),
            encoding="utf-8",
        )
        (epoch_path / "routing_mode.json").write_text(
            json.dumps({
                "gate_implementation": args.gate_implementation,
                "initialize_from_original": not args.gate_random_init,
            }, indent=2),
            encoding="utf-8",
        )
        (epoch_path / "evaluation_config.json").write_text(
            json.dumps({
                "fixed_hyperparameter_overrides": args.fixed_eval_hyperparameters,
                "nn_train_epochs": args.nn_train_epochs,
            }, indent=2),
            encoding="utf-8",
        )

        # ── 1. Generate NN architectures + evaluate (like nn_gen in tune()) ──

        session.save(epoch_path / "gate_pre_generation")
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
            prm_json=args.fixed_eval_hyperparameters,
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

        try:
            result, optimizer = train_gates(
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
                "weights but not attention/MLP activations. Keep gradient "
                "checkpointing enabled (the default), reduce --max-length, "
                "or reduce --batch-size."
            )
            raise
        post_training_vram = _log_cuda_memory(
            f"epoch={epoch} after gate training"
        )

        validation_loss = None
        if val_loader is not None:
            validation_loss = evaluate_language_model_loss(
                session.model,
                val_loader,
                max_steps=args.validation_steps,
            )

        if args.gradient_checkpointing:
            if hasattr(session.model, "disable_input_require_grads"):
                session.model.disable_input_require_grads()
            if hasattr(session.model, "gradient_checkpointing_disable"):
                session.model.gradient_checkpointing_disable()

        # Save post-training checkpoint + metrics
        session.save(epoch_path / "gate_post_train", optimizer=optimizer)
        training_metrics = {
            "epoch": epoch,
            "active_gate_layers": sorted(active_layers),
            "examples": len(train_dataset),
            "sequence_lengths": train_length_stats,
            "gradient_checkpointing": args.gradient_checkpointing,
            "vram_before_training": pre_training_vram,
            "vram_after_training": post_training_vram,
            "steps": result.steps,
            "mean_train_loss": result.mean_train_loss,
            "validation_loss": validation_loss,
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
        (epoch_path / "gate_training_metrics.json").write_text(
            json.dumps(training_metrics, indent=2), encoding="utf-8",
        )
        (epoch_path / "routing_stats_post_train.json").write_text(
            json.dumps(collect_gate_metrics(session.installs), indent=2),
            encoding="utf-8",
        )

        # ── Record this inner epoch for the outer gate score ──
        # Placed after training so the gate's own training loss is included.
        if outer_mode:
            epoch_metrics = _harvest_epoch_metrics(
                epoch, epoch_path, gate_feedback, cycle_results,
                cycle_models_trained, result.mean_train_loss, validation_loss,
                active_layers,
            )
            ctx.gate_epoch_metrics.append(epoch_metrics)
            write_json(
                epoch_dir(ctx.gate_root, ctx.candidate_index, epoch) / "metrics.json",
                epoch_metrics,
            )

        print(
            f"[epoch={epoch}] train_loss={result.mean_train_loss:.6f} "
            f"val_loss={validation_loss}"
        )

    return epoch_paths


def _harvest_epoch_metrics(
    epoch: int,
    epoch_path: Path,
    gate_feedback: dict,
    cycle_results: dict,
    models_trained: int,
    train_loss: float | None,
    validation_loss: float | None,
    active_layers: set[int],
) -> dict:
    """Collect one inner epoch's measurements for the outer gate score.

    ``cv_accuracy_mean`` averages the successfully measured CV candidates, which
    is robust to a single lucky model.  ``cv_accuracy_best`` is retained for
    reference.  Failures are counted, never scored as zero.
    """
    accuracies = [
        outcome["accuracy"]
        for outcome in gate_feedback.get("candidate_outcomes", [])
        if outcome.get("status") == "measured" and outcome.get("accuracy") is not None
    ]
    n_generated = int(gate_feedback.get("n_generated", 0))
    mean = sum(accuracies) / len(accuracies) if accuracies else None
    return {
        "epoch": epoch,
        "cv_accuracy_mean": mean,
        "cv_accuracy_best": max(accuracies) if accuracies else None,
        "n_measured": len(accuracies),
        "n_attempted": n_generated,
        "n_trained": int(models_trained),
        "n_generated": n_generated,
        "mean_accuracy_delta": gate_feedback.get("mean_accuracy_delta"),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "active_gate_layers": sorted(active_layers),
        "cycle_success": bool(cycle_results.get("success")),
        "epoch_path": str(epoch_path),
    }


def _write_gate_record(ctx: RunContext) -> dict:
    """Finalize one gate candidate: summarize, persist, and return the record."""
    from .gate_store import summarize_epoch_metrics, write_gate_summary

    score = summarize_epoch_metrics(ctx.gate_epoch_metrics)
    record = {
        "gate_id": ctx.candidate_index,
        "gate_code": ctx.gate_source,
        "structural_hash": structural_hash(ctx.gate_source),
        "reference_gate_id": ctx.reference_gate_id,
        "author_mode": ctx.args.gate_author,
        "author_gate_id": ctx.author_gate_id,
        "rematch_for": getattr(ctx, "rematch_for", None),
        "llm_version": getattr(ctx, "llm_version", "base"),
        "seeded": getattr(ctx, "seeded", False),
        # gate_pairs.build_gate_pairs groups comparable records on exactly
        # these three keys; without them no Phase B pairs can ever form.
        "task": "img-classification",
        "dataset": getattr(ctx.args, "dataset", "cifar-10"),
        "metric": "accuracy",
        "score": score,
        "epochs": ctx.gate_epoch_metrics,
    }
    write_gate_summary(ctx.gate_root, ctx.candidate_index, record)
    return record


def _candidate_feedback(epoch_paths: list[Path], candidate_index: int) -> str:
    """Summarize one completed candidate for the next outer-loop proposal."""
    entries = []
    for epoch_path in epoch_paths:
        path = epoch_path / "nas_feedback.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            entries.append({key: data.get(key) for key in (
                "n_generated", "n_trained", "best_accuracy", "mean_accuracy_delta",
                "improved_candidates",
            )})
    return f"Independent gate candidate {candidate_index} results: {json.dumps(entries)}"


def _write_final_summary(ctx: RunContext, candidate_epochs: list[list[Path]]) -> None:
    """Write the aggregated per-epoch summary next to the run artifacts."""
    from ab.gpt.util.Const import nngpt_dir

    args = ctx.args
    summary = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "gate_implementation": args.gate_implementation,
        "progressive_unfreeze_descending": args.progressive_unfreeze_descending,
        "generation_backend": "upstream_default",
        "fixed_evaluation_prompts": False,
        "train_prompt_config": args.train_prompt_config,
        "test_prompt_config": args.test_prompt_config,
        "dataset": args.dataset,
        "fixed_evaluation_hyperparameters": args.fixed_eval_hyperparameters,
        "nn_train_epochs": args.nn_train_epochs,
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
        "gate_candidates": args.gate_candidates,
        "gate_author": args.gate_author,
        "gate_outer_search": ctx.outer_summary,
        "run_root": str(ctx.run_root),
    }
    for candidate_index, epoch_paths in enumerate(candidate_epochs):
        candidate_root = nngpt_dir / "gate_candidates" / f"candidate_{candidate_index:03d}"
        candidate: dict[str, Any] = {
            "gate_source": str(candidate_root / "gate_source" / "gate.py"),
            "independent_native_initialization": True,
        }
        for epoch, epoch_path in enumerate(epoch_paths):
            training_path = epoch_path / "gate_training_metrics.json"
            cycle_path = epoch_path / "cycle_results.json"
            entry: dict[str, Any] = {"epoch": epoch}
            if training_path.is_file():
                entry["gate_training"] = json.loads(training_path.read_text(encoding="utf-8"))
            if cycle_path.is_file():
                entry["cv_results"] = json.loads(cycle_path.read_text(encoding="utf-8"))
            candidate[f"A{epoch}"] = entry
        summary[f"candidate_{candidate_index:03d}"] = candidate

    (nngpt_dir / "one_cycle_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(f"MoE gate cycle completed: {ctx.run_root}")


def _propose_gate(ctx: RunContext, gate_id: int, seen_hashes: set[str]) -> None:
    """Propose the next gate under the currently installed routing.

    With ``--gate-author native`` (and for round 0) that is the native router.
    With ``last``/``best`` the caller has already installed the corresponding
    trained gate, so the proposal is authored by the improved composite.

    The prompt is one LEMUR-CV pairing: the best prior gate's code with its
    measured accuracy as the reference, its score as the goal. Round 0 (or an
    empty store) bootstraps from the baseline contract without scores.
    """
    from .gate_prompt import BASELINE_GATE_CODE
    from .gate_store import best_gate

    reference = best_gate(ctx.gate_root)
    reference_source = ""
    reference_accuracy = None
    goal_accuracy = None
    ctx.reference_gate_id = None
    if reference is not None:
        reference_source = reference.get("gate_code") or ""
        ctx.reference_gate_id = reference.get("gate_id")
        score = (reference.get("score") or {}).get("accuracy")
        if isinstance(score, float):
            reference_accuracy = score
            goal_accuracy = score
    elif gate_id > 0:
        # No eligible prior gate yet: fall back to the baseline contract so the
        # model still has an explicit reference to diverge from.
        reference_source = BASELINE_GATE_CODE

    _prepare_gate_candidate(
        ctx,
        gate_id,
        reference_source=reference_source,
        reference_accuracy=reference_accuracy,
        goal_accuracy=goal_accuracy,
        dataset=ctx.args.dataset,
        seen_hashes=seen_hashes,
    )
    print(
        f"[GATE SEARCH] proposed gate {gate_id:03d} "
        f"(reference={ctx.reference_gate_id if ctx.reference_gate_id is not None else 'baseline'}, "
        f"distinct architectures so far={len(seen_hashes)})"
    )


def _prepare_seed_candidate(ctx: RunContext, gate_id: int) -> None:
    """Cold-start bootstrap: install a built-in compliant seed gate.

    Seeds bypass the LLM proposal entirely so the store accumulates scored,
    valid, creative references even when the proposer cannot produce one.
    Seeds are validated with the same rules as LLM proposals and fail loudly
    if they ever stop complying.
    """
    from .gate_seeds import seed_gate_name, seed_gate_source
    from .gate_store import gate_dir

    ctx.candidate_index = gate_id
    ctx.candidate_root = gate_dir(ctx.gate_root, gate_id)
    ctx.gate_dir = ctx.candidate_root / "gate_source"
    ctx.gate_dir.mkdir(parents=True, exist_ok=True)
    source = seed_gate_source(gate_id)
    _validate_gate_source(source, ctx.shapes)
    (ctx.candidate_root / "gate.py").write_text(source.rstrip() + "\n", encoding="utf-8")
    (ctx.gate_dir / "proposal_prompt.txt").write_text(
        f"[cold-start bootstrap] built-in seed gate '{seed_gate_name(gate_id)}' "
        "installed without an LLM proposal\n",
        encoding="utf-8",
    )
    ctx.gate_source = source
    ctx.seeded = True
    ctx.previous_feedback_summary = _initial_feedback_summary(source)
    ctx.used_prompts.clear()
    print(
        f"[GATE SEARCH] gate {gate_id:03d} seeded (cold-start bootstrap): "
        f"{seed_gate_name(gate_id)}"
    )


def _record_failed_candidate(ctx: RunContext, gate_id: int, exc: Exception) -> None:
    """Persist a failed candidate so the run continues and the store keeps the lesson."""
    from .gate_store import write_gate_summary

    reason = f"{type(exc).__name__}: {exc}"
    print(f"[GATE SEARCH] gate {gate_id:03d} FAILED, continuing: {reason}")
    record = {
        "gate_id": gate_id,
        "gate_code": ctx.gate_source,
        "structural_hash": structural_hash(ctx.gate_source) if ctx.gate_source else None,
        "reference_gate_id": ctx.reference_gate_id,
        "author_mode": ctx.args.gate_author,
        "author_gate_id": ctx.author_gate_id,
        "rematch_for": ctx.rematch_for,
        "llm_version": ctx.llm_version,
        "seeded": getattr(ctx, "seeded", False),
        "score": {
            "objective": "mean_of_epoch_means_v1",
            "accuracy": None,
            "eligible": False,
            "n_measured": 0,
            "n_attempted": 0,
        },
        "epochs": [],
        "failure": reason[:2000],
    }
    write_gate_summary(ctx.gate_root, gate_id, record)


def _run_candidate_once(
    ctx: RunContext, gate_id: int, seen_hashes: set[str], *,
    rematch_for: int | None = None, seed: bool = False,
) -> list[Path]:
    """One outer candidate: author selection, proposal, install, inner loop.

    ``rematch_for`` re-runs a prior gate's code without a new proposal
    (king-of-the-hill re-measure after a Phase B batch). Failures never
    propagate: the candidate is recorded as failed and an empty list returns
    so the search continues.
    """
    args = ctx.args
    ctx.rematch_for = rematch_for
    # Reset per-candidate state: a failed proposal must not leave the previous
    # candidate's code/id in the failed record.
    ctx.gate_source = None
    ctx.reference_gate_id = None
    ctx.gate_epoch_metrics = []
    ctx.seeded = False
    author_gate_id = (
        None if rematch_for is not None
        else _select_author_gate_id(args.gate_author, gate_id, ctx.gate_root)
    )
    ctx.author_gate_id = author_gate_id
    if args.gate_author == "best" and gate_id > 0 and author_gate_id is None:
        print("[GATE AUTHOR] no eligible best gate yet; proposing under native routing")
    try:
        with ctx.session:
            if rematch_for is not None:
                _prepare_rematch_candidate(ctx, gate_id, rematch_for)
            elif seed:
                # Cold-start bootstrap: built-in compliant gates fill the store
                # with scored references before the proposer is trusted.
                _prepare_seed_candidate(ctx, gate_id)
            else:
                if author_gate_id is not None:
                    try:
                        _install_author_gate(ctx, author_gate_id)
                    except FileNotFoundError as exc:
                        print(
                            f"[GATE AUTHOR] trained gate {author_gate_id:03d} unavailable "
                            f"({exc}); proposing under native routing instead"
                        )
                        author_gate_id = None
                        ctx.author_gate_id = None
                    else:
                        _prepare_model_for_generation(ctx)
                        _propose_gate(ctx, gate_id, seen_hashes)
                        ctx.session.restore()
                if author_gate_id is None:
                    # Round 0 and native mode: the proposal is authored with the
                    # native router active, outside the session.
                    _propose_gate(ctx, gate_id, seen_hashes)
            _install_and_verify(ctx)
            epoch_paths = _run_epochs(ctx)
        record = _write_gate_record(ctx)
        score = record["score"]
        accuracy = score.get("accuracy")
        print(
            f"[GATE SEARCH] gate {gate_id:03d} finished: "
            f"mean CV accuracy "
            f"{f'{accuracy:.4f}' if isinstance(accuracy, float) else 'unavailable'} "
            f"({score.get('n_measured', 0)}/{score.get('n_attempted', 0)} measured)"
            + (f" [rematch of {rematch_for:03d}]" if rematch_for is not None else "")
        )
        return epoch_paths
    except Exception as exc:
        _record_failed_candidate(ctx, gate_id, exc)
        return []


def _prepare_rematch_candidate(ctx: RunContext, gate_id: int, source_gate_id: int) -> None:
    """Re-install a prior gate's code verbatim for a re-measure.

    Duplicate-structure rejection is intentionally bypassed: the rematch
    exists precisely to re-score the incumbent under the current proposer.
    """
    from .gate_store import gate_dir as store_gate_dir

    ctx.candidate_index = gate_id
    ctx.candidate_root = store_gate_dir(ctx.gate_root, gate_id)
    ctx.gate_dir = ctx.candidate_root / "gate_source"
    ctx.gate_dir.mkdir(parents=True, exist_ok=True)
    source_file = store_gate_dir(ctx.gate_root, source_gate_id) / "gate.py"
    source = source_file.read_text(encoding="utf-8")
    (ctx.candidate_root / "gate.py").write_text(source, encoding="utf-8")
    ctx.gate_source = source
    ctx.reference_gate_id = source_gate_id
    ctx.previous_feedback_summary = _initial_feedback_summary(source)
    ctx.used_prompts.clear()
    print(
        f"[GATE SEARCH] rematch: re-measuring gate {source_gate_id:03d} under "
        f"proposer {ctx.llm_version} as gate {gate_id:03d}"
    )


def _run_phase_b_batch(ctx: RunContext, batch_index: int) -> str | None:
    """Train the proposer on accumulated pairs; return the new version tag.

    Returns None when there is not enough comparable data yet (nothing is
    trained, the version does not change).  Pairs measured under the current
    proposer version are preferred; earlier-version pairs are only used when
    the current version alone cannot fill --gate-min-pairs, because scores
    across versions carry generator drift on top of gate quality.
    """
    from .gate_pairs import build_gate_pairs
    from .gate_store import load_gate_summaries
    from .phase_b import llm_version as version_of
    from .phase_b import train_proposer

    all_pairs = build_gate_pairs(ctx.gate_root)
    if not all_pairs:
        print(f"[PHASE B] skipping batch {batch_index}: no comparable gate pairs yet")
        return None
    version_by_id = {
        summary["gate_id"]: summary.get("llm_version")
        for summary in load_gate_summaries(ctx.gate_root)
    }
    current_version_pairs = [
        pair for pair in all_pairs
        if version_by_id.get(pair["lower_gate_id"]) == ctx.llm_version
        and version_by_id.get(pair["higher_gate_id"]) == ctx.llm_version
    ]
    if len(current_version_pairs) >= getattr(ctx.args, "gate_min_pairs", 2):
        pairs = current_version_pairs
    else:
        pairs = all_pairs
    if len(pairs) < getattr(ctx.args, "gate_min_pairs", 2):
        print(
            f"[PHASE B] skipping batch {batch_index}: {len(pairs)} comparable "
            f"pair(s) < --gate-min-pairs {getattr(ctx.args, 'gate_min_pairs', 2)}"
        )
        return None
    adapter_dir = ctx.run_root / "proposer_adapters" / f"batch_{batch_index:03d}"
    parent_version = ctx.llm_version
    train_proposer(
        ctx.session,
        ctx.tokenizer,
        pairs,
        ctx.args,
        ctx.shapes,
        adapter_dir,
    )
    new_version = version_of(ctx.session.model, ctx.args, merged_from=parent_version)
    if ctx.args.gate_store is not None:
        # Persist the newest adapter so the next run continues from it
        # (Tune.py-style weight compounding across runs).
        persisted = Path(ctx.args.gate_store).resolve() / "proposer"
        (persisted / "adapter").mkdir(parents=True, exist_ok=True)
        ctx.session.model.save_pretrained(persisted / "adapter")
        (persisted / "lineage.json").write_text(
            json.dumps({
                "version": new_version,
                "parent_version": parent_version,
                "batch": batch_index,
                "mode": ctx.args.gate_sft_mode,
                "pairs": len(pairs),
                "source_run": str(ctx.run_root),
            }, indent=2),
            encoding="utf-8",
        )
        print(
            f"[GATE SEARCH] outer-loop SFT: persisted proposer adapter → {persisted / 'adapter'}"
        )
    print(f"[GATE SEARCH] outer-loop SFT: proposer version {parent_version} -> {new_version}")
    return new_version


def _run_outer_search(ctx: RunContext, args: Namespace) -> list[list[Path]]:
    """Self-improving gate search: propose, replace, train inner loop, measure, repeat.

    Bad candidates are rejected and recorded, never fatal. With
    ``--gate-outer-sft`` the proposer itself is fine-tuned every
    ``--gate-sft-every`` candidates, followed by a king-of-the-hill re-measure
    of the incumbent under the new proposer version.
    """
    from .gate_store import best_gate, load_gate_summaries, next_gate_id, summarize_gate

    candidate_epochs: list[list[Path]] = []
    # A persistent store carries dedup history across runs; a fresh store only
    # knows the baseline contract, so the first proposal cannot echo it.
    seen_hashes = {
        summary["structural_hash"]
        for summary in load_gate_summaries(ctx.gate_root)
        if summary.get("structural_hash")
    }
    from .gate_prompt import BASELINE_GATE_CODE

    seen_hashes.add(structural_hash(BASELINE_GATE_CODE))
    # Seed architectures are also off-limits for verbatim LLM re-emission.
    from .gate_seeds import SEED_GATES

    for seed_source, _seed_name in SEED_GATES:
        seen_hashes.add(structural_hash(seed_source))
    sft_batch = 0
    candidates_run = 0

    while candidates_run < args.gate_candidates:
        # Fresh id each iteration: rematch candidates also consume store ids,
        # so a precomputed counter would collide with them.
        candidate_epochs.append(
            _run_candidate_once(
                ctx, next_gate_id(ctx.gate_root), seen_hashes,
                seed=candidates_run < getattr(args, "gate_seed_candidates", 0),
            )
        )
        candidates_run += 1
        # Fire after the batch completes, not before the next one starts:
        # with candidates == sft_every the "before" placement never triggers.
        if (
            getattr(args, "gate_outer_sft", False)
            and candidates_run % getattr(args, "gate_sft_every", 10) == 0
        ):
            sft_batch += 1
            new_version = _run_phase_b_batch(ctx, sft_batch)
            if new_version is not None:
                ctx.llm_version = new_version
                if not getattr(args, "gate_no_rematch", False):
                    incumbent = best_gate(ctx.gate_root)
                    if incumbent is not None:
                        candidate_epochs.append(
                            _run_candidate_once(
                                ctx, next_gate_id(ctx.gate_root),
                                seen_hashes, rematch_for=incumbent["gate_id"],
                            )
                        )

        record_probe = best_gate(ctx.gate_root)
        ctx.outer_summary = {
            "gate_root": str(ctx.gate_root),
            "gate_candidates": args.gate_candidates,
            "llm_version": ctx.llm_version,
            "best": (summarize_gate(record_probe) if record_probe else None),
            "phase_b_ready": _phase_b_status(ctx.gate_root),
        }
    return candidate_epochs


def _phase_b_status(gate_root: Path) -> str:
    """How close Phase A is to having usable Phase B training pairs."""
    from .gate_pairs import build_gate_pairs, describe_pairs

    return describe_pairs(build_gate_pairs(gate_root))


def main() -> None:
    from .generation_dtype import ensure_generation_dtype_policy
    from .eval_safety import ensure_moe_eval_safety

    ensure_generation_dtype_policy()
    # Keep malformed LLM-generated hp payloads from aborting the shared
    # evaluator without changing Tune.py or Eval.py themselves.
    ensure_moe_eval_safety()
    args = parse_args()
    _validate_args(args)
    ctx = _setup_run(args)
    if args.gate_outer_search:
        candidate_epochs = _run_outer_search(ctx, args)
    else:
        candidate_epochs = []
        outer_feedback = ""
        for candidate_index in range(args.gate_candidates):
            _prepare_gate_candidate(ctx, candidate_index, outer_feedback)
            # The context restores native routers after every candidate, so no
            # candidate inherits another candidate's trained gate parameters.
            with ctx.session:
                _install_and_verify(ctx)
                epoch_paths = _run_epochs(ctx)
            candidate_epochs.append(epoch_paths)
            outer_feedback += "\n" + _candidate_feedback(epoch_paths, candidate_index)
    _write_final_summary(ctx, candidate_epochs)
