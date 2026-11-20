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
import time
from argparse import Namespace
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cli import parse_args
from .feedback import _build_gate_feedback_summary
from .gate_source import _generate_gate, _validate_gate_source
from .gate_store import fingerprint, structural_hash
from .morphism import (
    _capture_native_gate_inputs,
    _dtype,
    _model_logits,
    _perturb_gate_weights,
    _seed_all,
    carry_gate_weights,
    verify_parent_function_carry,
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
    proposal_prompt: str | None = None
    proposal_attempts: list = field(default_factory=list)
    install_metrics: dict | None = None
    created_at: str | None = None
    used_prompts: list = field(default_factory=list)
    candidate_index: int = 0
    candidate_root: Path | None = None
    gate_root: Path | None = None
    gate_epoch_metrics: list = field(default_factory=list)
    reference_gate_id: int | None = None
    reference_accuracy: float | None = None
    goal_accuracy: float | None = None
    outer_summary: dict | None = None
    author_gate_id: int | None = None
    llm_version: str = "base"
    rematch_for: int | None = None
    gate_pool: list = field(default_factory=list)
    seeded: bool = False
    carry_source_id: int | None = None
    parent_logits: Any = None
    carry_forward_metrics: dict | None = None


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


def _ts() -> str:
    """Wall-clock tag for milestone prints.

    nohup logs carry no timestamps on plain print() output, which made
    post-run forensics ("which gate ran when, for how long") needless file
    archaeology; every [CYCLE] milestone line below is anchored in time.
    """
    return datetime.now().strftime("%H:%M:%S")


def _release_cuda_memory() -> None:
    """Return fragmented CUDA reserve to the driver between candidates.

    A candidate that OOMs leaves GiBs reserved-but-unallocated; the next
    candidate's install-verification forward then inherits the shortage and
    fails for no architectural reason (run 20260922_222052: gates 032/033
    were rolled back this way right after gate 031's training OOM).
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    carry_mode = getattr(args, "gate_carry_forward", None)
    if carry_mode is not None:
        if not args.gate_outer_search:
            raise ValueError(
                "--gate-carry-forward carries trained gate weights between outer "
                "candidates and requires --gate-outer-search"
            )
        if args.gate_author not in ("native", carry_mode):
            raise ValueError(
                f"--gate-carry-forward {carry_mode} conflicts with "
                f"--gate-author {args.gate_author}; pick one gate selection mode"
            )
        if args.gate_init_noise_scale != 0:
            raise ValueError(
                "--gate-carry-forward continues from trained weights; "
                "--gate-init-noise-scale must remain 0"
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
        gate_dir=gate_root / "gate_000",
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
                            seen_hashes: set[str] | None = None,
                            source: str | None = None) -> None:
    """Prepare one gate architecture while the native router is active.

    ``source`` comes from the proposal pool (harvested in an earlier round);
    without it a proposal round runs, the first harvested gate is used here
    and the remainder join the pool for later candidates.
    """
    from .gate_store import gate_dir

    args = ctx.args
    ctx.candidate_index = candidate_index
    ctx.candidate_root = gate_dir(ctx.gate_root, candidate_index)
    ctx.gate_dir = ctx.candidate_root
    ctx.candidate_root.mkdir(parents=True, exist_ok=True)
    ctx.proposal_prompt = None
    ctx.proposal_attempts = []
    if args.gate_source is not None:
        if not args.gate_source.is_file():
            raise FileNotFoundError(f"Gate source file does not exist: {args.gate_source}")
        source = args.gate_source.read_text(encoding="utf-8")
        _validate_gate_source(
            source, ctx.shapes, class_name=args.gate_class,
            require_base=not args.gate_random_init,
        )
        ctx.proposal_prompt = f"[external gate source] {args.gate_source}"
    elif source is None:
        sources = _generate_gate(
            proposal_chat or ctx.chat_bot, ctx.shapes, args.gate_generation_attempts,
            args.gate_max_new_tokens, ctx.candidate_root,
            reference_source=reference_source,
            reference_accuracy=reference_accuracy,
            goal_accuracy=goal_accuracy,
            dataset=dataset,
            seen_hashes=seen_hashes,
            inherit_reference=getattr(args, "gate_carry_forward", None) is not None,
            write_artifacts=False,
            prompt_out=ctx.proposal_attempts,
        )
        source = sources[0]
        ctx.proposal_prompt = (
            ctx.proposal_attempts[-1] if ctx.proposal_attempts else None
        )
        if len(sources) > 1:
            ctx.gate_pool.extend(sources[1:])
            print(
                f"[GATE SEARCH] {len(sources) - 1} extra valid gate(s) -> "
                f"proposal pool ({len(ctx.gate_pool)} waiting)"
            )
    else:
        ctx.proposal_prompt = "[pooled valid gate from an earlier proposal round]"
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


def _latest_trained_checkpoint(gate_root: Path, gate_id: int) -> Path | None:
    """Newest trained checkpoint: ``weights/`` first, legacy epochs after."""
    from .gate_store import WEIGHTS_DIR
    from .gate_store import gate_dir as store_gate_dir

    entry = store_gate_dir(gate_root, gate_id)
    weights = entry / WEIGHTS_DIR
    if (weights / "metadata.json").is_file() and (weights / "gate_weights.pt").is_file():
        return weights
    epochs_root = entry / "epochs"
    if not epochs_root.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for directory in epochs_root.iterdir():
        name = directory.name
        if not (name.startswith("A") and name[1:].isdigit()):
            continue
        checkpoint = directory / "gate_post_train"
        if not (checkpoint / "metadata.json").is_file() or not (
            checkpoint / "gate_weights.pt"
        ).is_file():
            continue
        epoch = int(name[1:])
        if best is None or epoch > best[0]:
            best = (epoch, checkpoint)
    return best[1] if best else None


def _install_author_gate(ctx: RunContext, author_gate_id: int, *,
                         mode: str | None = None,
                         purpose: str | None = None) -> None:
    """Install a previously trained gate so the LLM works under its routing.

    Used while the LLM authors the next proposal and while proposer SFT
    batches run. No equivalence checks: trained weights are not bit-exact
    with the native router by design. Must run inside ``with ctx.session:``
    with no gates currently installed.
    """
    args = ctx.args
    from .gate_store import gate_dir as store_gate_dir

    author_root = store_gate_dir(ctx.gate_root, author_gate_id)
    checkpoint = _latest_trained_checkpoint(ctx.gate_root, author_gate_id)
    if checkpoint is None:
        raise FileNotFoundError(
            f"Author gate checkpoint not found under {author_root / 'epochs'}"
        )
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
    if purpose is None:
        print(
            f"[GATE AUTHOR] gate {mode or args.gate_author} mode: next proposal is authored "
            f"under trained gate {author_gate_id:03d} ({checkpoint})"
        )
    else:
        print(
            f"[GATE AUTHOR] gate {mode or args.gate_author} mode: trained gate "
            f"{author_gate_id:03d} installed for {purpose} ({checkpoint})"
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

    # A previous candidate's gate training leaves the model in train mode
    # with use_cache disabled and stale flags on detached native gates; every
    # install must start from the clean generation state.
    _prepare_model_for_generation(ctx)
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
    carried = False
    carry_source_id = getattr(ctx, "carry_source_id", None)
    if getattr(args, "gate_carry_forward", None) is not None and carry_source_id is not None:
        checkpoint = _latest_trained_checkpoint(ctx.gate_root, carry_source_id)
        if checkpoint is None:
            print(
                f"[GATE CARRY] gate {carry_source_id:03d} has no trained checkpoint "
                "yet; installing from the native copy"
            )
        else:
            ctx.carry_forward_metrics = carry_gate_weights(session.installs, checkpoint)
            ctx.carry_forward_metrics.update({
                "mode": args.gate_carry_forward,
                "source_gate_id": carry_source_id,
                "target_gate_id": ctx.candidate_index,
            })
            carried = True
            print(
                f"[GATE CARRY] trained weights of gate {carry_source_id:03d} -> "
                f"gate {ctx.candidate_index:03d} initialization "
                f"({ctx.carry_forward_metrics['n_parameters_copied']} parameters copied, "
                f"{ctx.carry_forward_metrics['n_parameters_fresh']} fresh)"
            )
    initialization_metrics = [] if args.gate_random_init else _perturb_gate_weights(
        session.installs, args.gate_init_noise_scale, args.seed
    )
    replacement_logits = _model_logits(session.model, ctx.sample_input)
    if not torch.isfinite(replacement_logits).all():
        raise RuntimeError("Replacement gates produced non-finite model logits")
    max_abs_diff = float((ctx.native_logits - replacement_logits).abs().max().item())
    equivalent = bool(torch.allclose(ctx.native_logits, replacement_logits, rtol=1e-5, atol=1e-5))
    parent_check = None
    parent_logits = getattr(ctx, "parent_logits", None)
    if carried and parent_logits is not None:
        parent_check = verify_parent_function_carry(parent_logits, replacement_logits)
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
        "carried_trained_weights": carried,
        "carry_source_gate_id": carry_source_id,
        "parent_function_equivalent": parent_check["equivalent"] if parent_check else None,
        "parent_logit_bit_exact": parent_check["bit_exact"] if parent_check else None,
        "parent_logit_max_abs_difference": (
            parent_check["max_abs_difference"] if parent_check else None
        ),
    }
    ctx.install_metrics = {
        "step_zero": equivalence,
        "initialization": initialization_metrics,
        "carry_forward": getattr(ctx, "carry_forward_metrics", None),
    }
    # Compact install receipt: the full metrics live in record.json; this
    # line makes the install auditable straight from the log (native-copy
    # vs carried, divergence size).
    print(
        f"[{_ts()}] [CYCLE] gate_{getattr(ctx, 'candidate_index', 0):03d} INSTALLED: "
        f"init={'gate_%03d-weights' % carry_source_id if carried else 'native-copy'}, "
        f"step-zero max_abs_diff={max_abs_diff:.3g}, equivalent={equivalent}, "
        f"parent-check={'n/a' if parent_check is None else parent_check['equivalent']}"
    )
    if parent_check is not None and not parent_check["equivalent"]:
        # Reject-and-continue: the candidate is recorded as failed by the
        # caller, the incumbent stays the self.
        raise RuntimeError(
            "Function-preserving carry failed: the successor gate does not "
            "reproduce the parent gate's step-zero logits "
            f"(max_abs_difference={parent_check['max_abs_difference']:.6g}); "
            "keep the inherited modules unchanged and put new capacity in "
            "zero-initialized branches"
        )
    skip_step_zero_verify = getattr(args, "gate_skip_step_zero_verify", False)
    if not equivalent and skip_step_zero_verify:
        print(
            "[GATE VERIFY] step-zero mismatch accepted (--gate-skip-step-zero-verify): "
            f"max_abs_difference={max_abs_diff}"
        )
    if carried and not equivalent:
        print(
            "[GATE CARRY] step-zero logits diverge from native as expected after "
            f"weight carry: max_abs_difference={max_abs_diff}"
        )
    if (
        not args.gate_random_init
        and args.gate_init_noise_scale == 0
        and not equivalent
        and not skip_step_zero_verify
        and not carried
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
                "feedback": _feedback_record(gate_feedback),
                "cv": cycle_results,
                "config": _epoch_config(args),
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
                "weights but not attention/MLP activations. Harvesting the "
                "epochs measured so far and stopping this candidate early "
                "instead of discarding its score; reduce --max-length or "
                "--batch-size to avoid a recurrence."
            )
            _release_cuda_memory()
            oom_recovered = True
            result = None
            optimizer = None
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
            "feedback": _feedback_record(gate_feedback),
            "cv": cycle_results,
            "config": _epoch_config(args),
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


def _harvest_epoch_metrics(
    epoch: int,
    epoch_path: Path,
    gate_feedback: dict,
    cycle_results: dict,
    models_trained: int,
    train_loss: float | None,
    validation_loss: float | None,
    active_layers: set[int],
    duration_ns: int = 0,
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
        "duration": duration_ns,
        "synth_nn": f"epochs/A{epoch}/synth_nn",
    }


def _feedback_record(feedback: dict) -> dict:
    """Compact per-epoch feedback kept in the record for improvement analysis."""
    keys = (
        "summary", "n_generated", "n_trained", "n_measured",
        "mean_accuracy", "std_accuracy", "best_accuracy", "mean_accuracy_delta",
        "improved_candidates", "datasets", "tasks",
        "candidate_outcomes", "failure_reasons",
    )
    return {key: feedback.get(key) for key in keys}


def _epoch_config(args: Namespace) -> dict:
    return {
        "progressive_unfreeze_descending": args.progressive_unfreeze_descending,
        "gate_implementation": args.gate_implementation,
        "initialize_from_original": not args.gate_random_init,
        "fixed_hyperparameter_overrides": args.fixed_eval_hyperparameters,
        "nn_train_epochs": args.nn_train_epochs,
    }


def _save_latest_weights(ctx: RunContext) -> Path | None:
    """Atomically replace ``<gate>/weights`` with the current trained gate."""
    from .gate_store import WEIGHTS_DIR

    if ctx.candidate_root is None:
        return None
    target = ctx.candidate_root / WEIGHTS_DIR
    temporary = ctx.candidate_root / f".{WEIGHTS_DIR}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        ctx.session.save(temporary)
        if not temporary.exists():
            return None
        if target.exists():
            shutil.rmtree(target)
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _protocol(args: Namespace) -> dict:
    """Measurement conditions stored with every gate and in the archive."""
    protocol = {
        "model": getattr(args, "model", None),
        "seed": getattr(args, "seed", None),
        "dataset": getattr(args, "dataset", None),
        "task": "img-classification",
        "metric": "accuracy",
        "layers": getattr(args, "layers", None),
        "router_top_k": getattr(args, "router_top_k", None),
        "epochs": getattr(args, "epochs", None),
        "nn_train_epochs": getattr(args, "nn_train_epochs", None),
        "test_nn": getattr(args, "test_nn", None),
        "fixed_eval_hyperparameters": getattr(args, "fixed_eval_hyperparameters", None),
        "gate_train_steps": getattr(args, "gate_train_steps", None),
        "gate_learning_rate": getattr(args, "gate_learning_rate", None),
        "gate_init_noise_scale": getattr(args, "gate_init_noise_scale", None),
        "train_prompt_config": getattr(args, "train_prompt_config", None),
        "test_prompt_config": getattr(args, "test_prompt_config", None),
    }
    protocol["fingerprint"] = fingerprint(protocol)
    return protocol


def _gate_archive_root(args: Namespace) -> Path:
    configured = getattr(args, "gate_archive_dir", None)
    if configured is not None:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[1] / "gate_archive"


def _slug(value: Any, fallback: str = "unknown") -> str:
    text = str(value or fallback)
    slug = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in text
    ).strip("-")
    return slug or fallback


def _archive_measured_gate(ctx: RunContext, record: dict) -> Path | None:
    """Copy a measured gate's code + LEMUR-shaped record into the archive.

    The archive is a data directory under ``nn-gpt`` (``gate_archive`` by
    default). It is copy-only: the live store remains the source of truth, a
    failed archive write never fails the candidate, and only gates with a
    measured CV accuracy are archived.
    """
    accuracy = (record.get("score") or {}).get("accuracy")
    if not isinstance(accuracy, (int, float)):
        return None
    candidate_root = getattr(ctx, "candidate_root", None)
    if candidate_root is None:
        return None
    source = candidate_root / "gate.py"
    record_path = candidate_root / "record.json"
    if not source.is_file() or not record_path.is_file():
        return None
    archive_root = _gate_archive_root(ctx.args)
    run_name = ctx.run_root.name if ctx.run_root is not None else "run"
    target = (
        archive_root
        / _slug(getattr(ctx.args, "model", None), "model")
        / _slug(run_name, "run")
        / f"gate_{ctx.candidate_index:03d}"
    )
    try:
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target / "gate.py")
        shutil.copy2(record_path, target / "record.json")
    except OSError as exc:
        print(f"[GATE ARCHIVE] gate {ctx.candidate_index:03d} archive skipped: {exc}")
        return None
    print(f"[GATE ARCHIVE] gate {ctx.candidate_index:03d} -> {target}")
    return target


def _write_gate_record(ctx: RunContext, *, finalize: bool = True) -> dict:
    """Persist one gate candidate, incrementally and at completion.

    ``finalize=False`` writes a ``running`` record after each inner epoch so a
    killed run still leaves usable measurements. The final write sets the
    terminal status and archives measured gates.
    """
    from .gate_store import summarize_epoch_metrics, write_gate_record

    score = summarize_epoch_metrics(ctx.gate_epoch_metrics)
    gate_source = getattr(ctx, "gate_source", None)
    record = {
        "gate_id": ctx.candidate_index,
        "gate_code": gate_source,
        "structural_hash": structural_hash(gate_source) if gate_source else None,
        "reference_gate_id": getattr(ctx, "reference_gate_id", None),
        "author_mode": (
            getattr(ctx.args, "gate_carry_forward", None)
            or getattr(ctx.args, "gate_author", None)
        ),
        "author_gate_id": getattr(ctx, "author_gate_id", None),
        "carry_source_gate_id": getattr(ctx, "carry_source_id", None),
        "carry_forward": getattr(ctx, "carry_forward_metrics", None),
        "rematch_for": getattr(ctx, "rematch_for", None),
        "llm_version": getattr(ctx, "llm_version", "base"),
        "seeded": getattr(ctx, "seeded", False),
        "created_at": getattr(ctx, "created_at", None),
        "running": not finalize,
        "install": getattr(ctx, "install_metrics", None),
        "proposal": {
            "prompt": getattr(ctx, "proposal_prompt", None),
            "reference_gate_id": getattr(ctx, "reference_gate_id", None),
            "reference_accuracy": getattr(ctx, "reference_accuracy", None),
            "goal_accuracy": getattr(ctx, "goal_accuracy", None),
            "attempts": len(getattr(ctx, "proposal_attempts", [])),
        },
        "protocol": _protocol(ctx.args),
        # gate_pairs.build_gate_pairs groups comparable records on exactly
        # these three keys; without them no Phase B pairs can ever form.
        "task": "img-classification",
        "dataset": getattr(ctx.args, "dataset", "cifar-10"),
        "metric": "accuracy",
        "score": score,
        "epochs": ctx.gate_epoch_metrics,
    }
    write_gate_record(
        ctx.gate_root,
        ctx.candidate_index,
        record,
        run_root=getattr(ctx, "run_root", None),
        model=getattr(ctx.args, "model", None),
    )
    if finalize:
        _archive_measured_gate(ctx, record)
    return record


def _candidate_feedback(epoch_metrics: list[dict], candidate_index: int) -> str:
    """Summarize one completed candidate for the next outer-loop proposal."""
    entries = [
        {
            key: (metrics.get("feedback") or {}).get(key)
            for key in (
                "n_generated", "n_trained", "best_accuracy",
                "mean_accuracy_delta", "improved_candidates",
            )
        }
        for metrics in epoch_metrics
    ]
    return f"Independent gate candidate {candidate_index} results: {json.dumps(entries)}"


def _write_final_summary(ctx: RunContext) -> None:
    """Write the run-level summary from the canonical gate records."""
    from ab.gpt.util.Const import nngpt_dir
    from .gate_store import load_gate_summaries

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
        "gate_source": (
            str((ctx.candidate_root or ctx.gate_dir) / "gate.py")
            if ctx.gate_source is not None else None
        ),
        "gate_shapes": ctx.shapes,
        "replaced_gates": len(
            ctx.sites
            if args.layers is None
            else [s for s in ctx.sites if s.layer_index in args.layers]
        ),
        "epochs": args.epochs,
        "gate_candidates": args.gate_candidates,
        "gate_author": args.gate_author,
        "gate_carry_forward": getattr(args, "gate_carry_forward", None),
        "gate_outer_search": ctx.outer_summary,
        "run_root": str(ctx.run_root),
    }
    for record in load_gate_summaries(ctx.gate_root):
        provenance = record.get("provenance") or {}
        if provenance.get("run_root") not in (None, str(ctx.run_root)):
            continue
        gate_id = record.get("gate_id")
        if not isinstance(gate_id, int):
            continue
        summary[f"candidate_{gate_id:03d}"] = {
            "gate_source": (record.get("artifacts") or {}).get("source"),
            "status": record.get("status"),
            "score": record.get("score"),
            "protocol": record.get("protocol"),
            "epochs": record.get("epochs"),
        }

    (nngpt_dir / "one_cycle_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(f"MoE gate cycle completed: {ctx.run_root}")


def _propose_gate(ctx: RunContext, gate_id: int, seen_hashes: set[str],
                  source: str | None = None) -> None:
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
    ctx.reference_accuracy = reference_accuracy
    ctx.goal_accuracy = goal_accuracy

    _prepare_gate_candidate(
        ctx,
        gate_id,
        reference_source=reference_source,
        reference_accuracy=reference_accuracy,
        goal_accuracy=goal_accuracy,
        dataset=ctx.args.dataset,
        seen_hashes=seen_hashes,
        source=source,
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
    ctx.gate_dir = ctx.candidate_root
    ctx.candidate_root.mkdir(parents=True, exist_ok=True)
    source = seed_gate_source(gate_id)
    _validate_gate_source(source, ctx.shapes)
    (ctx.candidate_root / "gate.py").write_text(source.rstrip() + "\n", encoding="utf-8")
    ctx.proposal_prompt = (
        f"[cold-start bootstrap] built-in seed gate '{seed_gate_name(gate_id)}' "
        "installed without an LLM proposal"
    )
    ctx.proposal_attempts = []
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
    from .gate_store import write_gate_record

    reason = f"{type(exc).__name__}: {exc}"
    print(f"[GATE SEARCH] gate {gate_id:03d} FAILED, continuing: {reason}")
    record = {
        "gate_id": gate_id,
        "gate_code": ctx.gate_source,
        "structural_hash": structural_hash(ctx.gate_source) if ctx.gate_source else None,
        "reference_gate_id": ctx.reference_gate_id,
        "author_mode": (
            getattr(ctx.args, "gate_carry_forward", None)
            or getattr(ctx.args, "gate_author", None)
        ),
        "author_gate_id": ctx.author_gate_id,
        "carry_source_gate_id": getattr(ctx, "carry_source_id", None),
        "carry_forward": getattr(ctx, "carry_forward_metrics", None),
        "rematch_for": getattr(ctx, "rematch_for", None),
        "llm_version": getattr(ctx, "llm_version", "base"),
        "seeded": getattr(ctx, "seeded", False),
        "created_at": getattr(ctx, "created_at", None),
        "running": False,
        "install": getattr(ctx, "install_metrics", None),
        "proposal": {
            "prompt": getattr(ctx, "proposal_prompt", None),
            "reference_gate_id": getattr(ctx, "reference_gate_id", None),
            "reference_accuracy": getattr(ctx, "reference_accuracy", None),
            "goal_accuracy": getattr(ctx, "goal_accuracy", None),
            "attempts": len(getattr(ctx, "proposal_attempts", [])),
        },
        "protocol": _protocol(ctx.args),
        "score": {
            "objective": "mean_of_epoch_means_v1",
            "accuracy": None,
            "eligible": False,
            "n_measured": 0,
            "n_attempted": 0,
        },
        # Partial measurements survive the failure: the candidate stays
        # ineligible (it never completed the protocol) but its epochs remain
        # analysable instead of being discarded with the record.
        "epochs": ctx.gate_epoch_metrics,
        "failure": reason[:2000],
    }
    write_gate_record(
        ctx.gate_root,
        gate_id,
        record,
        run_root=getattr(ctx, "run_root", None),
        model=getattr(ctx.args, "model", None),
    )


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
    ctx.reference_accuracy = None
    ctx.goal_accuracy = None
    ctx.gate_epoch_metrics = []
    ctx.seeded = False
    ctx.carry_source_id = None
    ctx.parent_logits = None
    ctx.carry_forward_metrics = None
    ctx.install_metrics = None
    ctx.proposal_prompt = None
    ctx.proposal_attempts = []
    ctx.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    carry_mode = getattr(args, "gate_carry_forward", None)
    author_mode = carry_mode or args.gate_author
    author_gate_id = (
        None if rematch_for is not None
        else _select_author_gate_id(author_mode, gate_id, ctx.gate_root)
    )
    ctx.author_gate_id = author_gate_id
    if author_mode == "best" and gate_id > 0 and author_gate_id is None:
        print("[GATE AUTHOR] no eligible best gate yet; proposing under native routing")
    pooled_source = None
    if not seed and rematch_for is None and ctx.gate_pool:
        pooled_source = ctx.gate_pool.pop(0)
        print(
            f"[GATE SEARCH] gate {gate_id:03d} taken from proposal pool "
            f"({len(ctx.gate_pool)} still waiting)"
        )
    try:
        with ctx.session:
            if rematch_for is not None:
                _prepare_rematch_candidate(ctx, gate_id, rematch_for)
                if carry_mode is not None:
                    # Function-carry verification needs the parent's logits:
                    # reinstall the re-measured gate with its trained weights,
                    # capture, and restore before the successor is installed.
                    try:
                        _install_author_gate(ctx, rematch_for, mode=carry_mode)
                    except FileNotFoundError:
                        pass
                    else:
                        _prepare_model_for_generation(ctx)
                        ctx.parent_logits = _model_logits(
                            ctx.session.model, ctx.sample_input
                        )
                        ctx.session.restore()
            elif seed:
                # Cold-start bootstrap: built-in compliant gates fill the store
                # with scored references before the proposer is trusted.
                _prepare_seed_candidate(ctx, gate_id)
            else:
                if author_gate_id is not None:
                    try:
                        _install_author_gate(ctx, author_gate_id, mode=author_mode)
                    except FileNotFoundError as exc:
                        print(
                            f"[GATE AUTHOR] trained gate {author_gate_id:03d} unavailable "
                            f"({exc}); proposing under native routing instead"
                        )
                        author_gate_id = None
                        ctx.author_gate_id = None
                    else:
                        _prepare_model_for_generation(ctx)
                        ctx.parent_logits = _model_logits(
                            ctx.session.model, ctx.sample_input
                        )
                        _propose_gate(ctx, gate_id, seen_hashes, source=pooled_source)
                        ctx.session.restore()
                if author_gate_id is None:
                    # Round 0 and native mode: the proposal is authored with the
                    # native router active, outside the session.
                    _propose_gate(ctx, gate_id, seen_hashes, source=pooled_source)
            if carry_mode is not None:
                # Weight-carry source: the incumbent for fresh proposals, or the
                # re-measured gate itself for a rematch. Seeds bootstrap from the
                # native copy like every first-generation gate.
                if rematch_for is not None:
                    ctx.carry_source_id = rematch_for
                elif not seed:
                    ctx.carry_source_id = author_gate_id
            # One consolidated header per candidate: kind, routing it was
            # authored under, weight-carry source, and the proposer version —
            # everything a post-run log analysis needs on a single line.
            candidate_kind = (
                f"rematch of gate_{rematch_for:03d}" if rematch_for is not None
                else "seed" if seed
                else "pool" if pooled_source is not None
                else "llm-proposal"
            )
            print(
                f"[{_ts()}] [CYCLE] gate_{gate_id:03d} START: {candidate_kind}, "
                f"author={'native' if author_gate_id is None else f'gate_{author_gate_id:03d}'}, "
                f"carry={'none' if ctx.carry_source_id is None else f'gate_{ctx.carry_source_id:03d}'}, "
                f"proposer={ctx.llm_version}"
            )
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
    finally:
        # Whether the candidate succeeded or was rejected, hand fragmented
        # CUDA reserve back to the driver so the next install-verification
        # forward is not doomed by this candidate's peak usage.
        _release_cuda_memory()


def _prepare_rematch_candidate(ctx: RunContext, gate_id: int, source_gate_id: int) -> None:
    """Re-install a prior gate's code verbatim for a re-measure.

    Duplicate-structure rejection is intentionally bypassed: the rematch
    exists precisely to re-score the incumbent under the current proposer.
    """
    from .gate_store import gate_dir as store_gate_dir

    ctx.candidate_index = gate_id
    ctx.candidate_root = store_gate_dir(ctx.gate_root, gate_id)
    ctx.gate_dir = ctx.candidate_root
    ctx.candidate_root.mkdir(parents=True, exist_ok=True)
    source_file = store_gate_dir(ctx.gate_root, source_gate_id) / "gate.py"
    source = source_file.read_text(encoding="utf-8")
    (ctx.candidate_root / "gate.py").write_text(source, encoding="utf-8")
    ctx.proposal_prompt = (
        f"[rematch] re-measuring gate {source_gate_id:03d} under proposer "
        f"{ctx.llm_version}"
    )
    ctx.proposal_attempts = []
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
        version_scope = "current-version"
    else:
        pairs = all_pairs
        version_scope = "cross-version fallback"
    if len(pairs) < getattr(ctx.args, "gate_min_pairs", 2):
        print(
            f"[PHASE B] skipping batch {batch_index}: {len(pairs)} comparable "
            f"pair(s) < --gate-min-pairs {getattr(ctx.args, 'gate_min_pairs', 2)}"
        )
        return None
    # SFT batch receipt: WHICH pairs trained the proposer and under which
    # version scope — the first question of any post-run analysis of the
    # self-improvement loop ("what did the LoRA actually learn from?").
    pair_ids = ", ".join(pair["pair_id"] for pair in pairs[:8]) + (
        f" (+{len(pairs) - 8} more)" if len(pairs) > 8 else ""
    )
    print(
        f"[{_ts()}] [CYCLE] proposer SFT batch {batch_index}: {len(pairs)} pair(s) "
        f"[{pair_ids}] ({version_scope}), parent proposer {ctx.llm_version}"
    )
    adapter_dir = ctx.run_root / "proposer_adapters" / f"batch_{batch_index:03d}"
    parent_version = ctx.llm_version
    training_summary = train_proposer(
        ctx.session,
        ctx.tokenizer,
        pairs,
        ctx.args,
        ctx.shapes,
        adapter_dir,
    )
    new_version = version_of(ctx.session.model, ctx.args, merged_from=parent_version)
    print(
        f"[{_ts()}] [CYCLE] proposer SFT batch {batch_index} TRAINED: "
        f"steps={training_summary.get('steps')}, pairs={training_summary.get('pairs')}, "
        f"mode={training_summary.get('mode')}, adapter={training_summary.get('output_dir')}"
    )
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


def _run_sft_batch_under_best_gate(ctx: RunContext, batch_index: int) -> str | None:
    """Run one proposer SFT batch with the best inner-loop gate installed.

    The SFT loss is teacher-forced on gate-code text, but its forward pass is
    still routed by the currently installed gates. The batch therefore runs
    with the best gate candidate's architecture and trained weights active —
    the same carried routing state the proposer will author proposals from —
    instead of the native routers the session restores after every candidate.
    Gate parameters themselves stay frozen; only the proposer LoRA trains
    (full-model adaptation, unlike the gate-only training of the inner loop).

    Without a scored incumbent, without a trained checkpoint, or when
    installing/training under the carried gate fails, the batch falls back to
    native routing so a broken incumbent can never block proposer training.
    """
    from .gate_store import best_gate

    incumbent = best_gate(ctx.gate_root)
    if incumbent is not None and _latest_trained_checkpoint(
        ctx.gate_root, incumbent["gate_id"]
    ) is not None:
        try:
            with ctx.session:
                _install_author_gate(
                    ctx, incumbent["gate_id"], mode="sft-carry",
                    purpose=f"proposer SFT batch {batch_index}",
                )
                for install in ctx.session.installs:
                    for parameter in install.new_gate.parameters():
                        parameter.requires_grad_(False)
                return _run_phase_b_batch(ctx, batch_index)
        except Exception as exc:
            print(
                "[GATE SEARCH] proposer SFT batch under trained gate "
                f"{incumbent['gate_id']:03d} failed ({type(exc).__name__}: {exc}); "
                "falling back to native routing"
            )
    return _run_phase_b_batch(ctx, batch_index)


def _run_outer_search(ctx: RunContext, args: Namespace) -> list[list[Path]]:
    """Self-improving gate search: propose, replace, train inner loop, measure, repeat.

    Bad candidates are rejected and recorded, never fatal. With
    ``--gate-outer-sft`` the proposer itself is fine-tuned every
    ``--gate-sft-every`` candidates — under the best trained gate's routing —
    followed by a king-of-the-hill re-measure of the incumbent under the new
    proposer version.
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
    from .gate_seeds import SEED_GATES, seed_gate_source

    for index in range(len(SEED_GATES)):
        seen_hashes.add(structural_hash(seed_gate_source(index)))
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
            try:
                new_version = _run_sft_batch_under_best_gate(ctx, sft_batch)
            except Exception as exc:
                # The proposer SFT has never run end-to-end on this hardware;
                # a broken batch must not take the whole search down with it.
                _release_cuda_memory()
                print(
                    f"[GATE SEARCH] proposer SFT batch {sft_batch} failed "
                    f"({type(exc).__name__}: {exc}); continuing with proposer "
                    f"{ctx.llm_version}"
                )
                new_version = None
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
            if outer_feedback:
                # Pre-dates the keyword-only signature; accumulated results only
                # reach later candidates through the log and the best-gate
                # reference prompt.
                print(outer_feedback)
            ctx.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            ctx.proposal_attempts = []
            ctx.install_metrics = None
            _prepare_gate_candidate(ctx, candidate_index)
            # The context restores native routers after every candidate, so no
            # candidate inherits another candidate's trained gate parameters.
            with ctx.session:
                _install_and_verify(ctx)
                epoch_paths = _run_epochs(ctx)
            _write_gate_record(ctx)
            candidate_epochs.append(epoch_paths)
            outer_feedback += "\n" + _candidate_feedback(
                ctx.gate_epoch_metrics, candidate_index,
            )
    _write_final_summary(ctx)
