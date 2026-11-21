"""Argument validation and run initialization for the MoE gate cycle."""

from __future__ import annotations

import json
import os
import shutil
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

from .context import RunContext, _log_cuda_memory
from .morphism import _capture_native_gate_inputs, _dtype, _model_logits, _seed_all


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

    from ab.gpt.util.Const import conf_test_dir, epoch_dir, nngpt_dir
    from ab.gpt.util.llm.Chatbot import ChatBot
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
        else Path(__file__).resolve().parents[1] / "gate_library"
    )
    gate_root.mkdir(parents=True, exist_ok=True)
    if any(gate_root.glob("gate_*")) or any(gate_root.glob("gate-*")):
        print(
            f"[GATE SEARCH] persistent store: continuing from existing records in {gate_root}"
        )
    print(f"[GATE SEARCH] gate root: {gate_root}")

    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("--load-in-8bit and --load-in-4bit are mutually exclusive")
    if args.load_in_4bit or args.load_in_8bit:
        from importlib.util import find_spec

        if find_spec("bitsandbytes") is None:
            raise RuntimeError(
                "--load-in-8bit/--load-in-4bit require the bitsandbytes package "
                "(pip install bitsandbytes)"
            )
        from transformers import BitsAndBytesConfig

        # Routers must stay unquantized: exact weight-copy seeding of the
        # replacement gates (and step-zero bit-exact equivalence) depends on
        # reading the native router's true float weights.
        # NOTE: once any explicit skip list is supplied, transformers drops its
        # automatic output-head protection, so lm_head must be listed here too.
        keep_unquantized = ["gate", "lm_head"]

    # A quantized load of the whole backbone also implies quantized experts: the
    # fused stacks are the bulk of the weights and stay in bf16 otherwise.
    expert_quant = getattr(args, "gate_expert_quant", "none")
    if args.load_in_4bit and expert_quant == "none":
        expert_quant = "nf4"

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
    if expert_quant != "none":
        # The fused MoE expert stacks are 3-D nn.Parameters that bitsandbytes'
        # nn.Linear walker cannot see, so a direct GPU load keeps ~92% of the
        # weights in bf16 and peaks at the full model size on the device
        # (transformers#43032). Stage the load on the CPU instead, quantize the
        # expert stacks afterwards (see expert_quantization), then move the
        # model to the GPU.
        model_kwargs["device_map"] = {"": "cpu"}
    elif args.device_map.lower() != "none":
        device_map = args.device_map
        if (
            device_map == "auto"
            and (args.load_in_8bit or args.load_in_4bit)
            and torch.cuda.device_count() == 1
        ):
            # Accelerate's automatic mapper reserves enough headroom to push the
            # final DeepSeek layers to CPU on a 24-GiB card.  The complete
            # quantized model fits on this GPU, and gate replacement/training
            # requires real (non-meta) router tensors on every layer.
            device_map = "cuda:0"
        model_kwargs["device_map"] = device_map

    if args.load_in_4bit or args.load_in_8bit:
        # Transformers pre-allocates a buffer the size of the *unquantized*
        # model before loading quantized weights. On a shared 24 GiB card that
        # warmup can exceed free VRAM even though the NF4 model itself fits.
        # The warmup is only an allocator optimization; skip it for QLoRA.
        try:
            import transformers.modeling_utils as _modeling_utils
            if hasattr(_modeling_utils, "caching_allocator_warmup"):
                _modeling_utils.caching_allocator_warmup = lambda *a, **k: None
                print("[CYCLE] skipped Transformers allocator warmup for quantized load")
        except Exception as exc:
            print(f"[CYCLE] allocator warmup patch unavailable: {exc}")

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

    if expert_quant != "none":
        from .expert_quantization import quantize_fused_experts

        target_device = (
            "cuda:0"
            if args.device_map.lower() != "none" and torch.cuda.is_available()
            else "cpu"
        )
        quantized = quantize_fused_experts(
            session.model, device=target_device, quant_type=expert_quant
        )
        if quantized:
            print(
                f"[CYCLE] {expert_quant}-quantized {quantized} fused MoE expert "
                f"stacks on {target_device}"
            )
        if target_device != "cpu":
            session.model.to(target_device)
        _log_cuda_memory("model on device after expert quantization")

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
        if not args.gate_fresh_proposer:
            persisted = gate_root / "proposer"
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
    # The chat model has no stop-sequence support, so a generation runs to the
    # token cap even when the requested block closed long before. Stopping at
    # the closing tag is what makes a large CV sweep affordable.
    from .generation_stop import install_tag_stopping_criterion

    install_tag_stopping_criterion(session.model, tokenizer)
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
