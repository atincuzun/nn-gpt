"""End-to-end check: ask DeepSeek-V2-Lite for a gate, install it, verify it works.

Steps:
  1. Load DeepSeek-V2-Lite-Chat (4-bit) through MoEGateSession.
  2. Discover the native MoEGate sites and capture native model logits.
  3. Ask the LLM (via moe_cycle.gate_source._generate_gate) for a replacement
     gate using the DeepSeek-V2-Lite contract prompt.
  4. Install the generated gate over every router, seeding it from the native
     weight, and verify step-zero model-logit equivalence.
  5. Check the replaced model still runs and generates coherent tokens.
  6. Restore the native routers.

Run from the nn-gpt directory:
    .venv/bin/python verify_llm_gate_replace.py --max-new-tokens 768
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch


class _LastTokenLogits(torch.nn.Module):
    """Compute vocabulary logits for the final position only.

    The stock DeepSeek forward materializes [batch, prompt_len, 102400] float32
    logits during generation, which needs >256 MiB for the gate prompt. The
    smoke test only needs the next-token distribution, so slicing the hidden
    states before lm_head keeps the same routing/step-zero semantics at a tiny
    fraction of the memory.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.inner(hidden_states[:, -1:, :])


from moe_cycle.generation_dtype import ensure_generation_dtype_policy

ensure_generation_dtype_policy()

from ab.gpt.util.llm.Chatbot import ChatBot  # noqa: E402
from moe_cycle.gate_prompt import gate_proposal_prompt  # noqa: E402
from moe_cycle.gate_source import _generate_gate, _validate_gate_source  # noqa: E402
from moe_cycle.morphism import (  # noqa: E402
    _capture_native_gate_inputs,
    _model_logits,
    _seed_all,
)
from moe_gate_only import MoEGateSession  # noqa: E402


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    parser.add_argument("--output", type=Path, default=Path("out/llm_gate_replace_check"))
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument(
        "--device-map",
        default="cuda:0",
        help="Transformers device map; use 'cpu' for a CPU-only run",
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=None,
        help="Override the rotary cache size (DeepSeek defaults to a 20 MiB "
        "cache for 163840 positions; 8192 is enough for this smoke test)",
    )
    parser.add_argument(
        "--last-token-logits",
        action="store_true",
        help="Compute vocabulary logits only for the final position (saves "
        "prompt_len * vocab * 4 bytes; step-zero check uses the same slice)",
    )
    parser.add_argument(
        "--compact-prompt",
        action="store_true",
        help="Use a shorter contract prompt to reduce generation attention memory",
    )
    parser.add_argument(
        "--gate-source",
        type=Path,
        help="Skip LLM generation and use this gate Python file instead",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.output / stamp).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Run root: {run_root}", flush=True)

    _seed_all(args.seed)

    # ---------------------------------------------------------------- load
    _section("LOADING MODEL")
    model_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": args.local_files_only,
        "device_map": args.device_map,
    }
    if args.max_position_embeddings is not None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        config.max_position_embeddings = args.max_position_embeddings
        model_kwargs["config"] = config
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["gate", "lm_head"],
        )
    session = MoEGateSession.from_pretrained(
        args.model,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"local_files_only": args.local_files_only},
    )
    tokenizer = session.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loaded. dtype={torch.bfloat16}, 4bit={args.load_in_4bit}")
    if args.last_token_logits:
        session.model.lm_head = _LastTokenLogits(session.model.lm_head)
        print("LM head restricted to the final position (memory-saving)")

    # ------------------------------------------------------- discover gates
    _section("DISCOVERING NATIVE ROUTER GATES")
    sites = session.inspect()
    if not sites:
        raise SystemExit("No MoE gate sites discovered")
    shapes = sorted({(site.model_dim, site.num_experts) for site in sites})
    print(f"Found {len(sites)} MoE router gate site(s); shapes={shapes}")
    print(f"  pattern={sites[0].pattern}  attr={sites[0].gate_attr}  "
          f"class={type(sites[0].gate).__name__}")

    sample_input = tokenizer(
        "Generate an improved PyTorch neural network architecture.",
        return_tensors="pt",
    )
    session.model.eval()
    native_logits, _ = _capture_native_gate_inputs(session.model, sites, sample_input)
    print(f"Native logits shape={tuple(native_logits.shape)} "
          f"finite={bool(torch.isfinite(native_logits).all())}")

    # ------------------------------------------------------- ask the LLM
    _section("QUERYING THE LLM FOR A GATE")
    gate_dir = run_root / "gate_source"
    if args.gate_source is not None:
        source = args.gate_source.read_text(encoding="utf-8")
        _validate_gate_source(source, shapes)
        (gate_dir).mkdir(parents=True, exist_ok=True)
        (gate_dir / "proposal_prompt.txt").write_text(
            "(external gate source supplied; LLM not queried)", encoding="utf-8"
        )
        print(f"Using external gate source: {args.gate_source}")
    else:
        chat_bot = ChatBot(
            session.model,
            tokenizer,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if args.compact_prompt:
            prompt = (
                "Write a PyTorch Mixture-of-Experts router gate.\n"
                "Contract:\n"
                "* Start the module with `import torch` and `import torch.nn as nn`.\n"
                "* Define class Net(nn.Module) with "
                "__init__(self, model_dim: int, num_experts: int) and "
                "forward(self, x: torch.Tensor) -> torch.Tensor.\n"
                "* Expose exactly self.base = nn.Linear(model_dim, num_experts, "
                "bias=False); the native router weights are copied into self.base.\n"
                "* Immediately after init the complete gate must reproduce "
                "self.base(x) exactly.\n"
                "* It must add trainable parameters beyond self.base connected "
                "to the output (for example a zero-initialized residual branch), "
                "so training can change routing.\n"
                "* Token-independent; output finite raw logits with shape "
                "(..., num_experts); deterministic; only PyTorch; no softmax, "
                "top-k, dropout, batch norm, or randomness.\n"
                f"* Shapes: (model_dim, num_experts) = {shapes}.\n"
                "Return only the complete Python module between <gate> and </gate>."
            )
        else:
            prompt = gate_proposal_prompt(shapes)
        (gate_dir).mkdir(parents=True, exist_ok=True)
        (gate_dir / "rendered_prompt.txt").write_text(prompt, encoding="utf-8")
        print(f"Prompt length: {len(prompt)} chars")
        torch.cuda.empty_cache()
        sources = _generate_gate(
            chat_bot,
            shapes,
            args.attempts,
            args.max_new_tokens,
            gate_dir,
            reference_source="",
            seen_hashes=set(),
            prompt_override=prompt if args.compact_prompt else None,
        )
        source = sources[0]
    print(f"\nGenerated gate source ({len(source)} chars):\n{'-' * 70}")
    print(source)
    print("-" * 70)

    # --------------------------------------------------------- install it
    _section("REPLACING ROUTERS AND VERIFYING STEP ZERO")
    installs = session.replace_source(
        source,
        class_name="Net",
        layers=None,
        sample_input=sample_input,
        verify=True,
        top_k=None,
        allow_remote_code=True,
        dynamic_discovery=False,
        initialize_from_original=True,
    )
    print(f"Installed {len(installs)} replacement gate(s)")
    print(f"  installed wrapper: {type(installs[0].new_gate).__name__}")
    print(f"  trainable gate params: {len(session.gate_parameters())}")

    replacement_logits = _model_logits(session.model, sample_input)
    finite = bool(torch.isfinite(replacement_logits).all())
    max_abs = float((native_logits - replacement_logits).abs().max().item())
    equivalent = bool(
        torch.allclose(native_logits, replacement_logits, rtol=1e-5, atol=1e-5)
    )
    print(f"  replacement logits finite: {finite}")
    print(f"  max abs logit difference: {max_abs:.6f}")
    print(f"  step-zero equivalent: {equivalent}")

    # --------------------------------------------------- run the model on
    _section("RUNNING THE REPLACED MODEL (GENERATION)")
    prompt_text = "The capital of France is"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(session.model.device)
    with torch.no_grad():
        generated = session.model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Prompt: {prompt_text!r}")
    print(f"Output: {text!r}")

    # ------------------------------------------------------- route sanity
    _section("ROUTING SANITY (GATE LOGIT HOOKS)")
    for install in installs[:3]:
        wrapper = install.new_gate
        logits = getattr(wrapper, "_last_gate_logits", None)
        idx = getattr(wrapper, "_last_topk_idx", None)
        print(f"  {install.site.path}: "
              f"gate_logits={None if logits is None else tuple(logits.shape)} "
              f"topk_idx={None if idx is None else tuple(idx.shape)}")

    # ------------------------------------------------------------ results
    report = {
        "model": args.model,
        "load_in_4bit": args.load_in_4bit,
        "gate_sites": len(sites),
        "shapes": shapes,
        "prompt_length": None if args.gate_source else len(prompt),
        "generated_source": source,
        "installed": len(installs),
        "installed_wrapper": type(installs[0].new_gate).__name__,
        "replacement_logits_finite": finite,
        "max_abs_logit_difference": max_abs,
        "step_zero_equivalent": equivalent,
        "generation_output": text,
    }
    (run_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _section("RESTORING NATIVE ROUTERS")
    session.restore()
    print("Restored.")

    _section("SUMMARY")
    print(f"LLM gate installed on {len(installs)} sites: YES")
    print(f"Step-zero equivalent: {equivalent} (max abs diff {max_abs:.6f})")
    print(f"Finite logits: {finite}")
    print(f"Generation ran: {'YES' if text.strip() else 'NO'}")
    print(f"Report: {run_root / 'report.json'}")
    if not (finite and equivalent):
        raise SystemExit("FAILED: replacement gate is not step-zero equivalent")


if __name__ == "__main__":
    main()
