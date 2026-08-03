"""Diagnostic: print MoE gate layer names, weights, and routing maps before/after replacement."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)


def _print_gate_weights(gate: nn.Module, prefix: str = "") -> None:
    for name, param in gate.named_parameters():
        data = param.detach().float().cpu()
        print(f"  {prefix}{name}: shape={list(param.shape)}, dtype={param.dtype}")
        flat = data.reshape(-1)
        n = min(8, flat.numel())
        vals = ", ".join(f"{v:.6f}" for v in flat[:n].tolist())
        extra = ", ..." if flat.numel() > n else ""
        print(f"    first {n} values: [{vals}{extra}]")
        print(f"    mean={data.mean().item():.6f}, std={data.std().item():.6f}, "
              f"min={data.min().item():.6f}, max={data.max().item():.6f}, "
              f"norm={data.norm().item():.6f}")


def _print_routing(label: str, topk_idx, topk_weight, num_experts: int) -> None:
    if topk_idx is None:
        print(f"  {label}: no routing data captured")
        return
    idx = topk_idx.detach().cpu()
    wts = topk_weight.detach().cpu() if topk_weight is not None else None
    print(f"  {label}:")
    print(f"    topk_idx shape: {list(idx.shape)}")
    print(f"    topk_idx[:8]: {idx.reshape(-1)[:8].tolist()}")
    if wts is not None:
        print(f"    topk_weight[:8]: {wts.reshape(-1)[:8].tolist()}")
    counts = torch.bincount(idx.reshape(-1), minlength=num_experts).int()
    total = counts.sum().item()
    print(f"    expert assignment counts (total={total}): {counts.tolist()}")
    fractions = counts.float() / max(total, 1)
    print(f"    expert fractions: {[f'{f:.4f}' for f in fractions.tolist()]}")


def main() -> None:
    from moe_gate_only import MoEGateSession, collect_gate_metrics

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    parser.add_argument("--gate", default="linear", help="Replacement gate factory name")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--initialize-from-original", action="store_true",
                        help="Copy native router weights into replacement gate base.weight")
    parser.add_argument("--teacher-student", action="store_true",
                        help="Use teacher-student mode")
    parser.add_argument("--prompt", default="What is 2+2?",
                        help="Sample prompt for routing map capture")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # ---- Load model ----
    _print_section("LOADING MODEL")
    session = MoEGateSession.from_pretrained(
        args.model,
        trust_remote_code=True,
        model_kwargs={
            "torch_dtype": dtype,
            "device_map": args.device_map,
            "local_files_only": args.local_files_only,
        },
        tokenizer_kwargs={"local_files_only": args.local_files_only},
    )
    model = session.model
    tokenizer = session.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model: {args.model}")
    print(f"Device map: {getattr(model, 'hf_device_map', 'N/A')}")

    # ---- Prepare sample input ----
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    print(f"Sample prompt: {args.prompt!r}")
    print(f"Input IDs shape: {list(input_ids.shape)}")

    # ---- Discover gates ----
    _print_section("DISCOVERING MOE GATES (BEFORE REPLACEMENT)")
    sites = session.inspect()
    print(f"Found {len(sites)} MoE gate site(s)\n")

    for site in sites:
        print(f"  Site: {site.path}")
        print(f"    layer_index: {site.layer_index}")
        print(f"    pattern: {site.pattern}")
        print(f"    model_dim: {site.model_dim}")
        print(f"    num_experts: {site.num_experts}")
        print(f"    gate_attr: {site.gate_attr}")
        print(f"    gate class: {type(site.gate).__name__}")
        print()

    # ---- Print native gate weights ----
    _print_section("NATIVE GATE WEIGHTS (BEFORE REPLACEMENT)")
    for site in sites:
        print(f"--- {site.path} ({type(site.gate).__name__}) ---")
        _print_gate_weights(site.gate, prefix="  ")
        print()

    # ---- Forward pass BEFORE replacement to capture routing ----
    _print_section("ROUTING MAPS (BEFORE REPLACEMENT)")
    model.eval()
    with torch.no_grad():
        outputs_before = model(input_ids=input_ids)
    for site in sites:
        gate = site.gate
        idx = getattr(gate, "_last_topk_idx", None)
        wts = getattr(gate, "_last_topk_weight", None)
        _print_routing(site.path, idx, wts, site.num_experts)
        print()

    # ---- Replace gates ----
    _print_section(f"REPLACING GATES WITH: {args.gate}")
    installs = session.replace(
        args.gate,
        initialize_from_original=args.initialize_from_original,
        teacher_student=args.teacher_student,
    )
    print(f"Installed {len(installs)} replacement gate(s)\n")

    for install in installs:
        site = install.site
        new_gate = install.new_gate
        print(f"  Site: {site.path}")
        print(f"    old gate class: {type(install.old_gate).__name__}")
        print(f"    new gate class: {type(new_gate).__name__}")
        print(f"    mode: {install.mode}")
        if install.student_gate is not None:
            print(f"    student gate class: {type(install.student_gate).__name__}")
        if install.teacher_gate is not None:
            print(f"    teacher gate class: {type(install.teacher_gate).__name__}")
        print()

    # ---- Print replacement gate weights ----
    _print_section("REPLACEMENT GATE WEIGHTS (AFTER REPLACEMENT)")
    for install in installs:
        site = install.site
        print(f"--- {site.path} (installed as {type(install.new_gate).__name__}) ---")
        if install.student_gate is not None:
            print("  [Student gate (trainable)]:")
            _print_gate_weights(install.student_gate, prefix="    ")
            print("  [Teacher gate (frozen)]:")
            _print_gate_weights(install.teacher_gate, prefix="    ")
        else:
            _print_gate_weights(install.new_gate, prefix="  ")
        print()

    # ---- Forward pass AFTER replacement to capture routing ----
    _print_section("ROUTING MAPS (AFTER REPLACEMENT)")
    with torch.no_grad():
        outputs_after = model(input_ids=input_ids)
    for install in installs:
        module = install.new_gate
        idx = getattr(module, "_last_topk_idx", None)
        wts = getattr(module, "_last_topk_weight", None)
        _print_routing(install.site.path, idx, wts, install.site.num_experts)
        if install.student_gate is not None and install.teacher_gate is not None:
            t_idx = getattr(module, "_last_teacher_topk_idx", None)
            s_idx = getattr(module, "_last_student_topk_idx", None)
            if t_idx is not None:
                print(f"    teacher topk_idx[:8]: {t_idx.detach().cpu().reshape(-1)[:8].tolist()}")
            if s_idx is not None:
                print(f"    student topk_idx[:8]: {s_idx.detach().cpu().reshape(-1)[:8].tolist()}")
        print()

    # ---- Compare logits ----
    _print_section("OUTPUT LOGIT COMPARISON (BEFORE vs AFTER)")
    logits_before = outputs_before.logits.detach().float().cpu()
    logits_after = outputs_after.logits.detach().float().cpu()
    diff = (logits_after - logits_before).abs()
    print(f"  logits shape: {list(logits_before.shape)}")
    print(f"  max abs diff: {diff.max().item():.6f}")
    print(f"  mean abs diff: {diff.mean().item():.6f}")
    print(f"  L2 diff norm: {diff.norm().item():.6f}")
    print(f"  cosine similarity (flattened): "
          f"{torch.nn.functional.cosine_similarity(logits_before.reshape(1, -1), logits_after.reshape(1, -1)).item():.6f}")

    # ---- Gate metrics ----
    _print_section("GATE METRICS (AFTER REPLACEMENT)")
    metrics = collect_gate_metrics(installs)
    print(json.dumps(metrics, indent=2, default=str))

    _print_section("DONE")
    print("Gate inspection completed successfully.")


if __name__ == "__main__":
    main()
