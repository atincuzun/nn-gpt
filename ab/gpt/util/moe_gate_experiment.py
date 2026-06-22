# ab/gpt/util/moe_gate_experiment.py
"""
MoE Gate Experiment -- replaces MoE gates with LLM-generated modules and trains
only the gate network while the base LLM remains frozen.

Integrates with the Tune.py pipeline:
  - Installs gates via moe_gate_only.universal
  - Trains gates on formatted NN-code prompt pairs (same data LoRA would use)
  - Reuses nn_gen (generate CV code) and NNEval (evaluate) unchanged
  - Saves per-gate, per-epoch checkpoints and aggregated metrics
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from ab.gpt.util.Const import nngpt_gate_dir, new_out_file, new_nn_file, synth_dir, epoch_dir
from ab.gpt.util.Util import create_file, exists, extract_str

# ── moe_gate_only imports ────────────────────────────────────────────────────
try:
    from moe_gate_only.universal import (
        find_moe_gates,
        install_gates as mg_install_gates,
        freeze_except_gates,
        count_parameters,
        GateInstall,
    )
except ImportError:
    raise ImportError(
        "moe_gate_only package not found. Ensure it is in your PYTHONPATH "
        "(e.g., export PYTHONPATH=<path-to-moe_gate_only>)."
    )

# ── Default baseline gate ────────────────────────────────────────────────────
_BASELINE_GATE_CODE = r"""
import torch
import torch.nn as nn

class BaselineGate(nn.Module):
    def __init__(self, model_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(model_dim, num_experts, bias=False)
        nn.init.normal_(self.fc.weight, std=0.02)

    def forward(self, x):
        return self.fc(x)
""".strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Gate code extraction & validation
# ═══════════════════════════════════════════════════════════════════════════════

def extract_gate_code(full_output: str) -> Optional[str]:
    """Extract the <gate>...</gate> block from LLM output."""
    return extract_str(full_output, "<gate>", "</gate>")


def validate_gate_code(
    gate_code: str, model_dim: int, num_experts: int, device=None, dtype=None
) -> bool:
    """Compile gate code, instantiate, and smoke-test forward pass."""
    ns: dict = {}
    try:
        compiled = compile(gate_code, "<gate>", "exec")
        exec(compiled, ns)
    except Exception:
        return False

    GateCls = None
    for obj in ns.values():
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
            GateCls = obj
            break

    if GateCls is None:
        return False

    try:
        gate = GateCls(model_dim, num_experts)
        x = torch.randn(2, model_dim)
        if device is not None:
            gate = gate.to(device)
            x = x.to(device)
        if dtype is not None and dtype.is_floating_point:
            gate = gate.to(dtype)
            x = x.to(dtype)
        y = gate(x)
        return y.shape == (2, num_experts)
    except Exception:
        return False


def compile_gate_class(gate_code: str) -> type:
    """Compile gate code and return the first nn.Module subclass found."""
    ns: dict = {}
    exec(compile(gate_code, "<gate>", "exec"), ns)
    for obj in ns.values():
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
            return obj
    raise ValueError("No nn.Module subclass found in gate code.")


# ═══════════════════════════════════════════════════════════════════════════════
# Gate installation & management
# ═══════════════════════════════════════════════════════════════════════════════

def _make_gate_factory(gate_cls: type, device, dtype):
    """Return a callable (model_dim, num_experts) -> gate module."""
    def factory(model_dim: int, num_experts: int) -> nn.Module:
        return gate_cls(model_dim, num_experts).to(device=device, dtype=dtype)
    return factory


def install_gates_on_model(
    model: nn.Module,
    gate_cls: type,
    layers: Optional[List[int]] = None,
    top_k: Optional[int] = None,
) -> List[GateInstall]:
    """Install gate_cls at every discovered MoE gate. Returns installations."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    factory = _make_gate_factory(gate_cls, device, dtype)
    try:
        installs = mg_install_gates(
            model,
            factory,
            layers=layers,
            verify=False,
            top_k=top_k,
        )
    except Exception:
        installs = mg_install_gates(
            model,
            factory,
            layers=layers,
            verify=False,
            top_k=top_k,
        )
    return installs


def rollback_installs(model: nn.Module, installs: List[GateInstall]) -> None:
    """Restore original gates (reverse order)."""
    for inst in reversed(installs):
        try:
            owner = inst.owner or inst.site.block
            attr_ = inst.attr or inst.site.gate_attr
            old = inst.old_child or inst.old_gate
            setattr(owner, attr_, old)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Gate training loop
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _eval_loss(model: nn.Module, loader, max_steps: int = 5) -> float:
    model.eval()
    losses: List[float] = []
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        ids = batch["input_ids"].to(model.device)
        labels = batch.get("labels", ids.clone()).to(model.device)
        out = model(input_ids=ids, labels=labels)
        losses.append(out.loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train_gates_epoch(
    model: nn.Module,
    installs: List[GateInstall],
    train_loader,
    val_loader,
    *,
    lr: float = 1e-3,
    steps_per_epoch: int = 50,
    grad_clip: float = 1.0,
    optimizer_state: Optional[dict] = None,
) -> Tuple[float, float, dict]:
    """Train gate parameters for one epoch. Returns (train_loss, val_loss, opt_state)."""
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return 0.0, 0.0, optimizer_state if optimizer_state else {}

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    epoch_losses: List[float] = []
    train_iter = iter(train_loader)

    for step in range(steps_per_epoch):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(model.device)
        labels = batch.get("labels", ids.clone()).to(model.device)

        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=labels)
        loss = out.loss
        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        epoch_losses.append(loss.item())

    train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
    val_loss_val = _eval_loss(model, val_loader) if val_loader is not None else train_loss
    opt_state = optimizer.state_dict()
    gc.collect()
    return train_loss, val_loss_val, opt_state


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

def _gate_epoch_dir(gate_id: int, inner_epoch: int) -> Path:
    return nngpt_gate_dir / f"gate_{gate_id:03d}" / f"epoch_{inner_epoch:02d}"


def _save_gate_snapshot(
    gate_id: int,
    inner_epoch: int,
    installs: List[GateInstall],
    accuracy: Optional[float],
    train_loss: float,
    val_loss: float,
    gate_code: str,
) -> None:
    """Save gate weights, gate code, and metrics for this epoch."""
    dir_path = _gate_epoch_dir(gate_id, inner_epoch)
    dir_path.mkdir(parents=True, exist_ok=True)

    gate_state: dict = {}
    for inst in installs:
        try:
            owner = inst.owner or inst.site.block
            attr_ = inst.attr or inst.site.gate_attr
            gate_mod = getattr(owner, attr_)
            gate_state[f"L{inst.site.layer_index}"] = {
                k: v.cpu() for k, v in gate_mod.state_dict().items()
            }
        except Exception:
            pass

    torch.save({"gate_state": gate_state}, dir_path / "gate_weights.pt")
    create_file(dir_path, "gate.py", gate_code)

    metrics: dict = {
        "gate_id": gate_id,
        "epoch": inner_epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    if accuracy is not None:
        metrics["cv_accuracy"] = accuracy

    with open(dir_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def _save_gate_summary(gate_id: int, epoch_results: List[dict], gate_code: str) -> None:
    """Write aggregated summary for one gate across all inner epochs."""
    dir_path = nngpt_gate_dir / f"gate_{gate_id:03d}"
    dir_path.mkdir(parents=True, exist_ok=True)

    acc_values = [r.get("cv_accuracy") for r in epoch_results if r.get("cv_accuracy") is not None]
    summary: dict = {
        "gate_id": gate_id,
        "num_epochs": len(epoch_results),
        "epochs": epoch_results,
    }
    if acc_values:
        summary["mean_cv_accuracy"] = sum(acc_values) / len(acc_values)
        summary["max_cv_accuracy"] = max(acc_values)
        summary["min_cv_accuracy"] = min(acc_values)

    with open(dir_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    create_file(dir_path, "gate.py", gate_code)


def _read_last_accuracy(gen_out_path: Path) -> Optional[float]:
    """Read best_accuracy from the cycle_results.json inside nn_gen output."""
    cycle_file = gen_out_path.parent / "cycle_results.json" if gen_out_path.name.startswith("B") else gen_out_path / "cycle_results.json"
    for candidate in [gen_out_path / "cycle_results.json", gen_out_path.parent / "cycle_results.json"]:
        if candidate.is_file():
            try:
                with open(candidate) as f:
                    data = json.load(f)
                acc = data.get("evaluation", {}).get("best_accuracy") or data.get("best_accuracy")
                if acc is not None:
                    return float(acc)
            except Exception:
                pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator -- the main entry point called from Tune.py
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_GATE_CODE = _BASELINE_GATE_CODE


def gate_experiment_run(
    model: nn.Module,
    tokenizer,
    *,
    gen_fn: Callable[[int, Path], None],
    train_dataset_builder: Callable,
    n_gates: int = 5,
    n_inner_epochs: int = 10,
    gate_layers: Optional[List[int]] = None,
    gate_train_lr: float = 1e-3,
    gate_train_steps: int = 50,
    gate_grad_clip: float = 1.0,
    baseline_gate_code: Optional[str] = None,
    chat_bot=None,
    nn_gen_args: Optional[dict] = None,
    top_k: Optional[int] = None,
) -> List[dict]:
    """Run the full gate experiment.

    Parameters
    ----------
    model: HF MoE model (mutable -- gates are installed/rolled back).
    tokenizer: matching tokenizer.
    gen_fn: callable (inner_epoch, out_path) that runs nn_gen + NNEval.
    train_dataset_builder: callable () -> (train_loader, val_loader).
    n_gates: number of gate candidates to evaluate.
    n_inner_epochs: training epochs per gate.
    gate_layers: layer indices to replace (None = all).
    baseline_gate_code: Python source for the initial gate module.
    chat_bot: optional ChatBot instance (updated per gate).
    top_k: optional top-k override for gate wrapper.

    Returns list of per-gate summary dicts.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    first_gate_code = baseline_gate_code or _DEFAULT_GATE_CODE

    all_gate_summaries: List[dict] = []

    for gate_id in range(n_gates):
        print(f"\n{'=' * 60}")
        print(f"  GATE CANDIDATE {gate_id + 1} / {n_gates}")
        print(f"{'=' * 60}")

        # ── Determine gate code ──────────────────────────────────────────
        if gate_id == 0:
            gate_code = first_gate_code
        else:
            fresh_gate_code = _generate_next_gate_code(chat_bot, model, gate_id)
            gate_code = fresh_gate_code if fresh_gate_code else first_gate_code

        gate_code = gate_code.strip()
        if not gate_code:
            print(f"  [WARN] No gate code for gate {gate_id}, using baseline")
            gate_code = _DEFAULT_GATE_CODE

        # ── Compile and validate ─────────────────────────────────────────
        gate_cls = compile_gate_class(gate_code)

        # ── Install gates ────────────────────────────────────────────────
        print(f"  Installing gates (layers={gate_layers}) ...")
        try:
            installs = install_gates_on_model(model, gate_cls, layers=gate_layers, top_k=top_k)
        except Exception as e:
            print(f"  [ERROR] Gate installation failed: {e}")
            all_gate_summaries.append({"gate_id": gate_id, "error": str(e)})
            continue

        if not installs:
            print(f"  [ERROR] No MoE gates found in model")
            all_gate_summaries.append({"gate_id": gate_id, "error": "No MoE gates found"})
            continue

        print(f"  Installed {len(installs)} gates across {len({inst.site.layer_index for inst in installs})} layers")

        # ── Freeze base ──────────────────────────────────────────────────
        freeze_except_gates(model, installs)
        trainable_n, total_n = count_parameters(model)
        print(f"  Trainable: {trainable_n:,} / {total_n:,} ({100 * trainable_n / max(1, total_n):.4f}%)")

        # ── Data loaders ─────────────────────────────────────────────────
        try:
            train_loader, val_loader = train_dataset_builder()
        except Exception as e:
            print(f"  [ERROR] Dataset builder failed: {e}")
            rollback_installs(model, installs)
            all_gate_summaries.append({"gate_id": gate_id, "error": f"Dataset: {e}"})
            continue

        # ── Update chat_bot ──────────────────────────────────────────────
        if chat_bot is not None:
            chat_bot.model = model

        # ── Inner epoch loop ─────────────────────────────────────────────
        epoch_results: List[dict] = []
        opt_state = None

        for inner_epoch in range(n_inner_epochs):
            print(f"\n  --- Inner epoch {inner_epoch + 1}/{n_inner_epochs} ---")

            # Generate CV code + evaluate
            out_path = _gate_epoch_dir(gate_id, inner_epoch)
            out_path.mkdir(parents=True, exist_ok=True)
            try:
                gen_fn(inner_epoch, out_path)
            except Exception as e:
                print(f"  [ERROR] gen_fn failed: {e}")

            # Read accuracy
            accuracy = _read_last_accuracy(out_path)
            if accuracy is not None:
                print(f"  CV accuracy: {accuracy:.4f}")
            else:
                print(f"  [WARN] No CV accuracy found")

            # Train gates
            train_loss, val_loss, opt_state = train_gates_epoch(
                model,
                installs,
                train_loader,
                val_loader,
                lr=gate_train_lr,
                steps_per_epoch=gate_train_steps,
                grad_clip=gate_grad_clip,
                optimizer_state=opt_state,
            )
            print(f"  Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}")

            # Save snapshot
            _save_gate_snapshot(
                gate_id, inner_epoch, installs, accuracy, train_loss, val_loss, gate_code,
            )

            epoch_results.append({
                "inner_epoch": inner_epoch,
                "cv_accuracy": accuracy,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

        # ── Save gate summary ────────────────────────────────────────────
        _save_gate_summary(gate_id, epoch_results, gate_code)

        # ── Rollback ─────────────────────────────────────────────────────
        print(f"  Rolling back gates...")
        rollback_installs(model, installs)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_gate_summaries.append({
            "gate_id": gate_id,
            "epoch_results": epoch_results,
        })

    # ── Write experiment summary ─────────────────────────────────────────────
    exp_summary_path = nngpt_gate_dir / "experiment_summary.json"
    exp_summary = {
        "n_gates": n_gates,
        "n_inner_epochs": n_inner_epochs,
        "gates": [
            {
                "gate_id": g["gate_id"],
                "mean_accuracy": (
                    sum(r["cv_accuracy"] for r in g.get("epoch_results", []) if r.get("cv_accuracy") is not None)
                    / max(1, sum(1 for r in g.get("epoch_results", []) if r.get("cv_accuracy") is not None))
                ) if g.get("epoch_results") else None,
            }
            for g in all_gate_summaries
        ],
    }
    with open(exp_summary_path, "w") as f:
        json.dump(exp_summary, f, indent=2)
    print(f"\nExperiment summary written to {exp_summary_path}")

    return all_gate_summaries


def _generate_next_gate_code(chat_bot, model, gate_id: int) -> Optional[str]:
    """Ask the LLM to generate a new gate module. Returns code or None."""
    if chat_bot is None:
        return None

    sample_sites = find_moe_gates(model)
    if not sample_sites:
        return None

    s = sample_sites[0]
    prompt = (
        f"You are an expert PyTorch neural network architect.\n"
        f"Generate a complete, runnable PyTorch gate module for a Mixture-of-Experts (MoE) model.\n"
        f"The gate receives hidden states of shape (batch, {s.model_dim}) and must output "
        f"logits of shape (batch, {s.num_experts}) for routing to {s.num_experts} experts.\n"
        f"Define exactly one `nn.Module` subclass with constructor `__init__(self, model_dim, num_experts)` "
        f"and `forward(self, x) -> torch.Tensor`.\n"
        f"Output only the code between <gate> and </gate> tags, no explanation."
    )
    try:
        code, hp, tr, full_out = chat_bot.chat(prompt, max_new_tokens=1024)
        gate_code = extract_gate_code(full_out)
        if gate_code and validate_gate_code(gate_code, s.model_dim, s.num_experts):
            return gate_code
    except Exception:
        pass
    return None
