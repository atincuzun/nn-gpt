#!/usr/bin/env python3
"""
Reusable Hugging Face + Tutel MoE loader/editor demo.

Key ideas:
1) Load a Hugging Face MoE model from a preset or explicit model id.
2) Optionally edit native HF MoE routing attributes in-place.
3) Optionally attach a Tutel adapter to the causal LM forward path in-place.
4) Return the edited model immediately; training is optional, not required.
5) Report routing changes, adapter configuration, and immediate output deltas.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, MixtralConfig, MixtralForCausalLM

try:
    from ab.gpt.moe.tutel_components import CustomExpertMLP, CustomLinearGate
except ImportError:
    from tutel_components import CustomExpertMLP, CustomLinearGate

try:
    from tutel import moe as tutel_moe
    from tutel import system
except ImportError as exc:
    raise RuntimeError(
        "Tutel is not installed in the active virtual environment. Install it with "
        "`python -m pip install -r req-no-isolation.txt --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu126`."
    ) from exc


HF_MOE_MODEL_PRESETS: Dict[str, Tuple[str, ...]] = {
    "tiny-mixtral": (
        "hf-internal-testing/tiny-random-MixtralForCausalLM",
    ),
    "mixtral-8x7b-base": (
        "mistralai/Mixtral-8x7B-v0.1",
    ),
    "mixtral-8x7b-instruct": (
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ),
    "qwen1.5-moe-a2.7b": (
        "Qwen/Qwen1.5-MoE-A2.7B",
    ),
    "deepseek-moe-16b-base": (
        "deepseek-ai/deepseek-moe-16b-base",
    ),
}


@dataclass
class MoEEditConfig:
    hf_top_k_override: Optional[int] = None
    attach_tutel_adapter: bool = True
    num_local_experts: int = 2
    top_k: int = 2
    hidden_mult: int = 2
    use_custom_gate: bool = False
    use_custom_expert: bool = False
    capacity_factor: float = 1.0
    gate_noise: float = 0.0
    gate_temperature: float = 0.8
    normalize_gate: bool = True
    aux_weight: float = 1e-2


@dataclass
class MoEEditResult:
    selected_model_id: str
    routing_before: List[str]
    routing_after: List[str]
    routing_changes: List[str]
    adapter_attached: bool
    adapter_summary: List[str]


@dataclass
class LoadedHFMoEModel:
    model: nn.Module
    selected_model_id: str
    device: torch.device
    dtype: torch.dtype
class TutelAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_local_experts: int,
        top_k: int,
        hidden_mult: int,
        use_custom_gate: bool,
        use_custom_expert: bool,
        capacity_factor: float,
        gate_noise: float,
        gate_temperature: float,
        normalize_gate: bool,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.use_custom_gate = use_custom_gate
        self.use_custom_expert = use_custom_expert
        self.top_k = int(top_k)
        self.capacity_factor = float(capacity_factor)
        self.gate_noise = float(gate_noise)
        self.gate_temperature = float(gate_temperature)
        self.normalize_gate = bool(normalize_gate)

        gate_type = (
            {
                "type": "custom",
                "module": CustomLinearGate,
                "k": top_k,
                "temperature": self.gate_temperature,
                "capacity_factor": self.capacity_factor,
                "gate_noise": self.gate_noise,
            }
            if use_custom_gate
            else {
                "type": "top",
                "k": top_k,
                "capacity_factor": self.capacity_factor,
                "gate_noise": self.gate_noise,
            }
        )

        if use_custom_expert:
            experts = {
                "type": "custom",
                "module": CustomExpertMLP,
                "hidden_size_per_expert": hidden_mult * hidden_size,
                "num_experts_per_device": num_local_experts,
            }
        else:
            experts = {
                "type": "ffn",
                "num_experts_per_device": num_local_experts,
                "hidden_size_per_expert": hidden_mult * hidden_size,
                "activation_fn": lambda x: F.silu(x),
            }

        self.moe = tutel_moe.moe_layer(
            gate_type=gate_type,
            experts=experts,
            model_dim=hidden_size,
            normalize_gate=self.normalize_gate,
            scan_expert_func=lambda _n, p: setattr(p, "skip_allreduce", True),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        adapted = self.moe(hidden_states)
        return self.layer_norm(hidden_states + adapted)

    def describe(self) -> List[str]:
        gate = self.moe.gates[0]
        lines = [
            f"custom_gate={self.use_custom_gate}",
            f"custom_expert={self.use_custom_expert}",
            f"num_global_experts={int(self.moe.num_global_experts)}",
            f"gate_class={gate.__class__.__name__}",
            f"top_k={getattr(gate, 'top_k', 'N/A')}",
            f"capacity_factor={getattr(gate, 'capacity_factor', 'N/A')}",
            f"gate_noise={getattr(gate, 'gate_noise', 'N/A')}",
            f"normalize_gate={self.normalize_gate}",
        ]
        if self.use_custom_gate:
            lines.append(f"gate_temperature={self.gate_temperature}")
        return lines


class HFMoEEditor:
    """Apply HF routing edits and an optional Tutel adapter in-place."""

    def __init__(self, config: MoEEditConfig):
        self.config = config

    def apply(self, model: nn.Module, selected_model_id: str = "unknown") -> Tuple[nn.Module, MoEEditResult]:
        routing_before = collect_moe_routing_info(model)
        routing_changes = modify_hf_moe_routing(model, self.config.hf_top_k_override)
        adapter_summary = ["adapter_attached=False"]

        if self.config.attach_tutel_adapter:
            hidden_size = infer_hidden_size(model)
            adapter_device, adapter_dtype = infer_model_device_dtype(model)
            adapter = TutelAdapter(
                hidden_size=hidden_size,
                num_local_experts=self.config.num_local_experts,
                top_k=self.config.top_k,
                hidden_mult=self.config.hidden_mult,
                use_custom_gate=self.config.use_custom_gate,
                use_custom_expert=self.config.use_custom_expert,
                capacity_factor=self.config.capacity_factor,
                gate_noise=self.config.gate_noise,
                gate_temperature=self.config.gate_temperature,
                normalize_gate=self.config.normalize_gate,
            ).to(device=adapter_device, dtype=adapter_dtype)
            attach_tutel_adapter_inplace(model, adapter, aux_weight=self.config.aux_weight)
            adapter_summary = adapter.describe()

        routing_after = collect_moe_routing_info(model)
        result = MoEEditResult(
            selected_model_id=selected_model_id,
            routing_before=routing_before,
            routing_after=routing_after,
            routing_changes=routing_changes,
            adapter_attached=self.config.attach_tutel_adapter,
            adapter_summary=adapter_summary,
        )
        setattr(model, "_moe_edit_result", result)
        return model, result


def load_hf_moe_model(
    model_id: Optional[str] = None,
    model_preset: Optional[str] = None,
    fallback_model_ids: Sequence[str] = (),
    device: Optional[str] = None,
    local_files_only: bool = False,
    trust_remote_code: bool = True,
    allow_local_fallback: bool = True,
) -> LoadedHFMoEModel:
    runtime_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = torch.bfloat16 if runtime_device.type == "cuda" else torch.float32
    candidate_ids = resolve_candidate_model_ids(model_id, model_preset, fallback_model_ids)
    model, selected_id = load_first_available_model(
        candidate_ids,
        model_dtype,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        allow_local_fallback=allow_local_fallback,
    )
    model = model.to(runtime_device)
    return LoadedHFMoEModel(
        model=model,
        selected_model_id=selected_id,
        device=runtime_device,
        dtype=model_dtype,
    )


def load_and_edit_hf_moe_model(
    *,
    edit_config: Optional[MoEEditConfig] = None,
    model_id: Optional[str] = None,
    model_preset: Optional[str] = None,
    fallback_model_ids: Sequence[str] = (),
    device: Optional[str] = None,
    local_files_only: bool = False,
    trust_remote_code: bool = True,
    allow_local_fallback: bool = True,
) -> Tuple[nn.Module, MoEEditResult, LoadedHFMoEModel]:
    loaded = load_hf_moe_model(
        model_id=model_id,
        model_preset=model_preset,
        fallback_model_ids=fallback_model_ids,
        device=device,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        allow_local_fallback=allow_local_fallback,
    )
    editor = HFMoEEditor(edit_config or MoEEditConfig())
    edited_model, edit_result = editor.apply(loaded.model, selected_model_id=loaded.selected_model_id)
    return edited_model, edit_result, loaded


def unique_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--model-preset", type=str, choices=sorted(HF_MOE_MODEL_PRESETS), default=None)
    parser.add_argument("--list-model-presets", action="store_true")
    parser.add_argument("--fallback-model-ids", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hf-top-k-override", type=int, default=None)
    parser.add_argument("--disable-tutel-adapter", action="store_true")
    parser.add_argument("--tutel-top-k", type=int, default=2)
    parser.add_argument("--tutel-num-local-experts", type=int, default=2)
    parser.add_argument("--adapter-hidden-mult", type=int, default=2)
    parser.add_argument("--aux-weight", type=float, default=1e-2)
    parser.add_argument("--use-custom-gate", action="store_true")
    parser.add_argument("--use-custom-expert", action="store_true")
    parser.add_argument("--tutel-capacity-factor", type=float, default=1.0)
    parser.add_argument("--tutel-gate-noise", type=float, default=0.0)
    parser.add_argument("--tutel-gate-temperature", type=float, default=0.8)
    parser.add_argument("--disable-normalize-gate", action="store_true")
    parser.add_argument("--train-base-model", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--disable-remote-code", action="store_true")
    parser.add_argument("--no-local-fallback", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def init_env(device: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        backend = "nccl" if device == "cuda" else "gloo"
        return system.init_data_model_parallel(group_count=1, backend=backend)

    local_device = torch.device("cuda", 0) if (device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
    if local_device.type == "cuda":
        torch.cuda.set_device(local_device)

    return SimpleNamespace(
        global_size=1,
        global_rank=0,
        local_device=local_device,
        dist_print=print,
        is_distributed=False,
    )


def infer_hidden_size(model: nn.Module) -> int:
    for key in ("hidden_size", "d_model", "n_embd"):
        value = getattr(model.config, key, None)
        if isinstance(value, int):
            return value
    raise ValueError("Unable to infer hidden size from model config.")


def infer_model_device_dtype(model: nn.Module) -> Tuple[torch.device, torch.dtype]:
    for parameter in model.parameters():
        if torch.is_floating_point(parameter):
            return parameter.device, parameter.dtype
    return torch.device("cpu"), torch.float32


def build_tiny_local_mixtral(dtype: torch.dtype) -> nn.Module:
    cfg = MixtralConfig(
        vocab_size=4096,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=256,
        num_local_experts=4,
        num_experts_per_tok=2,
        rms_norm_eps=1e-5,
    )
    model = MixtralForCausalLM(cfg)
    return model.to(dtype=dtype)


def resolve_candidate_model_ids(model_id: Optional[str], model_preset: Optional[str], fallback_model_ids: Sequence[str]) -> List[str]:
    candidate_ids: List[str] = []
    if model_id:
        candidate_ids.append(model_id)
    if model_preset:
        candidate_ids.extend(HF_MOE_MODEL_PRESETS[model_preset])
    candidate_ids.extend(fallback_model_ids)
    if not candidate_ids:
        candidate_ids.extend(HF_MOE_MODEL_PRESETS["tiny-mixtral"])
    return unique_preserve_order(candidate_ids)


def load_first_available_model(
    model_ids: Iterable[str],
    dtype: torch.dtype,
    local_files_only: bool,
    trust_remote_code: bool,
    allow_local_fallback: bool,
) -> Tuple[nn.Module, str]:
    last_err: Optional[Exception] = None
    for model_id in model_ids:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            return model, model_id
        except Exception as err:
            last_err = err
            print(f"[WARN] Failed to load '{model_id}': {err}")

    if allow_local_fallback:
        print("[WARN] Falling back to locally instantiated tiny Mixtral model (offline mode).")
        return build_tiny_local_mixtral(dtype), "local-tiny-mixtral"

    raise RuntimeError("Could not load any HF MoE model id.") from last_err


def collect_moe_routing_info(model: nn.Module) -> List[str]:
    rows: List[str] = []
    for module_name, module in model.named_modules():
        parts: List[str] = []
        if hasattr(module, "top_k"):
            parts.append(f"top_k={getattr(module, 'top_k')}")
        if hasattr(module, "num_experts_per_tok"):
            parts.append(f"num_experts_per_tok={getattr(module, 'num_experts_per_tok')}")
        if parts:
            rows.append(f"{module_name}: " + ", ".join(parts))
    return rows


def is_probably_moe_model(model: nn.Module) -> bool:
    config_keys = (
        "num_local_experts",
        "num_experts_per_tok",
        "num_experts",
        "moe_intermediate_size",
    )
    if any(hasattr(model.config, key) for key in config_keys):
        return True
    return len(collect_moe_routing_info(model)) > 0


def modify_hf_moe_routing(model: nn.Module, top_k_override: Optional[int]) -> List[str]:
    if top_k_override is None:
        return []

    changes: List[str] = []
    for module_name, module in model.named_modules():
        if hasattr(module, "top_k") and isinstance(getattr(module, "top_k"), int):
            old = int(getattr(module, "top_k"))
            new = max(1, min(old, top_k_override))
            if new != old:
                setattr(module, "top_k", new)
                changes.append(f"{module_name}.top_k: {old} -> {new}")

        if hasattr(module, "num_experts_per_tok") and isinstance(getattr(module, "num_experts_per_tok"), int):
            old = int(getattr(module, "num_experts_per_tok"))
            new = max(1, min(old, top_k_override))
            if new != old:
                setattr(module, "num_experts_per_tok", new)
                changes.append(f"{module_name}.num_experts_per_tok: {old} -> {new}")

    return changes


def attach_tutel_adapter_inplace(model: nn.Module, adapter: TutelAdapter, aux_weight: float) -> nn.Module:
    if getattr(model, "_tutel_adapter_attached", False):
        raise RuntimeError("A Tutel adapter is already attached to this model.")
    if model.get_output_embeddings() is None:
        raise ValueError("The selected causal LM does not expose output embeddings.")

    model.tutel_adapter = adapter
    model.moe_aux_weight = float(aux_weight)
    model._tutel_original_forward = model.forward
    model._tutel_adapter_attached = True

    def patched_forward(self, *args, **kwargs):
        requested_hidden_states = bool(kwargs.get("output_hidden_states", False))
        requested_return_dict = kwargs.get("return_dict")
        if requested_return_dict is None:
            requested_return_dict = bool(getattr(self.config, "use_return_dict", True))

        labels = kwargs.get("labels")
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True

        outputs = self._tutel_original_forward(*args, **kwargs)
        hidden_states = outputs.hidden_states[-1]
        edited_hidden = self.tutel_adapter(hidden_states)
        logits = self.get_output_embeddings()(edited_hidden)
        outputs.logits = logits

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            l_aux = getattr(self.tutel_adapter.moe, "l_aux", None)
            if l_aux is not None:
                loss = loss + float(self.moe_aux_weight) * l_aux
            outputs.loss = loss

        if not requested_hidden_states:
            outputs.hidden_states = None

        if not requested_return_dict:
            return outputs.to_tuple()
        return outputs

    model.forward = MethodType(patched_forward, model)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def configure_trainable_parameters(model: nn.Module, train_base_model: bool) -> None:
    if train_base_model:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("tutel_adapter"))


def random_batch(vocab_size: int, batch_size: int, seq_len: int, device: torch.device):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def summarize_hf_routing(rows: List[str], max_rows: int = 8) -> List[str]:
    if not rows:
        return ["<none>"]
    shown = rows[:max_rows]
    if len(rows) > max_rows:
        shown = shown + [f"... and {len(rows) - max_rows} more"]
    return shown


def probe_adapter_routing(adapter: TutelAdapter, hidden_states: torch.Tensor) -> List[str]:
    gate = adapter.moe.gates[0]
    x = hidden_states.reshape(-1, hidden_states.size(-1))
    with torch.no_grad():
        logits = gate(x)
        probs = F.softmax(logits, dim=-1)
        k = int(getattr(gate, "top_k", 1))
        k = max(1, min(k, probs.size(-1)))
        topk_ids = torch.topk(probs, k=k, dim=-1).indices.reshape(-1)
        counts = torch.bincount(topk_ids, minlength=probs.size(-1)).float()
        token_share = counts / max(1.0, float(topk_ids.numel()))
    preview = ", ".join([f"e{i}:{float(v):.3f}" for i, v in enumerate(token_share[: min(8, token_share.numel())])])
    return [
        f"tokens={x.size(0)} experts={probs.size(-1)} k={k}",
        f"mean_gate_entropy={float((-probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()):.5f}",
        f"token_share_preview={preview}",
    ]


def forward_snapshot(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
):
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )


def print_model_presets() -> None:
    print("Available HF MoE presets:")
    for preset_name, model_ids in HF_MOE_MODEL_PRESETS.items():
        print(f"  - {preset_name}: {', '.join(model_ids)}")


def main(cli_args: Optional[argparse.Namespace] = None):
    args = cli_args or parse_args()
    if args.list_model_presets:
        print_model_presets()
        return

    torch.manual_seed(0)

    env = init_env(args.device)
    device = env.local_device
    dist_print = env.dist_print if hasattr(env, "dist_print") else print

    loaded = load_hf_moe_model(
        model_id=args.model_id,
        model_preset=args.model_preset,
        fallback_model_ids=args.fallback_model_ids,
        device=str(device),
        local_files_only=args.local_files_only,
        trust_remote_code=not args.disable_remote_code,
        allow_local_fallback=not args.no_local_fallback,
    )
    base_model = loaded.model
    selected_id = loaded.selected_model_id
    candidate_ids = resolve_candidate_model_ids(args.model_id, args.model_preset, args.fallback_model_ids)

    dist_print(f"[INFO] Loaded model: {selected_id}")
    dist_print(f"[INFO] Probable HF MoE model: {is_probably_moe_model(base_model)}")

    vocab_size = int(getattr(base_model.config, "vocab_size", 32000))
    fixed_input_ids, fixed_attention_mask, fixed_labels = random_batch(
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
    )

    base_model.eval()
    with torch.no_grad():
        snapshot_before = forward_snapshot(
            base_model,
            fixed_input_ids,
            fixed_attention_mask,
            labels=fixed_labels,
        )
        logits_before_edit = snapshot_before.logits.detach().clone()

    editor = HFMoEEditor(
        MoEEditConfig(
            hf_top_k_override=args.hf_top_k_override,
            attach_tutel_adapter=not args.disable_tutel_adapter,
            num_local_experts=args.tutel_num_local_experts,
            top_k=args.tutel_top_k,
            hidden_mult=args.adapter_hidden_mult,
            use_custom_gate=args.use_custom_gate,
            use_custom_expert=args.use_custom_expert,
            capacity_factor=args.tutel_capacity_factor,
            gate_noise=args.tutel_gate_noise,
            gate_temperature=args.tutel_gate_temperature,
            normalize_gate=not args.disable_normalize_gate,
            aux_weight=args.aux_weight,
        )
    )
    edited_model, edit_result = editor.apply(base_model, selected_model_id=selected_id)

    dist_print(f"[INFO] Candidate model ids: {candidate_ids}")
    dist_print(f"[INFO] HF routing modules before edit: {len(edit_result.routing_before)}")
    for row in summarize_hf_routing(edit_result.routing_before):
        dist_print(f"  [HF-BEFORE] {row}")

    if edit_result.routing_changes:
        dist_print("[INFO] HF routing edits:")
        for line in edit_result.routing_changes[:12]:
            dist_print(f"  - {line}")
        if len(edit_result.routing_changes) > 12:
            dist_print(f"  - ... and {len(edit_result.routing_changes) - 12} more")
    else:
        dist_print("[INFO] No mutable HF routing attributes changed.")

    for row in summarize_hf_routing(edit_result.routing_after):
        dist_print(f"  [HF-AFTER] {row}")

    if edit_result.adapter_attached:
        dist_print("[INFO] Tutel adapter config:")
        for line in edit_result.adapter_summary:
            dist_print(f"  [TUTEL-CONFIG] {line}")
    else:
        dist_print("[INFO] Tutel adapter attachment disabled; only native HF routing edits were applied.")

    edited_model.eval()
    with torch.no_grad():
        snapshot_after_edit = forward_snapshot(
            edited_model,
            fixed_input_ids,
            fixed_attention_mask,
            labels=fixed_labels,
        )
        logits_after_edit = snapshot_after_edit.logits.detach().clone()

    edit_delta = (logits_after_edit - logits_before_edit).abs()
    mean_edit_delta = float(edit_delta.mean())
    max_edit_delta = float(edit_delta.max())

    if getattr(edited_model, "_tutel_adapter_attached", False):
        probe_after_edit = probe_adapter_routing(edited_model.tutel_adapter, snapshot_after_edit.hidden_states[-1])
        dist_print("[INFO] Tutel routing probe immediately after edit:")
        for row in probe_after_edit:
            dist_print(f"  [TUTEL-EDITED] {row}")

    did_optimize = False
    total_params = trainable_params = 0
    logits_after_training = logits_after_edit

    if args.steps > 0:
        configure_trainable_parameters(edited_model, train_base_model=args.train_base_model)
        total_params, trainable_params = count_parameters(edited_model)
        dist_print(
            f"[INFO] Params total={total_params:,} trainable={trainable_params:,} "
            f"({100.0 * trainable_params / max(1, total_params):.4f}%)"
        )

        trainable = [p for p in edited_model.parameters() if p.requires_grad]
        if trainable:
            optimizer = torch.optim.AdamW(trainable, lr=args.lr)
            edited_model.train()
            for step in range(args.steps):
                input_ids, attention_mask, labels = random_batch(
                    vocab_size=vocab_size,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                out = edited_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                    return_dict=True,
                )
                loss = out.loss
                loss.backward()
                optimizer.step()

                l_aux = None
                if getattr(edited_model, "_tutel_adapter_attached", False):
                    l_aux = getattr(edited_model.tutel_adapter.moe, "l_aux", None)
                l_aux_value = float(l_aux.detach()) if l_aux is not None else 0.0
                dist_print(f"[STEP {step:03d}] loss={float(loss.detach()):.5f} l_aux={l_aux_value:.5f}")
            did_optimize = True
        else:
            dist_print("[WARN] No trainable parameters found for optimization. Skipping training loop.")

        edited_model.eval()
        with torch.no_grad():
            snapshot_after_training = forward_snapshot(
                edited_model,
                fixed_input_ids,
                fixed_attention_mask,
                labels=fixed_labels,
            )
            logits_after_training = snapshot_after_training.logits.detach().clone()
    else:
        dist_print("[INFO] Adapter/base-model training skipped. The edited model is still usable immediately after edit.")

    final_delta = (logits_after_training - logits_before_edit).abs()
    mean_final_delta = float(final_delta.mean())
    max_final_delta = float(final_delta.max())

    dist_print("[VERIFY] Edited HF/Tutel MoE model summary:")
    dist_print(f"  - selected_model: {selected_id}")
    dist_print(f"  - adapter_attached: {edit_result.adapter_attached}")
    dist_print(f"  - hf_routing_changes: {len(edit_result.routing_changes)}")
    dist_print(f"  - logits_mean_abs_delta_after_edit: {mean_edit_delta:.8f}")
    dist_print(f"  - logits_max_abs_delta_after_edit: {max_edit_delta:.8f}")
    dist_print(f"  - optimization_ran: {did_optimize}")
    if args.steps > 0:
        dist_print(f"  - logits_mean_abs_delta_after_training: {mean_final_delta:.8f}")
        dist_print(f"  - logits_max_abs_delta_after_training: {max_final_delta:.8f}")
    dist_print("[DONE] Complete.")


if __name__ == "__main__":
    main()
