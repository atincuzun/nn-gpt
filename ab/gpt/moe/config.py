from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from ab.gpt.moe.hf_moe_editor import MoEEditConfig


DEFAULT_MOE_EDIT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "mode": "disabled",
    "attach_tutel_adapter": True,
    "hf_top_k_override": None,
    "tutel_top_k": 2,
    "tutel_num_local_experts": 2,
    "adapter_hidden_mult": 2,
    "use_custom_gate": False,
    "use_custom_expert": False,
    "tutel_capacity_factor": 1.0,
    "tutel_gate_noise": 0.0,
    "tutel_gate_temperature": 0.8,
    "normalize_gate": True,
    "aux_weight": 1e-2,
    "train_lora": True,
    "joint_lora": True,
    "persist_adapter": True,
    "metadata_filename": "moe_edit_metadata.json",
    "adapter_state_filename": "tutel_adapter.pt",
    "prompt_key": "moe_recipe_tracking",
}


def normalize_moe_edit_config(raw_config: Optional[Dict[str, Any]], base_model_name: Optional[str] = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_MOE_EDIT_CONFIG)
    if raw_config:
        if not isinstance(raw_config, dict):
            raise TypeError(f"moe_edit config must be a dict, got {type(raw_config)}")
        config.update(raw_config)

    mode = str(config.get("mode", "disabled"))
    if mode not in {"disabled", "inference_only", "trainable_adapter"}:
        raise ValueError(f"Unsupported moe_edit.mode='{mode}'")

    config["enabled"] = bool(config.get("enabled", False)) and mode != "disabled"
    config["mode"] = mode
    config["attach_tutel_adapter"] = bool(config.get("attach_tutel_adapter", True))
    config["use_custom_gate"] = bool(config.get("use_custom_gate", False))
    config["use_custom_expert"] = bool(config.get("use_custom_expert", False))
    config["normalize_gate"] = bool(config.get("normalize_gate", True))
    config["train_lora"] = bool(config.get("train_lora", True))
    config["joint_lora"] = bool(config.get("joint_lora", True))
    config["persist_adapter"] = bool(config.get("persist_adapter", True))
    config["adapter_trainable"] = bool(config["enabled"] and mode == "trainable_adapter")
    config["base_model_name"] = base_model_name
    return config


def is_moe_edit_enabled(raw_config: Optional[Dict[str, Any]]) -> bool:
    if raw_config is None:
        return False
    return bool(raw_config.get("enabled", False)) and str(raw_config.get("mode", "disabled")) != "disabled"


def build_edit_config(raw_config: Optional[Dict[str, Any]]) -> MoEEditConfig:
    config = normalize_moe_edit_config(raw_config)
    return MoEEditConfig(
        hf_top_k_override=config.get("hf_top_k_override"),
        attach_tutel_adapter=config.get("attach_tutel_adapter", True),
        num_local_experts=int(config.get("tutel_num_local_experts", 2)),
        top_k=int(config.get("tutel_top_k", 2)),
        hidden_mult=int(config.get("adapter_hidden_mult", 2)),
        use_custom_gate=config.get("use_custom_gate", False),
        use_custom_expert=config.get("use_custom_expert", False),
        capacity_factor=float(config.get("tutel_capacity_factor", 1.0)),
        gate_noise=float(config.get("tutel_gate_noise", 0.0)),
        gate_temperature=float(config.get("tutel_gate_temperature", 0.8)),
        normalize_gate=bool(config.get("normalize_gate", True)),
        aux_weight=float(config.get("aux_weight", 1e-2)),
    )


def build_moe_recipe_record(runtime_config: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    record = {
        "base_model_name": runtime_config.get("base_model_name"),
        "selected_model_id": runtime_config.get("selected_model_id"),
        "mode": runtime_config.get("mode"),
        "attach_tutel_adapter": runtime_config.get("attach_tutel_adapter"),
        "hf_top_k_override": runtime_config.get("hf_top_k_override"),
        "tutel_top_k": runtime_config.get("tutel_top_k"),
        "tutel_num_local_experts": runtime_config.get("tutel_num_local_experts"),
        "adapter_hidden_mult": runtime_config.get("adapter_hidden_mult"),
        "use_custom_gate": runtime_config.get("use_custom_gate"),
        "use_custom_expert": runtime_config.get("use_custom_expert"),
        "tutel_capacity_factor": runtime_config.get("tutel_capacity_factor"),
        "tutel_gate_noise": runtime_config.get("tutel_gate_noise"),
        "tutel_gate_temperature": runtime_config.get("tutel_gate_temperature"),
        "normalize_gate": runtime_config.get("normalize_gate"),
        "adapter_trainable": runtime_config.get("adapter_trainable", False),
        "train_lora": runtime_config.get("train_lora", True),
        "joint_lora": runtime_config.get("joint_lora", True),
        "routing_changes": runtime_config.get("routing_changes", []),
        "adapter_summary": runtime_config.get("adapter_summary", []),
        "epoch": runtime_config.get("epoch", 0),
        "best_accuracy": runtime_config.get("best_accuracy", "N/A"),
        "avg_accuracy": runtime_config.get("avg_accuracy", "N/A"),
    }
    if extra:
        record.update(extra)
    return record
