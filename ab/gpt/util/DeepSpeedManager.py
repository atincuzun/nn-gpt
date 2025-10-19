# ab/gpt/util/deepspeed_manager.py
import json
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import deepspeed
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def get_local_rank() -> int:
    return env_int("LOCAL_RANK", 0)


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> None:
    """
    Initialize torch.distributed using standard env vars:
    - MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK
    Works with torchrun, SLURM, k8s launchers, etc.
    """
    if is_dist_available_and_initialized():
        return

    world_size = env_int("WORLD_SIZE", 1)
    if world_size > 1:
        torch.cuda.set_device(get_local_rank())
        dist.init_process_group(backend=backend, init_method="env://")
        # Optional NCCL sanity
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_DEBUG", "WARN")


def barrier() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()


def cleanup() -> None:
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def load_ds_config(config_path_or_dict):
    if isinstance(config_path_or_dict, (str, os.PathLike)):
        with open(config_path_or_dict) as f:
            return json.load(f)
    return dict(config_path_or_dict)


def patch_runtime_from_training_args(ds_cfg: dict, training_args) -> dict:
    """
    Keep your JSON file static; override per-run knobs from TrainingArguments.
    """
    if hasattr(training_args, "per_device_train_batch_size"):
        ds_cfg["train_micro_batch_size_per_gpu"] = int(training_args.per_device_train_batch_size)
    if hasattr(training_args, "gradient_accumulation_steps"):
        ds_cfg["gradient_accumulation_steps"] = int(training_args.gradient_accumulation_steps)
    if hasattr(training_args, "fp16") and training_args.fp16:
        ds_cfg.setdefault("fp16", {"enabled": True})
    elif getattr(training_args, "bf16", False):
        ds_cfg.setdefault("bf16", {"enabled": True})
    return ds_cfg


def create_engine(model,
                  ds_config,
                  training_args=None,
                  optimizer=None):
    """
    Wrap model with DeepSpeed. If your JSON defines optimizer/scheduler,
    pass only model_parameters so DS builds them. Otherwise supply 'optimizer'.
    """
    ds_cfg = load_ds_config(ds_config)
    if training_args is not None:
        ds_cfg = patch_runtime_from_training_args(ds_cfg, training_args)

    # Train only LoRA params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=trainable_params,
        optimizer=optimizer,
        config=ds_cfg
    )
    return engine


def build_dataloader(dataset,
                     tokenizer: PreTrainedTokenizerBase,
                     per_device_bs: int,
                     shuffle: bool = True,
                     num_workers: int = 2,
                     pin_memory: bool = True):
    """
    dataset: HF Dataset or torch Dataset containing 'input_ids' (+ optional 'attention_mask').
    """
    sampler = None
    if get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dl = DataLoader(
        dataset,
        batch_size=per_device_bs,
        shuffle=(sampler is None) and shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator
    )
    return dl, sampler


@contextmanager
def zero3_gathered_params(model):
    """
    Make save_pretrained work under ZeRO-3 when LoRA params may be partitioned.
    """
    if not deepspeed.zero.is_enabled():
        yield
        return

    params = [p for p in model.parameters() if p.requires_grad]
    from deepspeed.runtime.zero.partition_parameters import GatheredParameters
    with GatheredParameters(params, modifier_rank=0):
        yield
