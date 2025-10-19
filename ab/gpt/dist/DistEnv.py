# ab/gpt/dist/DistEnv.py
import os
import time
import inspect
from datetime import timedelta
from typing import Callable, Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _spawn_entry(local_rank: int, entry_fn: Callable[..., Any], args: Tuple[any, ...], nprocs: int):
    # Local single-node fallback spawn (we avoid this under torchrun/deepspeed).
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(nprocs)
    entry_fn(local_rank, *args)


class DistEnv:
    """
    Safe distributed bootstrap that:
      - DOES NOT respawn under torchrun/deepspeed (env already set)
      - sets CUDA device *before* init_process_group (avoids NCCL hangs)
      - supports `seed` kwarg
      - passes `device_id`/`device_ids` to init/barrier when supported (PyTorch 2.7+)
    """
    _inited = False
    _rank = 0
    _local_rank = 0
    _world_size = 1

    # -------- Launch helpers (no-op under torchrun) --------
    @staticmethod
    def spawn_if_needed(entry_fn: Callable[..., Any], *args, nprocs: Optional[int] = None) -> None:
        if any(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            entry_fn(local_rank, *args)
            return
        if nprocs is None:
            nprocs = torch.cuda.device_count() or 1
        mp.spawn(_spawn_entry, nprocs=nprocs, args=(entry_fn, args, nprocs), join=True, daemon=False)

    # Back-compat name (older code may call this)
    autospawn_if_needed = spawn_if_needed

    # -------- Core init/destroy --------
    @staticmethod
    def init(backend: str = "nccl", timeout_s: int = 1800, seed: Optional[int] = None) -> None:
        if DistEnv._inited:
            return

        DistEnv._rank = int(os.environ.get("RANK", "0"))
        DistEnv._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        DistEnv._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        # Bind CUDA device EARLY
        if torch.cuda.is_available():
            torch.cuda.set_device(DistEnv._local_rank)

        # Initialize PG if multi-proc and not already done
        if DistEnv._world_size > 1 and not dist.is_initialized():
            kwargs = dict(
                backend=backend,
                init_method="env://",
                timeout=timedelta(seconds=timeout_s),
                rank=DistEnv._rank,
                world_size=DistEnv._world_size,
            )
            try:
                # PyTorch 2.7+ supports device_id in init_process_group
                if "device_id" in inspect.signature(dist.init_process_group).parameters:
                    kwargs["device_id"] = DistEnv._local_rank
                dist.init_process_group(**kwargs)
            except TypeError:
                # Older PyTorch w/o device_id
                kwargs.pop("device_id", None)
                dist.init_process_group(**kwargs)

        # Seed: if None, rank-0 picks and broadcasts
        if seed is None:
            base = int(time.time()) & 0x7FFFFFFF
            if dist.is_initialized():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                t = torch.tensor([base], dtype=torch.int64, device=device)
                dist.broadcast(t, src=0)
                seed = int(t.item())
            else:
                seed = base

        DistEnv.set_seed(seed)
        DistEnv._inited = True

    @staticmethod
    def destroy() -> None:
        if dist.is_initialized():
            try:
                DistEnv.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
        DistEnv._inited = False

    @staticmethod
    def barrier() -> None:
        if dist.is_initialized():
            # PyTorch 2.7+: barrier(device_ids=[...]) to avoid mapping warning
            kwargs = {}
            try:
                if "device_ids" in inspect.signature(dist.barrier).parameters and torch.cuda.is_available():
                    kwargs["device_ids"] = [DistEnv._local_rank]
            except Exception:
                pass
            dist.barrier(**kwargs)

    # -------- Utilities --------
    @staticmethod
    def set_seed(seed: int) -> None:
        rs = int(seed) + DistEnv.rank()
        torch.manual_seed(rs)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rs)

    @staticmethod
    def rank() -> int:
        return DistEnv._rank

    @staticmethod
    def local_rank() -> int:
        return DistEnv._local_rank

    @staticmethod
    def world_size() -> int:
        return DistEnv._world_size

    @staticmethod
    def is_main() -> bool:
        return DistEnv.rank() == 0