from typing import Optional, Dict, Tuple, Callable, Any
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def compute_cv_reward_simple(
    *,
    built_ok: bool,
    forward_ok: bool,
    trained_step_ok: bool,
    val_metric: Optional[float],
    val_metric_baseline: Optional[float] = None,
    latency_ms: Optional[float] = None,
    params_m: Optional[float] = None,
    flops_g: Optional[float] = None,
    critic_score: Optional[float] = None,
    kl_div: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Simple scalar reward for CV model generation.
    Returns a dict with 'reward' and component terms.
    """
    w = {
        "build": 0.2,         # bonus if model builds
        "forward": 0.2,       # bonus if forward works
        "trainstep": 0.2,     # bonus if 1 mini-batch trains
        "metric_gain": 1.0,   # (val - baseline)
        "eff_latency": 0.0,   # optional: -latency_ms * coeff
        "eff_params":  0.0,   # optional: -params_m  * coeff
        "eff_flops":   0.0,   # optional: -flops_g   * coeff
        "critic": 0.5,        # optional: critic score in [0,1]
        "kl": 0.02,           # optional: KL penalty coeff
        "clip_lo": -2.0,
        "clip_hi":  2.0,
    }
    if weights:
        w.update(weights)

    r_build    = w["build"]   if built_ok else 0.0
    r_forward  = w["forward"] if forward_ok else 0.0
    r_train    = w["trainstep"] if trained_step_ok else 0.0

    r_metric = 0.0
    if (val_metric is not None) and (val_metric_baseline is not None):
        r_metric = w["metric_gain"] * (val_metric - val_metric_baseline)

    r_eff = 0.0
    if latency_ms is not None and w["eff_latency"] != 0.0:
        r_eff += - w["eff_latency"] * float(latency_ms)
    if params_m is not None and w["eff_params"] != 0.0:
        r_eff += - w["eff_params"]  * float(params_m)
    if flops_g is not None and w["eff_flops"] != 0.0:
        r_eff += - w["eff_flops"]   * float(flops_g)

    r_critic = 0.0
    if critic_score is not None:
        critic_score = max(0.0, min(1.0, float(critic_score)))
        r_critic = w["critic"] * critic_score

    r_kl = 0.0
    if kl_div is not None and kl_div > 0.0:
        r_kl = - w["kl"] * float(kl_div)

    reward = r_build + r_forward + r_train + r_metric + r_eff + r_critic + r_kl
    reward = max(w["clip_lo"], min(w["clip_hi"], reward))

    return {
        "reward": reward,
        "r_build": r_build,
        "r_forward": r_forward,
        "r_trainstep": r_train,
        "r_metric": r_metric,
        "r_eff": r_eff,
        "r_critic": r_critic,
        "r_kl": r_kl,
    }

def _count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def _quick_forward(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: str = "cpu"
) -> Tuple[bool, Optional[float]]:
    """
    Try a single forward pass. Returns (ok, latency_ms).
    """
    model.eval()
    x = torch.randn(*input_shape, device=device)
    try:
        t0 = time.time()
        y = model(x)
        torch.cuda.synchronize() if device.startswith("cuda") and torch.cuda.is_available() else None
        dt = (time.time() - t0) * 1000.0
        _ = tuple(y.shape) if hasattr(y, "shape") else None  
        return True, dt
    except Exception:
        return False, None


def _toy_loader(
    n: int = 128,
    input_shape: Tuple[int, int, int, int] = (2, 3, 32, 32),
    n_classes: int = 10,
    device: str = "cpu",
    batch_size: int = 16
) -> DataLoader:
    """
    Fallback DataLoader with random data for quick sanity train/val.
    """
    N = max(n, batch_size)
    x = torch.randn((N,) + input_shape[1:], device=device)
    y = torch.randint(0, n_classes, (N,), device=device)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def _train_steps(
    model: nn.Module,
    train_loader: DataLoader,
    steps: int = 1,
    device: str = "cpu"
) -> bool:
    """
    Mini-batch training steps. Returns ok flag.
    """
    model.train()
    try:
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        k = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                return False
            loss.backward()
            opt.step()
            k += 1
            if k >= steps:
                break
        return True
    except Exception:
        return False


@torch.no_grad()
def _quick_validate_acc(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
    max_batches: int = 10
) -> float:
    """
    Evaluation. Returns accuracy in [0,1].
    """
    try:
        model.eval()
        correct, total, bs = 0, 0, 0
        for x, y in val_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)                   
            if logits.dim() != 2:
                raise RuntimeError(f"logits must be (N,C), got {tuple(logits.shape)}")
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            bs += 1
            if bs >= max_batches:
                break
        return (correct / total) if total > 0 else 0.0
    except Exception:
        return 0.0


# ------------------------------------
# Public API: evaluate & produce reward
# ------------------------------------

@dataclass
class EvalConfig:
    device: str = "cpu"
    input_shape: Tuple[int, int, int, int] = (2, 3, 32, 32)  # (B,C,H,W)
    n_classes: int = 10
    train_steps: int = 1
    max_val_batches: int = 10
    # Optional efficiency logging
    measure_latency: bool = True
    # Optional PPO/critic
    kl_div: Optional[float] = None
    critic_fn: Optional[Callable[[nn.Module, Dict[str, Any]], float]] = None
    # Optional reward weights
    weights: Optional[Dict[str, float]] = None


def evaluate_and_reward(
    *,
    # Either pass a ready model OR a builder that returns nn.Module
    model: Optional[nn.Module] = None,
    build_fn: Optional[Callable[[], nn.Module]] = None,

    # Optional loaders; if None, random toy loaders will be used
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,

    # Baseline metric for delta reward (e.g., last round or best-so-far)
    val_metric_baseline: Optional[float] = None,

    # Config & weights
    cfg: EvalConfig = EvalConfig(),
) -> Dict[str, Any]:
    """
    End-to-end:
      1) Build model (if builder provided)
      2) Forward sanity
      3) 1..K train steps
      4) Quick validation (accuracy)
      5) (optional) critic score / KL
      6) Compute scalar reward
    Returns:
      {
        'reward': float,
        'components': {...},
        'val_metric': float or None,
        'built_ok': bool,
        'forward_ok': bool,
        'trained_step_ok': bool,
        'latency_ms': float or None,
        'params_m': float,
    }
    """
    device = cfg.device
    built_ok = False
    forward_ok = False
    trained_step_ok = False
    latency_ms = None
    params_m = None
    val_metric = None
    critic_score = None

    # 1) Build or use provided model
    mdl = model
    if mdl is None and build_fn is not None:
        try:
            mdl = build_fn()
        except Exception:
            mdl = None
    if mdl is not None and isinstance(mdl, nn.Module):
        try:
            mdl.to(device)
            params_m = _count_params_m(mdl)
            built_ok = True
        except Exception:
            built_ok = False
            mdl = None

    # If build failed, compute minimal reward and return
    if not built_ok or mdl is None:
        components = compute_cv_reward_simple(
            built_ok=False,
            forward_ok=False,
            trained_step_ok=False,
            val_metric=None,
            val_metric_baseline=val_metric_baseline,
            latency_ms=None,
            params_m=None,
            flops_g=None,
            critic_score=None,
            kl_div=cfg.kl_div,
            weights=cfg.weights
        )
        return {
            "reward": components["reward"],
            "components": components,
            "val_metric": None,
            "built_ok": False,
            "forward_ok": False,
            "trained_step_ok": False,
            "latency_ms": None,
            "params_m": None,
        }

    # 2) Forward sanity (and optional latency)
    if cfg.measure_latency:
        forward_ok, latency_ms = _quick_forward(mdl, cfg.input_shape, device=device)
    else:
        forward_ok, _ = _quick_forward(mdl, cfg.input_shape, device=device)

    # 3) Mini training steps
    if train_loader is None:
        train_loader = _toy_loader(
            n=128,
            input_shape=cfg.input_shape,
            n_classes=cfg.n_classes,
            device=device,
            batch_size=max(2, cfg.input_shape[0])
        )
    trained_step_ok = _train_steps(mdl, train_loader, steps=cfg.train_steps, device=device)

    # 4) Quick validation (accuracy)
    if val_loader is None:
        val_loader = _toy_loader(
            n=128,
            input_shape=cfg.input_shape,
            n_classes=cfg.n_classes,
            device=device,
            batch_size=max(2, cfg.input_shape[0])
        )
    val_metric = _quick_validate_acc(mdl, val_loader, device=device, max_batches=cfg.max_val_batches)

    # 5) Optional critic score
    if cfg.critic_fn is not None:
        try:
            critic_score = float(cfg.critic_fn(mdl, {
                "built_ok": built_ok,
                "forward_ok": forward_ok,
                "trained_step_ok": trained_step_ok,
                "val_metric": val_metric,
                "params_m": params_m,
                "latency_ms": latency_ms
            }))
            critic_score = max(0.0, min(1.0, critic_score))
        except Exception:
            critic_score = None

    # 6) Compute reward
    components = compute_cv_reward_simple(
        built_ok=built_ok,
        forward_ok=forward_ok,
        trained_step_ok=trained_step_ok,
        val_metric=val_metric,
        val_metric_baseline=val_metric_baseline,
        latency_ms=latency_ms,
        params_m=params_m,
        flops_g=None,                 # flops left as None (optional to integrate)
        critic_score=critic_score,
        kl_div=cfg.kl_div,
        weights=cfg.weights
    )

    return {
        "reward": components["reward"],
        "components": components,
        "val_metric": val_metric,
        "built_ok": built_ok,
        "forward_ok": forward_ok,
        "trained_step_ok": trained_step_ok,
        "latency_ms": latency_ms,
        "params_m": params_m,
    }


def _safe_exec_code(code: str) -> Dict[str, Any]:
    """
    Execute user-provided model code with torch/nn in scope and return the namespace.
    Expect class Net(in_shape, out_shape, prm, device) to exist.
    """
    ns: Dict[str, Any] = {"__builtins__": __builtins__, "torch": torch, "nn": nn}
    exec(code, ns, ns)
    return ns


def build_fn_from_code(
    code: str,
    in_shape: Tuple[int, int, int, int],
    out_shape: Tuple[int, ...],
    prm: Dict[str, Any],
    device_str: str = "cpu",
) -> Callable[[], nn.Module]:
    """
    Returns a builder function that instantiates Net(in_shape, out_shape, prm, device).
    """
    ns = _safe_exec_code(code)
    Net = ns.get("Net", None)
    if not isinstance(Net, type):
        raise RuntimeError("Net class not found or invalid in provided code.")
    device = torch.device(device_str)

    def _builder() -> nn.Module:
        return Net(in_shape, out_shape, prm, device)

    return _builder


def evaluate_code_and_reward(
    code: str,
    *,
    in_shape: Tuple[int, int, int, int] = (8, 3, 32, 32),
    out_shape: Tuple[int, ...] = (10,),
    prm: Dict[str, Any] = None,
    device: str = "cpu",
    val_metric_baseline: Optional[float] = None,
    cfg: Optional[EvalConfig] = None,
) -> Dict[str, Any]:
    """
    High-level entry: take code string, build Net, and run evaluate_and_reward to get a scalar reward.
    """
    if prm is None:
        prm = {"lr": 1e-2, "momentum": 0.9}
    defaults = {"lr": 1e-2, "momentum": 0.9, "batch": 32, "epoch": 1, "transform": None}
    prm = {**defaults, **prm} 
    if cfg is None:
        # make sure n_classes matches out_shape[0], and B,C,H,W from in_shape
        cfg = EvalConfig(
            device=device,
            input_shape=in_shape,
            n_classes=int(out_shape[0]),
            train_steps=1,
            max_val_batches=10,
            measure_latency=True,
            kl_div=None,
            critic_fn=None,
            weights=None,
        )

    try:
        builder = build_fn_from_code(code, in_shape, out_shape, prm, device)
    except Exception as e:
        print("Building function from code failed.", e)
        return {
            "reward": 0.0,
            "components": {"reward": 0.0, "r_build": 0.0, "r_forward": 0.0,
                           "r_trainstep": 0.0, "r_metric": 0.0, "r_eff": 0.0,
                           "r_critic": 0.0, "r_kl": 0.0},
            "val_metric": None,
            "built_ok": False, "forward_ok": False, "trained_step_ok": False,
            "latency_ms": None, "params_m": None,
        }
    try: 
        res = evaluate_and_reward(
            build_fn=builder,
            train_loader=None,
            val_loader=None,
            val_metric_baseline=val_metric_baseline,
            cfg=cfg,
        )
        if res["reward"] == 0.0:
            res["reward"] = -1.0
        return res
    except Exception as e:
        print("Evaluation failed for provided code. Exception:", e)
        return {
            "reward": 0.0,
            "components": {"reward": 0.0, "r_build": 0.0, "r_forward": 0.0,
                           "r_trainstep": 0.0, "r_metric": 0.0, "r_eff": 0.0,
                           "r_critic": 0.0, "r_kl": 0.0},
            "val_metric": None,
            "built_ok": False, "forward_ok": False, "trained_step_ok": False,
            "latency_ms": None, "params_m": None,
        }

if __name__ == "__main__":
    demo_code = r'''
import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        
                           
        self.conv1 = nn.Conv2d(in_shape[0], prm['conv1_filters'], kernel_size=prm['conv1_kernel'], stride=prm['conv1_stride'])
        self.bn1 = nn.BatchNorm2d(prm['conv1_filters']) if prm['use_batchnorm'] else None
        self.conv2 = nn.Conv2d(prm['conv1_filters'], prm['conv2_filters'], kernel_size=prm['conv2_kernel'])
        self.bn2 = nn.BatchNorm2d(prm['conv2_filters']) if prm['use_batchnorm'] else None
        self.conv3 = nn.Conv2d(prm['conv2_filters'], prm['conv3_filters'], kernel_size=prm['conv3_kernel'])
        self.bn3 = nn.BatchNorm2d(prm['conv3_filters']) if prm['use_batchnorm'] else None
        self.conv4 = nn.Conv2d(prm['conv3_filters'], prm['conv4_filters'], kernel_size=prm['conv4_kernel'])
        self.bn4 = nn.BatchNorm2d(prm['conv4_filters']) if prm['use_batchnorm'] else None
        self.conv5 = nn.Conv2d(prm['conv4_filters'], prm['conv5_filters'], kernel_size=prm['conv5_kernel'])
        self.bn5 = nn.BatchNorm2d(prm['conv5_filters']) if prm['use_batchnorm'] else None
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(prm['fc1_neurons'], prm['fc2_neurons'])
        self.fc2 = nn.Linear(prm['fc2_neurons'], out_shape)
        
                       
        self.dropout = nn.Dropout(p=prm['dropout'])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x'''
    res = evaluate_code_and_reward(
        demo_code,
        in_shape=(1, 3, 224, 224),
        out_shape=(10,),
        prm={"lr": 1e-2, "momentum": 0.9, "dropout": 0.5},
        device="cpu",
        val_metric_baseline=0.10,
    )
    print(res)