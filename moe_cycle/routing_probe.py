"""Routing and output diagnostics measured against the untouched model.

A replacement gate can only change what the LLM generates if it routes tokens
to different experts. ``RoutingProbe`` makes that observable. It is captured
once from the native model -- before any gate is installed or adapter is
injected -- on a fixed set of probe sequences, and every later snapshot is
compared with it:

* per MoE layer, the fraction of probe tokens whose top-k expert set changed
  (``flip_any``) and the mean overlap of the top-k sets (``overlap``);
* the next-token distribution: an estimate of KL(native || current) on the
  native top-64 support, plus top-1 agreement;
* the teacher-forced NLL of the probe completions.

Native sites are hooked where the router lives (``site.block.<gate_attr>``);
replaced sites are hooked on the installed module, whose contract output is
the same ``(logits, weights, indices)`` the native router returns.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch

#: Native next-token support kept for the KL estimate.
TOPK_SUPPORT = 64


def _router_module(site: Any, installs_by_layer: dict[int, Any]) -> Any:
    install = installs_by_layer.get(site.layer_index)
    if install is not None and install.owner is not None and install.attr is not None:
        return getattr(install.owner, install.attr)
    return getattr(site.block, site.gate_attr)


def _selected_experts(output: Any, top_k: int) -> torch.Tensor | None:
    """The ``[tokens, k]`` expert indices a router call selected."""
    tensors = output if isinstance(output, (tuple, list)) else (output,)
    integer = [
        value for value in tensors
        if isinstance(value, torch.Tensor)
        and not value.is_floating_point()
        and value.dtype != torch.bool
        and value.ndim >= 1
    ]
    if integer:
        indices = integer[-1]
        return indices.reshape(-1, indices.shape[-1])
    floating = [
        value for value in tensors
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ]
    if not floating:
        return None
    logits = floating[0]
    return logits.reshape(-1, logits.shape[-1]).topk(top_k, dim=-1).indices


class RoutingProbe:
    """Fixed probe sequences plus the native reference captured from them."""

    def __init__(self, sequences: list[dict[str, torch.Tensor]], top_k: int):
        if not sequences:
            raise ValueError("RoutingProbe needs at least one probe sequence")
        self.sequences = sequences
        self.top_k = int(top_k)
        self.reference: dict[str, Any] | None = None

    @classmethod
    def from_dataset(cls, dataset: Iterable[dict[str, Any]], *, count: int,
                     max_tokens: int, top_k: int) -> "RoutingProbe":
        """Take the last ``max_tokens`` tokens of the first ``count`` rows.

        The tail is kept because it holds the completion (the generated
        model code), which is what the LLM is trained and measured on.
        """
        sequences: list[dict[str, torch.Tensor]] = []
        for row in dataset:
            input_ids = list(row["input_ids"])[-max_tokens:]
            labels = list(row["labels"])[-max_tokens:]
            if len(input_ids) < 8:
                continue
            sequences.append({
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "labels": torch.tensor([labels], dtype=torch.long),
            })
            if len(sequences) >= count:
                break
        return cls(sequences, top_k)

    @torch.no_grad()
    def _run(self, model: Any, sites: list[Any], installs: Iterable[Any],
             reference: dict[str, Any] | None) -> dict[str, Any]:
        installs_by_layer = {install.site.layer_index: install for install in installs}
        device = model.get_input_embeddings().weight.device
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for site in sites:
            module = _router_module(site, installs_by_layer)

            def hook(_module, _inputs, output, layer=site.layer_index):
                indices = _selected_experts(output, self.top_k)
                if indices is not None:
                    captured[layer] = indices.detach().to("cpu", torch.int16)

            handles.append(module.register_forward_hook(hook))

        was_training = model.training
        use_cache = getattr(model.config, "use_cache", None)
        model.eval()
        if use_cache is not None:
            model.config.use_cache = False
        routes: list[dict[int, torch.Tensor]] = []
        support: list[tuple[torch.Tensor, torch.Tensor]] = []
        nll_sum = 0.0
        nll_tokens = 0
        kl_sum = 0.0
        top1_agree = 0
        kl_tokens = 0
        try:
            for index, sequence in enumerate(self.sequences):
                captured.clear()
                input_ids = sequence["input_ids"].to(device)
                logits = model(input_ids=input_ids, use_cache=False).logits[0].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                routes.append(dict(captured))

                labels = sequence["labels"][0].to(device)
                targets = labels[1:]
                mask = targets != -100
                if mask.any():
                    picked = log_probs[:-1][mask].gather(1, targets[mask].unsqueeze(1))
                    nll_sum += float(-picked.sum().item())
                    nll_tokens += int(mask.sum().item())

                values, indices = log_probs.topk(TOPK_SUPPORT, dim=-1)
                if reference is None:
                    support.append((values.to("cpu", torch.float32), indices.to("cpu", torch.int32)))
                else:
                    ref_values, ref_indices = reference["support"][index]
                    ref_values = ref_values.to(device, torch.float32)
                    ref_indices = ref_indices.to(device, torch.long)
                    current = log_probs.gather(1, ref_indices)
                    p = ref_values.exp()
                    rest_p = (1.0 - p.sum(dim=-1)).clamp_min(1e-8)
                    rest_q = (1.0 - current.exp().sum(dim=-1)).clamp_min(1e-8)
                    kl = (p * (ref_values - current)).sum(dim=-1)
                    kl = kl + rest_p * (rest_p.log() - rest_q.log())
                    kl_sum += float(kl.clamp_min(0).sum().item())
                    top1_agree += int((indices[:, 0] == ref_indices[:, 0]).sum().item())
                    kl_tokens += int(kl.numel())
                del logits, log_probs
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)
            if use_cache is not None:
                model.config.use_cache = use_cache

        snapshot: dict[str, Any] = {
            "routes": routes,
            "nll": nll_sum / nll_tokens if nll_tokens else None,
            "nll_tokens": nll_tokens,
        }
        if reference is None:
            snapshot["support"] = support
        else:
            snapshot["kl"] = kl_sum / kl_tokens if kl_tokens else None
            snapshot["top1_agreement"] = top1_agree / kl_tokens if kl_tokens else None
        return snapshot

    def capture_reference(self, model: Any, sites: list[Any]) -> dict[str, Any]:
        """Native routing and output distribution; call before any change."""
        self.reference = self._run(model, sites, (), None)
        return {
            "probe_sequences": len(self.sequences),
            "probe_tokens": sum(int(s["input_ids"].shape[-1]) for s in self.sequences),
            "native_nll": self.reference["nll"],
            "layers_observed": sorted(self.reference["routes"][0]) if self.reference["routes"] else [],
        }

    def compare(self, model: Any, sites: list[Any], installs: Iterable[Any],
                replaced_layers: Iterable[int] = ()) -> dict[str, Any]:
        """Routing change and output drift relative to the native reference."""
        if self.reference is None:
            raise RuntimeError("capture_reference() must run before compare()")
        current = self._run(model, sites, installs, self.reference)
        per_layer: dict[str, dict[str, float]] = {}
        for layer in sorted(self.reference["routes"][0]) if self.reference["routes"] else []:
            overlaps: list[torch.Tensor] = []
            for native_routes, current_routes in zip(self.reference["routes"], current["routes"]):
                native = native_routes.get(layer)
                now = current_routes.get(layer)
                if native is None or now is None or native.shape != now.shape:
                    continue
                matches = (native.unsqueeze(-1) == now.unsqueeze(-2)).any(dim=-1)
                overlaps.append(matches.float().mean(dim=-1))
            if not overlaps:
                continue
            overlap = torch.cat(overlaps)
            per_layer[str(layer)] = {
                "overlap": float(overlap.mean().item()),
                "flip_any": float((overlap < 1.0).float().mean().item()),
            }
        replaced = {str(layer) for layer in replaced_layers}

        def _mean(keys: Iterable[str], field: str) -> float | None:
            values = [per_layer[key][field] for key in keys if key in per_layer]
            return sum(values) / len(values) if values else None

        native_nll = self.reference["nll"]
        return {
            "flip_any_all_layers": _mean(per_layer, "flip_any"),
            "overlap_all_layers": _mean(per_layer, "overlap"),
            "flip_any_replaced_layers": _mean(replaced, "flip_any"),
            "overlap_replaced_layers": _mean(replaced, "overlap"),
            "kl_native_to_current": current["kl"],
            "top1_agreement": current["top1_agreement"],
            "probe_nll": current["nll"],
            "native_probe_nll": native_nll,
            "probe_nll_delta": (
                current["nll"] - native_nll
                if current["nll"] is not None and native_nll is not None
                else None
            ),
            "per_layer": per_layer,
        }


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
