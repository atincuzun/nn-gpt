from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn as nn

from .universal import GateInstall, gate_trainable_parameters


@dataclass(frozen=True)
class GateTrainingResult:
    steps: int
    mean_train_loss: float
    validation_loss: Optional[float]


def gate_parameters(installs: Iterable[GateInstall]) -> list[nn.Parameter]:
    return gate_trainable_parameters(installs)


def create_gate_optimizer(
    installs: Iterable[GateInstall],
    *,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    parameters = gate_parameters(installs)
    if not parameters:
        raise ValueError("No trainable replacement-gate parameters were found")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def model_input_device(model: nn.Module) -> torch.device:
    try:
        embeddings = model.get_input_embeddings()
        return next(embeddings.parameters()).device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    return batch


def _forward_loss(model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    output = model(**batch)
    loss = getattr(output, "loss", None)
    if loss is None:
        raise ValueError("Model output has no loss; provide labels in each training batch")
    return loss


def train_gate_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    parameters: Optional[Iterable[nn.Parameter]] = None,
    grad_clip: Optional[float] = 1.0,
    auxiliary_loss_fn: Optional[Callable[[nn.Module], torch.Tensor]] = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = move_batch_to_device(batch, model_input_device(model))
    loss = _forward_loss(model, batch)
    if auxiliary_loss_fn is not None:
        auxiliary_loss = auxiliary_loss_fn(model)
        loss = loss + auxiliary_loss.to(loss.device)
    loss.backward()
    trainable = list(parameters) if parameters is not None else [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(trainable, grad_clip)
    optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def evaluate_language_model_loss(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    max_steps: Optional[int] = None,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    device = model_input_device(model)
    for step, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        loss = _forward_loss(model, move_batch_to_device(batch, device))
        losses.append(float(loss.item()))
    model.train(was_training)
    if not losses:
        raise ValueError("Validation dataloader produced no batches")
    return sum(losses) / len(losses)


def train_gates(
    model: nn.Module,
    installs: Iterable[GateInstall],
    dataloader: Iterable[Mapping[str, Any]],
    *,
    steps: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    grad_clip: Optional[float] = 1.0,
    validation_loader: Optional[Iterable[Mapping[str, Any]]] = None,
    validation_steps: Optional[int] = None,
    auxiliary_loss_fn: Optional[Callable[[nn.Module], torch.Tensor]] = None,
    on_step: Optional[Callable[[int, float], None]] = None,
) -> tuple[GateTrainingResult, torch.optim.Optimizer]:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    installs = list(installs)
    parameters = gate_parameters(installs)
    optimizer = optimizer or create_gate_optimizer(
        installs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    iterator = iter(dataloader)
    losses: list[float] = []
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise ValueError("Training dataloader produced no batches") from exc
        loss = train_gate_step(
            model,
            batch,
            optimizer,
            parameters=parameters,
            grad_clip=grad_clip,
            auxiliary_loss_fn=auxiliary_loss_fn,
        )
        losses.append(loss)
        if on_step is not None:
            on_step(step, loss)
    validation_loss = None
    if validation_loader is not None:
        validation_loss = evaluate_language_model_loss(
            model,
            validation_loader,
            max_steps=validation_steps,
        )
    result = GateTrainingResult(
        steps=steps,
        mean_train_loss=sum(losses) / len(losses),
        validation_loss=validation_loss,
    )
    return result, optimizer
