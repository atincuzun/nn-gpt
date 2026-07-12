from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def build_nngenprompt_dataloaders(
    tokenizer: Any,
    prompt_config: str | Path,
    *,
    context_length: int = 4096,
    max_prompts: Optional[int] = None,
    max_new_tokens: int = 4096,
    only_best_accuracy: bool = False,
    batch_size: int = 1,
    validation_fraction: float = 0.1,
    seed: int = 42,
    pad_to_multiple_of: Optional[int] = 8,
):
    """Build gate-training loaders from NNGPT's existing ``NNGenPrompt``.

    This optional adapter deliberately imports NNGPT dependencies lazily. The
    core gate package therefore remains usable without LEMUR/``ab.nn``, while
    standalone experiments can consume the exact chat-formatted NN generation
    examples used by the normal tuning pipeline.
    """
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling

    from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")

    processor = NNGenPrompt(context_length, tokenizer, Path(prompt_config))
    dataset = processor.get_dataset(
        only_best_accuracy=only_best_accuracy,
        seed=seed,
        max_prompts=max_prompts,
        max_new_tokens=max_new_tokens,
    )
    if len(dataset) == 0:
        raise ValueError("NNGenPrompt produced no examples")

    validation_dataset = None
    if validation_fraction > 0 and len(dataset) > 1:
        split = dataset.train_test_split(test_size=validation_fraction, seed=seed)
        dataset = split["train"]
        validation_dataset = split["test"]

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )
    return train_loader, validation_loader, dataset
