from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional


@contextmanager
def _runtime_prompt_config(
    prompt_config: str | Path,
    *,
    dataset_name: Optional[str],
    nn_prefixes: Optional[tuple[str, ...]],
):
    prompt_path = Path(prompt_config)
    if dataset_name is None and nn_prefixes is None:
        yield prompt_path
        return

    prompt_dict = json.loads(prompt_path.read_text(encoding="utf-8"))
    for key_config in prompt_dict.values():
        if dataset_name is not None:
            key_config["dataset"] = dataset_name
        if nn_prefixes is not None:
            key_config["nn_prefixes"] = list(nn_prefixes)

    with TemporaryDirectory(prefix="nngpt_moe_prompt_") as temp_dir:
        runtime_path = Path(temp_dir) / prompt_path.name
        runtime_path.write_text(json.dumps(prompt_dict, indent=2), encoding="utf-8")
        yield runtime_path


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
    gate_summary: Optional[str] = None,
    dataset_name: Optional[str] = None,
    nn_prefixes: Optional[tuple[str, ...]] = None,
    **_ignored,
):
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling
    from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")

    with _runtime_prompt_config(
        prompt_config,
        dataset_name=dataset_name,
        nn_prefixes=nn_prefixes,
    ) as runtime_prompt_config:
        processor = NNGenPrompt(
            context_length,
            tokenizer,
            runtime_prompt_config,
            extra_static_values={"gate_summary": gate_summary or ""},
        )
        dataset = processor.get_dataset(
            only_best_accuracy=only_best_accuracy,
            seed=seed,
            max_prompts=max_prompts,
            max_new_tokens=max_new_tokens,
        )

    if len(dataset) == 0:
        raise ValueError(
            "NNGenPrompt produced no usable examples after token-length filtering; "
            f"context_length={context_length}, max_new_tokens={max_new_tokens}"
        )

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
