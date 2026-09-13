from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional


def _fit_completion_to_context(
    tokenized: dict[str, Any],
    context_length: int,
    *,
    protected_prefix_tokens: int = 2,
) -> dict[str, Any]:
    """Trim prompt tokens while preserving the labelled assistant completion."""
    result = dict(tokenized)
    input_ids = list(result["input_ids"])
    attention_mask = list(result["attention_mask"])
    labels = list(result["labels"])
    response_length = int(result.get("response_length", 0))
    if response_length > context_length:
        result["fits_context"] = False
        result["truncated_prompt_tokens"] = 0
        return result

    overflow = max(0, len(input_ids) - context_length)
    prompt_length = len(input_ids) - response_length
    protected = min(protected_prefix_tokens, prompt_length)
    removable = max(0, prompt_length - protected)
    if overflow > removable:
        result["fits_context"] = False
        result["truncated_prompt_tokens"] = 0
        return result
    if overflow:
        keep = list(range(protected)) + list(range(protected + overflow, len(input_ids)))
        input_ids = [input_ids[index] for index in keep]
        attention_mask = [attention_mask[index] for index in keep]
        labels = [labels[index] for index in keep]
    result.update({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "fits_context": True,
        "truncated_prompt_tokens": overflow,
    })
    return result


def _completion_only_dataset(processor: Any, *, context_length: int, max_prompts: Optional[int],
                             max_new_tokens: int, only_best_accuracy: bool, seed: int):
    """Tokenize NNGenPrompt rows and mask system/user tokens from the LM loss."""
    from datasets import Dataset

    raw = processor.get_raw_dataset(only_best_accuracy, max_prompts)
    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        # NNGenPrompt already rendered the exact configured system/user/assistant
        # conversation. Split that text at the final assistant response instead
        # of reconstructing it and accidentally dropping the system prompt.
        full_text = str(row["text"])
        response_text = str(row["response"])
        response_start = full_text.rfind(response_text)
        if response_start < 0:
            continue
        prompt_text = full_text[:response_start]
        # These temporary tokenizations can exceed the model's advertised
        # context because the completion must be measured before the prompt is
        # trimmed below.  The overlength sequence is never sent to the model;
        # suppress the tokenizers warning that otherwise suggests it is.
        prompt_ids = processor.tokenizer(
            prompt_text,
            add_special_tokens=True,
            verbose=False,
        )["input_ids"]
        encoded = processor.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        response_length = len(input_ids) - prompt_length
        if response_length <= 0 or response_length >= max_new_tokens:
            continue
        fitted = _fit_completion_to_context({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "response_length": response_length,
        }, context_length)
        if fitted["fits_context"]:
            fitted.pop("fits_context", None)
            fitted.pop("truncated_prompt_tokens", None)
            fitted.pop("response_length", None)
            rows.append(fitted)
    dataset = Dataset.from_list(rows)
    return dataset.shuffle(seed=seed) if len(dataset) else dataset


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
    from transformers import DataCollatorForTokenClassification
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
        dataset = _completion_only_dataset(
            processor,
            context_length=context_length,
            max_prompts=max_prompts,
            max_new_tokens=max_new_tokens,
            only_best_accuracy=only_best_accuracy,
            seed=seed,
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

    # Preserve the precomputed -100 prompt mask; an LM collator would replace it.
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
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
