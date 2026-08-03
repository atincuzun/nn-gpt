from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _fit_completion_to_context(
    tokenized: dict[str, list[int] | int],
    context_length: int,
) -> dict[str, list[int] | int | bool]:
    input_ids = list(tokenized["input_ids"])
    attention_mask = list(tokenized["attention_mask"])
    labels = list(tokenized["labels"])
    completion_start = next(
        (index for index, label in enumerate(labels) if label != -100),
        len(labels),
    )
    completion_length = len(labels) - completion_start
    if completion_length > context_length:
        return {
            **tokenized,
            "fits_context": False,
            "truncated_prompt_tokens": 0,
        }
    if len(input_ids) <= context_length:
        return {
            **tokenized,
            "fits_context": True,
            "truncated_prompt_tokens": 0,
        }

    available_prompt = context_length - completion_length
    prefix_ids = input_ids[:completion_start]
    prefix_mask = attention_mask[:completion_start]
    head_length = (available_prompt + 1) // 2
    tail_length = available_prompt - head_length
    kept_prefix_ids = prefix_ids[:head_length]
    kept_prefix_mask = prefix_mask[:head_length]
    if tail_length:
        kept_prefix_ids += prefix_ids[-tail_length:]
        kept_prefix_mask += prefix_mask[-tail_length:]
    return {
        **tokenized,
        "input_ids": kept_prefix_ids + input_ids[completion_start:],
        "attention_mask": kept_prefix_mask + attention_mask[completion_start:],
        "labels": [-100] * available_prompt + labels[completion_start:],
        "fits_context": True,
        "truncated_prompt_tokens": completion_start - available_prompt,
    }


def _tokenize_completion(
    example: dict[str, str],
    tokenizer: Any,
    context_length: int,
) -> dict[str, list[int] | int | bool]:
    text = example["text"]
    response = example["response"]
    response_start = text.rfind(response)
    if response_start < 0:
        raise ValueError("Assistant response was not found in the rendered chat text")
    encoded = tokenizer(text, truncation=False)
    prefix = tokenizer(text[:response_start], truncation=False)
    input_ids = encoded["input_ids"]
    prompt_length = min(len(prefix["input_ids"]), len(input_ids))
    labels = [-100] * prompt_length + list(input_ids[prompt_length:])
    if all(label == -100 for label in labels):
        raise ValueError("Completion-only example contains no supervised assistant tokens")
    response_ids = tokenizer(response, truncation=False, add_special_tokens=False)["input_ids"]
    return _fit_completion_to_context({
        "input_ids": input_ids,
        "attention_mask": encoded.get("attention_mask", [1] * len(input_ids)),
        "labels": labels,
        "response_length": len(response_ids),
    }, context_length)


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
    completion_only: bool = True,
):
    """Build gate-training loaders from NNGPT's existing ``NNGenPrompt``.

    This optional adapter deliberately imports NNGPT dependencies lazily. The
    core gate package therefore remains usable without LEMUR/``ab.nn``, while
    standalone experiments can consume the exact chat-formatted NN generation
    examples used by the normal tuning pipeline.
    """
    from torch.utils.data import DataLoader
    from datasets import Dataset
    from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq

    from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")

    processor = NNGenPrompt(context_length, tokenizer, Path(prompt_config))
    if completion_only:
        raw = processor.get_raw_dataset(only_best_accuracy, max_prompts)
        dataset = Dataset.from_pandas(raw)
        dataset = dataset.map(
            lambda example: _tokenize_completion(example, tokenizer, context_length),
            remove_columns=dataset.column_names,
        )
        tokenized_examples = len(dataset)
        truncated_examples = sum(dataset["truncated_prompt_tokens"])
        oversized_responses = sum(
            not fits for fits in dataset["fits_context"]
        )
        dataset = dataset.filter(
            lambda example: example["fits_context"]
            and example["response_length"] <= max_new_tokens
        )
        dataset = dataset.remove_columns([
            "response_length",
            "fits_context",
            "truncated_prompt_tokens",
        ])
        print(
            "NNGenPrompt tokenization: "
            f"raw={tokenized_examples} kept={len(dataset)} "
            f"oversized_responses={oversized_responses} "
            f"trimmed_prompt_tokens={truncated_examples}",
            flush=True,
        )
        dataset = dataset.shuffle(seed=seed)
    else:
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

    if completion_only:
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            label_pad_token_id=-100,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
    else:
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
