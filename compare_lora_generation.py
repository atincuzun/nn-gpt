"""Compare base and LoRA neural-network generation on NNGen prompts."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftConfig, PeftModel

from ab.gpt.util.Util import extract_all_to_train
from ab.gpt.util.nn_sftcodegen_rag import validate_code_for_nneval
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
from moe_gate_only.training import model_input_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="Override the adapter's recorded base model")
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=Path("ab/gpt/conf/prompt/train/NN_gen.json"),
    )
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-source-prompts", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_response(response: str) -> tuple[dict[str, Any], tuple[str | None, str | None, str | None]]:
    code, hyperparameters, transform = extract_all_to_train(response)
    hp_valid = False
    hp_error = "missing <hp> block"
    if hyperparameters:
        try:
            hp_valid = isinstance(json.loads(hyperparameters), dict)
            hp_error = "" if hp_valid else "<hp> is not a JSON object"
        except json.JSONDecodeError as exc:
            hp_error = str(exc)
    code_valid, code_error = validate_code_for_nneval(code) if code else (False, "missing <nn> block")
    checks = {
        "has_hp": hyperparameters is not None,
        "has_transform": transform is not None,
        "has_nn": code is not None,
        "correct_tag_order": 0 <= response.find("<hp>") < response.find("<tr>") < response.find("<nn>"),
        "hyperparameters_valid": hp_valid,
        "hyperparameters_error": hp_error,
        "nneval_interface_valid": code_valid,
        "nneval_interface_error": code_error,
    }
    checks["compatible"] = all(
        checks[key]
        for key in (
            "has_hp",
            "has_transform",
            "has_nn",
            "correct_tag_order",
            "hyperparameters_valid",
            "nneval_interface_valid",
        )
    )
    return checks, (code, hyperparameters, transform)


def save_candidate(
    root: Path,
    arm: str,
    index: int,
    response: str,
    artifacts: tuple[str | None, str | None, str | None],
) -> None:
    code, hyperparameters, transform = artifacts
    candidate_dir = root / arm / f"{index:03d}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "response.txt").write_text(response, encoding="utf-8")
    if code:
        (candidate_dir / "new_nn.py").write_text(code.rstrip() + "\n", encoding="utf-8")
    if hyperparameters:
        (candidate_dir / "hyperparameters.json").write_text(hyperparameters.rstrip() + "\n", encoding="utf-8")
    if transform:
        (candidate_dir / "transform.py").write_text(transform.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_prompts < 1 or args.max_source_prompts < args.num_prompts:
        raise ValueError("num-prompts must be positive and no larger than max-source-prompts")
    seed_all(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    peft_config = PeftConfig.from_pretrained(args.adapter, local_files_only=args.local_files_only)
    base_model_name = args.model or peft_config.base_model_name_or_path
    tokenizer_source = args.adapter if (args.adapter / "tokenizer_config.json").exists() else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False)
    model.eval()

    prompt_config = json.loads(args.prompt_config.read_text(encoding="utf-8"))
    if len(prompt_config) != 1:
        raise ValueError("Generation comparison currently requires a prompt config with exactly one entry")
    system_prompt = "\n".join(next(iter(prompt_config.values())).get("system", []))
    processor = NNGenPrompt(args.max_length, tokenizer, args.prompt_config)
    prompts = processor.get_raw_dataset(False, args.max_source_prompts)
    prompts = prompts.sample(n=args.num_prompts, random_state=args.seed).reset_index(drop=True)

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    device = model_input_device(model)
    for index, row in prompts.iterrows():
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": row["instruction"]})
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=args.max_length)
        encoded = {key: value.to(device) for key, value in encoded.items()}

        for arm in ("base", "finetuned"):
            seed_all(args.seed + index)
            adapter_context = model.disable_adapter() if arm == "base" else nullcontext()
            with adapter_context, torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(
                generated[0, encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            checks, artifacts = validate_response(completion)
            save_candidate(args.output, arm, index, completion, artifacts)
            records.append({"prompt_index": index, "arm": arm, **checks})
            print(f"prompt={index} arm={arm} compatible={checks['compatible']}", flush=True)

    summary: dict[str, Any] = {
        "base_model": base_model_name,
        "adapter": str(args.adapter.resolve()),
        "prompt_config": str(args.prompt_config),
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "results": records,
    }
    for arm in ("base", "finetuned"):
        arm_records = [record for record in records if record["arm"] == arm]
        summary[arm] = {
            "compatible_count": sum(record["compatible"] for record in arm_records),
            "compatible_rate": sum(record["compatible"] for record in arm_records) / len(arm_records),
            "nneval_interface_valid_count": sum(record["nneval_interface_valid"] for record in arm_records),
            "valid_hyperparameters_count": sum(record["hyperparameters_valid"] for record in arm_records),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"base": summary["base"], "finetuned": summary["finetuned"]}, indent=2))


if __name__ == "__main__":
    main()
