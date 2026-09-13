"""Persistent LoRA proposer, attached only while learning/writing gate code."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from .gate_benchmark import write_json
from .gate_prompt import gate_sft_examples


def tokenize_gate_examples(tokenizer, examples: list[dict], max_length: int) -> list[dict]:
    """Label only complete gate-code responses; never truncate gate source."""
    rows = []
    for example in examples:
        messages = example["messages"]
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        input_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        # Some chat templates insert reasoning markers only during generation.
        # Find the actual response boundary rather than guessing a token offset.
        response_start = full.rfind(messages[-1]["content"])
        if response_start < 0:
            raise ValueError("Chat template omitted the gate-code target")
        if not full.startswith(prefix):
            prefix_ids = tokenizer(full[:response_start], add_special_tokens=False)["input_ids"]
        boundary = 0
        for left, right in zip(prefix_ids, input_ids):
            if left != right:
                break
            boundary += 1
        if len(input_ids) > max_length or boundary >= len(input_ids) or boundary < 1:
            continue
        rows.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
                     "labels": [-100] * boundary + input_ids[boundary:]})
    return rows


@contextmanager
def proposal_adapter(base_model, args, checkpoint: Path | None):
    """Never merge proposer weights; restore base trainability/config on exit."""
    from peft import LoraConfig, PeftModel, get_peft_model

    if isinstance(base_model, PeftModel):
        raise ValueError("Outer proposer expects an unwrapped frozen base model")
    original_parameters = [(p, p.requires_grad) for p in base_model.parameters()]
    original_training = base_model.training
    original_cache = base_model.config.use_cache
    original_checkpointing = getattr(base_model, "is_gradient_checkpointing", False)
    wrapped = None
    try:
        if checkpoint is not None:
            wrapped = PeftModel.from_pretrained(base_model, str(checkpoint), is_trainable=True)
        else:
            targets = [name for name, _ in base_model.named_modules()
                       if name.split(".")[-1] in args.gate_proposer_targets
                       and not any(part in name.lower() for part in ("expert", "gate", "router"))]
            if not targets:
                raise ValueError("No attention projections matched --gate-proposer-targets")
            wrapped = get_peft_model(base_model, LoraConfig(
                r=args.gate_proposer_rank, lora_alpha=2 * args.gate_proposer_rank,
                lora_dropout=0.0, bias="none", target_modules=targets, task_type="CAUSAL_LM",
            ))
        # No prepare_model_for_kbit_training: it permanently casts the base.
        # Frozen quantized weights already work in the inner gate-training path.
        for name, parameter in wrapped.named_parameters():
            parameter.requires_grad_("lora_" in name)
        yield wrapped
    finally:
        if wrapped is not None:
            wrapped.unload()  # Remove adapters WITHOUT modifying base weights.
        for parameter, trainable in original_parameters:
            parameter.requires_grad_(trainable)
        base_model.config.use_cache = original_cache
        if original_checkpointing:
            base_model.gradient_checkpointing_enable()
        elif hasattr(base_model, "gradient_checkpointing_disable"):
            base_model.gradient_checkpointing_disable()
        base_model.train(original_training)


def train_proposer(model, tokenizer, rows: list[dict], args, output: Path,
                   previous: Path | None) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForTokenClassification
    from moe_gate_only.training import model_input_device, move_batch_to_device

    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Proposer has no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.gate_proposer_learning_rate)
    if previous is not None and (previous / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(previous / "optimizer.pt", map_location="cpu", weights_only=True))
        for group in optimizer.param_groups:
            group["lr"] = args.gate_proposer_learning_rate
    loader = DataLoader(rows, batch_size=args.gate_proposer_batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=DataCollatorForTokenClassification(
                            tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8))
    model.train()
    model.config.use_cache = False
    hook = None
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        def require_embedding_grad(_module, _inputs, output):
            output.requires_grad_(True)
        hook = model.get_input_embeddings().register_forward_hook(require_embedding_grad)
    losses = []
    iterator = iter(loader)
    try:
        for step in range(args.gate_proposer_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**move_batch_to_device(batch, model_input_device(model))).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite gate proposer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            print(f"[GATE PROPOSER] step={step + 1} loss={losses[-1]:.6f}")
        output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output)
        torch.save(optimizer.state_dict(), output / "optimizer.pt")
    finally:
        if hook is not None:
            hook.remove()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()
    return {"steps": len(losses), "examples": len(rows), "mean_loss": sum(losses) / len(losses)}


class GateProposer:
    def __init__(self, args, root: Path, protocol_id: str):
        self.args = args
        self.root = root / "gate_proposer"
        self.protocol_id = protocol_id
        self.checkpoint = args.gate_proposer_checkpoint
        if self.checkpoint is not None:
            metadata = json.loads((self.checkpoint / "proposer_state.json").read_text())
            if metadata["protocol_id"] != protocol_id:
                raise ValueError("Proposer checkpoint belongs to a different gate benchmark protocol")

    @contextmanager
    def for_round(self, ctx, store, index: int):
        from ab.gpt.util.Chatbot import ChatBot
        from .morphism import _seed_all

        args = self.args
        selected = store.training_subset(
            self.protocol_id, max_examples=args.gate_proposer_max_examples,
            top_fraction=args.gate_proposer_top_fraction, seed=args.seed + index,
        )
        examples = gate_sft_examples(selected, ctx.shapes)
        rows = tokenize_gate_examples(ctx.tokenizer, examples, args.gate_proposer_max_length)
        output = self.root / f"round_{index:03d}"
        write_json(output / "training_examples.json", examples)
        can_train = len(rows) >= args.gate_proposer_min_examples
        if not can_train and self.checkpoint is None:
            write_json(output / "training_metrics.json", {"training_skipped": True,
                       "reason": "Not enough eligible, complete gate examples; bootstrap with native proposer",
                       "examples": len(rows)})
            _seed_all(args.seed + index)
            yield ChatBot(ctx.session.model, ctx.tokenizer, temperature=args.temperature,
                          top_k=args.top_k, top_p=args.top_p)
            return
        _seed_all(args.seed + index)
        with proposal_adapter(ctx.session.model, args, self.checkpoint) as model:
            if can_train:
                checkpoint = output / "adapter"
                metrics = train_proposer(model, ctx.tokenizer, rows, args, checkpoint, self.checkpoint)
                write_json(checkpoint / "proposer_state.json", {
                    "protocol_id": self.protocol_id, "round": index,
                    "parent_checkpoint": str(self.checkpoint) if self.checkpoint else None,
                    "source_trial_ids": [r["trial_id"] for r in selected],
                })
                self.checkpoint = checkpoint
                write_json(self.root / "latest.json", {"checkpoint": str(checkpoint.resolve()),
                                                       "protocol_id": self.protocol_id})
            else:
                metrics = {"training_skipped": True, "reason": "Not enough fitting gate examples"}
            write_json(output / "training_metrics.json", metrics)
            model.eval()
            model.config.use_cache = True
            yield ChatBot(model, ctx.tokenizer, temperature=args.temperature,
                          top_k=args.top_k, top_p=args.top_p)
