"""Shared contract for gate-code proposal and completion-only proposer SFT."""

from contextlib import contextmanager


GATE_SYSTEM_PROMPT = (
    "You design small, trainable MoE router scorers, not CV networks. "
    "Output only a complete Python module between <gate> and </gate>. "
    "Never output <nn>, <hp>, <tr>, training loops, or expert dispatch."
)


def gate_proposal_prompt(shapes, feedback: str = "") -> str:
    return f"""Write exactly one torch.nn.Module class named LLMGeneratedGate.
- Import only torch and torch.nn.
- Constructor: __init__(self, model_dim: int, num_experts: int).
- Define self.base = nn.Linear(model_dim, num_experts, bias=False).
- Native router weights will be copied into self.base before training.
- forward(self, x) accepts (..., model_dim) and returns finite raw logits (..., num_experts).
- Return self.base(x), optionally plus a small differentiable residual branch.
- Zero-initialize ONLY the residual's final projection for native step-zero routing.
- Do not use softmax, top-k, random sampling in forward, dropout, or batch normalization.
- Do not hard-code dimensions, change devices in forward, or return tuples.
- Preserve the native projection at initialization; improve adaptation after gate-only training.
- Keep parameter/runtime overhead small. Finish the entire class; no ellipses or placeholders.
Router shapes: {shapes!r}.
{feedback}
Return only <gate> followed by the complete source and </gate>."""


@contextmanager
def gate_prompt_scope(chat_bot):
    previous = getattr(chat_bot, "system_prompt", None)
    chat_bot.system_prompt = GATE_SYSTEM_PROMPT
    try:
        yield
    finally:
        chat_bot.system_prompt = previous


def gate_sft_examples(records: list[dict], shapes) -> list[dict]:
    """Requirements -> measured high-performing gate (elite SFT).

    Conditioning on a gate's own score/source would leak its target. The quality
    signal selects the target, rather than being placed in its input prompt.
    """
    prompt = gate_proposal_prompt(shapes)
    return [{
        "trial_id": r["trial_id"], "architecture_id": r["architecture_id"],
        "selection_trials": r["selection_trials"],
        "selection_accuracy": r["selection_accuracy"],
        "messages": [
            {"role": "system", "content": GATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "<gate>\n" + r["source"].strip() + "\n</gate>"},
        ],
    } for r in records]
