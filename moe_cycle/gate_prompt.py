"""Gate proposal prompt, deduplication, and Phase B gate-pair formatter.

The prompt mirrors the CV pipeline's conditioning: one reference artefact with
its measured score, plus a compact feedback block.  Round 0 has no prior gate,
so the baseline linear gate is used as the reference.
"""

from __future__ import annotations

import re
from contextlib import contextmanager

from .gate_store import structural_hash

GATE_SYSTEM_PROMPT = (
    "You design small, trainable MoE router scorers, not CV networks. "
    "Output only a complete Python module between <gate> and </gate>. "
    "Never output <nn>, <hp>, <tr>, training loops, or expert dispatch."
)

# Baseline reference for the first round, mirroring Gate_gen_with_gate.json.
BASELINE_GATE_CODE = (
    "import torch\n"
    "import torch.nn as nn\n\n"
    "class LLMGeneratedGate(nn.Module):\n"
    "    def __init__(self, model_dim: int, num_experts: int):\n"
    "        super().__init__()\n"
    "        self.base = nn.Linear(model_dim, num_experts, bias=False)\n\n"
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        return self.base(x)\n"
)


def gate_proposal_prompt(shapes, *, reference_source: str = "", feedback: str = "") -> str:
    """Single-reference gate proposal prompt.

    ``reference_source`` and ``feedback`` are optional so round 0 can bootstrap
    from the baseline contract alone.
    """
    reference_block = ""
    if reference_source.strip():
        reference_block = (
            "The current best MoE router gate is:\n"
            f"<gate>\n{reference_source.strip()}\n</gate>\n"
        )
    feedback_block = (
        "Measured results of earlier gate candidates in this run:\n"
        f"{feedback.strip()}\n" if feedback.strip() else ""
    )
    return f"""Write exactly one torch.nn.Module class named LLMGeneratedGate.
Requirements:
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
- Keep parameter and runtime overhead small. Finish the entire class; no ellipses or placeholders.
- Propose a structurally DIFFERENT gate from the reference below; do not re-emit it verbatim.
Router shapes: {shapes!r}.
{reference_block}{feedback_block}
Return only <gate> followed by the complete source and </gate>."""


@contextmanager
def gate_prompt_scope(chat_bot):
    """Isolate the gate proposal from any CV system prompt on the chatbot."""
    previous = getattr(chat_bot, "system_prompt", None)
    chat_bot.system_prompt = GATE_SYSTEM_PROMPT
    try:
        yield
    finally:
        chat_bot.system_prompt = previous


_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[^\s]")
_SHINGLE_N = 7


def code_minhash(code: str, num_perm: int = 128):
    """MinHash over token shingles, used as a near-duplicate *warning* only."""
    try:
        from datasketch import MinHash
    except ImportError:
        return None
    tokens = _TOKEN_RE.findall(code)
    shingle = MinHash(num_perm=num_perm)
    for index in range(max(1, len(tokens) - _SHINGLE_N + 1)):
        shingle.update(" ".join(tokens[index:index + _SHINGLE_N]).encode("utf-8"))
    return shingle


def near_duplicate_similarity(source: str, references: list[str]) -> float | None:
    """Highest MinHash Jaccard against prior gate sources, or None if unavailable.

    Gate code is highly templated, so this is advisory.  Exact duplicates are
    caught by :func:`is_duplicate_gate` using the AST hash.
    """
    candidate = code_minhash(source)
    if candidate is None:
        return None
    best = 0.0
    for reference in references:
        other = code_minhash(reference)
        if other is not None:
            best = max(best, candidate.jaccard(other))
    return round(best, 4)


def is_duplicate_gate(source: str, seen_hashes: set[str]) -> bool:
    """True when the gate's structure is byte-equivalent after normalisation."""
    try:
        return structural_hash(source) in seen_hashes
    except SyntaxError:
        return False


def gate_sft_examples(pairs: list[dict], shapes) -> list[dict]:
    """Phase B supervision: requirements -> higher-scoring gate source.

    Quality selects the target; it is deliberately not shown as an input, so the
    model learns the architecture rather than the score.
    """
    prompt = gate_proposal_prompt(shapes)
    examples = []
    for pair in pairs:
        examples.append({
            "pair_id": pair["pair_id"],
            "messages": [
                {"role": "system", "content": GATE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": "<gate>\n" + pair["higher_source"].strip() + "\n</gate>",
                },
            ],
        })
    return examples
