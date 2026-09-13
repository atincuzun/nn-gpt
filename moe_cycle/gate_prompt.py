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
    return f"""Write exactly one complete Python module defining the class LLMGeneratedGate.

You are INVENTING a trainable router scorer. You are not copying the native
router and not imitating any example.

WHAT THIS MODULE IS
It receives x of shape (..., model_dim) and returns raw, finite logits of shape
(..., num_experts). Nothing else. The surrounding DeepSeek code performs softmax,
top-k selection, correction bias, scaling, dispatch, and auxiliary balancing.
Do NOT implement any of those.

NON-NEGOTIABLE BASE PATH
You MUST define self.base = nn.Linear(model_dim, num_experts, bias=False).
Native router weights are copied into self.base after construction, so its shape
must stay exactly (num_experts, model_dim). In forward it is used directly as
self.base(x). Do not wrap, normalize, rescale, or replace it.

THE EXACT-INITIALIZATION CONTRACT
Before training, the gate must reproduce the native router EXACTLY, so:
    output = self.base(x) + branch_1(x) + branch_2(x) + ...
and EVERY added branch must be identically zero at initialization.
Achieve this by making each branch's FINAL operation an nn.Linear that produces
num_experts values, with its weight zero-initialized, and its bias too if present:
    nn.init.zeros_(your_final.weight)
    nn.init.zeros_(your_final.bias)      # only if that layer has a bias
Anything may happen BEFORE that final zero projection (normalization,
activations, bottlenecks, extra layers). Because the terminal projection is zero,
the whole branch contributes exactly zero initially, so the initial output is
exactly self.base(x). This is verified numerically.

A REAL BRANCH IS MANDATORY
`return self.base(x)` alone is INVALID and will be rejected. At least one genuine
additional branch is REQUIRED. A branch counts only if it (1) depends on x,
(2) computes something distinct from self.base, (3) produces num_experts values,
(4) ends in a zero-initialized projection to num_experts, and (5) is actually
added in forward. Declaring unused layers, or creating a constant or a tensor that
does not depend on x, does NOT satisfy this.

TRAINABILITY RULE (important)
The terminal projection must be zero at init, but the layers BEFORE it should use
their normal default initialization. Do NOT multiply an already-zero projection
by an additional learnable scalar initialized to zero: that makes the gradients
zero for both and the branch can never start learning.

SHAPE RULES
self.base(x) has shape (..., num_experts). EVERY tensor you add to it must also
end in num_experts. Never add x directly:
    return self.base(x) + x        # INVALID: x ends in model_dim, not num_experts
Never hard-code 2048, 64, or any concrete size. Use model_dim, num_experts, and
widths derived from them (for example a rank of max(4, model_dim // 64)).

DESIGN SPACE (ingredients, not a template - invent a coherent architecture)
nn.Linear, nn.LayerNorm, nn.GELU, nn.SiLU, nn.Tanh, elementwise addition,
elementwise multiplication, bottlenecks, expansions, parallel feature paths, and
residual transforms inside a branch. Combine them yourself. Keep the parameter
count modest.

STRUCTURAL NOVELTY IS REQUIRED
If a reference gate is supplied below, it is provided to DIFFER FROM, not to
imitate. Do NOT reproduce the reference, reproduce it with renamed attributes,
reproduce it with only different widths or constants, or fall back to the plain
baseline. Change at least TWO meaningful architectural properties, for example:
number of branches; branch depth; expansion versus compression; where
normalization sits; activation arrangement; parallel versus sequential transforms;
multiplicative feature interaction; full-width versus bottleneck processing;
shared versus independent intermediate features.

REQUIRED SKELETON (fill in YOUR OWN branch; this is not a complete answer)
    import torch
    import torch.nn as nn

    class LLMGeneratedGate(nn.Module):
        def __init__(self, model_dim: int, num_experts: int):
            super().__init__()
            self.base = nn.Linear(model_dim, num_experts, bias=False)
            # define YOUR branch here, with its own layer names and structure
            # its final projection to num_experts must be zero-initialized:
            #   nn.init.zeros_(YOUR_LAST.weight)
            #   nn.init.zeros_(YOUR_LAST.bias)     # only if it has a bias

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # return self.base(x) plus YOUR branch output; shapes (..., num_experts)
Write the real code, not comments. Every layer you use must be created in
__init__ and used in forward.

FORBIDDEN
softmax; top-k; expert selection; dropout; batch normalization; random sampling;
rand*, normal, bernoulli, multinomial; non-torch imports; hard-coded dimensions;
moving devices inside forward; returning tuples. forward must be deterministic.

SILENT CHECK BEFORE ANSWERING
A. self.base exists with shape (num_experts, model_dim) and is used directly.
B. At least one genuine branch depends on x and is added in forward.
C. Every added branch ends in a projection to num_experts.
D. Every such final projection has zero-initialized weight (and bias if present).
E. Therefore every added branch is exactly zero at initialization.
F. The returned tensor has final dimension num_experts.
G. No tensor whose final dimension is model_dim is added to the logits.
H. No concrete model size is hard-coded.
I. The structure differs from the supplied reference, if any.
J. The source is complete, syntactically valid, and has no placeholders.

Router shapes: {shapes!r}.
{reference_block}{feedback_block}
Return ONLY the complete source between <gate> and </gate>. No markdown fences,
no explanation, no text before <gate> or after </gate>."""


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
