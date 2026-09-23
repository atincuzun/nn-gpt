"""Gate proposal prompt, deduplication, and Phase B gate-pair formatter.

The prompt asks for architectural improvement inside the runtime contract,
without prescribing a network. Like the CV pipeline, it conditions proposals
on a reference artefact and measured feedback when available.
"""

from __future__ import annotations

import re
from contextlib import contextmanager

from .gate_store import structural_hash

GATE_SYSTEM_PROMPT = (
    "You are an expert PyTorch neural network architect designing trainable MoE router gates. "
    "Invent expressive, nonlinear router networks using PyTorch's rich building blocks. "
    "The supplied interface constrains compatibility, not your architectural creativity."
)

# Baseline reference for the first round. MUST satisfy the same validator as
# every proposal: it carries a zero-initialised residual branch (silent at
# step zero, trainable afterwards), so the LLM is shown a COMPLIANT pattern
# instead of a plain linear gate our own degenerate-gate check would reject.
BASELINE_GATE_CODE = (
    "import torch\n"
    "import torch.nn as nn\n\n"
    "class LLMGeneratedGate(nn.Module):\n"
    "    def __init__(self, model_dim: int, num_experts: int):\n"
    "        super().__init__()\n"
    "        self.base = nn.Linear(model_dim, num_experts, bias=False)\n"
    "        self.branch = nn.Sequential(\n"
    "            nn.Linear(model_dim, 4 * num_experts),\n"
    "            nn.Tanh(),\n"
    "            nn.Linear(4 * num_experts, num_experts, bias=False),\n"
    "        )\n"
    "        nn.init.zeros_(self.branch[-1].weight)\n\n"
    "    def forward(self, x):\n"
    "        return self.base(x) + self.branch(x)\n"
)


def gate_proposal_prompt(
    shapes,
    *,
    reference_source: str = "",
    reference_accuracy: float | None = None,
    goal_accuracy: float | None = None,
    dataset: str | None = None,
    inherit_reference: bool = False,
) -> str:
    """Single-reference gate proposal prompt, LEMUR-CV pairing style.

    Mirrors NN_gen.json's improvement rows: one reference artifact with its
    measured score, one goal score, no history tables. With no scored
    reference (fresh store) the accuracy clauses degrade gracefully.
    """
    dataset_text = f" on the '{dataset}' image-classification task" if dataset else ""
    if goal_accuracy is not None:
        goal_sentence = (
            "Your task is to write a router gate that increases the mean CV "
            f"accuracy of the neural networks generated through it to at least "
            f"{goal_accuracy:.4f}{dataset_text}."
        )
    else:
        goal_sentence = (
            "Your task is to write a router gate that increases the mean CV "
            f"accuracy of the neural networks generated through it{dataset_text}."
        )
    if reference_source.strip():
        if inherit_reference:
            if reference_accuracy is not None:
                reference_sentence = (
                    "The following reference gate is your current router "
                    f"architecture; it achieved a mean CV accuracy of {reference_accuracy:.4f} "
                    "under identical evaluation conditions. Its trained weights are "
                    "copied into your new code by parameter name. Improve it by "
                    "extending it."
                )
            else:
                reference_sentence = (
                    "The following reference gate is your current router "
                    "architecture. Its trained weights are copied into your new "
                    "code by parameter name. Improve it by extending it."
                )
            closing = (
                "Keep the reference's module and parameter names for every part "
                "you reuse; put all new ideas in new submodules.\n\n"
                "Respond with only the complete Python module between `<gate>` and `</gate>`."
            )
        elif reference_accuracy is not None:
            reference_sentence = (
                "Use the following reference gate, which achieved a mean CV "
                f"accuracy of {reference_accuracy:.4f} under identical "
                "evaluation conditions, as the baseline for architectural "
                "inspiration. Improve it by making fundamental changes to the "
                "gate design."
            )
            closing = (
                "Develop your own architecture rather than copying the reference. "
                "Your design must go beyond the reference's mechanism, not restate it.\n\n"
                "Respond with only the complete Python module between `<gate>` and `</gate>`."
            )
        else:
            reference_sentence = (
                "Use the following reference gate as the baseline for "
                "architectural inspiration. Improve it by making fundamental "
                "changes to the gate design."
            )
            closing = (
                "Develop your own architecture rather than copying the reference. "
                "Your design must go beyond the reference's mechanism, not restate it.\n\n"
                "Respond with only the complete Python module between `<gate>` and `</gate>`."
            )
        reference_block = (
            ("Current gate code:" if inherit_reference else "Baseline gate code:") + "\n"
            f"<gate>\n{reference_source.strip()}\n</gate>\n"
        )
    else:
        reference_sentence = ""
        reference_block = ""
        closing = "Respond with only the complete Python module between `<gate>` and `</gate>`."

    parts = [
        "Generate a PyTorch neural network to serve as the router gate of a "
        "Mixture-of-Experts language model.",
        "",
        "Your task is to design the gating network itself while strictly "
        "satisfying the interface and behavioral contract below.",
        "",
        "Design goal:",
        "Create a coherent trainable architecture for token-dependent expert "
        "routing. The architecture should have meaningful learnable capacity "
        "beyond the required native-weight interface and should be capable of "
        "developing substantially different routing behavior during training.",
        "Choose the architecture, internal representations, transformations, "
        "nonlinearities, parameterization, and composition yourself. Do not "
        "rely on unnecessary operations or components that have no effect on "
        "the returned logits.",
        "The required initialization behavior is a constraint on the initial "
        "function of the network, not a restriction on its eventual learned "
        "function. The complete architecture must therefore satisfy the "
        "initialization condition while retaining its additional trainable "
        "capacity.",
        "",
        goal_sentence,
    ]
    if reference_sentence:
        parts.append(reference_sentence)
    parts.append("")
    parts.append("Gate contract:")
    parts.append("""* Define `class LLMGeneratedGate(nn.Module)` with:
  `__init__(self, model_dim: int, num_experts: int)`

* Define:
  `forward(self, x: torch.Tensor) -> torch.Tensor`

* Input shape:
  `(..., model_dim)`

* Output shape:
  `(..., num_experts)`

* Each token must be processed independently. Preserve all leading dimensions.

* The output must be a single tensor containing finite raw router logits.

* Expose exactly:
  `self.base = nn.Linear(model_dim, num_experts, bias=False)`""")
    if inherit_reference:
        parts.append("""
* The host copies the reference gate's trained weights into your modules by parameter name: every name you keep, including `self.base`, receives the reference's trained value.

* After that copy, the complete gate must reproduce the reference gate's output exactly for arbitrary valid inputs. Keep every reused module computationally unchanged, and make each new branch contribute exactly zero at initialization (zero-initialize its output layer) so only training moves the gate away from the reference's behavior.""")
    else:
        parts.append("""
* The host system copies the native router weights into `self.base`.

* Immediately after initialization and after those native weights have been copied, the complete gate must reproduce `self.base(x)` exactly for arbitrary valid inputs.""")
    parts.append("""
* Additional trainable parameters must nevertheless be connected to the output computation so that optimization can move the gate away from its initialization behavior.

* Softmax, expert selection, top-k routing, auxiliary routing losses, and expert dispatch are handled externally. Do not implement them.

* Use only PyTorch.

* Register all trainable parameters and submodules in `__init__`.

* Respect the module's device and dtype.

* `forward` must be deterministic.

* Do not use dropout, batch normalization, random sampling, or stochastic behavior.

* Derive all internal dimensions from `model_dim` and `num_experts`.

* Expected router shape:
  `(model_dim, num_experts) = {shapes!r}`""".format(shapes=shapes))
    if reference_block:
        parts.append("")
        parts.append(reference_block.rstrip())
    parts.append("")
    parts.append(closing)
    return "\n".join(parts)


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


def gate_sft_examples(pairs: list[dict], shapes, *,
                      inherit_reference: bool = False) -> list[dict]:
    """Outer-loop SFT supervision, LEMUR-CV pairing style (NN_gen.json mirror).

    Each pair becomes one training row exactly like the CV improvement rows:
    the user message shows the lower-scoring gate as the reference with its
    measured accuracy and states the higher gate's accuracy as the goal
    ("increase ... to at least X"), and the assistant message is the higher
    gate's code. No history, no tables — one pair is one example.
    """
    examples = []
    for pair in pairs:
        prompt = gate_proposal_prompt(
            shapes,
            reference_source=pair.get("lower_source", ""),
            reference_accuracy=pair.get("lower_accuracy"),
            goal_accuracy=pair.get("higher_accuracy"),
            dataset=pair.get("dataset"),
            inherit_reference=inherit_reference,
        )
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
