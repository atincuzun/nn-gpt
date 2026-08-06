"""The replacement-gate contract and external gate loading.

The gate system does **not** ship concrete gate implementations.  A
replacement gate is provided externally — user code, LLM-generated source,
another package — and must satisfy the contract below.  Everything else
(discovery, replacement, integration, initialization, verification, freezing,
training, checkpoints) is handled by the rest of the package.

Contract
--------
A replacement gate is a ``torch.nn.Module`` such that:

- Constructor: ``__init__(self, model_dim: int, num_experts: int)``.
- Forward: ``forward(self, x)`` accepts ``(..., model_dim)`` and returns raw
  router logits ``(..., num_experts)`` (floating point, finite).  Softmax,
  sigmoid, top-k, auxiliary losses, and expert dispatch are left to the native
  MoE wrapper installed around the gate by :mod:`moe_gate_only.install`.
- Optional ``base = nn.Linear(model_dim, num_experts, bias=False)``: lets the
  system copy the native router projection into the gate at step zero
  (:func:`moe_gate_only.morphism.initialize_gate_from_projection`), which also
  enables A0 routing-equivalence checks.
- Optional ``initialize_from_projection(self, weight, bias=None) -> dict``:
  function-preserving initialization protocol.  The gate takes full control of
  its own morphism and reports a ``morphism_metrics`` dict.
- Optional ``morphism_metrics`` dict attribute: self-reported initialization
  diagnostics consumed by the cycle's morphism-metrics service.

Factories
---------
Gates are supplied to :func:`moe_gate_only.install_gates` (or
``MoEGateSession.replace``) either as

- a callable factory ``(model_dim, num_experts) -> nn.Module``,
- a name resolved through a caller-provided ``gate_registry`` mapping, or
- compiled source via :func:`compile_gate_from_string` /
  ``MoEGateSession.replace_source``.
"""

from __future__ import annotations


def compile_gate_from_string(code_string: str, class_name: str = "LLMGeneratedGate") -> type:
    """Compile a PyTorch ``nn.Module`` class from a Python code string.

    Parameters
    ----------
    code_string:
        Raw Python source. Must contain a class definition of ``class_name``.
    class_name:
        Name of the class to extract from the compiled module.

    Returns
    -------
    type
        The compiled class (not an instance).  Call it with
        ``(model_dim, num_experts)`` to instantiate a gate.

    Example
    -------
    >>> code = '''
    ... class LLMGeneratedGate(nn.Module):
    ...     def __init__(self, model_dim, num_experts):
    ...         super().__init__()
    ...         self.proj = nn.Linear(model_dim, num_experts)
    ...     def forward(self, x):
    ...         return self.proj(x)
    ... '''
    >>> Gate = compile_gate_from_string(code)
    >>> gate = Gate(64, 8)
    """
    namespace: dict = {}
    exec(compile(code_string, "<llm_generated>", "exec"), namespace)
    gate_class = namespace.get(class_name)
    if gate_class is None:
        raise ValueError(f"Code string must define a class named {class_name!r}")
    return gate_class
