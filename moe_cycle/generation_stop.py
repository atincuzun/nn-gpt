"""Stop generation at the closing tag of the block that was asked for.

The chat model has no stop-sequence support, so every generation runs to
``max_new_tokens`` even when the payload it was asked for -- a CV model or a
gate module -- is complete after a fraction of that. The completion then
continues with unrelated text, and the pipeline pays for all of it.

The criterion stops as soon as the completion contains an opening and a closing
tag of the same kind, which is exactly where the requested block ends. Only the
tail of the completion is decoded, so the check is cheap and cannot be fooled
by the reference material echoed from the prompt.
"""

from __future__ import annotations

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

#: (opening, closing) pairs of the blocks the pipeline asks for.
_TAG_PAIRS: tuple[tuple[str, str], ...] = (("<nn>", "</nn>"), ("<gate>", "</gate>"))

#: How many trailing tokens are decoded for the check.
_TAIL_TOKENS = 256


class _TagStoppingCriteria(StoppingCriteria):
    """Stop once a requested block is closed.

    Only the completion is inspected: the prompt shows the reference model
    inside ``<tr>``/``<nn>`` tags, so a closed block in the prompt would
    otherwise stop the generation before it began.
    """

    def __init__(self, tokenizer, prompt_len: int = 0, pairs=_TAG_PAIRS,
                 tail_tokens: int = _TAIL_TOKENS):
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.pairs = pairs
        self.tail_tokens = tail_tokens

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if input_ids is None or len(input_ids) == 0:
            return False
        completion = input_ids[0][self.prompt_len :]
        if completion.shape[-1] == 0:
            return False
        tail = completion[-self.tail_tokens :]
        try:
            text = self.tokenizer.decode(tail, skip_special_tokens=True)
        except Exception:
            return False
        for opening, closing in self.pairs:
            start = text.find(opening)
            if start < 0:
                continue
            if text.find(closing, start) >= 0:
                return True
        return False


def install_tag_stopping_criterion(model, tokenizer) -> bool:
    """Make the model's ``generate`` stop at the closing tag.

    Returns True when the model was wrapped. The wrapper only adds a stopping
    criterion to calls that do not already carry one, so an explicit criterion
    from the caller still wins.
    """
    if model is None or tokenizer is None:
        return False
    if getattr(model, "_moe_tag_stopping", False):
        return True

    original_generate = model.generate

    def generate(*args, **kwargs):
        prompt_len = 0
        for key in ("input_ids", "inputs"):
            value = kwargs.get(key)
            if value is not None and hasattr(value, "shape"):
                prompt_len = int(value.shape[-1])
                break
        if prompt_len == 0 and args:
            value = args[0]
            if hasattr(value, "shape"):
                prompt_len = int(value.shape[-1])
        criterion = _TagStoppingCriteria(tokenizer, prompt_len=prompt_len)
        criteria = kwargs.get("stopping_criteria")
        if criteria is None:
            kwargs["stopping_criteria"] = StoppingCriteriaList([criterion])
        elif isinstance(criteria, StoppingCriteriaList):
            criteria.append(criterion)
        else:
            kwargs["stopping_criteria"] = StoppingCriteriaList(
                list(criteria) + [criterion]
            )
        return original_generate(*args, **kwargs)

    model.generate = generate
    model._moe_tag_stopping = True
    return True
