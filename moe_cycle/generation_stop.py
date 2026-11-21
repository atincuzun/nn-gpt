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
    """Stop each sequence once a requested block is closed.

    Only the completion is inspected: the prompt shows the reference model
    inside ``<tr>``/``<nn>`` tags, so a closed block in the prompt would
    otherwise stop the generation before it began.

    The verdict is per row, so batched generation stops each sequence at its
    own closing tag instead of stopping the whole batch when row 0 finishes
    (which truncated every other row). The opening tag is remembered once
    seen: a block longer than the decoded tail -- most CV models are -- has
    its opening tag scrolled out of the window by the time the closing tag
    arrives, and a check that needs both tags in the tail never fired for it,
    so those generations ran to the token cap.
    """

    def __init__(self, tokenizer, prompt_len: int = 0, pairs=_TAG_PAIRS,
                 tail_tokens: int = _TAIL_TOKENS):
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.pairs = pairs
        self.tail_tokens = tail_tokens
        self._opened: dict[int, int] = {}
        self._done: set[int] = set()

    def _row_done(self, row: int, ids) -> bool:
        if row in self._done:
            return True
        completion = ids[self.prompt_len :]
        if completion.shape[-1] == 0:
            return False
        tail = completion[-self.tail_tokens :]
        try:
            text = self.tokenizer.decode(tail, skip_special_tokens=True)
        except Exception:
            return False
        for index, (opening, closing) in enumerate(self.pairs):
            start = text.rfind(opening)
            if start >= 0:
                self._opened[row] = index
                if text.find(closing, start + len(opening)) >= 0:
                    self._done.add(row)
                    return True
                continue
            # The opening tag scrolled out of the tail: everything still in
            # the window was generated after it.
            if self._opened.get(row) == index and closing in text:
                self._done.add(row)
                return True
        return False

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids is None or len(input_ids) == 0:
            return False
        verdicts = [self._row_done(row, input_ids[row]) for row in range(input_ids.shape[0])]
        return torch.tensor(verdicts, dtype=torch.bool, device=input_ids.device)


def _ends_in_loop(text: str, *, min_unit: int = 24, max_unit: int = 400,
                  repeats: int = 6) -> bool:
    """True when the text ends with one block repeated ``repeats`` times.

    Exact periodicity only: legitimate code repeats similar lines, but an
    identical multi-line block six times in a row is a sampling loop that
    never reaches its closing tag.
    """
    length = len(text)
    for unit in range(min_unit, min(max_unit, length // repeats) + 1):
        tail = text[length - unit:]
        if all(
            text[length - k * unit: length - (k - 1) * unit] == tail
            for k in range(2, repeats + 1)
        ):
            return True
    return False


class _LoopStoppingCriteria(StoppingCriteria):
    """Stop a sequence that has fallen into an exact repetition loop.

    At low temperature the model sometimes emits the same block until the
    token cap. Such a completion can never close its ``<nn>`` block, so it
    cannot become a usable model; in batched generation it also holds every
    other row of the batch until the cap. Checked every ``every`` steps on
    the decoded tail of each row.
    """

    def __init__(self, tokenizer, prompt_len: int = 0, tail_tokens: int = 1024,
                 every: int = 32):
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.tail_tokens = tail_tokens
        self.every = every
        self.calls = 0
        self.stopped: set[int] = set()

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids is None or len(input_ids) == 0:
            return False
        self.calls += 1
        verdicts = []
        for row in range(input_ids.shape[0]):
            if row in self.stopped:
                verdicts.append(True)
                continue
            done = False
            if self.calls % self.every == 0:
                completion = input_ids[row][self.prompt_len:]
                if completion.shape[-1] >= 256:
                    try:
                        text = self.tokenizer.decode(
                            completion[-self.tail_tokens:], skip_special_tokens=True
                        )
                    except Exception:
                        text = ""
                    if text and _ends_in_loop(text):
                        self.stopped.add(row)
                        done = True
            verdicts.append(done)
        return torch.tensor(verdicts, dtype=torch.bool, device=input_ids.device)


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
        added = [
            _TagStoppingCriteria(tokenizer, prompt_len=prompt_len),
            _LoopStoppingCriteria(tokenizer, prompt_len=prompt_len),
        ]
        criteria = kwargs.get("stopping_criteria")
        if criteria is None:
            kwargs["stopping_criteria"] = StoppingCriteriaList(added)
        elif isinstance(criteria, StoppingCriteriaList):
            criteria.extend(added)
        else:
            kwargs["stopping_criteria"] = StoppingCriteriaList(list(criteria) + added)
        return original_generate(*args, **kwargs)

    model.generate = generate
    model._moe_tag_stopping = True
    return True
