"""Recover LLM code that was returned as one line with literal escapes.

Some chat checkpoints emit a complete gate or NN module with every newline
escaped (``\\n``) instead of real line breaks.  The XML/code extractors keep
the payload verbatim, so the saved module cannot be imported.  Unescaping is
safe only when the whole payload is a single line: normal code is returned
unchanged, and callers compile-check the result before using it.
"""

from __future__ import annotations

import ast
import re


#: XML framing the LEMUR chat format wraps generated payloads in.  A payload
#: that keeps them is not importable as Python.  ``code`` is not part of the
#: LEMUR vocabulary, but models emit it often enough to be worth stripping.
_XML_TAG_RE = re.compile(r"</?\s*(?:nn|tr|hp|gate|code)\s*>", re.IGNORECASE)


def strip_xml_tags(text: str) -> str:
    """Remove the LEMUR XML framing from a payload.

    Only the tags themselves are removed; the payload between them is kept
    verbatim, so this is only safe for callers that compile-check the result.
    """
    return _XML_TAG_RE.sub("", text or "").strip()


def _call_name(node: ast.AST) -> str:
    """Return ``a.b.c`` for a call target, or an empty string."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _int_value(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    return None


def repair_classifier_width(source: str) -> str | None:
    """Recompute a classifier width that contradicts the feature map.

    A generated model routinely builds a conv stack without updating its
    channel counter, so the width it hands to the classifier's first Linear
    is the width it never receives and the model dies at the first forward
    ("mat1 and mat2 shapes cannot be multiplied").

    The width is recomputed from the stack's last Conv2d and the pooling
    size, and only written when both are statically known. The caller is
    expected to run the repaired model before keeping it.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    conv_outputs: list[int] = []
    pool_area: int | None = None
    classifier_linear: ast.Call | None = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name == "nn.Conv2d" and len(node.args) >= 2:
                out_channels = _int_value(node.args[1])
                if out_channels is None:
                    return None
                conv_outputs.append(out_channels)
            elif name in ("nn.AdaptiveAvgPool2d", "nn.AdaptiveMaxPool2d"):
                if node.args and isinstance(node.args[0], ast.Tuple):
                    dims = [_int_value(item) for item in node.args[0].elts]
                    if len(dims) == 2 and all(d is not None for d in dims):
                        pool_area = dims[0] * dims[1]
        elif isinstance(node, ast.Assign) and classifier_linear is None:
            targets = [
                target.attr for target in node.targets
                if isinstance(target, ast.Attribute)
            ]
            if not any("classifier" in target for target in targets):
                continue
            for inner in ast.walk(node.value):
                if isinstance(inner, ast.Call) and _call_name(inner.func) == "nn.Linear":
                    classifier_linear = inner
                    break

    if not conv_outputs or pool_area is None or classifier_linear is None:
        return None
    if not classifier_linear.args:
        return None
    width_arg = classifier_linear.args[0]
    if not isinstance(width_arg, ast.Name):
        return None

    width = conv_outputs[-1] * pool_area
    lines = source.splitlines(keepends=True)
    start = width_arg.lineno - 1
    end = width_arg.end_lineno - 1
    if start != end:
        return None
    line = lines[start]
    lines[start] = line[: width_arg.col_offset] + str(width) + line[width_arg.end_col_offset :]
    return "".join(lines)


def unescape_single_line_code(text: str) -> str:
    """Return *text* with literal escape sequences expanded when it is one line."""
    if "\n" in text or "\\n" not in text:
        return text
    return (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def unescape_embedded_escapes(text: str) -> str:
    """Expand literal escape sequences inside an otherwise multi-line payload.

    A generated module can arrive with real line breaks *and* literal ``\\n``
    inside the same block, which no parser accepts. Unlike
    :func:`unescape_single_line_code` this one always rewrites, so callers
    must compile-check the result and keep the original when it does not parse.
    """
    if "\\n" not in text:
        return text
    return (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
    )


_CV_HP = "\n\ndef supported_hyperparameters():\n    # Hyperparameters accepted by this LEMUR model.\n    return {\"lr\", \"momentum\"}\n"
_CV_TRAIN = """    def train_setup(self, prm):
        # Standard LEMUR training setup.
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=float(prm.get("lr", 1e-3)),
            momentum=float(prm.get("momentum", 0.9)),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.device = next(self.parameters()).device

    def learn(self, train_data):
        # Standard LEMUR classification training loop.
        self.train()
        correct = 0
        total = 0
        total_loss = 0.0
        for batch in train_data:
            inputs, labels = batch[0], batch[1]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total += len(labels)
        return correct / max(total, 1), total_loss / max(total, 1)

"""


def ensure_lemur_cv_interface(source: str) -> str:
    """Ensure generated CV code has LEMUR's module and training contract."""
    tree = ast.parse(source)
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Net"
    ]
    if not classes:
        return source
    cls = classes[0]
    has_hp = any(
        isinstance(node, ast.FunctionDef) and node.name == "supported_hyperparameters"
        for node in tree.body
    )
    names = {
        node.name for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    lines = source.splitlines(keepends=True)
    if not {"train_setup", "learn"} <= names:
        forward = next(
            (node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward"),
            None,
        )
        insert_at = (forward.lineno - 1) if forward is not None else cls.end_lineno
        while insert_at > 0 and not lines[insert_at - 1].strip() and forward is not None:
            insert_at -= 1
        lines.insert(insert_at, _CV_TRAIN)
        source = "".join(lines)
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Net")
    if not has_hp:
        lines = source.splitlines(keepends=True)
        lines.insert(cls.lineno - 1, _CV_HP)
        source = "".join(lines)
    return source
