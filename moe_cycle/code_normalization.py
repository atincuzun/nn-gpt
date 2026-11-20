"""Recover LLM code that was returned as one line with literal escapes.

Some chat checkpoints emit a complete gate or NN module with every newline
escaped (``\\n``) instead of real line breaks.  The XML/code extractors keep
the payload verbatim, so the saved module cannot be imported.  Unescaping is
safe only when the whole payload is a single line: normal code is returned
unchanged, and callers compile-check the result before using it.
"""

from __future__ import annotations


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
