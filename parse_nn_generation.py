"""Standalone thinking-stripping parser for LLM generation outputs.

The in-pipeline extractors (``ab/gpt/util/Util.py``) run directly on the raw
chat text. Reasoning models wrap their deliberation in ``<think>...</think>``
and similar markers; the pipeline's anchor-based extraction can then start
inside the thinking block, contaminating or losing the parsed artifacts.

This script:
  1. strips reasoning/thinking sections first (closed and unclosed forms),
  2. extracts ``<hp>/<tr>/<nn>`` blocks from the remaining text,
  3. validates (JSON for hp, ``ast.parse`` for code),
  4. flags truncation when a closing tag is missing.

It never modifies the pipeline or the original files.

Usage:
  python parse_nn_generation.py FILE [FILE ...]        parse specific outputs
  python parse_nn_generation.py DIR [DIR ...]          scan for **/full_output.txt
  python parse_nn_generation.py ... --write OUTDIR     write hp.txt/tr.py/new_nn.py
  python parse_nn_generation.py ... --json             machine-readable summary
  python parse_nn_generation.py ... --quiet            failures only
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

THINK_NAMES = ("think", "thinking", "thought", "reasoning", "analysis", "scratchpad")
_CLOSED_XML = re.compile(
    r"<(?:" + "|".join(THINK_NAMES) + r")>.*?</(?:" + "|".join(THINK_NAMES) + r")>",
    re.S | re.I,
)
_UNCLOSED_XML = re.compile(
    r"<(?:" + "|".join(THINK_NAMES) + r")>.*", re.S | re.I,
)
_CLOSED_BRACKET = re.compile(
    r"\[("
    + "|".join(THINK_NAMES)
    + r")\].*?\[/\1\]",
    re.S | re.I,
)
_UNCLOSED_BRACKET = re.compile(
    r"\[(?:"
    + "|".join(THINK_NAMES)
    + r")\].*",
    re.S | re.I,
)

_CODE_ANCHOR = re.compile(r"(?m)^(import |from |class |def |supported_hyperparameters)")
_FENCE_OPEN = re.compile(r"(?m)^(?:```(?:python|py)?\s*)$")
_FENCE_CLOSE = re.compile(r"(?m)^```\s*$")


def strip_thinking(text: str) -> str:
    """Remove reasoning/thinking sections from chat output."""
    text = _CLOSED_XML.sub("", text)
    text = _CLOSED_BRACKET.sub("", text)
    text = _UNCLOSED_XML.sub("", text)
    text = _UNCLOSED_BRACKET.sub("", text)
    return text


def extract_block(text: str, name: str, prefer_last: bool) -> tuple[str | None, bool]:
    """Extract the ``<name>...</name>`` region.

    Reasoning sections are stripped first. Returns ``(content, truncated)``;
    ``truncated`` is True when the closing tag is missing and the content runs
    to the end of the text.
    """
    text = strip_thinking(text)
    open_tag = f"<{name}>"
    close_tag = f"</{name}>"
    opens = [m.start() for m in re.finditer(re.escape(open_tag), text)]
    closes = [m.start() for m in re.finditer(re.escape(close_tag), text)]

    if closes:
        end = closes[-1]
        preceding = [o for o in opens if o < end]
        if not preceding:
            return None, False
        start = (preceding[-1] if prefer_last else preceding[0]) + len(open_tag)
        return text[start:end].strip(), False

    if opens:
        start = (opens[-1] if prefer_last else opens[0]) + len(open_tag)
        return text[start:].strip(), True

    return None, False


def extract_hyperparam(text: str) -> tuple[str | None, bool]:
    return extract_block(text, "hp", prefer_last=False)


def extract_transform(text: str) -> tuple[str | None, bool]:
    return extract_block(text, "tr", prefer_last=False)


def extract_nn_code(text: str) -> tuple[str | None, bool]:
    text = strip_thinking(text)
    code, truncated = extract_block(text, "nn", prefer_last=True)
    if code is not None:
        return code, truncated

    fence_match = _FENCE_OPEN.search(text)
    if fence_match:
        after = text[fence_match.end():]
        fence_end = _FENCE_CLOSE.search(after)
        if fence_end:
            return after[:fence_end.start()].strip(), False
        return after.strip(), True

    if "class Net" in text and "def forward" in text:
        cut = max(
            text.rfind("</tr>") + len("</tr>"),
            text.rfind("<tr>") + len("<tr>"),
            text.rfind("</hp>") + len("</hp>"),
            text.rfind("<hp>") + len("<hp>"),
            0,
        )
        region_end = text.rindex("</nn>") if "</nn>" in text else len(text)
        truncated = "</nn>" not in text
        seg = text[cut:region_end]
        anchor = _CODE_ANCHOR.search(seg)
        if anchor is not None:
            return seg[anchor.start():].strip(), truncated

    return None, False


def parse_hp_json(content: str) -> tuple[dict | None, str | None]:
    if content is None:
        return None, "no <hp> block"
    try:
        return json.loads(content), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON at char {exc.pos}: {exc.msg}"


def parse_code(content: str) -> tuple[bool, str | None]:
    if content is None:
        return False, "no block found"
    try:
        ast.parse(content)
        return True, None
    except SyntaxError as exc:
        return False, f"syntax error line {exc.lineno}: {exc.msg}"


def parse_output(text: str) -> dict:
    cleaned = strip_thinking(text)
    hp, hp_truncated = extract_hyperparam(cleaned)
    tr, tr_truncated = extract_transform(cleaned)
    nn, nn_truncated = extract_nn_code(cleaned)

    hp_obj, hp_error = parse_hp_json(hp)
    tr_ok, tr_error = parse_code(tr)
    nn_ok, nn_error = parse_code(nn)

    return {
        "hp": {
            "found": hp is not None,
            "truncated": hp_truncated,
            "valid": hp_obj is not None,
            "error": hp_error,
            "content": hp,
        },
        "tr": {
            "found": tr is not None,
            "truncated": tr_truncated,
            "valid": tr_ok,
            "error": tr_error,
            "content": tr,
        },
        "nn": {
            "found": nn is not None,
            "truncated": nn_truncated,
            "valid": nn_ok,
            "error": nn_error,
            "content": nn,
        },
    }


def collect_inputs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("full_output.txt")))
        elif path.is_file():
            files.append(path)
    return files


def write_artifacts(source: Path, result: dict, outdir: Path) -> Path:
    target = outdir / source.parent.name
    target.mkdir(parents=True, exist_ok=True)
    for key, filename in (("hp", "hp.txt"), ("tr", "tr.py"), ("nn", "new_nn.py")):
        block = result[key]
        if block["found"] and block["content"]:
            (target / filename).write_text(block["content"] + "\n")
        else:
            (target / filename).write_text("")
    return target


def report(source: Path, result: dict, write_dir: Path | None) -> dict:
    summary = {
        "file": str(source),
        "hp": "OK" if result["hp"]["valid"] else "FAIL",
        "tr": "OK" if result["tr"]["valid"] else "FAIL",
        "nn": "OK" if result["nn"]["valid"] else "FAIL",
    }
    lines = [f"{source}"]
    for name in ("hp", "tr", "nn"):
        block = result[name]
        status = "OK" if block["valid"] else "FAIL"
        flags = []
        if not block["found"]:
            flags.append("MISSING")
        if block["truncated"]:
            flags.append("TRUNCATED")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        error = f" -- {block['error']}" if block["error"] else ""
        lines.append(f"  {name}: {status}{suffix}{error}")
    if result["nn"]["found"] and result["nn"]["content"]:
        first = result["nn"]["content"].splitlines()[0]
        lines.append(f"  nn first line: {first}")
    if write_dir is not None:
        lines.append(f"  wrote -> {write_dir / source.parent.name}")
    print("\n".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="full_output.txt files or directories to scan")
    parser.add_argument("--write", type=Path, help="write cleaned hp.txt/tr.py/new_nn.py here")
    parser.add_argument("--json", action="store_true", help="print machine-readable summary")
    parser.add_argument("--quiet", action="store_true", help="report only failed parses")
    args = parser.parse_args()

    files = collect_inputs(args.paths)
    if not files:
        print(f"no full_output.txt files found under {args.paths}", file=sys.stderr)
        return 1

    summaries = []
    failed = 0
    for source in files:
        text = source.read_text(encoding="utf-8", errors="replace")
        result = parse_output(text)
        if args.write is not None:
            write_artifacts(source, result, args.write)
        if args.json:
            summaries.append({
                "file": str(source),
                **{k: {kk: vv for kk, vv in v.items() if kk != "content"}
                   for k, v in result.items()},
            })
        elif not args.quiet or not all(result[k]["valid"] for k in ("hp", "tr", "nn")):
            report(source, result, args.write)
        if not all(result[k]["valid"] for k in ("hp", "tr", "nn")):
            failed += 1

    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        print(f"\n{len(files) - failed}/{len(files)} fully valid parses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
