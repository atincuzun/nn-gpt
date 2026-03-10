#!/usr/bin/env python3
"""Thin CLI wrapper for the reusable HF/Tutel MoE editor."""

try:
    from ab.gpt.moe.hf_moe_editor import main
except ImportError:
    from hf_moe_editor import main


if __name__ == "__main__":
    main()
