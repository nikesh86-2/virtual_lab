from __future__ import annotations

import re
from pathlib import Path

def _safe_file_tag(value: str | None, default: str = "target") -> str:
    """
    Convert a PDB ID/path/name into a filesystem-safe compact tag.
    """
    if not value:
        return default

    raw = str(value).strip()

    # If a file path was passed, use its stem.
    if "/" in raw or raw.endswith(".pdb") or raw.endswith(".pdbqt"):
        raw = Path(raw).stem

    raw = raw.upper()

    # Remove common suffixes so /path/8k75.pdb -> 8K75,
    # but /path/8k75_receptor.pdbqt -> 8K75_RECEPTOR.
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    raw = raw.strip("_")

    return raw or default

def truncate_str(text: str, max_chars: int = 1000) -> str:
    """
    Truncate long strings while making truncation explicit.
    """
    if not text:
        return ""

    text = str(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "... [TRUNCATED — conclusions must not rely on omitted content]"


def clean_llm_output(text: str) -> str:
    """
    Remove common chat/template artifacts from local LLM output.
    """
    if not text:
        return ""

    text = str(text).replace("\\\\", "\\")

    for token in [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|begin_of_text|>",
    ]:
        text = text.replace(token, "")

    return text.strip()


def clean_structural_output(text: str) -> str:
    """
    Normalise structural wrapper output for LLM summarisation.
    """
    if not text:
        return ""

    text = str(text).replace("\\\\", "\\")
    text = " ".join(text.split())

    return text


def validate_text_block(text: str) -> tuple[bool, str]:
    """
    Basic text sanity validation.
    """
    if not text or len(str(text).strip()) < 10:
        return False, "Text too short"

    return True, "OK"