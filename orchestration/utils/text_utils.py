from __future__ import annotations


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