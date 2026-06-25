"""
convert_to_hf_dataset.py

Converts clean JSONL into HuggingFace dataset format.

Input:
  training/clean_dataset.jsonl

Output:
  training/hf_dataset
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset


INPUT_PATH = Path("training/clean_dataset.jsonl")
SAVE_PATH = Path("training/hf_dataset")


SYSTEM_PROMPT = (
    "You are a computational biophysics assistant helping operate a multi-agent "
    "RNA design, folding, MD, literature, and HDOCK workflow. Use HDOCK scores "
    "only as relative docking scores, not physical binding free energies."
)


def render_instruction_example(item: dict) -> str:
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output_text = item.get("output", "")

    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}\n\nINPUT:\n{input_text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{output_text}\n"
        "<|im_end|>\n"
    )


def render_chat_example(item: dict) -> str:
    parts = []

    for msg in item.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")

    return "\n".join(parts) + "\n"


def main() -> None:
    rows = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                continue

            if "messages" in item:
                text = render_chat_example(item)
            else:
                text = render_instruction_example(item)

            if text.strip():
                rows.append({"text": text})

    dataset = Dataset.from_list(rows)
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(SAVE_PATH))

    print(f"Saved HF dataset to {SAVE_PATH}")
    print(f"Examples: {len(dataset)}")


if __name__ == "__main__":
    main()