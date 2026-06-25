"""
validate_dataset.py

Validates and cleans JSONL training data before fine-tuning.

Reads:
  postrun_training_data/supervised.jsonl
  postrun_training_data/examples.jsonl
  postrun_training_data/literature_evidence.jsonl

Writes:
  training/clean_dataset.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


INPUT_FILES = [
    Path("postrun_training_data/supervised.jsonl"),
    Path("postrun_training_data/examples.jsonl"),
    Path("postrun_training_data/literature_evidence.jsonl"),
]

OUTPUT_PATH = Path("training/clean_dataset.jsonl")

MIN_OUTPUT_LENGTH = 30
MAX_INPUT_CHARS = 12000
MAX_OUTPUT_CHARS = 4000


def load_jsonl(path: Path) -> list[dict]:
    data = []
    if not path.exists():
        return data

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    data.append(item)
            except Exception:
                continue

    return data


def is_valid_entry(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False

    # Instruction-format examples.
    if "instruction" in entry and "output" in entry:
        if not str(entry.get("instruction", "")).strip():
            return False
        if len(str(entry.get("output", "")).strip()) < MIN_OUTPUT_LENGTH:
            return False
        return True

    # Chat-format examples.
    if "messages" in entry and isinstance(entry["messages"], list):
        return len(entry["messages"]) >= 2

    return False


def normalise_entry(entry: dict) -> dict | None:
    if not is_valid_entry(entry):
        return None

    if "instruction" in entry:
        out = dict(entry)
        out["instruction"] = str(out.get("instruction", ""))[:MAX_INPUT_CHARS]
        out["input"] = str(out.get("input", ""))[:MAX_INPUT_CHARS]
        out["output"] = str(out.get("output", ""))[:MAX_OUTPUT_CHARS]
        return out

    return entry


def deduplicate(data: Iterable[dict]) -> list[dict]:
    seen = set()
    out = []

    for item in data:
        key = json.dumps(
            {
                "instruction": item.get("instruction"),
                "input": item.get("input"),
                "output": item.get("output"),
                "messages": item.get("messages"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )[:3000]

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


def main() -> None:
    all_rows = []

    for path in INPUT_FILES:
        rows = load_jsonl(path)
        print(f"Loaded {len(rows)} rows from {path}")
        all_rows.extend(rows)

    cleaned = []
    for row in all_rows:
        normalised = normalise_entry(row)
        if normalised:
            cleaned.append(normalised)

    cleaned = deduplicate(cleaned)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in cleaned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Clean rows: {len(cleaned)}")
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()