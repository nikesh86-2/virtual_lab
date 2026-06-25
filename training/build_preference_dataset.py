"""
build_preference_dataset.py

Converts postrun preferences.jsonl into a simple preference dataset.

Input:
  postrun_training_data/preferences.jsonl

Output:
  training/preference_dataset.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path


INPUT = Path("postrun_training_data/preferences.jsonl")
OUTPUT = Path("training/preference_dataset.jsonl")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with INPUT.open("r", encoding="utf-8") as f_in, OUTPUT.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            try:
                item = json.loads(line)
            except Exception:
                continue

            row = {
                "prompt": item.get("prompt", ""),
                "chosen": item.get("chosen", ""),
                "rejected": item.get("rejected", ""),
                "preference_type": item.get("preference_type"),
                "metadata": item.get("metadata", {}),
            }

            if row["prompt"] and row["chosen"] and row["rejected"]:
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1

    print(f"Wrote {n} preference rows to {OUTPUT}")


if __name__ == "__main__":
    main()