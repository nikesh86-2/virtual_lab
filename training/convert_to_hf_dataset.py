"""
convert_to_hf_dataset.py

Converts your JSONL training_data into HuggingFace dataset format.
"""

import json
import os
from datasets import Dataset


def load_jsonl_files(folder="training_data"):
    data = []

    for file in os.listdir(folder):
        if file.endswith(".jsonl"):
            path = os.path.join(folder, file)

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)

                        text = build_prompt(item)

                        data.append({
                            "text": text
                        })

                    except Exception:
                        continue

    return data


def build_prompt(item):
    """
    Converts your instruction format into training text
    """

    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output_text = item.get("output", "")

    # ✅ Qwen / LLaMA style chat format
    return (
        "<|system|>\nYou are a scientific research assistant.\n"
        "<|user|>\n"
        f"{instruction}\n{input_text}\n"
        "<|assistant|>\n"
        f"{output_text}"
    )


def main():
    data = load_jsonl_files()

    dataset = Dataset.from_list(data)

    save_path = "training/hf_dataset"
    dataset.save_to_disk(save_path)

    print(f"✅ Dataset saved to {save_path}")
    print(f"✅ Size: {len(dataset)} examples")


if __name__ == "__main__":
    main()