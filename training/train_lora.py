"""
auto_retrain.py

Automated pipeline:
postrun JSONL → clean → HF dataset → LoRA training

Usage:
  PY=/path/to/python python training/auto_retrain.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


PY = os.getenv("PY", "python")
POSTRUN_DIR = Path("postrun_training_data")
TRAINING_DIR = Path("training")


def run(cmd: str) -> None:
    print(f"\n>>> {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    print("=== AUTO RETRAIN PIPELINE ===")

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Validate + clean selected SFT data.
    run(f"{PY} training/validate_dataset.py")

    cleaned_path = TRAINING_DIR / "clean_dataset.jsonl"

    if not cleaned_path.exists():
        print("Clean dataset not found — skipping training")
        return

    num_lines = count_lines(cleaned_path)
    size_bytes = cleaned_path.stat().st_size

    print(f"Clean dataset rows: {num_lines}")
    print(f"Clean dataset size: {size_bytes} bytes")

    if num_lines < int(os.getenv("VLAB_MIN_TRAIN_ROWS", "20")) or size_bytes < int(os.getenv("VLAB_MIN_TRAIN_BYTES", "5000")):
        print("Dataset too small — skipping training")
        return

    # 2. Convert to HF dataset.
    run(f"{PY} training/convert_to_hf_dataset.py")

    # 3. Train LoRA.
    run(f"{PY} training/train_lora.py")

    print("\nRetrain complete")


if __name__ == "__main__":
    main()