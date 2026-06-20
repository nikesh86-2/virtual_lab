"""
auto_retrain.py

Automated pipeline:
JSONL → clean → HF dataset → LoRA training
"""

import os
import subprocess

def run(cmd):
    print(f"\n>>> {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)


def main():
    print("=== AUTO RETRAIN PIPELINE ===")

    # ------------------------------------------------------------
    # 1. COMBINE DATA
    # ------------------------------------------------------------
    run("cat training_data/*.jsonl > training_data/combined.jsonl")

    # ------------------------------------------------------------
    # 2. VALIDATE + CLEAN
    # ------------------------------------------------------------
    run("${PY} training/validate_dataset.py")

    # ✅ --------------------------------------------------------
    # 3. DATASET SIZE CHECK (PLACE IT HERE)
    # ------------------------------------------------------------
    cleaned_path = "training/clean_dataset.jsonl"
    
    with open(cleaned_path) as f:
        num_lines = sum(1 for _ in f)


    if not os.path.exists(cleaned_path):
        print("⚠️ Clean dataset not found — skipping training")
        return

    if num_lines < 20 or os.path.getsize(cleaned_path) < 5000:
        print("⚠️ Dataset too small — skipping training")
        return

    # ------------------------------------------------------------
    # 4. CONVERT TO HF DATASET
    # ------------------------------------------------------------
    run("${PY} training/convert_to_hf_dataset.py")

    # ------------------------------------------------------------
    # 5. TRAIN LORA
    # ------------------------------------------------------------
    run("${PY}n training/train_lora.py")

    print("\n✅ RETRAIN COMPLETE")


if __name__ == "__main__":
    main()
