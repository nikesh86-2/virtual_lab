"""
auto_retrain.py

Automated pipeline:
postrun JSONL → clean → HF dataset → LoRA training → evaluation → merge → model replacement
"""

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger("virtual_lab.training")


def run(cmd: str) -> None:
    """Run a command and log output."""
    print(f"\n>>> {cmd}\n")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def count_lines(path: Path) -> int:
    """Count lines in a file."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def evaluate_model() -> bool:
    """Evaluate the trained model to check if it produces reasonable outputs."""
    print("\n=== EVALUATING TRAINED MODEL ===")

    py = os.getenv("PY", "python")

    try:
        run(f"{py} training/evaluate_model.py")
        print("Model evaluation passed")
        return True
    except Exception as e:
        print(f"Model evaluation failed: {e}")
        return False


def merge_model() -> bool:
    """Merge LoRA adapter into base model."""
    print("\n=== MERGING LORA ADAPTER ===")

    py = os.getenv("PY", "python")

    try:
        run(f"{py} training/merge_lora.py")
        print("Model merge successful")
        return True
    except Exception as e:
        print(f"Model merge failed: {e}")
        return False


def replace_model() -> bool:
    """Replace vLLM model with merged fine-tuned model."""
    print("\n=== REPLACING VLLM MODEL ===")

    # Check if auto-replace is enabled
    if not os.getenv("VLAB_AUTO_MODEL_REPLACE", "0") == "1":
        print("Auto model replacement disabled (VLAB_AUTO_MODEL_REPLACE != 1)")
        print("Merged model available at training/merged_model")
        return False

    merged_path = Path("training/merged_model")
    if not merged_path.exists():
        print(f"Merged model not found at {merged_path}")
        return False

    # Update VLLM_MODEL_PATH environment variable
    # This requires modifying the run script or restarting with new env
    print("Model replacement requires vLLM restart with updated VLLM_MODEL_PATH")
    print(f"Set VLLM_MODEL_PATH={merged_path.absolute()} and restart vLLM")

    # For now, just log the instruction
    # In production, this would restart the vLLM service
    return True


def main() -> None:
    print("=== AUTO RETRAIN PIPELINE ===")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    POSTRUN_DIR = Path("postrun_training_data")
    TRAINING_DIR = Path("training")

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    py = os.getenv("PY", "python")

    # 1. Validate + clean selected SFT data.
    print("\n=== STEP 1: VALIDATING AND CLEANING DATA ===")
    run(f"{py} training/validate_dataset.py")

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
    print("\n=== STEP 2: CONVERTING TO HF DATASET ===")
    run(f"{py} training/convert_to_hf_dataset.py")

    # 3. Train LoRA.
    print("\n=== STEP 3: TRAINING LORA ===")
    try:
        run(f"{py} training/train_lora.py")
    except Exception as e:
        print(f"Training failed: {e}")
        return

    # 4. Evaluate model.
    print("\n=== STEP 4: EVALUATING MODEL ===")
    if not evaluate_model():
        print("Model evaluation failed, skipping merge and replacement")
        return

    # 5. Merge model.
    print("\n=== STEP 5: MERGING MODEL ===")
    if not merge_model():
        print("Model merge failed, skipping replacement")
        return

    # 6. Replace model in vLLM.
    print("\n=== STEP 6: REPLACING MODEL ===")
    replace_model()

    print("\n=== AUTO RETRAIN PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
