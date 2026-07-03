"""
merge_lora.py

Merge trained LoRA adapter weights into base model for deployment.

Usage:
  PY=/path/to/python python training/merge_lora.py
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.model_version_tracker import (
    cleanup_old_versions,
    register_model_version,
    set_active_model,
)


log = logging.getLogger("virtual_lab.training")

CONFIG_PATH = Path("training/training_config.yaml")
LORA_PATH = Path("training/output_model")
MERGED_OUTPUT_DIR = Path("training/merged_model")
LOGS_DIR = Path("training/logs")


def setup_logging() -> None:
    """Setup logging configuration."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "merge.log"),
            logging.StreamHandler(),
        ],
    )


def load_config() -> dict:
    """Load training configuration from YAML file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_lora_adapter() -> None:
    """Merge LoRA adapter into base model."""
    setup_logging()

    log.info("=== LoRA MERGE START ===")

    # Load configuration
    config = load_config()
    base_model_name = config["model_name"]

    log.info(f"Base model: {base_model_name}")
    log.info(f"LoRA adapter: {LORA_PATH}")
    log.info(f"Output directory: {MERGED_OUTPUT_DIR}")

    # Check if LoRA adapter exists
    if not LORA_PATH.exists():
        raise FileNotFoundError(f"LoRA adapter not found at {LORA_PATH}")

    # Load base model
    log.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load tokenizer
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    log.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    # Merge adapter
    log.info("Merging LoRA adapter into base model...")
    merged_model = model.merge_and_unload()

    # Prepare output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_output = MERGED_OUTPUT_DIR / f"version_{timestamp}"
    versioned_output.mkdir(parents=True, exist_ok=True)

    # Save merged model
    log.info(f"Saving merged model to {versioned_output}")
    merged_model.save_pretrained(versioned_output)
    tokenizer.save_pretrained(versioned_output)

    # Also save to default path for easy access
    if MERGED_OUTPUT_DIR.exists():
        shutil.rmtree(MERGED_OUTPUT_DIR)
    shutil.copytree(versioned_output, MERGED_OUTPUT_DIR)

    log.info(f"Merged model saved to {MERGED_OUTPUT_DIR}")
    log.info(f"Versioned copy saved to {versioned_output}")

    # Validate merged model
    log.info("Validating merged model...")
    try:
        test_model = AutoModelForCausalLM.from_pretrained(
            MERGED_OUTPUT_DIR,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        log.info("Merged model validation successful")
    except Exception as e:
        log.error(f"Merged model validation failed: {e}")
        raise

    # Register model version
    log.info("Registering model version...")
    version_id = register_model_version(
        str(MERGED_OUTPUT_DIR),
        metadata={
            "base_model": base_model_name,
            "lora_path": str(LORA_PATH),
            "merged_at": datetime.now().isoformat(),
        },
    )

    # Set as active model
    set_active_model(str(MERGED_OUTPUT_DIR), version_id)

    # Cleanup old versions
    max_versions = int(os.getenv("VLAB_MAX_MODEL_VERSIONS", "3"))
    cleanup_old_versions(max_versions)

    log.info("=== LoRA MERGE COMPLETE ===")

    return str(versioned_output)


if __name__ == "__main__":
    merge_lora_adapter()
