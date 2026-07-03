"""
train_lora.py

Actual LoRA fine-tuning implementation using transformers/peft/accelerate.

Usage:
  PY=/path/to/python python training/train_lora.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


log = logging.getLogger("virtual_lab.training")

CONFIG_PATH = Path("training/training_config.yaml")
HF_DATASET_PATH = Path("training/hf_dataset")
OUTPUT_DIR = Path("training/output_model")
LOGS_DIR = Path("training/logs")


def load_config() -> dict:
    """Load training configuration from YAML file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging() -> None:
    """Setup logging configuration."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "training.log"),
            logging.StreamHandler(),
        ],
    )


def load_model_and_tokenizer(config: dict):
    """Load base model and tokenizer."""
    model_name = config["model_name"]

    log.info(f"Loading model: {model_name}")
    log.info(f"Loading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("Set pad_token to eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    return model, tokenizer


def setup_lora(model: torch.nn.Module, config: dict) -> torch.nn.Module:
    """Configure and apply LoRA to the model."""
    lora_config = config.get("lora", {})

    lora_cfg = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_config.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    log.info(f"LoRA config: r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, "
             f"target_modules={lora_cfg.target_modules}")

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model


def load_dataset() -> dict:
    """Load HuggingFace dataset from disk."""
    if not HF_DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {HF_DATASET_PATH}. "
                                "Run training/convert_to_hf_dataset.py first.")

    log.info(f"Loading dataset from {HF_DATASET_PATH}")
    dataset = load_from_disk(str(HF_DATASET_PATH))

    log.info(f"Dataset loaded: {len(dataset)} examples")
    return dataset


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize dataset examples."""
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


def prepare_dataset(dataset: dict, tokenizer, max_length: int) -> dict:
    """Prepare dataset for training."""
    log.info("Tokenizing dataset...")

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    return tokenized_dataset


def main() -> None:
    setup_logging()

    log.info("=== LoRA TRAINING START ===")

    # Load configuration
    config = load_config()
    log.info(f"Configuration loaded from {CONFIG_PATH}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Setup LoRA
    model = setup_lora(model, config)

    # Load and prepare dataset
    dataset = load_dataset()
    max_length = config.get("max_length", 4096)
    tokenized_dataset = prepare_dataset(dataset, tokenizer, max_length)

    # Split into train/val if not already split
    if "train" not in tokenized_dataset:
        split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset.get("test", tokenized_dataset["validation"])

    log.info(f"Train dataset size: {len(train_dataset)}")
    log.info(f"Eval dataset size: {len(eval_dataset)}")

    # Training arguments
    training_config = config.get("training", {})
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=training_config.get("epochs", 1),
        per_device_train_batch_size=training_config.get("batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", 2e-5),
        warmup_steps=100,
        logging_steps=training_config.get("logging_steps", 5),
        save_steps=training_config.get("save_steps", 100),
        save_total_limit=2,
        bf16=True,
        logging_dir=str(LOGS_DIR),
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    log.info(f"Training args: {training_args}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train
    log.info("Starting training...")
    trainer.train()

    # Save final model
    log.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    log.info("=== LoRA TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()