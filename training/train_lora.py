"""
train_lora.py

Fine-tunes your model using LoRA on your lab-generated dataset
"""

import torch
import yaml
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


# ------------------------------------------------------------
# LOAD CONFIG
# ------------------------------------------------------------
with open("training/training_config.yaml") as f:
    config = yaml.safe_load(f)

model_name = config["model_name"]
output_dir = config["output_dir"]

train_config = config["training"]
lora_config = config["lora"]
max_length = config["max_length"]

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)


# ------------------------------------------------------------
# APPLY LORA
# ------------------------------------------------------------
peft_config = LoraConfig(
    r=lora_config["r"],
    lora_alpha=lora_config["alpha"],
    lora_dropout=lora_config["dropout"],
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, peft_config)


# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
dataset = load_from_disk("training/hf_dataset")


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


dataset = dataset.map(tokenize, batched=True)


# ------------------------------------------------------------
# TRAINER
# ------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=train_config["batch_size"],
    gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
    num_train_epochs=train_config["epochs"],
    learning_rate=float(train_config["learning_rate"]),
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("\n✅ Training complete")
print(f"✅ Model saved to: {output_dir}")