"""
evaluate_model.py

Simple prompt test for trained LoRA model.

If output_model is a LoRA adapter, load base model then adapter.
"""

from __future__ import annotations

import yaml
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


CONFIG_PATH = "training/training_config.yaml"


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_model = config["model_name"]
    adapter_path = config["output_dir"]

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    prompt = (
        "<|im_start|>system\n"
        "You are a computational biophysics assistant. Use HDOCK scores only as relative scores.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Summarise this result: target_pdb=None, dock_valid=2, interface_pass=1, "
        "one pose clashed, one clean pose. What should the next PI action be?\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
        )

    print(tokenizer.decode(out[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()