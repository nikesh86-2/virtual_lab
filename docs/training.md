# Training Documentation

## Overview

Training examples and adapter fine-tuning for the BioAgent platform.

### Data Extraction
- `scripts/extract_training_examples.py` extracts compact JSONL examples from lab run artifacts and result JSONs
- `data/training_schema.md` documents the JSON fields and recommended practices
- `examples.jsonl` is the output of the extractor and can be used for:
  - Building a RAG store (vector DB + retriever)
  - Adapter/LoRA fine-tuning for domain adaptation

## Quick Extraction

```bash
python3 scripts/extract_training_examples.py --checkpoint lab_checkpoint.json --out examples.jsonl
```

## Low-cost Adapter Fine-tuning (LoRA)

### 1. Prepare Dataset

Prepare a small dataset in JSONL where each line is `{"input":..., "output":...}` or adapt `examples.jsonl` to a supervised format.

### 2. Install Dependencies

```bash
pip install transformers peft datasets accelerate
```

### 3. Example Training Script

```python
# pseudocode: fine-tune a small LLM using PEFT/LoRA on examples.jsonl
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ds = load_dataset('json', data_files='examples.jsonl')['train']
# convert ds to {'input_text':..., 'target_text':...} as needed

model_name = 'meta-llama/Llama-2-7b'  # replace with an accessible model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

config = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj','v_proj'], lora_dropout=0.1)
peft_model = get_peft_model(model, config)

# then use Trainer or custom training loop to fine-tune on the prepared dataset
```

## Recommended Pipeline

- Start with RAG + retrieval from `examples.jsonl` and papers index for immediate gains
- Use small adapter updates (LoRA) frequently on curated high-quality examples to avoid catastrophic forgetting
- Maintain a held-out validation set and monitor factuality, task success, and human evaluation
- For large-scale updates, use a proper training cluster with checkpointing and mixed precision

## Safety & Curation

- Curate examples to remove low-quality or incorrect outputs before training
- Keep provenance (`source_files`) so humans can audit training items

## Additional Resources

For detailed LoRA fine-tuning instructions, see [`docs/lora_finetuning.md`](lora_finetuning.md).

---
*Last updated: 2026-07-15*
