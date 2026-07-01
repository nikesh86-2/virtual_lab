# LoRA Fine‑tuning Documentation

## Overview
Low‑Rank Adaptation (LoRA) is used to fine‑tune the large language model that powers the VLAB2 agents without retraining the entire model. The pipeline is located under `bot/VLAB2/finetune/` and is invoked when the environment variable `VLAB_LORA_TRAIN=1` is set.

## Workflow
1. **Data preparation**
   - Training data consists of JSONL records with `prompt` and `completion` fields.
   - The script `finetune/prepare_dataset.py` filters the agent interaction logs (e.g., `logs/*.out`) and creates `train.jsonl` and `val.jsonl`.
2. **LoRA configuration**
   - Config file `finetune/lora_config.yaml` defines:
     ```yaml
     model_name: "meta-llama/Meta-Llama-3.1-8B"
     lora_r: 8               # rank
     lora_alpha: 16
     target_modules: ["q_proj", "v_proj"]
     learning_rate: 2e-4
     epochs: 3
     batch_size: 8
     ```
3. **Training**
   - Executed via the helper script `finetune/train_lora.sh` which runs:
     ```bash
     accelerate launch --config_file accelerate_config.yaml \
       finetune/train.py --config finetune/lora_config.yaml
     ```
   - The script logs progress to `finetune/logs/`.
4. **Merging & Deployment**
   - After training, `finetune/merge_lora.sh` merges the LoRA weights into the base model and writes the merged checkpoint to `${BASE_DIR}/models/merged_lora/`.
   - `run_virtual_lab_conda.sh` can point the agents to the merged model by setting `VLAB_MODEL_PATH`.

## Environment Variables
| Variable | Default | Description |
|---|---|---|
| `VLAB_LORA_TRAIN` | `0` | Enable LoRA fine‑tuning when set to `1`. |
| `VLAB_LORA_EPOCHS` | `3` | Number of training epochs. |
| `VLAB_LORA_RANK` | `8` | LoRA rank (`lora_r`). |
| `VLAB_LORA_ALPHA` | `16` | LoRA scaling factor. |

## Files
- `finetune/prepare_dataset.py` – extracts logs and creates JSONL files.
- `finetune/lora_config.yaml` – hyper‑parameter configuration.
- `finetune/train_lora.sh` – wrapper script for training.
- `finetune/merge_lora.sh` – merges LoRA weights into the base model.
- `run_virtual_lab_conda.sh` – exports the above environment variables.

---
*Last updated: 2026‑07‑01*