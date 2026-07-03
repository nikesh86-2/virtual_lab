# LoRA Fine‑tuning Documentation

## Overview
Low‑Rank Adaptation (LoRA) is used to fine‑tune the large language model that powers the VLAB2 agents without retraining the entire model. The pipeline is located under `bot/VLAB2/training/` and can be invoked automatically when the environment variable `VLAB_LORA_TRAIN=1` is set.

## Workflow
1. **Data preparation**
   - Training data is automatically collected during VLAB2 runs via `orchestration/postrun.py`
   - Data is extracted to `postrun_training_data/` with files:
     - `examples.jsonl` - general training examples
     - `supervised.jsonl` - supervised learning examples
     - `preferences.jsonl` - preference pairs for RLHF-style training
     - `docking_rows.jsonl` - docking results with interface metrics
     - `literature_evidence.jsonl` - literature search results

2. **Data validation**
   - `training/validate_dataset.py` validates and cleans training data
   - Checks for minimum quality thresholds (configurable via environment variables)
   - Outputs `training/clean_dataset.jsonl`

3. **Dataset conversion**
   - `training/convert_to_hf_dataset.py` converts JSONL to HuggingFace dataset format
   - Applies chat template formatting for instruction tuning
   - Outputs `training/hf_dataset/`

4. **LoRA training**
   - `training/train_lora.py` performs actual LoRA fine-tuning using transformers/peft
   - Configuration from `training/training_config.yaml`
   - Uses accelerate for multi-GPU training (config in `training/accelerate_config.yaml`)
   - Logs to `training/logs/training.log`
   - Outputs LoRA adapter to `training/output_model/`

5. **Model evaluation**
   - `training/evaluate_model.py` tests the trained model with sample prompts
   - Validates that the model produces reasonable outputs

6. **Model merging**
   - `training/merge_lora.py` merges LoRA adapter into base model
   - Registers model version in `training/merged_model_versions/`
   - Sets active model in `training/active_model.json`
   - Outputs merged model to `training/merged_model/`

7. **Model deployment**
   - When `VLAB_AUTO_MODEL_REPLACE=1`, the system can automatically switch to the fine-tuned model
   - Requires updating `VLLM_MODEL_PATH` and restarting vLLM
   - Model versioning provides rollback capability via `training/model_version_tracker.py`

## Configuration

### Training Config (`training/training_config.yaml`)
```yaml
model_name: /path/to/base/model
output_dir: training/output_model
max_length: 4096

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  epochs: 1
  learning_rate: 2.0e-5
  save_steps: 100
  logging_steps: 5

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
```

### Accelerate Config (`training/accelerate_config.yaml`)
- Multi-GPU configuration for 2 GPUs
- Mixed precision (bf16)
- Tensor parallelism for efficient training

## Environment Variables
| Variable | Default | Description |
|---|---|---|
| `VLAB_LORA_TRAIN` | `0` | Enable automatic LoRA fine‑tuning when set to `1`. |
| `VLAB_MIN_TRAIN_ROWS` | `20` | Minimum number of training examples required. |
| `VLAB_MIN_TRAIN_BYTES` | `5000` | Minimum dataset size in bytes. |
| `VLAB_MIN_INTERFACE_VALID` | `5` | Minimum interface-valid docking results required. |
| `VLAB_MIN_LITERATURE_EVIDENCE` | `2` | Minimum literature evidence records required. |
| `VLAB_AUTO_MODEL_REPLACE` | `0` | Enable automatic model replacement after training. |
| `VLAB_MAX_MODEL_VERSIONS` | `3` | Maximum number of model versions to keep. |

## Automatic Training Pipeline
When `VLAB_LORA_TRAIN=1`, the system automatically:
1. Collects training data after each VLAB2 run
2. Checks data quality against thresholds
3. Triggers training if quality is sufficient
4. Evaluates the trained model
5. Merges LoRA weights if evaluation passes
6. Optionally replaces the active model

Training runs asynchronously in the background to avoid blocking the main pipeline.

## Manual Training
To manually trigger training:
```bash
PY=/path/to/python python training/auto_retrain.py
```

This runs the full pipeline: validation → conversion → training → evaluation → merge → versioning.

## Model Version Management
```bash
# List all model versions
python training/model_version_tracker.py list

# Show active model
python training/model_version_tracker.py active

# Rollback to a specific version
python training/model_version_tracker.py rollback <version_id>

# Cleanup old versions (keep max N)
python training/model_version_tracker.py cleanup [max_versions]
```

## Files
- `training/train_lora.py` – actual LoRA training implementation
- `training/merge_lora.py` – merges LoRA weights into base model
- `training/auto_retrain.py` – automated training pipeline
- `training/validate_dataset.py` – validates and cleans training data
- `training/convert_to_hf_dataset.py` – converts to HF dataset format
- `training/evaluate_model.py` – evaluates trained model
- `training/model_version_tracker.py` – model versioning and rollback
- `training/training_config.yaml` – training hyperparameters
- `training/accelerate_config.yaml` – accelerate multi-GPU config
- `orchestration/postrun.py` – automatic training trigger
- `run_virtual_lab_conda.sh` – exports training environment variables

## Troubleshooting
- **Training not triggering**: Check `VLAB_LORA_TRAIN=1` and data quality thresholds
- **Out of memory**: Reduce `batch_size` or increase `gradient_accumulation_steps`
- **Poor model quality**: Increase training epochs or collect more high-quality data
- **Model replacement failing**: Check `VLAB_AUTO_MODEL_REPLACE=1` and vLLM configuration

---
*Last updated: 2026‑07‑01*