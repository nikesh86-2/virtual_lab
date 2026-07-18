# Training Pipeline Documentation

## Overview

The **Training Pipeline** enables VLAB2 to learn from its own research cycles through closed-loop instruction tuning. The system automatically collects high-quality training data from agent reasoning, optimisation decisions, and simulation outcomes, then uses this data for LoRA fine-tuning of the base LLM.

## Two Learning Loops

### 1. Surrogate Learning (Fast Loop)

**Purpose**: Improve the neural surrogate model used by NSGA-II optimisation

**Flow**:
```
MD Simulation → Surrogate Model Training → NSGA-II Improvement
```

- The `SimpleGP` surrogate model learns from evaluated RNA sequences
- After each generation, the surrogate is retrained on all evaluated candidates
- Acquisition function balances exploitation (known good sequences) and exploration (uncertain regions)
- Active learning: Top candidates are validated with real MD simulations to update the surrogate

**Location**: `optimisation/optimisation_engine.py`

### 2. LLM Fine-tuning (Slow Loop)

**Purpose**: Improve the LLM's reasoning patterns and scientific judgment

**Flow**:
```
Agent Logs → Dataset Construction → LoRA Training → Model Update
```

- Every research cycle produces instruction-tuning examples
- Post-run pipeline converts raw results into supervised learning format
- LoRA fine-tuning adapts the base model without catastrophic forgetting
- Model versions are tracked and managed

**Location**: `training/`

## Data Collection

### TrainingDataCollector

Located in: `core/training_data_collector.py`

The `TrainingDataCollector` class captures training examples throughout the research pipeline:

```python
class TrainingDataCollector:
    def __init__(self, output_dir="training_data", prefix="biophysics_tuning")
    
    def collect_cycle(topic, hypothesis, evidence, critique, final_synthesis)
    def capture_agent_interaction(agent_name, task, response, input_context)
    def capture_optimisation_step(population_summary, selected_sequences, mutation_bias)
    def capture_mutation_pattern(bias, iteration)
    def summarise_population(population, decode_fn)
```

### Data Types Captured

#### 1. Research Cycles
```json
{
  "instruction": "Evaluate a biophysics research cycle.",
  "input": "{\"topic\": \"...\", \"hypothesis\": \"...\", \"evidence\": [...], \"critique\": \"...\"}",
  "output": "Final synthesis text...",
  "metadata": {
    "type": "research_cycle",
    "session_id": "20260701_120000",
    "timestamp": "2026-07-01T12:00:00Z"
  }
}
```

#### 2. Agent Reasoning
```json
{
  "instruction": "Act as PI Agent. Task: Optimise RNA sequences",
  "input": "Context: target=6M71, sequences=[...], critique=...",
  "output": "{\"selected_sequences\": [...], \"mutation_bias\": {...}}",
  "metadata": {
    "type": "agent_reasoning",
    "agent": "pi_agent",
    "session_id": "20260701_120000"
  }
}
```

#### 3. Optimisation Steps
```json
{
  "instruction": "Analyse optimisation behaviour and sequence selection.",
  "input": "Population summary text...",
  "output": "{\"selected_sequences\": [...], \"mutation_bias\": {...}}",
  "metadata": {
    "type": "optimisation_step",
    "session_id": "20260701_120000"
  }
}
```

#### 4. Mutation Patterns
```json
{
  "instruction": "Learn mutation bias patterns",
  "input": "{\"thermo\": 0.8, \"structure\": 0.6, ...}",
  "output": "Iteration 3",
  "metadata": {
    "type": "mutation_pattern",
    "session_id": "20260701_120000"
  }
}
```

## Dataset Construction

### Step 1: Collect JSONL Data

Training data is automatically collected during runs and saved as JSONL files:
```
training_data/biophysics_tuning_20260701_120000.jsonl
```

### Step 2: Build Preference Dataset

Script: `training/build_preference_dataset.py`

Converts raw JSONL into preference pairs for DPO (Direct Preference Optimization):
- Positive examples: Successful research cycles
- Negative examples: Failed or low-quality cycles

### Step 3: Convert to HuggingFace Dataset

Script: `training/convert_to_hf_dataset.py`

```bash
python training/convert_to_hf_dataset.py \
    --input training_data/biophysics_tuning_*.jsonl \
    --output training/hf_dataset
```

Creates a HuggingFace dataset with `train` and `test` splits.

### Step 4: Validate Dataset

Script: `training/validate_dataset.py`

Checks:
- Minimum row count (`VLAB_MIN_TRAIN_ROWS`, default: 20)
- Minimum file size (`VLAB_MIN_TRAIN_BYTES`, default: 5000)
- Data format validity
- Field completeness

## LoRA Fine-tuning

### Configuration

Located in: `training/training_config.yaml`

```yaml
model_name: /path/to/Qwen2.5-32B-Instruct
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

### Training Script

Located in: `training/train_lora.py`

**Key steps**:
1. Load base model and tokenizer
2. Apply LoRA configuration
3. Load and tokenize dataset
4. Train with `Trainer` API
5. Save adapter weights

```bash
PY=/path/to/python python training/train_lora.py
```

### Training Arguments

```python
TrainingArguments(
    output_dir="training/output_model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=5,
    save_steps=100,
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)
```

### LoRA Configuration

```python
LoraConfig(
    r=16,                      # Rank of update matrices
    lora_alpha=32,             # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

## Model Management

### Model Version Tracker

Located in: `training/model_version_tracker.py`

Tracks trained model versions and manages rotation:
- Stores metadata for each trained adapter
- Enforces maximum version count (`VLAB_MAX_MODEL_VERSIONS`, default: 3)
- Supports automatic model replacement (`VLAB_AUTO_MODEL_REPLACE`)

### Merge LoRA Weights

Script: `training/merge_lora.py`

Merges LoRA adapter weights into base model for deployment:
```bash
python training/merge_lora.py \
    --base-model /path/to/base \
    --lora-adapter training/output_model \
    --output training/merged_model
```

### Reload Model

Script: `training/reload_model.sh`

Restores trained model to vLLM serving:
```bash
bash training/reload_model.sh
```

## Evaluation

### Evaluate Model

Script: `training/evaluate_model.py`

Evaluates trained model on held-out test set:
- Perplexity score
- Task-specific metrics
- Comparison with base model

```bash
python training/evaluate_model.py
```

## Training Controls

Environment variables control training behavior:

| Variable | Default | Description |
|---|---|---|
| `VLAB_LORA_TRAIN` | `0` | Enable LoRA training (set to `1` to train) |
| `VLAB_MIN_TRAIN_ROWS` | `20` | Minimum training examples required |
| `VLAB_MIN_TRAIN_BYTES` | `5000` | Minimum dataset size in bytes |
| `VLAB_MIN_INTERFACE_VALID` | `5` | Minimum interface-valid examples |
| `VLAB_MIN_LITERATURE_EVIDENCE` | `2` | Minimum literature evidence examples |
| `VLAB_AUTO_MODEL_REPLACE` | `0` | Automatically replace vLLM model after training |
| `VLAB_MAX_MODEL_VERSIONS` | `3` | Maximum stored model versions |

## Post-Run Pipeline

Located in: `orchestration/postrun.py`

Automatically runs after each Virtual Lab execution:

```python
run_postrun_pipeline(topic, final_result)
```

**Steps**:
1. **Bootstrap knowledge base** — Load literature and failure memory
2. **Build training data** — Convert results to instruction-tuning format
3. **Validate dataset** — Check minimum requirements
4. **Trigger training** (if enabled) — Run LoRA fine-tuning
5. **Update model** (if enabled) — Replace vLLM model with trained adapter

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Run (vLLM)                      │
│                                                             │
│  PI → Researcher → Bioinfo → Structural → MD → Protein    │
│                                              ↓             │
│                              Inhibitor (conditional)        │
│                                              ↓             │
│                                          Skeptic            │
└─────────────────────────────────────────────────────────────┘
            │                                         │
            │                                         │
            ▼                                         ▼
┌──────────────────────┐              ┌──────────────────────────────┐
│ TrainingDataCollector│              │  Post-Run Pipeline           │
│                      │              │                              │
│ - Research cycles    │              │ 1. Build training data       │
│ - Agent interactions │              │ 2. Validate dataset          │
│ - Optimisation steps │              │ 3. Trigger LoRA training     │
│ - Mutation patterns  │              │ 4. Update model              │
└──────────────────────┘              └──────────────────────────────┘
            │                                         │
            ▼                                         ▼
┌──────────────────────┐              ┌──────────────────────────────┐
│  training_data/*.jsonl│              │  training/output_model/      │
│                      │              │  (LoRA adapter weights)      │
│ - biophysics_tuning_ │              │                              │
│   20260701_*.jsonl   │              │  - adapter_config.json       │
└──────────────────────┘              │  - pytorch_model.bin         │
                                      └──────────────────────────────┘
                                              │
                                              ▼
                                      ┌──────────────────────┐
                                      │  Merged Model (optional)│
                                      │  training/merged_model/ │
                                      └──────────────────────┘
```

## Key Design Principles

1. **Automatic collection** — No manual data curation required; data is captured during normal operation
2. **Quality filtering** — Training examples are filtered by quality metrics (interface validity, evidence count)
3. **Version control** — Model versions are tracked and rotated automatically
4. **Non-disruptive** — Training runs separately from research execution; doesn't block the pipeline
5. **Reproducible** — All training data includes timestamps and session IDs for auditability
6. **Modular** — Each training step can be run independently for debugging

## Troubleshooting

### Training doesn't start
- Check `VLAB_LORA_TRAIN=1`
- Verify dataset exists and meets minimum requirements
- Check `training/hf_dataset/` directory

### Out of memory during training
- Reduce `batch_size` in `training_config.yaml`
- Increase `gradient_accumulation_steps`
- Use `gradient_checkpointing=True` (already enabled)

### Model quality degradation
- Reduce `learning_rate` (try `1e-5`)
- Reduce `epochs` to 1 (already default)
- Increase `lora_dropout` to `0.1`
- Curate dataset to remove low-quality examples

---
*Last updated: 2026-07-01*