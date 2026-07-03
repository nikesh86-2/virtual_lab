# Training Pipeline (Closed-Loop Learning)

## Overview

Enables the system to learn from its own experiments.

---

## Two Learning Loops

### 1. Surrogate learning (fast loop)

MD → surrogate → NSGA improvement

---

### 2. LLM fine-tuning (slow loop)

agent logs → dataset → LoRA training

---

## Pipeline Steps

1. Collect JSONL data
2. Validate + clean dataset
3. Convert to training format
4. Train LoRA
5. Reload model

---

## Data Sources

- agent reasoning traces
- optimisation logs
- simulation outputs

---

## Goal

Transform system from:


rule-based

to:


self-improving

---

## Key Insight

The system learns:

- reasoning patterns (LLM)
- physical behaviour (surrogate)