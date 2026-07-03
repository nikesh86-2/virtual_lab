
# Training Pipeline

## Overview

Transforms experiment logs into training data.

---

## Pipeline

1. Collect data (JSONL)
2. Validate + clean dataset
3. Convert to training format
4. Train LoRA model
5. Reload model

---

## Components

### Data Collector
Captures:
- agent reasoning
- optimisation decisions

---

### Validator
Filters:
- low-quality outputs
- errors
- duplicates

---

### Training
Uses:
- LoRA fine-tuning
- HuggingFace datasets

---

## Goal

Enable:
run → learn → improve → rerun


---

## Key Insight

This enables:

> a self-improving scientific system