# Neural Surrogate Model (Active Learning Core)

## Overview

Approximates MD simulation using a neural network.

Replaces expensive physics simulation in inner optimisation loop.

---

## Input Features

- SFold pair density
- GC content
- sequence length (normalised)
- motif signal

---

## Output

- predicted MD stability
- prediction uncertainty (via MC dropout)

---

## Uncertainty Estimation

Uses **Monte Carlo Dropout**:

- multiple forward passes
- variance → uncertainty estimate

---

## Active Learning Loop
Predict → Measure uncertainty → Select samples → Run MD → Train

---

## Training

- accumulates real MD results
- trains via MSE loss
- improves over time

---

## Behaviour

Early:
- high uncertainty
- fallback to heuristics

Later:
- accurate predictions
- reduced MD calls

---

## Role in System
SFold → surrogate → NSGA decision

---

## Key Insight

The surrogate learns:

> an approximation of molecular dynamics from data