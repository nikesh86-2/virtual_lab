# PI Agent (Adaptive Evolution Controller)

## Overview

The PI agent runs **multi-objective optimisation** using NSGA-II
guided by a neural surrogate model.

---

## Responsibilities

- Run NSGA-II optimisation
- Use surrogate model for fast evaluation
- Select Pareto-optimal sequences
- Apply conservation constraints
- Log optimisation behaviour

---

## Optimisation Pipeline
Generate population
↓
Evaluate via surrogate
↓
Pareto front extraction
↓
Select sequences
---

## Surrogate Integration

Instead of running MD:
NSGA uses:
surrogate prediction + uncertainty


---

## Adaptive Mutation

Stores:
state["mutation_bias"]

Used to bias:
- motif insertion
- nucleotide selection
- structure preference

---

## Pareto Strategy

Selects:
- best overall solution
- extreme trade-off solutions

---

## Conservation Filtering

High conservation:
→ reduces exploration space
→ focuses search

---

## Failure Handling

Fallback:
- uses previous best sequence
- avoids pipeline collapse

---

## Key Insight

The PI agent implements:

> evolutionary optimisation with learned physics approximation