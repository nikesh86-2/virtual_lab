# Research Agent (Adaptive Signal Engine)

## Overview

Transforms scientific literature into optimisation guidance.

This module connects:
-literature → signals → optimisation weights

---

## Responsibilities

- Retrieve papers (PubMed + Semantic Scholar)
- Extract scientific signals
- Build multi-objective weights
- Guide optimisation behaviour

---

## Key Outputs

### Objective Weights
{
thermo,
structure,
motif,
binding,
kinetic
}
These weights control NSGA-II objective priorities.

---

## Signal Extraction

Identifies:

- Thermodynamic stability
- Secondary structure motifs
- Sequence motifs (RGAG/RAAG)
- Binding relevance
- Kinetics

---

## Adaptive Learning

### `update_weights()`

- learns from top-performing sequences
- dynamically rebalances objectives
- prevents objective collapse

---

## Mutation Guidance

### `mutation_bias()`

Provides:

- motif insertion probability
- structure bias
- GC bias

Used directly in NSGA mutation

---

## Role in System
Literature
↓
Signal extraction
↓
Objective weighting
↓
NSGA-II behaviour

---

## Key Insight

This agent turns:

> biological knowledge into mathematical optimisation signals