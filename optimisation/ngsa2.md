# NSGA-II Optimisation (Surrogate + Active Learning)

## Overview

Performs multi-objective evolutionary optimisation for RNA design.

---

## Objectives

- Thermodynamic stability
- Structural integrity
- Motif presence
- Binding relevance
- Kinetic behaviour

---

## Key Improvements

### 1. No MD inside loop
→ uses surrogate instead

### 2. Caching
→ avoids repeated evaluations

### 3. Active learning
→ selectively runs MD

---

## Workflow

Initial population
↓
Evaluate via surrogate
↓
Pareto selection
↓
Select candidates via acquisition
↓
Run real MD (few samples)
↓
Update surrogate

---

## Performance

| Component | Cost |
|----------|------|
| SFold | cheap |
| surrogate | very cheap |
| MD | expensive (limited) |

---

## Benefits

- 10–50x faster optimisation
- improved exploration
- reduced compute cost

---

## Key Insight

NSGA is no longer brute-force:

> it is guided by a learned model of physics