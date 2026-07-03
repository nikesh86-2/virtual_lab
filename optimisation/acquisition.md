# Acquisition Function (Active Learning Strategy)

## Overview

Selects sequences for real MD evaluation using uncertainty.

---

## Method

Uses **Upper Confidence Bound (UCB)**:
score = prediction + β × uncertainty

---

## Behaviour

- High prediction → exploitation
- High uncertainty → exploration

---

## Parameters

- β controls exploration level
    - low β → exploit
    - high β → explore

---

## Workflow

1. Evaluate all candidates using surrogate
2. Rank using acquisition score
3. Select top K
4. Run MD only on these

---

## Role
Surrogate → acquisition → real experiments

---

## Key Insight

This component ensures:

> efficient use of expensive simulations