# Virtual Lab Orchestrator

## Overview

The Virtual Lab Orchestrator coordinates a **multi-agent, closed-loop scientific workflow** for RNA design and evaluation.

This system integrates:
- hypothesis-driven design (LLM)
- biological simulation
- evolutionary optimisation (NSGA-II)
- active learning via neural surrogate models

---

## Core Innovation

The system is a **closed-loop active learning pipeline**:

- Design → Simulate → Optimise → Learn surrogate → Repeat

---

## Agent Graph

PI → {Researcher, Bioinfo, Structural}  
Structural → {MD, Protein}  
All → Skeptic → loop or END  

---

## Responsibilities

- Maintain global state (`LabState`)
- Execute LangGraph workflow
- Control iteration loop
- Coordinate optimisation cycles
- Persist checkpoints
- Trigger learning pipeline

---

## Execution Flow

1. PI runs NSGA-II optimisation (surrogate-guided)
2. Researcher gathers literature
3. Bioinfo applies evolutionary constraints
4. Structural agent designs RNA
5. MD agent performs dynamic validation
6. Protein agent evaluates binding
7. Skeptic critiques results
8. Loop or terminate

---

## Active Learning Integration

- NSGA uses **surrogate model instead of MD**
- Only top/uncertain sequences run real MD
- Surrogate is updated with new physics data

---

## Outputs

- JSON state output
- Markdown research report
- Training data logs
- Surrogate training data

---

## Key Insight

This is not a pipeline — it is:

> a **self-improving scientific optimisation system**