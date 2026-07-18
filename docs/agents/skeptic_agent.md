# Skeptic Agent

## Overview



The Skeptic Agent performs critical evaluation of results generated across the multi-agent pipeline. It acts as a quality gate, challenging assumptions, identifying inconsistencies, and ensuring that conclusions are well-supported by evidence.
---

## Responsibilities





- **Analyze multi-modal data**: Evaluate binding energy, structural, and molecular dynamics (MD) simulation results for coherence and reliability.
- **Assess hypothesis validity**: Determine whether proposed designs or hypotheses are supported by the available data.
- **Detect convergence**: Monitor whether iterative refinements are converging toward stable, high-quality solutions or stagnating.
- **Identify artifacts and biases**: Flag potential sources of error, such as force field limitations, sampling insufficiency, or overfitting.
- **Provide constructive criticism**: Offer actionable feedback to guide the next iteration of design or analysis.
---

## Key Metrics





| Metric | Description |
|---|---|
| Binding energy spread | Variance in predicted binding energies across replicates or conformations |
| Consistency across sequences | Agreement of results when evaluated on related or mutated sequences |
| Structural plausibility | Assessment of whether predicted structures are physically realistic |
| Convergence rate | How quickly the system stabilizes across iterations |
| Confidence score | Overall certainty in the validity of the current solution |
---

## Outputs




- **Critique text**: Detailed written evaluation highlighting strengths, weaknesses, and areas for improvement.
- **Iteration summary**: A concise overview of the current state of the system, including progress and recommended next steps.
- **Red flags**: List of critical issues that must be addressed before proceeding.
---

## Workflow
