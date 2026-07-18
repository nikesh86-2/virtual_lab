
# Structural Agent

## Overview

The Structural Agent designs RNA sequences and evaluates their folding properties. It generates candidate sequences using LLM-guided design, validates and diversifies them, runs folding simulations via SFold and ViennaRNA, optionally incorporates conservation-aware RNAalifold analysis, and selects the best-folding candidate for downstream MD and protein docking pipelines.

---

## Responsibilities

### Sequence Design

- **LLM-guided generation**: Prompts an LLM to design 3–5 diverse RNA sequences (25–40 nt) with stem-loop structures and exposed recognition motifs.
- **Motif prioritisation**: Encourages target motifs (configurable via `VLAB_RNA_TARGET_MOTIFS`) with soft preferences for GAG, AAG, CAG, GUC, AGU.
- **Diversity enforcement**: Ensures candidates differ by at least 5 positions and use varied motif types, stem lengths, and loop sizes.
- **Skeptic integration**: Incorporates redesign recommendations from the Skeptic Agent when available.

### Sequence Validation & Normalisation

- **Cleaning and deduplication**: Removes invalid characters, normalises T→U, and deduplicates sequences.
- **Duplicate avoidance**: Rejects sequences too similar (Hamming distance ≤ 4) to prior or existing candidates.
- **Motif enforcement**: Optionally requires target motifs (`VLAB_RNA_REQUIRE_TARGET_MOTIF=1`).
- **Conserved region locking**: Respects `conserved_regions` to preserve critical nucleotide positions.
- **Interface-aware anchoring**: Incorporates known clean-interface sequences as anchors and generates local variants.
- **Fallback generation**: Uses grammar-based structured RNA generation when LLM output is insufficient.

### Folding Simulation

- **SFold analysis**: Runs stochastic folding simulations for each candidate.
- **ViennaRNA analysis**: Runs deterministic folding via ViennaRNA package.
- **RNAalifold integration**: Runs consensus folding when multiple sequence alignment (MSA) data is available.
- **Conservation-aware gating**: Treats RNAalifold as advisory when conservation fitness is low (`VLAB_ALIFOLD_MIN_CONSERVATION_FITNESS=0.5`).

### Candidate Selection

- **Fold quality ranking**: Scores candidates using a composite metric: `0.45 × pair_density + 0.35 × |mfe_per_nt| + 0.20 × pass_bonus`.
- **Interface-aware ordering**: Prioritises clean-interface anchors, then local variants, then exploratory candidates.
- **Best selection**: Selects the top-ranked candidate as `target_sequence` for downstream processing.
- **Candidate retention**: Returns up to 5 candidates for PI/NSGA-II optimisation seeding.

---

## Key Metrics

| Metric | Description |
|---|---|
| Pair density | Fraction of nucleotides in base pairs (threshold ≥ 0.24) |
| MFE per nucleotide | Minimum free energy normalised by sequence length (threshold ≤ -0.15 kcal/mol/nt) |
| Fold pass/fail | Boolean indicating whether all fold thresholds are met |
| Rank score | Composite score: `0.45 × pd + 0.35 × mfe_score + 0.20 × pass_bonus` |
| MSA conservation fitness | Quality measure for multiple sequence alignment data |
| Interface evidence | Known interface validation status from protein agent results |

---

## Outputs

- **Folding stability**: LLM-interpreted stability assessment (STABLE/UNSTABLE/MARGINAL) with MFE, pair density, and ensemble diversity values.
- **Structural plausibility**: Motif presence/absence, conservation likelihood, and RNAalifold support status.
- **Sequence candidates**: Up to 5 ranked RNA sequences for downstream PI/NSGA-II optimisation.
- **Target sequence**: Best-folding candidate selected for MD simulation and protein docking.
- **Structural analysis**: Comprehensive interpretation including all candidate summaries and raw folding outputs.
- **Fold quality metadata**: Detailed threshold pass/fail information with reasons.

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `VLAB_RNA_TARGET_MOTIFS` | `GAG:0.55,AAG:0.55,CAG:0.35,GUC:0.35,AGU:0.35` | Comma-separated motif definitions with optional weights |
| `VLAB_RNA_REQUIRE_TARGET_MOTIF` | `0` | Whether motif presence is mandatory |
| `VLAB_ALIFOLD_MIN_CONSERVATION_FITNESS` | `0.5` | Minimum conservation fitness for RNAalifold hard gating |
| `VLAB_ALIFOLD_ADVISORY_WHEN_LOW_CONSERVATION` | `1` | Treat RNAalifold as advisory when conservation is weak |

---

## Selection Policy

```
Anchor (clean interface) → Local Variants → Exploration → All Others
```

1. **Anchor**: Known clean-interface sequence (if available and passing fold thresholds).
2. **Local variants**: Sequences within Hamming distance ≤ 2 of the anchor.
3. **Exploratory**: Other passing candidates ranked by fold quality.
4. **All others**: Remaining candidates in ranked order.

---

## Role in the Pipeline

The Structural Agent serves as the **sequence design and folding validation** engine. It:

- Translates biological hypotheses into concrete RNA sequence candidates.
- Ensures structural plausibility through multi-tool folding analysis.
- Incorporates conservation data when available for enhanced confidence.
- Feeds selected sequences to MD simulation and protein docking pipelines.
- Maintains diversity to enable robust optimisation in downstream stages.

---

## Integration Points

- **Inputs**: Hypothesis/research topic, prior sequences, skeptic critique, MSA data, conserved regions.
- **Outputs**: Designed sequences, target sequence, fold quality, structural analysis.
- **Downstream**: MD Agent receives target sequence for simulation; Protein Agent receives candidates for docking.
- **Upstream**: Receives redesign guidance from the Skeptic Agent in iterative loops.