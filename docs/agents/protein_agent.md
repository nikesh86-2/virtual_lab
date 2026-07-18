# Protein Agent

## Overview

The Protein Agent evaluates RNA–protein binding interactions using HDOCK docking simulations. It dynamically selects protein targets from the RCSB PDB, runs docking calculations, validates interfaces, and ranks candidate sequences based on binding affinity and structural plausibility.
---

## Responsibilities

### Target Selection

- **Dynamic PDB selection**: Queries RCSB PDB for relevant viral RNA-binding proteins, nucleocapsid proteins, or RNA-protein complexes.
- **Target memory reuse**: Reuses partial-success targets and avoids hard-failed targets using `FailureMemory`.
- **Fallback strategies**: Supports LLM-generated candidates, RCSB dynamic ranking, and environment-based fallback PDBs.
- **Blacklist enforcement**: Respects `VLAB_TARGET_BLACKLIST` and hard/soft failure records.

### Docking & Binding Evaluation

- **Sequence collection**: Prioritises NSGA-promoted sequences, best clean-interface sequences, and designed candidates.
- **Fold gating**: Skips docking if RNA fold thresholds fail (configurable via `VLAB_RNA_ENFORCE_MIN_FOLD`).
- **HDOCK execution**: Prepares receptor and RNA ligand PDBs, runs HDOCK docking simulations.
- **Score validation**: Rejects implausible HDOCK scores (e.g., ~-1000) that indicate preparation artefacts.
- **Interface analysis**: Computes interface contacts, steric clash detection, and signed interface quality scores.

### Ranking & Output

- **Binding rank scoring**: Combines HDOCK scores with proxy scores using a weighted formula: `0.7 × dock_score + 0.3 × proxy_dg`.
- **Interface-aware ranking**: Prefers clean-interface poses over those with steric clashes, even if raw scores differ.
- **Target status classification**:
  - `accepted_target`: Sufficient clean-interface poses meet all thresholds.
  - `partial_success_target`: Docking-valid but insufficient clean interfaces — reusable as lower-confidence target.
  - `failed_target`: No usable docking results or interface validation failed.
---

## Key Metrics

| Metric | Description |
|---|---|
| Docking-valid count | Number of sequences with valid HDOCK docking scores |
| Interface-clean count | Sequences with clean interfaces (no steric clashes) |
| Steric clash count | Sequences rejected due to steric clashes at the interface |
| Best HDOCK relative score | Lowest (most favourable) HDOCK docking score |
| Score spread | Variance in docking scores across valid results |
| Binding rank score | Composite score: `0.7 × dock_score + 0.3 × proxy_dg` |
---

## Outputs

- **ΔG binding energies**: HDOCK relative docking scores (not physical kcal/mol free energies).
- **Valid sequence rankings**: Ranked list of sequences by binding affinity with interface validation status.
- **Selected protein target**: Best PDB target with `accepted_target`, `partial_success_target`, or `failed_target` status.
- **Interface contact metrics**: Residue-level interface contacts, clash severity, and quality scores.
- **Docking preferences**: Training data examples preferring clean-interface poses over clashing ones.
- **Partial success records**: Structured records for reusable lower-confidence targets.

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `VLAB_AGENT_EVAL_TOP_N` | `3` | Number of RNA sequences to evaluate per target (bounded 2–3) |
| `VLAB_MAX_DOCKINGS_PER_TARGET` | `eval_top_n` | Maximum docking attempts per target |
| `VLAB_MIN_VALID_HDOCK_SCORE` | `-30` | Minimum acceptable HDOCK score threshold |
| `VLAB_MAX_TARGET_SCORE_SPREAD` | `25` | Maximum allowable score spread before target rejection |
| `VLAB_REQUIRE_DOCKING` | `1` | Whether docking is required (0 = proxy-only allowed) |
| `VLAB_RNA_ENFORCE_MIN_FOLD` | `1` | Whether to block docking if RNA fold thresholds fail |
| `VLAB_ALLOW_PARTIAL_INTERFACE_TARGET` | `true` | Allow partial-success target classification |
| `VLAB_TARGET_BLACKLIST` | `""` | Comma-separated list of excluded PDB IDs |
| `VLAB_ALLOW_GENERIC_TARGET_FALLBACK` | `false` | Enable generic fallback PDBs from `VLAB_FALLBACK_PDBS` |
| `VLAB_MAX_RECEPTOR_ATOMS` | `60000` | Maximum receptor PDB atom count |

---

## Target Status Workflow
- **HDOCK execution**: Prepares receptor and RNA ligand PDBs, runs HDOCK docking simulations.
- **Score validation**: Rejects implausible HDOCK scores (e.g., ~-1000) that indicate preparation artefacts.
- **Interface analysis**: Computes interface contacts, steric clash detection, and signed interface quality scores.

### Ranking & Output

- **Binding rank scoring**: Combines HDOCK scores with proxy scores using a weighted formula: `0.7 × dock_score + 0.3 × proxy_dg`.
- **Interface-aware ranking**: Prefers clean-interface poses over those with steric clashes, even if raw scores differ.
- **Target status classification**:
  - `accepted_target`: Sufficient clean-interface poses meet all thresholds.
  - `partial_success_target`: Docking-valid but insufficient clean interfaces — reusable as lower-confidence target.
  - `failed_target`: No usable docking results or interface validation failed.
---







## Key Metrics

| Metric | Description |
|---|---|
| Docking-valid count | Number of sequences with valid HDOCK docking scores |
| Interface-clean count | Sequences with clean interfaces (no steric clashes) |
| Steric clash count | Sequences rejected due to steric clashes at the interface |
| Best HDOCK relative score | Lowest (most favourable) HDOCK docking score |
| Score spread | Variance in docking scores across valid results |
| Binding rank score | Composite score: `0.7 × dock_score + 0.3 × proxy_dg` |
---



## Outputs

- **ΔG binding energies**: HDOCK relative docking scores (not physical kcal/mol free energies).
- **Valid sequence rankings**: Ranked list of sequences by binding affinity with interface validation status.
- **Selected protein target**: Best PDB target with `accepted_target`, `partial_success_target`, or `failed_target` status.
- **Interface contact metrics**: Residue-level interface contacts, clash severity, and quality scores.
- **Docking preferences**: Training data examples preferring clean-interface poses over clashing ones.
- **Partial success records**: Structured records for reusable lower-confidence targets.

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `VLAB_AGENT_EVAL_TOP_N` | `3` | Number of RNA sequences to evaluate per target (bounded 2–3) |
| `VLAB_MAX_DOCKINGS_PER_TARGET` | `eval_top_n` | Maximum docking attempts per target |
| `VLAB_MIN_VALID_HDOCK_SCORE` | `-30` | Minimum acceptable HDOCK score threshold |
| `VLAB_MAX_TARGET_SCORE_SPREAD` | `25` | Maximum allowable score spread before target rejection |
| `VLAB_REQUIRE_DOCKING` | `1` | Whether docking is required (0 = proxy-only allowed) |
| `VLAB_RNA_ENFORCE_MIN_FOLD` | `1` | Whether to block docking if RNA fold thresholds fail |
| `VLAB_ALLOW_PARTIAL_INTERFACE_TARGET` | `true` | Allow partial-success target classification |
| `VLAB_TARGET_BLACKLIST` | `""` | Comma-separated list of excluded PDB IDs |
| `VLAB_ALLOW_GENERIC_TARGET_FALLBACK` | `false` | Enable generic fallback PDBs from `VLAB_FALLBACK_PDBS` |
| `VLAB_MAX_RECEPTOR_ATOMS` | `60000` | Maximum receptor PDB atom count |

---

## Target Status Workflow
