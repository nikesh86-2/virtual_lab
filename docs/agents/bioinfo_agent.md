# Bioinformatics Agent (Conservation Analysis Engine)

## Overview

The bioinformatics agent analyzes evolutionary conservation patterns in viral RNA sequences to provide conservation-aware signals for optimisation.

This agent connects:
- evolutionary data → conservation metrics → structural constraints

## System Architecture

The Bioinformatics Agent has two implementation paths:

### 1. BioinfoWrapper (Preferred Path)
Located in `core/bioinfo_wrapper.py` - Full-featured implementation with:
- NCBI Entrez genome fetching with organism inference
- MAFFT/MUSCLE/Clustal Omega MSA support
- RNAalifold integration for consensus structure prediction
- Position-level conservation scoring with entropy calculations
- Motif-level conservation analysis (RGAG, RAAG, etc.)
- NSGA-ready conservation signal output

### 2. Legacy BioinfoWrapper
Located in `wrappers/bioinfo_wrapper.py` - Simplified version with:
- Basic NCBI fetching
- MAFFT/MUSCLE/Clustal Omega MSA
- Column-based conservation analysis
- Conserved region detection

### 3. Orchestration Layer
Located in `orchestration/agents/bioinfo_agent.py` - Main agent entry point that:
- Coordinates between wrapper implementations
- Normalizes conservation signals
- Applies quality gates
- Integrates with LabState

---

## Responsibilities

### Primary Tasks
- Identify organism via NCBI Taxon ID or organism query inference
- Retrieve related genomes from NCBI Nucleotide database
- Perform Multiple Sequence Alignment (MSA) using MAFFT (primary), MUSCLE or Clustal Omega (fallbacks)
- Compute position-level conservation metrics
- Detect conserved regions and motifs
- Generate conservation fitness scores for optimisation
- Optional RNAalifold consensus structure prediction

### Genome Fetching Strategy
1. **Taxon ID Path**: Direct lookup via NCBI Taxon ID (e.g., txid12110 for HPeV1)
2. **Organism Query Path**: Infer organism name from topic text and search
3. **Local Fallback**: Use designed/target sequences if no external genomes found
4. **Target Window Extraction**: Extract regions similar to target sequence from related genomes

### MSA Execution
- **Primary**: MAFFT with `--auto` mode (auto-selects algorithm)
- **Fallback 1**: MUSCLE (`-align -output`)
- **Fallback 2**: Clustal Omega (`--outfmt=fasta --force`)
- Timeout: 180 seconds (configurable via `MAFFT_TIMEOUT`)
- Sequence filtering: min length 12-50nt, max 200 records, deduplication by identity

### Conservation Analysis
- **Position-level scores**: Entropy-based conservation per nucleotide position
- **Consensus sequence**: Most frequent base at each position
- **Conserved regions**: Windows with high conservation (default threshold: 0.85)
- **Motif conservation**: Special scoring for RGAG, RAAG, GAG, AAG, CAG, GUC, AGU motifs
- **RNAalifold**: Optional consensus structure prediction with MFE and pair density

---

## Key Outputs

### Conservation Signal
```json
{
  "valid": true,
  "consensus": "ACGUACGU...",
  "position_scores": [0.95, 0.87, 0.12, ...],
  "entropy_scores": [0.05, 0.13, 0.88, ...],
  "conserved_regions": [[0, 12], [45, 58], ...],
  "motif_scores": {"RGAG": 0.92, "RAAG": 0.85},
  "selected_motifs": [
    {
      "motif": "RGAG",
      "matched": "RGAG",
      "start": 5,
      "end": 9,
      "weight": 1.0,
      "position_conservation": 0.92,
      "motif_conservation": 0.92,
      "selection_score": 0.92
    }
  ],
  "conservation_pct": 45.5,
  "conservation_fitness": 0.67
}
```

### Scalar Conservation Fitness
- Computed from position scores: `mean(position_scores)`
- Normalised to 0.0-1.0 range
- Used by PI agent as conservation objective signal
- Penalized by 75% if quality gate fails

### MSA Data
- Full multiple sequence alignment in FASTA format
- Stored in `msa_data` field of agent output

### Analysis Summary
```
MSA: 15 seqs, 120 nt
Conservation: 45.5%
Conserved regions: 8
Selected motifs: RGAG (0.92), RAAG (0.85)
RNAalifold: valid/missing/failed
```

### Agent Output Fields
- `bioinfo_analysis`: Full JSON analysis report
- `msa_data`: MSA in FASTA format
- `conservation_signal`: Normalised conservation metrics
- `conserved_regions`: List of (start, end) tuples
- `bioinfo_num_sequences`: Number of sequences in MSA
- `bioinfo_alignment_length`: Length of alignment
- `bioinfo_quality_passed`: Boolean quality gate result
- `bioinfo_quality_reasons`: List of quality issues if failed
- `_run_system_selected_motifs`: Top conserved motifs for optimisation

---

## Conservation Metrics Explained

### Position-Level Conservation
For each position in the alignment:
1. Count base frequencies (excluding gaps)
2. Calculate Shannon entropy: `H = -Σ(p * log2(p))`
3. Normalise entropy: `norm_entropy = min(1.0, H / 2.0)`
4. Conservation score: `conservation = max(0.0, 1.0 - norm_entropy)`

Range: 0.0 (variable) to 1.0 (fully conserved)

### Conserved Regions
- Sliding window approach (default window: 6 nt)
- Minimum score threshold: 0.85 (configurable via `VLAB_CONSERVATION_REGION_MIN_SCORE`)
- Extends region while consecutive positions meet threshold
- Returns list of (start, end) tuples

### Motif Conservation
For each target motif (RGAG, RAAG, GAG, AAG, CAG, GUC, AGU):
1. Find all motif occurrences in consensus (supports IUPAC codes)
2. Calculate average position conservation within motif window
3. Apply motif-specific weight from `VLAB_RNA_TARGET_MOTIFS`
4. Combined score: `0.55 * position_conservation + 0.45 * motif_weight`

### Conservation Fitness (Optimisation Signal)
```python
conservation_fitness = mean(position_scores)
```
- Normalised to 0.0-1.0
- Penalized to 25% of original if quality gate fails
- Used as `conservation` objective in NSGA-II

---

## Quality Gates

### Minimum Requirements
Configurable via environment variables:
- `VLAB_BIOINFO_MIN_MSA_SEQUENCES`: Minimum sequences (default: 3, min: 1)
- `VLAB_BIOINFO_MIN_ALIGNMENT_LENGTH`: Minimum alignment length (default: 20, min: 1)

### Quality Failure Consequences
If quality gate fails:
- `conservation_fitness` multiplied by 0.25 (75% penalty)
- `quality_passed` set to False
- `quality_reasons` populated with failure details
- `valid` flag set to False

### Common Quality Reasons
- `too_few_sequences:2<3` - Not enough sequences in MSA
- `alignment_too_short:15<20` - Alignment shorter than minimum

---

## RNAalifold Integration

### When Used
Optional consensus structure prediction when RNAalifold binary is available.

### Configuration
- Binary path: `RNAALIFOLD_BIN` env var or `rnaalifold_bin` param
- Timeout: `RNAALIFOLD_TIMEOUT` env var (default: 120s)

### Output
```json
{
  "valid": true,
  "structure": "(((...)))....((...",
  "mfe": -45.3,
  "mfe_per_nt": -0.38,
  "pair_density": 0.35,
  "passes_min_fold_thresholds": true,
  "threshold_reasons": []
}
```

### Thresholds
- `VLAB_ALIFOLD_MIN_PAIR_DENSITY`: Minimum pair density (default: 0.10)
- `VLAB_ALIFOLD_MIN_MFE_PER_NT`: Minimum MFE per nucleotide (default: -0.03)

---

## Role in Pipeline

```
Evolutionary Data (NCBI)
        ↓
  Genome Fetching (Taxon ID / Organism Query)
        ↓
  Sequence Filtering (dedup, length, identity)
        ↓
  MSA Construction (MAFFT → MUSCLE → Clustal Omega)
        ↓
  Conservation Analysis (position scores, regions, motifs)
        ↓
  RNAalifold (optional consensus structure)
        ↓
  Quality Gate Validation
        ↓
  Conservation Fitness Signal
        ↓
  NSGA-II Conservation Objective
```

### Integration Points
- **Feeds into**: Structural Agent (conservation-aware structure prediction), PI Agent (conservation fitness)
- **Consumes from**: LabState (virus family, genus, name, target sequence, designed sequences)
- **Informs**: Conservation objective weight in multi-objective optimisation

---

## Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `ENTREZ_EMAIL` | `fbsnpat@leeds.ac.uk` | Email for NCBI Entrez API |
| `MAFFT_BIN` | `mafft` | Path to MAFFT binary |
| `RNAALIFOLD_BIN` | `RNAalifold` | Path to RNAalifold binary |
| `MAFFT_TIMEOUT` | `180` | MAFFT timeout in seconds |
| `RNAALIFOLD_TIMEOUT` | `120` | RNAalifold timeout in seconds |
| `VLAB_BIOINFO_FETCH_LIMIT` | `20` | Max genomes to fetch from NCBI |
| `VLAB_BIOINFO_MIN_MSA_SEQUENCES` | `3` | Minimum sequences for quality gate |
| `VLAB_BIOINFO_MIN_ALIGNMENT_LENGTH` | `20` | Minimum alignment length |
| `VLAB_BIOINFO_MAX_MSA_IDENTITY` | `0.995` | Max identity for deduplication |
| `VLAB_CONSERVATION_REGION_MIN_SCORE` | `0.85` | Min score for conserved regions |
| `VLAB_CONSERVATION_WINDOW` | `6` | Window size for region detection |
| `VLAB_RNA_TARGET_MOTIFS` | `RGAG:1.0,RAAG:1.0,GAG:0.45,AAG:0.45,CAG:0.25,GUC:0.25,AGU:0.25` | Target motifs and weights |

### Motif Configuration Format
```
MOTIF1:weight1,MOTIF2:weight2,...
```
- Supports IUPAC ambiguity codes (R, Y, S, W, K, M, B, D, H, V, N)
- Weights scale motif conservation scores
- Default motifs optimized for viral RNA packaging signals

---

## Key Insight

This agent transforms:

> evolutionary constraints into quantifiable optimisation signals

By analyzing natural sequence conservation, the Bioinformatics Agent identifies functionally important regions that guide the design of novel RNA sequences with preserved biological functionality.

---

## File Reference

| File | Purpose |
|------|---------|
| `orchestration/agents/bioinfo_agent.py` | Main agent entry, signal normalization, quality gates |
| `core/bioinfo_wrapper.py` | Full-featured wrapper with RNAalifold, motif analysis |
| `wrappers/bioinfo_wrapper.py` | Simplified legacy wrapper |

---
*Last updated: 2026-07-01*