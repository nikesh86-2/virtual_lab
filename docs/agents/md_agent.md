# MD Agent Documentation

## Overview
The **md_agent** validates structural stability of RNA sequences through molecular dynamics (MD) simulations. It serves as a validation layer in the VLAB2 pipeline, providing high-fidelity stability metrics that complement cheaper computational methods.

## Responsibilities
- Evaluate structural stability of candidate RNA sequences via MD simulations
- Extract and standardise energy metrics (min energy, mean energy, fluctuation)
- Cache MD results to avoid redundant computations
- Gate MD execution based on fold quality thresholds
- Track and report failed sequences

## State Inputs
The agent reads the following fields from `LabState`:

| Field | Type | Description |
|---|---|---|
| `target_sequence` | `str` | Primary target sequence |
| `designed_sequences` | `list[str]` | Sequences from design agents |
| `structural_candidates` | `list[dict]` | Structural candidates with sequences |
| `fold_thresholds_passed` | `bool` | Whether fold quality thresholds passed |
| `fold_threshold_reasons` | `list[str]` | Reasons for fold threshold failures |
| `wrappers["md"]` | `object` | MD simulation wrapper instance |
| `_md_cache` | `dict` | Cached MD results (persisted across iterations) |

## State Outputs
The agent adds the following fields to `LabState`:

| Field | Type | Description |
|---|---|---|
| `md_analysis` | `str` | Human-readable summary of MD results |
| `md_results` | `list[dict]` | Detailed MD results for each sequence |
| `failed_sequences` | `list[dict]` | Sequences that failed MD simulation |
| `_md_cache` | `dict` | Persisted MD result cache |

Each `md_results` entry contains:
```python
{
    "sequence": str,           # The RNA sequence
    "result": {
        "valid": bool,         # Whether simulation succeeded
        "min_energy": float,   # Minimum energy observed
        "mean_energy": float,  # Mean energy over trajectory
        "energy_fluctuation": float,  # Energy variance (max - min)
        "energy_trajectory": list[float],  # Optional energy trace
        "error": str           # Error message if failed
    }
}
```

## Environment Variables
| Variable | Default | Description |
|---|---|---|
| `VLAB_AGENT_EVAL_TOP_N` | `3` | Maximum number of sequences to evaluate |
| `VLAB_RNA_ENFORCE_MIN_FOLD` | `1` | Enforce fold quality gate before MD |

## Core Components

### `_extract_md_metrics(output: dict) -> dict`
Standardises MD outputs into consistent fields. Supports:
- Direct values from MD wrapper
- Energy trajectory lists (infer min/mean/fluctuation)

### `md_agent(state: LabState) -> dict`
Main agent function that orchestrates MD evaluation.

## Agent Workflow

1. **Sequence Selection** – Collects sequences from:
   - `target_sequence`
   - `designed_sequences`
   - `structural_candidates` (those with valid sequences)
   - Deduplicates and limits to `VLAB_AGENT_EVAL_TOP_N` sequences

2. **Fold Quality Gate** – If `fold_thresholds_passed=False` and `VLAB_RNA_ENFORCE_MIN_FOLD=1`:
   - Checks if any structural candidate passed fold quality
   - If none passed, skips MD and returns early with failure summary

3. **MD Execution** – For each sequence:
   - Checks `_md_cache` for existing results (avoids redundant computation)
   - If not cached, runs `md.run_md(seq)` via the MD wrapper
   - Catches exceptions and records them as failed results
   - Clears GPU memory after each simulation
   - Stores result in cache for future iterations

4. **Metric Extraction** – Calls `_extract_md_metrics()` to standardise outputs:
   - Extracts `min_energy`, `mean_energy`, `energy_fluctuation`
   - If direct values missing, infers from `energy_trajectory` or `energies` list

5. **Result Classification** – Separates results into:
   - `valid_results`: Simulations that completed successfully
   - `failed_sequences`: Simulations that failed with error messages

6. **Summary Generation** – Creates readable summary showing top 5 valid results with:
   - Sequence (truncated to 20 chars)
   - Minimum energy
   - Mean energy
   - Energy fluctuation

7. **Checkpoint & Return** – Saves state with results and cache, returns update dict

## Error Handling & Edge Cases

The agent handles several failure modes:

- **No sequences** – Returns empty results with `md_analysis="No sequences"`
- **Fold quality gate failure** – Skips MD entirely if fold thresholds not met
- **All simulations fail** – Returns all results with failure summary
- **Individual simulation failures** – Records errors per sequence, continues with valid results
- **Wrapper unavailable** – Caught by outer try/except, returns error summary
- **General exceptions** – Logs stack trace, returns error summary with empty results
- **GPU memory** – Cleared after each simulation and in finally block

## Integration Points

- **Preceded by**: PI Agent (provides designed sequences)
- **Followed by**: Skeptic Agent (uses MD stability signals)
- **State persistence**: Results and cache persisted via `save_checkpoint()`
- **GPU management**: Uses `clear_gpu()` to free memory after simulations

## Key Design Principles

1. **Caching** – MD results are cached by sequence to avoid redundant expensive computations across iterations
2. **Validation-only role** – MD is used ONLY for validation and surrogate training, not for routine screening
3. **Cheap optimization → Expensive validation** – NSGA-II with surrogate models handles initial screening; MD validates top candidates
4. **Graceful degradation** – Partial failures don't block the pipeline; valid results are still reported

---
*Last updated: 2025-07-01*