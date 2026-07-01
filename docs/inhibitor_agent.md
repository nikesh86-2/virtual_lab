# Inhibitor Agent Documentation

## Overview
The **inhibitor_agent** is a new LangGraph node added to the VLAB2 pipeline between the `protein_agent` and the `skeptic_agent`. It is responsible for:
- Identifying the RNA‑binding pocket from the protein‑RNA interface.
- Fetching or generating small‑molecule inhibitors and docking them with **AutoDock Vina**.
- Designing peptide inhibitors via an LLM, generating 3‑D structures with OpenBabel, and docking them with **HDOCK**.
- Comparing inhibitor poses with RNA‑binding poses and summarising the results for downstream critique.

## State Fields
The agent adds the following fields to `LabState` (see `orchestration/state_schema.py`):
```python
inhibitor_small_molecules: List[dict]   # docking results for small molecules
inhibitor_peptides: List[dict]          # docking results for peptide inhibitors
inhibitor_analysis: str                 # textual summary of inhibitor analysis
inhibitor_binding_site_overlap: float   # overlap score (0‑1) with RNA binding site
```

## Environment Variables
| Variable | Default | Description |
|---|---|---|
| `VLAB_INHIBITOR_ENABLED` | `1` | Enable/disable the inhibitor node. |
| `VLAB_INHIBITOR_MAX_SMALL_MOLECULES` | `10` | Maximum number of small‑molecule ligands to dock. |
| `VLAB_INHIBITOR_MAX_PEPTIDES` | `5` | Maximum number of peptide sequences to generate and dock. |
| `VLAB_VINA_EXHAUSTIVENESS` | `8` | Vina exhaustiveness parameter. |
| `VLAB_INHIBITOR_BOX_PADDING` | `8.0` | Padding (Å) around the pocket for the Vina search box. |

These variables are exported in `run_virtual_lab_conda.sh`.

## Core Modules
- `core/small_molecule_prep.py` – fetches compounds from PubChem and prepares PDBQT files.
- `core/peptide_prep.py` – generates peptide 3‑D structures using OpenBabel.
- `core/inhibitor_docking.py` – orchestrates Vina and HDOCK docking runs.
- `core/rna_binding_pocket.py` – derives pocket centre and size from interface contacts.

## Agent Workflow
1. **Pocket extraction** – Uses `rna_binding_pocket.compute_pocket_*` to obtain Vina box.
2. **Small‑molecule fetching** – Calls `small_molecule_prep.fetch_pubchem_compounds` and prepares ligands.
3. **Peptide design** – Calls `peptide_prep.design_inhibitor_peptides` (LLM‑driven) and builds PDBs.
4. **Docking** – Runs Vina for small molecules and HDOCK for peptides.
5. **Comparison** – `inhibitor_docking.compare_inhibitor_with_rna_pose` computes RMSD/overlap.
6. **State update** – Populates the new state fields and produces a summary string.

## Integration Points
- **Orchestrator** – Added as node `inhibitor` with edges `protein → inhibitor → skeptic`.
- **Skeptic Agent** – Updated prompt to include inhibitor scores and overlap.
- **Reporting** – `orchestration/reporting.py` now includes an *Inhibitor Results* section.

## Usage
The agent runs automatically when `VLAB_INHIBITOR_ENABLED=1`. No manual invocation is required; it participates in the LangGraph execution flow.

---
*Last updated: 2026‑07‑01*