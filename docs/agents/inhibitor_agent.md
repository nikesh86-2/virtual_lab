# Inhibitor Agent Documentation

## Overview
The **inhibitor_agent** is a LangGraph node in the VLAB2 pipeline that screens candidate inhibitors against the RNA-binding pocket identified from the `protein_agent`'s interface contacts. It is responsible for:
- Identifying the RNA‑binding pocket from the protein‑RNA interface.
- Fetching or generating small‑molecule inhibitors and docking them with **AutoDock Vina**.
- Designing peptide inhibitors via an LLM, generating 3‑D structures with OpenBabel, and docking them with **HDOCK**.
- Comparing inhibitor poses with RNA‑binding poses and summarising the results for downstream critique.

## State Fields
The agent adds the following fields to `LabState` (see `orchestration/state_schema.py`):

| Field | Type | Description |
|---|---|---|
| `inhibitor_small_molecules` | `list[dict]` | Docking results for small molecules |
| `inhibitor_peptides` | `list[dict]` | Docking results for peptide inhibitors |
| `inhibitor_analysis` | `str` | Detailed textual summary of inhibitor analysis |
| `inhibitor_summary` | `str` | Concise summary string for downstream consumption |
| `inhibitor_binding_site_overlap` | `float` | Overlap score in percent (0‑100), legacy-compatible |
| `inhibitor_binding_site_overlap_score` | `float` | Overlap score in 0‑1 scale |
| `inhibitor_pose_comparison` | `dict` | Full pose comparison data between inhibitors and RNA interface |
| `inhibitor_small_molecule_comparison` | `dict \\| None` | Per-small-molecule comparison details |
| `inhibitor_peptide_comparison` | `dict \\| None` | Per-peptide comparison details |
| `inhibitor_docking_box` | `dict` | Docking box definition (center, size, pocket residues) |
| `inhibitor_snapshot_paths` | `list[str]` | Paths to rendered PyMOL snapshot images |
| `inhibitor_best_small_molecule` | `dict \\| None` | Top-ranked small molecule result |
| `inhibitor_best_peptide` | `dict \\| None` | Top-ranked peptide result |
| `inhibitor_vina_seed` | `int` | Random seed used for Vina docking |
| `inhibitor_vina_cache_hits` | `int` | Number of Vina cache hits |
| `inhibitor_vina_cache_misses` | `int` | Number of Vina cache misses |
| `inhibitor_vina_cache_enabled` | `bool` | Whether Vina caching was enabled |
| `inhibitor_enabled` | `bool` | Whether inhibitor screening was active |
| `stage_outputs` | `list[dict]` | Structured stage output for reporting |
| `conversation_history` | `list[dict]` | Conversation history entry for this agent |

## Environment Variables
| Variable | Default | Description |
|---|---|---|
| `VLAB_INHIBITOR_ENABLED` | `1` | Enable/disable the inhibitor node. |
| `VLAB_INHIBITOR_MAX_SMALL_MOLECULES` | `10` | Maximum number of small‑molecule ligands to dock. |
| `VLAB_INHIBITOR_MAX_PEPTIDES` | `5` | Maximum number of peptide sequences to generate and dock. |
| `VLAB_VINA_EXHAUSTIVENESS` | `8` | Vina exhaustiveness parameter. |
| `VLAB_VINA_SEED` | `1` | Random seed for Vina docking reproducibility. |
| `VLAB_INHIBITOR_BOX_PADDING` | `8.0` | Padding (Å) around the pocket for the Vina search box. |
| `VLAB_INHIBITOR_CACHE_DIR` | `/tmp/vlab_inhibitor_cache` | Base directory for inhibitor caching. |
| `VLAB_VINA_FORCE_REDOCK` | `False` | Force redocking even if cached results exist. |
| `VLAB_PUBCHEM_QUERIES` | `""` | Pipe-separated PubChem search queries (e.g., `"adenine|ATP|GTP"`). |

These variables are exported in `run_virtual_lab_conda.sh`.

## Core Modules
- `core/small_molecule_prep.py` – fetches compounds from PubChem and prepares PDBQT files.
  - `fetch_known_rna_binding_inhibitors()` – retrieves known RNA-binding inhibitors from local database.
  - `fetch_and_prepare_compounds()` – queries PubChem for compounds by name/SMILES.
  - `prepare_receptor_pdbqt()` – converts protein PDB to PDBQT format for Vina.
- `core/peptide_prep.py` – generates peptide 3‑D structures using OpenBabel.
  - `get_known_antiviral_peptides()` – retrieves known antiviral peptide sequences.
  - `design_inhibitor_peptides()` – LLM-driven peptide design targeting the pocket.
  - `prepare_peptides()` – converts peptide sequences to 3D PDB structures.
- `core/inhibitor_docking.py` – orchestrates Vina and HDOCK docking runs.
  - `run_inhibitor_screen()` – main docking orchestrator for both small molecules and peptides.
- `core/rna_binding_pocket.py` – derives pocket centre and size from interface contacts.
  - `define_docking_box()` – computes Vina box parameters from RNA-protein interface.
- `core/protein_prep.py` – utility for resolving PDB IDs to local PDB files.
  - `ensure_protein_pdb()` – fetches and validates protein PDB structure.
- `orchestration/utils/docking_visuals.py` – renders PyMOL snapshots.
  - `render_inhibitor_snapshots_for_results()` – generates visual snapshots of docking poses.

## Agent Workflow
1. **Target resolution** – Resolves the target PDB from state (`target_pdb_path`, `target_pdb_id`, or partial targets). Downloads from PDB if needed.
2. **Pocket extraction** – Uses `rna_binding_pocket.define_docking_box()` to obtain Vina box parameters (center, size, pocket residues) from interface contacts.
3. **Small‑molecule fetching** – Calls `small_molecule_prep.fetch_known_rna_binding_inhibitors()` for known inhibitors, then `fetch_and_prepare_compounds()` for PubChem queries (from `VLAB_PUBCHEM_QUERIES`). Results are deduplicated by SMILES → CID → name priority.
4. **Peptide design** – Calls `peptide_prep.get_known_antiviral_peptides()` for known peptides, then `peptide_prep.design_inhibitor_peptides()` (LLM‑driven) for novel designs. Results are deduplicated by sequence.
5. **Docking** – Runs AutoDock Vina for small molecules and HDOCK for peptides via `run_inhibitor_screen()`. Vina supports caching for reproducibility.
6. **Pose comparison** – `inhibitor_docking` computes RMSD/overlap between inhibitor poses and RNA-binding poses.
7. **Snapshot rendering** – Generates PyMOL visual snapshots of key docking poses.
8. **State update** – Populates all inhibitor-related state fields and produces summary/analysis strings.

## Integration Points
- **Orchestrator** – Added as node `inhibitor` in the LangGraph pipeline.
- **Skeptic Agent** – Updated prompt to include inhibitor scores and overlap.
- **Reporting** – `orchestration/reporting.py` includes an *Inhibitor Results* section.
- **Checkpointing** – Results are saved via `save_checkpoint()` for pipeline recovery.

## Error Handling & Edge Cases
The agent gracefully handles several failure modes:
- **Disabled screening** – Returns empty results with `inhibitor_enabled=False` when `VLAB_INHIBITOR_ENABLED` is not set.
- **Missing target PDB** – Returns early if no valid protein structure can be resolved.
- **Missing interface contacts** – Returns early if `interface_contacts` is not available from `protein_agent`.
- **Pocket definition failure** – Continues with available box parameters if pocket computation fails.
- **PubChem fetch failure** – Logs warning and continues with known inhibitors only.
- **Peptide design failure** – Logs warning and continues with known peptides only.
- **Receptor PDBQT preparation failure** – Skips small-molecule Vina docking but continues with peptide HDOCK.
- **Snapshot rendering failure** – Logs warning but does not block result reporting.
- **General exceptions** – Catches all errors, logs stack trace, and returns a structured error payload.

## Usage
The agent runs automatically when `VLAB_INHIBITOR_ENABLED=1` (default). No manual invocation is required; it participates in the LangGraph execution flow.

To customize inhibitor screening:
- Set `VLAB_PUBCHEM_QUERIES` to search specific compounds from PubChem.
- Adjust `VLAB_INHIBITOR_MAX_SMALL_MOLECULES` and `VLAB_INHIBITOR_MAX_PEPTIDES` to control screening breadth.
- Set `VLAB_VINA_FORCE_REDOCK=True` to bypass cached Vina results.
- Modify `VLAB_VINA_SEED` for reproducible docking runs.

---
*Last updated: 2026-07-01*