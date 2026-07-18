# Peptide Production Documentation

## Purpose
The peptide production pipeline generates peptide inhibitors for RNA‑binding viral proteins. It is used by the **inhibitor_agent** to create peptide candidates that are subsequently docked with **HDOCK**.

## Steps
1. **Design peptide sequences**
   - The LLM is prompted with the target protein PDB and pocket residues.
   - Desired peptide length is configurable via `VLAB_INHIBITOR_PEPTIDE_LENGTH` (default `5-15`).
   - The agent returns a list of candidate sequences.
2. **Generate 3‑D structures**
   - Each sequence is converted to a PDB file using OpenBabel:
     ```bash
     obabel -:"ACDEFG" -opdb --gen3d -O peptide.pdb
     ```
   - The resulting PDBs are stored under `${BASE_DIR}/output_data/peptide_structures/`.
3. **Dock peptides**
   - HDOCK is invoked in *protein‑protein* mode, treating the peptide as a small protein.
   - Docking parameters are defined in `core/inhibitor_docking.py`.
4. **Analyse results**
   - Docking scores and poses are parsed and stored in the `inhibitor_peptides` state field.
   - Overlap with the RNA‑binding site is computed to rank peptide efficacy.

## Configuration
| Variable | Default | Description |
|---|---|---|
| `VLAB_INHIBITOR_MAX_PEPTIDES` | `5` | Maximum number of peptide candidates to generate. |
| `VLAB_INHIBITOR_PEPTIDE_LENGTH` | `5-15` | Allowed peptide length range (min‑max). |
| `VLAB_HDOCK_PATH` | `$(which hdock)` | Path to the HDOCK executable. |

## Files
- `core/peptide_prep.py` – Implements `design_inhibitor_peptides` and `generate_peptide_pdb`.
- `core/inhibitor_docking.py` – Calls HDOCK for each peptide PDB.
- `run_virtual_lab_conda.sh` – Exports the environment variables above.

---
*Last updated: 2026-07-15*