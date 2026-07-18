# Core Utilities Documentation

## Overview

The **Core Utilities** module (`core/`) provides the scientific software wrappers and helper functions that interface VLAB2 with real biophysical simulation tools. These wrappers translate between VLAB2's state format and the input/output formats of external programs like HDOCK, SimRNA, ViennaRNA, and OpenBabel.

**Location**: `core/`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VLAB2 Agents                             │
│         (PI, Structural, MD, Protein, etc.)                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Core Wrappers Layer                        │
│                                                             │
│  MD Wrapper    → SimRNA molecular dynamics                  │
│  Protein Wrap  → RNA-protein docking (SimRNA/HDOCK)         │
│  HDOCK Wrap    → HDOCKlite docking server                   │
│  ViennaRNA     → RNA secondary structure prediction         │
│  SFold         → Alternative folding prediction             │
│  Bioinfo Wrap  → NCBI/BLAST/MAFFT for MSA                   │
│  Vina Wrap     → AutoDock Vina for small molecules          │
│                                                             │
│  Supporting utilities:                                      │
│  - PDB/RNA preparation                                      │
│  - Docking utilities                                        │
│  - Peptide/small-molecule prep                              │
│  - GPU management                                           │
│  - Training data collection                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              External Scientific Tools                      │
│         (HDOCK, SimRNA, ViennaRNA, OpenBabel, etc.)         │
└─────────────────────────────────────────────────────────────┘
```

## Wrapper Classes

### MDWrapper (`md_wrapper.py`)

**Purpose**: Run SimRNA molecular dynamics simulations to assess RNA stability.

**Key Methods**:
```python
class MDWrapper:
    def run_md(sequence: str) -> dict
    def get_rna_structure(sequence: str) -> str
```

**Input**: RNA sequence string
**Output**: Dictionary with:
- `valid`: bool
- `energy_fluctuation`: float
- `min_energy`: float
- `rna_pdb_path`: str (path to generated PDB)
- `error`: str (if failed)

**Workflow**:
1. Generates initial RNA structure using SimRNA
2. Runs coarse-grained MD simulation
3. Calculates energy fluctuation as stability metric
4. Returns PDB path for downstream analysis

### ProteinWrapper (`protein_wrapper.py`)

**Purpose**: Perform RNA-protein docking using SimRNA or HDOCK.

**Key Methods**:
```python
class ProteinWrapper:
    def evaluate_sequences(pdb_id: str, sequences: List[str]) -> List[dict]
```

**Input**: Target PDB ID, list of RNA sequences
**Output**: List of docking results, each containing:
- `valid`: bool
- `dock_valid`: bool
- `binding_mode`: str
- `dock_score`: float
- `dg`: float (binding energy)
- `complex_file`: str (docked complex PDB path)
- `error`: str (if failed)

**Workflow**:
1. Prepares target protein PDB
2. For each RNA sequence:
   - Generates RNA structure
   - Runs docking simulation
   - Extracts binding score and interface contacts
3. Returns ranked results

### HDockDocking (`hdock_wrapper.py`)

**Purpose**: Interface with HDOCKlite server for protein-RNA docking.

**Key Methods**:
```python
class HDockDocking:
    def dock(protein_pdb: str, rna_pdb: str, box_center: dict) -> dict
    def parse_results(output_dir: str) -> List[dict]
```

**Features**:
- Automatic server management
- Result parsing and ranking
- Interface contact extraction
- PyMOL snapshot generation

### ViennaRNAWrapper (`viennarna_wrapper.py`)

**Purpose**: Predict RNA secondary structure using ViennaRNA package.

**Key Methods**:
```python
class ViennaRNAWrapper:
    def run_rnafold(sequence: str) -> dict
    def get_mfe(sequence: str) -> float
    def get_structure(sequence: str) -> str
```

**Output**: Dictionary with:
- `valid`: bool
- `mfe`: float (minimum free energy)
- `structure`: str (dot-bracket notation)
- `ecm`: float (ensemble clustering measure)

### SFoldWrapper (`sfold_wrapper.py`)

**Purpose**: Alternative RNA folding prediction using SFold statistical ensemble.

**Key Methods**:
```python
class SFoldWrapper:
    def run_sfold(sequence: str) -> dict
```

**Use case**: Provides alternative folding prediction when ViennaRNA fails or for validation.

### BioinfoWrapper (`bioinfo_wrapper.py`)

**Purpose**: Retrieve genomic sequences and perform multiple sequence alignment (MSA).

**Key Methods**:
```python
class BioinfoWrapper:
    def retrieve_sequences(virus_name: str) -> List[str]
    def perform_msa(sequences: List[str]) -> dict
    def analyze_conservation(msa_data: str) -> dict
```

**Workflow**:
1. Queries NCBI/GenBank for viral sequences
2. Performs MSA using MAFFT
3. Calculates conservation scores per position
4. Identifies conserved regions

**Output**:
- `msa_data`: Aligned sequences
- `conservation_fitness`: float (0-1)
- `conserved_regions`: List of position ranges
- `num_sequences`: int

### ViennaRNAWrapper (`viennarna_wrapper.py`)

**Purpose**: RNA secondary structure prediction using ViennaRNA package.

**Key Methods**:
```python
class ViennaRNAWrapper:
    def run_rnafold(sequence: str) -> dict
    def get_mfe(sequence: str) -> float
    def get_structure(sequence: str) -> str
```

### VinaWrapper (`vina_wrapper.py`)

**Purpose**: Small-molecule docking using AutoDock Vina.

**Key Methods**:
```python
class VinaWrapper:
    def dock_molecule(protein_pdbqt: str, ligand_pdbqt: str, box: dict) -> dict
```

**Use case**: Inhibitor screening for small molecules and peptides.

## Preparation Utilities

### Protein Preparation (`protein_prep.py`)

**Purpose**: Prepare protein PDB files for docking.

**Functions**:
```python
def prepare_protein_pdb(pdb_path: str, chain: str = "A") -> str
def extract_chain(pdb_path: str, chain_id: str) -> str
def clean_pdb(pdb_path: str) -> str
```

### RNA Preparation (`rna_prep.py`)

**Purpose**: Prepare RNA sequences for simulation.

**Functions**:
```python
def prepare_rna_sequence(sequence: str) -> str
def validate_rna_sequence(sequence: str) -> bool
def convert_T_to_U(sequence: str) -> str
```

### PDBQT Utilities (`pdbqt_utils.py`)

**Purpose**: Convert PDB to PDBQT format for docking.

**Functions**:
```python
def pdb_to_pdbqt(pdb_path: str) -> str
def add_charges(pdbqt_path: str) -> str
def define_docking_box(pdb_path: str, center: dict, size: dict) -> dict
```

### Peptide Preparation (`peptide_prep.py`)

**Purpose**: Prepare peptide inhibitors for docking.

**Functions**:
```python
def prepare_peptide(peptide_sequence: str) -> dict
def generate_peptide_pdbqt(peptide_sequence: str) -> str
```

### Small Molecule Preparation (`small_molecule_prep.py`)

**Purpose**: Prepare small-molecule inhibitors for docking.

**Functions**:
```python
def prepare_small_molecule(smiles: str) -> dict
def smiles_to_pdbqt(smiles: str) -> str
def query_pubchem(cid: str) -> dict
```

## Docking Utilities (`docking_utils.py`)

**Purpose**: Common docking analysis functions.

**Functions**:
```python
def parse_docking_results(output_dir: str) -> List[dict]
def calculate_interface_contacts(complex_pdb: str) -> dict
def extract_binding_energy(docking_output: str) -> float
def rank_docking_results(results: List[dict]) -> List[dict]
```

## GPU Management (`gpu_manager.py`)

**Purpose**: Manage GPU resource allocation for simulations.

**Functions**:
```python
def get_available_gpu() -> int
def set_gpu_device(device_id: int) -> None
def check_gpu_memory(min_mb: int) -> bool
```

## Structure Core (`structure_core.py`)

**Purpose**: Common structure manipulation utilities.

**Functions**:
```python
def calculate_rmsd(pdb1: str, pdb2: str) -> float
def superpose_structures(pdb1: str, pdb2: str) -> str
def extract_binding_pocket(pdb_path: str, ligand: str) -> dict
```

## RNA Binding Pocket (`rna_binding_pocket.py`)

**Purpose**: Identify and characterize RNA binding pockets on proteins.

**Functions**:
```python
def identify_binding_pocket(protein_pdb: str) -> dict
def calculate_pocket_volume(pocket: dict) -> float
def match_pocket_to_interface(pocket: dict, interface: dict) -> float
```

## Mutation Engine (`mutation_engine.py`)

**Purpose**: Apply mutations to RNA sequences with various strategies.

**Functions**:
```python
def point_mutation(sequence: str, position: int, nucleotide: str) -> str
def insert_mutation(sequence: str, position: int, inserted: str) -> str
def delete_mutation(sequence: str, position: int, length: int) -> str
def apply_mutation_bias(sequence: str, bias: dict) -> str
```

## Training Data Collector (`training_data_collector.py`)

**Purpose**: Collect and format training data from research cycles.

**Class**:
```python
class TrainingDataCollector:
    def __init__(self, output_dir: str = "training_data", prefix: str = "biophysics_tuning")
    
    def collect_cycle(topic: str, hypothesis: str, evidence: list, 
                     critique: str, final_synthesis: str) -> None
    def capture_agent_interaction(agent_name: str, task: str, 
                                 response: str, input_context: str) -> None
    def capture_optimisation_step(population_summary: str, 
                                 selected_sequences: list, 
                                 mutation_bias: dict) -> None
    def capture_mutation_pattern(bias: dict, iteration: int) -> None
    def summarise_population(population: list, decode_fn: callable) -> str
```

**Data Format**: JSONL files in `training_data/` directory.

## Wrapper Bundle Construction

Located in: `orchestration/wrappers.py`

```python
def build_wrapper_bundle() -> dict:
    """Build all computational wrappers once at startup."""
    md = MDWrapper()
    protein = ProteinWrapper(simrna=md)
    hdock = HDockDocking()
    
    wrappers = {
        "protein": protein,
        "md": md,
        "sfold": SFoldWrapper(),
        "vienna": ViennaRNAWrapper(),
        "bioinfo": BioinfoWrapper(),
        "hdock": hdock,
    }
    return wrappers
```

The wrapper bundle is passed through state to agents that need computational tools.

## Error Handling

All wrappers implement graceful degradation:

1. **Try/except blocks**: Catch simulation failures
2. **Fallback values**: Return sensible defaults on error
3. **Error logging**: Log stack traces for debugging
4. **Valid flags**: Each result includes `valid: bool`
5. **Error messages**: Include descriptive error strings

```python
def run_md(sequence: str) -> dict:
    try:
        # ... simulation code ...
        return {"valid": True, "energy_fluctuation": fluct, ...}
    except Exception as e:
        log.warning("MD failed for %s: %s", sequence[:25], e)
        return {"valid": False, "error": str(e)}
```

## Integration with Agents

Each agent uses specific wrappers:

| Agent | Wrappers Used |
|---|---|
| **PI Agent** | All (via state) |
| **Structural Agent** | ViennaRNA, SFold |
| **MD Agent** | MDWrapper |
| **Protein Agent** | ProteinWrapper, HDockDocking |
| **Bioinformatics Agent** | BioinfoWrapper |
| **Inhibitor Agent** | VinaWrapper, small_molecule_prep, peptide_prep |

## Key Design Principles

- **State-driven**: Wrappers read from and write to LabState, not global variables
- **Pure functions**: Given same input, always produce same output
- **Fault-tolerant**: Graceful degradation with valid flags and error messages
- **Serialisable**: All outputs are JSON-serialisable dictionaries
- **Modular**: Each wrapper is independent and can be swapped
- **Cached**: Repeated computations are cached to save time
- **Logged**: All operations are logged for debugging and audit

---
*Last updated: 2026-07-01*