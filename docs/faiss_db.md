# FAISS Vector Database Documentation

## Purpose
FAISS (Facebook AI Similarity Search) is used in VLAB2 to store and retrieve high‑dimensional embeddings of protein‑RNA complexes, inhibitor docking poses, and LLM‑generated summaries. It enables fast nearest‑neighbour queries for:
- Finding similar binding pockets.
- Retrieving previously docked inhibitor poses.
- Supporting the **retrieval‑augmented generation** (RAG) workflow in the `researcher_agent`.

## Index Types
Two FAISS indexes are maintained:
1. **Flat L2 index** – `faiss_index_flat.idx`
   - Used for exact nearest‑neighbour search on small datasets (<10k vectors).
2. **IVF‑PQ index** – `faiss_index_ivfpq.idx`
   - Scalable to >100k vectors with sub‑millisecond query latency.

Both indexes store vectors of dimension `768` (the output of the `sentence‑transformers/all‑MiniLM‑L6‑v2` encoder).

## Data Flow
1. **Embedding generation**
   - `core/embedding.py` provides `embed_text(text: str) -> np.ndarray`.
   - Called by agents when a new docking result or report is produced.
2. **Indexing**
   - `faiss_utils.add_to_index(vector, metadata)` stores the vector and a JSON‑serialised metadata blob (e.g., PDB ID, ligand ID, score).
   - Indexes are persisted under `${BASE_DIR}/faiss/`.
3. **Querying**
   - `faiss_utils.search(query_vector, k=5)` returns the top‑k nearest neighbours with their metadata.
   - Used by `researcher_agent` to retrieve similar prior experiments.

## Configuration
| Variable | Default | Description |
|---|---|---|
| `VLAB_FAISS_INDEX_TYPE` | `flat` | Choose `flat` or `ivfpq`. |
| `VLAB_FAISS_DIM` | `768` | Dimensionality of embedding vectors. |
| `VLAB_FAISS_K` | `5` | Number of neighbours to return on a query. |

These variables are exported in `run_virtual_lab_conda.sh`.

## Maintenance Scripts
- `faiss_utils/create_index.py` – (re)creates the chosen index from scratch.
- `faiss_utils/add_embeddings.py` – batch‑adds embeddings from a CSV of records.
- `faiss_utils/cleanup.sh` – removes stale entries older than `VLAB_FAISS_MAX_AGE_DAYS`.

## Usage Example (Python)
```python
from faiss_utils import embed_text, add_to_index, search

# Add a new docking pose
vector = embed_text("Docking pose of ligand X with score -78.5")
metadata = {"pdb": "6XYZ", "ligand": "X", "score": -78.5}
add_to_index(vector, metadata)

# Query similar poses
query_vec = embed_text("Find poses similar to ligand Y")
results = search(query_vec, k=3)
for r in results:
    print(r["metadata"]["ligand"], r["distance"])
```

---
*Last updated: 2026-07-15*

