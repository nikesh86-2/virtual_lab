# Research Agent (Adaptive Signal Engine)

## Overview

Transforms scientific literature into optimisation guidance for RNA-viral protein docking and inhibitor screening tasks.

This agent connects:
- literature → signals → optimisation weights

## System Architecture

The Research Agent is composed of three integrated components:

1. **Orchestration Layer** (`orchestration/agents/researcher_agent.py`)
   - Main agent entry point
   - Manages literature query generation and execution
   - Handles evidence collection and deduplication
   - Integrates with Skeptic critique parser

2. **Research Core** (`research/`)
   - `research_agent_adaptive.py` - Semantic Scholar search, FAISS vector DB, weight adaptation
   - `research_agent_loop.py` - Multi-agent research controller for population analysis
   - `self_driving_lab.py` - Autonomous hypothesis generation and strategy updates
   - `literature_refiner.py` - Query refinement based on Skeptic critique
   - `streaming_literature_agent.py` - Background literature ingestion

3. **Knowledge Base**
   - FAISS vector index for local literature storage
   - Semantic Scholar API for external paper retrieval
   - Skeptic critique parser for evidence validation

---

## Responsibilities

### Primary Tasks
- Retrieve papers from PubMed (via Entrez) and Semantic Scholar
- Extract scientific signals through topic-aware filtering
- Build multi-objective optimisation weights
- Guide NSGA-II mutation and selection behaviour
- Background streaming of relevant literature into FAISS

### Query Generation
- Builds topic-aware literature profile from state (virus family, genus, name, task type)
- Generates primary query using LLM (temperature=0.0)
- Creates deterministic fallback queries based on virus family
- Refines queries based on Skeptic critique output
- Enforces query filters to exclude irrelevant literature (ML, robotics, clinical trials)

### Literature Filtering
- **Biomedical Relevance**: Filters for RNA, capsid, nucleocapsid, packaging, binding terms
- **Topic Relevance**: Matches against virus-specific profiles
- **Inhibitor Screening Mode**: Stricter filtering for capsid/RNA-binding pocket inhibitors vs polymerase inhibitors
- **Deduplication**: By DOI, normalised title/year/content hash

---

## Key Outputs

### Objective Weights
```json
{
  "thermo": 0.25,
  "structure": 0.20,
  "motif": 0.15,
  "binding": 0.20,
  "kinetic": 0.10,
  "conservation": 0.10
}
```
These weights control NSGA-II objective priorities and are dynamically updated.

### Evidence Items
Each evidence item contains:
```json
{
  "schema_version": "literature_evidence.v2",
  "title": "Paper Title",
  "abstract": "Abstract text...",
  "source": "semantic_scholar|local_faiss",
  "query": "Search query used",
  "year": 2024,
  "doi": "10.xxxx/xxxxx",
  "url": "https://...",
  "retrieval_score": 0.85,
  "topic_name": "Topic identifier",
  "virus_family": "Coronaviridae",
  "task_type": "rna_docking|inhibitor_screening"
}
```

### Research State
- `streaming_started`: Whether background streaming is active
- `streamed_queries`: List of queries used for streaming
- `streamed_query_keys`: Unique keys for deduplication
- `research_query`: Primary query string
- `literature_query_bundle`: All queries attempted
- `literature_topic_profile`: Full topic profile used

---

## Signal Extraction

Identifies and scores six key biological signals:

### 1. Thermodynamic Stability (`thermo`)
- Derived from minimum free energy (MFE) or molecular dynamics energy
- Higher stability = more negative energy values
- Normalised via weighting

### 2. Secondary Structure (`structure`)
- Based on pair density, base pair probability mean, and base pair count
- Formula: `0.4 * pair_density + 0.4 * bpp_mean + 0.2 * (base_pairs / length)`

### 3. Sequence Motifs (`motif`)
- Detects RGAG and RAAG patterns using regex
- Score = fraction of motifs found

### 4. Binding Affinity (`binding`)
- From docking scores or binding energy
- Normalised via tanh function: `tanh(-score / 100)` for docking, `tanh(-score / 10)` for binding energy

### 5. Kinetics (`kinetic`)
- Combines ensemble entropy and energy fluctuation
- Formula: `0.5 * entropy + 0.5 * (1 / (1 + fluctuation))`

### 6. Conservation (`conservation`)
- Based on conserved region coverage in sequence
- Ratio of conserved positions to total length

---

## Adaptive Learning

### Weight Update Mechanism (`update_weights()`)

```python
def update_weights(population_scores, current_weights, lr=0.2) -> Dict:
```

**Process:**
1. Sort population by total fitness
2. Select top third of population
3. Calculate average scores for each objective in top performers
4. Normalise to probability distribution
5. Blend with current weights using learning rate:
   ```
   new_weight = (1 - lr) * current_weight + lr * top_performer_avg
   ```
6. Enforce minimum weight (0.05) to prevent objective collapse
7. Renormalise to sum to 1.0

**Purpose:**
- Learns from top-performing sequences
- Dynamically rebalances objectives based on what works
- Prevents any single objective from dominating permanently
- Adapts to emergent patterns in sequence space

### Mutation Guidance

While not a single function, mutation parameters are influenced by:
- **Motif insertion probability**: Higher when motif scores are low
- **Structure bias**: Adjusted based on structure objective weight
- **GC content bias**: Tied to thermodynamic optimisation
- **Mutation rate**: Scales with focus objective weight (`0.1 + weight[focus]`)
- **Population size**: Scales with focus objective (`40 + 60 * weight[focus]`)

---

## Research Agent Loop

### Autonomous Research Cycle (`self_driving_lab.py`)

```
Generation N:
  1. Interpret Results → average scores, dominant signal
  2. Generate Hypothesis → based on dominant weight
  3. Every 5 generations:
     a. Analyse population behaviour
     b. Propose new literature query
     c. Expand knowledge via literature search
     d. Merge weight models
```

### Hypothesis Generation
Maps dominant objective to testable hypotheses:
- `thermo` → "Higher GC content improves RNA structural stability"
- `structure` → "Secondary structure density drives functional behaviour"
- `motif` → "RGAG/RAAG motifs increase functional efficiency"
- `binding` → "Binding affinity motifs dominate performance"
- `kinetic` → "Folding kinetics constrain optimal structures"

### Query Proposal Logic
Based on population averages:
- Low motif score → "RNA motifs regulatory elements RGAG RAAG binding"
- High structure + low kinetic → "RNA folding kinetics vs thermodynamic stability"
- Low binding → "RNA protein binding affinity determinants"
- Default → "RNA secondary structure stability mechanisms"

---

## Background Literature Streaming

### Streaming Architecture (`streaming_literature_agent.py`)

**Features:**
- Background threads fetch papers continuously
- One stream per unique query + topic profile combination
- Configurable interval (default: 30s), batch size (default: 20)
- Automatic stop after idle cycles or max cycles

**Deduplication Pipeline:**
1. Paper-level dedup by DOI or normalised title/year/content hash
2. FAISS-level dedup by document key
3. Batch-level dedup within each fetch

**Relevance Filtering:**
- **BIO_TERMS**: 50+ biomedical terms (stem-loop, capsid, RNA binding, etc.)
- **DOMAIN_ANCHORS**: Virus-specific anchor terms
- **BAD_TERMS**: Excludes ML, robotics, clinical, manufacturing papers
- **Inhibitor mode**: Stricter filtering for target relevance

**Reranking:**
- Optional CrossEncoder reranker (`ms-marco-MiniLM-L-6-v2`)
- Reranks local FAISS results by query relevance
- Configurable via `VLAB_RERANKER_MODEL`

---

## Query Generation Pipeline

### Step 1: Topic Profile Construction
Extracts from `LabState`:
- `topic_name`, `research_topic`, `hypothesis`
- `virus_family`, `virus_genus`, `virus_name`
- Infers `task_type`: `rna_docking` or `inhibitor_screening`
- Builds `target_concepts` and `motif_concepts` lists
- Family-specific concept expansion (Coronaviridae, Picornaviridae)

### Step 2: LLM Query Generation
- System prompt enforces: 5-9 words, no punctuation, must contain "RNA", must contain viral concept
- Temperature = 0.0 for deterministic output
- Filters out framework/LLM/clinical terms

### Step 3: Fallback Queries
Deterministic queries based on virus family and task type:
- Coronavirus: "coronavirus nucleocapsid RNA binding", "coronavirus RNA packaging signal"
- Picornaviridae: "poliovirus capsid RNA binding", "enterovirus RNA packaging capsid"
- Inhibitor screening: "{organism} RNA binding protein inhibitors", etc.

### Step 4: Skeptic-Driven Refinement
Parses Skeptic critique to generate targeted queries:
- Conservation concerns → "Viral RNA stem-loop conservation packaging"
- Docking concerns → "viral capsid RNA docking binding energy"
- MD concerns → "viral RNA stem-loop molecular dynamics stability"
- Solvent concerns → "viral RNA capsid binding ions solvent"

---

## Role in System

```
Literature (Semantic Scholar + FAISS)
        ↓
  Query Generation (LLM + Fallbacks + Skeptic-refined)
        ↓
  Background Streaming (continuous FAISS updates)
        ↓
  Evidence Collection (deduplicated, metadata-rich)
        ↓
  Signal Extraction (thermo, structure, motif, binding, kinetic, conservation)
        ↓
  Objective Weighting (adaptive update from top performers)
        ↓
  NSGA-II Mutation & Selection (guided by weights)
        ↓
  Population Evaluation → Feedback to Research Agent (every 5 generations)
```

---

## Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `VLAB_FAISS_INDEX_PATH` | `./cache/faiss_index` | Path to FAISS vector index |
| `VLAB_LIT_MAX_TOPIC_STREAMS` | `5` | Max concurrent topic streams |
| `VLAB_LIT_MAX_REFINED_STREAMS` | `3` | Max Skeptic-refined streams |
| `VLAB_STREAM_INTERVAL` | `30` | Seconds between stream fetches |
| `VLAB_STREAM_BATCH_SIZE` | `20` | Papers per fetch |
| `VLAB_STREAM_MAX_IDLE` | `0` | Max idle cycles before stop (0=infinite) |
| `VLAB_STREAM_MAX_CYCLES` | `0` | Max total cycles (0=infinite) |
| `VLAB_LIT_DEDUP_ABSTRACT_PREFIX_LEN` | `500` | Abstract prefix length for dedup |
| `VLAB_LIT_STRICT_INHIBITOR_FILTER` | `1` | Enable strict inhibitor filtering |
| `VLAB_RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | CrossEncoder model for reranking |
| `VLAB_RERANKER_LOCAL_ONLY` | `1` | Use only local reranker files |
| `VLAB_RESEARCH_EVIDENCE_CAP` | `20` | Maximum evidence items to retain |
| `VLAB_STREAM_WARMUP_SECONDS` | `6.0` | Wait time after starting streams |
| `S2_API_KEY` | (required) | Semantic Scholar API key |
| `ENTREZ_EMAIL` | `virtual.lab@example.com` | Entrez email for PubMed |

---

## Key Insight

This agent turns:

> biological knowledge into mathematical optimisation signals

Through continuous literature ingestion, adaptive weight updating, and autonomous hypothesis generation, the Research Agent creates a closed-loop system where experimental results feed back into literature queries, which in turn guide future experimental design.

---

## File Reference

| File | Purpose |
|------|---------|
| `orchestration/agents/researcher_agent.py` | Main agent entry, evidence collection |
| `research/research_agent_adaptive.py` | Semantic Scholar, FAISS, weight adaptation |
| `research/research_agent_loop.py` | Population analysis, query proposal |
| `research/self_driving_lab.py` | Autonomous hypothesis and strategy |
| `research/literature_refiner.py` | Skeptic-driven query refinement |
| `research/streaming_literature_agent.py` | Background literature ingestion |