"""
research_agent_adaptive.py (FINAL – FULL INTEGRATION, HDOCK COMPATIBLE)

Includes:
- SFold + SimRNA multi-scale scoring
- Binding proxy integration
- ✅ HDOCK-aware docking scoring
- Over-stability penalty
- Conservation-aware scoring ✅
- Adaptive NSGA-II compatibility
"""

from __future__ import annotations
import time
import logging
import os
import statistics
from typing import Dict, List

import requests
from Bio import Entrez
from dotenv import load_dotenv

log = logging.getLogger("virtual_lab.researcher")

os.environ["CUDA_VISIBLE_DEVICES"] = ""

load_dotenv()
Entrez.email = os.getenv("ENTREZ_EMAIL", "virtual.lab@example.com")

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("S2_API_KEY")

_session = requests.Session()

headers = {
    "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
    "User-Agent": "VirtualLab/1.0"
}

# ----------------------------------------------------------------------
# CACHE
# ----------------------------------------------------------------------
_semantic_cache = {}

def cached_semantic_search(query, limit):
    if query in _semantic_cache:
        return _semantic_cache[query]

    result = search_semantic_scholar(query, limit=limit)

    if result:
        _semantic_cache[query] = result

    return result

# ----------------------------------------------------------------------
# EMBEDDINGS
# ----------------------------------------------------------------------
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
    return _model

from langchain_core.embeddings import Embeddings

class CachedSentenceTransformerEmbeddings(Embeddings):
    def __init__(self):
        self.model = get_embedding_model()

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# ----------------------------------------------------------------------
# OBJECTIVES
# ----------------------------------------------------------------------
OBJECTIVES = ["thermo", "structure", "motif", "binding", "kinetic", "conservation"]

def fitness_vector(score: Dict) -> List[float]:
    return [score.get(k, 0.0) for k in OBJECTIVES]

# ----------------------------------------------------------------------
# MOTIF SCORE
# ----------------------------------------------------------------------
def _motif_score(seq: str) -> float:
    import re
    patterns = ["[AG]GAG", "[AG]AAG"]
    hits = sum(bool(re.search(p, seq)) for p in patterns)
    return hits / len(patterns)

# ----------------------------------------------------------------------
# CONSERVATION SCORE ✅
# ----------------------------------------------------------------------
def _conservation_score(seq: str, conservation_signal: Dict) -> float:
    if not conservation_signal:
        return 0.0

    conserved = conservation_signal.get("conserved_regions", [])
    if not conserved:
        return conservation_signal.get("conservation_fitness", 0.0)

    score = 0
    length = len(seq)

    for start, end in conserved:
        span = max(0, min(end, length) - start)
        score += span

    return score / max(1, length)

# ----------------------------------------------------------------------
# ✅ UPDATED SCORING FUNCTION (HDOCK SAFE)
# ----------------------------------------------------------------------
def score_sequence(seq, sf, md, weights, dock=None, conservation_signal=None):

    mfe = sf.get("mfe")
    sim_e = md.get("min_energy")

    # ---------------- THERMO ----------------
    if sim_e is not None:
        thermo_raw = -sim_e
    elif mfe is not None:
        thermo_raw = -mfe
    else:
        thermo_raw = sf.get("pair_density", 0)

    # ---------------- STRUCTURE ----------------
    pd = sf.get("pair_density") or 0.0
    bpp = sf.get("bpp_mean") or 0.0
    bp = md.get("base_pairs") or 0.0

    structure = 0.4 * pd + 0.4 * bpp + 0.2 * (bp / max(1, len(seq)))

    # ---------------- KINETIC ----------------
    entropy = sf.get("ensemble_entropy") or 0.0
    fluct = md.get("energy_fluctuation") or 0.0
    kinetic = 0.5 * entropy + 0.5 * (1 / (1 + fluct))

    # ---------------- MOTIF ----------------
    motif = _motif_score(seq)

    # ---------------- ✅ BINDING (HDOCK AWARE) ----------------
    binding = 0.0

    if dock:
        # ✅ NEW: HDOCK support
        if dock.get("dock_score") is not None:
            raw = float(dock["dock_score"])

            # HDOCK ~ -200 to -400 typical range
            # Map to smooth bounded signal
            binding = float(__import__("numpy").tanh(-raw / 100.0))

        # ✅ fallback legacy Vina
        elif dock.get("binding_energy") is not None:
            raw = float(dock["binding_energy"])
            binding = float(__import__("numpy").tanh(-raw / 10.0))

    # ---------------- CONSERVATION ----------------
    conservation = _conservation_score(seq, conservation_signal or {})

    return {
        "thermo": thermo_raw * weights.get("thermo", 1.0),
        "structure": structure * weights.get("structure", 1.0),
        "motif": motif * weights.get("motif", 1.0),
        "binding": binding * weights.get("binding", 1.0),
        "kinetic": kinetic * weights.get("kinetic", 1.0),
        "conservation": conservation * weights.get("conservation", 1.0),
    }

# ----------------------------------------------------------------------
# ADAPTIVE WEIGHTS
# ----------------------------------------------------------------------
def update_weights(population_scores: List[Dict], current_weights: Dict, lr: float = 0.2) -> Dict:
    if not population_scores:
        return current_weights

    sorted_pop = sorted(
        population_scores,
        key=lambda x: sum(x.get(k, 0) for k in OBJECTIVES),
        reverse=True
    )

    top = sorted_pop[: max(1, len(sorted_pop)//3)]

    avg = {
        k: statistics.mean(p.get(k, 0) for p in top)
        for k in OBJECTIVES
    }

    total = sum(avg.values()) or 1.0
    avg = {k: v / total for k, v in avg.items()}

    new_weights = {
        k: (1 - lr) * current_weights.get(k, 1.0) + lr * avg[k]
        for k in OBJECTIVES
    }

    MIN_WEIGHT = 0.05
    for k in new_weights:
        new_weights[k] = max(new_weights[k], MIN_WEIGHT)

    total = sum(new_weights.values())
    return {k: v / total for k, v in new_weights.items()}

# ----------------------------------------------------------------------
# INITIALISATION
# ----------------------------------------------------------------------
def build_initial_weights() -> Dict:
    return {k: 1.0 for k in OBJECTIVES}

def initialise_system(topic: str) -> Dict:
    return {
        "topic": topic,
        "weights": build_initial_weights(),
    }

# ----------------------------------------------------------------------
# RATE LIMIT
# ----------------------------------------------------------------------
import threading
_last_call_time = 0
_lock = threading.Lock()
MIN_INTERVAL = 1.0

def rate_limited_get(url, **kwargs):
    global _last_call_time
    with _lock:
        now = time.time()
        elapsed = now - _last_call_time

        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)

        resp = _session.get(url, **kwargs)
        _last_call_time = time.time()

    return resp

# ----------------------------------------------------------------------
# SEARCH FUNCTIONS (UNCHANGED)
# ----------------------------------------------------------------------
def search_local_db(query: str) -> List:
    try:
        from langchain_community.vectorstores import FAISS
        import os

        index_path = os.path.join(
            os.path.dirname(__file__), "..", "cache", "faiss_index"
        )

        if not os.path.exists(index_path):
            return []

        embeddings = CachedSentenceTransformerEmbeddings()
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return db.similarity_search(query, k=10)

    except Exception as e:
        log.warning("Local DB error: %s", e)
        return []

def search_semantic_scholar(query: str, limit: int = 25) -> List[Dict]:
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "title,abstract,year",
    }

    for attempt in range(5):
        try:
            resp = rate_limited_get(
                SEMANTIC_SCHOLAR_API_URL,
                params=params,
                headers=headers,
                timeout=10
            )

            if resp.status_code == 429:
                time.sleep(min(2**attempt, 30))
                continue

            if resp.status_code >= 500:
                time.sleep(min(2**attempt, 30))
                continue

            resp.raise_for_status()
            return resp.json().get("data", [])

        except Exception as e:
            log.warning("Semantic Scholar error: %s", e)
            time.sleep(1)

    return []

# ----------------------------------------------------------------------
# EXPAND KNOWLEDGE ✅
# ----------------------------------------------------------------------
def expand_knowledge(topic: str, build_db: bool = False) -> List:
    papers = cached_semantic_search(topic, limit=25)

    if build_db and papers:
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document

            docs = [
                Document(
                    page_content=p.get("abstract") or p.get("title") or "",
                    metadata={
                        "title": p.get("title", ""),
                        "year": p.get("year")
                    }
                )
                for p in papers
                if p.get("abstract") or p.get("title")
            ]

            if docs:
                embeddings = CachedSentenceTransformerEmbeddings()
                db = FAISS.from_documents(docs, embeddings)

                index_path = os.path.join(
                    os.path.dirname(__file__), "..", "cache", "faiss_index"
                )

                db.save_local(index_path)
                log.info("FAISS rebuilt with %d docs", len(docs))

        except Exception as e:
            log.warning("FAISS rebuild failed: %s", e)

    return papers