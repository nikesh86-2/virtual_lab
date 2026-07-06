from __future__ import annotations

import logging
import os
import statistics
import threading
import time
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
    "User-Agent": "VirtualLab/1.0",
}

_semantic_cache = {}


def cached_semantic_search(query, limit):
    cache_key = (str(query or "").strip().lower(), int(limit or 25))

    if cache_key in _semantic_cache:
        return _semantic_cache[cache_key]

    result = search_semantic_scholar(query, limit=limit)

    if result:
        _semantic_cache[cache_key] = result

    return result


_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
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


OBJECTIVES = ["thermo", "structure", "motif", "binding", "kinetic", "conservation"]


def fitness_vector(score: Dict) -> List:
    return [score.get(k, 0.0) for k in OBJECTIVES]


def _motif_score(seq: str) -> float:
    import re

    patterns = ["[AG]GAG", "[AG]AAG"]
    hits = sum(bool(re.search(p, seq)) for p in patterns)
    return hits / len(patterns)


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


def score_sequence(seq, sf, md, weights, dock=None, conservation_signal=None):
    mfe = sf.get("mfe")
    sim_e = md.get("min_energy")

    if sim_e is not None:
        thermo_raw = -sim_e
    elif mfe is not None:
        thermo_raw = -mfe
    else:
        thermo_raw = sf.get("pair_density", 0)

    pd = sf.get("pair_density") or 0.0
    bpp = sf.get("bpp_mean") or 0.0
    bp = md.get("base_pairs") or 0.0

    structure = 0.4 * pd + 0.4 * bpp + 0.2 * (bp / max(1, len(seq)))

    entropy = sf.get("ensemble_entropy") or 0.0
    fluct = md.get("energy_fluctuation") or 0.0
    kinetic = 0.5 * entropy + 0.5 * (1 / (1 + fluct))

    motif = _motif_score(seq)

    binding = 0.0

    if dock:
        if dock.get("dock_score") is not None:
            raw = float(dock["dock_score"])
            binding = float(__import__("numpy").tanh(-raw / 100.0))
        elif dock.get("binding_energy") is not None:
            raw = float(dock["binding_energy"])
            binding = float(__import__("numpy").tanh(-raw / 10.0))

    conservation = _conservation_score(seq, conservation_signal or {})

    return {
        "thermo": thermo_raw * weights.get("thermo", 1.0),
        "structure": structure * weights.get("structure", 1.0),
        "motif": motif * weights.get("motif", 1.0),
        "binding": binding * weights.get("binding", 1.0),
        "kinetic": kinetic * weights.get("kinetic", 1.0),
        "conservation": conservation * weights.get("conservation", 1.0),
    }


def update_weights(population_scores: List[Dict], current_weights: Dict, lr: float = 0.2) -> Dict:
    if not population_scores:
        return current_weights

    sorted_pop = sorted(
        population_scores,
        key=lambda x: sum(x.get(k, 0) for k in OBJECTIVES),
        reverse=True,
    )

    top = sorted_pop[: max(1, len(sorted_pop) // 3)]

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

    min_weight = 0.05
    for k in new_weights:
        new_weights[k] = max(new_weights[k], min_weight)

    total = sum(new_weights.values())
    return {k: v / total for k, v in new_weights.items()}


def build_initial_weights() -> Dict:
    return {k: 1.0 for k in OBJECTIVES}


def initialise_system(topic: str) -> Dict:
    return {
        "topic": topic,
        "weights": build_initial_weights(),
    }


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


def _faiss_index_path() -> str:
    try:
        from VLAB2.research.streaming_literature_agent import get_faiss_index_path

        return get_faiss_index_path()
    except Exception:
        return (
            os.getenv("VLAB_FAISS_INDEX_PATH")
            or os.getenv("FAISS_INDEX_PATH")
            or os.path.join(os.path.dirname(__file__), "..", "cache", "faiss_index")
        )


def search_local_db(query: str) -> List:
    try:
        from langchain_community.vectorstores import FAISS

        index_path = _faiss_index_path()

        if not os.path.exists(index_path):
            return []

        embeddings = CachedSentenceTransformerEmbeddings()
        db = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return db.similarity_search(query, k=10)

    except Exception as e:
        log.warning("Local DB error: %s", e)
        return []


def search_semantic_scholar(query: str, limit: int = 25) -> List:
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "title,abstract,year,url,externalIds",
    }

    for attempt in range(5):
        try:
            resp = rate_limited_get(
                SEMANTIC_SCHOLAR_API_URL,
                params=params,
                headers=headers,
                timeout=10,
            )

            if resp.status_code == 429:
                time.sleep(min(2**attempt, 30))
                continue

            if resp.status_code >= 500:
                time.sleep(min(2**attempt, 30))
                continue

            resp.raise_for_status()
            data = resp.json().get("data", []) or []

            out = []
            for p in data:
                ext = p.get("externalIds") or {}
                row = {
                    "title": p.get("title"),
                    "abstract": p.get("abstract"),
                    "year": p.get("year"),
                    "url": p.get("url"),
                    "doi": ext.get("DOI"),
                    "pmid": ext.get("PubMed"),
                    "source": "semantic_scholar",
                }
                out.append(row)

            return out

        except Exception as e:
            log.warning("Semantic Scholar error: %s", e)
            time.sleep(1)

    return []


def expand_knowledge(topic: str, build_db: bool = False) -> List:
    papers = cached_semantic_search(topic, limit=25)

    if build_db and papers:
        try:
            from langchain_core.documents import Document
            from VLAB2.research.streaming_literature_agent import append_to_faiss

            docs = []

            for p in papers:
                title = p.get("title") or ""
                abstract = p.get("abstract") or ""
                text = f"{title}\n\n{abstract}".strip()

                if not text:
                    continue

                doi = p.get("doi") or p.get("DOI")
                year = p.get("year")

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "title": title,
                            "abstract": abstract,
                            "year": year,
                            "doi": doi,
                            "url": p.get("url"),
                            "source": p.get("source", "semantic_scholar"),
                            "query": topic,
                            "search_query": topic,
                        },
                    )
                )

            if docs:
                added = append_to_faiss(docs)
                log.info("FAISS appended with %d docs for topic '%s'", added, topic)

        except Exception as e:
            log.warning("FAISS append failed: %s", e)

    return papers