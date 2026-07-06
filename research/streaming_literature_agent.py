"""
streaming_literature_agent.py

Background literature ingestion for Virtual Lab.

Responsibilities:
  - Start one background stream per normalized query/topic profile.
  - Fetch papers using cached_semantic_search.
  - Filter for topic-aware biomedical RNA/capsid/nucleocapsid/inhibitor relevance.
  - Deduplicate papers by DOI, normalized title/year/source, and content hash.
  - Append genuinely new documents to FAISS.
  - Optionally rerank retrieved local documents with CrossEncoder.

Important behaviour:
  - "No new papers" is logged at DEBUG, not INFO, to avoid log spam.
  - FAISS path is taken from research_agent_adaptive if available, otherwise
    from VLAB_FAISS_INDEX_PATH, otherwise VLAB2/cache/faiss_index.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Set

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

import VLAB2.research.research_agent_adaptive as research_mod

from VLAB2.research.research_agent_adaptive import (
    CachedSentenceTransformerEmbeddings,
    cached_semantic_search,
)

log = logging.getLogger("virtual_lab.streaming")


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_stream_cache: Set[str] = set()
_active_queries: Dict[str, threading.Event] = {}
_active_threads: Dict[str, threading.Thread] = {}
_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_FAISS_INDEX_PATH = str(
    Path(__file__).resolve().parents[1] / "cache" / "faiss_index"
)

FAISS_INDEX_PATH = (
    os.getenv("VLAB_FAISS_INDEX_PATH")
    or getattr(research_mod, "FAISS_INDEX_PATH", None)
    or getattr(research_mod, "FAISS_PATH", None)
    or DEFAULT_FAISS_INDEX_PATH
)

STREAM_INTERVAL = int(os.getenv("VLAB_STREAM_INTERVAL", "30"))
STREAM_BATCH_SIZE = int(os.getenv("VLAB_STREAM_BATCH_SIZE", "20"))
STREAM_MAX_IDLE = int(os.getenv("VLAB_STREAM_MAX_IDLE", "0"))
STREAM_MAX_CYCLES = int(os.getenv("VLAB_STREAM_MAX_CYCLES", "0"))
DEDUP_ABSTRACT_PREFIX_LEN = int(os.getenv("VLAB_LIT_DEDUP_ABSTRACT_PREFIX_LEN", "500"))

# If enabled, inhibitor-screening topics require closer inhibitor/target relevance.
STRICT_INHIBITOR_FILTER = os.getenv("VLAB_LIT_STRICT_INHIBITOR_FILTER", "1").strip() != "0"


BIO_TERMS = [
    "stem-loop",
    "stem loop",
    "5' utr",
    "5 utr",
    "utr",
    "rna",
    "ribonucleic",
    "capsid",
    "coat protein",
    "ires",
    "internal ribosome entry site",
    "viral",
    "virus",
    "virion",
    "packaging",
    "assembly",
    "rna binding",
    "rna-binding",
    "rna-protein",
    "rna protein",
    "rna packaging",
    "rna packaging signal",
    "genomic rna",
    "subgenomic rna",
    "nucleocapsid",
    "nucleoprotein",
    "n protein",
    "n-terminal domain",
    "coronavirus",
    "coronaviridae",
    "sars-cov",
    "sars cov",
    "picornaviridae",
    "picornavirus",
    "parechovirus",
    "hpev",
    "enterovirus",
    "poliovirus",
    "rhinovirus",
    "ev-a71",
    "inhibitor",
    "inhibitors",
    "small molecule",
    "small-molecule",
    "compound",
    "antiviral",
    "peptide inhibitor",
    "competitive inhibition",
    "binding pocket",
    "capsid-binding",
    "capsid binding",
]

DOMAIN_ANCHORS = [
    "virus",
    "viral",
    "virion",
    "capsid",
    "coat protein",
    "packaging",
    "assembly",
    "rna binding",
    "rna-binding",
    "rna-protein",
    "rna protein",
    "rna packaging",
    "rna binding pocket",
    "nucleocapsid",
    "nucleoprotein",
    "n protein",
    "coronavirus",
    "coronaviridae",
    "sars-cov",
    "sars cov",
    "picornaviridae",
    "picornavirus",
    "parechovirus",
    "hpev",
    "enterovirus",
    "poliovirus",
    "rhinovirus",
    "ev-a71",
]

BAD_TERMS = [
    "multi-agent reinforcement learning",
    "multi-agent llm",
    "large language model",
    "llm planning",
    "jailbreak",
    "manufacturing systems",
    "pose graph",
    "clinical trial multi-agent",
    "robot",
    "slam",
    "phosphate glass",
    "plantaricin",
    "anti-cancer",
    "anticancer",
]


# ---------------------------------------------------------------------------
# Normalisation / dedupe
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalise_title(title: str) -> str:
    return _normalise_text(title)


def _stable_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _paper_key(p: dict) -> str:
    title = _normalise_title(p.get("title") or "")
    doi = _normalise_text(p.get("doi") or p.get("DOI") or "")
    year = str(p.get("year") or "").strip()
    source = _normalise_text(p.get("source") or "semantic_search")

    if doi:
        return f"doi::{doi}"

    abstract = _normalise_text(p.get("abstract") or "")[:DEDUP_ABSTRACT_PREFIX_LEN]
    content_sig = _stable_hash(f"{title}|{year}|{abstract}")

    if title:
        return f"title::{title}|year::{year}|sig::{content_sig}"

    return f"source::{source}|year::{year}|sig::{content_sig}"


def _doc_key(d: Document) -> str:
    metadata = d.metadata or {}
    title = _normalise_title(metadata.get("title") or "")
    doi = _normalise_text(metadata.get("doi") or metadata.get("DOI") or "")
    year = str(metadata.get("year") or "").strip()
    source = _normalise_text(metadata.get("source") or "semantic_search")
    abstract = _normalise_text(d.page_content or "")[:DEDUP_ABSTRACT_PREFIX_LEN]
    content_sig = _stable_hash(f"{title}|{year}|{abstract}")

    if doi:
        return f"doi::{doi}"

    if title:
        return f"title::{title}|year::{year}|sig::{content_sig}"

    return f"source::{source}|year::{year}|sig::{content_sig}"


def _load_existing_faiss_keys(db: FAISS) -> Set[str]:
    keys: Set[str] = set()

    try:
        docs = list(db.docstore._dict.values())
    except Exception:
        docs = []

    for d in docs:
        try:
            keys.add(_doc_key(d))
        except Exception:
            continue

    return keys


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

def _load_reranker():
    if CrossEncoder is None:
        log.warning("CrossEncoder unavailable because sentence_transformers import failed.")
        return None

    model_name = os.getenv(
        "VLAB_RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    local_only = os.getenv("VLAB_RERANKER_LOCAL_ONLY", "1").strip() == "1"

    try:
        reranker_model = CrossEncoder(
            model_name,
            local_files_only=local_only,
        )
        log.info("CrossEncoder reranker loaded successfully: %s", model_name)
        return reranker_model

    except Exception as e:
        log.warning("Reranker unavailable: %s", e)
        return None


reranker = _load_reranker()


# ---------------------------------------------------------------------------
# Topic-aware relevance
# ---------------------------------------------------------------------------

def _topic_profile_terms(topic_profile: dict | None) -> list[str]:
    if not topic_profile:
        return []

    terms: list[str] = []

    for key in ("virus_family", "virus_genus", "virus_name"):
        value = topic_profile.get(key)
        if value:
            terms.append(str(value).lower())

    terms.extend(
        str(x).lower()
        for x in topic_profile.get("target_concepts", []) or []
    )
    terms.extend(
        str(x).lower()
        for x in topic_profile.get("motif_concepts", []) or []
    )

    return terms


def _query_tokens(query: str | None) -> list[str]:
    q = (query or "").lower()
    return [
        tok for tok in re.split(r"[^a-z0-9]+", q)
        if len(tok) >= 4
    ]


def _is_inhibitor_topic(topic_profile: dict | None, query: str | None = None) -> bool:
    profile_task = (topic_profile or {}).get("task_type")
    if profile_task == "inhibitor_screening":
        return True

    q = (query or "").lower()
    return (
        "inhibitor" in q
        or "small molecule" in q
        or "small-molecule" in q
        or "peptide inhibitor" in q
        or "competitive" in q
    )


def _passes_inhibitor_topic_filter(text: str, query: str | None, topic_profile: dict | None) -> bool:
    """
    Stricter filter for inhibitor-screening topics.

    Goal:
      Prefer capsid/nucleocapsid/RNA-binding-pocket inhibitor papers.

    Avoid:
      Polymerase/OSBP/receptor/membrane-only inhibitor papers unless they also
      explicitly discuss capsid, nucleocapsid, or RNA-binding pocket relevance.
    """
    if not STRICT_INHIBITOR_FILTER:
        return True

    t = text.lower()
    q = (query or "").lower()

    inhibitor_terms = [
        "inhibitor",
        "inhibitors",
        "small molecule",
        "small-molecule",
        "compound",
        "compounds",
        "antiviral",
        "capsid-binding",
        "capsid binding",
        "binding pocket",
        "competitive",
        "outcompeting",
    ]

    desired_target_terms = [
        "capsid",
        "coat protein",
        "nucleocapsid",
        "nucleoprotein",
        "n protein",
        "rna binding",
        "rna-binding",
        "rna binding domain",
        "rna-binding domain",
        "rna binding pocket",
        "rna-binding pocket",
        "enterovirus capsid",
        "poliovirus capsid",
        "picornavirus capsid",
        "rhinovirus capsid",
        "viral capsid",
    ]

    bad_for_this_task = [
        "polymerase",
        "rdrp",
        "rna-dependent rna polymerase",
        "osbp",
        "oxysterol-binding protein",
        "receptor",
        "cellular receptor",
        "membrane protein",
        "scarb2",
        "protease",
    ]

    has_inhibitor = any(term in t for term in inhibitor_terms)
    has_desired_target = any(term in t for term in desired_target_terms)
    has_task_bad = any(term in t for term in bad_for_this_task)

    # Query itself can express inhibitor intent even if abstract uses "compound".
    query_has_inhibitor_intent = any(term in q for term in inhibitor_terms)

    # Strong accept.
    if has_inhibitor and has_desired_target:
        return True

    # For target-specific inhibitor topics, reject host/receptor/polymerase-only hits.
    if has_task_bad and not has_desired_target:
        return False

    # If query asks for inhibitors, require either inhibitor marker or desired target.
    if query_has_inhibitor_intent:
        return has_inhibitor or has_desired_target

    # Otherwise defer to generic biomedical relevance.
    return True


def is_biomed_relevant(
    text: str,
    query: str | None = None,
    topic_profile: dict | None = None,
) -> bool:
    if not text:
        return False

    t = text.lower()

    has_bad_term = any(term in t for term in BAD_TERMS)
    if has_bad_term:
        return False

    has_rna_or_structure = any(term in t for term in BIO_TERMS)
    has_domain_anchor = any(term in t for term in DOMAIN_ANCHORS)

    profile_terms = _topic_profile_terms(topic_profile)
    query_tokens = _query_tokens(query)

    has_profile_match = any(term and term in t for term in profile_terms)
    has_query_overlap = sum(1 for tok in query_tokens if tok in t) >= 2

    generic_pass = (
        has_rna_or_structure
        and (has_domain_anchor or has_profile_match or has_query_overlap)
    )

    if not generic_pass:
        return False

    if _is_inhibitor_topic(topic_profile, query=query):
        return _passes_inhibitor_topic_filter(
            text=text,
            query=query,
            topic_profile=topic_profile,
        )

    return True


def paper_to_document(
    p: dict,
    query: str | None = None,
    topic_profile: dict | None = None,
) -> Document | None:
    title = p.get("title") or ""
    abstract = p.get("abstract") or ""
    year = p.get("year")

    combined_text = f"{title}\n\n{abstract}".strip()

    if not combined_text:
        return None

    if not is_biomed_relevant(
        combined_text,
        query=query,
        topic_profile=topic_profile,
    ):
        log.debug("Filtered non-biomedical/topic-mismatched paper: %s", title[:120])
        return None

    pkey = _paper_key(p)
    profile = topic_profile or {}

    return Document(
        page_content=combined_text,
        metadata={
            "title": title,
            "year": year,
            "source": p.get("source", "semantic_search"),
            "doi": p.get("doi") or p.get("DOI"),
            "pmid": p.get("pmid") or p.get("PMID"),
            "url": p.get("url"),
            "dedup_key": pkey,
            "query": query,
            "search_query": query,
            "topic_name": profile.get("topic_name"),
            "research_topic": profile.get("research_topic"),
            "virus_family": profile.get("virus_family"),
            "virus_genus": profile.get("virus_genus"),
            "virus_name": profile.get("virus_name"),
            "task_type": profile.get("task_type"),
        },
    )


# ---------------------------------------------------------------------------
# FAISS append
# ---------------------------------------------------------------------------

def append_to_faiss(docs: List[Document]) -> int:
    if not docs:
        return 0

    embeddings = CachedSentenceTransformerEmbeddings()
    index_path = Path(FAISS_INDEX_PATH)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        try:
            db = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            existing_keys = _load_existing_faiss_keys(db)

            unique_docs = []
            seen_batch = set()

            for d in docs:
                key = d.metadata.get("dedup_key") or _doc_key(d)

                if key in existing_keys or key in seen_batch:
                    log.debug(
                        "Skipping duplicate FAISS document: %s",
                        d.metadata.get("title", "")[:120],
                    )
                    continue

                d.metadata["dedup_key"] = key
                seen_batch.add(key)
                unique_docs.append(d)

            if not unique_docs:
                return 0

            db.add_documents(unique_docs)
            db.save_local(str(index_path))
            return len(unique_docs)

        except Exception as e:
            log.warning(
                "Creating new FAISS index at %s because load failed: %s",
                index_path,
                e,
            )

            unique_docs = []
            seen_batch = set()

            for d in docs:
                key = d.metadata.get("dedup_key") or _doc_key(d)

                if key in seen_batch:
                    continue

                d.metadata["dedup_key"] = key
                seen_batch.add(key)
                unique_docs.append(d)

            if not unique_docs:
                return 0

            db = FAISS.from_documents(unique_docs, embeddings)
            db.save_local(str(index_path))
            return len(unique_docs)


# ---------------------------------------------------------------------------
# Streaming worker
# ---------------------------------------------------------------------------

def stream_papers(
    query: str,
    stop_event: threading.Event,
    interval: int = STREAM_INTERVAL,
    batch_size: int = STREAM_BATCH_SIZE,
    topic_profile: dict | None = None,
):
    """
    Continuously pull papers in background and append relevant biomedical papers to FAISS.
    """
    normalized_query = " ".join((query or "").split()).strip()
    key = _stream_key(normalized_query, topic_profile)

    idle_cycles = 0
    total_cycles = 0

    log.info("Streaming worker active for query: %s", normalized_query)

    try:
        while not stop_event.is_set():
            total_cycles += 1

            try:
                try:
                    papers = cached_semantic_search(normalized_query, limit=batch_size)
                except TypeError:
                    papers = cached_semantic_search(normalized_query, batch_size)

                papers = papers or []
                new_docs = []
                batch_keys: Set[str] = set()

                with _lock:
                    for p in papers:
                        if not isinstance(p, dict):
                            continue

                        pkey = _paper_key(p)

                        if not pkey or pkey in _stream_cache or pkey in batch_keys:
                            continue

                        doc = paper_to_document(
                            p,
                            query=normalized_query,
                            topic_profile=topic_profile,
                        )

                        if doc is None:
                            continue

                        batch_keys.add(pkey)
                        _stream_cache.add(pkey)
                        new_docs.append(doc)

                if new_docs:
                    added = append_to_faiss(new_docs)

                    if added:
                        idle_cycles = 0
                        log.info(
                            "Streaming: added %d new biomedical papers for query: %s",
                            added,
                            normalized_query,
                        )
                    else:
                        idle_cycles += 1
                        log.debug(
                            "Streaming: candidate papers already existed in FAISS for query: %s "
                            "(idle cycle %d)",
                            normalized_query,
                            idle_cycles,
                        )
                else:
                    idle_cycles += 1
                    log.debug(
                        "Streaming: no new relevant biomedical papers for query: %s "
                        "(idle cycle %d)",
                        normalized_query,
                        idle_cycles,
                    )

                if STREAM_MAX_IDLE > 0 and idle_cycles >= STREAM_MAX_IDLE:
                    log.info(
                        "Stopping stream for query after %d idle cycles: %s",
                        idle_cycles,
                        normalized_query,
                    )
                    break

                if STREAM_MAX_CYCLES > 0 and total_cycles >= STREAM_MAX_CYCLES:
                    log.info(
                        "Stopping stream for query after %d total cycles: %s",
                        total_cycles,
                        normalized_query,
                    )
                    break

            except Exception as e:
                log.warning(
                    "Streaming ingestion error for query '%s': %s",
                    normalized_query,
                    e,
                )

            stop_event.wait(interval)

    finally:
        with _lock:
            _active_queries.pop(key, None)
            _active_threads.pop(key, None)

        log.info("Streaming worker stopped for query: %s", normalized_query)


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    docs: List[Document],
    top_k: int = 5,
    topic_profile: dict | None = None,
):
    if not docs:
        return []

    unique_docs = []
    seen = set()

    for d in docs:
        key = d.metadata.get("dedup_key") or _doc_key(d)
        if key in seen:
            continue

        seen.add(key)
        unique_docs.append(d)

    docs = [
        d for d in unique_docs
        if is_biomed_relevant(
            f"{d.metadata.get('title', '')}\n\n{d.page_content}",
            query=query,
            topic_profile=topic_profile,
        )
    ]

    if not docs:
        log.warning("Rerank received no biomedical/topic-relevant docs after filtering.")
        return []

    if reranker is None:
        return docs[:top_k]

    pairs = [
        (
            query,
            f"{d.metadata.get('title', '')}\n\n{d.page_content}",
        )
        for d in docs
    ]

    try:
        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        return [d for d, _ in ranked[:top_k]]

    except Exception as e:
        log.warning("Reranking failed, returning first %d docs: %s", top_k, e)
        return docs[:top_k]


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _stream_key(query: str, topic_profile: dict | None = None) -> str:
    query = " ".join((query or "").split()).strip().lower()

    if not topic_profile:
        return query

    topic_key = "|".join(
        str(topic_profile.get(k) or "")
        for k in (
            "topic_name",
            "virus_family",
            "virus_genus",
            "virus_name",
            "task_type",
        )
    ).lower()

    return f"{query}::{topic_key}"


def start_streaming(query: str, topic_profile: dict | None = None):
    query = " ".join((query or "").split()).strip()

    if not query:
        log.warning("Refusing to start streaming for empty query.")
        return False

    key = _stream_key(query, topic_profile)

    with _lock:
        if key in _active_queries:
            log.info("Streaming already active for query/topic: %s", query)
            return False

        stop_event = threading.Event()
        _active_queries[key] = stop_event

        thread = threading.Thread(
            target=stream_papers,
            args=(query, stop_event),
            kwargs={"topic_profile": topic_profile},
            daemon=True,
            name=f"literature-stream::{key[:40]}",
        )

        _active_threads[key] = thread
        thread.start()

    log.info("Started streaming literature ingestion for query: %s", query)
    return True


def stop_streaming(query: str, topic_profile: dict | None = None) -> bool:
    query = " ".join((query or "").split()).strip()

    if not query:
        return False

    key = _stream_key(query, topic_profile)

    with _lock:
        stop_event = _active_queries.get(key)

        if stop_event is None:
            return False

        stop_event.set()
        return True


def stop_all_streaming() -> int:
    with _lock:
        events = list(_active_queries.values())

        for ev in events:
            ev.set()

        return len(events)


def active_streams() -> list[str]:
    with _lock:
        return list(_active_queries.keys())


def get_faiss_index_path() -> str:
    return FAISS_INDEX_PATH