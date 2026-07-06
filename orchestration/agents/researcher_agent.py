from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, List

from langchain_core.messages import HumanMessage, SystemMessage

from VLAB2.orchestration.llm import get_llm
from VLAB2.orchestration.state_schema import (
    LabState,
    add_conversation_entry,
    record_stage_output,
)
from VLAB2.orchestration.utils.checkpointing import save_checkpoint
from VLAB2.orchestration.utils.literature_utils import (
    fallback_literature_query,
    keep_literature_text,
    keep_research_query,
    normalise_lit_query,
)
from VLAB2.research.research_agent_adaptive import (
    search_local_db,
    search_semantic_scholar,
)

log = logging.getLogger("virtual_lab")

__all__ = ["researcher_agent"]


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def _evidence_key(item: Any) -> str:
    """
    Generate a stable deduplication key from evidence dicts or legacy strings.
    """
    if isinstance(item, dict):
        doi = str(item.get("doi") or "").strip().lower()
        if doi:
            return f"doi::{doi}"

        title = str(item.get("title") or "").strip().lower()
        year = str(item.get("year") or "").strip()
        abstract = str(item.get("abstract") or item.get("text") or "")[:240].strip().lower()
        raw = f"{title}|{year}|{abstract}"
    else:
        raw = str(item or "")
        raw = raw.split(" | ")[0].replace("Title: ", "").strip().lower()

    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    return " ".join(raw.split())


def _evidence_to_text(item: Any) -> str:
    """
    Convert evidence dict/legacy string into compact display text.
    """
    if isinstance(item, dict):
        title = item.get("title") or "Unknown"
        abstract = item.get("abstract") or item.get("text") or item.get("content") or ""
        source = item.get("source") or "unknown"
        query = item.get("query") or item.get("search_query") or "unknown"
        return (
            f"Title: {title} | Source: {source} | Query: {query} | "
            f"Abstract: {str(abstract)[:300]}..."
        )

    return str(item)


def _safe_doc_title(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return metadata.get("title", "Unknown")


def _safe_doc_abstract(doc) -> str:
    return getattr(doc, "page_content", "") or ""


def _doc_to_evidence_dict(doc, fallback_query: str, profile: dict) -> dict:
    metadata = getattr(doc, "metadata", {}) or {}
    title = _safe_doc_title(doc)
    abstract = _safe_doc_abstract(doc)

    return {
        "schema_version": "literature_evidence.v2",
        "title": title,
        "abstract": abstract,
        "source": metadata.get("source", "local_faiss"),
        "query": metadata.get("query") or metadata.get("search_query") or fallback_query,
        "search_query": metadata.get("query") or metadata.get("search_query") or fallback_query,
        "year": metadata.get("year"),
        "doi": metadata.get("doi") or metadata.get("DOI"),
        "url": metadata.get("url"),
        "retrieval_score": metadata.get("score") or metadata.get("retrieval_score"),
        "topic_name": profile.get("topic_name"),
        "research_topic": profile.get("research_topic"),
        "virus_family": profile.get("virus_family"),
        "virus_genus": profile.get("virus_genus"),
        "virus_name": profile.get("virus_name"),
        "task_type": profile.get("task_type"),
    }


def _paper_to_evidence_dict(paper: dict, query: str, profile: dict) -> dict:
    return {
        "schema_version": "literature_evidence.v2",
        "title": paper.get("title", "Unknown"),
        "abstract": paper.get("abstract") or paper.get("text") or "",
        "source": paper.get("source", "semantic_scholar"),
        "query": paper.get("query") or paper.get("search_query") or query,
        "search_query": paper.get("query") or paper.get("search_query") or query,
        "year": paper.get("year"),
        "doi": paper.get("doi") or paper.get("DOI"),
        "url": paper.get("url"),
        "retrieval_score": paper.get("score") or paper.get("retrieval_score"),
        "topic_name": profile.get("topic_name"),
        "research_topic": profile.get("research_topic"),
        "virus_family": profile.get("virus_family"),
        "virus_genus": profile.get("virus_genus"),
        "virus_name": profile.get("virus_name"),
        "task_type": profile.get("task_type"),
    }


# ---------------------------------------------------------------------------
# Topic profile / query generation
# ---------------------------------------------------------------------------

def _topic_profile_from_state(state: LabState, hypothesis: str) -> dict:
    """
    Build a topic-aware literature profile so streaming adapts to the current run.
    """
    topic_name = state.get("topic_name") or state.get("name") or ""
    research_topic = (
        state.get("research_topic")
        or state.get("topic_description")
        or state.get("description")
        or hypothesis
    )
    virus_family = state.get("virus_family") or ""
    virus_genus = state.get("virus_genus") or ""
    virus_name = state.get("virus_name") or ""

    text = " ".join(
        str(x or "")
        for x in [
            topic_name,
            research_topic,
            hypothesis,
            virus_family,
            virus_genus,
            virus_name,
        ]
    ).lower()

    if (
        "inhibitor" in text
        or "small molecule" in text
        or "small-molecule" in text
        or "peptide inhibitor" in text
        or "competitive inhibition" in text
    ):
        task_type = "inhibitor_screening"
    else:
        task_type = "rna_docking"

    target_concepts = [
        "RNA binding",
        "viral capsid",
        "coat protein",
        "nucleocapsid",
        "RNA packaging",
    ]

    motif_concepts = [
        "RNA stem-loop",
        "RNA hairpin",
        "conserved RNA motif",
        "packaging signal",
    ]

    family_l = virus_family.lower()
    text_l = text

    if family_l == "coronaviridae" or "coronavirus" in text_l or "sars-cov" in text_l:
        target_concepts.extend(
            [
                "coronavirus nucleocapsid",
                "N protein RNA binding domain",
                "SARS-CoV nucleocapsid",
                "nucleoprotein RNA binding",
            ]
        )
        motif_concepts.extend(
            [
                "coronavirus 5 UTR stem-loop",
                "coronavirus packaging signal",
                "coronavirus genomic RNA motif",
            ]
        )

    if (
        family_l == "picornaviridae"
        or "poliovirus" in text_l
        or "enterovirus" in text_l
        or "picornavirus" in text_l
    ):
        target_concepts.extend(
            [
                "picornavirus capsid",
                "poliovirus capsid",
                "enterovirus capsid",
                "viral RNA packaging",
                "capsid RNA binding pocket",
            ]
        )
        motif_concepts.extend(
            [
                "IRES RNA structure",
                "picornavirus RNA stem-loop",
                "enterovirus RNA motif",
            ]
        )

    if task_type == "inhibitor_screening":
        target_concepts.extend(
            [
                "RNA binding pocket inhibitors",
                "small molecule inhibitors",
                "peptide inhibitors",
                "competitive inhibition",
            ]
        )

    return {
        "topic_name": topic_name,
        "research_topic": research_topic,
        "hypothesis": hypothesis,
        "virus_family": virus_family,
        "virus_genus": virus_genus,
        "virus_name": virus_name,
        "task_type": task_type,
        "target_concepts": sorted(set(target_concepts)),
        "motif_concepts": sorted(set(motif_concepts)),
    }


def _topic_fallback_queries(profile: dict) -> list[str]:
    """
    Deterministic topic-specific fallback queries.
    """
    family = profile.get("virus_family") or ""
    genus = profile.get("virus_genus") or ""
    virus = profile.get("virus_name") or ""
    task_type = profile.get("task_type")

    organism_terms = [x for x in [virus, genus, family] if x]
    organism = organism_terms[0] if organism_terms else "viral"

    queries: list[str] = []

    if task_type == "inhibitor_screening":
        queries.extend(
            [
                f"{organism} RNA binding protein inhibitors",
                f"{organism} capsid RNA binding inhibitors",
                f"{organism} RNA binding pocket small molecule",
                f"{organism} peptide inhibitors RNA binding",
                f"{family or organism} RNA packaging inhibitors",
            ]
        )
    else:
        queries.extend(
            [
                f"{organism} RNA stem loop binding",
                f"{organism} RNA packaging capsid protein",
                f"{organism} conserved RNA motif binding",
                f"{family or organism} nucleocapsid RNA binding",
                f"{family or organism} viral RNA stem loop",
            ]
        )

    if str(family).lower() == "coronaviridae":
        queries.extend(
            [
                "coronavirus nucleocapsid RNA binding",
                "coronavirus RNA packaging signal",
                "SARS CoV nucleocapsid RNA binding domain",
                "Coronaviridae RNA stem loop nucleocapsid",
            ]
        )

    if str(family).lower() == "picornaviridae":
        queries.extend(
            [
                "poliovirus capsid RNA binding",
                "enterovirus RNA packaging capsid",
                "picornavirus RNA stem loop capsid",
                "poliovirus RNA binding pocket inhibitors",
            ]
        )

    out: list[str] = []
    seen: set[str] = set()

    for q in queries:
        q = normalise_lit_query(q)
        if q and q not in seen and keep_research_query(q):
            seen.add(q)
            out.append(q)

    return out


def _build_query_bundle(
    state: LabState,
    hypothesis: str,
    profile: dict,
) -> tuple[str, list[str]]:
    """
    Generate a dynamic primary query plus deterministic topic-specific fallbacks.
    """
    query_msg = get_llm(temperature=0.0).invoke(
        [
            SystemMessage(
                content=(
                    "You are a biomedical literature specialist generating "
                    "PubMed-style keyword search queries.\n\n"
                    "Rules:\n"
                    "- 5–9 words, no punctuation, no quotes, no operators\n"
                    "- Must contain RNA\n"
                    "- Must contain at least one viral protein, capsid, packaging, "
                    "nucleocapsid, coat protein, or RNA-binding concept\n"
                    "- Must include the organism, viral genus, or viral family if provided\n"
                    "- If the task is inhibitor screening, include inhibitor, small molecule, "
                    "peptide, pocket, or competitive binding when appropriate\n"
                    "- Avoid: framework, multi-agent, LLM, manufacturing, clinical trial\n\n"
                    f"Topic profile:\n"
                    f"virus_family={profile.get('virus_family') or 'N/A'}\n"
                    f"virus_genus={profile.get('virus_genus') or 'N/A'}\n"
                    f"virus_name={profile.get('virus_name') or 'N/A'}\n"
                    f"task_type={profile.get('task_type')}\n"
                    f"target_concepts={', '.join(profile.get('target_concepts', []))}\n"
                    f"motif_concepts={', '.join(profile.get('motif_concepts', []))}\n\n"
                    "Return ONLY the query string."
                )
            ),
            HumanMessage(content=hypothesis),
        ]
    )

    query = normalise_lit_query(query_msg.content)

    if not keep_research_query(query):
        log.warning("Initial query rejected by filter: %s", query)
        query = fallback_literature_query(hypothesis)

    fallback_queries = _topic_fallback_queries(profile)

    query_bundle: list[str] = []
    seen_queries: set[str] = set()

    for q in [query] + fallback_queries:
        q = normalise_lit_query(q)
        if not q or q in seen_queries:
            continue
        if not keep_research_query(q):
            continue
        seen_queries.add(q)
        query_bundle.append(q)

    max_topic_queries = int(os.getenv("VLAB_LIT_MAX_TOPIC_STREAMS", "5"))
    query_bundle = query_bundle[:max_topic_queries]

    if not query_bundle and query:
        query_bundle = [query]

    return query, query_bundle


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

def researcher_agent(state: LabState) -> dict:
    """
    Literature evidence collection agent.

    Topic-reactive behaviour:
      - builds profile from topic/family/genus/virus/task
      - streams query bundle, not one hardcoded query
      - local FAISS search across full bundle
      - Semantic Scholar fallback across full bundle
      - stores evidence as metadata-rich dicts
    """
    try:
        from VLAB2.orchestration.skeptic_parser import parse_skeptic_output
        from VLAB2.research.literature_refiner import generate_refined_queries
        from VLAB2.research.streaming_literature_agent import start_streaming, rerank

        log.info("--- RESEARCHER AGENT: Querying Knowledge Base ---")

        streamed = set(state.get("streamed_queries", []))
        streaming_started = bool(state.get("streaming_started", False))

        seed_questions = state.get("seed_questions", [])
        idx = int(state.get("current_question_idx", 0) or 0)

        if seed_questions:
            idx = max(0, min(idx, len(seed_questions) - 1))
            hypothesis = seed_questions[idx]
        else:
            hypothesis = (
                state.get("hypothesis")
                or state.get("topic_description")
                or state.get("research_topic", "")
            )

        profile = _topic_profile_from_state(state, hypothesis)
        query, query_bundle = _build_query_bundle(state, hypothesis, profile)

        log.info("Primary literature query: %s", query)
        log.info("Literature query bundle: %s", query_bundle)

        started_any_stream = False

        for q in query_bundle:
            stream_key = f"{q}::{profile.get('topic_name')}::{profile.get('virus_family')}::{profile.get('task_type')}"
            if stream_key in streamed:
                log.info("Streaming already started for query/topic: %s", q)
                continue

            started = start_streaming(q, topic_profile=profile)
            streamed.add(stream_key)

            if started:
                streaming_started = True
                started_any_stream = True

        parsed = parse_skeptic_output(state.get("critique", ""))

        try:
            refined_queries = generate_refined_queries(parsed, hypothesis=hypothesis)
        except TypeError:
            refined_queries = generate_refined_queries(parsed)

        refined_queries = [normalise_lit_query(q) for q in refined_queries if q]
        refined_queries = [q for q in refined_queries if q]

        accepted_refined = 0
        max_refined = int(os.getenv("VLAB_LIT_MAX_REFINED_STREAMS", "3"))

        for q in refined_queries:
            if accepted_refined >= max_refined:
                break

            if not keep_research_query(q):
                log.info("Skipping broad/refined query: %s", q)
                continue

            stream_key = f"{q}::{profile.get('topic_name')}::{profile.get('virus_family')}::{profile.get('task_type')}"
            if stream_key in streamed:
                log.info("Refined query already streamed: %s", q)
                continue

            started = start_streaming(q, topic_profile=profile)
            streamed.add(stream_key)

            if started:
                streaming_started = True
                started_any_stream = True
                accepted_refined += 1
                query_bundle.append(q)

        if started_any_stream:
            time.sleep(float(os.getenv("VLAB_STREAM_WARMUP_SECONDS", "6.0")))

        # ------------------------------------------------------------
        # Search local DB across query bundle
        # ------------------------------------------------------------
        all_local_results = []

        for q in query_bundle:
            try:
                docs = search_local_db(q)
            except Exception as e:
                log.warning("Local DB search failed for query '%s': %s", q, e)
                docs = []

            if docs:
                all_local_results.extend(docs)

        if all_local_results:
            local_results = rerank(query, all_local_results, top_k=10, topic_profile=profile)
        else:
            fresh_results = []

            for q in query_bundle:
                try:
                    docs = search_local_db(q)
                except Exception as e:
                    log.warning("Fresh local DB search failed for query '%s': %s", q, e)
                    docs = []

                if docs:
                    fresh_results.extend(docs)

            local_results = rerank(query, fresh_results, top_k=10, topic_profile=profile) if fresh_results else []

        filtered_local_results = []

        for doc in local_results:
            title = _safe_doc_title(doc)
            abstract = _safe_doc_abstract(doc)

            if keep_literature_text(title, abstract):
                filtered_local_results.append(doc)
            else:
                log.info("Filtered local DB result as off-topic: %s", str(title)[:120])

        local_results = filtered_local_results[:10]
        evidence: List[dict] = []

        if local_results:
            log.info("Found %d relevant papers in local DB.", len(local_results))

            for doc in local_results:
                evidence.append(_doc_to_evidence_dict(doc, fallback_query=query, profile=profile))

        else:
            log.info("Nothing found in local DB. Falling back to Semantic Scholar...")

            ss_results_all = []

            for q in query_bundle:
                try:
                    ss_results = search_semantic_scholar(q, limit=10)
                except Exception as e:
                    log.warning("Semantic Scholar fallback failed for query '%s': %s", q, e)
                    ss_results = []

                for paper in ss_results:
                    if isinstance(paper, dict):
                        paper = dict(paper)
                        paper["query"] = q
                        paper["search_query"] = q
                        ss_results_all.append(paper)

            for paper in ss_results_all:
                title = paper.get("title", "Unknown")
                abstract = paper.get("abstract") or ""

                if not keep_literature_text(title, abstract):
                    log.info(
                        "Filtered Semantic Scholar result as off-topic: %s",
                        str(title)[:120],
                    )
                    continue

                evidence.append(
                    _paper_to_evidence_dict(
                        paper,
                        query=paper.get("query") or query,
                        profile=profile,
                    )
                )

        # ------------------------------------------------------------
        # Deduplicate, merge with existing evidence, cap
        # ------------------------------------------------------------
        existing_evidence = state.get("evidence", []) or []

        seen = set()
        deduped_new = []

        for item in existing_evidence:
            key = _evidence_key(item)
            if key:
                seen.add(key)

        for item in evidence:
            key = _evidence_key(item)
            if not key or key in seen:
                continue

            seen.add(key)
            deduped_new.append(item)

        combined = existing_evidence + deduped_new

        seen = set()
        evidence_out = []

        for item in combined:
            key = _evidence_key(item)
            if not key or key in seen:
                continue

            seen.add(key)
            evidence_out.append(item)

        evidence_cap = int(os.getenv("VLAB_RESEARCH_EVIDENCE_CAP", "20"))
        evidence_out = evidence_out[-evidence_cap:]

        evidence_text = "\n\n".join(_evidence_to_text(x) for x in evidence_out)

        log.info("Final evidence count: %d", len(evidence_out))

        result = {
            "evidence": evidence_out,
            "streaming_started": streaming_started,
            "streamed_queries": list(streamed),
            "research_query": query,
            "literature_query_bundle": query_bundle,
            "literature_topic_profile": profile,
            "stage_outputs": [
                record_stage_output(
                    state,
                    "researcher",
                    evidence_text,
                    summary=f"Collected {len(evidence_out)} evidence items",
                    metadata={
                        "query": query,
                        "query_bundle": query_bundle,
                        "topic_profile": profile,
                        "new_evidence_count": len(deduped_new),
                        "total_evidence_count": len(evidence_out),
                        "streamed_queries": list(streamed),
                        "accepted_refined_queries": accepted_refined,
                    },
                )
            ],
            "conversation_history": [
                add_conversation_entry(
                    state,
                    "assistant",
                    "Evidence collected.",
                    "researcher",
                )
            ],
        }

        data_collector = state.get("data_collector")

        if data_collector is not None:
            try:
                data_collector.capture_agent_interaction(
                    "Researcher",
                    "Gather literature evidence",
                    evidence_text,
                )
            except Exception as e:
                log.warning("Researcher data collection failed/non-fatal: %s", e)

        save_checkpoint({**state, **result})
        return result

    except Exception as e:
        log.exception("RESEARCHER AGENT ERROR")

        return {
            "evidence": [f"Researcher failed: {e}"],
            "streaming_started": state.get("streaming_started", False),
            "streamed_queries": list(state.get("streamed_queries", [])),
            "research_query": "",
        }