from __future__ import annotations

import logging
import os
import re
import time
from typing import List

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


def _evidence_key(item: str) -> str:
    """
    Generate a stable deduplication key from a stored evidence string.
    """
    title = item.split(" | ")[0].replace("Title: ", "").strip().lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return " ".join(title.split())


def _safe_doc_title(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return metadata.get("title", "Unknown")


def _safe_doc_abstract(doc) -> str:
    return getattr(doc, "page_content", "") or ""


def researcher_agent(state: LabState) -> dict:
    """
    Literature evidence collection agent.

    Preserved original behaviour:
      - seed-question aware hypothesis selection
      - constrained biomedical query generation
      - streaming ingestion for primary + refined queries
      - Skeptic-driven refined queries
      - local FAISS search first
      - local reranking
      - fresh local retry if no results
      - Semantic Scholar fallback
      - off-topic filtering
      - merge with existing evidence
      - cap evidence to latest 20 items
      - stage output, conversation history, checkpoint
    """
    try:
        from VLAB2.orchestration.skeptic_parser import parse_skeptic_output
        from VLAB2.research.literature_refiner import generate_refined_queries
        from VLAB2.research.streaming_literature_agent import start_streaming, rerank

        log.info("--- RESEARCHER AGENT: Querying Knowledge Base ---")

        streamed = set(state.get("streamed_queries", []))
        streaming_started = bool(state.get("streaming_started", False))

        # ------------------------------------------------------------
        # Preserve seed-question handling from original monolith
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Generate constrained primary literature query
        # ------------------------------------------------------------
        query_msg = get_llm(temperature=0.0).invoke(
            [
                SystemMessage(
                    content=(
                        "You are a biomedical literature specialist generating "
                        "PubMed-style keyword search queries.\n\n"
                        "Rules:\n"
                        "- 5–8 words, no punctuation, no quotes, no operators\n"
                        "- Must contain RNA\n"
                        "- Must contain at least one viral/capsid/packaging concept\n"
                        "- Must include organism or viral family if named\n"
                        "- Prefer: RNA stem-loop, viral capsid, RNA packaging, "
                        "coat protein, RNA binding, Picornaviridae, HPeV1\n"
                        "- Avoid: framework, multi-agent, LLM, manufacturing, clinical trial\n\n"
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

        log.info("    Search query: %s", query)

        # ------------------------------------------------------------
        # Start streaming literature ingestion for primary query
        # ------------------------------------------------------------
        started_any_stream = False

        if keep_research_query(query) and query not in streamed:
            start_streaming(query)
            streamed.add(query)
            streaming_started = True
            started_any_stream = True
        elif query in streamed:
            log.info("Streaming already started for query: %s", query)

        # ------------------------------------------------------------
        # Generate and stream Skeptic-refined queries
        # ------------------------------------------------------------
        parsed = parse_skeptic_output(state.get("critique", ""))

        try:
            refined_queries = generate_refined_queries(parsed, hypothesis=hypothesis)
        except TypeError:
            refined_queries = generate_refined_queries(parsed)

        refined_queries = [normalise_lit_query(q) for q in refined_queries if q]
        refined_queries = [q for q in refined_queries if q]

        accepted_refined = 0

        for q in refined_queries:
            if accepted_refined >= 3:
                break

            if not keep_research_query(q):
                log.info("Skipping broad/refined query: %s", q)
                continue

            if q in streamed:
                log.info("Refined query already streamed: %s", q)
                continue

            start_streaming(q)
            streamed.add(q)
            streaming_started = True
            started_any_stream = True
            accepted_refined += 1

        if started_any_stream:
            time.sleep(float(os.getenv("VLAB_STREAM_WARMUP_SECONDS", "6.0")))

        # ------------------------------------------------------------
        # Search local DB
        # ------------------------------------------------------------
        try:
            local_results = search_local_db(query)
        except Exception as e:
            log.warning("Local DB search failed for query '%s': %s", query, e)
            local_results = []

        if local_results:
            local_results = rerank(query, local_results, top_k=7)
        else:
            # Preserve original fresh local retry after streaming warmup
            try:
                fresh_results = search_local_db(query)
            except Exception as e:
                log.warning("Fresh local DB search failed for query '%s': %s", query, e)
                fresh_results = []

            local_results = rerank(query, fresh_results, top_k=7) if fresh_results else []

        # ------------------------------------------------------------
        # Filter local results
        # ------------------------------------------------------------
        filtered_local_results = []

        for doc in local_results:
            title = _safe_doc_title(doc)
            abstract = _safe_doc_abstract(doc)

            if keep_literature_text(title, abstract):
                filtered_local_results.append(doc)
            else:
                log.info("Filtered local DB result as off-topic: %s", str(title)[:120])

        local_results = filtered_local_results[:7]
        evidence: List[str] = []

        if local_results:
            log.info("Found %d relevant papers in local DB.", len(local_results))

            for doc in local_results:
                title = _safe_doc_title(doc)
                abstract = _safe_doc_abstract(doc)

                evidence.append(
                    f"Title: {title} | Abstract: {abstract[:300]}..."
                )

        else:
            # --------------------------------------------------------
            # Semantic Scholar fallback
            # --------------------------------------------------------
            log.info("Nothing found in local DB. Falling back to Semantic Scholar...")

            try:
                ss_results = search_semantic_scholar(query, limit=10)
            except Exception as e:
                log.warning("Semantic Scholar fallback failed for query '%s': %s", query, e)
                ss_results = []

            for paper in ss_results:
                title = paper.get("title", "Unknown")
                abstract = paper.get("abstract") or ""

                if not keep_literature_text(title, abstract):
                    log.info(
                        "Filtered Semantic Scholar result as off-topic: %s",
                        str(title)[:120],
                    )
                    continue

                evidence.append(
                    f"Title: {title} | Abstract: {abstract[:300]}..."
                )

        # ------------------------------------------------------------
        # Deduplicate, merge with existing evidence, and cap to 20
        # ------------------------------------------------------------
        existing_evidence = state.get("evidence", []) or []

        seen = set()
        deduped_new = []

        # Mark existing evidence as seen
        for item in existing_evidence:
            key = _evidence_key(item)
            if key:
                seen.add(key)

        # Add only new unique evidence
        for item in evidence:
            key = _evidence_key(item)

            if not key or key in seen:
                continue

            seen.add(key)
            deduped_new.append(item)

        # Preserve old evidence + append new unique, cap to latest 20
        combined = existing_evidence + evidence

        seen = set()
        evidence_out = []

        for item in combined:
            key = _evidence_key(item)

            if not key or key in seen:
                continue

            seen.add(key)
            evidence_out.append(item)

        evidence_out = evidence_out[-20:]

        log.info("Final evidence count: %d", len(evidence_out))

        result = {
            "evidence": evidence_out,
            "streaming_started": streaming_started,
            "streamed_queries": list(streamed),
            "research_query": query,
            "stage_outputs": [
                record_stage_output(
                    state,
                    "researcher",
                    "\n\n".join(evidence_out),
                    summary=f"Collected {len(evidence_out)} evidence items",
                    metadata={
                        "query": query,
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

        # Optional collector if present in state
        data_collector = state.get("data_collector")

        if data_collector is not None:
            try:
                data_collector.capture_agent_interaction(
                    "Researcher",
                    "Gather literature evidence",
                    "\n\n".join(evidence_out),
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