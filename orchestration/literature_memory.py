"""
literature_memory.py

Persistent memory for literature-derived biological hints.

Tracks:
- target class hints, e.g. nucleocapsid, capsid, RNA-binding domain
- motif/scaffold hints, e.g. stem-loop, packaging signal, G-rich motif
- avoid hints, e.g. spike-only, antibody, polymerase, protease
- query provenance

Used by:
- Researcher Agent
- Protein target selection
- PI Agent
- postrun training data builder
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _clean_text(x: Any) -> str:
    return str(x or "").strip()


def _normalise_hint(x: Any) -> str:
    x = str(x or "").lower().strip()
    x = re.sub(r"[^a-z0-9\- ]+", " ", x)
    return " ".join(x.split())


class LiteratureMemory:
    TARGET_TERMS = {
        "nucleocapsid": 1.0,
        "nucleoprotein": 1.0,
        "capsid": 0.8,
        "coat protein": 0.8,
        "rna-binding domain": 1.0,
        "rna binding": 0.8,
        "ribonucleoprotein": 0.9,
        "packaging protein": 0.8,
        "n protein": 1.0,
        "n-terminal domain": 0.7,
        "c-terminal domain": 0.7,
        "rna chaperone": 0.8,
        "oligomerization domain": 0.6,
    }

    MOTIF_TERMS = {
        "stem-loop": 1.0,
        "stem loop": 1.0,
        "hairpin": 0.8,
        "packaging signal": 1.0,
        "g-rich": 0.7,
        "conserved motif": 0.7,
        "rna motif": 0.7,
    }

    AVOID_TERMS = {
        "spike": 1.0,
        "fusion core": 1.0,
        "antibody": 0.9,
        "fab": 0.8,
        "polymerase": 0.7,
        "protease": 0.7,
        "membrane protein": 0.6,
    }

    def __init__(self, path: str | None = None):
        self.path = path or os.getenv(
            "VLAB_LITERATURE_MEMORY_PATH",
            "literature_memory.json",
        )

        self.memory = {
            "schema_version": "literature_memory.v1",
            "evidence_records": [],
            "target_hint_counts": {},
            "motif_hint_counts": {},
            "avoid_hint_counts": {},
            "query_counts": {},
            "topic_records": [],
        }

        self._load()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        defaults = {
            "schema_version": "literature_memory.v1",
            "evidence_records": [],
            "target_hint_counts": {},
            "motif_hint_counts": {},
            "avoid_hint_counts": {},
            "query_counts": {},
            "topic_records": [],
        }

        for k, v in defaults.items():
            self.memory.setdefault(k, v)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)

            if isinstance(loaded, dict):
                self.memory.update(loaded)

        except Exception:
            pass

    def save(self) -> None:
        try:
            self._compact()
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self.memory, handle, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _compact(self) -> None:
        records = self.memory.get("evidence_records", []) or []

        seen = {}
        for r in records:
            if not isinstance(r, dict):
                continue

            key = (
                _clean_text(r.get("title")).lower()
                or _clean_text(r.get("doi")).lower()
                or _clean_text(r.get("pmid")).lower()
                or _clean_text(r.get("abstract"))[:200].lower()
                or _clean_text(r.get("text"))[:200].lower()
            )

            if key:
                seen[key] = r

        self.memory["evidence_records"] = list(seen.values())[-500:]
        self.memory["topic_records"] = (self.memory.get("topic_records", []) or [])[-200:]

    def _extract_hints(self, text: str) -> tuple[list[str], list[str], list[str]]:
        lower = text.lower()

        target_hints = [
            term for term in self.TARGET_TERMS
            if term in lower
        ]

        motif_hints = [
            term for term in self.MOTIF_TERMS
            if term in lower
        ]

        avoid_hints = [
            term for term in self.AVOID_TERMS
            if term in lower
        ]

        return target_hints, motif_hints, avoid_hints

    def ingest_evidence_item(
        self,
        item: Any,
        research_topic: str | None = None,
        query: str | None = None,
    ) -> dict | None:
        """
        Ingest one evidence item.

        Supports:
        - dict papers
        - LangChain Document-like objects
        - raw strings
        """
        title = ""
        abstract = ""
        source = "unknown"
        pmid = None
        doi = None
        year = None
        score = None

        if isinstance(item, dict):
            title = _clean_text(item.get("title"))
            abstract = _clean_text(
                item.get("abstract")
                or item.get("text")
                or item.get("content")
                or item.get("page_content")
            )
            source = _clean_text(item.get("source") or "dict")
            pmid = item.get("pmid")
            doi = item.get("doi")
            year = item.get("year")
            score = item.get("score") or item.get("retrieval_score")

            query = query or item.get("query") or item.get("search_query")

        elif hasattr(item, "page_content"):
            metadata = getattr(item, "metadata", {}) or {}
            title = _clean_text(metadata.get("title"))
            abstract = _clean_text(getattr(item, "page_content", ""))
            source = _clean_text(metadata.get("source") or "document")
            pmid = metadata.get("pmid")
            doi = metadata.get("doi")
            year = metadata.get("year")
            score = metadata.get("score")

        else:
            abstract = _clean_text(item)
            source = "raw_text"

        if not title and not abstract:
            return None

        text = f"{title}\n{abstract}"
        target_hints, motif_hints, avoid_hints = self._extract_hints(text)

        record = {
            "schema_version": "literature_evidence.v1",
            "timestamp": _now_iso(),
            "research_topic": research_topic,
            "query": query,
            "source": source,
            "title": title,
            "abstract": abstract,
            "year": year,
            "pmid": pmid,
            "doi": doi,
            "retrieval_score": score,
            "target_hints": sorted(set(target_hints)),
            "motif_hints": sorted(set(motif_hints)),
            "avoid_hints": sorted(set(avoid_hints)),
        }

        self.memory["evidence_records"].append(record)

        self._increment_counts("target_hint_counts", target_hints)
        self._increment_counts("motif_hint_counts", motif_hints)
        self._increment_counts("avoid_hint_counts", avoid_hints)

        if query:
            self._increment_counts("query_counts", [query])

        return record

    def ingest_state(self, state: dict) -> list[dict]:
        """
        Ingest literature evidence from a final/intermediate LabState.
        """
        if not isinstance(state, dict):
            return []

        research_topic = state.get("research_topic")
        query = (
            state.get("research_query")
            or state.get("literature_query")
            or state.get("topic_description")
            or research_topic
        )

        records = []

        for item in state.get("evidence", []) or []:
            rec = self.ingest_evidence_item(
                item,
                research_topic=research_topic,
                query=query,
            )

            if rec:
                records.append(rec)

        self.memory["topic_records"].append(
            {
                "timestamp": _now_iso(),
                "research_topic": research_topic,
                "query": query,
                "evidence_count": len(records),
                "top_target_hints": self.get_target_policy_hints(limit=8),
                "top_motif_hints": self.get_motif_hints(limit=8),
                "top_avoid_hints": self.get_avoid_hints(limit=8),
            }
        )

        self.save()
        return records

    def _increment_counts(self, collection: str, hints: list[str]) -> None:
        counts = Counter(self.memory.get(collection, {}) or {})

        for h in hints:
            h = _normalise_hint(h)

            if h:
                counts[h] += 1

        self.memory[collection] = dict(counts)

    def _top_keys(self, collection: str, limit: int = 10) -> list[str]:
        counts = Counter(self.memory.get(collection, {}) or {})
        return [k for k, _ in counts.most_common(limit)]

    def get_target_policy_hints(self, limit: int = 10) -> list[str]:
        return self._top_keys("target_hint_counts", limit=limit)

    def get_motif_hints(self, limit: int = 10) -> list[str]:
        return self._top_keys("motif_hint_counts", limit=limit)

    def get_avoid_hints(self, limit: int = 10) -> list[str]:
        return self._top_keys("avoid_hint_counts", limit=limit)

    def build_target_policy_text(self) -> str:
        target_hints = self.get_target_policy_hints(limit=8)
        motif_hints = self.get_motif_hints(limit=8)
        avoid_hints = self.get_avoid_hints(limit=8)

        return (
            "Literature-derived target policy:\n"
            f"- Prefer target classes: {', '.join(target_hints) or 'N/A'}\n"
            f"- Prefer RNA motifs/scaffolds: {', '.join(motif_hints) or 'N/A'}\n"
            f"- Avoid target classes: {', '.join(avoid_hints) or 'N/A'}"
        )