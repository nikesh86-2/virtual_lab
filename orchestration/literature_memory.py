"""
literature_memory.py

Persistent memory for literature-derived biological hints.

Tracks:
- target class hints, e.g. nucleocapsid, capsid, RNA-binding domain
- motif/scaffold hints, e.g. stem-loop, packaging signal, IRES
- avoid hints, e.g. spike-only, antibody, polymerase, protease
- query provenance
- topic/profile provenance

Used by:
- Researcher Agent
- Protein target selection
- PI Agent
- postrun training data builder

Current schema:
    literature_memory.v2
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _clean_text(x: Any) -> str:
    return str(x or "").strip()


def _normalise_hint(x: Any) -> str:
    x = str(x or "").lower().strip()
    x = re.sub(r"[^a-z0-9\- ]+", " ", x)
    return " ".join(x.split())


def _normalise_key_text(x: Any) -> str:
    x = str(x or "").lower().strip()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return " ".join(x.split())


def _looks_like_long_description_query(query: Any) -> bool:
    """
    Detect legacy query strings that are really full topic descriptions.
    """
    q = _clean_text(query)
    if not q:
        return False

    words = q.split()
    if len(words) >= 18:
        return True

    lower = q.lower()
    markers = [
        "this test validates",
        "design and optimise",
        "evaluate candidates using",
        "screen small-molecule",
        "autodock vina",
        "hdock peptide docking",
        "molecular dynamics",
    ]

    return any(m in lower for m in markers)


def _short_query_from_profile(
    research_topic: str | None,
    virus_family: str | None,
    virus_genus: str | None,
    virus_name: str | None,
    task_type: str | None,
) -> str:
    """
    Build a short fallback query when legacy state only has a long description.
    """
    organism = virus_name or virus_genus or virus_family or "viral"

    if task_type == "inhibitor_screening":
        return f"{organism} capsid RNA binding inhibitors"

    if virus_family and str(virus_family).lower() == "coronaviridae":
        return "coronavirus nucleocapsid RNA binding"

    if virus_family and str(virus_family).lower() == "picornaviridae":
        return f"{organism} capsid RNA binding"

    if research_topic:
        lower = research_topic.lower()
        if "inhibitor" in lower:
            return f"{organism} RNA binding protein inhibitors"
        if "coronavirus" in lower or "coronaviridae" in lower:
            return "coronavirus nucleocapsid RNA binding"
        if "poliovirus" in lower or "enterovirus" in lower:
            return f"{organism} capsid RNA binding"

    return "viral RNA stem loop capsid binding"


class LiteratureMemory:
    TARGET_TERMS = {
        # Generic RNA/protein target concepts
        "nucleocapsid": 1.0,
        "nucleoprotein": 1.0,
        "capsid": 0.8,
        "coat protein": 0.8,
        "rna-binding domain": 1.0,
        "rna binding domain": 1.0,
        "rna binding": 0.8,
        "rna-binding": 0.8,
        "ribonucleoprotein": 0.9,
        "packaging protein": 0.8,
        "n protein": 1.0,
        "n-terminal domain": 0.7,
        "c-terminal domain": 0.7,
        "rna chaperone": 0.8,
        "oligomerization domain": 0.6,
        "rna binding pocket": 0.9,
        "binding pocket": 0.5,

        # Coronaviridae-specific
        "coronavirus nucleocapsid": 1.0,
        "sars-cov nucleocapsid": 1.0,
        "sars cov nucleocapsid": 1.0,
        "nucleocapsid phosphoprotein": 1.0,

        # Picornaviridae/Enterovirus-specific
        "picornavirus capsid": 0.9,
        "poliovirus capsid": 0.9,
        "enterovirus capsid": 0.9,
        "ev-a71 capsid": 0.8,
        "rhinovirus capsid": 0.8,
        "capsid-binding inhibitor": 0.8,
        "capsid binding inhibitor": 0.8,
    }

    MOTIF_TERMS = {
        "stem-loop": 1.0,
        "stem loop": 1.0,
        "hairpin": 0.8,
        "packaging signal": 1.0,
        "g-rich": 0.7,
        "conserved motif": 0.7,
        "rna motif": 0.7,
        "5 utr": 0.7,
        "5' utr": 0.7,
        "ires": 0.8,
        "internal ribosome entry site": 0.8,
        "genomic rna": 0.6,
        "subgenomic rna": 0.6,
        "stem-loop iv": 0.9,
        "stem loop iv": 0.9,
    }

    AVOID_TERMS = {
        "spike": 1.0,
        "fusion core": 1.0,
        "antibody": 0.9,
        "fab": 0.8,
        "polymerase": 0.7,
        "rdrp": 0.7,
        "protease": 0.7,
        "membrane protein": 0.6,
        "osbp": 0.6,
        "receptor": 0.5,
        "scarb2": 0.5,
    }

    def __init__(self, path: str | None = None):
        self.path = path or os.getenv(
            "VLAB_LITERATURE_MEMORY_PATH",
            "literature_memory.json",
        )

        self.memory = {
            "schema_version": "literature_memory.v2",
            "evidence_records": [],
            "target_hint_counts": {},
            "motif_hint_counts": {},
            "avoid_hint_counts": {},
            "query_counts": {},
            "topic_records": [],
            "topic_query_counts": {},
        }

        self._load()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema / persistence
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        defaults = {
            "schema_version": "literature_memory.v2",
            "evidence_records": [],
            "target_hint_counts": {},
            "motif_hint_counts": {},
            "avoid_hint_counts": {},
            "query_counts": {},
            "topic_records": [],
            "topic_query_counts": {},
        }

        for k, v in defaults.items():
            self.memory.setdefault(k, v)

        previous_schema = self.memory.get("schema_version")

        self._migrate_legacy_records()
        self._compact()

        # Rebuild counts after migration/compaction so legacy duplicate inflation
        # is reduced and v2 hint terms are applied consistently.
        if previous_schema != "literature_memory.v2":
            self.rebuild_hint_counts()

        self.memory["schema_version"] = "literature_memory.v2"

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)

            if isinstance(loaded, dict):
                self.memory.update(loaded)

        except Exception:
            # Memory is non-critical; corrupt/partial files should not break runs.
            pass

    def save(self) -> None:
        try:
            self._compact()
            self.memory["schema_version"] = "literature_memory.v2"

            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self.memory, handle, indent=2, ensure_ascii=False)

        except Exception:
            # Persistent memory should never fail the main workflow.
            pass

    def _legacy_title_abstract_from_raw(self, raw: str) -> tuple[str, str]:
        """
        Parse legacy evidence strings:
            Title: ... | Abstract: ...
        """
        raw = _clean_text(raw)

        if not raw:
            return "", ""

        if raw.startswith("Title: "):
            parts = raw.split(" | Abstract:", 1)
            if len(parts) == 2:
                title = parts[0].replace("Title:", "").strip()
                abstract = parts[1].strip()
                return title, abstract

            parts = raw.split(" | ", 1)
            title = parts[0].replace("Title:", "").strip()

            if len(parts) > 1:
                abstract = parts[1].replace("Abstract:", "").strip()
            else:
                abstract = ""

            return title, abstract

        return "", raw

    def _infer_task_type(
        self,
        research_topic: Any = None,
        query: Any = None,
        title: Any = None,
        abstract: Any = None,
    ) -> str:
        text = " ".join(
            str(x or "")
            for x in [research_topic, query, title, abstract]
        ).lower()

        if (
            "inhibitor" in text
            or "small molecule" in text
            or "small-molecule" in text
            or "peptide inhibitor" in text
            or "competitive inhibition" in text
            or "vina" in text
        ):
            return "inhibitor_screening"

        return "rna_docking"

    def _migrate_legacy_records(self) -> None:
        """
        Upgrade legacy literature_memory.v1/v1-like records into v2 shape.

        Fixes:
          - raw_text title/abstract packing
          - missing topic_name
          - missing task_type
          - schema_version mismatch
        """
        records = self.memory.get("evidence_records", []) or []

        for r in records:
            if not isinstance(r, dict):
                continue

            r["schema_version"] = "literature_evidence.v2"

            title = _clean_text(r.get("title"))
            abstract = _clean_text(r.get("abstract"))

            # Legacy record has:
            # title=""
            # abstract="Title: ... | Abstract: ..."
            if not title and abstract.startswith("Title: "):
                parsed_title, parsed_abstract = self._legacy_title_abstract_from_raw(abstract)
                if parsed_title:
                    r["title"] = parsed_title
                if parsed_abstract:
                    r["abstract"] = parsed_abstract

            if r.get("source") == "raw_text":
                r["source"] = "legacy_raw_text"

            if not r.get("topic_name") and r.get("research_topic"):
                r["topic_name"] = r.get("research_topic")

            if not r.get("task_type"):
                r["task_type"] = self._infer_task_type(
                    research_topic=r.get("research_topic"),
                    query=r.get("query"),
                    title=r.get("title"),
                    abstract=r.get("abstract"),
                )

            # Promote obvious family/genus/name metadata when absent.
            text = " ".join(
                str(x or "")
                for x in [
                    r.get("research_topic"),
                    r.get("query"),
                    r.get("title"),
                    r.get("abstract"),
                ]
            ).lower()

            if not r.get("virus_family"):
                if "picornaviridae" in text or "poliovirus" in text or "enterovirus" in text:
                    r["virus_family"] = "Picornaviridae"
                elif "coronaviridae" in text or "coronavirus" in text or "sars-cov" in text:
                    r["virus_family"] = "Coronaviridae"

            if not r.get("virus_genus"):
                if "enterovirus" in text or "poliovirus" in text:
                    r["virus_genus"] = "Enterovirus"

            if not r.get("virus_name"):
                if "poliovirus" in text:
                    r["virus_name"] = "Poliovirus"

        # Upgrade topic records shape too.
        for tr in self.memory.get("topic_records", []) or []:
            if not isinstance(tr, dict):
                continue

            if not tr.get("topic_name") and tr.get("research_topic"):
                tr["topic_name"] = tr.get("research_topic")

            tr.setdefault("query_bundle", [])
            tr.setdefault("topic_profile", {})

        self.memory["schema_version"] = "literature_memory.v2"

    def _compact(self) -> None:
        """
        Dedupe evidence records and bound memory growth.
        """
        records = self.memory.get("evidence_records", []) or []

        seen: dict[str, dict] = {}

        for r in records:
            if not isinstance(r, dict):
                continue

            doi = _clean_text(r.get("doi")).lower()
            pmid = _clean_text(r.get("pmid")).lower()
            title = _clean_text(r.get("title")).lower()
            abstract_prefix = _clean_text(r.get("abstract"))[:300].lower()
            year = _clean_text(r.get("year"))

            if doi:
                key = f"doi::{doi}"
            elif pmid:
                key = f"pmid::{pmid}"
            elif title:
                key = f"title::{_normalise_key_text(title)}|year::{year}|abs::{_normalise_key_text(abstract_prefix)[:160]}"
            elif abstract_prefix:
                key = f"abstract::{_normalise_key_text(abstract_prefix)[:220]}"
            else:
                continue

            # Prefer richer v2/dict records over legacy.
            old = seen.get(key)
            if old is None:
                seen[key] = r
            else:
                old_score = self._record_richness_score(old)
                new_score = self._record_richness_score(r)
                if new_score >= old_score:
                    seen[key] = r

        self.memory["evidence_records"] = list(seen.values())[-500:]
        self.memory["topic_records"] = (self.memory.get("topic_records", []) or [])[-200:]

    def _record_richness_score(self, r: dict) -> int:
        score = 0

        for key in (
            "title",
            "abstract",
            "doi",
            "pmid",
            "year",
            "url",
            "virus_family",
            "virus_genus",
            "virus_name",
            "task_type",
            "topic_name",
            "query",
        ):
            if r.get(key):
                score += 1

        if r.get("source") != "legacy_raw_text":
            score += 2

        if r.get("schema_version") == "literature_evidence.v2":
            score += 2

        return score

    # ------------------------------------------------------------------
    # Hint extraction / rebuild
    # ------------------------------------------------------------------

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

    def rebuild_hint_counts(self) -> None:
        """
        Recompute hint/query counts from current compacted evidence records.

        Useful after migrating old v1 memory or after changing hint vocabulary.
        """
        self.memory["target_hint_counts"] = {}
        self.memory["motif_hint_counts"] = {}
        self.memory["avoid_hint_counts"] = {}
        self.memory["query_counts"] = {}
        self.memory["topic_query_counts"] = {}

        for r in self.memory.get("evidence_records", []) or []:
            if not isinstance(r, dict):
                continue

            text = f"{r.get('title', '')}\n{r.get('abstract', '')}"
            target_hints, motif_hints, avoid_hints = self._extract_hints(text)

            self._increment_counts("target_hint_counts", target_hints)
            self._increment_counts("motif_hint_counts", motif_hints)
            self._increment_counts("avoid_hint_counts", avoid_hints)

            query = r.get("query")
            if query:
                self._increment_counts("query_counts", [query])

            research_topic = r.get("research_topic")
            if research_topic and query:
                topic_key = (
                    f"{_normalise_key_text(research_topic)}::"
                    f"{_normalise_key_text(query)}"
                )
                self._increment_counts("topic_query_counts", [topic_key])

    def _increment_counts(self, collection: str, hints: list[str]) -> None:
        counts = Counter(self.memory.get(collection, {}) or {})

        for h in hints:
            h = _normalise_hint(h)

            if h:
                counts[h] += 1

        self.memory[collection] = dict(counts)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_evidence_item(
        self,
        item: Any,
        research_topic: str | None = None,
        query: str | None = None,
        topic_name: str | None = None,
        topic_profile: dict | None = None,
    ) -> dict | None:
        """
        Ingest one evidence item.

        Supports:
        - dict papers/evidence
        - LangChain Document-like objects
        - legacy raw strings
        """
        title = ""
        abstract = ""
        source = "unknown"
        pmid = None
        doi = None
        year = None
        score = None
        url = None

        virus_family = None
        virus_genus = None
        virus_name = None
        task_type = None

        profile = topic_profile or {}

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
            url = item.get("url")
            score = item.get("score") or item.get("retrieval_score")

            query = query or item.get("query") or item.get("search_query")
            research_topic = research_topic or item.get("research_topic")
            topic_name = topic_name or item.get("topic_name")

            virus_family = item.get("virus_family") or profile.get("virus_family")
            virus_genus = item.get("virus_genus") or profile.get("virus_genus")
            virus_name = item.get("virus_name") or profile.get("virus_name")
            task_type = item.get("task_type") or profile.get("task_type")

        elif hasattr(item, "page_content"):
            metadata = getattr(item, "metadata", {}) or {}
            title = _clean_text(metadata.get("title"))
            abstract = _clean_text(getattr(item, "page_content", ""))
            source = _clean_text(metadata.get("source") or "document")
            pmid = metadata.get("pmid")
            doi = metadata.get("doi")
            year = metadata.get("year")
            url = metadata.get("url")
            score = metadata.get("score") or metadata.get("retrieval_score")

            query = query or metadata.get("query") or metadata.get("search_query")
            research_topic = research_topic or metadata.get("research_topic")
            topic_name = topic_name or metadata.get("topic_name")

            virus_family = metadata.get("virus_family") or profile.get("virus_family")
            virus_genus = metadata.get("virus_genus") or profile.get("virus_genus")
            virus_name = metadata.get("virus_name") or profile.get("virus_name")
            task_type = metadata.get("task_type") or profile.get("task_type")

        else:
            raw = _clean_text(item)
            source = "legacy_raw_text"

            parsed_title, parsed_abstract = self._legacy_title_abstract_from_raw(raw)
            title = parsed_title
            abstract = parsed_abstract

        if not title and not abstract:
            return None

        if not topic_name and research_topic:
            topic_name = research_topic

        if not task_type:
            task_type = self._infer_task_type(
                research_topic=research_topic,
                query=query,
                title=title,
                abstract=abstract,
            )

        if _looks_like_long_description_query(query):
            query = _short_query_from_profile(
                research_topic=research_topic,
                virus_family=virus_family,
                virus_genus=virus_genus,
                virus_name=virus_name,
                task_type=task_type,
            )

        text = f"{title}\n{abstract}"
        target_hints, motif_hints, avoid_hints = self._extract_hints(text)

        record = {
            "schema_version": "literature_evidence.v2",
            "timestamp": _now_iso(),
            "research_topic": research_topic,
            "topic_name": topic_name,
            "query": query,
            "source": source,
            "title": title,
            "abstract": abstract,
            "year": year,
            "pmid": pmid,
            "doi": doi,
            "url": url,
            "retrieval_score": score,
            "virus_family": virus_family,
            "virus_genus": virus_genus,
            "virus_name": virus_name,
            "task_type": task_type,
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

        if research_topic and query:
            topic_key = (
                f"{_normalise_key_text(research_topic)}::"
                f"{_normalise_key_text(query)}"
            )
            self._increment_counts("topic_query_counts", [topic_key])

        return record

    def ingest_state(self, state: dict) -> list[dict]:
        """
        Ingest literature evidence from a final/intermediate LabState.
        """
        if not isinstance(state, dict):
            return []

        topic_profile = state.get("literature_topic_profile", {}) or {}

        research_topic = (
            state.get("research_topic")
            or topic_profile.get("research_topic")
            or state.get("topic_name")
        )

        topic_name = (
            state.get("topic_name")
            or topic_profile.get("topic_name")
            or research_topic
        )

        query_bundle = state.get("literature_query_bundle", []) or []

        query = (
            state.get("research_query")
            or (query_bundle[0] if query_bundle else None)
            or state.get("literature_query")
            or research_topic
        )

        if _looks_like_long_description_query(query):
            query = _short_query_from_profile(
                research_topic=research_topic,
                virus_family=state.get("virus_family") or topic_profile.get("virus_family"),
                virus_genus=state.get("virus_genus") or topic_profile.get("virus_genus"),
                virus_name=state.get("virus_name") or topic_profile.get("virus_name"),
                task_type=topic_profile.get("task_type"),
            )

        records: list[dict] = []

        for item in state.get("evidence", []) or []:
            rec = self.ingest_evidence_item(
                item,
                research_topic=research_topic,
                query=query,
                topic_name=topic_name,
                topic_profile=topic_profile,
            )

            if rec:
                records.append(rec)

        self.memory["topic_records"].append(
            {
                "timestamp": _now_iso(),
                "research_topic": research_topic,
                "topic_name": topic_name,
                "query": query,
                "query_bundle": query_bundle,
                "topic_profile": topic_profile,
                "evidence_count": len(records),
                "top_target_hints": self.get_target_policy_hints(limit=8),
                "top_motif_hints": self.get_motif_hints(limit=8),
                "top_avoid_hints": self.get_avoid_hints(limit=8),
            }
        )

        self.save()
        return records

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def _top_keys(self, collection: str, limit: int = 10) -> list:
        counts = Counter(self.memory.get(collection, {}) or {})
        return [k for k, _ in counts.most_common(limit)]

    def get_target_policy_hints(self, limit: int = 10) -> list:
        return self._top_keys("target_hint_counts", limit=limit)

    def get_motif_hints(self, limit: int = 10) -> list:
        return self._top_keys("motif_hint_counts", limit=limit)

    def get_avoid_hints(self, limit: int = 10) -> list:
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