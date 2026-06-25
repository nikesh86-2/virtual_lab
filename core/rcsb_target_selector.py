from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


log = logging.getLogger("virtual_lab.rcsb_target_selector")


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_POLYMER_ENTITY_URL = (
    "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
)


@dataclass
class PDBTargetCandidate:
    pdb_id: str
    score: float
    title: str = ""
    release_date: str = ""
    year: Optional[int] = None
    methods: List[str] = None
    resolution: Optional[float] = None
    entity_descriptions: List[str] = None
    polymer_types: List[str] = None
    organisms: List[str] = None
    reasons: List[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["methods"] = d["methods"] or []
        d["entity_descriptions"] = d["entity_descriptions"] or []
        d["polymer_types"] = d["polymer_types"] or []
        d["organisms"] = d["organisms"] or []
        d["reasons"] = d["reasons"] or []
        d["metadata"] = d["metadata"] or {}
        return d


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))

    try:
        return int(raw)
    except Exception:
        log.warning("Invalid integer env %s=%r; using %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))

    try:
        return float(raw)
    except Exception:
        log.warning("Invalid float env %s=%r; using %.3f", name, raw, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _http_json(
    url: str,
    payload: Optional[dict] = None,
    timeout: float = 30.0,
    retries: int = 2,
) -> dict:
    data = None
    headers = {
        "Accept": "application/json",
        "User-Agent": "VLAB2-RCSBTargetSelector/1.1",
    }

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    last_err = None

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method="POST" if payload is not None else "GET",
            )

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", None)
                raw = resp.read()
                body = raw.decode("utf-8", errors="replace")

                if not body.strip():
                    raise RuntimeError(
                        f"Empty response from {url}; status={status}"
                    )

                try:
                    return json.loads(body)
                except Exception as json_err:
                    raise RuntimeError(
                        f"Non-JSON response from {url}; status={status}; "
                        f"body_prefix={body[:500]!r}; json_error={json_err}"
                    )

        except Exception as e:
            last_err = e

            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                break

    raise RuntimeError(f"RCSB request failed for {url}: {last_err}")


def _clean_terms(text: str) -> List[str]:
    if not text:
        return []

    text = re.sub(r"[^A-Za-z0-9\-\s]", " ", text)
    words = [w.strip() for w in text.split() if len(w.strip()) >= 3]

    stop = {
        "the",
        "and",
        "for",
        "with",
        "into",
        "will",
        "than",
        "after",
        "score",
        "spread",
        "lower",
        "higher",
        "less",
        "more",
        "best",
        "binding",
        "relative",
        "hdock",
        "using",
        "current",
        "compared",
        "optimized",
        "optimised",
        "stronger",
        "weaker",
        "stable",
        "stability",
    }

    out = []

    for w in words:
        wl = w.lower()

        if wl in stop:
            continue

        out.append(w)

    return list(dict.fromkeys(out))


def build_rcsb_query_text(
    topic: str = "",
    hypothesis: str = "",
    virus_name: str = "",
    virus_family: str = "",
) -> str:
    """
    Backwards-compatible single-query builder.

    The selector now uses build_rcsb_query_texts(...) by default, but this is kept
    for older callers and emergency fallback inside select_pdb_targets(...).
    """
    env_keywords = os.getenv(
        "VLAB_TARGET_KEYWORDS",
        "RNA binding,RNA protein complex,ribonucleoprotein,nucleocapsid,capsid",
    )

    keyword_terms = [x.strip() for x in env_keywords.split(",") if x.strip()]

    biological_terms: List[str] = []

    if virus_name:
        biological_terms.append(virus_name)

    if virus_family:
        biological_terms.append(virus_family)

    if not biological_terms:
        biological_terms.extend(["viral", "RNA"])

    extracted_terms = _clean_terms(topic)[:4] + _clean_terms(hypothesis)[:4]

    query_terms = biological_terms + extracted_terms + keyword_terms[:3]

    return " ".join(dict.fromkeys(x for x in query_terms if x)).strip()


def build_rcsb_query_texts(
    topic: str = "",
    hypothesis: str = "",
    virus_name: str = "",
    virus_family: str = "",
) -> List[str]:
    """
    Build multiple adaptive RCSB full-text queries.

    Avoids over-reliance on one brittle query like:
      'Coronaviridae capsid coat protein viral assembly'

    The query set stays general and does not require hardcoded family profiles.
    """
    base_terms: List[str] = []

    if virus_name:
        base_terms.append(str(virus_name).strip())

    if virus_family:
        base_terms.append(str(virus_family).strip())

    extracted_terms = _clean_terms(topic)[:6] + _clean_terms(hypothesis)[:8]
    extracted_terms = list(dict.fromkeys(extracted_terms))

    anchors = [
        "RNA binding protein",
        "RNA protein complex",
        "ribonucleoprotein",
        "viral RNA binding",
        "nucleocapsid RNA binding",
        "capsid RNA binding",
        "RNA packaging protein",
    ]

    queries: List[str] = []

    for anchor in anchors:
        terms = base_terms + extracted_terms[:4] + [anchor]
        q = " ".join(dict.fromkeys(x for x in terms if x)).strip()

        if q:
            queries.append(q)

    if virus_family:
        queries.append(f"{virus_family} RNA protein")
        queries.append(f"{virus_family} ribonucleoprotein")
        queries.append(f"{virus_family} nucleocapsid")
        queries.append(f"{virus_family} RNA binding")

    if virus_name:
        queries.append(f"{virus_name} RNA binding")
        queries.append(f"{virus_name} nucleocapsid")
        queries.append(f"{virus_name} RNA protein complex")

    if topic:
        queries.append(topic)

    return list(dict.fromkeys(q for q in queries if q))


def search_rcsb_entries(query_text: str, rows: int = 50) -> List[str]:
    """
    Search RCSB by full text and return PDB IDs.

    The Search API returns identifiers; metadata is fetched separately from
    the Data API.
    """
    rows = max(1, rows)

    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {
                "value": query_text,
            },
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": rows,
            },
        },
    }

    try:
        data = _http_json(RCSB_SEARCH_URL, payload=payload)
    except Exception as e:
        log.warning("RCSB full-text search failed: %s", e)
        raise

    result_set = data.get("result_set", []) or []

    ids = []

    for item in result_set:
        identifier = item.get("identifier")

        if identifier:
            ids.append(str(identifier).upper())

    return list(dict.fromkeys(ids))


def fetch_entry_metadata(pdb_id: str) -> dict:
    pdb_id = str(pdb_id).upper()
    return _http_json(RCSB_ENTRY_URL.format(pdb_id=pdb_id))


def fetch_polymer_entities(pdb_id: str, entry_data: dict) -> List[dict]:
    pdb_id = str(pdb_id).upper()

    container = entry_data.get("rcsb_entry_container_identifiers", {}) or {}
    entity_ids = container.get("polymer_entity_ids", []) or []

    out = []

    for entity_id in entity_ids:
        try:
            entity = _http_json(
                RCSB_POLYMER_ENTITY_URL.format(
                    pdb_id=pdb_id,
                    entity_id=entity_id,
                )
            )
            out.append(entity)
        except Exception as e:
            log.warning("Failed to fetch polymer entity %s_%s: %s", pdb_id, entity_id, e)

    return out


def _parse_year(date_str: str) -> Optional[int]:
    if not date_str:
        return None

    try:
        return datetime.fromisoformat(date_str.replace("Z", "")).year
    except Exception:
        pass

    m = re.search(r"\b(19|20)\d{2}\b", str(date_str))

    if m:
        return int(m.group(0))

    return None


def _extract_resolution(entry_data: dict) -> Optional[float]:
    info = entry_data.get("rcsb_entry_info", {}) or {}
    vals = info.get("resolution_combined") or []

    if not vals:
        return None

    nums = []

    for v in vals:
        try:
            nums.append(float(v))
        except Exception:
            pass

    return min(nums) if nums else None


def _extract_methods(entry_data: dict) -> List[str]:
    methods = []

    for item in entry_data.get("exptl", []) or []:
        method = item.get("method")

        if method:
            methods.append(str(method))

    return list(dict.fromkeys(methods))


def _extract_title(entry_data: dict) -> str:
    struct = entry_data.get("struct", {}) or {}
    return str(struct.get("title", "") or "")


def _entity_description(entity: dict) -> str:
    desc = ""

    rpe = entity.get("rcsb_polymer_entity", {}) or {}

    if rpe.get("pdbx_description"):
        desc = str(rpe.get("pdbx_description"))

    if not desc:
        ep = entity.get("entity_poly", {}) or {}
        desc = str(ep.get("pdbx_strand_id", "") or "")

    return desc


def _entity_polymer_type(entity: dict) -> str:
    ep = entity.get("entity_poly", {}) or {}
    rpt = ep.get("rcsb_entity_polymer_type") or ep.get("type") or ""
    return str(rpt)


def _entity_organisms(entity: dict) -> List[str]:
    orgs = []

    for key in ["rcsb_entity_source_organism", "entity_src_gen", "entity_src_nat"]:
        val = entity.get(key)

        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    name = (
                        item.get("scientific_name")
                        or item.get("pdbx_gene_src_scientific_name")
                        or item.get("pdbx_organism_scientific")
                    )

                    if name:
                        orgs.append(str(name))

        elif isinstance(val, dict):
            name = (
                val.get("scientific_name")
                or val.get("pdbx_gene_src_scientific_name")
                or val.get("pdbx_organism_scientific")
            )

            if name:
                orgs.append(str(name))

    return list(dict.fromkeys(orgs))


def _token_set(text: str) -> set[str]:
    return {x.lower() for x in _clean_terms(text or "")}


def _entry_atom_count(entry_data: dict) -> Optional[int]:
    info = entry_data.get("rcsb_entry_info", {}) or {}

    for key in [
        "deposited_atom_count",
        "modeled_polymer_monomer_count",
        "deposited_polymer_monomer_count",
    ]:
        value = info.get(key)

        try:
            if value is not None:
                return int(value)
        except Exception:
            pass

    return None


def _adaptive_relevance_score(
    candidate_text: str,
    query_text: str,
    polymer_text: str,
    entry_data: dict,
    virus_name: str = "",
    virus_family: str = "",
) -> tuple[float, List[str]]:
    """
    General-purpose relevance scoring.

    IMPORTANT:
      candidate_text must contain ONLY candidate metadata.
      query_text contains query / state context.

    This prevents query leakage where every candidate falsely inherits terms like
    'RNA', 'nucleocapsid', or 'ribonucleoprotein' from the search query itself.
    """
    reasons: List[str] = []
    score = 0.0

    candidate_text_l = (candidate_text or "").lower()
    query_text_l = (query_text or "").lower()

    query_tokens = _token_set(query_text_l)
    candidate_tokens = _token_set(candidate_text_l)

    overlap = query_tokens.intersection(candidate_tokens)

    if overlap:
        score += min(4.0, 0.35 * len(overlap))
        reasons.append(f"text_overlap:{len(overlap)}")

    biological_terms = [
        "rna",
        "binding",
        "rna-binding",
        "ribonucleoprotein",
        "nucleocapsid",
        "nucleoprotein",
        "n protein",
        "capsid",
        "core protein",
        "viral",
        "genome",
        "packaging",
        "encapsidation",
    ]

    bio_hits = [t for t in biological_terms if t in candidate_text_l]

    if bio_hits:
        score += min(4.0, 0.45 * len(bio_hits))
        reasons.extend(f"bio_term:{t}" for t in bio_hits[:6])

    receptor_terms = [
        "nucleocapsid",
        "nucleoprotein",
        "n protein",
        "rna-binding",
        "rna binding",
        "ribonucleoprotein",
        "core protein",
        "capsid protein",
    ]

    receptor_hits = [t for t in receptor_terms if t in candidate_text_l]

    if receptor_hits:
        score += min(4.0, 1.0 * len(receptor_hits))
        reasons.append(f"receptor_term:{len(receptor_hits)}")

    polymer_text_l = (polymer_text or "").lower()

    has_protein = "protein" in polymer_text_l
    has_rna = "rna" in polymer_text_l or "polyribonucleotide" in polymer_text_l

    if has_protein and has_rna:
        score += 5.0
        reasons.append("rna_protein_complex")
    elif has_rna:
        score += 1.5
        reasons.append("rna_entity")
    elif has_protein:
        score += 0.75
        reasons.append("protein_entity")

    for label, term in [("virus_name", virus_name), ("virus_family", virus_family)]:
        if term:
            term_tokens = _token_set(term)
            n = len(term_tokens.intersection(candidate_tokens))

            if n:
                score += 1.0 * n
                reasons.append(f"{label}_overlap:{n}")

    noise_terms = [
        "antibody",
        "fab",
        "nanobody",
        "spike",
        "ace2",
        "receptor binding domain",
        "polymerase",
        "protease",
        "fusion core",
        "hr1",
        "hr2",
        "membrane protein",
        "surface glycoprotein",
    ]

    noise_hits = [t for t in noise_terms if t in candidate_text_l]

    if noise_hits:
        score -= min(7.5, 1.5 + 1.0 * len(noise_hits))
        reasons.append(f"noise_penalty:{len(noise_hits)}")

    atom_count = _entry_atom_count(entry_data)

    if atom_count:
        max_atoms = _env_int("VLAB_MAX_RECEPTOR_ATOMS", 60000)

        if atom_count > max_atoms:
            score -= 5.0
            reasons.append(f"too_large:{atom_count}")

    return score, reasons


def score_pdb_candidate(
    pdb_id: str,
    entry_data: dict,
    entities: List[dict],
    query_text: str,
    virus_name: str = "",
    virus_family: str = "",
) -> Optional[PDBTargetCandidate]:
    min_year = _env_int("VLAB_TARGET_MIN_YEAR", 2015)
    max_resolution = _env_float("VLAB_TARGET_MAX_RESOLUTION", 4.0)

    require_experimental = _env_bool("VLAB_TARGET_REQUIRE_EXPERIMENTAL", True)
    require_protein_target = _env_bool("VLAB_TARGET_REQUIRE_PROTEIN", True)

    allow_cryoem = _env_bool("VLAB_TARGET_ALLOW_CRYOEM", True)
    allow_xray = _env_bool("VLAB_TARGET_ALLOW_XRAY", True)
    allow_nmr = _env_bool("VLAB_TARGET_ALLOW_NMR", False)

    title = _extract_title(entry_data)
    methods = _extract_methods(entry_data)
    resolution = _extract_resolution(entry_data)

    accession = entry_data.get("rcsb_accession_info", {}) or {}
    release_date = accession.get("initial_release_date", "") or ""
    status = str(accession.get("status_code", "") or "").upper()

    if status in {"OBS", "OBSOLETE", "REMOVED"}:
        return None

    year = _parse_year(release_date)
    method_text = " ".join(methods).lower()

    if require_experimental and not methods:
        return None

    allowed_method = False

    if allow_xray and "x-ray" in method_text:
        allowed_method = True

    if allow_cryoem and ("electron microscopy" in method_text or "cryo" in method_text):
        allowed_method = True

    if allow_nmr and "nmr" in method_text:
        allowed_method = True

    if methods and not allowed_method:
        return None

    entity_descriptions = []
    polymer_types = []
    organisms = []

    for entity in entities:
        desc = _entity_description(entity)
        ptype = _entity_polymer_type(entity)

        if desc:
            entity_descriptions.append(desc)

        if ptype:
            polymer_types.append(ptype)

        organisms.extend(_entity_organisms(entity))

    polymer_text = " ".join(polymer_types).lower()

    has_protein = "protein" in polymer_text
    has_rna = "rna" in polymer_text or "polyribonucleotide" in polymer_text

    if require_protein_target and not has_protein:
        return None

    candidate_text = " ".join(
        [title]
        + entity_descriptions
        + organisms
    ).lower()

    query_context_text = " ".join(
        [query_text, virus_name, virus_family]
    ).lower()

    reasons = []
    score = 0.0

    relevance_score, relevance_reasons = _adaptive_relevance_score(
        candidate_text=candidate_text,
        query_text=query_context_text,
        polymer_text=polymer_text,
        entry_data=entry_data,
        virus_name=virus_name,
        virus_family=virus_family,
    )

    score += relevance_score
    reasons.extend(relevance_reasons)

    if "x-ray" in method_text:
        score += 0.75
        reasons.append("xray")

    if "electron microscopy" in method_text or "cryo" in method_text:
        score += 0.75
        reasons.append("cryoem")

    if resolution is not None:
        if resolution <= max_resolution:
            score += max(0.0, 1.5 - (resolution / max_resolution))
            reasons.append(f"resolution_ok:{resolution:.2f}")
        else:
            score -= min(2.0, (resolution - max_resolution) / 2.0)
            reasons.append(f"resolution_penalty:{resolution:.2f}")
    else:
        score -= 0.25
        reasons.append("missing_resolution")

    if year is not None:
        if year >= min_year:
            score += min(1.5, (year - min_year + 1) / 8.0)
            reasons.append(f"recent:{year}")
        else:
            score -= min(1.0, (min_year - year) / 20.0)
            reasons.append(f"older:{year}")
    else:
        score -= 0.25
        reasons.append("missing_release_year")

    if not any(
        r.startswith("text_overlap")
        or r.startswith("bio_term")
        or r.startswith("receptor_term")
        or r in {"rna_protein_complex", "protein_entity"}
        for r in reasons
    ):
        score -= 2.5
        reasons.append("low_text_relevance")

    return PDBTargetCandidate(
        pdb_id=pdb_id,
        score=round(score, 4),
        title=title,
        release_date=release_date,
        year=year,
        methods=methods,
        resolution=resolution,
        entity_descriptions=list(dict.fromkeys(entity_descriptions)),
        polymer_types=list(dict.fromkeys(polymer_types)),
        organisms=list(dict.fromkeys(organisms)),
        reasons=reasons,
        metadata={
            "query_text": query_text,
            "status": status,
            "adaptive_selector": True,
            "require_protein_target": require_protein_target,
            "has_protein": has_protein,
            "has_rna": has_rna,
            "atom_count": _entry_atom_count(entry_data),
        },
    )


def select_pdb_targets(
    topic: str = "",
    hypothesis: str = "",
    virus_name: str = "",
    virus_family: str = "",
    exclude_pdbs: Optional[List[str]] = None,
) -> List[dict]:
    """
    Search and rank recent/high-quality/relevant PDB targets for RNA-protein docking.

    Returns list of dictionaries sorted by descending target score.
    """
    max_candidates = _env_int("VLAB_TARGET_MAX_CANDIDATES", 8)
    exclude = {str(x).upper() for x in (exclude_pdbs or []) if x}

    query_texts = build_rcsb_query_texts(
        topic=topic,
        hypothesis=hypothesis,
        virus_name=virus_name,
        virus_family=virus_family,
    )

    if not query_texts:
        query_texts = [
            build_rcsb_query_text(
                topic=topic,
                hypothesis=hypothesis,
                virus_name=virus_name,
                virus_family=virus_family,
            )
        ]

    search_rows = max(25, max_candidates * 8)
    rows_per_query = max(10, search_rows // max(len(query_texts), 1))

    candidates: List[PDBTargetCandidate] = []
    ids: List[str] = []

    successful_queries: List[str] = []
    failed_queries: List[str] = []

    for query_text in query_texts:
        try:
            log.info("RCSB target query: %s", query_text)
            query_ids = search_rcsb_entries(query_text, rows=rows_per_query)

            if query_ids:
                successful_queries.append(query_text)
                ids.extend(query_ids)
            else:
                failed_queries.append(query_text)

        except Exception as e:
            failed_queries.append(query_text)
            log.warning("RCSB target search failed for query %r: %s", query_text, e)

    ids = list(dict.fromkeys(ids))
    query_context = " | ".join(successful_queries or query_texts)

    for pdb_id in ids:
        pdb_id = pdb_id.upper()

        if pdb_id in exclude:
            continue

        try:
            entry = fetch_entry_metadata(pdb_id)
            entities = fetch_polymer_entities(pdb_id, entry)

            candidate = score_pdb_candidate(
                pdb_id=pdb_id,
                entry_data=entry,
                entities=entities,
                query_text=query_context,
                virus_name=virus_name,
                virus_family=virus_family,
            )

            if candidate is not None:
                candidates.append(candidate)

        except Exception as e:
            log.warning("Failed to score PDB target %s: %s", pdb_id, e)

    candidates.sort(key=lambda x: x.score, reverse=True)

    out = [c.to_dict() for c in candidates[:max_candidates]]

    allow_generic_fallback = _env_bool("VLAB_ALLOW_GENERIC_TARGET_FALLBACK", False)
    min_real_candidate_score = _env_float("VLAB_MIN_REAL_TARGET_SCORE", 1.0)

    best_real_score = max((c.score for c in candidates), default=-999.0)
    need_fallback = not out or best_real_score < min_real_candidate_score

    if allow_generic_fallback and need_fallback:
        fallback_env = os.getenv("VLAB_FALLBACK_PDBS", "")
        fallback_ids = [
            x.strip().upper()
            for x in fallback_env.split(",")
            if x.strip()
        ]

        existing = {x["pdb_id"] for x in out}

        log.warning(
            "Using generic fallback PDBs because no adequate RCSB target was found "
            "(best_real_score=%.3f, min_real_candidate_score=%.3f): %s",
            best_real_score,
            min_real_candidate_score,
            fallback_ids,
        )

        for pdb_id in fallback_ids:
            if pdb_id in exclude or pdb_id in existing:
                continue

            out.append(
                {
                    "pdb_id": pdb_id,
                    "score": -999.0,
                    "title": "Generic fallback target",
                    "release_date": "",
                    "year": None,
                    "methods": [],
                    "resolution": None,
                    "entity_descriptions": [],
                    "polymer_types": [],
                    "organisms": [],
                    "reasons": ["env_fallback"],
                    "metadata": {
                        "query_text": query_context,
                        "successful_queries": successful_queries,
                        "failed_queries": failed_queries,
                    },
                }
            )

            existing.add(pdb_id)

            if len(out) >= max_candidates:
                break

    return out[:max_candidates]


def select_pdb_ids(
    topic: str = "",
    hypothesis: str = "",
    virus_name: str = "",
    virus_family: str = "",
    exclude_pdbs: Optional[List[str]] = None,
) -> List[str]:
    ranked = select_pdb_targets(
        topic=topic,
        hypothesis=hypothesis,
        virus_name=virus_name,
        virus_family=virus_family,
        exclude_pdbs=exclude_pdbs,
    )

    return [x["pdb_id"] for x in ranked if x.get("pdb_id")]