import re
from typing import Any


def _normalise_query(q: str) -> str:
    if not q:
        return ""

    q = q.strip().strip('"').strip("'")
    q = re.sub(r"[^A-Za-z0-9'\-\s]", " ", q)
    q = " ".join(q.split())

    words = q.split()
    if len(words) > 9:
        q = " ".join(words[:9])

    return q


def _query_allowed(q: str) -> bool:
    """
    Local copy of query filter to avoid importing orchestrator code here.
    """
    if not q:
        return False

    t = q.lower()

    has_rna = any(
        x in t
        for x in [
            "rna",
            "stem-loop",
            "stem loop",
            "ires",
            "utr",
        ]
    )

    has_viral_anchor = any(
        x in t
        for x in [
            "virus",
            "viral",
            "capsid",
            "coat protein",
            "packaging",
            "assembly",
            "rna binding",
            "rna-protein",
            "rna protein",
            "docking",
        ]
    )

    forbidden = [
        "multi-agent",
        "language model",
        "llm",
        "framework",
        "manufacturing",
        "pose graph",
        "robot",
        "slam",
        "education",
        "clinical trial",
        "jailbreak",
        "glass",
        "plantaricin",
        "cancer",
        "dna secondary structure",
        "thermal melting",
    ]

    return has_rna and has_viral_anchor and not any(x in t for x in forbidden)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []

    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)

    return out


def generate_refined_queries(
    parsed: dict[str, Any] | None,
    hypothesis: str | None = None,
    max_queries: int = 6,
) -> list[str]:
    """
    Generate focused literature queries from parsed Skeptic output.

    Design goal:
      Query refinement should support the RNA-capsid docking task without
      polluting FAISS with generic RNA, generic MD, ML, materials, or clinical
      literature.

    Returns:
      List of filtered keyword queries.
    """
    parsed = parsed or {}
    h = (hypothesis or "").lower()

    concerns = parsed.get("concerns", []) or []
    missing = parsed.get("missing_controls", []) or []

    if isinstance(concerns, str):
        concerns_text = concerns.lower()
    else:
        concerns_text = " ".join(map(str, concerns)).lower()

    if isinstance(missing, str):
        missing_text = missing.lower()
    else:
        missing_text = " ".join(map(str, missing)).lower()

    all_text = f"{h} {concerns_text} {missing_text}"

    # ------------------------------------------------------------------
    # Always-use core queries for this project domain
    # ------------------------------------------------------------------
    queries = [
        "viral coat protein RNA stem-loop binding",
    ]

    # ------------------------------------------------------------------
    # Add targeted queries based on missing controls / critique
    # ------------------------------------------------------------------
    if any(x in all_text for x in ["conservation", "alifold", "msa", "phylogen"]):
        queries.extend(
            [
                "Viral RNA stem-loop conservation packaging",
                "RNA structure conservation capsid packaging",
            ]
        )

    if any(x in all_text for x in ["vina", "docking", "binding energy", "affinity"]):
        queries.extend(
            [
                "viral capsid RNA docking binding energy",
                "coat protein RNA stem-loop docking",
                "viral RNA capsid binding affinity",
            ]
        )

    if any(x in all_text for x in ["md", "molecular dynamics", "stability", "trajectory"]):
        queries.extend(
            [
                "viral RNA stem-loop molecular dynamics stability",
                "RNA stem-loop capsid binding molecular dynamics",
            ]
        )

    if any(x in all_text for x in ["solvent", "ion", "magnesium", "force-field", "force field"]):
        queries.extend(
            [
                "viral RNA capsid binding ions solvent",
                "RNA stem-loop capsid interaction magnesium ions",
            ]
        )

    if any(x in all_text for x in ["entropy", "ensemble", "pair_density", "base pair"]):
        queries.extend(
            [
                "viral RNA stem-loop ensemble capsid binding",
                "RNA stem-loop base pairing viral packaging",
            ]
        )

    if any(x in all_text for x in ["packaging", "assembly", "capsid"]):
        queries.extend(
            [
                "viral RNA packaging signal capsid assembly",
            ]
        )

    # ------------------------------------------------------------------
    # Normalise, filter, dedupe
    # ------------------------------------------------------------------
    queries = [_normalise_query(q) for q in queries]
    queries = [q for q in queries if _query_allowed(q)]
    queries = _dedupe_keep_order(queries)

    return queries[:max_queries]
