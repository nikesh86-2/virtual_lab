from __future__ import annotations


def normalise_lit_query(q: str) -> str:
    """
    Clean and shorten literature query strings.
    """
    if not q:
        return ""

    q = str(q).strip().strip('"').strip("'")

    for ch in [":", ";", ",", "(", ")"]:
        q = q.replace(ch, " ")

    q = " ".join(q.split())

    words = q.split()

    if len(words) > 10:
        q = " ".join(words[:10])

    return q


def keep_research_query(q: str) -> bool:
    """
    Filter generated literature queries to avoid broad/off-domain retrieval.
    """
    if not q:
        return False

    t = q.lower()

    required_any_rna = [
        "rna",
        "stem-loop",
        "stem loop",
        "ires",
        "utr",
    ]

    domain_anchor = [
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
        "phosphate glass",
        "plantaricin",
        "cancer drug",
        "anti-cancer",
        "anticancer",
        "dna secondary structure",
        "thermal melting curves",
        "algorithm for dna",
    ]

    return (
        any(x in t for x in required_any_rna)
        and any(x in t for x in domain_anchor)
        and not any(x in t for x in forbidden)
    )


def keep_literature_text(title: str, abstract: str = "") -> bool:
    """
    Filter literature results to keep viral RNA/capsid relevant papers only.
    """
    text = f"{title or ''} {abstract or ''}".lower()

    if not text.strip():
        return False

    required_any_rna = [
        "rna",
        "stem-loop",
        "stem loop",
        "ires",
        "utr",
        "ribonucleic",
    ]

    domain_anchor = [
        "virus",
        "viral",
        "capsid",
        "coat protein",
        "packaging",
        "virion",
        "assembly",
        "rna-binding",
        "rna binding",
        "rna-protein",
        "rna protein",
    ]

    forbidden = [
        "multi-agent",
        "large language model",
        "llm",
        "manufacturing",
        "pose graph",
        "robot",
        "slam",
        "jailbreak",
        "clinical trial multi-agent",
        "phosphate glass",
        "plantaricin",
        "anti-cancer drug",
        "anticancer drug",
    ]

    return (
        any(x in text for x in required_any_rna)
        and any(x in text for x in domain_anchor)
        and not any(x in text for x in forbidden)
    )


def fallback_literature_query(hypothesis: str = "") -> str:
    """
    Fallback query when LLM-generated query is rejected.
    """
    h = (hypothesis or "").lower()

    if "hpev" in h or "parechovirus" in h:
        return "HPeV1 RNA stem-loop viral capsid binding"

    if "picornaviridae" in h:
        return "Picornaviridae RNA stem-loop capsid packaging"

    return "viral RNA stem-loop capsid binding"