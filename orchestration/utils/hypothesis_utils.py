from __future__ import annotations

import re

from VLAB2.orchestration.utils.env_utils import getenv_float
from VLAB2.orchestration.utils.text_utils import truncate_str


def critique_reports_high_spread(critique: str) -> bool:
    """
    Detect whether Skeptic critique reports excessive docking score spread.
    """
    if not critique:
        return False

    max_spread = getenv_float("VLAB_MAX_ACCEPT_SCORE_SPREAD", 10.0)

    matches = re.findall(
        r"spread\s*=?\s*([-+]?\d+(?:\.\d+)?)",
        critique,
        flags=re.IGNORECASE,
    )

    for m in matches:
        try:
            if float(m) > max_spread:
                return True
        except Exception:
            pass

    if "high" in critique.lower() and "spread" in critique.lower():
        return True

    return False


def strip_metric_literals(text: str) -> str:
    """
    Remove overly specific numeric metric literals from critique text before
    sending it to the PI hypothesis-refinement LLM.

    Prevents hypotheses like:
      'score spread will be less than 68.99677938164322...'
    """
    if not text:
        return ""

    text = re.sub(r"[-+]?\d+\.\d{2,}", "<metric_value>", text)

    text = re.sub(
        r"(less than|greater than|below|above|under|over)\s+[-+]?\d+(?:\.\d+)?",
        r"\1 <metric_value>",
        text,
        flags=re.IGNORECASE,
    )

    return text


def critique_signal_summary_for_pi(critique: str) -> str:
    """
    Convert detailed Skeptic critique into qualitative PI guidance.

    Preserves scientific direction without copying exact numerical thresholds.
    """
    if not critique:
        return "No prior critique available."

    c = critique.lower()
    signals = []

    if "spread" in c:
        signals.append(
            "Docking-score variability is too high; favour designs with more consistent HDOCK-relative ranks."
        )

    if "weak" in c or "support: weak" in c:
        signals.append(
            "Current evidence provides weak support; require stronger structural and docking consistency."
        )

    if "fold" in c or "pair_density" in c or "mfe" in c:
        signals.append(
            "Fold support is marginal; favour higher pair density and more favourable mfe_per_nt."
        )

    if "conservation" in c:
        signals.append(
            "Conservation support is limited; favour motifs and regions with stronger conservation signal."
        )

    if "md" in c or "trajectory" in c or "fluctuation" in c:
        signals.append(
            "MD stability is uncertain; favour candidates with lower fluctuation and stable RNA conformations."
        )

    if "redesign_sequences" in c:
        signals.append(
            "Skeptic recommends sequence redesign rather than accepting the current hypothesis."
        )

    if not signals:
        signals.append(strip_metric_literals(truncate_str(critique, 800)))

    return "\n".join(f"- {s}" for s in signals)


def hypothesis_contains_bad_metric_threshold(text: str) -> bool:
    """
    Detect hypotheses that copied exact numeric thresholds from previous results.
    """
    if not text:
        return False

    patterns = [
        r"(less than|greater than|below|above|under|over)\s*-?\d+(?:\.\d+)?",
        r"[<>]=?\s*-?\d+(?:\.\d+)?",
        r"\b\d+\.\d{3,}\b",
        r"score spread will be less than",
        r"best_binding_score will be lower than",
    ]

    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def fallback_qualitative_hypothesis(state: dict) -> str:
    """
    Safe fallback hypothesis when the LLM copies exact thresholds.
    """
    return (
        "Redesigned conserved viral RNA stem-loop candidates with exposed packaging-like "
        "motifs will show stronger and less variable HDOCK-relative docking ranks against "
        "relevant viral capsid RNA-binding targets than prior candidates."
    )