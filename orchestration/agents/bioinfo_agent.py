from __future__ import annotations

import json
import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage

from VLAB2.orchestration.llm import get_llm
from VLAB2.orchestration.state_schema import (
    LabState,
    safe_jsonable,
    add_conversation_entry,
    record_stage_output,
)

from VLAB2.orchestration.utils.checkpointing import save_checkpoint
from VLAB2.orchestration.utils.sequence_utils import clean_rna, dedupe_rna_sequences


log = logging.getLogger("virtual_lab")


__all__ = ["bioinfo_agent"]


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except Exception:
        return max(minimum, default)


def _count_fasta_records(text: str) -> int:
    return sum(1 for line in str(text or "").splitlines() if line.startswith(">"))


def _alignment_length(msa: str) -> int:
    lengths = []
    current = []
    for line in str(msa or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                lengths.append(len("".join(current)))
                current = []
        else:
            current.append(line)
    if current:
        lengths.append(len("".join(current)))
    return max(lengths) if lengths else 0


def _normalise_fraction(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def _normalise_position_scores(values) -> list[float]:
    out = []
    for value in values or []:
        try:
            value = float(value)
        except Exception:
            continue
        if value < 0.0:
            value = 0.0
        out.append(value)
    if not out:
        return []
    # Entropy-like vectors are converted to conservation-like values.
    if max(out) > 1.5:
        out = [1.0 / (1.0 + value) for value in out]
    return [max(0.0, min(1.0, value)) for value in out]


def _quality_gate_conservation(
    conservation_signal: dict,
    analysis_dict: dict,
    msa: str,
) -> dict:
    signal = dict(conservation_signal or {})
    num_sequences = int(
        analysis_dict.get("num_sequences")
        or signal.get("num_sequences")
        or _count_fasta_records(msa)
        or 0
    )
    aln_len = int(
        analysis_dict.get("alignment_length")
        or signal.get("alignment_length")
        or _alignment_length(msa)
        or 0
    )
    min_sequences = _env_int("VLAB_BIOINFO_MIN_MSA_SEQUENCES", 3, 1)
    min_alignment_length = _env_int("VLAB_BIOINFO_MIN_ALIGNMENT_LENGTH", 20, 1)
    quality_reasons = []
    if num_sequences < min_sequences:
        quality_reasons.append(
            f"too_few_sequences:{num_sequences}<{min_sequences}"
        )
    if aln_len < min_alignment_length:
        quality_reasons.append(
            f"alignment_too_short:{aln_len}<{min_alignment_length}"
        )
    signal["num_sequences"] = num_sequences
    signal["alignment_length"] = aln_len
    signal["quality_reasons"] = quality_reasons
    signal["quality_passed"] = not quality_reasons
    signal["valid"] = bool(signal.get("valid")) and not quality_reasons
    return signal


def bioinfo_agent(state: LabState) -> dict:
    """
    Run conservation analysis for the current topic/design context.

    Preferred path:
      BioinfoWrapper.run_pipeline()

    Legacy fallback:
      - infer NCBI taxon ID using LLM
      - fetch related genomes
      - run MSA
      - calculate conservation
    """
    log.info("--- BIOINFO AGENT: Checking Conservation ---")

    try:
        bw = state["wrappers"]["bioinfo"]

        taxonomy_context = " ".join(
            x for x in [
                state.get("virus_name", ""),
                state.get("virus_genus", ""),
                state.get("virus_family", ""),
            ]
            if x
        )

        topic_text = " ".join(
            x for x in [
                taxonomy_context,
                state.get("hypothesis")
                or state.get("topic_description")
                or state.get("research_topic")
                or "",
            ]
            if x
        )

        target_sequence = clean_rna(state.get("target_sequence", "")) or None

        designed_sequences = dedupe_rna_sequences(
            state.get("designed_sequences", [])
        )
        clean_interface_sequence = clean_rna(
            state.get("best_interface_clean_sequence")
            or state.get("_run_system_best_interface_sequence")
            or ""
        )
        if clean_interface_sequence:
            designed_sequences = dedupe_rna_sequences(
                [clean_interface_sequence, *designed_sequences]
            )

        motif_hints = state.get("literature_motif_hints", []) or []
        target_hints = state.get("literature_target_hints", []) or []
        if motif_hints or target_hints:
            topic_text = " ".join(
                x for x in [
                    topic_text,
                    "target hints " + " ".join(map(str, target_hints[:8])),
                    "motif hints " + " ".join(map(str, motif_hints[:8])),
                ]
                if x
            )

        # ------------------------------------------------------------
        # Preferred upgraded BioinfoWrapper path
        # ------------------------------------------------------------
        if hasattr(bw, "run_pipeline"):
            analysis_dict = bw.run_pipeline(
                topic=topic_text,
                target_sequence=target_sequence,
                designed_sequences=designed_sequences,
                limit=int(os.getenv("VLAB_BIOINFO_FETCH_LIMIT", "20")),
            )

        else:
            # --------------------------------------------------------
            # Legacy fallback
            # --------------------------------------------------------
            lookup = get_llm(temperature=0.0).invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a bioinformatician with expert knowledge of NCBI taxonomy.\n\n"
                            "Extract the NCBI Taxon ID for the primary virus or organism named.\n"
                            "Return ONLY a single integer. If ambiguous, return 0.\n\n"
                            "Examples:\n"
                            "HIV-1 → 11676\n"
                            "SARS-CoV-2 → 2697049\n"
                            "HCV → 11103\n"
                            "HPeV1 → 12110\n"
                            "Influenza A → 11520\n"
                            "Poliovirus → 12081\n"
                            "E. coli → 562\n"
                            "Homo sapiens → 9606"
                        )
                    ),
                    HumanMessage(content=topic_text),
                ]
            )

            txid = "".join(c for c in lookup.content if c.isdigit())

            if not txid or txid == "0":
                return {
                    "bioinfo_analysis": (
                        "Skipped: No reliable NCBI Taxon ID identified. "
                        "Conservation analysis requires an unambiguous organism name."
                    ),
                    "conservation_signal": {"valid": False},
                }

            fasta = bw.fetch_related_genomes(
                txid,
                limit=int(os.getenv("VLAB_BIOINFO_FETCH_LIMIT", "10"))
            )

            if ">" not in fasta:
                return {
                    "bioinfo_analysis": "No genomes found for conservation analysis.",
                    "conservation_signal": {"valid": False},
                }

            msa = bw.run_msa(fasta)
            analysis_dict = bw.calculate_conservation(msa)
            analysis_dict["msa"] = msa
            analysis_dict["fasta"] = fasta

        # ------------------------------------------------------------
        # Normalise conservation signal
        # ------------------------------------------------------------
        msa = analysis_dict.get("msa", "")
        conservation_signal = analysis_dict.get("conservation_signal", {})

        if not conservation_signal:
            conservation_signal = {
                "valid": bool(analysis_dict.get("valid")),
                "consensus": analysis_dict.get("consensus", ""),
                "position_scores": analysis_dict.get("position_scores", []),
                "entropy_scores": analysis_dict.get("entropy_scores", []),
                "conserved_regions": analysis_dict.get("conserved_regions", []),
                "motif_scores": analysis_dict.get("motif_scores", {}),
                "selected_motifs": analysis_dict.get("selected_motifs", []),
                "conservation_pct": analysis_dict.get("conservation_pct", 0.0),
            }

        # Scalar conservation_fitness expected by PI agent. Position scores
        # are normalised first so entropy-like and percentage-like wrapper
        # outputs cannot silently inflate the PI signal.
        position_scores = _normalise_position_scores(
            conservation_signal.get("position_scores", [])
        )
        conservation_signal["position_scores"] = position_scores

        if position_scores:
            conservation_fitness = sum(position_scores) / len(position_scores)
        else:
            conservation_fitness = _normalise_fraction(
                analysis_dict.get(
                    "conservation_pct",
                    conservation_signal.get("conservation_pct", 0.0),
                )
            )

        conservation_signal["conservation_fitness"] = max(
            0.0,
            min(1.0, conservation_fitness),
        )
        conservation_signal = _quality_gate_conservation(
            conservation_signal,
            analysis_dict,
            msa,
        )
        if not conservation_signal.get("quality_passed", False):
            conservation_signal["conservation_fitness"] *= 0.25

        log.info(
            "Conservation fitness: %.3f | regions=%d | motifs=%s",
            conservation_signal["conservation_fitness"],
            len(conservation_signal.get("conserved_regions", [])),
            conservation_signal.get("selected_motifs", [])[:5],
        )

        analysis_str = json.dumps(safe_jsonable(analysis_dict), indent=2)

        result = {
            "bioinfo_analysis": analysis_str,
            "msa_data": msa,
            "conservation_signal": conservation_signal,
            "conserved_regions": conservation_signal.get("conserved_regions", []),
            "bioinfo_num_sequences": conservation_signal.get("num_sequences", 0),
            "bioinfo_alignment_length": conservation_signal.get("alignment_length", 0),
            "bioinfo_quality_passed": conservation_signal.get("quality_passed", False),
            "bioinfo_quality_reasons": conservation_signal.get("quality_reasons", []),
            "_run_system_selected_motifs": conservation_signal.get("selected_motifs", []),
            "stage_outputs": [
                record_stage_output(
                    state,
                    "bioinfo",
                    analysis_str,
                    summary=(
                        f"MSA conservation: "
                        f"{analysis_dict.get('conservation_pct', 0):.1f}% "
                        f"over {analysis_dict.get('num_sequences', '?')} seqs"
                    ),
                    metadata={
                        "conservation_valid": conservation_signal.get("valid", False),
                        "msa_size": len(msa or ""),
                        "conservation_fitness": conservation_signal.get(
                            "conservation_fitness",
                            0.0,
                        ),
                        "selected_motifs": conservation_signal.get(
                            "selected_motifs",
                            [],
                        )[:10],
                        "num_sequences": conservation_signal.get("num_sequences", 0),
                        "alignment_length": conservation_signal.get("alignment_length", 0),
                        "quality_passed": conservation_signal.get("quality_passed", False),
                        "quality_reasons": conservation_signal.get("quality_reasons", []),
                    },
                )
            ],
            "conversation_history": [
                add_conversation_entry(state, "assistant", analysis_str, "bioinfo")
            ],
        }

        # Optional collector if present in state.
        data_collector = state.get("data_collector")

        if data_collector is not None:
            try:
                data_collector.capture_agent_interaction(
                    "Bioinfo",
                    "Compute MSA and conservation",
                    analysis_str,
                )
            except Exception as e:
                log.warning("Bioinfo data collection failed/non-fatal: %s", e)

        save_checkpoint({**state, **result})
        return result

    except Exception as e:
        log.exception("BIOINFO AGENT ERROR")

        return {
            "bioinfo_analysis": f"Conservation analysis failed: {e}",
            "conservation_signal": {"valid": False},
        }