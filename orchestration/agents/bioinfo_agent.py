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
from VLAB2.orchestration.utils.sequence_utils import dedupe_rna_sequences


log = logging.getLogger("virtual_lab")


__all__ = ["bioinfo_agent"]


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

        target_sequence = state.get("target_sequence")

        designed_sequences = dedupe_rna_sequences(
            state.get("designed_sequences", [])
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

        # Scalar conservation_fitness expected by PI agent.

        position_scores = conservation_signal.get("position_scores", []) or []

        if position_scores:
            vals = [float(x) for x in position_scores]

            # heuristic: detect entropy-like scale
            if max(vals) > 1.5:
                vals = [1.0 / (1.0 + v) for v in vals]

            conservation_fitness = sum(vals) / max(1, len(vals))
        else:
            conservation_fitness = (
                float(analysis_dict.get("conservation_pct", 0.0) or 0.0)
                / 100.0
            )

        conservation_signal["conservation_fitness"] = max(
            0.0,
            min(1.0, conservation_fitness),
        )

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