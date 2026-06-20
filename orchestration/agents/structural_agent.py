from __future__ import annotations

import json
import logging
import os
import random
import re

from langchain_core.messages import HumanMessage, SystemMessage

from VLAB2.optimisation.rna_grammar_generator import generate_structured_rna

from VLAB2.orchestration.llm import get_llm
from VLAB2.orchestration.state_schema import (
    LabState,
    safe_jsonable,
    add_conversation_entry,
    record_stage_output,
)

from VLAB2.orchestration.utils.checkpointing import save_checkpoint
from VLAB2.orchestration.utils.env_utils import getenv_float
from VLAB2.orchestration.utils.fold_utils import extract_fold_quality_from_outputs
from VLAB2.orchestration.utils.sequence_utils import (
    clean_rna,
    dedupe_rna_sequences,
    hamming_distance,
    mutate_sequence,
    validate_sequence,
)
from VLAB2.orchestration.utils.text_utils import (
    clean_llm_output,
    clean_structural_output,
    truncate_str,
    validate_text_block,
)


log = logging.getLogger("virtual_lab")

__all__ = ["structural_agent"]


def _parse_motif_names_from_env() -> list[str]:
    """
    Parse motif names from VLAB_RNA_TARGET_MOTIFS.

    Example:
      GAG:0.55,AAG:0.55,AGAG:0.30
    """
    raw = os.getenv(
        "VLAB_RNA_TARGET_MOTIFS",
        "GAG:0.55,AAG:0.55,CAG:0.35,GUC:0.35,AGU:0.35",
    )

    motifs = []

    for item in raw.split(","):
        item = item.strip()

        if not item:
            continue

        motif = item.split(":", 1)[0].strip().upper().replace("T", "U")

        if motif:
            motifs.append(motif)

    return list(dict.fromkeys(motifs))


def _motif_to_regex(motif: str) -> str:
    """
    Convert simple IUPAC RNA motif to regex.

    Supports R=A/G. Other non-ACGU characters are escaped literally.
    """
    mapping = {
        "A": "A",
        "C": "C",
        "G": "G",
        "U": "U",
        "R": "[AG]",
        "Y": "[CU]",
        "N": "[ACGU]",
    }

    return "".join(mapping.get(c, re.escape(c)) for c in motif.upper())


def _has_required_motif(seq: str, motifs: list[str]) -> bool:
    seq = clean_rna(seq)

    if not seq:
        return False

    for motif in motifs:
        pattern = _motif_to_regex(motif)

        if re.search(pattern, seq):
            return True

    return False


def _add_mfe_per_nt_if_missing(obj: dict, seq: str) -> dict:
    """
    Add mfe_per_nt to a folding output dict if it has mfe but lacks mfe_per_nt.
    """
    if not isinstance(obj, dict):
        return {}

    out = dict(obj)

    if out.get("mfe_per_nt") is None and out.get("mfe") is not None:
        try:
            out["mfe_per_nt"] = float(out["mfe"]) / max(1, len(seq))
        except Exception:
            pass

    return out


def _rank_fold_quality(fq: dict) -> float:
    """
    Rank fold quality using pair density, mfe_per_nt, and pass/fail status.

    Higher score is better.
    """
    pd = fq.get("best_pair_density")
    mfe_nt = fq.get("best_mfe_per_nt")

    pd_score = 0.0 if pd is None else min(1.0, max(0.0, float(pd) / 0.35))

    if mfe_nt is None:
        mfe_score = 0.0
    else:
        # More negative is better up to a useful cap.
        mfe_score = min(1.0, max(0.0, abs(float(mfe_nt)) / 0.30))

    # Keep pass/fail important, but do not let a barely-passing fold dominate
    # a much stronger near-threshold fold.
    pass_bonus = 0.7 if fq.get("passed") else 0.0

    return 0.45 * pd_score + 0.35 * mfe_score + 0.20 * pass_bonus


def _parse_sequences_from_design_output(text: str) -> list[str]:
    """
    Parse RNA sequences from LLM design output.

    Preferred format:
      SEQ: ACGU...

    Fallback:
      regex extraction of 25–60 nt RNA/DNA-like strings.
    """
    raw_sequences = []

    for line in str(text or "").strip().splitlines():
        if line.upper().startswith("SEQ:"):
            seq = line.split(":", 1)[1].strip().upper().replace("T", "U")
            seq = "".join(c for c in seq if c in "ACGU")

            if seq:
                raw_sequences.append(seq)

    if not raw_sequences:
        tokens = re.findall(r"[ACGUTacgut]{25,60}", str(text or ""))

        for token in tokens:
            seq = token.upper().replace("T", "U")
            seq = "".join(c for c in seq if c in "ACGU")

            if seq:
                raw_sequences.append(seq)

    return dedupe_rna_sequences(raw_sequences)


def _build_locked_positions(conserved_regions) -> set[int]:
    """
    Convert conserved region intervals into locked nucleotide positions.
    """
    locked_positions = set()

    for item in conserved_regions or []:
        try:
            start, end = item
            locked_positions.update(range(int(start), int(end)))
        except Exception:
            continue

    return locked_positions


def _too_close_to_existing(seq: str, existing: list[str], max_distance: int = 4) -> bool:
    for other in existing:
        if len(seq) == len(other) and hamming_distance(seq, other) <= max_distance:
            return True

    return False


def _generate_fallback_sequence(
    locked_positions: set[int],
    motifs: list[str],
    require_motif: bool,
) -> str:
    """
    Generate a fallback structured RNA and repair if needed.
    """
    try:
        seq = generate_structured_rna(length=30)
    except Exception:
        seq = "".join(random.choice("ACGU") for _ in range(30))

    seq = clean_rna(seq)
    seq = mutate_sequence(seq, 2, locked_positions)

    if require_motif and motifs and not _has_required_motif(seq, motifs):
        # Insert a literal motif if possible. For ambiguous motifs, pick a safe
        # concrete version.
        motif = motifs[0].replace("R", "A").replace("Y", "C").replace("N", "A")
        motif = clean_rna(motif)

        if motif and len(seq) >= len(motif) + 4:
            pos = max(2, min(len(seq) - len(motif) - 2, len(seq) // 2))
            seq = seq[:pos] + motif + seq[pos + len(motif):]

    return clean_rna(seq)


def _normalise_candidate_sequences(
    raw_sequences: list[str],
    prior_seqs: list[str],
    locked_positions: set[int],
) -> list[str]:
    """
    Validate, clean, dedupe, motif-check, and repair candidate RNA sequences.
    """
    final_sequences = []

    motifs = _parse_motif_names_from_env()
    require_motif = os.getenv("VLAB_RNA_REQUIRE_TARGET_MOTIF", "0").strip() == "1"

    for seq in raw_sequences:
        seq = clean_rna(seq)

        is_valid, msg = validate_sequence(seq)

        if not is_valid:
            log.warning(
                "Invalid designed sequence (%s) — replacing with grammar-generated candidate",
                msg,
            )

            seq = _generate_fallback_sequence(
                locked_positions=locked_positions,
                motifs=motifs,
                require_motif=require_motif,
            )

            is_valid, msg = validate_sequence(seq)

            if not is_valid:
                log.warning("Fallback sequence also invalid (%s): %s", msg, seq)
                continue

        if not seq:
            continue

        if seq in prior_seqs or _too_close_to_existing(seq, prior_seqs, max_distance=2):
            seq = mutate_sequence(seq, 2, locked_positions)
            seq = clean_rna(seq)

            is_valid, msg = validate_sequence(seq)

            if not is_valid:
                log.warning("Mutated duplicate became invalid (%s): %s", msg, seq)
                continue

        if require_motif and motifs and not _has_required_motif(seq, motifs):
            log.warning("Dropping candidate lacking required motif: %s", seq)
            continue

        if _too_close_to_existing(seq, final_sequences, max_distance=4):
            log.debug("Dropping low-diversity candidate: %s", seq)
            continue

        final_sequences.append(seq)

    final_sequences = dedupe_rna_sequences(final_sequences)

    # Ensure at least 3 candidates if possible.
    attempts = 0

    while len(final_sequences) < 3 and attempts < 40:
        attempts += 1

        seq = _generate_fallback_sequence(
            locked_positions=locked_positions,
            motifs=motifs,
            require_motif=require_motif,
        )

        seq = clean_rna(seq)
        is_valid, _ = validate_sequence(seq)

        if not is_valid:
            continue

        if require_motif and motifs and not _has_required_motif(seq, motifs):
            continue

        if _too_close_to_existing(seq, final_sequences, max_distance=4):
            continue

        final_sequences.append(seq)
        final_sequences = dedupe_rna_sequences(final_sequences)

    return final_sequences[:5]


def _run_alifold_if_available(state: LabState, vienna) -> dict | str:
    """
    Run RNAalifold where MSA data is available.

    Non-fatal: returns empty string on failure/no data.
    """
    msa = state.get("msa_data", "")

    if not (msa and ">" in msa):
        return ""

    try:
        if hasattr(vienna, "run_rnaalifold"):
            return vienna.run_rnaalifold(msa)

        if (
            "bioinfo" in state.get("wrappers", {})
            and hasattr(state["wrappers"]["bioinfo"], "run_rnaalifold")
        ):
            return state["wrappers"]["bioinfo"].run_rnaalifold(msa)

        return ""

    except Exception as e:
        log.warning("RNAalifold failed/non-fatal: %s", e)
        return ""


def _should_use_alifold_for_hard_gate(state: LabState, alifold_output) -> bool:
    """
    Decide whether RNAalifold contributes to hard fold threshold gating.

    If conservation is weak, RNAalifold is advisory by default because
    poor/incoherent MSA can produce fully unpaired consensus folds.
    """
    if not isinstance(alifold_output, dict):
        return False

    conservation_signal = state.get("conservation_signal", {}) or {}
    conservation_fitness = float(
        conservation_signal.get("conservation_fitness", 0.0) or 0.0
    )

    alifold_min_conservation = getenv_float(
        "VLAB_ALIFOLD_MIN_CONSERVATION_FITNESS",
        0.5,
    )

    alifold_advisory_when_low_conservation = (
        os.getenv("VLAB_ALIFOLD_ADVISORY_WHEN_LOW_CONSERVATION", "1").strip() == "1"
    )

    return (
        conservation_fitness >= alifold_min_conservation
        or not alifold_advisory_when_low_conservation
    )


def _add_alifold_advisory_warning(
    state: LabState,
    fold_quality: dict,
    alifold_output,
) -> dict:
    """
    Attach advisory warning when RNAalifold was run but excluded from hard gating.
    """
    if not isinstance(alifold_output, dict):
        return fold_quality

    conservation_signal = state.get("conservation_signal", {}) or {}
    conservation_fitness = float(
        conservation_signal.get("conservation_fitness", 0.0) or 0.0
    )

    alifold_min_conservation = getenv_float(
        "VLAB_ALIFOLD_MIN_CONSERVATION_FITNESS",
        0.5,
    )

    fold_quality.setdefault("advisory_warnings", [])
    fold_quality["advisory_warnings"].append(
        {
            "source": "alifold",
            "reason": (
                "RNAalifold excluded from hard fold gate because conservation "
                f"fitness={conservation_fitness:.3f} < {alifold_min_conservation:.3f}"
            ),
            "alifold_pair_density": alifold_output.get("pair_density"),
            "alifold_mfe_per_nt": alifold_output.get("mfe_per_nt"),
            "alifold_threshold_reasons": alifold_output.get("threshold_reasons", []),
        }
    )

    return fold_quality


def structural_agent(state: LabState) -> dict:
    """
    Multi-candidate RNA structural design and folding agent.

    Behaviour:
      - asks LLM for 3–5 diverse RNA candidates
      - validates/deduplicates candidates
      - runs SFold and ViennaRNA for every candidate
      - optionally runs RNAalifold if MSA is available
      - treats RNAalifold as advisory when conservation is weak
      - selects the best-folding candidate as target_sequence
      - returns structural_candidates for MD/protein downstream
    """
    log.info("--- STRUCTURAL AGENT: Running Simulations ---")

    try:
        prior_seqs = dedupe_rna_sequences(state.get("designed_sequences", []))
        critique = state.get("critique", "") or ""

        motifs = _parse_motif_names_from_env()
        motif_text = ", ".join(motifs[:12]) if motifs else "GAG, AAG, CAG, GUC, AGU"

        exclusion_block = ""

        if prior_seqs:
            exclusion_block = (
                "\nPreviously designed sequences (must not duplicate):\n"
                + "\n".join(f"- {s}" for s in prior_seqs[-10:])
            )

        redesign_note = ""

        if critique:
            for line in critique.splitlines():
                if line.startswith("RECOMMENDATION:") or line.startswith("CONCERNS:"):
                    redesign_note += line + "\n"

            if redesign_note:
                redesign_note = f"\nSkeptic guidance:\n{redesign_note.strip()}"

        design_msg = get_llm(temperature=0.35).invoke(
            [
                SystemMessage(
                    content=(
                        "You are an RNA structural biologist designing candidate "
                        "functional RNA sequences.\n\n"
                        "Design 3–5 DIVERSE RNA sequences, each 25–40 nt, satisfying:\n\n"
                        "Core requirements:\n"
                        "1. Each sequence forms a stem-loop with an exposed recognition motif\n"
                        f"2. Prefer motifs from this soft-prior set: {motif_text}\n"
                        "3. GC content should be 40–60% where possible\n"
                        "4. No homopolymer runs of 4 or more\n"
                        "5. Predicted MFE < -5 kcal/mol and pair_density >= 0.24\n"
                        "6. Distinct from previous sequences\n\n"
                        "Diversity constraints, mandatory:\n"
                        "- Sequences must differ from each other by at least 5 positions where possible\n"
                        "- Use at least two different motif types across the set where possible\n"
                        "- Vary motif position, loop length, or stem length across candidates\n"
                        "- Avoid trivial single-mutation variants of the same design\n\n"
                        "Design strategy guidance:\n"
                        "- Vary stem length between 6–10 paired bases\n"
                        "- Vary loop size between 3–8 nt\n"
                        "- Place motifs either in the loop centre or at the stem-loop junction\n"
                        "- Include at least one stability-prioritised candidate and one motif-exposure-prioritised candidate\n"
                        "- Prefer pair_density >= 0.35 where possible\n"
                        "- Prefer mfe_per_nt <= -0.15 where possible\n\n"
                        "Output EXACTLY this repeated format, with no extra text:\n"
                        "MOTIF: <motif>@<start-end>\n"
                        "SEQ: <uppercase RNA sequence>\n\n"
                        "MOTIF: <motif>@<start-end>\n"
                        "SEQ: <uppercase RNA sequence>"
                    )
                ),
                HumanMessage(
                    content=(
                        (state.get("hypothesis") or state.get("research_topic", ""))
                        + exclusion_block
                        + redesign_note
                    )
                ),
            ]
        )

        raw_sequences = _parse_sequences_from_design_output(design_msg.content)

        conserved_regions = state.get("conserved_regions", [])
        locked_positions = _build_locked_positions(conserved_regions)

        final_sequences = _normalise_candidate_sequences(
            raw_sequences=raw_sequences,
            prior_seqs=prior_seqs,
            locked_positions=locked_positions,
        )

        if not final_sequences:
            raise RuntimeError("No valid RNA sequences generated by structural agent.")

        log.info(
            "    Generated %d candidate RNA sequences: %s",
            len(final_sequences),
            ", ".join(final_sequences),
        )

        sfold = state["wrappers"]["sfold"]
        vienna = state["wrappers"]["vienna"]

        alifold_output = _run_alifold_if_available(state, vienna)

        candidate_records = []

        for seq in final_sequences:
            try:
                sfold_output = sfold.run_sfold(seq)
            except Exception as e:
                log.warning("SFold failed for %s: %s", seq[:15], e)
                sfold_output = {}

            try:
                vienna_output = vienna.run_rnafold(seq)
            except Exception as e:
                log.warning("ViennaRNA failed for %s: %s", seq[:15], e)
                vienna_output = {}

            sfold_output = _add_mfe_per_nt_if_missing(
                sfold_output if isinstance(sfold_output, dict) else {},
                seq,
            )

            vienna_output = _add_mfe_per_nt_if_missing(
                vienna_output if isinstance(vienna_output, dict) else {},
                seq,
            )

            use_alifold_for_hard_gate = _should_use_alifold_for_hard_gate(
                state,
                alifold_output,
            )

            fold_quality = extract_fold_quality_from_outputs(
                sfold_output=sfold_output,
                vienna_output=vienna_output,
                alifold_output=(
                    alifold_output
                    if use_alifold_for_hard_gate and isinstance(alifold_output, dict)
                    else {}
                ),
            )

            if isinstance(alifold_output, dict) and not use_alifold_for_hard_gate:
                fold_quality = _add_alifold_advisory_warning(
                    state,
                    fold_quality,
                    alifold_output,
                )

            candidate_records.append(
                {
                    "sequence": seq,
                    "sfold_output": sfold_output,
                    "vienna_output": vienna_output,
                    "fold_quality": fold_quality,
                    "rank_score": _rank_fold_quality(fold_quality),
                }
            )

        candidate_records.sort(
            key=lambda r: (
                bool(r["fold_quality"].get("passed")),
                float(r.get("rank_score", 0.0)),
            ),
            reverse=True,
        )

        for rec in candidate_records[:5]:
            fq = rec["fold_quality"]
            log.info(
                "    Candidate fold rank: seq=%s score=%.3f pass=%s pd=%s mfe_nt=%s",
                rec["sequence"][:20],
                float(rec.get("rank_score", 0.0)),
                fq.get("passed"),
                fq.get("best_pair_density"),
                fq.get("best_mfe_per_nt"),
            )

        selected = candidate_records[0]

        sequence = selected["sequence"]
        sfold_output = selected["sfold_output"]
        vienna_output = selected["vienna_output"]
        fold_quality = selected["fold_quality"]

        log.info(
            "    Selected target candidate (%d nt): %s | fold_pass=%s | pd=%s | mfe_nt=%s",
            len(sequence),
            sequence,
            fold_quality.get("passed"),
            fold_quality.get("best_pair_density"),
            fold_quality.get("best_mfe_per_nt"),
        )

        clean_sfold = clean_structural_output(str(sfold_output))
        clean_vienna = clean_structural_output(str(vienna_output))
        clean_alifold = clean_structural_output(str(alifold_output))

        candidate_summary_lines = []

        for i, rec in enumerate(candidate_records, start=1):
            fq = rec["fold_quality"]

            candidate_summary_lines.append(
                (
                    f"{i}. {rec['sequence']} | "
                    f"passed={fq.get('passed')} | "
                    f"rank_score={rec.get('rank_score')} | "
                    f"pair_density={fq.get('best_pair_density')} | "
                    f"mfe_per_nt={fq.get('best_mfe_per_nt')} | "
                    f"reasons={fq.get('reasons', [])}"
                )
            )

        candidate_summary = "\n".join(candidate_summary_lines)

        interpret_msg = get_llm(temperature=0.0).invoke(
            [
                SystemMessage(
                    content=(
                        "You are reviewing RNA folding simulation outputs.\n\n"
                        "Important interpretation rules:\n"
                        "- If RNAalifold was excluded from hard fold gating due to low conservation fitness, "
                        "do not use RNAalifold alone to recommend REDESIGN.\n"
                        "- Treat excluded RNAalifold results as advisory conservation caveats only.\n"
                        "- Judge the selected sequence primarily using its individual SFold/ViennaRNA "
                        "fold metrics when fold_quality.passed=True.\n"
                        "- If pair_density and mfe_per_nt pass thresholds, do not call the fold unstable "
                        "solely because RNAalifold pair_density is 0.0.\n\n"
                        "Respond in EXACTLY this format:\n"
                        "STABILITY: [STABLE/UNSTABLE/MARGINAL] — MFE=<value or N/A> kcal·mol⁻¹, pair_density=<value or N/A>, ensemble_diversity=<value or N/A>\n"
                        "MOTIF: [PRESENT/ABSENT/AMBIGUOUS] — <motif and nt positions or N/A>\n"
                        "CONSERVATION: [LIKELY/UNLIKELY/UNKNOWN] — <reason or N/A>\n"
                        "ALIFOLD: [SUPPORTED/UNSUPPORTED/NO_DATA/ADVISORY_ONLY] — <evidence or N/A>\n"
                        "VERDICT: [PROCEED/REDESIGN] — <one sentence citing a number>"
                    )
                )
,
                HumanMessage(
                    content=(
                        f"Selected sequence ({len(sequence)} nt): {sequence}\n\n"
                        f"All candidate fold summary:\n{truncate_str(candidate_summary, 1500)}\n\n"
                        f"Selected SFold output:\n{truncate_str(clean_sfold, 1500)}\n\n"
                        f"Selected ViennaRNA output:\n{truncate_str(clean_vienna, 1000)}\n\n"
                        f"RNAalifold output:\n{truncate_str(clean_alifold, 1000)}"
                    )
                ),
            ]
        )

        interpreted = clean_llm_output(interpret_msg.content)
        ok, _ = validate_text_block(interpreted)

        if not ok:
            interpreted = (
                "STABILITY: N/A\n"
                "MOTIF: N/A\n"
                "CONSERVATION: UNKNOWN — insufficient data\n"
                "ALIFOLD: NO_DATA\n"
                "VERDICT: REDESIGN — structural outputs could not be parsed"
            )

        analysis = (
            f"Structural Analysis for selected sequence {sequence}\n\n"
            f"{interpreted}\n\n"
            f"All generated candidates:\n"
            f"{candidate_summary}\n\n"
            f"Selected fold threshold status:\n"
            f"{json.dumps(safe_jsonable(fold_quality), indent=2)}\n\n"
            f"Raw selected SFold:\n{truncate_str(clean_sfold, 1200)}\n\n"
            f"Raw selected ViennaRNA:\n{truncate_str(clean_vienna, 800)}\n\n"
            f"Raw RNAalifold:\n{truncate_str(clean_alifold, 800)}"
        )

        result = {
            "structural_analysis": analysis,

            # Return 3–5 candidates to seed PI/NSGA-II.
            "designed_sequences": final_sequences,

            # Selected best-folding candidate for MD/protein path.
            "target_sequence": sequence,

            "fold_quality": fold_quality,
            "fold_thresholds_passed": fold_quality.get("passed", False),
            "fold_threshold_reasons": fold_quality.get("reasons", []),

            "structural_candidates": [
                {
                    "sequence": rec["sequence"],
                    "fold_quality": rec["fold_quality"],
                    "rank_score": rec["rank_score"],
                }
                for rec in candidate_records
            ],

            "stage_outputs": [
                record_stage_output(
                    state,
                    "structural",
                    analysis,
                    summary=(
                        f"Generated {len(final_sequences)} RNA candidates; "
                        f"selected best fold candidate with "
                        f"pair_density={fold_quality.get('best_pair_density')}, "
                        f"mfe_per_nt={fold_quality.get('best_mfe_per_nt')}"
                    ),
                    metadata={
                        "selected_sequence": sequence,
                        "candidate_count": len(final_sequences),
                        "designed_sequences": final_sequences,
                        "fold_thresholds_passed": fold_quality.get("passed", False),
                        "pair_density": fold_quality.get("best_pair_density"),
                        "mfe_per_nt": fold_quality.get("best_mfe_per_nt"),
                    },
                )
            ],
            "conversation_history": [
                add_conversation_entry(state, "assistant", analysis, "structural")
            ],
        }

        data_collector = state.get("data_collector")

        if data_collector is not None:
            try:
                data_collector.capture_agent_interaction(
                    "Structural",
                    "Run multi-candidate structural analysis",
                    analysis,
                )
            except Exception as e:
                log.warning("Structural data collection failed/non-fatal: %s", e)

        save_checkpoint({**state, **result})
        return result

    except Exception as e:
        log.exception("STRUCTURAL AGENT ERROR")
        return {"structural_analysis": f"Structural analysis failed: {e}"}