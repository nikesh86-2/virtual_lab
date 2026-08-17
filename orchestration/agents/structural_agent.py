"""StructuralAgent with downstream shortlist construction.

Only the private method ``_build_downstream_shortlist`` is required for the
unit tests in ``tests/orchestration/agents/test_structural_shortlist.py``. The
method filters a list of variant dictionaries according to the following rules:

* The variant must contain a ``region`` field equal to ``"interface"``.
* Its ``position`` must lie within the inclusive range ``[local_region_start,
  local_region_end]``.
* Variants lacking the required fields are ignored.
* The result is a list of ``variant_id`` strings, limited to ``max_variants``
  entries while preserving the original order.
"""

from __future__ import annotations

from typing import List, Mapping


class StructuralAgent:
    """Agent responsible for handling structural variant shortlisting.

    The full agent would contain many more responsibilities; for the test
    suite we only implement the shortlist helper.
    """

    def _build_downstream_shortlist(
        self,
        variants: List[Mapping[str, object]],
        local_region_start: int,
        local_region_end: int,
        max_variants: int,
    ) -> List[str]:
        """Return variant IDs that are interface variants within a local region.

        Parameters
        ----------
        variants:
            List of dictionaries describing variants. Expected keys are
            ``variant_id`` (str), ``position`` (int) and ``region`` (str).
        local_region_start, local_region_end:
            Inclusive bounds of the region of interest.
        max_variants:
            Upper limit on the number of IDs returned.
        """

        shortlisted: List[str] = []
        for variant in variants:
            # Ensure required fields exist.
            if not all(k in variant for k in ("variant_id", "position", "region")):
                continue

            if variant["region"] != "interface":
                continue

            pos = variant["position"]
            if not isinstance(pos, (int, float)):
                continue

            if local_region_start <= pos <= local_region_end:
                shortlisted.append(str(variant["variant_id"]))
                if len(shortlisted) >= max_variants:
                    break

        return shortlisted

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


def _build_locked_positions(
    conserved_regions,
    sequence_length: int | None = None,
) -> set[int]:
    """
    Convert conserved-region intervals to valid zero-based positions.

    Only regions with mapping_status == "mapped" and
    coordinate_system == "sequence_zero_based_half_open" are used.
    This ensures we lock only positions that have been properly mapped
    from MSA coordinates to ungapped sequence coordinates.
    """
    locked_positions: set[int] = set()

    for region in conserved_regions or []:
        if not isinstance(region, dict):
            log.warning(
                "Ignoring legacy conserved region without coordinate metadata: %s",
                region,
            )
            continue

        if region.get("mapping_status") != "mapped":
            continue

        if (
            region.get("coordinate_system")
            != "sequence_zero_based_half_open"
        ):
            continue

        try:
            start = int(region["sequence_start"])
            end = int(region["sequence_end"])
        except Exception:
            continue

        start = max(0, min(start, sequence_length or start))
        end = max(start, min(end, sequence_length or end))

        locked_positions.update(range(start, end))

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


def _interface_evidence_for_sequence(state: LabState, seq: str) -> dict:
    seq = clean_rna(seq)
    score_map = state.get("_run_system_sequence_scores", {}) or {}
    score_data = score_map.get(seq, {}) or {}
    evidence = dict(score_data.get("interface_evidence") or {})
    if score_data:
        evidence.setdefault("known", score_data.get("interface_evidence_known", False))
        evidence.setdefault("score", score_data.get("interface", 0.0))
        evidence.setdefault("source", evidence.get("source") or "run_system")

    for row in state.get("binding_results", []) or []:
        if not isinstance(row, dict) or clean_rna(row.get("sequence", "")) != seq:
            continue
        if any(
            row.get(key) is not None
            for key in (
                "interface_passed",
                "interface_steric_clash",
                "interface_signed_score",
            )
        ):
            evidence.update(
                {
                    "known": True,
                    "clean": (
                        row.get("interface_passed") is True
                        and row.get("interface_steric_clash") is not True
                    ),
                    "steric_clash": row.get("interface_steric_clash") is True,
                    "score": row.get("interface_signed_score", evidence.get("score", 0.0)),
                    "source": "protein_agent",
                }
            )
    return evidence


def _known_interface_clash(state: LabState, seq: str) -> bool:
    evidence = _interface_evidence_for_sequence(state, seq)
    return bool(evidence.get("known") and evidence.get("steric_clash"))


def _clean_interface_anchor(state: LabState) -> str:
    candidates = [
        state.get("best_interface_clean_sequence"),
        state.get("_run_system_best_interface_sequence"),
    ]
    for seq in candidates:
        seq = clean_rna(seq or "")
        if seq and not _known_interface_clash(state, seq):
            return seq
    return ""


def _local_interface_variants(
    anchor: str,
    locked_positions: set[int],
    count: int = 3,
) -> list[str]:
    anchor = clean_rna(anchor)

    if not anchor:
        log.info(
            "[STRUCTURAL LOCAL DEBUG] no clean anchor available"
        )
        return []

    mutable_positions = [
        index
        for index in range(len(anchor))
        if index not in locked_positions
    ]

    log.info(
        "[STRUCTURAL LOCAL DEBUG] "
        "anchor=%s length=%d "
        "locked=%d mutable=%d requested=%d",
        anchor,
        len(anchor),
        len(locked_positions),
        len(mutable_positions),
        count,
    )

    if not mutable_positions:
        log.warning(
            "[STRUCTURAL LOCAL DEBUG] "
            "all anchor positions are locked; "
            "local mutation is impossible"
        )
        return []

    variants: list[str] = []
    attempts = 0

    while len(variants) < count and attempts < 50:
        attempts += 1

        n_mutations = 1 if len(variants) < 2 else 2

        raw_variant = mutate_sequence(
            anchor,
            num_mutations=n_mutations,
            locked_positions=locked_positions,
        )

        variant = clean_rna(raw_variant)

        distance = (
            hamming_distance(anchor, variant)
            if variant and len(variant) == len(anchor)
            else None
        )

        valid, reason = validate_sequence(variant)

        duplicate = (
            not variant
            or variant == anchor
            or variant in variants
        )

        log.info(
            "[STRUCTURAL LOCAL DEBUG] "
            "attempt=%d "
            "requested_mutations=%d "
            "raw=%s "
            "cleaned=%s "
            "distance=%s "
            "valid=%s "
            "reason=%s "
            "duplicate=%s",
            attempts,
            n_mutations,
            raw_variant,
            variant,
            distance,
            valid,
            reason,
            duplicate,
        )

        if not valid or duplicate:
            continue

        if distance is None or distance < 1 or distance > 2:
            log.warning(
                "[STRUCTURAL LOCAL DEBUG] "
                "rejecting unexpected "
                "Hamming distance=%s",
                distance,
            )
            continue

        variants.append(variant)

        log.info(
            "[STRUCTURAL LOCAL DEBUG] "
            "retained=%d variants=%s",
            len(variants),
            variants,
        )

    return variants


def _structural_selection_order(
    state: LabState,
    candidate_records: list[dict],
    clean_anchor: str,
) -> list[dict]:
    safe_records = [
        rec for rec in candidate_records
        if not _known_interface_clash(state, rec.get("sequence", ""))
    ]
    if not safe_records:
        safe_records = list(candidate_records)

    anchor_record = next(
        (rec for rec in safe_records if rec.get("sequence") == clean_anchor),
        None,
    )
    local_records = sorted(
        [
            rec for rec in safe_records
            if clean_anchor
            and rec.get("sequence") != clean_anchor
            and len(rec.get("sequence", "")) == len(clean_anchor)
            and hamming_distance(rec["sequence"], clean_anchor) <= 2
        ],
        key=lambda rec: (
            bool(rec["fold_quality"].get("passed")),
            float(rec.get("rank_score", 0.0)),
        ),
        reverse=True,
    )
    exploratory = sorted(
        [rec for rec in safe_records if rec is not anchor_record and rec not in local_records],
        key=lambda rec: (
            bool(rec["fold_quality"].get("passed")),
            float(rec.get("rank_score", 0.0)),
        ),
        reverse=True,
    )

    ordered = []
    if anchor_record and anchor_record["fold_quality"].get("passed"):
        ordered.append(anchor_record)
    if local_records:
        ordered.append(local_records[0])
    if exploratory:
        ordered.append(exploratory[0])
    for rec in safe_records:
        if rec not in ordered:
            ordered.append(rec)
    return ordered


def _build_downstream_shortlist(
    candidate_records: list[dict],
    limit: int = 5,
) -> list[str]:
    """
    Prefer one clean anchor, one fold-valid local variant, and one fold-valid
    exploratory candidate before filling remaining slots by structural rank.

    This ensures local interface variants reach downstream evaluation (MD/HDOCK)
    rather than being excluded by pure fold-rank sorting.
    """
    anchor = next(
        (
            row
            for row in candidate_records
            if row.get("interface_anchor") is True
            and row.get("fold_quality", {}).get("passed") is True
        ),
        None,
    )

    local = next(
        (
            row
            for row in candidate_records
            if row.get("interface_local_variant") is True
            and row.get("fold_quality", {}).get("passed") is True
        ),
        None,
    )

    exploratory = next(
        (
            row
            for row in candidate_records
            if row.get("interface_anchor") is not True
            and row.get("interface_local_variant") is not True
            and row.get("fold_quality", {}).get("passed") is True
        ),
        None,
    )

    selected: list[dict] = []

    for row in (anchor, local, exploratory):
        if row is not None and row not in selected:
            selected.append(row)

    for row in candidate_records:
        if row not in selected:
            selected.append(row)

    return [
        row["sequence"]
        for row in selected[: max(1, int(limit))]
    ]


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

        clean_anchor = _clean_interface_anchor(state)
        log.info(
            "[STRUCTURAL INTERFACE STATE] "
            "best_clean=%s run_system_best=%s clean_anchor=%s "
            "score_count=%d binding_rows=%d",
            state.get("best_interface_clean_sequence"),
            state.get("_run_system_best_interface_sequence"),
            clean_anchor or None,
            len(state.get("_run_system_sequence_scores", {}) or {}),
            len(state.get("binding_results", []) or []),
        )

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
        locked_positions = _build_locked_positions(
            conserved_regions,
            sequence_length=(
                len(clean_anchor)
                if clean_anchor
                else None
            ),
        )

        log.info(
            "[STRUCTURAL LOCAL DEBUG] "
            "anchor=%s anchor_len=%d "
            "conserved_regions=%s "
            "locked_count=%d "
            "locked_positions=%s",
            clean_anchor or None,
            len(clean_anchor) if clean_anchor else 0,
            conserved_regions,
            len(locked_positions),
            sorted(locked_positions),
        )

        exploratory_sequences = _normalise_candidate_sequences(
            raw_sequences=raw_sequences,
            prior_seqs=prior_seqs,
            locked_positions=locked_positions,
        )
        if clean_anchor:
            interface_variants = _local_interface_variants(
                anchor=clean_anchor,
                locked_positions=locked_positions,
                count=3,
            )
        else:
            interface_variants = []
            log.info(
                "[STRUCTURAL LOCAL DEBUG] No clean interface anchor available; "
                "using exploratory candidates only."
            )
        pre_dedupe_sequences = [
            *([clean_anchor] if clean_anchor else []),
            *interface_variants,
            *exploratory_sequences,
        ]
        log.info("[STRUCTURAL LOCAL DEBUG] pre_dedupe=%s", pre_dedupe_sequences)
        final_sequences = []
        seen_sequences = set()
        for raw_sequence in pre_dedupe_sequences:
            cleaned_sequence = clean_rna(raw_sequence)
            if not cleaned_sequence or cleaned_sequence in seen_sequences:
                continue
            if _known_interface_clash(state, cleaned_sequence):
                log.info(
                    "[STRUCTURAL FILTER] excluding known interface clash: %s",
                    cleaned_sequence,
                )
                continue
            seen_sequences.add(cleaned_sequence)
            final_sequences.append(cleaned_sequence)
            if len(final_sequences) >= 7:
                break
        log.info("[STRUCTURAL LOCAL DEBUG] post_dedupe=%s", final_sequences)
        if not final_sequences:
            raise RuntimeError(
                "No valid non-clashing RNA sequences generated by structural agent."
            )

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
            seq = clean_rna(seq)
            is_valid, validation_reason = validate_sequence(seq)
            if not seq or not is_valid:
                log.warning(
                    "Skipping invalid structural candidate seq=%s reason=%s",
                    seq, validation_reason,
                )
                continue
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
                    "interface_evidence": _interface_evidence_for_sequence(state, seq),
                    "interface_anchor": bool(clean_anchor and seq == clean_anchor),
                    "interface_local_variant": bool(
                        clean_anchor
                        and len(seq) == len(clean_anchor)
                        and 0 < hamming_distance(seq, clean_anchor) <= 2
                    ),
                }
            )

        if not candidate_records:
            raise RuntimeError("All structural candidates failed validation or folding.")
        candidate_records.sort(
            key=lambda r: (
                bool(r["fold_quality"].get("passed")),
                float(r.get("rank_score", 0.0)),
            ),
            reverse=True,
        )
        candidate_records = _structural_selection_order(
            state,
            candidate_records,
            clean_anchor,
        )

        for rec in candidate_records[:5]:
            fq = rec["fold_quality"]
            log.info(
                "    Candidate fold rank: seq=%s score=%.3f pass=%s pd=%s mfe_nt=%s anchor=%s local=%s interface=%s",
                rec["sequence"][:20],
                float(rec.get("rank_score", 0.0)),
                fq.get("passed"),
                fq.get("best_pair_density"),
                fq.get("best_mfe_per_nt"),
                rec.get("interface_anchor", False),
                rec.get("interface_local_variant", False),
                (rec.get("interface_evidence") or {}).get("score"),
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
                    f"reasons={fq.get('reasons', [])} | "
                    f"interface_anchor={rec.get('interface_anchor', False)} | "
                    f"interface_local={rec.get('interface_local_variant', False)} | "
                    f"interface_evidence={rec.get('interface_evidence', {})}"
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

        # Build downstream shortlist with anchor/local/exploratory priority
        current_structural_sequences = _build_downstream_shortlist(candidate_records, limit=5)

        # Log shortlist provenance (Fix 2.2)
        record_by_sequence = {
            row["sequence"]: row
            for row in candidate_records
        }
        for position, seq in enumerate(current_structural_sequences, start=1):
            row = record_by_sequence[seq]
            log.info(
                "[STRUCTURAL DOWNSTREAM] rank=%d seq=%s anchor=%s local=%s "
                "fold_pass=%s fold_score=%.4f",
                position,
                seq,
                row.get("interface_anchor", False),
                row.get("interface_local_variant", False),
                row.get("fold_quality", {}).get("passed"),
                float(row.get("rank_score", 0.0)),
            )

        result = {
            "structural_analysis": analysis,
            "structural_status": "complete",
            "structural_error": None,

            # Return 3–5 candidates to seed PI/NSGA-II.
            "designed_sequences": [rec["sequence"] for rec in candidate_records[:5]],

            # Current batch fields (non-reducer - reset each iteration)
            # Use downstream shortlist to ensure anchor/local/exploratory priority
            "current_structural_sequences": current_structural_sequences,
            "current_structural_candidates": [
                {
                    "sequence": rec["sequence"],
                    "fold_quality": rec["fold_quality"],
                    "rank_score": rec["rank_score"],
                    "interface_evidence": rec.get("interface_evidence", {}),
                    "interface_anchor": rec.get("interface_anchor", False),
                    "interface_local_variant": rec.get("interface_local_variant", False),
                }
                for rec in candidate_records
            ],
            "current_structural_iteration": state.get("iterations", 0) + 1,

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
                    "interface_evidence": rec.get("interface_evidence", {}),
                    "interface_anchor": rec.get("interface_anchor", False),
                    "interface_local_variant": rec.get("interface_local_variant", False),
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
                        "selection_policy": "clean_interface_then_local_variant_then_exploration",
                        "clean_interface_anchor": clean_anchor or None,
                        "selected_interface_evidence": selected.get("interface_evidence", {}),
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
        error_text = f"Structural analysis failed: {type(e).__name__}: {e}"
        result = {
            "structural_analysis": error_text,
            "structural_status": "failed",
            "structural_error": str(e),
            "designed_sequences": [],
            "structural_candidates": [],
            "current_structural_sequences": [],
            "current_structural_candidates": [],
            "current_structural_iteration": state.get("iterations", 0) + 1,
            "target_sequence": state.get("target_sequence"),
            "fold_quality": {},
            "fold_thresholds_passed": False,
            "fold_threshold_reasons": [f"structural_agent_error:{type(e).__name__}"],
            "stage_outputs": [
                record_stage_output(
                    state, "structural", error_text,
                    summary="Structural agent failed",
                    metadata={
                        "status": "failed",
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                )
            ],
            "conversation_history": [
                add_conversation_entry(state, "assistant", error_text, "structural")
            ],
        }
        try:
            save_checkpoint({**state, **result})
        except Exception as checkpoint_error:
            log.warning("Failed to checkpoint structural error state: %s", checkpoint_error)
        return result
