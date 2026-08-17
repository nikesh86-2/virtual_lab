from __future__ import annotations

import time

from VLAB2.orchestration.state_schema import LabState
from VLAB2.orchestration.utils.text_utils import truncate_str


def format_motif_location(motif: dict) -> str:
    """
    Format a motif location string with correct coordinate system.
    
    Renders as:
    - GAG@sequence[5,8) if mapped to sequence coordinates
    - CUG@alignment[40,43) if unmapped (alignment coordinates)
    - CUG@unmapped if invalid
    
    Args:
        motif: A motif dict with keys like motif/matched, mapping_status, 
               coordinate_system, sequence_start/end, msa_start/end
    
    Returns:
        Formatted motif location string
    """
    name = str(motif.get("motif") or motif.get("matched") or "unknown")
    status = motif.get("mapping_status")
    coordinate_system = motif.get("coordinate_system")

    if (
        status == "mapped"
        and coordinate_system == "sequence_zero_based_half_open"
    ):
        start = motif.get("sequence_start")
        end = motif.get("sequence_end")

        if start is not None and end is not None:
            return f"{name}@sequence[{int(start)},{int(end)})"

    msa_start = motif.get("msa_start")
    msa_end = motif.get("msa_end")

    if msa_start is not None and msa_end is not None:
        return f"{name}@alignment[{int(msa_start)},{int(msa_end)})"

    return f"{name}@unmapped"


def generate_final_report(state: LabState) -> str:
    """
    Generate Markdown summary report for a completed Virtual Lab run.
    """
    report = f"# Research Report: {state.get('research_topic', 'Unknown')}\n\n"

    report += (
        f"## Scientific Hypothesis\n"
        f"{truncate_str(state.get('hypothesis', ''), 600)}\n\n"
    )

    report += (
        f"## PI Optimisation Summary\n"
        f"{truncate_str(state.get('pi_summary', ''), 800)}\n\n"
    )

    # Phase 4.1: Add NSGA health reporting
    report += "## NSGA-II Health\n"
    report += f"- Status: {state.get('optimisation_status', 'unknown')}\n"
    report += f"- Valid candidates: {state.get('nsga_valid_candidate_count', 0)}\n"
    report += f"- Invalid candidates: {state.get('nsga_invalid_candidate_count', 0)}\n"
    report += f"- Penalty-only candidates: {state.get('nsga_penalty_only_count', 0)}\n"
    report += f"- Fallback population used: {'yes' if state.get('nsga_used_fallback_population', False) else 'no'}\n"
    if state.get('nsga_best_sequence'):
        report += f"- Best sequence: {state.get('nsga_best_sequence')}\n"
    if state.get('nsga_failure_reason_counts'):
        report += f"- Failure reasons: {state.get('nsga_failure_reason_counts')}\n"
    report += "\n"

    report += f"## Target Protein\n{state.get('target_pdb', 'Not selected')}\n\n"
    report += f"## Binding Units\n{state.get('binding_units', 'hdock_relative_score')}\n\n"

    # Phase 3.4: Report latest-batch status separately from best-validated status
    latest_status = state.get("current_target_status")
    best_status = state.get("best_validated_target_status")
    latest_interface_count = state.get("clean_interface_count", 0)
    best_interface_count = state.get("best_validated_clean_interface_count", 0)

    report += "## Target Protein Status\n"
    report += f"Latest docking batch status: {latest_status or 'unknown'}\n"
    report += f"Best validated target status: {best_status or 'unknown'}\n"
    report += f"Latest clean interface count: {latest_interface_count}\n"
    report += f"Best validated clean interface count: {best_interface_count}\n\n"

    # Phase 4.2: Add method-specific cache performance reporting
    report += "## Cache Performance\n"
    report += f"- RNA HDOCK: {state.get('rna_hdock_cache_hits', 0)} hit(s), {state.get('rna_hdock_cache_misses', 0)} miss(es)\n"
    report += f"- Peptide HDOCK: {state.get('peptide_hdock_cache_hits', 0)} hit(s), {state.get('peptide_hdock_cache_misses', 0)} miss(es)\n"
    report += f"- Vina: {state.get('inhibitor_vina_cache_hits', 0)} hit(s), {state.get('inhibitor_vina_cache_misses', 0)} miss(es)\n"
    report += f"- Ligand metadata: {state.get('pubchem_manifest_hits', 0)} manifest, {state.get('pubchem_cache_hits', 0)} cache, {state.get('pubchem_live_requests', 0)} live\n\n"

    # Phase 4.2: Add method-specific score units
    report += "## Score Units\n"
    report += "- RNA HDOCK scores: relative scores, more negative ranked better within this method\n"
    report += "- Peptide HDOCK scores: relative scores, more negative ranked better within this method\n"
    report += "- Vina scores: kcal/mol\n\n"

    if state.get("docking_summary_csv") or state.get("docking_summary_json"):
        report += "## Docking Output Files\n"
        report += f"- JSON: {state.get('docking_summary_json', '')}\n"
        report += f"- CSV: {state.get('docking_summary_csv', '')}\n"
        report += f"- Markdown: {state.get('docking_summary_md', '')}\n\n"

    if state.get("joint_physics_feedback"):
        jf = state.get("joint_physics_feedback", {})
        report += "## Joint Physics Feedback\n"
        report += (
            f"- Binding signal: {jf.get('binding_signal')}\n"
            f"- Stability signal: {jf.get('stability_signal')}\n"
            f"- Fluctuation signal: {jf.get('fluctuation_signal')}\n"
            f"- Best binding score: {jf.get('best_binding_score')}\n"
            f"- Best MD min energy: {jf.get('best_md_min_energy')}\n\n"
        )

    # Inhibitor screening results
    if state.get("inhibitor_enabled"):
        report += "## Inhibitor Screening\n"
        report += f"{state.get('inhibitor_summary', 'No inhibitor summary available.')}\n\n"

        small_mols = state.get("inhibitor_small_molecules", []) or []
        peptides = state.get("inhibitor_peptides", []) or []

        if small_mols:
            report += "### Small Molecules\n"
            report += "| Name | Binding Energy (kcal/mol) | Valid |\n"
            report += "| :--- | :----------------------- | :---- |\n"
            for sm in small_mols:
                report += (
                    f"| {sm.get('name', 'unknown')} "
                    f"| {sm.get('binding_energy', 'N/A')} "
                    f"| {sm.get('valid', False)} |\n"
                )
            report += "\n"

        if peptides:
            report += "### Peptides\n"
            report += "| Sequence | Dock Score | Valid |\n"
            report += "| :------- | :-------- | :---- |\n"
            for pep in peptides:
                seq = pep.get('sequence', 'unknown')[:20]
                report += (
                    f"| {seq}... "
                    f"| {pep.get('score', 'N/A')} "
                    f"| {pep.get('valid', False)} |\n"
                )
            report += "\n"

        overlap = state.get("inhibitor_binding_site_overlap", 0.0)
        if overlap > 0:
            report += f"**Binding-site overlap with RNA interface:** {overlap:.1f}%\n\n"

        if state.get("inhibitor_analysis"):
            report += f"### Inhibitor Analysis\n{state.get('inhibitor_analysis')}\n\n"

    report += "## Iteration History\n"
    report += "| Iter | Target | Best Binding Score | Score Range | n_valid | Critique |\n"
    report += "| :--- | :----- | :----------------- | :---------- | :------ | :------- |\n"

    for item in state.get("results_log", []) or []:
        report += (
            f"| {item.get('iteration')} "
            f"| {item.get('target_pdb')} "
            f"| {item.get('best_binding_score', item.get('best_dg', 'N/A'))} "
            f"| {item.get('score_range', item.get('energy_range', 'N/A'))} "
            f"| {item.get('n_valid', 'N/A')} "
            f"| {truncate_str(item.get('critique', ''), 80)} |\n"
        )

    report += "\n## Agent Outputs\n\n"

    for entry in state.get("stage_outputs", []) or []:
        report += f"### {str(entry.get('agent', 'unknown')).upper()}\n"
        report += f"{entry.get('summary', '')}\n\n"

    return report


def print_user_friendly_summary(result: dict) -> None:
    """
    Print concise terminal summary and save Markdown report.
    """
    print("\n" + "=" * 70)
    print(" VIRTUAL LAB: RUN COMPLETE")
    print("=" * 70)

    print("\n[SCIENTIFIC HYPOTHESIS]")
    print(f"  {truncate_str(result.get('hypothesis', ''), 500)}")

    print("\n[PI OPTIMISATION SUMMARY]")
    print(f"  {truncate_str(result.get('pi_summary', ''), 700)}")

    print("\n[RESEARCH TOPIC]")
    print(f"  {result.get('research_topic')}")
    print(f"  {result.get('topic_description')}")

    print("\n[TARGET PROTEIN]")
    print(f"  {result.get('target_pdb', 'Not selected')}")

    print("\n[BINDING UNITS]")
    print(f"  {result.get('binding_units', 'hdock_relative_score')}")

    print("\n[AGENT TAKEAWAYS]")

    for entry in result.get("stage_outputs", []) or []:
        print(
            f"  {str(entry.get('agent', 'unknown')).upper():<12} : "
            f"{entry.get('summary', '')}"
        )

    print("\n" + "=" * 70)

    report = generate_final_report(result)
    report_file = f"summary_{int(time.time())}.md"

    try:
        with open(report_file, "w") as f:
            f.write(report)

        print(f"Final report saved to: {report_file}")

    except Exception as e:
        print(f"Failed to save report file: {e}")