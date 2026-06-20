import os
import re
from orchestration.state_schema import LabState

def parse_pdb_candidates(text: str) -> list[str]:
    candidates = []
    for line in text.splitlines():
        line = line.strip().upper()
        m = re.match(r"^[A-Z0-9]{4}$", line)
        if m:
            candidates.append(line)
    return list(dict.fromkeys(candidates))

def get_candidate_pdbs(state: LabState) -> list[str]:
    candidates = []

    if state.get("target_pdb"):
        candidates.append(state["target_pdb"])

    if state.get("target_pdb_candidates"):
        candidates.extend(state["target_pdb_candidates"])

    # Optional env-configured emergency fallbacks
    extra = os.getenv("VLAB_FALLBACK_PDBS", "")
    if extra.strip():
        candidates.extend([x.strip().upper() for x in extra.split(",") if x.strip()])

    failed = set(state.get("failed_target_pdbs", []))
    candidates = [c for c in dict.fromkeys(candidates) if c not in failed]

    return candidates

def select_valid_target_pdb(
    state: LabState,
    pw,
    sequences: list[str],
) -> tuple[str | None, list[dict], list[str]]:
    failed = list(state.get("failed_target_pdbs", []))
    candidates = get_candidate_pdbs(state)

    for pdb_id in candidates:
        results = pw.evaluate_sequences(pdb_id, sequences) or []
        valid = [r for r in results if r.get("valid")]
        if valid:
            return pdb_id, valid, failed
        failed.append(pdb_id)

    return None, [], failed
