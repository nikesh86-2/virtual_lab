# VLAB2 Run 7110842: Deterministic Bug Fix Plan

## Scope

This plan addresses deterministic defects observed in SLURM job `7110842`, run on 14 August 2026. It deliberately excludes changing the Qwen model because the run did not show failed chat requests, malformed model responses, or agent-cycle termination attributable to the language model.

The main objectives are:

1. separate the **latest evaluated batch** from the **best validated target state**;
2. correct motif alignment-to-sequence interval mapping;
3. correct motif coordinate formatting and provenance;
4. prevent incomplete Vina pocket coverage from being treated as a valid full-pocket screen;
5. pass the LLM client into peptide design only when generative peptide design is intended;
6. make final reporting internally consistent and scientifically auditable.

---

# Executive priority order

## P0: state correctness

1. Introduce separate latest-batch and best-validated target records.
2. Make target-state promotion monotonic and quality-aware.
3. Stop later exploratory batches from overwriting the final validated headline.
4. Update final reporting, convergence, routing, and training export to use the correct state view.

## P0: coordinate correctness

5. Fix alignment-to-sequence half-open interval mapping.
6. Enforce motif-length preservation for mapped intervals.
7. Reject invalid mapped intervals rather than shortening or clamping them.
8. Fix malformed motif output such as `GAG@alignment2,5)`.

## P0: docking-box correctness

9. Revalidate the final Vina box after recentering.
10. Require complete pocket-point coverage for a normal full-pocket screen.
11. Split oversized pockets or mark results explicitly exploratory.

## P1: peptide design wiring

12. Pass a model client into peptide design when enabled.
13. Keep deterministic known-peptide screening as the default production mode.

## P1: regression coverage

14. Add exact run-derived regression fixtures for target preservation, motif mappings, coordinate rendering, and Vina containment.

---

# 1. Separate latest-batch state from best validated target state

## Observed defect

During run `7110842`, `8K75` passed interface validation in earlier batches:

```text
Batch example 1:
- docking-valid rows: 3
- clean interfaces: 3
- steric clashes: 0
- result: accepted_target

Batch example 2:
- docking-valid rows: 3
- clean interfaces: 2
- steric clashes: 1
- result: accepted_target
```

A later exploratory batch produced:

```text
- docking-valid rows: 3
- clean interfaces: 1
- steric clashes: 2
- result: partial_success_target
```

The final PI headline then reported:

```text
Target PDB: 8K75 (partial success)
Target status: partial_success_target
```

This loses the strongest validated conclusion even though the same target had already passed the configured interface threshold.

## Required semantics

The application must maintain two independent views:

### Latest batch

The result of the most recently completed target-RNA docking and interface-analysis batch. This is diagnostic and may improve or degrade.

### Best validated state

The strongest accepted evidence observed during the current run, and optionally across remembered prior runs if that is the configured policy. This state must not be degraded by a weaker later exploratory batch.

The final headline should use the best validated state. The latest batch should appear in a separate iteration-diagnostics section.

## Files to change

```text
VLAB2/orchestration/state_schema.py
VLAB2/orchestration/state_factory.py
VLAB2/orchestration/agents/protein_agent.py
VLAB2/orchestration/agents/pi_agent.py
VLAB2/orchestration/agents/inhibitor_agent.py
VLAB2/orchestration/agents/skeptic_agent.py
VLAB2/orchestration/routing.py
VLAB2/orchestration/postrun.py
VLAB2/orchestration/reporting.py
VLAB2/orchestration/failure_memory.py
VLAB2/core/training_data_collector.py
```

Tests:

```text
VLAB2/tests/orchestration/agents/test_best_target_preservation.py
VLAB2/tests/orchestration/agents/test_protein_target_state_views.py
VLAB2/tests/orchestration/test_final_reporting.py
VLAB2/tests/orchestration/test_routing_target_state.py
VLAB2/tests/orchestration/test_training_export_target_state.py
```

---

## 1.1 Add structured target-evaluation records

Prefer a dedicated type over a growing set of loosely related scalar fields.

In `orchestration/state_schema.py`:

```python
from typing import Any, NotRequired, TypedDict


class TargetEvaluationRecord(TypedDict):
    target_pdb: str
    status: str
    status_reason: str
    iteration: int
    batch_id: str
    evaluated_at: str

    docking_valid_count: int
    docking_required_count: int
    clean_interface_count: int
    clean_interface_required_count: int
    steric_clash_count: int

    best_hdock_relative_score: float | None
    hdock_score_spread: float | None
    interface_quality_score: float | None
    aggregate_quality_score: float

    accepted: bool
    interface_validated: bool
    exploratory: bool

    binding_result_ids: list[str]
    clean_sequence_ids: list[str]
    clash_sequence_ids: list[str]

    docking_summary_json: str | None
    docking_summary_csv: str | None
    docking_summary_md: str | None

    metadata: dict[str, Any]
```

Add state fields:

```python
latest_target_evaluation: TargetEvaluationRecord | None
best_validated_target_evaluation: TargetEvaluationRecord | None
target_evaluation_history: list[TargetEvaluationRecord]
```

Keep compatibility fields temporarily:

```python
# Compatibility aliases. These must be derived, not independently mutated.
target_pdb: str | None
target_status: str | None
target_status_reason: str | None
best_validated_target_pdb: str | None
best_validated_target_status: str | None
best_validated_target_status_reason: str | None
```

## 1.2 Initialise both views explicitly

In `orchestration/state_factory.py`:

```python
"latest_target_evaluation": None,
"best_validated_target_evaluation": None,
"target_evaluation_history": [],

"target_pdb": None,
"target_status": None,
"target_status_reason": None,
"best_validated_target_pdb": None,
"best_validated_target_status": None,
"best_validated_target_status_reason": None,
```

Do not initialise `target_status` to a success-like value.

---

## 1.3 Define status strength centrally

Create:

```text
VLAB2/orchestration/utils/target_state_utils.py
```

```python
from __future__ import annotations

from typing import Any


TARGET_STATUS_RANK = {
    "failed_target": 0,
    "unvalidated_target": 1,
    "partial_success_target": 2,
    "accepted_target": 3,
}


def target_status_rank(status: str | None) -> int:
    return TARGET_STATUS_RANK.get(str(status or ""), -1)
```

Status alone is insufficient to compare two accepted records, so add a deterministic quality key:

```python
def target_quality_key(record: dict[str, Any] | None) -> tuple:
    if not record:
        return (-1, -1, -1, float("-inf"), float("-inf"), 0)

    status_rank = target_status_rank(record.get("status"))
    clean_count = int(record.get("clean_interface_count", 0) or 0)
    clash_count = int(record.get("steric_clash_count", 0) or 0)
    quality = float(record.get("aggregate_quality_score", 0.0) or 0.0)

    best_hdock = record.get("best_hdock_relative_score")
    hdock_rank = (
        -float(best_hdock)
        if best_hdock is not None
        else float("-inf")
    )

    iteration = int(record.get("iteration", 0) or 0)

    return (
        status_rank,
        clean_count,
        -clash_count,
        quality,
        hdock_rank,
        -iteration,
    )
```

The final `-iteration` tie-breaker preserves the earlier record when scientific quality is exactly equal. This avoids changing the declared best state solely because an equivalent batch ran later.

## 1.4 Define target promotion rules

```python
def is_validated_target_record(record: dict[str, Any] | None) -> bool:
    if not record:
        return False

    return bool(
        record.get("accepted")
        and record.get("interface_validated")
        and record.get("status") == "accepted_target"
        and int(record.get("docking_valid_count", 0) or 0)
            >= int(record.get("docking_required_count", 0) or 0)
        and int(record.get("clean_interface_count", 0) or 0)
            >= int(record.get("clean_interface_required_count", 0) or 0)
    )


def should_promote_best_target(
    candidate: dict[str, Any] | None,
    incumbent: dict[str, Any] | None,
) -> bool:
    if not is_validated_target_record(candidate):
        return False

    if not is_validated_target_record(incumbent):
        return True

    return target_quality_key(candidate) > target_quality_key(incumbent)
```

A `partial_success_target` record can never demote an accepted record.

---

## 1.5 Build one record per completed batch

In `orchestration/agents/protein_agent.py`, after docking and interface analysis are complete, create a single record.

```python
from datetime import datetime, timezone
import hashlib
import json


def _make_batch_id(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
```

Construct the record from final batch values only:

```python
batch_identity = {
    "target_pdb": target_pdb,
    "iteration": int(state.get("iterations", 0) or 0),
    "sequence_ids": sorted(result_ids),
    "scores": sorted(hdock_scores),
}

record: TargetEvaluationRecord = {
    "target_pdb": target_pdb,
    "status": target_status,
    "status_reason": target_status_reason,
    "iteration": int(state.get("iterations", 0) or 0),
    "batch_id": _make_batch_id(batch_identity),
    "evaluated_at": datetime.now(timezone.utc).isoformat(),

    "docking_valid_count": dock_valid_count,
    "docking_required_count": min_valid_required,
    "clean_interface_count": interface_clean_count,
    "clean_interface_required_count": min_clean_required,
    "steric_clash_count": steric_clash_count,

    "best_hdock_relative_score": best_hdock_score,
    "hdock_score_spread": score_spread,
    "interface_quality_score": interface_quality_score,
    "aggregate_quality_score": aggregate_quality_score,

    "accepted": target_status == "accepted_target",
    "interface_validated": interface_clean_count >= min_clean_required,
    "exploratory": bool(exploratory_batch),

    "binding_result_ids": result_ids,
    "clean_sequence_ids": clean_sequence_ids,
    "clash_sequence_ids": clash_sequence_ids,

    "docking_summary_json": docking_summary_json,
    "docking_summary_csv": docking_summary_csv,
    "docking_summary_md": docking_summary_md,

    "metadata": {
        "binding_units": "hdock_relative_score",
        "backend": "hdock",
        "receptor_atom_count": receptor_atom_count,
    },
}
```

If `aggregate_quality_score` does not already exist, calculate it deterministically:

```python
def calculate_target_quality_score(
    clean_count: int,
    clash_count: int,
    valid_count: int,
    interface_quality_scores: list[float],
) -> float:
    mean_interface = (
        sum(interface_quality_scores) / len(interface_quality_scores)
        if interface_quality_scores
        else 0.0
    )

    return (
        2.0 * float(clean_count)
        - 1.0 * float(clash_count)
        + 0.25 * float(valid_count)
        + mean_interface
    )
```

Do not include Vina scores in this target quality value.

---

## 1.6 Update latest and best views atomically

Still in `protein_agent.py`:

```python
from orchestration.utils.target_state_utils import (
    should_promote_best_target,
)


history = list(state.get("target_evaluation_history") or [])
history.append(record)

best = state.get("best_validated_target_evaluation")

update = {
    "latest_target_evaluation": record,
    "target_evaluation_history": history,

    # Legacy latest-batch aliases
    "target_pdb": record["target_pdb"],
    "target_status": record["status"],
    "target_status_reason": record["status_reason"],
}

if should_promote_best_target(record, best):
    update.update(
        {
            "best_validated_target_evaluation": record,
            "best_validated_target_pdb": record["target_pdb"],
            "best_validated_target_status": record["status"],
            "best_validated_target_status_reason": record[
                "status_reason"
            ],
        }
    )
```

Do not write best-valid aliases in any other branch.

## 1.7 Protect binding rows associated with the best state

The best target record must point to immutable result identifiers. Do not rely on mutable global `binding_results` ordering.

Add a stable identifier to every RNA docking row:

```python
def make_binding_result_id(result: dict) -> str:
    identity = {
        "target_pdb": result.get("target_pdb"),
        "sequence": result.get("sequence"),
        "cache_key": result.get("cache_key"),
        "hdock_run_id": result.get("hdock_run_id"),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
```

Store `binding_result_id` on each row before appending it to state.

---

## 1.8 Correct downstream consumers

### Final report

Use the best validated record for the main target headline:

```python
headline_target = (
    state.get("best_validated_target_evaluation")
    or state.get("latest_target_evaluation")
)
```

### Inhibitor screening

Use the best validated target and its accepted interface rows:

```python
screening_target = (
    state.get("best_validated_target_evaluation")
    or state.get("latest_target_evaluation")
)
```

Then resolve only binding rows named in:

```python
screening_target["binding_result_ids"]
```

Do not use whichever interface contacts happen to be last in global state.

### Convergence

Convergence should use:

- best validated target for `accepted target` requirements;
- latest batch for evidence that the current iteration improved;
- current conservation and current fold state for iteration-level convergence.

Example:

```python
best_target_ok = is_validated_target_record(
    state.get("best_validated_target_evaluation")
)

latest = state.get("latest_target_evaluation") or {}
latest_batch_clean = int(latest.get("clean_interface_count", 0) or 0)
```

### Failure memory

Record the latest weaker batch as an exploratory failure signal, but do not demote remembered target status.

### Training export

Tag examples explicitly:

```python
{
    "target_state_view": "latest_batch",
    "target_batch_id": latest_batch_id,
}
```

or:

```python
{
    "target_state_view": "best_validated",
    "target_batch_id": best_batch_id,
}
```

This prevents contradictory supervision labels for the same target.

---

## 1.9 Final report design

In `orchestration/postrun.py` and `orchestration/reporting.py`, render two sections.

### Headline

```text
Target conclusion
- Best validated target: 8K75
- Best validated status: accepted_target
- Reason: interface_validated
- Valid dockings: 3
- Clean interfaces: 3
- Steric clashes: 0
- Validated in iteration: 0
```

### Latest iteration diagnostics

```text
Latest target-evaluation batch
- Target: 8K75
- Latest status: partial_success_target
- Reason: hdock_passed_interface_partially_failed
- Valid dockings: 3
- Clean interfaces: 1
- Steric clashes: 2
- Interpretation: latest exploratory batch was weaker and did not replace the best validated state
```

### If no target was ever accepted

```text
Target conclusion
- Best validated target: none
- Latest evaluated target: 8K75
- Latest status: partial_success_target
```

Never label a target `accepted_target` based only on a remembered PDB ID. The best record itself must pass the validation predicate.

---

## 1.10 Exact regression test for run 7110842

In `tests/orchestration/agents/test_best_target_preservation.py`:

```python
def test_later_partial_batch_does_not_demote_best_accepted_target():
    accepted = make_target_record(
        target_pdb="8K75",
        status="accepted_target",
        iteration=0,
        clean=3,
        clashes=0,
        valid=3,
        required=2,
        min_clean=2,
        quality=7.8,
    )

    partial = make_target_record(
        target_pdb="8K75",
        status="partial_success_target",
        iteration=2,
        clean=1,
        clashes=2,
        valid=3,
        required=2,
        min_clean=2,
        quality=1.1,
    )

    state = initial_state()
    state = apply_target_evaluation(state, accepted)
    state = apply_target_evaluation(state, partial)

    assert state["latest_target_evaluation"] == partial
    assert state["best_validated_target_evaluation"] == accepted
    assert state["target_status"] == "partial_success_target"
    assert state["best_validated_target_status"] == "accepted_target"
```

Add report assertions:

```python
def test_final_report_separates_best_and_latest_target_states():
    report = build_final_report(state_with_accepted_then_partial())

    assert "Best validated status: accepted_target" in report
    assert "Latest status: partial_success_target" in report
    assert "did not replace the best validated state" in report
```

## Acceptance criteria

- A later partial batch never overwrites an accepted best record.
- A stronger later accepted batch can replace a weaker accepted best record.
- Final headline and inhibitor screening use the same best record.
- Latest diagnostics remain visible.
- Training rows state which target-state view generated them.
- Legacy aliases are derived from structured records and cannot drift independently.

---

# 2. Fix motif alignment-to-sequence mapping

## Observed defect

The run logged mappings including:

```text
motif=GAG  msa=[5,8) sequence=[5,7)
motif=RGAG msa=[4,8) sequence=[4,7)
motif=RAAG msa=[2,6) sequence=[2,5)
```

These mappings shorten every motif by one nucleotide:

```text
GAG:  alignment length 3, sequence length 2
RGAG: alignment length 4, sequence length 3
RAAG: alignment length 4, sequence length 3
```

## Likely root cause

The mapping function probably converts the exclusive MSA end using the same inclusive-column mapping used for the start, then fails to add one to form a half-open sequence end.

For half-open intervals `[msa_start, msa_end)`, the correct sequence interval is determined by counting non-gap reference residues before each boundary, not by mapping the final included column and treating it as exclusive.

## Files to change

```text
VLAB2/core/bioinfo_wrapper.py
VLAB2/orchestration/utils/sequence_utils.py
VLAB2/orchestration/agents/bioinfo_agent.py
```

Tests:

```text
VLAB2/tests/core/test_conservation_coordinate_mapping.py
VLAB2/tests/core/test_motif_coordinate_mapping.py
```

---

## 2.1 Implement boundary-based half-open mapping

In `orchestration/utils/sequence_utils.py`:

```python
def alignment_boundary_to_sequence_offset(
    aligned_reference: str,
    boundary: int,
) -> int | None:
    if boundary < 0 or boundary > len(aligned_reference):
        return None

    return sum(
        1
        for character in aligned_reference[:boundary]
        if character not in {"-", "."}
    )


def map_alignment_interval_to_sequence(
    aligned_reference: str,
    msa_start: int,
    msa_end: int,
) -> tuple[int, int] | None:
    if not isinstance(msa_start, int) or not isinstance(msa_end, int):
        return None

    if not 0 <= msa_start < msa_end <= len(aligned_reference):
        return None

    sequence_start = alignment_boundary_to_sequence_offset(
        aligned_reference,
        msa_start,
    )
    sequence_end = alignment_boundary_to_sequence_offset(
        aligned_reference,
        msa_end,
    )

    if sequence_start is None or sequence_end is None:
        return None

    if sequence_start >= sequence_end:
        return None

    return sequence_start, sequence_end
```

This handles the exclusive end correctly.

## 2.2 Add motif-specific validation

A motif mapping needs stricter validation than a generic conserved region.

```python
def validate_motif_mapping(
    matched_motif: str,
    sequence_start: int | None,
    sequence_end: int | None,
    reference_sequence: str,
) -> tuple[bool, str | None]:
    if sequence_start is None or sequence_end is None:
        return False, "missing_sequence_interval"

    if not 0 <= sequence_start < sequence_end <= len(reference_sequence):
        return False, "interval_out_of_bounds"

    expected_length = len(matched_motif)
    observed_length = sequence_end - sequence_start

    if observed_length != expected_length:
        return False, (
            "mapped_length_mismatch:"
            f"expected={expected_length},observed={observed_length}"
        )

    mapped_sequence = reference_sequence[sequence_start:sequence_end]

    if mapped_sequence != matched_motif:
        return False, (
            "mapped_sequence_mismatch:"
            f"expected={matched_motif},observed={mapped_sequence}"
        )

    return True, None
```

If the motif contains an ambiguity code such as `R`, validate against the concrete `matched` value, not the motif pattern.

## 2.3 Reject invalid mappings

In `core/bioinfo_wrapper.py`:

```python
mapped = map_alignment_interval_to_sequence(
    aligned_reference=aligned_reference,
    msa_start=msa_start,
    msa_end=msa_end,
)

if mapped is None:
    sequence_start = None
    sequence_end = None
    mapping_status = "unmapped"
    mapping_error = "alignment_interval_unmappable"
else:
    sequence_start, sequence_end = mapped
    mapping_valid, mapping_error = validate_motif_mapping(
        matched_motif=matched,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        reference_sequence=reference_sequence,
    )

    if mapping_valid:
        mapping_status = "mapped"
    else:
        sequence_start = None
        sequence_end = None
        mapping_status = "unmapped"
```

Store:

```python
"mapping_error": mapping_error,
```

Never repair the interval by clamping or subtracting one.

## 2.4 Correct mapping logs

```python
log.info(
    "[MOTIF MAPPING] motif=%s matched=%s msa=[%d,%d) "
    "sequence=%s status=%s valid=%s error=%s",
    motif,
    matched,
    msa_start,
    msa_end,
    (
        f"[{sequence_start},{sequence_end})"
        if sequence_start is not None and sequence_end is not None
        else "unmapped"
    ),
    mapping_status,
    mapping_status == "mapped",
    mapping_error,
)
```

## 2.5 Add exact tests

```python
def test_ungapped_half_open_motif_mapping_preserves_length():
    aligned = "AACUGAGUCC"

    assert map_alignment_interval_to_sequence(
        aligned,
        4,
        7,
    ) == (4, 7)


def test_gapped_half_open_mapping_counts_boundaries():
    aligned = "AA-CUGA-GUCC"

    mapped = map_alignment_interval_to_sequence(
        aligned,
        3,
        7,
    )

    assert mapped == (2, 6)


def test_gag_mapping_does_not_shorten_to_two_bases():
    matched = "GAG"
    start, end = map_alignment_interval_to_sequence(
        "AAAAAGAGCCCC",
        5,
        8,
    )

    assert end - start == 3
    assert "AAAAAGAGCCCC"[start:end] == matched


def test_invalid_motif_mapping_becomes_unmapped():
    valid, reason = validate_motif_mapping(
        matched_motif="RGAG".replace("R", "A"),
        sequence_start=4,
        sequence_end=7,
        reference_sequence="CCCCAGAGCCCC",
    )

    assert valid is False
    assert reason.startswith("mapped_length_mismatch")
```

Use the concrete matched sequence in real tests, for example `AGAG`, not `RGAG`.

## Acceptance criteria

- Every mapped motif preserves `sequence_end - sequence_start == len(matched)`.
- Every mapped slice equals the concrete matched sequence.
- Invalid mapping becomes `unmapped` with a reason.
- No mapper mixes inclusive and exclusive endpoint conventions.

---

# 3. Fix motif coordinate formatting and provenance

## Observed defect

The final report contains malformed strings such as:

```text
GAG@alignment2,5)
AAG@alignment45,48)
```

The opening bracket is missing. The selected coordinate system can also disagree with the mapping record.

## Files to change

```text
VLAB2/orchestration/reporting.py
VLAB2/orchestration/postrun.py
VLAB2/orchestration/agents/pi_agent.py
```

Tests:

```text
VLAB2/tests/orchestration/test_final_reporting.py
VLAB2/tests/orchestration/test_motif_formatting.py
```

## 3.1 Use one canonical formatter

In `orchestration/reporting.py`:

```python
def format_motif_location(motif: dict) -> str:
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
```

Do not build motif locations with ad hoc f-strings anywhere else.

## 3.2 Preserve source coordinates instead of flattened strings

PI state should hold motif dictionaries, not preformatted labels.

Correct:

```python
selected_motifs: list[dict]
```

Avoid:

```python
selected_motifs: list[str]
```

If compatibility strings are required, derive them at the report boundary.

## 3.3 Add a coordinate legend

```text
Coordinate convention: zero-based, half-open intervals [start,end).
```

## 3.4 Tests

```python
def test_format_mapped_motif_uses_sequence_coordinates():
    motif = {
        "motif": "GAG",
        "mapping_status": "mapped",
        "coordinate_system": "sequence_zero_based_half_open",
        "sequence_start": 5,
        "sequence_end": 8,
        "msa_start": 5,
        "msa_end": 8,
    }

    assert format_motif_location(motif) == "GAG@sequence[5,8)"


def test_format_unmapped_motif_uses_alignment_coordinates():
    motif = {
        "motif": "CUG",
        "mapping_status": "unmapped",
        "coordinate_system": "alignment_zero_based_half_open",
        "msa_start": 40,
        "msa_end": 43,
    }

    assert format_motif_location(motif) == "CUG@alignment[40,43)"
```

## Acceptance criteria

- No output contains `@alignment2,5)` or equivalent malformed text.
- Mapped motifs prefer validated sequence coordinates.
- Unmapped motifs retain alignment coordinates.
- The interval convention is declared once in the report.

---

# 4. Make Vina pocket-box handling scientifically valid

## Observed defect

Run `7110842` showed at least two incomplete boxes:

```text
Raw z extent: 40.714 Å
Capped z size: 32 Å
Excluded points: 9 of 40
```

and later:

```text
Excluded points: 2 of 124
```

The workflow continued after recentering another capped box. Recentring cannot guarantee coverage when the geometric extent exceeds the maximum allowed axis.

The logs also call atomic coordinates “pocket residues,” causing `4 residues` to become `40 pocket residues`.

## Files to change

```text
VLAB2/core/rna_binding_pocket.py
VLAB2/orchestration/agents/inhibitor_agent.py
VLAB2/core/inhibitor_docking.py
VLAB2/core/vina_wrapper.py
VLAB2/orchestration/state_schema.py
VLAB2/orchestration/state_factory.py
VLAB2/orchestration/postrun.py
```

Tests:

```text
VLAB2/tests/core/test_vina_box_constraints.py
VLAB2/tests/orchestration/agents/test_inhibitor_box_policy.py
```

---

## 4.1 Compute the true enclosing box

```python
import numpy as np


def minimum_enclosing_box(
    points,
    padding: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    xyz = np.asarray(points, dtype=float)

    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError("Pocket points must be a non-empty N x 3 array")

    lower = xyz.min(axis=0)
    upper = xyz.max(axis=0)
    center = (lower + upper) / 2.0
    size = upper - lower + 2.0 * float(padding)

    return tuple(center.tolist()), tuple(size.tolist())
```

Do not call a clipped box the minimum-required box.

## 4.2 Revalidate every final box

```python
def evaluate_box_coverage(points, center, size, tolerance=1e-6) -> dict:
    xyz = np.asarray(points, dtype=float)
    center_arr = np.asarray(center, dtype=float)
    half_size = np.asarray(size, dtype=float) / 2.0

    inside = np.all(
        np.abs(xyz - center_arr) <= half_size + tolerance,
        axis=1,
    )

    outside_indices = np.flatnonzero(~inside).tolist()
    covered = int(inside.sum())
    total = int(len(inside))

    return {
        "contains_all": covered == total,
        "coverage_fraction": covered / total if total else 0.0,
        "covered_point_count": covered,
        "outside_point_count": len(outside_indices),
        "outside_point_indices": outside_indices,
        "total_point_count": total,
    }
```

Call this after the final centre and final size are selected.

## 4.3 Add explicit policy

```bash
export VLAB_VINA_POCKET_POLICY=split
```

Allowed values:

```text
strict
split
exploratory_partial
```

### Strict

Skip Vina if one valid box cannot cover all required points.

### Split

Recursively split the pocket along its longest axis until each sub-pocket fits and has 100% local coverage.

### Exploratory partial

Allow the capped box, but mark every score as partial-pocket exploratory evidence.

Production default should be `split` or `strict`, not silent partial coverage.

## 4.4 Split oversized pockets deterministically

```python
def split_points_on_longest_axis(points):
    xyz = np.asarray(points, dtype=float)
    extents = xyz.max(axis=0) - xyz.min(axis=0)
    axis = int(np.argmax(extents))
    midpoint = float(np.median(xyz[:, axis]))

    first = xyz[xyz[:, axis] <= midpoint]
    second = xyz[xyz[:, axis] > midpoint]

    if len(first) == 0 or len(second) == 0:
        raise ValueError("Unable to split oversized pocket")

    return first, second
```

Continue until:

```text
all axis sizes <= configured maxima
volume <= configured maximum
coverage_fraction == 1.0
```

## 4.5 Correct terminology

Use:

```text
pocket_residue_count
pocket_point_count
outside_point_count
```

Log:

```python
log.warning(
    "Final Vina box covers %d/%d pocket points derived from %d residues",
    covered_point_count,
    pocket_point_count,
    pocket_residue_count,
)
```

## 4.6 Add score provenance

Every Vina result must carry:

```python
{
    "pocket_id": pocket_id,
    "pocket_policy": policy,
    "pocket_coverage_complete": coverage["contains_all"],
    "pocket_coverage_fraction": coverage["coverage_fraction"],
    "docking_interpretation": (
        "full_pocket"
        if coverage["contains_all"]
        else "exploratory_partial_pocket"
    ),
}
```

Include `pocket_id`, point-set hash, box centre, and box size in the Vina cache key.

## 4.7 Tests from the run geometry

```python
def test_40_point_pocket_with_40_714_extent_is_not_valid_in_32_box():
    points = make_points_with_z_extent(40.714, count=40)
    center, size = capped_box(points, max_axis=32.0)
    coverage = evaluate_box_coverage(points, center, size)

    assert coverage["contains_all"] is False


def test_recentered_box_is_revalidated():
    ...


def test_split_policy_covers_all_points():
    ...


def test_partial_policy_marks_results_exploratory():
    ...
```

## Acceptance criteria

- No full-pocket screen proceeds with incomplete coverage.
- A recentered box is never assumed valid without rechecking.
- Split sub-pockets each have 100% point coverage.
- Partial results are clearly labelled and excluded from validated full-pocket claims.
- Logs distinguish residues from atomic or sampled points.

---

# 5. Wire peptide design intentionally

## Observed warning

```text
No LLM provided for peptide design, using conservative defaults
```

This is not a Qwen failure. The peptide preparation path receives no model client.

## Recommended policy

Keep two distinct modes:

```text
known_panel
llm_design
```

Default production inhibitor validation to `known_panel` for reproducibility. Enable `llm_design` only for explicit exploratory peptide generation.

## Files to change

```text
VLAB2/orchestration/agents/inhibitor_agent.py
VLAB2/core/peptide_prep.py
VLAB2/orchestration/config.py
```

Tests:

```text
VLAB2/tests/core/test_peptide_design_mode.py
VLAB2/tests/orchestration/agents/test_inhibitor_peptide_mode.py
```

## 5.1 Configuration

```bash
export VLAB_PEPTIDE_MODE=known_panel
```

## 5.2 Pass the existing client only in design mode

```python
peptide_mode = os.getenv(
    "VLAB_PEPTIDE_MODE",
    "known_panel",
).strip().lower()

if peptide_mode == "llm_design":
    llm = get_llm_client(
        temperature=0.25,
        max_tokens=1200,
    )
else:
    llm = None

peptide_candidates = prepare_peptide_candidates(
    target_context=target_context,
    known_peptides=known_peptides,
    llm=llm,
    mode=peptide_mode,
)
```

## 5.3 Remove the misleading warning in known-panel mode

```python
if mode == "known_panel":
    log.info("Using deterministic known antiviral peptide panel")
elif mode == "llm_design" and llm is None:
    raise RuntimeError("LLM peptide design requested but no LLM client was supplied")
```

## 5.4 Validate generated peptides

Require:

```text
allowed amino-acid alphabet
configured length range
no duplicate sequence
no empty sequence
provenance=llm_design
model identifier
prompt version
```

LLM-designed peptides must not be merged with known peptides without a source label.

## Acceptance criteria

- Known-panel mode never emits a missing-LLM warning.
- LLM-design mode fails clearly when no client is available.
- Every peptide records its source and model metadata.

---

# 6. Final report consistency rules

## Files to change

```text
VLAB2/orchestration/postrun.py
VLAB2/orchestration/reporting.py
VLAB2/orchestration/agents/pi_agent.py
```

## Required report ordering

```text
1. Best validated scientific conclusion
2. Latest batch diagnostics
3. Conservation state
4. RNA docking results
5. Small-molecule Vina results
6. Peptide HDOCK results
7. Optimiser health
8. Warnings and limitations
```

## Required target wording

Do not render:

```text
Target PDB: 8K75 (partial success)
```

when a best accepted record exists.

Render:

```text
Best validated target: 8K75 (accepted_target)
Latest evaluated batch: 8K75 (partial_success_target)
```

## Required Vina wording

If coverage is incomplete:

```text
Vina interpretation: exploratory partial-pocket screen
Pocket coverage: 31/40 points (77.5%)
```

If split boxes cover the full pocket:

```text
Vina interpretation: complete multi-subpocket screen
Combined required-point coverage: 100%
```

## Required motif wording

```text
GAG@sequence[5,8)
CUG@alignment[40,43)
```

## Required score separation

```text
RNA HDOCK: relative score, more negative ranked better within HDOCK
Peptide HDOCK: relative score, more negative ranked better within HDOCK
AutoDock Vina: kcal/mol
```

Never combine HDOCK and Vina values into one numeric ranking.

---

# 7. Regression suite

## New or updated tests

```text
VLAB2/tests/core/test_conservation_coordinate_mapping.py
VLAB2/tests/core/test_motif_coordinate_mapping.py
VLAB2/tests/core/test_vina_box_constraints.py
VLAB2/tests/core/test_peptide_design_mode.py
VLAB2/tests/orchestration/agents/test_best_target_preservation.py
VLAB2/tests/orchestration/agents/test_protein_target_state_views.py
VLAB2/tests/orchestration/agents/test_inhibitor_box_policy.py
VLAB2/tests/orchestration/agents/test_inhibitor_peptide_mode.py
VLAB2/tests/orchestration/test_final_reporting.py
VLAB2/tests/orchestration/test_routing_target_state.py
VLAB2/tests/orchestration/test_training_export_target_state.py
```

## Mandatory scenarios

### Target state

- accepted then partial, best remains accepted;
- partial then accepted, best promotes to accepted;
- accepted then stronger accepted, best promotes;
- accepted then equal accepted, earlier record remains best;
- two different targets, stronger validated record wins;
- latest record always changes after a completed batch;
- best record never references missing binding rows.

### Motif mapping

- ungapped interval;
- gaps before interval;
- gap inside interval;
- endpoint at alignment length;
- mapped motif length preserved;
- mapped slice equals concrete match;
- invalid mapping becomes unmapped.

### Reporting

- headline uses best validated target;
- latest status remains visible;
- motif bracket formatting is exact;
- partial Vina coverage is labelled exploratory;
- score units remain separate.

### Vina box

- final capped box revalidated;
- run-derived 40.714 Å extent fails single 32 Å box;
- split policy reaches complete coverage;
- strict policy skips invalid screen;
- partial policy marks all outputs.

### Peptide mode

- known panel without LLM;
- LLM mode with client;
- LLM mode without client fails clearly;
- generated peptide provenance recorded.

---

# 8. Suggested implementation sequence

## Commit 1: target-state split

```text
state_schema.py
state_factory.py
target_state_utils.py
protein_agent.py
```

Add tests before touching reporting.

## Commit 2: downstream target-state consumers

```text
routing.py
inhibitor_agent.py
skeptic_agent.py
pi_agent.py
failure_memory.py
training_data_collector.py
```

## Commit 3: report semantics

```text
postrun.py
reporting.py
```

## Commit 4: motif mapping

```text
sequence_utils.py
bioinfo_wrapper.py
bioinfo_agent.py
```

## Commit 5: Vina geometry policy

```text
rna_binding_pocket.py
inhibitor_agent.py
inhibitor_docking.py
vina_wrapper.py
```

## Commit 6: peptide design mode

```text
config.py
peptide_prep.py
inhibitor_agent.py
```

## Commit 7: pre-SLURM regression gate

Add all focused tests to the launch script before vLLM startup.

---

# 9. Validation commands

## Compile check

```bash
python -m py_compile \
  core/bioinfo_wrapper.py \
  core/rna_binding_pocket.py \
  core/inhibitor_docking.py \
  core/vina_wrapper.py \
  core/peptide_prep.py \
  orchestration/state_schema.py \
  orchestration/state_factory.py \
  orchestration/utils/sequence_utils.py \
  orchestration/utils/target_state_utils.py \
  orchestration/agents/protein_agent.py \
  orchestration/agents/inhibitor_agent.py \
  orchestration/agents/pi_agent.py \
  orchestration/agents/skeptic_agent.py \
  orchestration/routing.py \
  orchestration/postrun.py \
  orchestration/reporting.py
```

## Focused tests

```bash
python -m pytest -q \
  tests/core/test_conservation_coordinate_mapping.py \
  tests/core/test_motif_coordinate_mapping.py \
  tests/core/test_vina_box_constraints.py \
  tests/core/test_peptide_design_mode.py \
  tests/orchestration/agents/test_best_target_preservation.py \
  tests/orchestration/agents/test_protein_target_state_views.py \
  tests/orchestration/agents/test_inhibitor_box_policy.py \
  tests/orchestration/agents/test_inhibitor_peptide_mode.py \
  tests/orchestration/test_final_reporting.py \
  tests/orchestration/test_routing_target_state.py \
  tests/orchestration/test_training_export_target_state.py
```

## Target-state mutation search

After implementation, audit direct mutations:

```bash
grep -R -n \
  -E 'target_status|best_validated_target_status|target_pdb' \
  orchestration \
  | grep -v 'target_state_utils.py'
```

Every mutation should either construct a batch record or derive a compatibility alias.

## Motif formatting search

```bash
grep -R -n \
  -E '@alignment|@sequence|Selected motifs' \
  orchestration
```

All formatting should flow through `format_motif_location()`.

---

# 10. Pre-run checklist

## Target state

- [ ] Latest and best target records are distinct objects.
- [ ] A partial record cannot demote an accepted best record.
- [ ] Best record promotion is deterministic.
- [ ] Inhibitor screening uses the best record’s binding-result IDs.
- [ ] Final headline uses best validated state.
- [ ] Latest batch is reported separately.
- [ ] Training exports label their state view.

## Motif coordinates

- [ ] MSA intervals are zero-based and half-open.
- [ ] Sequence intervals are zero-based and half-open.
- [ ] Mapped length equals concrete motif length.
- [ ] Mapped slice equals the concrete matched sequence.
- [ ] Invalid mappings become unmapped.
- [ ] Formatted coordinates contain `[` and `)`.

## Vina boxes

- [ ] The true minimum enclosing box is calculated without clipping.
- [ ] Every final box is revalidated.
- [ ] Full-pocket screens require 100% point coverage.
- [ ] Oversized pockets are split or skipped.
- [ ] Partial screens are explicitly exploratory.
- [ ] Cache identity includes pocket geometry and identity.

## Peptide mode

- [ ] Known-panel mode is deterministic.
- [ ] LLM-design mode receives a client.
- [ ] LLM-designed peptides have provenance metadata.

## Reporting

- [ ] Best target and latest target cannot be confused.
- [ ] HDOCK and Vina units remain separate.
- [ ] Limitations include incomplete conservation mapping or partial pocket coverage.

---

# Definition of done

The deterministic bug-fix batch is complete when a weaker later target-evaluation batch cannot replace a stronger accepted target in the scientific headline, while remaining visible as the latest diagnostic result; motif mappings preserve half-open interval length and sequence identity; motif coordinates render unambiguously; Vina never claims full-pocket validity with excluded required points; peptide design mode is explicit; and all run-derived regression tests pass before vLLM and the expensive docking workflow start.
