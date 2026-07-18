# Most Recent Virtual Lab Run Review

## Executive verdict

The run is substantially healthier: three-RNA evaluation worked, deterministic Vina and persistent caching worked, and the coverage audit completed. One concrete structural bug remains, plus two state-consistency issues that explain the contradictory target and interface summaries.

### Working correctly

- Three RNA candidates reached HDOCK.
- The first structural bootstrap completed normally.
- Vina used deterministic seed `1`.
- Persistent Vina entries were created successfully.
- Interface-aware target evaluation worked.
- Inhibitor screening worked for accepted and partial-success targets.
- Coverage completed at **60.3%** with eight zero-coverage files identified.
- The orchestrator exited with code `0`.

### Still requiring fixes

1. `_local_interface_variants()` calls `mutate_sequence()` with an unsupported keyword.
2. Structural failure did not prevent MD, protein, and inhibitor nodes from processing stale sequences.
3. Interface-clean counts appear to include accumulated rows from earlier iterations or targets.
4. PI resets an accepted target using proxy or joint-feedback scores rather than the current measured clean-interface docking rows.
5. The Vina cache populated correctly, but this run mostly exercised cache misses because pocket geometry changed between iterations.

---

## 1. Immediate structural fix

The remaining exception is:

```text
TypeError: mutate_sequence() got an unexpected keyword argument 'n_mutations'
```

It occurs in `_local_interface_variants()` on the first anchored structural pass.

Your actual `mutate_sequence()` signature evidently does not use `n_mutations`. The previous positional call worked:

```python
mutate_sequence(anchor, n_mutations, locked_positions)
```

Therefore, change:

```python
raw_variant = mutate_sequence(
    seq=anchor,
    n_mutations=n_mutations,
    locked_positions=locked_positions,
)
```

back to:

```python
raw_variant = mutate_sequence(
    anchor,
    n_mutations,
    locked_positions,
)
```

Alternatively, inspect the actual signature:

```bash
grep -n -A20   "def mutate_sequence"   orchestration/utils/sequence_utils.py
```

Then use its real parameter names. The safest immediate fix is the positional form because that previously executed successfully.

### Add a signature regression diagnostic

```python
import inspect

log.info(
    "[STRUCTURAL LOCAL DEBUG] mutate_sequence signature=%s",
    inspect.signature(mutate_sequence),
)
```

This can be temporary. It will reveal the exact expected names in the next run.

---

## 2. Locked-position conversion is currently wrong

The anchored iteration logged:

```text
conserved_regions=[(0, 7), (37, 43)]
anchor_len=29
locked_positions=[0,1,2,3,4,5,6,36,37,38,39,40,41,42]
```

The second conserved interval lies entirely beyond a 29-nt anchor. More importantly, a region beginning at `0` cannot be one-based. Your `_build_locked_positions()` documentation says the coordinates are one-based inclusive, but the wrapper is clearly returning at least some zero-based coordinates.

This did not directly cause the exception, but it corrupts local mutation constraints.

Replace `_build_locked_positions()` with a length-aware version:

```python
def _build_locked_positions(
    conserved_regions,
    sequence_length: int | None = None,
) -> set[int]:
    """
    Convert conserved-region intervals to valid zero-based positions.

    Bioinformatics output is treated as zero-based, end-exclusive when a
    region starts at 0. Otherwise one-based inclusive coordinates are
    converted conservatively.
    """
    locked_positions: set[int] = set()

    for item in conserved_regions or []:
        try:
            start, end = item
            start = int(start)
            end = int(end)
        except Exception:
            continue

        if end <= start:
            continue

        if start == 0:
            start_zero = start
            end_exclusive = end
        else:
            start_zero = start - 1
            end_exclusive = end

        if sequence_length is not None:
            start_zero = max(0, min(start_zero, sequence_length))
            end_exclusive = max(
                start_zero,
                min(end_exclusive, sequence_length),
            )

        locked_positions.update(
            range(start_zero, end_exclusive)
        )

    return locked_positions
```

Then call it only after the anchor is known:

```python
locked_positions = _build_locked_positions(
    conserved_regions,
    sequence_length=(
        len(clean_anchor)
        if clean_anchor
        else None
    ),
)
```

For the recorded 29-nt anchor, the out-of-range interval `(37, 43)` will contribute no locked positions.

---

## 3. Structural failure is still allowing stale downstream work

After the structural agent failed, MD immediately evaluated:

```text
GCGAACAGUCGAGGAGACCGGAGUCGAGC
UCCGAAUCGGCUCCGAUCCGGAGUCCGUU
CGGCUCCGAUCGGAAUCGGCUGUCGGCAA
```

Protein docking then proceeded with the same stale or inherited candidates.

The fixed `should_continue()` router acts only after Skeptic, so it cannot prevent same-cycle MD and protein execution. You need an additional conditional route immediately after structural.

### Add a post-structural router

```python
def after_structural(state: LabState) -> str:
    if state.get("structural_status") == "failed":
        log.warning(
            "Skipping MD/protein after structural failure: %s",
            state.get("structural_error"),
        )
        return "pi"

    if not state.get("designed_sequences"):
        log.warning(
            "Skipping MD/protein because structural output "
            "contains no designed sequences."
        )
        return "pi"

    return "md"
```

Graph wiring should resemble:

```python
graph.add_conditional_edges(
    "structural",
    after_structural,
    {
        "md": "md",
        "pi": "pi",
    },
)
```

If you want to stop rather than retry:

```python
return "end"
```

This is important because returning:

```python
"designed_sequences": []
```

does not necessarily clear an annotated reducer list. A list reducer may merge the empty list with old state and preserve stale sequences.

### Do not rely on empty-list updates to clear reducer-backed fields

Your `designed_sequences` field uses a deduplication reducer. Returning `[]` does not erase prior values. Therefore, on structural failure, use routing to prevent stale consumption rather than expecting an empty update to clear history.

A cleaner long-term split would be:

```python
designed_sequences_history
current_structural_sequences
```

with only the historical field using a reducer.

---

## 4. Interface-clean count is inconsistent

At the second partial-success target evaluation, the protein agent reported:

```text
interface_clean=1
steric_clash=2
```

but the inhibitor router then reported:

```text
clean_rows=2
```

That strongly suggests `_inhibitor_should_run()` counted clean rows accumulated from previous iterations rather than only the current docking batch.

The same accumulation is visible in the final joint feedback:

```text
clean_interface=2
clash=2
```

which represents more rows than one three-candidate batch.

### Filter inhibitor rows to the current docking run

Add a run or batch identifier to protein results, for example:

```python
"docking_iteration": state.get("iterations", 0),
```

Then filter:

```python
current_iteration = state.get("iterations", 0)

clean_rows = [
    row
    for row in state.get("binding_results", []) or []
    if isinstance(row, dict)
    and row.get("docking_iteration") == current_iteration
    and row.get("dock_valid") is True
    and row.get("interface_passed") is True
    and row.get("interface_steric_clash") is not True
]
```

A stronger key is:

```python
docking_batch_id
```

generated once per protein-agent invocation and returned alongside:

```python
current_binding_results
current_docking_batch_id
```

Then routing should use only `current_binding_results`, while `binding_results` remains historical.

---

## 5. PI target reset is using the wrong score layer

After an accepted target with measured HDOCK scores of:

```text
-57.56
-61.39
-66.69
```

and two clean interfaces, PI reset the target because joint feedback contained:

```text
best_relative_score=-49.0639
spread=6.6927
```

Those values do not match the current physical HDOCK batch.

The PI appears to be resetting the target using a proxy, aggregated, inherited, or transformed score rather than current measured docking results.

### Target reset should use current measured rows only

Before resetting an accepted target, compute:

```python
current_rows = [
    row
    for row in current_binding_results
    if row.get("dock_valid") is True
    and row.get("dock_method") == "hdock"
    and row.get("interface_passed") is True
    and row.get("interface_steric_clash") is not True
]

current_scores = [
    float(row["dock_score"])
    for row in current_rows
    if row.get("dock_score") is not None
]
```

Only reset if:

```python
len(current_scores) < min_required
```

or the measured clean-interface scores themselves fail the configured policy.

In the first batch, two clean interfaces existed and all three measured HDOCK scores were stronger than `-50`. The accepted target should not have been reset solely because an aggregated NSGA or joint-feedback value was `-49.06`.

---

## 6. Vina reproducibility and caching worked

The implementation behaved as intended:

- Vina received `--seed 1`.
- The executable reported `random seed: 1`.
- Successful poses and energies were saved into the persistent result cache.
- Cache keys were generated per receptor, ligand, box, exhaustiveness, seed, and binary fingerprint.

The first inhibitor pass naturally produced misses because it populated the cache.

Later passes had different pocket geometry.

First pocket:

```text
center=(4.32, -4.39, 3.65)
size=(40.8, 25.0, 50.5)
```

Second pocket:

```text
center=(6.26, 8.26, 8.90)
size=(29.3, 26.4, 26.0)
```

Different geometry correctly produces different cache keys.

### The large Vina box should be reviewed

Vina repeatedly warned that the first search-space volume exceeded `27,000 Å³`. The first box volume was approximately:

```text
40.8 × 25.0 × 50.5 ≈ 51,500 Å³
```

This is almost twice the warning threshold. The second box was much tighter, approximately:

```text
29.3 × 26.4 × 26.0 ≈ 20,100 Å³
```

The pocket-builder should cap each dimension or shrink the padding when the box is oversized.

Suggested controls:

```bash
export VLAB_MAX_VINA_BOX_VOLUME=27000
export VLAB_MAX_VINA_BOX_DIMENSION=32
export VLAB_POCKET_PADDING=6
```

These require corresponding support in `define_docking_box()`.

---

## 7. Final scientific outcome

The bootstrap structural candidate selection worked correctly. Five candidates passed into folding, and the selected sequence was:

```text
UCCGAAUCGGCUCCGAUCCGGAGUCCGUU
```

with:

```text
pair_density=0.4828
mfe_per_nt=-0.37241
fold_pass=True
```

Three candidates then reached HDOCK, confirming the batch script’s `eval_top_n=3` correction.

The first protein pass produced three docking-valid results, two clean interfaces, and one clash, which was enough to retain `8K75` as `accepted_target`. The later pass had only one clean interface and two clashes, correctly producing `partial_success_target`.

Bioinformatics improved after sequences became available, reaching conservation fitness `0.672`, two regions, and GAG/RGAG/GUC motif support.

The final report nevertheless ended with a partial-success target because of the later interface batch and the structural local-mutation failure.

---

## 8. Coverage findings

Overall coverage was:

```text
60.3%
12,527 statements
4,429 missed
4,830 branches
1,215 partial branches
```

Eight files had zero coverage in this run.

### Likely legacy optimisation stack

These modules were unused while `final_rna_design_system.py` reached 73.6% coverage:

```text
optimisation/acquisition.py
optimisation/adaptive_mutation.py
optimisation/neural_surrogate.py
optimisation/optimisation_backend.py
optimisation/optimisation_engine.py
optimisation/optimiser_engine.py
optimisation/surrogate_model.py
```

This strongly suggests the active PI path goes through `final_rna_design_system.py` and the remaining optimisation modules are legacy or alternative implementations. That is evidence for review, not yet proof that they are safe to delete.

Before deleting them:

```bash
grep -R   -e "optimisation_engine"   -e "optimiser_engine"   -e "neural_surrogate"   -e "surrogate_model"   -e "optimisation_backend"   -e "adaptive_mutation"   -e "acquisition"   .   --exclude-dir=output_data   --exclude-dir=hf_cache
```

`find_used_files.py` itself had zero coverage because it was not invoked by the orchestrator. That is expected for a standalone diagnostic script.

---

## Recommended priority order

### Must fix before the next scientific run

1. Restore the compatible positional call to `mutate_sequence()`.
2. Make conserved-position conversion length-aware.
3. Add a post-structural conditional edge so failure skips MD and protein.
4. Separate current docking rows from historical binding results.
5. Prevent PI from resetting accepted targets using proxy or aggregated scores.

### Useful optimisation

6. Tighten oversized Vina boxes.
7. Add persistent HDOCK result caching, since repeated RNA and peptide dockings consumed most of the runtime.
8. Rename the peptide message from “LLM designed” when conservative defaults were used.
9. Fix the LangChain `model_kwargs` warnings by passing supported generation arguments explicitly.
10. Isolate or archive the seven unused optimisation modules after repository-wide import checks.

## Immediate one-line correction

```python
raw_variant = mutate_sequence(
    anchor,
    n_mutations,
    locked_positions,
)
```

## Most important architectural correction

Route away from MD and protein immediately after:

```python
structural_status == "failed"
```

This prevents stale structural candidates and historical docking rows from being treated as current-cycle evidence.
