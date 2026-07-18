```python
# In pi_agent.py
from VLAB2.optimisation.optimisation_engine import optimise_sequences

# Build combined objectives from feedback
combined_objectives = _build_combined_objectives(
    state, parsed, extra_objectives, failure_weights,
    conservation_fitness, critique, score_spread, joint_feedback
)

# Run optimisation
selected_sequences = optimise_sequences(
    initial_sequences=state["designed_sequences"],
    wrapper_bundle=wrappers,
    pdb_id=target_pdb,
    generations=3,
    population_size=20,
)
Feedback Integration
The PI agent enhances the basic optimisation with:

Joint physics feedback: Combines docking, MD, and interface signals
Failure memory: Penalises previously failed designs
Literature constraints: Incorporates motif hints from research
Conservation pressure: Weights objectives by evolutionary conservation
Adaptive mutation bias: Adjusts mutation rates based on iteration
Environment Variables
Variable	Default	Description
VLAB_RNA_ENFORCE_MIN_FOLD	1	Enforce fold quality thresholds
VLAB_ACCEPT_BINDING_SCORE	-50.0	Threshold for favourable binding
VLAB_MAX_ACCEPT_SCORE_SPREAD	10.0	Max acceptable score spread
VLAB_MIN_VALID_HDOCK_SCORE	-30.0	Minimum valid HDOCK score
VLAB_TARGET_RESET_ON_HIGH_SPREAD	1	Auto-reset target on high spread
Key Design Principles
Multi-objective: Optimises binding, stability, folding, conservation simultaneously
Surrogate-guided: Uses fast neural predictions to rank candidates before expensive MD
Adaptive: Dynamically adjusts mutation rates and objective weights
Pareto-optimal: Preserves trade-off solutions rather than single best
Diversity-preserving: Prevents population collapse through filtering
Feedback-driven: Integrates MD, docking, and interface quality signals
Memory-aware: Learns from historical failures and literature
Last updated: 2026-07-15