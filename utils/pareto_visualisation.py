"""
pareto_visualisation.py

3D visualisation + interactive selection
"""

import numpy as np
import plotly.graph_objects as go


# ----------------------------------------------------------------------
# DATA EXTRACTION
# ----------------------------------------------------------------------
def extract_population(population, decode_fn):
    sequences = []
    objs = []

    for ind in population:
        seq = decode_fn(ind.X)
        sequences.append(seq)

        # invert objectives back to maximisation
        obj = [-v for v in ind.F]
        objs.append(obj)

    return sequences, np.array(objs)


# ----------------------------------------------------------------------
# 3D PLOT
# ----------------------------------------------------------------------
def plot_3d_pareto(population, decode_fn):
    sequences, objs = extract_population(population, decode_fn)

    x = objs[:, 0]  # structure
    y = objs[:, 1]  # binding
    z = objs[:, 2]  # MD

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=6,
                    color=z,
                    colorscale="Viridis",
                    opacity=0.8
                ),
                text=sequences,
                hovertemplate=(
                    "Sequence: %{text}<br>"
                    "Structure: %{x:.3f}<br>"
                    "Binding: %{y:.3f}<br>"
                    "MD: %{z:.3f}"
                ),
            )
        ]
    )

    fig.update_layout(
        title="Pareto Front (Structure vs Binding vs MD)",
        scene=dict(
            xaxis_title="Structure",
            yaxis_title="Binding",
            zaxis_title="MD Stability",
        )
    )

    fig.show()


# ----------------------------------------------------------------------
# AUTO SELECT BEST EXPERIMENT CANDIDATES
# ----------------------------------------------------------------------
def select_candidates(population, decode_fn):
    sequences, objs = extract_population(population, decode_fn)

    # normalise
    norm = (objs - objs.min(axis=0)) / (objs.max(axis=0) - objs.min(axis=0) + 1e-8)

    # ideal point
    ideal = np.ones(3)

    distances = np.linalg.norm(norm - ideal, axis=1)

    best_idx = np.argmin(distances)

    # extremes
    best_structure = np.argmax(objs[:, 0])
    best_binding = np.argmax(objs[:, 1])
    best_md = np.argmax(objs[:, 2])

    selected_indices = set([
        best_idx,
        best_structure,
        best_binding,
        best_md
    ])

    return [(i, sequences[i], objs[i]) for i in selected_indices]
