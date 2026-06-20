"""
visualisation.py

Plots:
- fitness convergence
- weight evolution
- diversity
- motif / structure signals
"""

import json
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------
def load_data(path: str):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# FITNESS EVOLUTION
# ----------------------------------------------------------------------
def plot_fitness(data):
    gens = data["generations"]

    best_scores = []
    avg_scores = []

    for g in gens:
        best = sum(g["best_score"].values())
        avg = sum(g["average_scores"].values())

        best_scores.append(best)
        avg_scores.append(avg)

    plt.figure()
    plt.plot(best_scores, label="Best fitness")
    plt.plot(avg_scores, label="Average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Convergence")
    plt.legend()
    plt.show()


# ----------------------------------------------------------------------
# WEIGHT EVOLUTION
# ----------------------------------------------------------------------
def plot_weights(data):
    gens = data["generations"]

    weights_over_time = {k: [] for k in gens[0]["weights"]}

    for g in gens:
        for k, v in g["weights"].items():
            weights_over_time[k].append(v)

    plt.figure()

    for k, vals in weights_over_time.items():
        plt.plot(vals, label=k)

    plt.xlabel("Generation")
    plt.ylabel("Weight")
    plt.title("Constraint Weight Evolution")
    plt.legend()
    plt.show()


# ----------------------------------------------------------------------
# DIVERSITY
# ----------------------------------------------------------------------
def plot_diversity(data):
    gens = data["generations"]

    diversity = [g["unique_sequences"] for g in gens]

    plt.figure()
    plt.plot(diversity)
    plt.xlabel("Generation")
    plt.ylabel("Unique sequences")
    plt.title("Population Diversity")
    plt.show()


# ----------------------------------------------------------------------
# RAW SIGNALS (STRUCTURE + MOTIF)
# ----------------------------------------------------------------------
def plot_signals(data):
    gens = data["generations"]

    pd_vals = []
    motif_vals = []

    for g in gens:
        raw_pd = []
        raw_motif = []

        for s in g["scores"]:
            # fallback if raw not stored
            raw_pd.append(s.get("thermo", 0))
            raw_motif.append(s.get("motif", 0))

        pd_vals.append(sum(raw_pd) / len(raw_pd))
        motif_vals.append(sum(raw_motif) / len(raw_motif))

    plt.figure()
    plt.plot(pd_vals, label="Structure signal")
    plt.plot(motif_vals, label="Motif signal")
    plt.xlabel("Generation")
    plt.ylabel("Signal strength")
    plt.title("Emergent Signals")
    plt.legend()
    plt.show()


# ----------------------------------------------------------------------
# FULL DASHBOARD
# ----------------------------------------------------------------------
def plot_all(json_path: str):
    data = load_data(json_path)

    plot_fitness(data)
    plot_weights(data)
