"""
pareto_analysis.py

Tools to:
- analyse Pareto front
- cluster solutions
- select "best compromise" sequences
"""

import numpy as np


# ----------------------------------------------------------------------
# EXTRACT DATA
# ----------------------------------------------------------------------
def extract_population(population, decode_fn):
    sequences = []
    objectives = []

    for ind in population:
        seq = decode_fn(ind.X)
        sequences.append(seq)

        # invert back (since pymoo uses minimisation)
        obj = [-v for v in ind.F]
        objectives.append(obj)

    return sequences, np.array(objectives)


# ----------------------------------------------------------------------
# NORMALISE OBJECTIVES
# ----------------------------------------------------------------------
def normalize_objectives(objs):
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)

    return (objs - mins) / (maxs - mins + 1e-8)


# ----------------------------------------------------------------------
# BEST COMPROMISE (IMPORTANT)
# ----------------------------------------------------------------------
def best_compromise(objs):
    """
    Finds sequence closest to ideal point
    """

    ideal = objs.max(axis=0)
    distances = np.linalg.norm(objs - ideal, axis=1)

    return np.argmin(distances)


# ----------------------------------------------------------------------
# EXTREME SOLUTIONS
# ----------------------------------------------------------------------
def extreme_solutions(objs):
    """
    Find sequences that maximise each objective
    """
    return [np.argmax(objs[:, i]) for i in range(objs.shape[1])]


# ----------------------------------------------------------------------
# SIMPLE CLUSTERING (k-means)
# ----------------------------------------------------------------------
def cluster_solutions(objs, k=3):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(objs)

    return labels


# ----------------------------------------------------------------------
# FULL ANALYSIS
# ----------------------------------------------------------------------
def analyse_pareto(population, decode_fn):
    sequences, objs = extract_population(population, decode_fn)

    norm = normalize_objectives(objs)

    best_idx = best_compromise(norm)
    extremes = extreme_solutions(norm)
    clusters = cluster_solutions(norm, k=3)

    print("\n=== PARETO ANALYSIS ===\n")

    print("Best compromise sequence:")
    print(sequences[best_idx], objs[best_idx])
    print()

    print("Extreme solutions:")
    for i, idx in enumerate(extremes):
        print(f"Objective {i}: {sequences[idx]} {objs[idx]}")
    print()

    print("Clusters:")
    for c in set(clusters):
        print(f"\nCluster {c}:")
        for i in range(len(sequences)):
            if clusters[i] == c:
                print(sequences[i])

    return {
        "best_idx": best_idx,
        "extremes": extremes,
        "clusters": clusters,
    }