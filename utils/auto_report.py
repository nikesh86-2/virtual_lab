"""
auto_report.py

Generates:
- summary stats
- key findings
- simple text report

Inputs:
    experiment JSON log
"""

import json
import os

# ----------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------
def load_data(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# SUMMARY METRICS
# ----------------------------------------------------------------------
def compute_summary(data):
    gens = data["generations"]

    final_gen = gens[-1]

    best_scores = [sum(g["best_score"].values()) for g in gens]
    avg_scores = [sum(g["average_scores"].values()) for g in gens]
    diversity = [g["unique_sequences"] for g in gens]

    return {
        "final_best_score": best_scores[-1],
        "final_avg_score": avg_scores[-1],
        "improvement": best_scores[-1] - best_scores[0],
        "final_diversity": diversity[-1],
        "min_diversity": min(diversity),
        "max_diversity": max(diversity),
        "final_weights": final_gen["weights"],
        "best_sequence": final_gen["best_sequence"],
    }


# ----------------------------------------------------------------------
# INTERPRETATION
# ----------------------------------------------------------------------
def interpret(summary):
    text = []

    # improvement
    if summary["improvement"] > 0.5:
        text.append("Strong optimisation observed.")
    elif summary["improvement"] > 0.1:
        text.append("Moderate optimisation observed.")
    else:
        text.append("Limited optimisation detected.")

    # diversity
    if summary["min_diversity"] < 5:
        text.append("Possible premature convergence (low diversity).")

    # weights
    dominant = max(summary["final_weights"], key=summary["final_weights"].get)
    text.append(f"Dominant constraint: {dominant}")

    return text


# ----------------------------------------------------------------------
# WRITE REPORT
# ----------------------------------------------------------------------
def generate_report(json_path, output_dir=None):
    data = load_data(json_path)

    summary = compute_summary(data)
    interpretation = interpret(summary)

    if output_dir is None:
        output_dir = os.path.dirname(json_path)

    report_path = os.path.join(output_dir, "report.txt")

    with open(report_path, "w") as f:
        f.write("=== RNA DESIGN EXPERIMENT REPORT ===\n\n")

        f.write(">> Summary\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

        f.write("\n>> Interpretation\n")
        for line in interpretation:
            f.write(f"- {line}\n")

    print(f"Report saved to: {report_path}")
