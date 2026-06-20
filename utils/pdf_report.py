"""
pdf_report.py

Creates:
- plots (saved as PNG)
- PDF report combining:
    - summary
    - interpretation
    - plots
"""

import os
import json

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet


# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------
def load_data(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# PLOT GENERATION
# ----------------------------------------------------------------------
def generate_plots(data, outdir):
    import matplotlib.pyplot as plt
    gens = data["generations"]

    # --- fitness ---
    best = [sum(g["best_score"].values()) for g in gens]
    avg = [sum(g["average_scores"].values()) for g in gens]

    plt.figure()
    plt.plot(best, label="Best")
    plt.plot(avg, label="Average")
    plt.legend()
    plt.title("Fitness")
    fitness_path = os.path.join(outdir, "fitness.png")
    plt.savefig(fitness_path)
    plt.close()

    # --- weights ---
    weights_keys = gens[0]["weights"].keys()
    weight_data = {k: [] for k in weights_keys}

    for g in gens:
        for k in weights_keys:
            weight_data[k].append(g["weights"][k])

    plt.figure()
    for k, vals in weight_data.items():
        plt.plot(vals, label=k)
    plt.legend()
    plt.title("Weights")
    weights_path = os.path.join(outdir, "weights.png")
    plt.savefig(weights_path)
    plt.close()

    # --- diversity ---
    diversity = [g["unique_sequences"] for g in gens]

    plt.figure()
    plt.plot(diversity)
    plt.title("Diversity")
    diversity_path = os.path.join(outdir, "diversity.png")
    plt.savefig(diversity_path)
    plt.close()

    return fitness_path, weights_path, diversity_path


# ----------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------
def compute_summary(data):
    gens = data["generations"]

    best_scores = [sum(g["best_score"].values()) for g in gens]

    return {
        "final_best": best_scores[-1],
        "improvement": best_scores[-1] - best_scores[0],
        "best_sequence": gens[-1]["best_sequence"],
        "weights": gens[-1]["weights"],
    }


# ----------------------------------------------------------------------
# PDF GENERATION
# ----------------------------------------------------------------------
def generate_pdf(json_path):
    data = load_data(json_path)
    outdir = os.path.dirname(json_path)

    fitness, weights, diversity = generate_plots(data, outdir)

    summary = compute_summary(data)

    pdf_path = os.path.join(outdir, "report.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("RNA DESIGN EXPERIMENT REPORT", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Summary:", styles["Heading2"]))
    for k, v in summary.items():
        elements.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Plots:", styles["Heading2"]))

    elements.append(Image(fitness, width=400, height=200))
    elements.append(Image(weights, width=400, height=200))
    elements.append(Image(diversity, width=400, height=200))

    doc.build(elements)

    print(f"PDF report saved to: {pdf_path}")