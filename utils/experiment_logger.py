"""
experiment_logger.py (ENHANCED)

Handles:
- structured logging (JSON)
- run metadata
- generation tracking
- best sequence tracking
- diversity metrics
- top sequence export
"""

import json
import os
import time
from typing import List, Dict


class ExperimentLogger:
    def __init__(self, base_dir="experiments"):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"run_{timestamp}")

        os.makedirs(self.run_dir, exist_ok=True)

        self.log_file = os.path.join(self.run_dir, "log.txt")
        self.data_file = os.path.join(self.run_dir, "data.json")
        self.top_file = os.path.join(self.run_dir, "top_sequences.txt")

        self.data = {
            "metadata": {},
            "generations": []
        }

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------
    def log_metadata(self, metadata: Dict):
        self.data["metadata"] = metadata
        self._write()

    # ------------------------------------------------------------------
    # GENERATION LOGGING
    # ------------------------------------------------------------------
    def log_generation(self, gen: int, sequences: List[str], scores: List[Dict], weights: Dict):
        if not sequences or not scores:
            return

        # -------------------------
        # Clean score dicts (remove internal keys)
        # -------------------------
        clean_scores = []
        for s in scores:
            clean_scores.append({k: v for k, v in s.items() if not k.startswith("_")})

        # -------------------------
        # Compute best sequence
        # -------------------------
        def objective_sum(s):
            return sum(v for k, v in s.items() if not k.startswith("_"))

        best_idx = max(range(len(clean_scores)), key=lambda i: objective_sum(clean_scores[i]))
        best_seq = sequences[best_idx]
        best_score = clean_scores[best_idx]

        # -------------------------
        # Diversity metric
        # -------------------------
        unique_sequences = len(set(sequences))

        # -------------------------
        # Aggregate averages
        # -------------------------
        avg_scores = {}
        for key in clean_scores[0]:
            avg_scores[key] = sum(s[key] for s in clean_scores) / len(clean_scores)

        # -------------------------
        # Entry
        # -------------------------
        entry = {
            "generation": gen,
            "sequences": sequences,
            "scores": clean_scores,
            "weights": weights,
            "best_sequence": best_seq,
            "best_score": best_score,
            "unique_sequences": unique_sequences,
            "average_scores": avg_scores,
        }

        self.data["generations"].append(entry)

        # -------------------------
        # Persist JSON
        # -------------------------
        self._write()

        # -------------------------
        # Human-readable log
        # -------------------------
        with open(self.log_file, "a") as f:
            f.write(f"\n--- Generation {gen} ---\n")
            f.write(f"Weights: {weights}\n")
            f.write(f"Best sequence: {best_seq}\n")
            f.write(f"Best score: {best_score}\n")
            f.write(f"Unique sequences: {unique_sequences}\n")

        # -------------------------
        # Top sequences file (append)
        # -------------------------
        with open(self.top_file, "a") as f:
            f.write(f"Gen {gen}: {best_seq} -> {best_score}\n")

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    def _write(self):
        # Write safely (avoid partial write corruption)
        tmp_file = self.data_file + ".tmp"

        with open(tmp_file, "w") as f:
            json.dump(self.data, f, indent=2)

        os.replace(tmp_file, self.data_file)