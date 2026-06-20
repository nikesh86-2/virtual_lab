"""
training_data_collector.py (UPGRADED)

Now captures:
- agent reasoning
- research cycles
- Pareto optimisation decisions
- mutation bias evolution
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("virtual_lab.training_data")


class TrainingDataCollector:

    def __init__(
        self,
        output_dir: str = "training_data",
        prefix: str = "biophysics_tuning",
        session_id: Optional[str] = None,
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            self.output_dir,
            f"{prefix}_{self.session_id}.jsonl",
        )

    # ------------------------------------------------------------------
    # CORE RESEARCH CYCLE
    # ------------------------------------------------------------------
    def collect_cycle(
        self,
        topic: str,
        hypothesis: str,
        evidence: List[Dict[str, Any]],
        critique: str,
        final_synthesis: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        entry = {
            "instruction": "Evaluate a biophysics research cycle.",
            "input": json.dumps({
                "topic": topic,
                "hypothesis": hypothesis,
                "evidence": evidence,
                "critique": critique,
            }),
            "output": final_synthesis,
            "metadata": {
                "timestamp": self._timestamp(),
                "session_id": self.session_id,
                "type": "research_cycle",
                **(extra_metadata or {}),
            },
        }

        self.save_entry(entry)

    # ------------------------------------------------------------------
    # AGENT INTERACTION
    # ------------------------------------------------------------------
    def capture_agent_interaction(
        self,
        agent_name: str,
        task: str,
        response: str,
        input_context: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        entry = {
            "instruction": f"Act as {agent_name}. Task: {task}",
            "input": input_context,
            "output": response,
            "metadata": {
                "timestamp": self._timestamp(),
                "session_id": self.session_id,
                "agent": agent_name,
                "type": "agent_reasoning",
                **(metadata or {}),
            },
        }

        self.save_entry(entry)

    # ------------------------------------------------------------------
    # ✅ NEW: OPTIMISATION STEP CAPTURE
    # ------------------------------------------------------------------
    def capture_optimisation_step(
        self,
        population_summary: str,
        selected_sequences: List[str],
        mutation_bias: Dict,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        entry = {
            "instruction": "Analyse optimisation behaviour and sequence selection.",
            "input": population_summary,
            "output": json.dumps({
                "selected_sequences": selected_sequences,
                "mutation_bias": mutation_bias,
            }, ensure_ascii=False),
            "metadata": {
                "timestamp": self._timestamp(),
                "session_id": self.session_id,
                "type": "optimisation_step",
                **(metadata or {}),
            },
        }

        self.save_entry(entry)

    # ------------------------------------------------------------------
    # ✅ NEW: MUTATION PATTERN TRACKING
    # ------------------------------------------------------------------
    def capture_mutation_pattern(
        self,
        bias: dict,
        iteration: int,
    ) -> None:

        self.save_entry({
            "instruction": "Learn mutation bias patterns",
            "input": json.dumps(bias, ensure_ascii=False),
            "output": f"Iteration {iteration}",
            "metadata": {
                "timestamp": self._timestamp(),
                "session_id": self.session_id,
                "type": "mutation_pattern",
            },
        })

    # ------------------------------------------------------------------
    # ✅ NEW: PARETO SUMMARY
    # ------------------------------------------------------------------
    def summarise_population(self, population, decode_fn) -> str:
        lines = []

        for ind in population[:10]:
            seq = decode_fn(ind.X)
            obj = [-v for v in ind.F]

            lines.append(f"{seq} | {obj}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    def save_entry(self, entry: Dict[str, Any]) -> None:
        safe_entry = self._safe_jsonable(entry)

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(safe_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning("Failed to save training data entry: %s", e)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _safe_jsonable(self, obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, list):
            return [self._safe_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): self._safe_jsonable(v) for k, v in obj.items()}
        return repr(obj)