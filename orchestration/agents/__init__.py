"""
Virtual Lab agent modules.

Each agent exposes one LangGraph-compatible function:

    agent_name(state: LabState) -> dict
"""

from VLAB2.orchestration.agents.pi_agent import pi_agent
from VLAB2.orchestration.agents.researcher_agent import researcher_agent
from VLAB2.orchestration.agents.bioinfo_agent import bioinfo_agent
from VLAB2.orchestration.agents.structural_agent import structural_agent
from VLAB2.orchestration.agents.md_agent import md_agent
from VLAB2.orchestration.agents.protein_agent import protein_agent
from VLAB2.orchestration.agents.skeptic_agent import skeptic_agent

__all__ = [
    "pi_agent",
    "researcher_agent",
    "bioinfo_agent",
    "structural_agent",
    "md_agent",
    "protein_agent",
    "skeptic_agent",
]