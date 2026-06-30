from __future__ import annotations

from langgraph.graph import END, StateGraph

from VLAB2.orchestration.state_schema import LabState

from VLAB2.orchestration.agents.pi_agent import pi_agent
from VLAB2.orchestration.agents.researcher_agent import researcher_agent
from VLAB2.orchestration.agents.bioinfo_agent import bioinfo_agent
from VLAB2.orchestration.agents.structural_agent import structural_agent
from VLAB2.orchestration.agents.md_agent import md_agent
from VLAB2.orchestration.agents.protein_agent import protein_agent
from VLAB2.orchestration.agents.inhibitor_agent import inhibitor_agent
from VLAB2.orchestration.agents.skeptic_agent import skeptic_agent

from VLAB2.orchestration.routing import should_continue, _inhibitor_should_run


def build_virtual_lab():
    """
    Build and compile the LangGraph Virtual Lab workflow.
    """
    workflow = StateGraph(LabState)

    workflow.add_node("pi", pi_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("bioinfo", bioinfo_agent)
    workflow.add_node("structural", structural_agent)
    workflow.add_node("md", md_agent)
    workflow.add_node("protein", protein_agent)
    workflow.add_node("inhibitor", inhibitor_agent)
    workflow.add_node("skeptic", skeptic_agent)

    workflow.set_entry_point("pi")

    workflow.add_edge("pi", "researcher")
    workflow.add_edge("researcher", "bioinfo")
    workflow.add_edge("bioinfo", "structural")
    workflow.add_edge("structural", "md")
    workflow.add_edge("md", "protein")

    # Conditional edge: run inhibitor screening if enabled, else skip to skeptic
    workflow.add_conditional_edges(
        "protein",
        _inhibitor_should_run,
        {
            "inhibitor": "inhibitor",
            "skeptic": "skeptic",
        },
    )

    workflow.add_edge("inhibitor", "skeptic")

    workflow.add_conditional_edges(
        "skeptic",
        should_continue,
        {
            "continue": "pi",
            "end": END,
        },
    )

    return workflow.compile()


virtual_lab = build_virtual_lab()