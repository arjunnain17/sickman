"""
Orchestrator — Medical Report Simplifier
-----------------------------------------
LangGraph StateGraph that wires together:
  1. Medical Agent
  2. Insurance Agent (stub until teammate's implementation is ready)
  3. Document Builder (stub until built)

Usage:
    from graph import run_pipeline
    result = run_pipeline("data/medical/current_report.pdf")
    print(result["final_document"])

Requires:
    pip install langgraph
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, END

from medical_agent import run_medical_agent, MedicalOutput
from insurance_agent import run_insurance_agent, InsuranceOutput


# ── Shared State ───────────────────────────────────────────────────────────────
# This is the single object that flows through the entire pipeline.
# Every node reads from it and writes back to it.
# No agent talks directly to another — everything goes through state.

class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    current_report_path: str        # path to the uploaded current report PDF
    policy_pdf_path: str        

    # ── Agent outputs (populated as pipeline runs) ─────────────────
    medical_output: MedicalOutput | None
    insurance_output: InsuranceOutput | None    
    final_document: str | None

    # ── Orchestrator bookkeeping ───────────────────────────────────
    current_step: str
    errors: list[str]


# ── Node 1: Medical Agent ──────────────────────────────────────────────────────

def medical_agent_node(state: AgentState) -> AgentState:
    """
    Runs the Medical Agent on the current report PDF.
    Populates state["medical_output"] on success.
    Populates state["errors"] on failure and sets medical_output to None.
    """
    print("\n[Orchestrator] → Running Medical Agent ...")
    try:
        result = run_medical_agent(state["current_report_path"])
        print(f"[Orchestrator] ✓ Medical Agent complete — {len(result.findings)} findings")
        return {
            **state,
            "medical_output": result,
            "current_step": "medical_done",
        }
    except Exception as e:
        error_msg = f"Medical Agent failed: {str(e)}"
        print(f"[Orchestrator] ✗ {error_msg}")
        return {
            **state,
            "medical_output": None,
            "current_step": "error",
            "errors": state["errors"] + [error_msg],
        }


# ── Node 2: Insurance Agent ────────────────────────────────────────────────────
# STUB — replace the body of this function with your teammate's implementation.
# The function signature and return shape must stay exactly the same.

def insurance_agent_node(state: AgentState) -> AgentState:
    """
    Runs the Insurance Agent using the Medical Agent's findings.
    """
    print("\n[Orchestrator] → Running Insurance Agent ...")
    try:
        result = run_insurance_agent(
            medical_output=state["medical_output"],
            policy_pdf_path=state["policy_pdf_path"],  # adjust to your exact filename
        )
        print(f"[Orchestrator] ✓ Insurance Agent complete — {len(result.coverage)} coverage items")
        return {
            **state,
            "insurance_output": result,
            "current_step": "insurance_done",
        }
    except Exception as e:
        error_msg = f"Insurance Agent failed: {str(e)}"
        print(f"[Orchestrator] ✗ {error_msg}")
        return {
            **state,
            "insurance_output": None,
            "current_step": "error",
            "errors": state["errors"] + [error_msg],
        }

# ── Node 3: Document Builder ───────────────────────────────────────────────────
# Add import at top of graph.py
from document_builder import build_document

# Replace document_builder_node body
def document_builder_node(state: AgentState) -> AgentState:
    print("\n[Orchestrator] → Running Document Builder ...")
    try:
        out_path = build_document(
            medical=state["medical_output"],
            insurance=state["insurance_output"],
            output_path="final_report.pdf",
        )
        final_doc = str(out_path)
        print(f"[Orchestrator] ✓ Document built → {final_doc}")
        return {**state, "final_document": final_doc, "current_step": "done"}
    except Exception as e:
        error_msg = f"Document Builder failed: {str(e)}"
        print(f"[Orchestrator] ✗ {error_msg}")
        return {**state, "final_document": None, "current_step": "error",
                "errors": state["errors"] + [error_msg]}

# ── Node 4: Error Handler ──────────────────────────────────────────────────────

def error_handler_node(state: AgentState) -> AgentState:
    """
    Called when any agent fails. Logs errors and sets a user-facing message.
    """
    print("\n[Orchestrator] ✗ Pipeline failed. Errors:")
    for err in state["errors"]:
        print(f"    - {err}")

    error_doc = (
        "# Report Generation Failed\n\n"
        "An error occurred while processing your report.\n\n"
        "**Errors:**\n"
        + "\n".join(f"- {e}" for e in state["errors"])
    )
    return {
        **state,
        "final_document": error_doc,
        "current_step": "error_handled",
    }


# ── Routing Functions ──────────────────────────────────────────────────────────
# These tell LangGraph which node to go to next based on state.

def after_medical_agent(state: AgentState) -> str:
    """Route after Medical Agent — go to Insurance Agent or Error Handler."""
    if state["current_step"] == "error" or state["medical_output"] is None:
        return "error_handler"
    return "insurance_agent"


def after_insurance_agent(state: AgentState) -> str:
    """Route after Insurance Agent — go to Document Builder or Error Handler."""
    if state["current_step"] == "error" or state["insurance_output"] is None:
        return "error_handler"
    return "document_builder"


# ── Build the Graph ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("medical_agent",    medical_agent_node)
    graph.add_node("insurance_agent",  insurance_agent_node)
    graph.add_node("document_builder", document_builder_node)
    graph.add_node("error_handler",    error_handler_node)

    # Entry point
    graph.set_entry_point("medical_agent")

    # Edges with conditional routing
    graph.add_conditional_edges("medical_agent",   after_medical_agent)
    graph.add_conditional_edges("insurance_agent", after_insurance_agent)

    # Terminal edges
    graph.add_edge("document_builder", END)
    graph.add_edge("error_handler",    END)

    return graph.compile()


# ── Public entry point ─────────────────────────────────────────────────────────

def run_pipeline(report_pdf_path: str | Path) -> AgentState:
    """
    Run the full pipeline on a current medical report PDF.

    Parameters
    ----------
    report_pdf_path : path to the current report PDF

    Returns
    -------
    AgentState : final state containing medical_output, insurance_output,
                 final_document, and any errors
    """

    insurance_dir = Path("data/insurance")
    policy_files = list(insurance_dir.glob("*.pdf"))

    if not policy_files:
        raise FileNotFoundError("No policy PDF found in data/insurance/")
    if len(policy_files) > 1:
        print(f"[Orchestrator] Multiple policy PDFs found, using: {policy_files[0].name}")

    policy_pdf_path = policy_files[0]

    initial_state: AgentState = {
        "current_report_path": str(report_pdf_path),
        "policy_pdf_path":     str(policy_pdf_path), 
        "medical_output":      None,
        "insurance_output":    None,
        "final_document":      None,
        "current_step":        "start",
        "errors":              [],
    }

    graph = build_graph()

    print("=" * 52)
    print("  Medical Report Simplifier — Pipeline Start")
    print("=" * 52)

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 52)
    print(f"  Pipeline {('complete ✓' if not final_state['errors'] else 'failed ✗')}")
    print("=" * 52)

    return final_state


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python graph.py path/to/current_report.pdf")
        sys.exit(1)

    result = run_pipeline(sys.argv[1])

    # Save final document
    if result["final_document"]:
        out_path = Path("final_report.md")
        out_path.write_text(result["final_document"], encoding="utf-8")
        print(f"\n✓ Final report saved → {out_path}")

    # Save full state for debugging
    import json
    state_path = Path("pipeline_state.json")
    serializable = {
        k: (v.model_dump() if hasattr(v, "model_dump") else v)
        for k, v in result.items()
    }
    state_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ Full pipeline state saved → {state_path}")