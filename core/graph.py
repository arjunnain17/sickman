"""
graph.py — LangGraph Orchestrator
-----------------------------------
Wires the pipeline: Medical Agent → Insurance Agent

AgentState keys (must match api.py exactly — do not add extra keys):
    report_pdf_path  : str
    policy_pdf_path  : str
    medical_output   : Optional[MedicalOutput]
    insurance_output : Optional[InsuranceOutput]
    error            : Optional[str]

Flow:
    START → medical_agent → [conditional] → insurance_agent → END
                                         ↘ END  (if medical failed)

Run standalone:
    python core/graph.py <report.pdf> <policy.pdf>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, TypedDict

# ── Ensure project root on sys.path ───────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env")

from langgraph.graph import END, START, StateGraph

from core.agents.medical_agent import MedicalOutput, run_medical_agent
from core.agents.insurance_agent import InsuranceOutput, run_insurance_agent


# ── Shared state ──────────────────────────────────────────────────────────────
# ONLY these five keys. No current_step, no errors list, no final_document.
# api.py builds the initial state with exactly these keys.

class AgentState(TypedDict):
    report_pdf_path:  str
    policy_pdf_path:  str
    medical_output:   Optional[MedicalOutput]
    insurance_output: Optional[InsuranceOutput]
    error:            Optional[str]


# ── Node 1: Medical Agent ─────────────────────────────────────────────────────

def medical_agent_node(state: AgentState) -> AgentState:
    print("[Orchestrator] → Running Medical Agent ...")
    try:
        result = run_medical_agent(state["report_pdf_path"])
        print(f"[Orchestrator] ✓ Medical Agent complete — {len(result.findings)} findings")
        return {
            **state,
            "medical_output": result,
            "error": None,
        }
    except Exception as exc:
        msg = f"Medical Agent failed: {exc}"
        print(f"[Orchestrator] ✗ {msg}")
        return {
            **state,
            "medical_output": None,
            "error": msg,
        }


# ── Node 2: Insurance Agent ───────────────────────────────────────────────────

def insurance_agent_node(state: AgentState) -> AgentState:
    print("[Orchestrator] → Running Insurance Agent ...")
    try:
        result = run_insurance_agent(
            medical_output=state["medical_output"],
            policy_pdf_path=state["policy_pdf_path"],
        )
        covered = sum(1 for c in result.coverage if c.covered)
        print(f"[Orchestrator] ✓ Insurance Agent complete — {covered}/{len(result.coverage)} findings covered")
        return {
            **state,
            "insurance_output": result,
            "error": None,
        }
    except Exception as exc:
        msg = f"Insurance Agent failed: {exc}"
        print(f"[Orchestrator] ✗ {msg}")
        # Non-fatal — medical output is still valid, insurance is just None
        return {
            **state,
            "insurance_output": None,
            "error": msg,
        }


# ── Conditional edge ──────────────────────────────────────────────────────────
# After medical_agent_node: if medical failed, skip insurance and go to END.

def after_medical_agent(state: AgentState) -> str:
    if state.get("error") or state.get("medical_output") is None:
        print("[Orchestrator] ✗ Medical Agent failed — skipping Insurance Agent.")
        return "end"
    return "insurance"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("medical_agent",    medical_agent_node)
    builder.add_node("insurance_agent",  insurance_agent_node)

    builder.add_edge(START, "medical_agent")

    builder.add_conditional_edges(
        "medical_agent",
        after_medical_agent,
        {
            "insurance": "insurance_agent",
            "end":       END,
        },
    )

    builder.add_edge("insurance_agent", END)

    return builder.compile()


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python core/graph.py <report.pdf> <policy.pdf>")
        sys.exit(1)

    graph = build_graph()
    initial_state: AgentState = {
        "report_pdf_path":  sys.argv[1],
        "policy_pdf_path":  sys.argv[2],
        "medical_output":   None,
        "insurance_output": None,
        "error":            None,
    }

    print("[Graph] Starting pipeline ...")
    final = graph.invoke(initial_state)
    print("[Graph] Pipeline complete.")

    if final.get("error") and final.get("medical_output") is None:
        print(f"\n✗ Fatal error: {final['error']}")
        sys.exit(1)

    med = final["medical_output"]
    ins = final["insurance_output"]

    print(f"\n✓ Medical  : {len(med.findings)} findings for {med.patient_name}")
    if ins:
        covered = sum(1 for c in ins.coverage if c.covered)
        print(f"✓ Insurance: {covered}/{len(ins.coverage)} findings covered")
    else:
        print(f"⚠ Insurance: not available — {final.get('error')}")