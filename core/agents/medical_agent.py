"""
Medical Agent — Medical Report Simplifier
------------------------------------------
Takes a current medical report PDF, retrieves relevant past history
from FAISS, and returns a structured MedicalOutput Pydantic object.

Usage:
    from medical_agent import run_medical_agent
    result = run_medical_agent("path/to/current_report.pdf")
    print(result.model_dump_json(indent=2))

Requires:
    - faiss_index/ built by ingest.py
    - GOOGLE_API_KEY in .env
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL  = "gemini-embedding-001"
LLM_MODEL        = "gemini-2.5-flash"
INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "faiss_index"
FAISS_INDEX_NAME = "medical_rag"


# ── Output Schema ──────────────────────────────────────────────────────────────
# This is the contract between the Medical Agent and everything downstream.
# The Insurance Agent and Document Builder both consume this object.

class FindingItem(BaseModel):
    name: str = Field(
        description="Name of the test or clinical finding. E.g. 'Serum Creatinine'"
    )
    value: str = Field(
        description="Reported value with units. E.g. '2.1 mg/dL'"
    )
    reference_range: str = Field(
        description="Normal reference range from the report. E.g. '0.7–1.3 mg/dL'"
    )
    flag: Literal["critical", "monitor", "normal"] = Field(
        description=(
            "critical = significantly outside range, requires immediate attention. "
            "monitor = mildly abnormal, needs follow-up. "
            "normal = within reference range."
        )
    )
    plain_explanation: str = Field(
        description=(
            "2-sentence plain English explanation for a patient with no medical background. "
            "Do NOT use jargon. State what the test measures and what the result means for them."
        )
    )
    trend: str = Field(
        description=(
            "Comparison to past history retrieved from records. "
            "E.g. '↑ Higher than Dec 2024 result (1.6 mg/dL)'. "
            "If no past data found, write: 'No prior data available.'"
        )
    )


class MedicalOutput(BaseModel):
    patient_name: str = Field(
        description="Patient name as it appears in the report. 'Unknown' if not found."
    )
    report_date: str = Field(
        description="Date of the current report. 'Unknown' if not found."
    )
    attending_physician: str = Field(
        description="Attending doctor's name. 'Unknown' if not found."
    )
    summary: str = Field(
        description=(
            "3–4 sentence plain English overall summary of the report. "
            "Mention the most important findings and whether they are a cause for concern."
        )
    )
    findings: list[FindingItem] = Field(
        description="List of all significant findings from the report."
    )
    follow_up_actions: list[str] = Field(
        description=(
            "Recommended next steps in plain language. "
            "E.g. 'Schedule a kidney function follow-up in 4 weeks.'"
        )
    )
    referrals: list[str] = Field(
        description=(
            "Specialist referrals recommended. "
            "E.g. 'Nephrology referral recommended.' "
            "Empty list if none."
        )
    )
    disclaimer: str = Field(
        default=(
            "This summary is AI-generated for informational purposes only. "
            "Please consult your doctor before making any medical decisions."
        ),
        description="Always include this disclaimer in the output."
    )


# ── Step 1: Parse incoming PDF ─────────────────────────────────────────────────

def parse_current_report(pdf_path: str | Path) -> str:
    """
    Parse the current report PDF into Markdown text using OpenDataLoader.
    Same parser as ingestion — tables come out as Markdown, headings preserved.
    """
    loader = OpenDataLoaderPDFLoader(
        file_path=str(pdf_path),
        format="markdown",
        quiet=True,
    )
    pages = loader.load()
    # Concatenate all pages into one string — current report is the query, not chunked
    full_text = "\n\n".join(p.page_content for p in pages if p.page_content.strip())
    return full_text


# ── Step 2: Retrieve past history ─────────────────────────────────────────────

def retrieve_past_history(query_text: str, k: int = 6) -> str:
    """
    Retrieve the most relevant chunks from past medical history in FAISS.
    Filters to source_type='medical' so insurance docs never bleed in.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        str(INDEX_DIR / FAISS_INDEX_NAME),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"source_type": "medical"},
        },
    )
    docs = retriever.invoke(query_text)

    if not docs:
        return "No relevant past medical history found."

    # Format retrieved chunks with source attribution for the LLM
    sections = []
    for i, doc in enumerate(docs):
        fname = doc.metadata.get("file_name", "unknown")
        page  = doc.metadata.get("page", "?")
        sections.append(f"[Past Record {i+1} | {fname} p{page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(sections)


# ── Step 3: Build prompt ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior medical professional helping patients understand their medical reports.

Your job:
1. Analyze the CURRENT MEDICAL REPORT provided.
2. Use the PAST MEDICAL HISTORY to identify trends (better, worse, unchanged).
3. Return a structured analysis following the exact schema provided.

Rules:
- Flag findings as 'critical' only if they are significantly outside the reference range.
- Every plain_explanation must be written for a patient with zero medical knowledge.
- For trends, always cite the specific past value and date if available in past history.
- If a critical flag is assigned, the plain_explanation must include: "Please discuss this with your doctor immediately."
- Do not invent values. If something is not in the report, say so honestly.
- follow_up_actions and referrals must be concrete and actionable, not generic.
"""

HUMAN_PROMPT = """## Current Medical Report
{current_report}

---

## Past Medical History (retrieved from records)
{past_history}

---

Analyze the current report in context of the past history and return the structured MedicalOutput.
"""

def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])


# ── Step 4: Run the agent ──────────────────────────────────────────────────────

def run_medical_agent(pdf_path: str | Path) -> MedicalOutput:
    """
    Full pipeline: PDF → parse → retrieve → LLM → MedicalOutput

    Parameters
    ----------
    pdf_path : path to the current medical report PDF

    Returns
    -------
    MedicalOutput : validated Pydantic object ready for the Insurance Agent
                    and Document Builder
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Report not found: {pdf_path}")

    print(f"[Medical Agent] Parsing {pdf_path.name} ...")
    current_report = parse_current_report(pdf_path)

    print(f"[Medical Agent] Retrieving past history ...")
    past_history = retrieve_past_history(current_report)

    print(f"[Medical Agent] Analyzing with {LLM_MODEL} ...")
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.1,   # low temp = consistent structured output
    )
    structured_llm = llm.with_structured_output(MedicalOutput)

    prompt   = build_prompt()
    chain    = prompt | structured_llm
    result   = chain.invoke({
        "current_report": current_report,
        "past_history":   past_history,
    })

    print(f"[Medical Agent] Done — {len(result.findings)} findings extracted.")
    return result


# ── CLI entrypoint (for testing) ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python medical_agent.py path/to/report.pdf")
        sys.exit(1)

    output = run_medical_agent(sys.argv[1])

    # Pretty print the result
    print("\n" + "=" * 52)
    print("  Medical Agent Output")
    print("=" * 52)
    print(output.model_dump_json(indent=2))

    # Save to file for inspection
    out_path = Path("medical_output.json")
    out_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n✓ Saved to {out_path}")