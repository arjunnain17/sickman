"""
insurance_agent.py
------------------
Insurance Agent for the Medical Report Simplifier pipeline.

Input:
  - MedicalOutput object (produced by medical_agent.py)
  - Path to the insurance policy PDF

Process:
  - Parses the insurance policy PDF via OpenDataLoader (format="markdown")
  - Retrieves top-6 relevant chunks from the FAISS index filtered to
    source_type="insurance", with priority on is_table_chunk=True chunks
  - For each FindingItem in MedicalOutput.findings, looks up what the
    policy covers and builds a CoverageItem (finding_name is a strict
    foreign key back to FindingItem.name)
  - Returns a validated InsuranceOutput Pydantic object
  - Saves insurance_output.json for inspection and handoff to Document Builder

Key constraints (do not change):
  - source_type="insurance" filter is mandatory on all FAISS retrieval calls
  - CoverageItem.finding_name must exactly match FindingItem.name
  - Temperature 0.1 for consistent structured output
  - OpenDataLoader with format="markdown" for all PDF parsing
"""

import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from pydantic import BaseModel, Field

# Reuse MedicalOutput schema for the input contract
from medical_agent import FindingItem, MedicalOutput

load_dotenv()

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class CoverageItem(BaseModel):
    finding_name: str = Field(
        ...,
        description=(
            "Must exactly match FindingItem.name from MedicalOutput — "
            "this is the foreign key linking findings to coverage rows."
        ),
    )
    covered: bool = Field(..., description="True if the policy covers this finding/procedure.")
    coverage_detail: str = Field(
        ...,
        description=(
            "Plain-language coverage summary, e.g. "
            "'Covered under Section 4B, max ₹80,000/year'."
        ),
    )
    pre_auth_required: bool = Field(
        ..., description="True if pre-authorisation is required before treatment/claim."
    )
    exclusions: str = Field(
        ...,
        description=(
            "Any exclusions or conditions that limit coverage. "
            "Use 'None' if there are no relevant exclusions."
        ),
    )


class ClaimStep(BaseModel):
    step_number: int = Field(..., description="Sequential step number starting at 1.")
    instruction: str = Field(
        ...,
        description=(
            "Plain-language instruction for the patient, e.g. "
            "'Submit Form XYZ with your discharge summary to your TPA office'."
        ),
    )
    deadline: str = Field(
        ...,
        description=(
            "Time limit for this step, e.g. 'Within 30 days of procedure'. "
            "Use 'No fixed deadline' if not specified in the policy."
        ),
    )


class InsuranceOutput(BaseModel):
    policy_holder_name: str = Field(..., description="Name of the insured person as stated in the policy.")
    policy_number: str = Field(..., description="Policy/certificate number.")
    insurer_name: str = Field(..., description="Name of the insurance company.")
    coverage: list[CoverageItem] = Field(
        ...,
        description="One CoverageItem per FindingItem in the MedicalOutput — no findings may be skipped.",
    )
    claim_steps: list[ClaimStep] = Field(
        ..., description="Ordered steps the patient must follow to file a claim."
    )
    documents_required: list[str] = Field(
        ...,
        description=(
            "Complete list of documents the patient must gather for the claim, "
            "e.g. ['Original discharge summary', 'Completed Claim Form A', 'Doctor's prescription']."
        ),
    )
    disclaimer: str = Field(
        ...,
        description=(
            "Non-negotiable disclaimer reminding the patient that this is an AI-generated "
            "summary and they should verify coverage details directly with their insurer."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAISS_INDEX_PATH = "faiss_index/medical_rag"  # same index, different filter
OUTPUT_PATH = "insurance_output.json"


def _load_policy_markdown(pdf_path: str) -> str:
    """Parse the insurance policy PDF to Markdown using OpenDataLoader."""
    loader = OpenDataLoaderPDFLoader(file_path=pdf_path, format="markdown")
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)


def _build_retriever(k_normal: int = 4, k_table: int = 2):
    """
    Returns two retrievers:
      - base_retriever: top-k insurance chunks (any type)
      - table_retriever: top-k insurance chunks where is_table_chunk=True
    Table chunks are prioritised because coverage limits live in policy tables.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,   
    )

    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k_normal,
            "filter": {"source_type": "insurance"},
        },
    )
    table_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k_table,
            "filter": {"source_type": "insurance", "is_table_chunk": True},
        },
    )
    return base_retriever, table_retriever


def _retrieve_policy_context(query: str, base_retriever, table_retriever) -> str:
    """
    Retrieves relevant insurance chunks for a given query.
    Table chunks are labelled [TABLE] so the LLM knows to treat them as
    authoritative for coverage limits and benefit schedules.
    """
    base_docs = base_retriever.invoke(query)
    table_docs = table_retriever.invoke(query)

    seen = set()
    chunks = []

    # Table chunks first — they carry coverage amounts and limits
    for doc in table_docs:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            chunks.append(f"[TABLE]\n{doc.page_content}")

    for doc in base_docs:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            chunks.append(doc.page_content)

    return "\n\n---\n\n".join(chunks)


def _build_prompt(
    policy_markdown: str,
    medical_output: MedicalOutput,
    policy_context: str,
) -> str:
    """Construct the prompt sent to Gemini."""

# NEW
    findings_list = "\n".join(
        f"  - {f.name} | value: {f.value} | flag: {f.flag} | "
        f"plain meaning: {f.plain_explanation}"
        for f in medical_output.findings
    )

    return f"""You are an expert insurance coverage analyst. Your job is to read a patient's
insurance policy and determine exactly what is covered for each of their medical findings.

## Patient findings that need coverage analysis
{findings_list}

Patient name: {medical_output.patient_name}
Report date: {medical_output.report_date}

---

## Full insurance policy text (parsed from PDF)
{policy_markdown}

---

## Relevant policy excerpts retrieved from the knowledge base
{policy_context}

---

## Your task
Analyse the policy and produce a structured InsuranceOutput object.

## Reasoning approach — CRITICAL, follow this exactly
Insurance policies cover CONDITIONS and TREATMENTS, not lab test names.
You must reason from finding → condition → coverage. Follow this process
for every single finding before checking the policy:

Step 1 — Infer the condition:
  - High Total Cholesterol + High LDL → Cardiovascular disease risk, dyslipidemia
  - Low Haemoglobin → Anaemia, may need iron therapy or transfusion
  - Abnormal WBC → Infection, immune disorder, may need hospitalization
  - High Blood Glucose / Urine Glucose → Diabetes mellitus management
  - High Creatinine / Urine Protein → Chronic kidney disease, renal treatment
  - Abnormal Platelets → Bleeding disorder, haematology treatment
  - Normal result (flag: "normal") → No condition to claim, skip coverage lookup

Step 2 — Search the policy for that CONDITION or its TREATMENT:
  - Look for the condition name, related procedures, hospitalization cover
  - Check the Scope of Coverage, Inclusions, and Benefit Schedule sections
  - A finding is only ❌ if the condition is EXPLICITLY EXCLUDED in the policy
  - "Not mentioned by test name" is NOT the same as "not covered"

Step 3 — Set covered_detail accordingly:
  - If covered: "Covered as [condition] under Section X, max ₹Y/year"
  - If normal: "No claim needed — result is within normal range"
  - If genuinely excluded: "Excluded under Section X — [reason]"
  - If policy is ambiguous: "Policy does not explicitly mention [condition].
    Likely covered under general hospitalization. Verify with insurer."

STRICT RULES:
1. `coverage` must contain exactly one CoverageItem for EVERY finding listed above.
2. `CoverageItem.finding_name` must be an exact character-for-character match of the
   finding name from the list above — do not paraphrase or abbreviate.
3. If a finding is not mentioned in the policy, set `covered: false` and explain why
   in `coverage_detail` (e.g. "Not listed in policy schedule; likely excluded as
   pre-existing condition — verify with insurer").
4. `claim_steps` must be specific to this policy and patient — not generic advice.
5. All language must be plain English a non-medical, non-legal reader can act on.
6. The `disclaimer` field must always read:
   "This is an AI-generated summary for informational purposes only. Coverage
   determinations are subject to the full policy terms and conditions. Always verify
   coverage details, pre-authorisation requirements, and claim procedures directly
   with your insurance provider or TPA before proceeding."
7. Do not invent coverage amounts not present in the policy text.
8. If finding.flag == "normal", always set covered=False and
   coverage_detail="No claim needed — result is within normal range."
   Do NOT look up coverage for normal findings.
9. Never mark a finding as not covered just because the lab test name
   doesn't appear in the policy. Always reason via the underlying condition.
"""


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

def run_insurance_agent(
    medical_output: MedicalOutput,
    policy_pdf_path: str,
    output_path: str = OUTPUT_PATH,
) -> InsuranceOutput:
    """
    Run the Insurance Agent.

    Args:
        medical_output: Validated MedicalOutput from medical_agent.py.
        policy_pdf_path: File path to the patient's insurance policy PDF.
        output_path: Where to write insurance_output.json.

    Returns:
        Validated InsuranceOutput Pydantic object.
    """
    print(f"[InsuranceAgent] Parsing policy PDF: {policy_pdf_path}")
    policy_markdown = _load_policy_markdown(policy_pdf_path)

    print("[InsuranceAgent] Building retrievers (source_type='insurance')...")
    base_retriever, table_retriever = _build_retriever()

    # Build a single composite query from all finding names so we pull
    # the most relevant coverage clauses in one retrieval pass.
    composite_query = (
        f"Insurance coverage for: "
        + ", ".join(f.name for f in medical_output.findings)
        + ". Claim procedure, pre-authorisation, benefit limits, exclusions."
    )
    print(f"[InsuranceAgent] Retrieving policy context for query: {composite_query[:80]}...")
    policy_context = _retrieve_policy_context(composite_query, base_retriever, table_retriever)

    print("[InsuranceAgent] Building prompt and calling Gemini...")
    prompt = _build_prompt(policy_markdown, medical_output, policy_context)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.1,
    )
    structured_llm = llm.with_structured_output(InsuranceOutput)

    insurance_output: InsuranceOutput = structured_llm.invoke(prompt)

    # ------------------------------------------------------------------
    # Validate foreign key integrity: every finding must have a coverage row
    # ------------------------------------------------------------------
    finding_names = {f.name for f in medical_output.findings}
    coverage_names = {c.finding_name for c in insurance_output.coverage}

    missing = finding_names - coverage_names
    if missing:
        raise ValueError(
            f"[InsuranceAgent] Foreign key violation — these findings have no "
            f"CoverageItem: {missing}. The LLM skipped them. Re-run or inspect the prompt."
        )

    extra = coverage_names - finding_names
    if extra:
        # Non-fatal: LLM added coverage rows for findings not in MedicalOutput.
        # Strip them to keep the contract clean.
        print(
            f"[InsuranceAgent] WARNING: LLM produced CoverageItems for unknown "
            f"findings {extra} — stripping them from output."
        )
        insurance_output.coverage = [
            c for c in insurance_output.coverage if c.finding_name in finding_names
        ]

    # ------------------------------------------------------------------
    # Persist output
    # ------------------------------------------------------------------
    output_file = Path(output_path)
    output_file.write_text(
        json.dumps(insurance_output.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[InsuranceAgent] Output saved → {output_file.resolve()}")

    return insurance_output


# ---------------------------------------------------------------------------
# CLI entry point (for standalone testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python insurance_agent.py <medical_output.json> <policy.pdf>")
        sys.exit(1)

    medical_json_path = sys.argv[1]
    policy_pdf = sys.argv[2]

    # Deserialise MedicalOutput from the JSON file produced by medical_agent.py
    with open(medical_json_path, encoding="utf-8") as f:
        raw = json.load(f)
    medical_output = MedicalOutput.model_validate(raw)

    result = run_insurance_agent(medical_output, policy_pdf)

    print("\n" + "=" * 60)
    print("INSURANCE AGENT OUTPUT")
    print("=" * 60)
    print(f"Policy holder : {result.policy_holder_name}")
    print(f"Policy number : {result.policy_number}")
    print(f"Insurer       : {result.insurer_name}")
    print(f"\nCoverage rows : {len(result.coverage)}")
    for item in result.coverage:
        status = "✅" if item.covered else "❌"
        auth = " [PRE-AUTH REQUIRED]" if item.pre_auth_required else ""
        print(f"  {status} {item.finding_name}{auth}")
        print(f"     {item.coverage_detail}")
    print(f"\nClaim steps   : {len(result.claim_steps)}")
    for step in result.claim_steps:
        print(f"  {step.step_number}. {step.instruction} ({step.deadline})")
    print(f"\nDocuments required:")
    for doc in result.documents_required:
        print(f"  • {doc}")
    print(f"\nDisclaimer: {result.disclaimer}")