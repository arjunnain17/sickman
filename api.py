"""
api.py — FastAPI Backend for Medical Report Simplifier
--------------------------------------------------------
Exposes the full LangGraph pipeline as a REST API.

Project structure expected:
    sickman/
    ├── api.py                          ← this file (always run from here)
    ├── .env
    ├── faiss_index/
    ├── data/
    └── core/
        ├── __init__.py
        ├── graph.py
        ├── ingest.py
        └── agents/
            ├── __init__.py
            ├── medical_agent.py
            ├── insurance_agent.py
            └── document_builder.py

Endpoints:
    GET  /health                — liveness check
    POST /ingest                — trigger FAISS index rebuild (background)
    POST /ingest/upload         — upload past PDFs + rebuild index (synchronous)
    POST /analyze               — upload 2 PDFs → full JSON response
    POST /analyze/download/json — upload 2 PDFs → download JSON file
    POST /analyze/download/pdf  — upload 2 PDFs → download rendered PDF
    GET  /docs                  — Swagger UI (built into FastAPI)

Run from project root:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, List, Optional

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Environment ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env")

# ── Internal imports ──────────────────────────────────────────────────────────
from core.graph import AgentState, build_graph
from core.agents.medical_agent import MedicalOutput
from core.agents.insurance_agent import InsuranceOutput
from core.agents.document_builder import build_document

# ── FastAPI imports ───────────────────────────────────────────────────────────
# BackgroundTask  (singular) — runs a single callable after a response streams.
# BackgroundTasks (plural)   — dependency-injection class for route parameters.
# These are DIFFERENT classes. Only BackgroundTask belongs in FileResponse.
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from starlette.background import BackgroundTask        # ← singular, for FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Medical Report Simplifier",
    description=(
        "Upload a current medical report PDF and an insurance policy PDF. "
        "Runs Medical Agent → Insurance Agent via LangGraph and returns a "
        "plain-English analysis with full insurance coverage details."
    ),
    version="1.0.0",
)

# NOTE: allow_credentials MUST be False when allow_origins=["*"].
# Combining credentials=True with wildcard origins violates the CORS spec
# and causes browsers to silently block all requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_pdf(upload: UploadFile, field_name: str) -> None:
    """Reject non-PDF uploads before touching the pipeline."""
    if not upload.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"'{field_name}' must be a .pdf file. Got: {upload.filename!r}",
        )


def _save_upload(upload: UploadFile, dest: Path) -> None:
    """Stream an uploaded file to disk without loading it fully into memory."""
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def _run_pipeline(report_path: Path, policy_path: Path) -> AgentState:
    """
    Compile and invoke the LangGraph pipeline.
    Returns the final AgentState with medical_output and insurance_output populated.
    Sequential: Medical Agent fires first, Insurance Agent is downstream.
    """
    graph = build_graph()
    initial_state: AgentState = {
        "report_pdf_path":  str(report_path),
        "policy_pdf_path":  str(policy_path),
        "medical_output":   None,
        "insurance_output": None,
        "error":            None,
    }
    return graph.invoke(initial_state)


def _check_state(state: AgentState) -> tuple[MedicalOutput, Optional[InsuranceOutput]]:
    """
    Extract and validate outputs from the final graph state.
    - Medical failure  → hard HTTPException (nothing usable).
    - Insurance failure → soft, returns None (medical output still served).
    """
    if state.get("error") and state.get("medical_output") is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {state['error']}",
        )

    medical: MedicalOutput = state.get("medical_output")
    insurance: Optional[InsuranceOutput] = state.get("insurance_output")

    if medical is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Medical agent returned no output. Check server logs.",
        )

    return medical, insurance


def _make_temp_file(suffix: str) -> Path:
    """
    Create a named temp file that persists on disk until explicitly deleted.
    NamedTemporaryFile(delete=False) actually creates the file — unlike
    tempfile.mktemp() which only returns a path string and can race.
    Pair with BackgroundTask(os.unlink, path) to delete after streaming.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    return Path(tmp.name)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health_check():
    """Liveness probe — returns 200 if the server is up."""
    return {
        "status": "ok",
        "service": "Medical Report Simplifier",
        "version": "1.0.0",
    }


@app.post("/ingest", tags=["Admin"])
def ingest(
    background_tasks: BackgroundTasks,
    medical_dir: str = "data/medical",
    insurance_dir: str = "data/insurance",
):
    """
    Trigger a FAISS index rebuild in the background.
    Drop PDFs into data/medical/ and data/insurance/ first, then call this.

    Query params:
        medical_dir   — relative path to medical PDFs   (default: data/medical)
        insurance_dir — relative path to insurance PDFs (default: data/insurance)
    """
    def _run():
        from core.ingest import run_ingest          # lazy import
        run_ingest(
            medical_dir=ROOT / medical_dir,
            insurance_dir=ROOT / insurance_dir,
        )

    background_tasks.add_task(_run)
    return {
        "status": "accepted",
        "message": "Ingestion started in background. Check server logs for progress.",
        "medical_dir": str(ROOT / medical_dir),
        "insurance_dir": str(ROOT / insurance_dir),
    }


@app.post("/ingest/upload", tags=["Admin"])
async def ingest_upload(files: List[UploadFile] = File(...)):
    """
    Receive past medical report PDFs from the frontend, save them to
    data/medical/, then rebuild the FAISS index synchronously.
    Called by the frontend BEFORE /analyze when past reports are uploaded.
    Must be synchronous so the index is ready before /analyze fires.
    """
    medical_dir = ROOT / "data" / "medical"
    medical_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        dest = medical_dir / f.filename
        _save_upload(f, dest)
        saved.append(f.filename)
        print(f"[Ingest] Saved: {dest}")

    if saved:
        from core.ingest import run_ingest
        run_ingest(
            medical_dir=medical_dir,
            insurance_dir=ROOT / "data" / "insurance",
        )
        print(f"[Ingest] Index rebuilt — {len(saved)} new file(s): {saved}")

    return {"status": "ok", "ingested": saved}


@app.post("/analyze", tags=["Analysis"])
async def analyze(
    report: UploadFile = File(..., description="Current medical report PDF"),
    policy: UploadFile = File(..., description="Insurance policy PDF"),
) -> JSONResponse:
    """
    Run the full pipeline on the uploaded PDFs and return structured JSON.

    Response JSON keys:
        medical          — MedicalOutput  (findings, trends, follow-up, referrals)
        insurance        — InsuranceOutput (coverage, claim steps, documents required)
                           null if insurance agent failed
        pipeline_warning — error string if insurance agent failed, null otherwise
    """
    _validate_pdf(report, "report")
    _validate_pdf(policy, "policy")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        report_path = tmp / report.filename
        policy_path = tmp / policy.filename

        _save_upload(report, report_path)
        _save_upload(policy, policy_path)

        try:
            state = _run_pipeline(report_path, policy_path)
        except HTTPException:
            raise
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected pipeline error: {exc}",
            )

        medical, insurance = _check_state(state)

        payload: dict[str, Any] = {
            "medical":          medical.model_dump(),
            "insurance":        insurance.model_dump() if insurance else None,
            "pipeline_warning": state.get("error") if insurance is None else None,
        }

        return JSONResponse(content=payload)


@app.post("/analyze/download/json", tags=["Analysis"])
async def analyze_download_json(
    report: UploadFile = File(..., description="Current medical report PDF"),
    policy: UploadFile = File(..., description="Insurance policy PDF"),
) -> FileResponse:
    """
    Run the full pipeline and return the structured analysis as a
    downloadable JSON file. Useful for archiving or feeding into other systems.
    """
    _validate_pdf(report, "report")
    _validate_pdf(policy, "policy")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        report_path = tmp / report.filename
        policy_path = tmp / policy.filename

        _save_upload(report, report_path)
        _save_upload(policy, policy_path)

        try:
            state = _run_pipeline(report_path, policy_path)
        except HTTPException:
            raise
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected pipeline error: {exc}",
            )

        medical, insurance = _check_state(state)

        payload: dict[str, Any] = {
            "medical":   medical.model_dump(),
            "insurance": insurance.model_dump() if insurance else None,
        }

        out_file = _make_temp_file(".json")
        out_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        patient_name = (medical.patient_name or "patient").replace(" ", "_")

        return FileResponse(
            path=str(out_file),
            media_type="application/json",
            filename=f"{patient_name}_analysis.json",
            background=BackgroundTask(os.unlink, str(out_file)),  # ← singular
        )


@app.post("/analyze/download/pdf", tags=["Analysis"])
async def analyze_download_pdf(
    report: UploadFile = File(..., description="Current medical report PDF"),
    policy: UploadFile = File(..., description="Insurance policy PDF"),
) -> FileResponse:
    """
    Run the full pipeline and return a rendered, patient-friendly PDF report.

    PDF is built by core/agents/document_builder.py — pure deterministic
    ReportLab rendering, zero LLM calls.

    Requires insurance_output to be available (document_builder needs both).
    If the insurance agent failed, returns HTTP 502.
    Use POST /analyze/download/json for a medical-only JSON fallback.
    """
    _validate_pdf(report, "report")
    _validate_pdf(policy, "policy")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        report_path = tmp / report.filename
        policy_path = tmp / policy.filename

        _save_upload(report, report_path)
        _save_upload(policy, policy_path)

        try:
            state = _run_pipeline(report_path, policy_path)
        except HTTPException:
            raise
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected pipeline error: {exc}",
            )

        medical, insurance = _check_state(state)

        if insurance is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "Insurance agent failed — PDF requires coverage data. "
                    f"Pipeline warning: {state.get('error')}. "
                    "Use POST /analyze/download/json for a medical-only JSON export."
                ),
            )

        # Render PDF — _make_temp_file creates the file on disk so ReportLab
        # can write to it. BackgroundTask deletes it only after streaming ends.
        out_pdf = _make_temp_file(".pdf")
        try:
            build_document(medical, insurance, output_path=out_pdf)
        except Exception as exc:
            os.unlink(str(out_pdf))       # clean up immediately on render failure
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"PDF rendering failed: {exc}",
            )

        patient_name = (medical.patient_name or "patient").replace(" ", "_")
        filename     = f"{patient_name}_medical_summary.pdf"

        print(f"[API] Streaming PDF: {out_pdf} ({out_pdf.stat().st_size} bytes) as {filename}")

        return FileResponse(
            path=str(out_pdf),
            media_type="application/pdf",
            filename=filename,
            background=BackgroundTask(os.unlink, str(out_pdf)),   # ← singular
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)