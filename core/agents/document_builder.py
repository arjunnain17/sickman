"""
Document Builder — Medical Report Simplifier
---------------------------------------------
Takes MedicalOutput + InsuranceOutput and renders a clean,
patient-friendly PDF report using ReportLab.

No LLM calls — purely deterministic template rendering.

Install:
    pip install reportlab

Usage:
    from document_builder import build_document
    build_document(medical_output, insurance_output, "final_report.pdf")

Or standalone:
    python document_builder.py medical_output.json insurance_output.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from core.agents.medical_agent import MedicalOutput, FindingItem
from core.agents.insurance_agent import InsuranceOutput, CoverageItem


# ── Font registration ──────────────────────────────────────────────────────────
# Arial ships with Windows and supports Unicode (₹, ↑, ↓, etc.)
# Falls back to Helvetica if not found

def _register_fonts():
    paths = {
        "regular": r"C:\Windows\Fonts\arial.ttf",
        "bold":    r"C:\Windows\Fonts\arialbd.ttf",
        "italic":  r"C:\Windows\Fonts\ariali.ttf",
    }
    if all(os.path.exists(p) for p in paths.values()):
        pdfmetrics.registerFont(TTFont("DocFont",       paths["regular"]))
        pdfmetrics.registerFont(TTFont("DocFont-Bold",  paths["bold"]))
        pdfmetrics.registerFont(TTFont("DocFont-Italic",paths["italic"]))
        return "DocFont", "DocFont-Bold", "DocFont-Italic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"

FONT, FONT_BOLD, FONT_ITALIC = _register_fonts()


# ── Sanitize text ──────────────────────────────────────────────────────────────
# Fallback for any character the font can't render

def sanitize(text: str) -> str:
    if FONT != "Helvetica":
        return text  # Unicode font loaded — no sanitization needed
    return (text
        .replace("₹", "Rs.")
        .replace("\u2191", "^")
        .replace("\u2193", "v")
        .replace("\u2013", "-")
        .replace("\u2014", "--")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


# ── Colours ────────────────────────────────────────────────────────────────────

C_CRITICAL    = colors.HexColor("#C0392B")
C_CRITICAL_BG = colors.HexColor("#FDEDEC")
C_MONITOR     = colors.HexColor("#D35400")
C_MONITOR_BG  = colors.HexColor("#FEF9E7")
C_NORMAL      = colors.HexColor("#1E8449")
C_NORMAL_BG   = colors.HexColor("#EAFAF1")
C_COVERED_BG  = colors.HexColor("#EBF5FB")
C_COVERED     = colors.HexColor("#1A5276")
C_HEADER_BG   = colors.HexColor("#1A252F")
C_SECTION_BG  = colors.HexColor("#F2F3F4")
C_BORDER      = colors.HexColor("#BDC3C7")
C_TEXT        = colors.HexColor("#2C3E50")
C_MUTED       = colors.HexColor("#7F8C8D")

FLAG_COLORS = {
    "critical": (C_CRITICAL, C_CRITICAL_BG),
    "monitor":  (C_MONITOR,  C_MONITOR_BG),
    "normal":   (C_NORMAL,   C_NORMAL_BG),
}
FLAG_LABELS = {
    "critical": "CRITICAL",
    "monitor":  "MONITOR",
    "normal":   "NORMAL",
}

# Page width minus margins (20mm left + 20mm right)
CONTENT_WIDTH = A4[0] - 40 * mm   # 170mm


# ── Styles ─────────────────────────────────────────────────────────────────────

def build_styles() -> dict:
    def s(name, **kw) -> ParagraphStyle:
        return ParagraphStyle(name, **kw)

    return {
        "title": s(
            "DocTitle",
            fontName=FONT_BOLD, fontSize=20,
            textColor=colors.white, alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": s(
            "DocSubtitle",
            fontName=FONT, fontSize=10,
            textColor=colors.HexColor("#BDC3C7"), alignment=TA_CENTER,
        ),
        "section_header": s(
            "SectionHeader",
            fontName=FONT_BOLD, fontSize=10,
            textColor=C_TEXT, spaceAfter=0,
        ),
        "field_label": s(
            "FieldLabel",
            fontName=FONT_BOLD, fontSize=8,
            textColor=C_MUTED, spaceAfter=1,
        ),
        "field_value": s(
            "FieldValue",
            fontName=FONT, fontSize=10,
            textColor=C_TEXT, spaceAfter=2,
        ),
        "body": s(
            "Body",
            fontName=FONT, fontSize=10,
            textColor=C_TEXT, leading=15, spaceAfter=4,
        ),
        "finding_name": s(
            "FindingName",
            fontName=FONT_BOLD, fontSize=11,
            textColor=C_TEXT, spaceAfter=2,
        ),
        "finding_meta": s(
            "FindingMeta",
            fontName=FONT, fontSize=9,
            textColor=C_MUTED, spaceAfter=3,
        ),
        "finding_explanation": s(
            "FindingExplanation",
            fontName=FONT, fontSize=10,
            textColor=C_TEXT, leading=14, spaceAfter=4,
        ),
        "trend": s(
            "Trend",
            fontName=FONT_ITALIC, fontSize=9,
            textColor=C_MUTED, leading=13, spaceAfter=4,
        ),
        "coverage_status": s(
            "CoverageStatus",
            fontName=FONT_BOLD, fontSize=9,
            spaceAfter=2,
        ),
        "coverage_detail": s(
            "CoverageDetail",
            fontName=FONT, fontSize=9,
            textColor=C_TEXT, leading=13,
        ),
        "bullet": s(
            "Bullet",
            fontName=FONT, fontSize=10,
            textColor=C_TEXT, leading=15,
            leftIndent=10, spaceAfter=3,
        ),
        "step_text": s(
            "StepText",
            fontName=FONT, fontSize=10,
            textColor=C_TEXT, leading=14,
        ),
        "step_deadline": s(
            "StepDeadline",
            fontName=FONT_ITALIC, fontSize=8,
            textColor=C_MUTED, spaceAfter=0,
        ),
        "disclaimer": s(
            "Disclaimer",
            fontName=FONT_ITALIC, fontSize=8,
            textColor=C_MUTED, leading=12, alignment=TA_CENTER,
            spaceBefore=6,
        ),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def section_bar(title: str, styles: dict) -> list:
    """Shaded section header bar."""
    tbl = Table(
        [[Paragraph(title.upper(), styles["section_header"])]],
        colWidths=[CONTENT_WIDTH],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_SECTION_BG),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return [tbl, Spacer(1, 6)]


# ── Header ─────────────────────────────────────────────────────────────────────

def build_header(medical: MedicalOutput, insurance: InsuranceOutput, styles: dict) -> list:
    elements = []

    # Title bar
    header_tbl = Table(
        [[Paragraph("MEDICAL REPORT SUMMARY", styles["title"])],
         [Paragraph("AI-Generated Patient Summary  —  Confidential", styles["subtitle"])]],
        colWidths=[CONTENT_WIDTH],
    )
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_HEADER_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    elements.append(header_tbl)
    elements.append(Spacer(1, 10))

    # Info grid — two columns
    def lbl(text): return Paragraph(sanitize(text), styles["field_label"])
    def val(text): return Paragraph(sanitize(str(text)), styles["field_value"])

    col = CONTENT_WIDTH / 2

    grid_data = [
        [lbl("PATIENT NAME"),        lbl("POLICY HOLDER")],
        [val(medical.patient_name),  val(insurance.policy_holder_name)],
        [lbl("REPORT DATE"),         lbl("INSURER")],
        [val(medical.report_date),   val(insurance.insurer_name)],
        [lbl("ATTENDING PHYSICIAN"), lbl("POLICY NUMBER")],
        [val(medical.attending_physician), val(insurance.policy_number)],
    ]
    grid = Table(grid_data, colWidths=[col, col])
    grid.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_COVERED_BG),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(grid)
    elements.append(Spacer(1, 12))
    return elements


# ── Summary ────────────────────────────────────────────────────────────────────

def build_summary(medical: MedicalOutput, styles: dict) -> list:
    elements = section_bar("Overall Summary", styles)
    elements.append(Paragraph(sanitize(medical.summary), styles["body"]))
    elements.append(Spacer(1, 6))
    return elements


# ── Findings ───────────────────────────────────────────────────────────────────

def build_findings(
    medical: MedicalOutput,
    insurance: InsuranceOutput,
    styles: dict,
) -> list:
    elements = section_bar("Findings & Coverage", styles)

    coverage_map: dict[str, CoverageItem] = {
        c.finding_name: c for c in insurance.coverage
    }

    # Card inner width = CONTENT_WIDTH minus left+right padding (12mm each = 24mm)
    CARD_INNER  = CONTENT_WIDTH - 24 * mm          # ~146mm
    # Coverage box inner width = CARD_INNER minus its own padding (8mm each = 16mm)
    COV_INNER   = CARD_INNER - 16 * mm             # ~130mm

    for finding in medical.findings:
        fg_color, bg_color = FLAG_COLORS.get(finding.flag, (C_MUTED, C_SECTION_BG))
        flag_label = FLAG_LABELS.get(finding.flag, finding.flag.upper())
        coverage   = coverage_map.get(finding.name)

        # ── Flag pill ──
        flag_pill = Table(
            [[Paragraph(
                f"  {flag_label}  ",
                ParagraphStyle("fp", fontName=FONT_BOLD, fontSize=8,
                               textColor=fg_color, alignment=TA_CENTER)
            )]],
            colWidths=[22 * mm],
        )
        flag_pill.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg_color),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
            ("BOX",           (0, 0), (-1, -1), 0.5, fg_color),
        ]))

        # ── Name | flag | value row ──
        # Widths must sum to CARD_INNER
        name_col  = CARD_INNER - 22 * mm - 28 * mm  # ~96mm
        value_col = 28 * mm

        name_row = Table(
            [[Paragraph(sanitize(finding.name), styles["finding_name"]),
              flag_pill,
              Paragraph(
                  sanitize(finding.value),
                  ParagraphStyle("fv", fontName=FONT_BOLD, fontSize=10,
                                 textColor=fg_color, alignment=TA_RIGHT)
              )]],
            colWidths=[name_col, 22 * mm, value_col],
        )
        name_row.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))

        # ── Inner content rows ──
        inner = [
            name_row,
            Paragraph(
                sanitize(f"Reference range: {finding.reference_range}"),
                styles["finding_meta"]
            ),
            Paragraph(sanitize(finding.plain_explanation), styles["finding_explanation"]),
            Paragraph(sanitize(f"Trend: {finding.trend}"), styles["trend"]),
        ]

        # ── Coverage box ──
        if coverage:
            excl = (f" | Exclusions: {coverage.exclusions}"
                    if coverage.exclusions and coverage.exclusions.lower() != "none"
                    else "")
            auth = " | Pre-auth required" if coverage.pre_auth_required else ""
            status_text = ("Covered" if coverage.covered else "Not covered") + auth
            status_color = C_COVERED if coverage.covered else C_CRITICAL

            cov_box = Table(
                [[Paragraph(
                    sanitize(status_text),
                    ParagraphStyle("cs", fontName=FONT_BOLD, fontSize=9,
                                   textColor=status_color)
                )],
                 [Paragraph(
                    sanitize(coverage.coverage_detail + excl),
                    styles["coverage_detail"]
                 )]],
                colWidths=[COV_INNER],
            )
            cov_box.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), C_COVERED_BG),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
            ]))
            inner.append(cov_box)
        else:
            inner.append(Paragraph(
                "Insurance: Not found in policy — contact your insurer.",
                styles["coverage_detail"]
            ))

        # ── Outer card ──
        card = Table(
            [[item] for item in inner],
            colWidths=[CARD_INNER],
        )
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.white),
            ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
            ("LINEBEFORE",    (0, 0), (0, -1),  4,   fg_color),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        elements.append(KeepTogether(card))
        elements.append(Spacer(1, 8))

    return elements


# ── Follow-up & referrals ──────────────────────────────────────────────────────

def build_followup(medical: MedicalOutput, styles: dict) -> list:
    elements = section_bar("Follow-up Actions", styles)
    for action in medical.follow_up_actions:
        elements.append(Paragraph(sanitize(f"• {action}"), styles["bullet"]))
    elements.append(Spacer(1, 4))

    if medical.referrals:
        elements += section_bar("Specialist Referrals", styles)
        for ref in medical.referrals:
            elements.append(Paragraph(sanitize(f"• {ref}"), styles["bullet"]))
        elements.append(Spacer(1, 4))

    return elements


# ── Claim steps ────────────────────────────────────────────────────────────────

def build_claim_section(insurance: InsuranceOutput, styles: dict) -> list:
    elements = section_bar("How to File a Claim", styles)

    NUM_COL  = 10 * mm
    TEXT_COL = CONTENT_WIDTH - NUM_COL - 8 * mm  # 8mm for inter-column gap

    for step in insurance.claim_steps:
        step_tbl = Table(
            [[Paragraph(
                  str(step.step_number),
                  ParagraphStyle("sn", fontName=FONT_BOLD, fontSize=11,
                                 textColor=colors.white, alignment=TA_CENTER)
              ),
              [Paragraph(sanitize(step.instruction), styles["step_text"]),
               Paragraph(sanitize(f"Deadline: {step.deadline}"), styles["step_deadline"])],
            ]],
            colWidths=[NUM_COL, TEXT_COL],
        )
        step_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), C_COVERED),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ]))
        elements.append(step_tbl)
        elements.append(Spacer(1, 5))

    elements.append(Spacer(1, 6))
    elements += section_bar("Documents Required for Claim", styles)
    for doc in insurance.documents_required:
        elements.append(Paragraph(sanitize(f"• {doc}"), styles["bullet"]))
    elements.append(Spacer(1, 8))
    return elements


# ── Disclaimer ─────────────────────────────────────────────────────────────────

def build_disclaimer(medical: MedicalOutput, styles: dict) -> list:
    return [
        HRFlowable(width="100%", thickness=0.5, color=C_BORDER),
        Paragraph(sanitize(medical.disclaimer), styles["disclaimer"]),
        Paragraph(
            f"Generated on {datetime.now().strftime('%d %B %Y at %H:%M')}",
            styles["disclaimer"]
        ),
    ]


# ── Main entry point ───────────────────────────────────────────────────────────

def build_document(
    medical: MedicalOutput,
    insurance: InsuranceOutput,
    output_path: str | Path = "final_report.pdf",
) -> Path:
    """
    Render the final patient report PDF.

    Parameters
    ----------
    medical      : MedicalOutput from medical_agent.py
    insurance    : InsuranceOutput from insurance_agent.py
    output_path  : destination path for the PDF

    Returns
    -------
    Path to the generated PDF
    """
    output_path = Path(output_path)
    styles      = build_styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm,  bottomMargin=20 * mm,
        title=f"Medical Report — {medical.patient_name}",
        author="Medical Report Simplifier",
    )

    story = []
    story += build_header(medical, insurance, styles)
    story += build_summary(medical, styles)
    story += build_findings(medical, insurance, styles)
    story += build_followup(medical, styles)
    story += build_claim_section(insurance, styles)
    story += build_disclaimer(medical, styles)

    doc.build(story)
    print(f"[DocumentBuilder] PDF saved → {output_path.resolve()}")
    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python document_builder.py medical_output.json insurance_output.json")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        medical = MedicalOutput.model_validate(json.load(f))

    with open(sys.argv[2], encoding="utf-8") as f:
        insurance = InsuranceOutput.model_validate(json.load(f))

    out = build_document(medical, insurance)
    print(f"Done — open {out}")