"""report_module.py — PDF report (no weights column in clinical table)"""
from __future__ import annotations
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, PageBreak, Paragraph,
                                 SimpleDocTemplate, Spacer, Table, TableStyle)

NAVY  = colors.HexColor("#1E3A8A")
LIGHT = colors.HexColor("#EFF6FF")
RED   = colors.HexColor("#DC2626")
ORG   = colors.HexColor("#D97706")
GRN   = colors.HexColor("#16A34A")
GRAY  = colors.HexColor("#64748B")


def _s(name, **kw): return ParagraphStyle(name, **kw)


def _risk(val, invert=False):
    score = (1-val) if invert else val
    col, lbl = (GRN, "Low Risk") if score >= 0.7 else (ORG, "Moderate") if score >= 0.4 else (RED, "High Risk")
    return Paragraph(f"<font color='#{col.hexval()[2:]}'><b>{lbl}</b></font>",
                     _s("RL", fontSize=9, fontName="Helvetica-Bold", alignment=TA_CENTER))


def generate_pdf_report(diagnosis, confidence, mmse, memory, adl, func, behav) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    sev = RED if "Moderate" in diagnosis else ORG if "Mild" in diagnosis else GRN
    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story += [
        Paragraph("NeuroSplain", _s("T", fontSize=22, leading=28, textColor=NAVY,
                  fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4)),
        Paragraph("Alzheimer's Multimodal Diagnostic Report",
                  _s("S", fontSize=11, textColor=GRAY, fontName="Helvetica",
                     alignment=TA_CENTER, spaceAfter=2)),
        Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y  |  %H:%M:%S')}",
                  _s("S2", fontSize=10, textColor=GRAY, fontName="Helvetica",
                     alignment=TA_CENTER, spaceAfter=2)),
        Spacer(1, 6),
        HRFlowable(width="100%", thickness=2, color=NAVY),
        Spacer(1, 10),
    ]

    # ── Diagnosis badge ───────────────────────────────────────────────────────
    badge = Table([
        [Paragraph("FINAL DIAGNOSIS", _s("BH", fontSize=9, textColor=GRAY,
                   fontName="Helvetica-Bold", alignment=TA_CENTER))],
        [Paragraph(diagnosis, _s("D", fontSize=16, textColor=sev,
                   fontName="Helvetica-Bold", alignment=TA_CENTER, leading=20))],
        [Paragraph(f"Confidence: <b>{confidence}%</b>",
                   _s("C", fontSize=11, textColor=sev,
                   fontName="Helvetica-Bold", alignment=TA_CENTER))],
    ], colWidths=[6.5*inch])
    badge.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), LIGHT),
        ("BOX",           (0,0),(-1,-1), 1.5, NAVY),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
    ]))
    story += [badge, Spacer(1, 14)]

    # ── Clinical table — NO weights column ───────────────────────────────────
    th = _s("TH", fontSize=10, fontName="Helvetica-Bold",
            textColor=colors.white, alignment=TA_CENTER)
    rows = [
        [Paragraph("<b>Parameter</b>", th),
         Paragraph("<b>Score</b>",     th),
         Paragraph("<b>Assessment</b>",th)],
        ["MMSE Score",            f"{mmse:.2f}",   _risk(mmse)],
        ["Functional Assessment", f"{func:.2f}",   _risk(func)],
        ["Memory Complaints",     f"{memory:.2f}", _risk(memory, True)],
        ["Behavioral Problems",   f"{behav:.2f}",  _risk(behav, True)],
        ["ADL Score",             f"{adl:.2f}",    _risk(adl)],
    ]
    ct = Table(rows, colWidths=[3.0*inch, 1.2*inch, 2.3*inch], hAlign="LEFT")
    ct.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1, 0), NAVY),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LIGHT]),
        ("BOX",           (0,0),(-1,-1), 0.5, NAVY),
        ("INNERGRID",     (0,0),(-1,-1), 0.25, GRAY),
        ("ALIGN",         (1,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
    ]))

    sec  = _s("SEC", fontSize=13, textColor=NAVY, fontName="Helvetica-Bold",
              spaceBefore=14, spaceAfter=4, leading=18)
    body = _s("BODY", fontSize=10, textColor=colors.black, fontName="Helvetica",
              alignment=TA_JUSTIFY, spaceAfter=6, leading=14)

    story += [
        Paragraph("Clinical Input Parameters", sec),
        HRFlowable(width="100%", thickness=0.5, color=NAVY),
        Spacer(1, 6), ct, Spacer(1, 14),

        Paragraph("MRI Neuroimaging (Grad-CAM)", sec),
        HRFlowable(width="100%", thickness=0.5, color=NAVY),
        Spacer(1, 6),
        Paragraph(
            "Grad-CAM applied to ResNet-50 identifies regions of diagnostic relevance. "
            "Red zones — primary atrophy (hippocampus, entorhinal cortex). "
            "Yellow — secondary structural changes. Blue — healthy tissue.",
            body),
        Spacer(1, 8),

        Paragraph("EEG Electrophysiological Analysis", sec),
        HRFlowable(width="100%", thickness=0.5, color=NAVY),
        Spacer(1, 6),
        Paragraph(
            "19-channel resting-state EEG processed using 334 features extracted from "
            "the 9 SHAP-selected channels (O2, T5, T3, F7, O1, P4, T6, F8, P3). "
            "Features include band power, statistical, Hjorth, spectral entropy, "
            "connectivity, and hemispheric asymmetry. SelectKBest reduces to 150 features. "
            "Prediction via stacking ensemble (XGBoost + LightGBM + RF + GBM).",
            body),
        PageBreak(),

        Paragraph("Disclaimer", sec),
        HRFlowable(width="100%", thickness=0.5, color=NAVY),
        Spacer(1, 6),
    ]

    disc = Table([
        [Paragraph("IMPORTANT DISCLAIMER", _s("DH", fontSize=11, fontName="Helvetica-Bold",
                   textColor=colors.white, alignment=TA_CENTER))],
        [Paragraph(
            "This report is an AI-generated estimation and is NOT a substitute for "
            "professional medical advice, diagnosis, or treatment. "
            "Consult a qualified neurologist for a complete clinical assessment.",
            _s("DB", fontSize=10, fontName="Helvetica-Bold",
               textColor=RED, alignment=TA_CENTER, leading=15))],
    ], colWidths=[6.5*inch])
    disc.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1, 0), NAVY),
        ("BACKGROUND",    (0,1),(-1,-1), colors.HexColor("#FEF2F2")),
        ("BOX",           (0,0),(-1,-1), 2, RED),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ("RIGHTPADDING",  (0,0),(-1,-1), 12),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story += [
        disc, Spacer(1, 8),
        HRFlowable(width="100%", thickness=0.5, color=GRAY),
        Spacer(1, 4),
        Paragraph(
            f"NeuroSplain  |  {datetime.now().strftime('%Y-%m-%d')}  |  "
            "Multimodal Alzheimer Detection System",
            _s("F", fontSize=8, textColor=GRAY, fontName="Helvetica", alignment=TA_CENTER)),
    ]

    doc.build(story)
    buf.seek(0)
    return buf.read()