"""
app.py  —  NeuroSplain Alzheimer's Detection System
Run:  streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np
from datetime import datetime
from PIL import Image
import streamlit as st

st.set_page_config(page_title="NeuroSplain", layout="wide", page_icon=None)

import torch
from modules import (
    INTEGRATED_CLASSES,
    clinical_predict,
    eeg_predict_v2, load_eeg_models,
    generate_pdf_report,
    load_fusion_model,
    mri_predict,
    run_integrated_analysis,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS — high-contrast, clean medical look, no emojis
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset & base ── */
html, body { background-color: #F1F5F9; }
.stApp     { background-color: #F1F5F9; }

/* Force Streamlit markdown text dark */
.stMarkdown, .stMarkdown p, .stMarkdown span,
[data-testid="stMarkdownContainer"] p {
    color: #1E293B !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
.stMarkdown h5, .stMarkdown h6 {
    color: #0F172A !important;
    font-weight: 700;
}

/* No custom loader background styling; keep Streamlit default loader appearance */

/* Streamlit markdown */
.stMarkdown, .stMarkdown p, .stMarkdown span,
[data-testid="stMarkdownContainer"] p { color: #1E293B !important; }

/* App header title should stay white on the blue banner */
.app-header h2, .app-header h2 * {
    color: #FFFFFF !important;
}
.app-header p, .app-header p * {
    color: #BFDBFE !important;
}

/* Force button text white for Streamlit buttons */
.stButton button,
.stButton button *,
.stButton > button,
.stButton > button *,
div.stButton button,
div.stButton button *,
div.stButton > button,
div.stButton > button * {
    color: #FFFFFF !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #0F172A !important;
    border-bottom: 3px solid #1E3A8A !important;
    background-color: #EFF6FF !important;
}

/* ── Buttons — high contrast ── */
div.stButton > button,
div.stButton button,
.stButton > button,
.stButton button,
.stButton > button span,
.stButton button span,
div.stButton > button span,
div.stButton button span,
.stButton > button *,
.stButton button *,
div.stButton > button *,
div.stButton button * {
    color: #FFFFFF !important;
}
div.stButton > button,
div.stButton button,
.stButton > button,
.stButton button {
    background-color: #1E3A8A !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    width: 100% !important;
    letter-spacing: 0.4px;
    box-shadow: 0 2px 8px rgba(30,58,138,0.3);
    padding: 0.9rem 1rem !important;
    line-height: 1.2 !important;
}
div.stButton > button:hover,
div.stButton button:hover,
.stButton > button:hover,
.stButton button:hover {
    background-color: #1e40af !important;
    color: #FFFFFF !important;
}
div.stButton > button:disabled,
div.stButton button:disabled,
.stButton > button:disabled,
.stButton button:disabled,
div.stButton > button:disabled span,
div.stButton button:disabled span,
.stButton > button:disabled span,
.stButton button:disabled span,
div.stButton > button:disabled *,
.stButton button:disabled * {
    background-color: #CBD5E1 !important;
    color: #64748B !important;
    box-shadow: none !important;
}

/* Download button — outline style */
div.stDownloadButton > button {
    background-color: #FFFFFF !important;
    color: #1E3A8A !important;
    border: 2px solid #1E3A8A !important;
    border-radius: 8px !important;
    height: 3.2em !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    width: 100% !important;
}
div.stDownloadButton > button:hover {
    background-color: #EFF6FF !important;
    color: #1E3A8A !important;
}

/* Sliders */
.stSlider label   { color: #1E293B !important; font-weight: 600 !important; font-size: 14px !important; }
.stSlider span    { color: #1E293B !important; }
[data-testid="stSliderThumbValue"] { color: #1E293B !important; }

/* File uploader */
[data-testid="stFileUploader"], .stFileUploader,
[data-testid="stFileUploader"] div, .stFileUploader div,
[data-testid="stFileUploader"] span, .stFileUploader span,
[data-testid="stFileUploader"] p, .stFileUploader p,
[data-testid="stFileUploader"] label, .stFileUploader label,
[data-testid="stFileUploader"] button, .stFileUploader button {
    color: #FFFFFF !important;
}
[data-testid="stFileUploader"], .stFileUploader {
    background-color: #1E293B !important;
    border-radius: 12px !important;
    padding: 18px 20px 16px !important;
}
[data-testid="stFileUploader"] *, .stFileUploader *,
[data-testid="stFileUploader"] button, .stFileUploader button,
[data-testid="stFileUploader"] label, .stFileUploader label {
    color: #FFFFFF !important;
}
[data-testid="stFileUploader"] button, .stFileUploader button {
    background-color: #1E40AF !important;
    border-color: transparent !important;
}
[data-testid="stFileUploader"] p, .stFileUploader p { color: #CBD5E1 !important; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: #FFFFFF; border-radius: 8px; padding: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
div[data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 700 !important; }
div[data-testid="stMetricLabel"] { color: #64748B !important; }

/* Alerts */
div[data-testid="stInfo"]    { background: #EFF6FF !important; border-color: #93C5FD !important; }
div[data-testid="stSuccess"] { background: #F0FDF4 !important; }
div[data-testid="stWarning"] { background: #FFFBEB !important; }
div[data-testid="stError"]   { background: #FEF2F2 !important; }
div[data-testid="stInfo"] p,
div[data-testid="stSuccess"] p,
div[data-testid="stWarning"] p,
div[data-testid="stError"] p  { color: #1E293B !important; }

/* Toast notifications */
[data-testid*="toast"], .stToast {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
}
[data-testid*="toast"] *, .stToast * {
    color: #0F172A !important;
}

/* Hide Streamlit built-in loading spinner/status bar */
div[data-testid="stSpinner"], div[data-testid="stLoading"], .stSpinner, .streamlit-spinner {
    display: none !important;
}
div[data-testid="stSpinner"] *, div[data-testid="stLoading"] *, .stSpinner *, .streamlit-spinner * {
    display: none !important;
}

/* Divider */
hr { border-color: #CBD5E1 !important; }

/* Caption */
.stCaption, [data-testid="stCaptionContainer"] p { color: #64748B !important; font-size: 12px; }

/* Scrollbars */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #94A3B8; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Header banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='app-header' style='background:#1E3A8A; padding:18px 26px; border-radius:10px; margin-bottom:22px;'>
  <h2 style='margin:0; font-size:24px; color:#FFFFFF !important;'>NeuroSplain</h2>
  <p style='margin:5px 0 0; font-size:13px; color:#BFDBFE !important;'>
    Multimodal Alzheimer's Detection &mdash; Clinical + MRI + EEG Fusion
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [
    ("analysis_done", False),
    ("results", None),
    ("fusion_inputs", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
#  Model loading (cached) with custom Alzheimer system loader text
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource()
def _cached_fusion_model():
    return load_fusion_model()

@st.cache_resource()
def _cached_eeg_models():
    return load_eeg_models()

_loading_overlay = st.empty()
_loading_overlay.markdown(
    "<div style='position:fixed; inset:0; z-index:100000; display:flex;"
    "align-items:center; justify-content:center; padding:24px; background:rgba(15,23,42,0.72);'>"
    "<div style='max-width:560px; width:100%; background:rgba(255,255,255,0.98);"
    "border-radius:24px; padding:28px 32px; box-shadow:0 24px 80px rgba(15,23,42,0.25); text-align:center;'>"
    "<h2 style='margin:0 0 12px; font-size:24px; color:#0F172A;'>NeuroSplain models are being loaded</h2>"
    "<p style='margin:0; font-size:15px; color:#475569; line-height:1.6;'>"
    "Please wait while MRI, EEG, and fusion models initialise.</p>"
    "</div></div>", unsafe_allow_html=True)

fusion_model = _cached_fusion_model()
eeg_models = _cached_eeg_models()
_loading_overlay.empty()

eeg_load_error = eeg_models.get("_load_error") if isinstance(eeg_models, dict) else None

eeg_ready = bool(eeg_models) and not bool(eeg_load_error)

if "model_toast_shown" not in st.session_state:
    def _show_model_toast(message, icon=None):
        if hasattr(st, "toast"):
            if icon:
                st.toast(message, icon=icon)
            else:
                st.toast(message)
        else:
            st.info(message)

    if fusion_model and eeg_ready:
        _show_model_toast("All models loaded successfully")
    elif not fusion_model and eeg_ready:
        st.error("MRI model not loaded properly")
    elif fusion_model and not eeg_ready:
        st.error("EEG model not loaded properly")
    else:
        st.error("MRI and EEG models not loaded properly")
    st.session_state.model_toast_shown = True

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: placeholder card
# ─────────────────────────────────────────────────────────────────────────────
def _placeholder():
    st.markdown(
        "<div style='background:#F8FAFC; border:1px dashed #CBD5E1; border-radius:10px;"
        "min-height:200px; display:flex; align-items:center; justify-content:center;'>"
        "<span style='color:#94A3B8; font-size:14px;'>Results will appear here</span></div>",
        unsafe_allow_html=True)


def _result_box(text, color):
    st.markdown(
        f"<div style='background:#FFFFFF; border-left:5px solid {color}; border-radius:8px;"
        f"padding:14px 20px; margin-top:12px; box-shadow:0 1px 3px rgba(0,0,0,0.07);'>"
        f"<span style='font-size:16px; font-weight:700; color:{color};'>{text}</span></div>",
        unsafe_allow_html=True)


def _safe_upload_name(uploaded_file):
    if uploaded_file is None:
        return None
    return getattr(uploaded_file, "name", None)


def _fusion_inputs_changed(current_inputs):
    previous = st.session_state.get("fusion_inputs")
    if previous != current_inputs:
        st.session_state.fusion_inputs = current_inputs
        if st.session_state.analysis_done:
            st.session_state.analysis_done = False
            st.session_state.results = None


# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_c, tab_m, tab_e, tab_f = st.tabs([
    "Clinical Assessment",
    "MRI Analysis",
    "EEG Analysis",
    "Fusion Suite",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CLINICAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_c:
    st.markdown("#### Clinical Assessment")
    st.caption("Normalised scores: 0 = worst, 1 = best. Memory Complaints and Behavioral Problems: 0 = none, 1 = severe.")
    st.divider()

    col_in, col_res = st.columns([1, 1], gap="large")
    with col_in:
        c_mmse  = st.slider("MMSE Score",            0.0, 1.0, 0.70, 0.01, key="c_mmse")
        c_func  = st.slider("Functional Assessment", 0.0, 1.0, 0.80, 0.01, key="c_func")
        c_mem   = st.slider("Memory Complaints",     0.0, 1.0, 0.20, 0.01, key="c_mem")
        c_behav = st.slider("Behavioral Problems",   0.0, 1.0, 0.10, 0.01, key="c_behav")
        c_adl   = st.slider("ADL Score",             0.0, 1.0, 0.90, 0.01, key="c_adl")
        st.markdown("")
        run_c = st.button("Run Clinical Analysis", type="primary", key="btn_c")

    with col_res:
        if run_c:
            with st.spinner("Analysing..."):
                fig, result = clinical_predict(c_mmse, c_func, c_mem, c_behav, c_adl)
            st.pyplot(fig, use_container_width=True)
            txt = result.replace("**", "")
            clr = "#DC2626" if "Risk" in txt else "#16A34A"
            _result_box(txt, clr)
        else:
            _placeholder()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MRI  (with Grad-CAM)
# ══════════════════════════════════════════════════════════════════════════════
with tab_m:
    st.markdown("#### MRI Analysis")
    st.caption("Upload a T1-weighted brain MRI scan. ResNet-50 classifies into 4 dementia stages. Grad-CAM highlights diagnostically active regions.")
    st.divider()

    col_in, col_res = st.columns([1, 1.6], gap="large")
    with col_in:
        mri_file = st.file_uploader("Upload MRI scan (PNG / JPG)", type=["png","jpg","jpeg"], key="mri_up")
        st.markdown("")
        run_m = st.button("Run MRI Analysis", type="primary", key="btn_m",
                          disabled=(mri_file is None))
        if mri_file is None:
            st.caption("Upload a scan to enable analysis.")

    with col_res:
        if run_m and mri_file:
            with st.spinner("Running inference + Grad-CAM..."):
                image = Image.open(mri_file)
                fig, result, probs, _ = mri_predict(image, fusion_model)
            st.pyplot(fig, use_container_width=True)
            txt = result.replace("**", "")
            clr = "#DC2626" if "Moderate" in txt else "#D97706" if "Mild" in txt else "#16A34A"
            _result_box(txt, clr)
            # Grad-CAM legend
            st.markdown(
                "<div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px;"
                "padding:10px 14px; font-size:12px; color:#475569; margin-top:8px;'>"
                "<strong style='color:#1E293B;'>Grad-CAM key:</strong>&nbsp;&nbsp;"
                "<strong style='color:#DC2626;'>Red</strong> — primary atrophy (hippocampus)&nbsp;&nbsp;"
                "<strong style='color:#D97706;'>Yellow</strong> — secondary changes&nbsp;&nbsp;"
                "<strong style='color:#2563EB;'>Blue</strong> — healthy tissue"
                "</div>", unsafe_allow_html=True)
        else:
            _placeholder()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — EEG
# ══════════════════════════════════════════════════════════════════════════════
with tab_e:
    st.markdown("#### EEG Analysis")
    st.caption("Upload a resting-state EEG in EEGLAB .set format (19-channel, ds004504 protocol). Features extracted from 9 SHAP-selected channels.")
    st.divider()

    col_in, col_res = st.columns([1, 1.4], gap="large")
    with col_in:
        eeg_file = st.file_uploader("Upload EEG (.set)", type=["set"], key="eeg_up")
        st.markdown("")
        run_e = st.button("Run EEG Analysis", type="primary", key="btn_e",
                          disabled=(eeg_file is None or not eeg_ready))

        if not eeg_ready:
            if eeg_load_error:
                st.error("EEG model is unavailable. See status above for details.")
            else:
                st.warning("EEG model files not found in models/ directory.")
        elif eeg_file is None:
            st.caption("Upload a .set file to enable analysis.")

    with col_res:
        if run_e and eeg_file and eeg_ready:
            with st.spinner("Segmenting and extracting 334 features..."):
                try:
                    fig, result, probs = eeg_predict_v2(eeg_file, eeg_models)
                    st.pyplot(fig, use_container_width=True)
                    txt = result.replace("**", "")
                    clr = "#DC2626" if "Alzheimer" in txt else "#16A34A"
                    _result_box(txt, clr)
                except Exception as e:
                    st.error(f"EEG analysis failed: {e}")
        else:
            _placeholder()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — FUSION SUITE
# ══════════════════════════════════════════════════════════════════════════════
with tab_f:
    st.markdown("#### Integrated Neuro-Diagnostic Suite")
    st.caption("Integrated Alzheimer's detection combining MRI, clinical, and EEG evidence.")
    st.divider()

    # Inputs
    col_files, col_clin = st.columns([1, 1], gap="large")
    with col_files:
        st.markdown("**Biomarker Uploads**")
        f_mri = st.file_uploader("MRI Scan (PNG / JPG)", type=["png","jpg","jpeg"], key="f_mri")
        f_eeg = st.file_uploader("EEG Recording (.set)", type=["set"], key="f_eeg")

    with col_clin:
        st.markdown("**Clinical Parameters**")
        f_mmse  = st.slider("MMSE Score",            0.0, 1.0, 0.70, 0.01, key="fi_mmse")
        f_func  = st.slider("Functional Assessment", 0.0, 1.0, 0.80, 0.01, key="fi_func")
        f_mem   = st.slider("Memory Complaints",     0.0, 1.0, 0.20, 0.01, key="fi_mem")
        f_behav = st.slider("Behavioral Problems",   0.0, 1.0, 0.10, 0.01, key="fi_behav")
        f_adl   = st.slider("ADL Score",             0.0, 1.0, 0.90, 0.01, key="fi_adl")

    fusion_inputs = (
        _safe_upload_name(f_mri),
        _safe_upload_name(f_eeg),
        f_mmse, f_func, f_mem, f_behav, f_adl,
    )
    _fusion_inputs_changed(fusion_inputs)

    st.divider()

    # ── Action row ────────────────────────────────────────────────────────────
    can_run = bool(f_mri and f_eeg)

    if not st.session_state.get("analysis_done"):
        # Before results: single centred button
        _, btn_col, _ = st.columns([2, 3, 2])
        with btn_col:
            run_f = st.button("RUN ANALYSIS", type="primary",
                              use_container_width=True, key="btn_f",
                              disabled=not can_run)
            if not can_run:
                st.caption("Upload both MRI and EEG files to run analysis.")
    else:
        # After results: Run + Download side by side
        _, c_run, c_dl, _ = st.columns([1, 2, 2, 1])
        with c_run:
            run_f = st.button("RUN ANALYSIS", type="primary",
                              use_container_width=True, key="btn_f")
        with c_dl:
            res = st.session_state.results
            pdf = generate_pdf_report(
                diagnosis  = res["diag"],
                confidence = f"{res['conf']:.2f}",
                mmse       = res["clinical"]["mmse"],
                memory     = res["clinical"]["mem"],
                adl        = res["clinical"]["adl"],
                func       = res["clinical"]["func"],
                behav      = res["clinical"]["behav"],
            )
            st.download_button(
                "DOWNLOAD REPORT (PDF)", data=pdf,
                file_name=f"NeuroSplain_{res['diag']}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf", use_container_width=True)

    # ── Computation ───────────────────────────────────────────────────────────
    if run_f and can_run:
        with st.spinner("Running multimodal fusion..."):
            try:
                mri_img  = Image.open(f_mri)
                clin_dat = (f_mmse, f_func, f_mem, f_behav, f_adl)
                f_probs, _, overlay, orig_rgb, eeg_ok = run_integrated_analysis(
                    mri_img=mri_img, eeg_file=f_eeg,
                    clinical_data=clin_dat,
                    fusion_model=fusion_model, eeg_models=eeg_models)

                pred_idx = int(np.argmax(f_probs))
                st.session_state.results = {
                    "diag":    INTEGRATED_CLASSES[pred_idx],
                    "conf":    float(f_probs[pred_idx] * 100),
                    "probs":   f_probs,
                    "overlay": overlay,
                    "orig":    orig_rgb,
                    "eeg_ok":  eeg_ok,
                    "clinical": {
                        "mmse": f_mmse, "func": f_func,
                        "mem":  f_mem,  "behav": f_behav, "adl": f_adl},
                }
                st.session_state.analysis_done = True
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback; traceback.print_exc()

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.get("analysis_done"):
        res   = st.session_state.results
        diag  = res["diag"]
        conf  = res["conf"]
        clr   = ("#DC2626" if "Moderate" in diag else
                 "#D97706" if "Mild"     in diag else "#16A34A")

        st.divider()

        # Top banner
        st.markdown(
            f"<div style='background:#FFFFFF; border-left:6px solid {clr}; border-radius:10px;"
            f"padding:16px 24px; box-shadow:0 1px 4px rgba(0,0,0,0.08);'>"
            f"<h3 style='margin:0; color:{clr} !important; font-size:20px;'>"
            f"{diag} &mdash; {conf:.1f}% confidence</h3>"
            f"</div>", unsafe_allow_html=True)

        if not res.get('eeg_ok'):
            st.warning("EEG fallback mode was used during fusion analysis.")

        st.markdown("")
        col_l, col_r = st.columns([1.3, 1], gap="large")

        with col_l:
            st.markdown("**Neural Heatmap (Grad-CAM)**")
            c1, c2 = st.columns(2)
            c1.image(res["orig"],    caption="Original MRI",  use_container_width=True)
            c2.image(res["overlay"], caption="Active Zones",  use_container_width=True)
            st.markdown(
                "<div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px;"
                "padding:10px 14px; font-size:12px; color:#475569 !important; margin-top:8px;'>"
                "<strong style='color:#1E293B;'>Grad-CAM key:</strong>&nbsp;&nbsp;"
                "<strong style='color:#DC2626;'>Red</strong> — primary atrophy&nbsp;&nbsp;"
                "<strong style='color:#D97706;'>Yellow</strong> — secondary changes&nbsp;&nbsp;"
                "<strong style='color:#2563EB;'>Blue</strong> — healthy tissue"
                "</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown("**Probability Distribution**")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            bar_colors = ["#16A34A", "#60A5FA", "#D97706", "#DC2626"]
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor("#FFFFFF"); ax.set_facecolor("#FFFFFF")
            bars = ax.bar(INTEGRATED_CLASSES, res["probs"], color=bar_colors,
                          edgecolor="#1E293B", linewidth=0.5, width=0.55)
            for bar, p in zip(bars, res["probs"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{p*100:.1f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#1E293B")
            ax.set_ylim(0, 1.18)
            ax.set_ylabel("Probability", fontsize=9, color="#1E293B")
            ax.tick_params(colors="#1E293B", labelsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            for spine in ax.spines.values(): spine.set_edgecolor("#CBD5E1")
            plt.xticks(rotation=18, ha="right", color="#1E293B", fontsize=8)
            plt.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # CDR mapping
            cdr_map = {
                "NonDemented":      ("CDR 0",   "No impairment",      "#16A34A"),
                "VeryMildDemented": ("CDR 0.5", "Very mild",          "#60A5FA"),
                "MildDemented":     ("CDR 1",   "Mild impairment",    "#D97706"),
                "ModerateDemented": ("CDR 2",   "Moderate impairment","#DC2626"),
            }
            cdr, desc, cdr_clr = cdr_map.get(diag, ("—", "—", "#64748B"))
            st.markdown(
                f"<div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px;"
                f"padding:12px 16px; margin-top:10px;'>"
                f"<p style='margin:0; font-size:12px; color:#64748B !important;'>Clinical Dementia Rating</p>"
                f"<p style='margin:4px 0 0; font-size:15px; font-weight:700; color:{cdr_clr} !important;'>"
                f"{cdr} &mdash; {desc}</p>"
                f"</div>", unsafe_allow_html=True)