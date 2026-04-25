"""clinical_module.py"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLINICAL_FEATURES = ["MMSE", "FunctionalAssessment", "MemoryComplaints",
                     "BehavioralProblems", "ADL"]
_W = {"func": 0.285, "adl": 0.249, "mem": 0.197, "mmse": 0.160, "behav": 0.112}


def clinical_risk_score(mmse, func, mem, behav, adl) -> float:
    p = ((1-func)*_W["func"] + (1-adl)*_W["adl"] + mem*_W["mem"]
         + (1-mmse)*_W["mmse"] + behav*_W["behav"])
    return float(np.clip(p, 0.05, 0.95))


def clinical_predict(mmse, func, mem, behav, adl):
    prob_ad = clinical_risk_score(mmse, func, mem, behav, adl)
    pred    = "Alzheimer's Risk" if prob_ad > 0.5 else "No Risk Detected"
    vals    = [mmse, func, mem, behav, adl]
    colors  = ["#16A34A" if v > 0.7 else "#D97706" if v > 0.4 else "#DC2626"
               for v in vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    bars = ax.bar(CLINICAL_FEATURES, vals, color=colors, edgecolor="#1E293B",
                  linewidth=0.5, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#1E293B")
    ax.set_ylim(0, 1.18)
    sev_color = "#DC2626" if prob_ad > 0.5 else "#16A34A"
    ax.set_title(f"{pred}   |   AD probability: {prob_ad*100:.1f}%",
                 fontsize=12, fontweight="bold", color=sev_color)
    ax.set_ylabel("Score", fontsize=9, color="#1E293B")
    ax.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=0.8, label="threshold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(colors="#1E293B", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#CBD5E1")
    plt.xticks(rotation=15, ha="right", color="#1E293B")
    plt.tight_layout()
    return fig, f"**{pred}** — {prob_ad*100:.1f}% Alzheimer's probability"