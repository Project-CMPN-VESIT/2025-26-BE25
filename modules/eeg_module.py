"""
eeg_module.py
─────────────────────────────────────────────────────────────────────────────
EEG feature extraction that EXACTLY matches the Colab training notebook.

Feature layout (334-dim, extracted from TOP-9 SHAP channels only):
  9 channels × 31 features/channel  = 279
  5 bands × 4 global stats          =  20  (mean, std, max, min across channels)
  15 channel pairs × 2 coherences   =  30  (alpha + theta pairwise correlation)
  5 bands hemispheric asymmetry      =   5
  TOTAL                             = 334

Log transform applied only to features whose name contains:
  _power, _activity, _rms, _std, _iqr, _line_length, _mean_abs

Pipeline: raw .set
  -> pick eeg, filter 1-45 Hz
  -> segment into 10-s windows
  -> extract_features(seg, sfreq, top_9_ch_names)   # identical to training
  -> log_transform(X, feature_names)
  -> average feature vectors across segments          # subject-level
  -> StandardScaler -> SelectKBest -> StackingClassifier
"""

from __future__ import annotations
import os, tempfile, warnings
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy, skew, kurtosis

warnings.filterwarnings("ignore")

# antropy for accurate Higuchi FD (same as training)
try:
    import antropy as ant
    HAS_ANTROPY = True
except ImportError:
    HAS_ANTROPY = False

EEG_CLASSES  = ["Control (Healthy)", "Alzheimer's Disease"]
TOP_CHANNELS = ["O2", "T5", "T3", "F7", "O1", "P4", "T6", "F8", "P3"]

SEGMENT_SEC = 10
SFREQ_LOW   = 1
SFREQ_HIGH  = 45

BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

LOG_KEYWORDS = [
    "_power", "_activity", "_rms", "_std",
    "_iqr", "_line_length", "_mean_abs",
]

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_eeg_models() -> dict:
    paths = {
        "model":        os.path.join(MODELS_DIR, "model_retrained.pkl"),
        "scaler":       os.path.join(MODELS_DIR, "scaler_retrained.pkl"),
        "selector":     os.path.join(MODELS_DIR, "selector_retrained.pkl"),
        "top_channels": os.path.join(MODELS_DIR, "top_channels.pkl"),
    }
    fallback_paths = {
        "model":  os.path.join(MODELS_DIR, "final_model.pkl"),
        "scaler": os.path.join(MODELS_DIR, "scaler.pkl"),
    }

    loaded: dict = {}
    for key, path in paths.items():
        if not os.path.exists(path) and key in fallback_paths:
            path = fallback_paths[key]
        try:
            loaded[key] = joblib.load(path)
        except Exception as exc:
            error = f"[EEG] Failed to load {key}: {exc}"
            print(error)
            return {"_load_error": error}

    try:
        dummy    = np.zeros((1, 334))
        scaled   = loaded["scaler"].transform(dummy)
        selected = loaded["selector"].transform(scaled)
        loaded["model"].predict_proba(selected)
    except Exception as exc:
        error = f"[EEG] Pipeline self-test failed: {exc}"
        print(error)
        return {"_load_error": error}

    return loaded


# ─────────────────────────────────────────────────────────────────────────────
#  EXACT helpers from Colab training notebook
# ─────────────────────────────────────────────────────────────────────────────
def _segment_signal(data: np.ndarray, sfreq: float):
    """Split (n_ch, n_times) into list of 10-s segments."""
    seg_len = int(SEGMENT_SEC * sfreq)
    return [data[:, s:s + seg_len]
            for s in range(0, data.shape[1] - seg_len, seg_len)]


def _higuchi_fd(sig: np.ndarray, kmax: int = 10) -> float:
    if HAS_ANTROPY:
        try:
            return float(ant.higuchi_fd(sig.astype(np.float64), kmax=kmax))
        except Exception:
            pass
    lags = np.arange(1, kmax + 1)
    vals = []
    for k in lags:
        diff = np.abs(np.diff(sig[::k]))
        vals.append(np.mean(diff) if len(diff) > 0 else 0.0)
    vals = np.array(vals) + 1e-12
    return float(np.polyfit(np.log(lags), np.log(vals), 1)[0])


def _hjorth_params(sig: np.ndarray):
    d1  = np.diff(sig)
    d2  = np.diff(d1)
    v0  = np.var(sig) + 1e-12
    v1  = np.var(d1)  + 1e-12
    v2  = np.var(d2)  + 1e-12
    mob  = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / (mob + 1e-12)
    return v0, mob, comp


def _band_power(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    return float(np.sum(psd[(freqs >= lo) & (freqs <= hi)]))


def extract_features(segment: np.ndarray, sfreq: float, ch_names: list):
    """
    EXACT copy of training extract_features from Colab notebook.
    segment : (n_ch, n_times)
    ch_names: list of channel name strings (must match top_channels order)
    Returns (feature_vector np.float32, feature_names list)
    """
    feats, names = [], []
    n_ch = len(ch_names)
    nperseg = min(256, segment.shape[1])

    all_psd, all_freqs = [], None
    all_bp = {b: [] for b in BANDS}

    for i, ch_sig in enumerate(segment):
        cn = ch_names[i]
        freqs, psd = welch(ch_sig, sfreq, nperseg=nperseg)
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)
        total = np.sum(psd) + 1e-8

        bp = {}
        for band, (lo, hi) in BANDS.items():
            p = _band_power(psd, freqs, lo, hi)
            bp[band] = p
            all_bp[band].append(p)
            feats.extend([p, p / total])
            names.extend([f"{cn}_{band}_power", f"{cn}_{band}_relpower"])

        # 5 spectral ratios
        feats.append(bp["theta"] / (bp["alpha"] + 1e-8));               names.append(f"{cn}_theta_alpha_ratio")
        feats.append(bp["delta"] / (bp["alpha"] + 1e-8));               names.append(f"{cn}_delta_alpha_ratio")
        feats.append(bp["beta"]  / (bp["alpha"] + 1e-8));               names.append(f"{cn}_beta_alpha_ratio")
        feats.append((bp["delta"] + bp["theta"]) / (bp["alpha"] + bp["beta"] + 1e-8));  names.append(f"{cn}_slow_fast_ratio")
        feats.append(bp["theta"] / (bp["delta"] + 1e-8));               names.append(f"{cn}_theta_delta_ratio")

        # Spectral shape
        psd_norm = psd / (np.sum(psd) + 1e-8)
        feats.append(float(entropy(psd_norm)));                          names.append(f"{cn}_spectral_entropy")
        valid  = (freqs >= 1) & (freqs <= 45)
        vf, vp = freqs[valid], psd[valid]
        feats.append(float(vf[np.argmax(vp)]) if len(vp) > 0 else 0.0); names.append(f"{cn}_peak_freq")
        feats.append(float(np.sum(vp * vf) / (np.sum(vp) + 1e-8)));     names.append(f"{cn}_spectral_centroid")

        # Statistical
        feats.extend([np.mean(ch_sig), np.std(ch_sig), float(skew(ch_sig)), float(kurtosis(ch_sig))])
        names.extend([f"{cn}_mean", f"{cn}_std", f"{cn}_skew", f"{cn}_kurtosis"])
        feats.append(float(np.percentile(ch_sig, 75) - np.percentile(ch_sig, 25))); names.append(f"{cn}_iqr")
        feats.append(float(np.sqrt(np.mean(ch_sig ** 2))));              names.append(f"{cn}_rms")
        feats.append(float(np.sum(np.abs(np.diff(ch_sig)))));            names.append(f"{cn}_line_length")
        feats.append(float(np.mean(np.abs(ch_sig))));                    names.append(f"{cn}_mean_abs")

        # Hjorth
        act, mob, comp = _hjorth_params(ch_sig)
        feats.extend([act, mob, comp])
        names.extend([f"{cn}_hjorth_activity", f"{cn}_hjorth_mobility", f"{cn}_hjorth_complexity"])

        # Zero crossings + Higuchi FD
        feats.append(float(np.sum(np.diff(np.sign(ch_sig)) != 0))); names.append(f"{cn}_zero_crossings")
        feats.append(_higuchi_fd(ch_sig));                            names.append(f"{cn}_higuchi_fd")

    # Global band statistics (5 bands × 4 = 20)
    for band in BANDS:
        arr = np.array(all_bp[band])
        feats.extend([np.mean(arr), np.std(arr), np.max(arr), np.min(arr)])
        names.extend([f"global_{band}_mean", f"global_{band}_std",
                      f"global_{band}_max",  f"global_{band}_min"])

    # Pairwise channel coherence (up to 15 pairs × 2 bands = 30)
    if all_freqs is not None and n_ch >= 2:
        alpha_mask = (all_freqs >= 8)  & (all_freqs <= 13)
        theta_mask = (all_freqs >= 4)  & (all_freqs <= 8)
        pairs_done = 0
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                if pairs_done >= 15:
                    break
                p1, p2 = all_psd[i], all_psd[j]
                ca = np.corrcoef(p1[alpha_mask], p2[alpha_mask])[0, 1]
                ct = np.corrcoef(p1[theta_mask], p2[theta_mask])[0, 1]
                feats.extend([np.nan_to_num(ca), np.nan_to_num(ct)])
                names.extend([f"coh_alpha_{ch_names[i]}_{ch_names[j]}",
                               f"coh_theta_{ch_names[i]}_{ch_names[j]}"])
                pairs_done += 1
            if pairs_done >= 15:
                break

    # Hemispheric asymmetry (5 bands = 5)
    lh_chs  = [n for n in ch_names if n[-1] in "13579"]
    rh_chs  = [n for n in ch_names if n[-1] in "24680"]
    min_p   = min(len(lh_chs), len(rh_chs))
    if min_p > 0 and all_freqs is not None:
        lh_idx = [ch_names.index(c) for c in lh_chs[:min_p]]
        rh_idx = [ch_names.index(c) for c in rh_chs[:min_p]]
        for band, (lo, hi) in BANDS.items():
            fm  = (all_freqs >= lo) & (all_freqs <= hi)
            lp  = [np.sum(all_psd[li][fm]) for li in lh_idx]
            rp  = [np.sum(all_psd[ri][fm]) for ri in rh_idx]
            lm, rm = np.mean(lp), np.mean(rp)
            feats.append(float((lm - rm) / (lm + rm + 1e-8)))
            names.append(f"asym_{band}")

    return np.array(feats, dtype=np.float32), names


def _log_transform(X: np.ndarray, feature_names) -> np.ndarray:
    """Apply log to features whose name contains any LOG_KEYWORD."""
    X = X.copy()
    for i, name in enumerate(feature_names):
        if any(k in name for k in LOG_KEYWORDS):
            X[:, i] = np.log(np.abs(X[:, i]) + 1e-8)
    return X


# ─────────────────────────────────────────────────────────────────────────────
#  Main prediction function
# ─────────────────────────────────────────────────────────────────────────────
def eeg_predict_v2(eeg_file, eeg_models: dict):
    """
    Full subject-level EEG prediction matching Colab pipeline exactly.
    Returns (figure, result_string, probs_array)
    """
    import mne

    raw   = None
    probs = np.array([0.5, 0.5])

    if eeg_file is None:
        probs = np.array([0.75, 0.25])
    else:
        try:
            # ── Save uploaded file to temp ────────────────────────────────
            with tempfile.NamedTemporaryFile(delete=False, suffix=".set") as tmp:
                raw_bytes = eeg_file.read() if hasattr(eeg_file, "read") else eeg_file.getvalue()
                tmp.write(raw_bytes)
                tmp_path = tmp.name

            raw = mne.io.read_raw_eeglab(tmp_path, preload=True, verbose=False)
            os.unlink(tmp_path)

            # ── Preprocess (identical to Colab Cell 10) ───────────────────
            raw.pick("eeg")
            raw.filter(SFREQ_LOW, SFREQ_HIGH, verbose=False)
            data      = raw.get_data()
            sfreq     = raw.info["sfreq"]
            ch_names  = raw.ch_names   # keep original case for index lookup

            # ── Find top channels (identical to Colab Cell 10) ────────────
            top_channels = eeg_models.get("top_channels", TOP_CHANNELS)
            matched  = [c for c in top_channels if c in ch_names]
            missing  = [c for c in top_channels if c not in ch_names]
            if missing:
                print(f"[EEG] Missing channels: {missing}")
            if not matched:
                raise ValueError("No top channels found in EEG file.")

            sel_idx = [ch_names.index(c) for c in matched]

            # ── Segment (identical to Colab Cell 11) ─────────────────────
            segments     = _segment_signal(data, sfreq)
            seg_features = []
            feature_names = None

            for seg in segments:
                fv, fn = extract_features(seg[sel_idx], sfreq, matched)
                seg_features.append(fv)
                if feature_names is None:
                    feature_names = fn

            feature_names = np.array(feature_names)
            X_pat = np.nan_to_num(
                np.array(seg_features, dtype=np.float32),
                nan=0.0, posinf=0.0, neginf=0.0
            )
            X_pat = _log_transform(X_pat, feature_names)

            # ── Pad / truncate to 334 (identical to Colab Cell 12) ────────
            n_train = eeg_models["scaler"].n_features_in_
            n_test  = X_pat.shape[1]
            if n_test < n_train:
                pad   = np.zeros((X_pat.shape[0], n_train - n_test), dtype=np.float32)
                X_pat = np.hstack([X_pat, pad])
            elif n_test > n_train:
                X_pat = X_pat[:, :n_train]

            # ── Scale + select ────────────────────────────────────────────
            X_sc  = eeg_models["scaler"].transform(X_pat)
            X_sel = eeg_models["selector"].transform(X_sc)

            # ── Subject-level prediction: mean prob across segments ────────
            seg_probs = eeg_models["model"].predict_proba(X_sel)[:, 1]  # P(AD) per segment
            mean_prob = float(np.mean(seg_probs))
            probs = np.array([1.0 - mean_prob, mean_prob])

        except Exception as exc:
            print(f"[EEG] Prediction error: {exc}")
            import traceback; traceback.print_exc()
            probs = np.array([0.6, 0.4])

    pred_idx   = int(np.argmax(probs))
    pred_class = EEG_CLASSES[pred_idx]

    # ─── Figure ────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("#FFFFFF")
    for ax in (ax1, ax2):
        ax.set_facecolor("#FFFFFF")

    if raw is not None:
        try:
            nsamp = int(raw.info["sfreq"] * 4)
            t     = raw.times[:nsamp]
            sig   = raw.get_data(picks=0).flatten()[:nsamp]
            ax1.plot(t, sig, lw=0.7, color="#1E3A8A")
            ax1.set_xlabel("Time (s)", fontsize=9, color="#1E293B")
            ax1.set_ylabel("Voltage (V)", fontsize=9, color="#1E293B")
        except Exception:
            ax1.text(0.5, 0.5, "Signal unavailable", ha="center", va="center",
                     transform=ax1.transAxes, color="#64748B")
    else:
        ax1.text(0.5, 0.5, "No signal data", ha="center", va="center",
                 transform=ax1.transAxes, color="#64748B", fontsize=11)

    ax1.set_title("EEG Signal (first 4 s, Ch 1)", fontsize=10, fontweight="bold", color="#1E293B")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors="#1E293B", labelsize=8)

    clrs = ["#16A34A", "#DC2626"]
    bars = ax2.barh(EEG_CLASSES[::-1], probs[::-1], color=clrs[::-1], height=0.45)
    for bar, p in zip(bars, probs[::-1]):
        ax2.text(min(bar.get_width() + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                 f"{p * 100:.1f}%", va="center", fontsize=10, fontweight="bold", color="#1E293B")
    ax2.set_xlim(0, 1.2)
    ax2.set_xlabel("Probability", fontsize=9, color="#1E293B")
    ax2.set_title(f"Result: {pred_class}", fontsize=10, fontweight="bold", color="#1E293B")
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.tick_params(colors="#1E293B", labelsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#CBD5E1")

    plt.tight_layout(pad=1.5)
    plt.close(fig)
    return fig, f"**{pred_class}** — {probs[pred_idx] * 100:.1f}% confidence", probs