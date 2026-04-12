"""fusion_module.py"""
from __future__ import annotations
import os, tempfile
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .clinical_module import clinical_risk_score
from .mri_module import FusionNet, overlay_gradcam, generate_gradcam

INTEGRATED_CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
_AUC = {"Clinical": 0.95, "MRI": 0.98, "EEG": 0.743}


def _weights():
    raw   = {k: max(v - 0.5, 0.01) for k, v in _AUC.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def _clinical_probs(mmse, func, mem, behav, adl):
    p = clinical_risk_score(mmse, func, mem, behav, adl)
    return np.array([1-p, p*0.6, p*0.3, p*0.1])


def _mri_probs(image: Image.Image, model: FusionNet):
    from .mri_module import _prepare_tensor
    img_t = _prepare_tensor(image)
    tab_t = torch.tensor([0.5]*32).float().unsqueeze(0)
    with torch.no_grad():
        out = model(img_t, tab_t)
    return F.softmax(out, dim=1).numpy()[0], img_t


def _eeg_probs(eeg_file, eeg_models, fallback):
    import mne
    from .eeg_module import (
        _segment_signal, extract_features, _log_transform, TOP_CHANNELS,
        SFREQ_LOW, SFREQ_HIGH
    )
    if eeg_file is None:
        return fallback.copy(), False

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".set") as tmp:
            data_b = eeg_file.getvalue() if hasattr(eeg_file, "getvalue") else eeg_file.read()
            tmp.write(data_b); tmp_path = tmp.name

        raw = mne.io.read_raw_eeglab(tmp_path, preload=True, verbose=False)
        raw.pick("eeg")
        raw.filter(SFREQ_LOW, SFREQ_HIGH, verbose=False)
        data, sfreq, ch_names = raw.get_data(), raw.info["sfreq"], raw.ch_names

        top_ch  = eeg_models.get("top_channels", TOP_CHANNELS)
        matched = [c for c in top_ch if c in ch_names]
        if not matched:
            return fallback.copy(), False
        sel_idx = [ch_names.index(c) for c in matched]

        segments, feature_names, seg_features = _segment_signal(data, sfreq), None, []
        for seg in segments:
            fv, fn = extract_features(seg[sel_idx], sfreq, matched)
            seg_features.append(fv)
            if feature_names is None: feature_names = fn

        feature_names = np.array(feature_names)
        X = np.nan_to_num(np.array(seg_features, dtype=np.float32), nan=0., posinf=0., neginf=0.)
        X = _log_transform(X, feature_names)
        n_train = eeg_models["scaler"].n_features_in_
        if X.shape[1] < n_train:
            X = np.hstack([X, np.zeros((X.shape[0], n_train - X.shape[1]), dtype=np.float32)])
        elif X.shape[1] > n_train:
            X = X[:, :n_train]

        X_sc   = eeg_models["scaler"].transform(X)
        X_sel  = eeg_models["selector"].transform(X_sc)
        p_ad   = float(np.mean(eeg_models["model"].predict_proba(X_sel)[:, 1]))
        return np.array([1-p_ad, p_ad*0.6, p_ad*0.3, p_ad*0.1]), True

    except Exception as exc:
        print(f"[Fusion/EEG] error: {exc}")
        noise = np.random.normal(0, 0.02, 4)
        p = np.clip(fallback + noise, 0, 1)
        return p / p.sum(), False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_integrated_analysis(mri_img, eeg_file, clinical_data, fusion_model, eeg_models):
    mmse, func, mem, behav, adl = clinical_data
    w        = _weights()
    clin_p   = _clinical_probs(mmse, func, mem, behav, adl)
    mri_p, img_tensor = _mri_probs(mri_img, fusion_model)
    eeg_p, eeg_ok     = _eeg_probs(eeg_file, eeg_models, clin_p)

    fused = clin_p*w["Clinical"] + mri_p*w["MRI"] + eeg_p*w["EEG"]
    fused = fused / (fused.sum() + 1e-12)

    pred_idx = int(np.argmax(fused))
    cam, _   = generate_gradcam(fusion_model, img_tensor, target_class=pred_idx)
    img_rgb  = mri_img.convert("RGB")
    overlay  = overlay_gradcam(img_rgb, cam)

    return fused, w, overlay, np.array(img_rgb.resize((224, 224))), eeg_ok