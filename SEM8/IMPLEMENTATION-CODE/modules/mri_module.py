"""
mri_module.py  —  MRI inference + Grad-CAM (FusionNet)
Restores exact original mri_fusion_05 logic + adds Grad-CAM to standalone MRI tab.
"""
from __future__ import annotations
import os, warnings
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")

MRI_CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


# ── Architecture (must match training exactly) ────────────────────────────────
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        base = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.proj     = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2048, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())
    def forward(self, x): return self.proj(self.features(x))


class TabularEncoder(nn.Module):
    def __init__(self, in_dim=32, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())
    def forward(self, x): return self.net(x)


class FusionNet(nn.Module):
    def __init__(self, tabular_dim=32, num_classes=4):
        super().__init__()
        self.img_enc     = ImageEncoder(512)
        self.tab_enc     = TabularEncoder(tabular_dim, 256)
        self.fusion_head = nn.Sequential(
            nn.Linear(512 + 256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes))
    def forward(self, img, tab):
        return self.fusion_head(torch.cat([self.img_enc(img), self.tab_enc(tab)], dim=1))


def load_fusion_model() -> FusionNet:
    checkpoint = torch.load(os.path.join(MODELS_DIR, "best_model.pth"), map_location="cpu")
    model = FusionNet(tabular_dim=32, num_classes=4)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def generate_gradcam(model: FusionNet, img_tensor: torch.Tensor, target_class=None):
    model.eval()
    target_layer   = model.img_enc.features[7][-1]
    activated, gradients = [], []
    h_fwd = target_layer.register_forward_hook(lambda m, i, o: activated.append(o))
    h_bwd = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    dummy_tab = torch.tensor([0.5] * 32).float().unsqueeze(0)
    output    = model(img_tensor, dummy_tab)
    if target_class is None:
        target_class = int(output.argmax(dim=1).item())

    model.zero_grad()
    output[0, target_class].backward()

    grads   = gradients[0].cpu().detach().numpy().squeeze()
    fmaps   = activated[0].cpu().detach().numpy().squeeze()
    weights = np.mean(grads, axis=(1, 2))
    cam     = np.zeros(fmaps.shape[1:], dtype=np.float32)
    for w, fmap in zip(weights, fmaps):
        cam += w * fmap
    cam   = np.maximum(cam, 0)
    cam   = cv2.resize(cam, (224, 224))
    denom = cam.max() - cam.min()
    cam   = (cam - cam.min()) / (denom if denom != 0 else 1.0)
    h_fwd.remove(); h_bwd.remove()
    return cam, target_class


def overlay_gradcam(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    arr = np.array(pil_img.resize((224, 224)))
    if arr.ndim == 2:       arr = np.stack([arr] * 3, axis=2)
    if arr.shape[2] == 4:   arr = arr[:, :, :3]
    hm  = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    hm  = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(arr, 1 - alpha, hm, alpha, 0)


# ── Image preprocessing ───────────────────────────────────────────────────────
def _prepare_tensor(image: Image.Image):
    """Exact original mri_fusion_05 preprocessing (no forced RGB)."""
    img   = image.resize((224, 224))
    arr   = np.array(img) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)


# ── Inference ─────────────────────────────────────────────────────────────────
def mri_predict(image: Image.Image, model: FusionNet):
    """
    Runs MRI inference + Grad-CAM.
    Returns (figure, label_string, probs, img_tensor)
    Figure shows: original | Grad-CAM overlay | probability bars
    """
    img_tensor      = _prepare_tensor(image)
    clinical_tensor = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5] + [0.0] * 27).float().unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor, clinical_tensor)
        probs  = F.softmax(output, dim=1).numpy()[0]

    pred_idx   = int(np.argmax(probs))
    pred_class = MRI_CLASSES[pred_idx]

    # Generate Grad-CAM for predicted class
    cam, _ = generate_gradcam(model, img_tensor, target_class=pred_idx)
    img_rgb = image.resize((224, 224)).convert("RGB")
    overlay = overlay_gradcam(img_rgb, cam)

    # ── 3-panel figure ────────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("#FFFFFF")
    for ax in (ax1, ax2):
        ax.set_facecolor("#000000")
    ax3.set_facecolor("#FFFFFF")

    ax1.imshow(img_rgb); ax1.axis("off")
    ax1.set_title("Input MRI", fontsize=10, fontweight="bold", color="#1E293B", pad=6)

    ax2.imshow(overlay); ax2.axis("off")
    ax2.set_title("Grad-CAM Active Zones", fontsize=10, fontweight="bold", color="#1E293B", pad=6)

    bar_clrs = ["#3B82F6", "#DC2626", "#16A34A", "#D97706"]
    bars = ax3.barh(MRI_CLASSES[::-1], probs[::-1], color=bar_clrs[::-1], height=0.45)
    for bar, p in zip(bars, probs[::-1]):
        ax3.text(min(bar.get_width() + 0.015, 0.99),
                 bar.get_y() + bar.get_height() / 2,
                 f"{p * 100:.1f}%", va="center", fontsize=9,
                 fontweight="bold", color="#1E293B")
    ax3.set_xlim(0, 1.2)
    ax3.set_xlabel("Probability", fontsize=9, color="#1E293B")
    ax3.set_title(f"Result: {pred_class} ({probs[pred_idx]*100:.1f}%)",
                  fontsize=10, fontweight="bold", color="#1E293B")
    ax3.grid(True, axis="x", alpha=0.3)
    ax3.tick_params(colors="#1E293B", labelsize=9)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#CBD5E1")

    plt.tight_layout(pad=1.5)
    plt.close(fig)

    return fig, f"**{pred_class}** — {probs[pred_idx]*100:.1f}% confidence", probs, img_tensor