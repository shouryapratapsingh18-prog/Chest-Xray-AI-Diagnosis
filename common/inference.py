"""
Test-Time Augmentation (TTA) inference for the hybrid model.

We average sigmoid probabilities over a small set of augmented views:
  - original
  - horizontal flip
  - slight center-crop + resize back (two crop ratios)

Grad-CAM for the explanation image is always computed on the plain,
un-augmented (original) view, so the arrow/heatmap lines up with the
image the user uploaded.
"""

from typing import List
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .model import IMG_SIZE, NIH_CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_base_resize = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def _center_crop_resize(img: Image.Image, crop_frac: float) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h))


def make_tta_views(pil_image: Image.Image) -> List[Image.Image]:
    """Returns a list of PIL images representing the TTA views."""
    img = pil_image.convert("RGB")
    views = [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        _center_crop_resize(img, 0.9),
        _center_crop_resize(img, 0.8),
    ]
    return views


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """Single image -> normalized (1, 3, 224, 224) tensor."""
    return _base_resize(pil_image.convert("RGB")).unsqueeze(0)


def predict_tta(model, pil_image: Image.Image, device: str = "cpu"):
    """
    Runs TTA inference. Returns:
        mean_probs: np.ndarray, shape (num_classes,)
        top_idx: index of the highest-probability class
    Does NOT compute Grad-CAM (that needs a single, gradient-tracked
    forward pass -- see gradcam.explain, run separately on the
    original, non-augmented view).
    """
    views = make_tta_views(pil_image)
    all_probs = []
    model.eval()
    with torch.no_grad():
        for v in views:
            tensor = _base_resize(v).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            all_probs.append(probs)
    mean_probs = np.mean(all_probs, axis=0)
    top_idx = int(np.argmax(mean_probs))
    return mean_probs, top_idx


def probs_to_report(probs: np.ndarray, threshold: float = 0.5):
    """Turns a probability vector into a sorted list of dicts for the UI."""
    report = [
        {"label": NIH_CLASSES[i], "probability": float(probs[i]), "positive": bool(probs[i] >= threshold)}
        for i in range(len(NIH_CLASSES))
    ]
    report.sort(key=lambda x: x["probability"], reverse=True)
    return report
