"""
Grad-CAM on the EfficientNetV2 branch's last spatial feature map, plus a
simple arrow annotation pointing at the most-important region so a
non-technical viewer can see at a glance *where* the model looked.

Usage:
    heatmap, overlay_img, arrow_img = explain(model, input_tensor, class_idx, orig_pil_image)
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


def compute_gradcam(model, input_tensor, class_idx):
    """
    Runs a forward + backward pass and returns a (H, W) numpy heatmap
    in the range [0, 1], resized to the model's input feature-map size.
    `input_tensor` must be a single image, shape (1, 3, H, W), with
    requires_grad not needed (we only need grads on the feature map).
    """
    model.zero_grad()
    logits = model(input_tensor)               # (1, num_classes)
    score = logits[0, class_idx]
    score.backward(retain_graph=True)

    fmap = model.last_eff_feature_map           # (1, C, h, w)
    grads = fmap.grad                            # (1, C, h, w)

    # Global-average-pool the gradients -> per-channel importance weight
    weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
    cam = (weights * fmap).sum(dim=1, keepdim=True)  # (1, 1, h, w)
    cam = F.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, torch.sigmoid(logits).detach().cpu().numpy()[0]


def overlay_heatmap(orig_pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    """Resize cam to the original image size and blend as a heatmap overlay."""
    orig = np.array(orig_pil_image.convert("RGB"))
    h, w = orig.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(orig * (1 - alpha) + heatmap * alpha)
    return Image.fromarray(overlay), cam_resized


def draw_arrow_to_hotspot(orig_pil_image: Image.Image, cam_resized: np.ndarray, min_area_frac: float = 0.01):
    """
    Finds the centroid of the strongest activation region (thresholded
    Grad-CAM) and draws an arrow pointing at it, plus a circle around
    the region, so the affected area is obvious even without reading a
    heatmap.
    """
    img = np.array(orig_pil_image.convert("RGB")).copy()
    h, w = img.shape[:2]

    thresh = (cam_resized > 0.6).astype(np.uint8) * 255
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # fall back to the single brightest pixel
        y, x = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)
        cx, cy = int(x), int(y)
        radius = int(0.08 * min(h, w))
    else:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area_frac * h * w:
            largest = max(contours, key=cv2.contourArea)  # keep it anyway, still show something
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        cx, cy, radius = int(cx), int(cy), max(int(radius), int(0.05 * min(h, w)))

    # Arrow tail: pick a corner far from the hotspot so the arrow doesn't
    # overlap the region itself.
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    tail = max(corners, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)

    # Stop the arrowhead just outside the circle, not on top of it.
    dx, dy = cx - tail[0], cy - tail[1]
    dist = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
    tip = (int(cx - dx / dist * (radius + 8)), int(cy - dy / dist * (radius + 8)))

    color = (255, 0, 0)
    cv2.circle(img, (cx, cy), radius, color, 3)
    cv2.arrowedLine(img, tail, tip, color, 3, tipLength=0.12)

    return Image.fromarray(img), (cx, cy, radius)


def explain(model, input_tensor, class_idx, orig_pil_image):
    """
    Full pipeline: Grad-CAM -> heatmap overlay image -> arrow-annotated image.
    Returns (raw_cam, probs, heatmap_overlay_pil, arrow_annotated_pil, hotspot_xyr)
    """
    cam, probs = compute_gradcam(model, input_tensor, class_idx)
    overlay_img, cam_resized = overlay_heatmap(orig_pil_image, cam)
    arrow_img, hotspot = draw_arrow_to_hotspot(orig_pil_image, cam_resized)
    return cam, probs, overlay_img, arrow_img, hotspot
