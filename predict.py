"""
Chest X-Ray Prediction + Grad-CAM Visualization

Model:
    EfficientNetV2-S + Swin-Tiny

Checkpoint:
    FINAL_BEST_EPOCH5_AUC_0.76148.pth

Output:
    - Disease probabilities
    - Top prediction
    - Grad-CAM heatmap
    - Original vs Grad-CAM comparison with refined prominent arrows
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from common.model import (
    load_trained_model,
    NIH_CLASSES,
    IMG_SIZE
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WEIGHTS_PATH = (
    "models/FINAL_BEST_EPOCH5_AUC_0.76148.pth"
)

IMAGE_PATH = "test_xray.png"

OUTPUT_DIR = "outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("CHEST X-RAY AI")
print("=" * 70)

print()
print("Device:", DEVICE)

print()
print("Loading model...")

model = load_trained_model(
    WEIGHTS_PATH,
    device=DEVICE,
    num_classes=len(NIH_CLASSES)
)

model.eval()

print()
print("Model loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

print()
print("Loading image:")
print(IMAGE_PATH)

if not os.path.exists(IMAGE_PATH):

    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


original_image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ============================================================
# CREATE MODEL INPUT
# ============================================================

image_tensor = transform(
    original_image
).unsqueeze(0)

image_tensor = image_tensor.to(
    DEVICE
)


# ============================================================
# FORWARD PASS
#
# IMPORTANT:
# DO NOT use torch.no_grad()
# because Grad-CAM needs gradients.
# ============================================================

model.zero_grad()

logits = model(
    image_tensor
)

probabilities = torch.sigmoid(
    logits
)[0]


# ============================================================
# CONVERT TO NUMPY
# ============================================================

probabilities_np = (
    probabilities
    .detach()
    .cpu()
    .numpy()
)


# ============================================================
# SORT PREDICTIONS
# ============================================================

sorted_indices = np.argsort(
    probabilities_np
)[::-1]


# ============================================================
# PRINT PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("CHEST X-RAY PREDICTIONS")
print("=" * 70)

for idx in sorted_indices:

    disease = NIH_CLASSES[idx]

    probability = (
        probabilities_np[idx] * 100
    )

    print(
        f"{disease:<25} : "
        f"{probability:6.2f}%"
    )


# ============================================================
# TOP PREDICTION
# ============================================================

target_class = sorted_indices[0]

target_name = NIH_CLASSES[
    target_class
]

target_probability = (
    probabilities_np[target_class] * 100
)


print()
print("=" * 70)
print("TOP PREDICTION")
print("=" * 70)

print()
print("Disease     :", target_name)

print(
    "Confidence  : "
    f"{target_probability:.2f}%"
)


# ============================================================
# GRAD-CAM
# ============================================================

print()
print("=" * 70)
print("GENERATING GRAD-CAM")
print("=" * 70)


# ------------------------------------------------------------
# Select the output corresponding to the top prediction
# ------------------------------------------------------------

target_logit = logits[
    0,
    target_class
]


# ------------------------------------------------------------
# Backpropagate
# ------------------------------------------------------------

model.zero_grad()

target_logit.backward(
    retain_graph=True
)


# ============================================================
# GET EFFICIENTNET FEATURE MAP
# ============================================================

feature_map = (
    model.last_eff_feature_map
)

if feature_map is None:

    raise RuntimeError(
        "Grad-CAM feature map is None. "
        "Check model.py."
    )


# ============================================================
# GET GRADIENTS
# ============================================================

gradients = feature_map.grad

if gradients is None:

    raise RuntimeError(
        "Gradients are None. "
        "Grad-CAM cannot be generated."
    )


# ============================================================
# GRAD-CAM CALCULATION
# ============================================================

feature_map = feature_map[
    0
].detach()

gradients = gradients[
    0
].detach()


# ------------------------------------------------------------
# Global average pooling of gradients
# ------------------------------------------------------------

weights = gradients.mean(
    dim=(1, 2)
)


# ------------------------------------------------------------
# Weighted combination of feature maps
# ------------------------------------------------------------

cam = torch.zeros(
    feature_map.shape[1:],
    device=feature_map.device
)


for i in range(
    feature_map.shape[0]
):

    cam += (
        weights[i] *
        feature_map[i]
    )


# ============================================================
# RELU
# ============================================================

cam = torch.relu(
    cam
)


# ============================================================
# NORMALIZE CAM
# ============================================================

cam = (
    cam -
    cam.min()
)

cam = (
    cam /
    (cam.max() + 1e-8)
)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

cam_np = (
    cam
    .cpu()
    .numpy()
)


# ============================================================
# RESIZE CAM TO ORIGINAL IMAGE
# ============================================================

original_width, original_height = (
    original_image.size
)

cam_image = Image.fromarray(
    np.uint8(
        cam_np * 255
    )
)

cam_image = cam_image.resize(
    (
        original_width,
        original_height
    ),
    Image.Resampling.BILINEAR
)

cam_np = (
    np.asarray(
        cam_image
    ).astype(
        np.float32
    ) / 255.0
)


# ============================================================
# FIND PEAK ACTIVATION COORDINATE FOR ARROW
# ============================================================

y_max, x_max = np.unravel_index(
    np.argmax(cam_np), 
    cam_np.shape
)


# ============================================================
# CREATE MATPLOTLIB FIGURE
# ============================================================

fig = plt.figure(
    figsize=(16, 7)
)


# ============================================================
# ORIGINAL IMAGE
# ============================================================

ax1 = plt.subplot(
    1,
    3,
    1
)

ax1.imshow(
    original_image,
    cmap="gray"
)

ax1.set_title(
    "Original Chest X-Ray",
    fontsize=14,
    fontweight="bold"
)

ax1.axis("off")


# ============================================================
# GRAD-CAM HEATMAP
# ============================================================

ax2 = plt.subplot(
    1,
    3,
    2
)

ax2.imshow(
    original_image,
    cmap="gray"
)

ax2.imshow(
    cam_np,
    cmap="jet",
    alpha=0.50
)

# Refined and sized-up main arrow (Yellow/Black contrast for high visibility)
ax2.annotate(
    "Peak Focus",
    xy=(x_max, y_max),
    xytext=(x_max + 45, y_max + 45),
    arrowprops=dict(
        facecolor="yellow", 
        edgecolor="black", 
        shrink=0.08, 
        width=2, 
        headwidth=9
    ),
    fontsize=9,
    color="yellow",
    weight="bold"
)

ax2.set_title(
    f"Grad-CAM\n{target_name} ({target_probability:.2f}%)",
    fontsize=14,
    fontweight="bold"
)

ax2.axis("off")


# ============================================================
# CLEAN OVERLAY
# ============================================================

ax3 = plt.subplot(
    1,
    3,
    3
)

ax3.imshow(
    original_image,
    cmap="gray"
)

heatmap = ax3.imshow(
    cam_np,
    cmap="jet",
    alpha=0.45
)

# Refined and sized-up main arrow on Overlay
ax3.annotate(
    "Peak Focus",
    xy=(x_max, y_max),
    xytext=(x_max + 45, y_max + 45),
    arrowprops=dict(
        facecolor="yellow", 
        edgecolor="black", 
        shrink=0.08, 
        width=2, 
        headwidth=9
    ),
    fontsize=9,
    color="yellow",
    weight="bold"
)

ax3.set_title(
    "Model Attention Overlay",
    fontsize=14,
    fontweight="bold"
)

ax3.axis("off")

plt.colorbar(
    heatmap,
    ax=ax3,
    fraction=0.046,
    pad=0.04
)


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    (
        "Chest X-Ray AI Analysis\n"
        f"Top Prediction: {target_name} "
        f"({target_probability:.2f}%)"
    ),
    fontsize=17,
    fontweight="bold"
)


plt.tight_layout()


# ============================================================
# SAVE RESULT
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_comparison.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)


# ============================================================
# DISPLAY
# ============================================================

plt.show()


# ============================================================
# SAVE HEATMAP SEPARATELY
# ============================================================

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_heatmap.png"
)

plt.imsave(
    heatmap_path,
    cam_np,
    cmap="jet"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("GRAD-CAM COMPLETE")
print("=" * 70)

print()
print("Top prediction:")
print(
    f"{target_name} "
    f"({target_probability:.2f}%)"
)

print()
print("Comparison saved to:")
print(
    output_path
)

print()
print("Heatmap saved to:")
print(
    heatmap_path
)

print()
print("=" * 70)