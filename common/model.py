"""
Hybrid EfficientNetV2-S + Swin-Tiny model for multi-label
NIH ChestX-ray14 classification.

IMPORTANT:
This architecture must match the architecture used during
Kaggle training.

Trained model:
    EfficientNetV2-S + Swin-Tiny
    Image size: 224
    Number of classes: 14
"""

import torch
import torch.nn as nn
import timm


# ============================================================
# NIH CHEST X-RAY 14 CLASSES
# ============================================================

NIH_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


# ============================================================
# IMAGE SIZE
# ============================================================

IMG_SIZE = 224


# ============================================================
# HYBRID MODEL
# ============================================================

class HybridEffNetSwin(nn.Module):

    def __init__(
        self,
        num_classes=14,
        effnet_name="tf_efficientnetv2_s",
        swin_name="swin_tiny_patch4_window7_224",
        pretrained=True,
        dropout=0.3,
    ):
        super().__init__()

        # ====================================================
        # EFFICIENTNETV2-S
        # ====================================================

        self.effnet = timm.create_model(
            effnet_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,)
        )

        eff_channels = self.effnet.feature_info[-1]["num_chs"]

        # ====================================================
        # SWIN-TINY
        #
        # IMPORTANT:
        # The trained Kaggle model uses Swin-Tiny.
        # ====================================================

        self.swin = timm.create_model(
            swin_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        swin_dim = self.swin.num_features

        # ====================================================
        # GLOBAL AVERAGE POOLING
        # ====================================================

        self.eff_pool = nn.AdaptiveAvgPool2d(1)

        # ====================================================
        # FEATURE FUSION
        # ====================================================

        fusion_dim = eff_channels + swin_dim

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
                512
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                512,
                num_classes
            )
        )

        # ====================================================
        # GRAD-CAM FEATURE MAP
        # ====================================================

        self.last_eff_feature_map = None

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # EfficientNet branch
        # ----------------------------------------------------

        eff_feat_map = self.effnet(x)[-1]

        # Keep feature map for Grad-CAM
        if eff_feat_map.requires_grad:
            eff_feat_map.retain_grad()

        self.last_eff_feature_map = eff_feat_map

        # Global average pooling

        eff_vec = self.eff_pool(
            eff_feat_map
        ).flatten(1)

        # ----------------------------------------------------
        # Swin-Tiny branch
        # ----------------------------------------------------

        swin_vec = self.swin(x)

        # ----------------------------------------------------
        # Feature fusion
        # ----------------------------------------------------

        fused = torch.cat(
            [
                eff_vec,
                swin_vec
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        logits = self.classifier(fused)

        return logits


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(
    num_classes=14,
    pretrained=True
):

    model = HybridEffNetSwin(
        num_classes=num_classes,
        pretrained=pretrained
    )

    return model


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_trained_model(
    weights_path,
    device="cpu",
    num_classes=14
):

    print("=" * 60)
    print("LOADING TRAINED CHEST X-RAY MODEL")
    print("=" * 60)

    print()
    print("Weights:")
    print(weights_path)

    print()
    print("Device:")
    print(device)

    print()
    print("Classes:")
    print(num_classes)

    # ========================================================
    # BUILD ARCHITECTURE
    #
    # pretrained=False because the checkpoint already
    # contains the trained parameters.
    # ========================================================

    model = build_model(
        num_classes=num_classes,
        pretrained=False
    )

    # ========================================================
    # LOAD CHECKPOINT
    #
    # PyTorch 2.6 changed the default behavior of torch.load()
    # to weights_only=True.
    #
    # Your trusted Kaggle checkpoint contains additional
    # serialized objects, so weights_only=False is required.
    # ========================================================

    print()
    print("Loading checkpoint...")

    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False
    )

    print()
    print("Checkpoint type:")
    print(type(checkpoint))

    # ========================================================
    # EXTRACT MODEL STATE DICTIONARY
    # ========================================================

    if isinstance(checkpoint, dict):

        print()
        print("Checkpoint keys:")

        for key in checkpoint.keys():
            print(" -", key)

        # Standard checkpoint format

        if "model_state_dict" in checkpoint:

            print()
            print("Using: model_state_dict")

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            print()
            print("Using: state_dict")

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            print()
            print(
                "No model_state_dict found."
            )

            print(
                "Assuming checkpoint itself "
                "is the state dictionary."
            )

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # ========================================================
    # HANDLE DATA PARALLEL CHECKPOINTS
    #
    # If the model was trained using DataParallel,
    # parameters may start with:
    #
    # module.effnet...
    #
    # Remove "module." if necessary.
    # ========================================================

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            new_key = key[
                len("module."):]
        else:

            new_key = key

        cleaned_state_dict[
            new_key
        ] = value

    state_dict = cleaned_state_dict

    # ========================================================
    # LOAD WEIGHTS
    # ========================================================

    print()
    print("Loading trained parameters...")

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False
    )

    # ========================================================
    # REPORT LOADING INFORMATION
    # ========================================================

    print()

    if len(missing_keys) == 0:

        print(
            "Missing keys: NONE"
        )

    else:

        print(
            "WARNING - Missing keys:"
        )

        for key in missing_keys:

            print(
                "  ",
                key
            )

    print()

    if len(unexpected_keys) == 0:

        print(
            "Unexpected keys: NONE"
        )

    else:

        print(
            "WARNING - Unexpected keys:"
        )

        for key in unexpected_keys:

            print(
                "  ",
                key
            )

    # ========================================================
    # MOVE MODEL TO DEVICE
    # ========================================================

    model.to(device)

    model.eval()

    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print("=" * 60)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 60)

    print()
    print("Architecture:")
    print(
        "EfficientNetV2-S + Swin-Tiny"
    )

    print()
    print("Number of classes:")
    print(num_classes)

    print()
    print("Image size:")
    print(IMG_SIZE)

    print()
    print("Device:")
    print(device)

    print()
    print("=" * 60)

    return model