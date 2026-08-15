"""
Dataset loader for the Kaggle dataset "NIH Chest X-rays (224x224 resized)".

Expected structure once you add the dataset via Kaggle "+ Add Input"
(the exact folder name may vary slightly by dataset version -- check
the left "Data" panel in your notebook and adjust IMAGE_DIR_CANDIDATES
/ CSV_PATH_CANDIDATES below if needed):

/kaggle/input/<dataset-name>/
    Data_Entry_2017.csv
    images/               (or images_224/, all .png flattened in one folder)

This loader:
  1. Reads Data_Entry_2017.csv
  2. Multi-hot encodes the "Finding Labels" column into the 14 NIH classes
     (drops "No Finding" rows' label but keeps the image as an
     all-zero row, i.e. a healthy example -- useful negative signal)
  3. Splits train/val by patient ID (avoids leakage of the same patient
     into both splits)
"""

import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.model import NIH_CLASSES


def find_dataset_paths():
    """Auto-detect the CSV and image folder under /kaggle/input/."""
    csv_candidates = glob.glob("/kaggle/input/**/Data_Entry_2017.csv", recursive=True)
    if not csv_candidates:
        raise FileNotFoundError(
            "Could not find Data_Entry_2017.csv under /kaggle/input/. "
            "Make sure you added the NIH ChestX-ray14 (224x224) dataset via '+ Add Input'."
        )
    csv_path = csv_candidates[0]
    dataset_root = os.path.dirname(csv_path)

    # image folder can be named differently across dataset versions
    img_dir = None
    for cand in ["images", "images_224", "Images", "images-224"]:
        p = os.path.join(dataset_root, cand)
        if os.path.isdir(p):
            img_dir = p
            break
    if img_dir is None:
        # fall back: search recursively for a folder that actually contains .png files
        for root, dirs, files in os.walk(dataset_root):
            if any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in files):
                img_dir = root
                break
    if img_dir is None:
        raise FileNotFoundError(f"Could not locate an image folder under {dataset_root}")

    return csv_path, img_dir


def build_label_matrix(df: pd.DataFrame) -> np.ndarray:
    labels = np.zeros((len(df), len(NIH_CLASSES)), dtype=np.float32)
    for i, findings in enumerate(df["Finding Labels"]):
        if findings == "No Finding":
            continue
        for f in findings.split("|"):
            f = f.strip()
            if f in NIH_CLASSES:
                labels[i, NIH_CLASSES.index(f)] = 1.0
    return labels


class NIHChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = build_label_matrix(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image Index"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def load_and_split(val_frac: float = 0.15, seed: int = 42):
    """Returns (train_df, val_df, img_dir) split by Patient ID to avoid leakage."""
    csv_path, img_dir = find_dataset_paths()
    df = pd.read_csv(csv_path)

    patient_ids = df["Patient ID"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)
    n_val = int(len(patient_ids) * val_frac)
    val_ids = set(patient_ids[:n_val])

    val_df = df[df["Patient ID"].isin(val_ids)]
    train_df = df[~df["Patient ID"].isin(val_ids)]
    return train_df, val_df, img_dir
