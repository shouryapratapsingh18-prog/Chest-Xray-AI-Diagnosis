# Chest X-ray AI Diagnosis Assistant

EfficientNetV2-S + Swin-Small hybrid model, trained on **NIH Chest X-ray 14**,
with Test-Time Augmentation (TTA) and Grad-CAM based explainability
(heatmap + arrow pointing at the region that drove the prediction).

```
chest-xray-project/
├── common/                 # shared model + Grad-CAM + inference code
│   ├── model.py             # HybridEffNetSwin architecture
│   ├── gradcam.py            # Grad-CAM + arrow annotation
│   └── inference.py          # TTA prediction pipeline
├── kaggle_train/
│   ├── train_notebook.ipynb  # <-- upload/run this on Kaggle
│   ├── dataset.py             # standalone dataset loader (reference)
│   └── requirements.txt
└── webapp/
    ├── app.py                 # Flask backend (upload -> predict -> explain)
    ├── templates/index.html
    ├── static/style.css, script.js
    ├── weights/                # <-- put best_model.pth here after training
    └── requirements.txt
```

## Step 1 — Train on Kaggle

1. Go to kaggle.com → **New Notebook**.
2. Upload / open `kaggle_train/train_notebook.ipynb`.
3. Right sidebar → **+ Add Input** → search **"NIH Chest X-rays (224x224 resized)"**
   (or the original NIH ChestX-ray14 dataset) → add it.
4. Notebook **Settings**: turn on **GPU** (T4 x2 / P100) and **Internet**
   (needed once, to download ImageNet-pretrained EfficientNetV2/Swin
   weights via `timm`).
5. Run all cells. Training takes a while (15 epochs by default — reduce
   `EPOCHS` in the config cell if you want a faster/rough first run).
6. When done, open the **Output** tab and download `best_model.pth`.

## Step 2 — Run the webapp locally

1. Put the downloaded file at `webapp/weights/best_model.pth`.
2. Install dependencies:
   ```bash
   cd webapp
   pip install -r requirements.txt
   ```
   (A GPU is optional for inference — CPU works fine for single-image
   prediction, just a bit slower.)
3. Run:
   ```bash
   python app.py
   ```
4. Open `http://localhost:5000` in your browser, upload a chest X-ray
   image, and click **Analyze X-ray**.

You'll get:
- The top predicted condition + probability
- A full bar chart of all 14 NIH classes
- A Grad-CAM heatmap overlay
- The same image with a **red arrow + circle** pointing at the region
  the model focused on for its top prediction

## Notes / things you may want to tune

- **Class list & order** live in `common/model.py` (`NIH_CLASSES`) — keep
  it identical between training and the webapp (they already share the
  same file, so this is automatic as long as you don't edit only one copy).
- **Threshold** for "positive" in the UI is 0.5 sigmoid probability
  (`common/inference.py: probs_to_report`) — NIH14 models are often
  better read as ranked risk scores rather than hard yes/no, feel free
  to adjust.
- **TTA views** (`common/inference.py: make_tta_views`): original,
  horizontal flip, and two center-crop scales. Add/remove views there.
- **Grad-CAM branch**: computed on the EfficientNetV2 branch's last
  conv feature map (`common/gradcam.py`) since it's spatial and easy to
  visualize; Swin's windowed attention is not used for the heatmap.
- If Kaggle's dataset folder names differ from what `dataset.py` /
  the notebook auto-detect, check the notebook's **Data** panel for the
  exact path and adjust the `find_dataset_paths()` candidates.
- This is a research/education demo, **not a medical device** — do not
  use it for real diagnosis.
