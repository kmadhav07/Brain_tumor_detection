# 🧠 Brain Tumor Detection — Modern PyTorch

A binary MRI classifier that detects the presence of brain tumors using a
fine-tuned **ResNet-50** model, implemented entirely in **PyTorch**.

---

## 📁 Project Structure

```
Vision-Model/
├── brain-tumor-detection-pytorch.ipynb   ← Run this notebook
├── brain_tumor_pytorch.py                ← Equivalent Python script
├── best_brain_tumor_model.pth            ← Saved after training
└── PROJECT_DETAILS.md                    ← This file
```

## 🔧 Tech Stack

| Component | Choice |
|---|---|
| Framework | PyTorch ≥ 2.0 |
| Backbone | `torchvision.models.resnet50` (ImageNet V2 weights) |
| Data Pipeline | `torchvision.datasets.ImageFolder` + `DataLoader` |
| Augmentation | Random flips, rotation, colour jitter, brain contour crop |
| Optimiser | Adam (`lr = 1e-4`, `weight_decay = 1e-4`) |
| Scheduler | `ReduceLROnPlateau` (factor 0.5, patience 2) |
| Loss | `CrossEntropyLoss` |
| Early Stopping | Manual, patience = 5 epochs |

## 🧬 Model Architecture

```
ResNet-50 (frozen convolutional backbone)
└── Custom Classification Head
    ├── Dropout (0.4)
    ├── Linear (2048 → 512)
    ├── ReLU
    ├── Dropout (0.2)
    └── Linear (512 → 2)
```

- **Total parameters:** ~24.6 M
- **Trainable parameters:** ~1.1 M (head only)
- The frozen backbone extracts powerful features learned from 1.2 M ImageNet images;
  only the lightweight head is trained on brain MRI data.

## 🖼️ Preprocessing Pipeline

1. **Brain Contour Crop** — OpenCV finds the largest contour in the MRI scan
   and crops to its bounding box, removing irrelevant black borders.
2. **Resize** — to 224 × 224 pixels.
3. **Augmentation** (training only) — horizontal/vertical flips, ±20° rotation,
   brightness/contrast jitter.
4. **Normalise** — ImageNet mean/std `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`.

## 📊 Visual Outputs Produced

| Output | Description |
|---|---|
| **Sample Batch** | Grid of 16 augmented training images with class labels |
| **Dataset Distribution** | Bar chart of class counts in train & validation sets |
| **Training Curves** | Side-by-side loss & accuracy plots per epoch |
| **Performance Dashboard** | Accuracy / Precision / Recall / F1 gauges |
| **Confusion Matrix** | Seaborn heatmap of true vs predicted labels |
| **Classification Report** | Per-class precision, recall, F1 (sklearn) |
| **Prediction Gallery** | 10 validation images with predicted label & confidence % |
| **Contour Crop Demo** | 4-step pipeline: original → contour → extremes → cropped |

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install torch torchvision matplotlib seaborn scikit-learn tqdm opencv-python pillow
   ```

2. **Set the dataset path** — open the notebook and edit `DATASET_ROOT`:
   ```python
   DATASET_ROOT = "C:/path/to/your/dataset"
   ```
   The folder must contain:
   ```
   dataset/
   ├── train/
   │   ├── yes/   (tumor images)
   │   └── no/    (healthy images)
   └── valid/
       ├── yes/
       └── no/
   ```

3. **Run all cells** — the notebook trains the model, saves the best weights to
   `best_brain_tumor_model.pth`, and produces all visual outputs automatically.

## 📚 Recommended Datasets

| Dataset | Classes | Images | Link |
|---|---|---|---|
| Brain MRI Dataset (7 200 images) | Glioma, Meningioma, Pituitary, No tumor | 7 200 | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Crystal Clean Brain Tumors MRI | 4 classes, cleaned & 224×224 | ~7 000 | [Kaggle](https://www.kaggle.com/datasets/mohammedkhalilia/crystal-clean-brain-tumors-mri-dataset) |
| BRATS 2019 Train/Test/Valid | Binary (yes/no) | ~3 000 | [Kaggle](https://www.kaggle.com/datasets) |

> For multi-class classification, change `num_classes` in the config cell and
> switch to `class_mode='categorical'` equivalents.

## 📝 Variable Naming Convention

All variables use descriptive, PEP-8 compliant `snake_case`:

| Old Name (Keras) | New Name (PyTorch) |
|---|---|
| `es` | `early_stop_patience` |
| `mc` | `MODEL_SAVE_PATH` |
| `cnn` | `brain_tumor_model` |
| `train` | `train_loader` |
| `validation` | `val_loader` |
| `extLeft` | `extreme_left` |
| `c` | `largest_contour` |
| `result` | `probabilities` |

---

*Generated automatically — March 2026*
