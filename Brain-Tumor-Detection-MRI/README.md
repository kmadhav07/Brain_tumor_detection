---
title: Brain Tumor Detection MRI
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

<div align="center">

# 🧠 Brain Tumor Detection — MRI Classification

### Deep Learning–Powered Binary MRI Classifier Using Fine-Tuned ResNet-50

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces/DumbMaddy/Brain-Tumor-Detection-MRI)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)

**A production-ready web application that detects brain tumors in MRI scans using a fine-tuned ResNet-50 model built with PyTorch. Achieves ~99% validation accuracy. Deployed live on Hugging Face Spaces.**

[🚀 Try the Live Demo](https://huggingface.co/spaces/DumbMaddy/Brain-Tumor-Detection-MRI) · [📖 Project Details](PROJECT_DETAILS.md) · [📓 Training Notebook](training/medical-image.ipynb)

</div>

---

## 🌐 Live Demo

> **Try it instantly — no setup required!**
>
> 👉 **[https://huggingface.co/spaces/DumbMaddy/Brain-Tumor-Detection-MRI](https://huggingface.co/spaces/DumbMaddy/Brain-Tumor-Detection-MRI)**

Hosted on **Hugging Face Spaces** via Docker. Includes 6 pre-loaded sample MRI scans from the BRATS 2019 dataset for instant testing.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **ResNet-50 Backbone** | Transfer learning from ImageNet with custom classification head |
| 🧹 **Smart Preprocessing** | OpenCV brain contour cropping removes irrelevant black borders |
| 📊 **Confidence Scores** | Softmax probability bars for Tumor & No Tumor classes |
| 🖼️ **Dual Input Modes** | Upload your own MRI **or** test with 6 pre-loaded samples |
| 🏷️ **Ground Truth Labels** | Sample images show actual labels alongside predictions |
| 📱 **Responsive UI** | Clean, modern design with Inter font — works on all devices |
| 🔄 **Drag & Drop** | Intuitive file upload with drag-and-drop |
| 📚 **Educational Pipeline** | Interactive "Know More" section explains architecture |

---

## 🏗️ System Architecture

```
Browser (Frontend)
  │
  │  Upload / Select Sample MRI
  │  HTTP JSON API
  ▼
Flask Server (Backend)
  │
  ├── 1. Load Image (PIL → RGB)
  ├── 2. Brain Contour Crop (OpenCV)
  │     ├─ Gaussian Blur (5×5)
  │     ├─ Binary Threshold (45)
  │     ├─ Morphological Ops (erode/dilate)
  │     └─ Bounding Box of Largest Contour
  ├── 3. Resize to 224×224
  ├── 4. Normalize (ImageNet mean/std)
  │
  ▼
PyTorch Model
  ├── ResNet-50 (frozen backbone) → 2048-dim features
  └── Custom Head
      ├── Dropout (0.4)
      ├── Linear (2048 → 728) + ReLU
      ├── Dropout (0.4)
      └── Linear (728 → 2) → Softmax → [P(No Tumor), P(Tumor)]
```

---

## 🧬 Model Details

### Architecture

```
ResNet-50 (frozen convolutional backbone)
└── Custom Classification Head
    ├── Dropout (p=0.4)
    ├── Linear (2048 → 728)
    ├── ReLU
    ├── Dropout (p=0.4)
    └── Linear (728 → 2)
```

| Metric | Value |
|--------|-------|
| Total Parameters | ~24.6 M |
| Trainable Parameters | ~1.5 M (head only) |
| Input Size | 224 × 224 × 3 (RGB) |
| Output Classes | 2 (`no` = Healthy, `yes` = Tumor) |

### Training Configuration

| Component | Choice |
|-----------|--------|
| Framework | PyTorch ≥ 2.0 |
| Backbone | `torchvision.models.resnet50` (ImageNet V2 weights) |
| Data Pipeline | `torchvision.datasets.ImageFolder` + `DataLoader` |
| Augmentation | Random flips, ±20° rotation, colour jitter, brain contour crop |
| Optimizer | Adam (`lr=1e-4`, `weight_decay=1e-4`) |
| Scheduler | `ReduceLROnPlateau` (factor 0.5, patience 2) |
| Loss | `CrossEntropyLoss` |
| Early Stopping | Patience = 5 epochs |

### Performance (BRATS 2019 Dataset)

| Metric | Score |
|--------|-------|
| Validation Accuracy | ~99% |
| F1-Score | ~99% |
| Precision | ~99% |
| Recall | ~99% |

---

## 🖼️ Preprocessing Pipeline

1. **Brain Contour Crop** — OpenCV finds the largest contour, crops to bounding box, removes skull & black borders
2. **Resize** — 224 × 224 pixels
3. **Augmentation** (training only) — horizontal/vertical flips, ±20° rotation, brightness/contrast jitter
4. **Normalize** — ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`

---

## 📁 Project Structure

```
Brain-Tumor-Detection-MRI/
├── app.py                          # Flask backend
├── best_brain_tumor_model.pth      # Trained model weights (~96 MB)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker config for HF Spaces
├── README.md                       # This file
├── PROJECT_DETAILS.md              # Extended documentation
├── templates/
│   └── index.html                  # Frontend UI (~38 KB)
├── default_images/                 # 6 sample MRI scans
│   ├── no_1400.jpg                 #   Healthy brain
│   ├── no_1450.jpg                 #   Healthy brain
│   ├── no_1499.jpg                 #   Healthy brain
│   ├── yes_1400.jpg                #   Brain with tumor
│   ├── yes_1450.jpg                #   Brain with tumor
│   └── yes_1499.jpg                #   Brain with tumor
├── training/
│   └── medical-image.ipynb         # Training notebook (~3 MB)
└── uploads/                        # Temporary uploaded images
```

---

## 🚀 Quick Start

### Option 1: Live Demo (No Setup)

👉 **[https://huggingface.co/spaces/DumbMaddy/Brain-Tumor-Detection-MRI](https://huggingface.co/spaces/DumbMaddy/Brain-Tumor-Detection-MRI)**

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://huggingface.co/spaces/DumbMaddy/Brain-Tumor-Detection-MRI
cd Brain-Tumor-Detection-MRI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

Open **http://localhost:5000** in your browser.

### Option 3: Docker

```bash
docker build -t brain-tumor-detection .
docker run -p 7860:7860 brain-tumor-detection
```

Open **http://localhost:7860**.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main web UI |
| `GET` | `/list-defaults` | List available sample images |
| `POST` | `/load-defaults` | Analyze a default sample. Body: `{"selected_file": "filename.jpg"}` |
| `POST` | `/upload` | Analyze uploaded MRI. Multipart form: `image` field |
| `GET` | `/serve-image/<source>/<filename>` | Serve image (`default` or `uploaded`) |

### Example Response

```json
{
  "success": true,
  "prediction": "yes",
  "confidence": 98.73,
  "tumor_prob": 98.73,
  "no_tumor_prob": 1.27
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch 2.1, TorchVision 0.16 |
| Web Framework | Flask 3.0 |
| Image Processing | OpenCV (headless), Pillow |
| Frontend | HTML5, Vanilla CSS, JavaScript |
| Typography | Google Fonts — Inter |
| Containerization | Docker (Python 3.11-slim) |
| Deployment | Hugging Face Spaces |

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` (local) / `7860` (Docker) | Server port |
| `MAX_CONTENT_LENGTH` | 16 MB | Max upload size |
| `IMAGE_SIZE` | 224 | Input image dimension |
| `ALLOWED_EXTENSIONS` | `png, jpg, jpeg` | Accepted formats |

---

## 📊 Training Notebook Outputs

The notebook `training/medical-image.ipynb` produces:

| Output | Description |
|--------|-------------|
| Sample Batch | 16 augmented training images with labels |
| Dataset Distribution | Class count bar chart |
| Training Curves | Loss & accuracy plots per epoch |
| Performance Dashboard | Accuracy/Precision/Recall/F1 gauges |
| Confusion Matrix | Seaborn heatmap |
| Classification Report | Per-class metrics (sklearn) |
| Prediction Gallery | 10 images with predicted labels & confidence |
| Contour Crop Demo | 4-step pipeline visualization |

---

## 📚 Recommended Datasets

| Dataset | Classes | Images | Link |
|---------|---------|--------|------|
| Brain MRI Dataset | Glioma, Meningioma, Pituitary, No Tumor | 7,200 | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Crystal Clean Brain Tumors | 4 classes, 224×224 | ~7,000 | [Kaggle](https://www.kaggle.com/datasets/mohammedkhalilia/crystal-clean-brain-tumors-mri-dataset) |
| BRATS 2019 | Binary (yes/no) | ~3,000 | [Kaggle](https://www.kaggle.com/datasets) |

---

## ⚠️ Disclaimer

> This is a **research/educational tool** — **NOT** for clinical diagnosis. Always consult qualified medical professionals. The model may not generalize to all MRI protocols or scanners.

---

## 📬 Contact

- **Email:** kmadhav0726@gmail.com
- **Phone:** 9693600978
- **Hugging Face:** [DumbMaddy](https://huggingface.co/DumbMaddy)

---

<div align="center">

**⭐ Star this project if you found it helpful!**

Made with ❤️ using PyTorch & Flask

</div>
