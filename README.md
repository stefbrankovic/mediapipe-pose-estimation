# Real-Time Pose Estimation & Yoga Pose Classification

Human pose analysis using **Google MediaPipe**, **OpenCV**, and **PyTorch**.  
Built as hands-on exploration for my thesis on Graph Convolutional Networks for Human Action Recognition.

---

## What it does

### 1. Real-time webcam demo
Live pose tracking with two features:
- **Bicep curl rep counter** — tracks elbow angle (shoulder → elbow → wrist) and counts full reps
- **Clap detector** — triggers when wrist distance drops below a threshold

### 2. Yoga pose classifier (3 classes: Tree, Warrior, T-Pose)

A three-step pipeline:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `src/preprocess.py` | Runs MediaPipe on each image, extracts 132 landmark features (x, y, z, visibility × 33 joints) → CSV |
| 2 | `src/train.py` | Trains a PyTorch MLP on the landmark coordinates |
| 3 | `src/realtime_classify.py` | Real-time classification from webcam using the trained model |

---

## Why landmarks instead of raw images?

Using skeleton coordinates rather than pixels:
- Reduces input from ~300k pixel values to **132 floats**
- Makes the classifier viewpoint- and scale-invariant (to a degree)
- Directly mirrors the input format of graph-based models (ST-GCN, AGCN) which treat landmarks as graph nodes — this was the key motivation

---

## Model

```
Input (132) → Linear(256) → BN → ReLU → Dropout(0.3)
            → Linear(128) → BN → ReLU → Dropout(0.3)
            → Linear(64)  → ReLU
            → Linear(3)   → CrossEntropyLoss
```

Trained with Adam + StepLR scheduler, 80/20 stratified split.

---

## Setup

```bash
# 1. Create and activate Anaconda environment
conda create -n pose-env python=3.9
conda activate pose-env

# 2. Install PyTorch (CPU) — for GPU visit https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies
pip install -r requirements.txt
```

---

## Usage

### Webcam demo (no dataset needed)
```bash
python src/webcam_demo.py
```

### Train the yoga classifier

1. Download the [Kaggle yoga dataset](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset) and place images under `Yoga/` with subfolders per class:
```
Yoga/
    tree/
    warrior/
    tpose/
```

2. Extract landmarks:
```bash
python src/preprocess.py --data_dir ./Yoga --out yoga_landmarks.csv
```

3. Train:
```bash
python src/train.py --csv yoga_landmarks.csv --epochs 60
# model saved to models/yoga_classifier.pth
```

4. Real-time inference:
```bash
python src/realtime_classify.py
```

---

## Project structure

```
pose-estimation/
├── src/
│   ├── webcam_demo.py         # Real-time bicep counter + clap detector
│   ├── preprocess.py          # Landmark extraction from dataset images
│   ├── train.py               # PyTorch MLP training
│   └── realtime_classify.py   # Real-time inference with trained model
├── requirements.txt
└── README.md
```

---

## Connection to thesis

This project served as a practical entry point into skeleton-based action understanding.  
The thesis extends this to **spatio-temporal graph modelling** (ST-GCN, AGCN) for full action sequence recognition, with prospective applications in collaborative robotics.

---

## Tech stack

`Python 3.9` · `MediaPipe` · `OpenCV` · `PyTorch` · `scikit-learn` · `pandas`
