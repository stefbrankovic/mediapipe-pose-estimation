"""
Train a PyTorch MLP on extracted MediaPipe landmark coordinates.

Input:  yoga_landmarks.csv  (output of preprocess.py)
Output: models/yoga_classifier.pth  +  models/meta.json (class names)

Usage:
    python src/train.py
    python src/train.py --csv yoga_landmarks.csv --epochs 80 --lr 0.001
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ── Model ─────────────────────────────────────────────────────────────────────

class PoseClassifier(nn.Module):
    """
    Lightweight MLP for pose classification from landmark coordinates.

    Input: 132 floats (33 landmarks × [x, y, z, visibility])
    Output: logits over N pose classes
    """
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    # Load CSV
    df = pd.read_csv(args.csv)
    y_raw = df.pop("target").values
    X = df.values.astype(np.float32)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)
    print(f"Classes: {classes}  |  Samples: {len(X)}")

    # Train / val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    to_tensor = lambda a, dtype: torch.tensor(a, dtype=dtype)
    train_ds = torch.utils.data.TensorDataset(
        to_tensor(X_tr, torch.float32), to_tensor(y_tr, torch.long)
    )
    val_ds = torch.utils.data.TensorDataset(
        to_tensor(X_val, torch.float32), to_tensor(y_val, torch.long)
    )
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=64)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model   = PoseClassifier(X.shape[1], len(classes)).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()

        # ── validate ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total   += len(yb)
        acc = correct / total

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{args.epochs}  val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            torch.save(model.state_dict(), args.save)

    print(f"\nBest val accuracy: {best_acc:.3f}  →  {args.save}")

    # Save class metadata next to the model
    meta_path = args.save.replace(".pth", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"classes": classes, "input_dim": int(X.shape[1])}, f)
    print(f"Metadata saved to {meta_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    default="yoga_landmarks.csv")
    p.add_argument("--epochs", type=int,   default=60)
    p.add_argument("--lr",     type=float, default=0.001)
    p.add_argument("--save",   default="models/yoga_classifier.pth")
    train(p.parse_args())
