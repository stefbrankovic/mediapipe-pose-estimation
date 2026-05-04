"""
Extract MediaPipe pose landmarks from yoga pose images → CSV.

Dataset layout expected under --data_dir:
    Yoga/
        tree/      (or Tree/)
        warrior/
        tpose/

Each image is processed by MediaPipe Pose (static mode).
Detected: 33 landmarks × 4 values (x, y, z, visibility) = 132 features.

Output: yoga_landmarks.csv  (132 feature columns + 'target' label column)

Usage:
    python src/preprocess.py --data_dir ./Yoga --out yoga_landmarks.csv
"""

import argparse
import os

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

CLASSES = ["tree", "warrior", "tpose"]   # adjust if your folder names differ


def get_label(root_folder: str) -> str:
    """Infer class label from parent folder name (case-insensitive)."""
    folder = os.path.basename(root_folder).lower()
    for cls in CLASSES:
        if cls in folder:
            return cls
    return "unknown"


def extract_row(img_path: str):
    """Return 132-element landmark list or None if pose not detected."""
    image = cv2.imread(img_path)
    if image is None:
        return None
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    row = []
    for lm in results.pose_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return row


def main(data_dir: str, out_csv: str):
    data, labels = [], []

    # Support both flat folder (all images + filename-based label)
    # and class-subfolder layout (preferred)
    subfolders = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    if subfolders:
        print(f"Found subfolders: {subfolders}")
        for folder in subfolders:
            label = get_label(folder)
            folder_path = os.path.join(data_dir, folder)
            images = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            print(f"  {folder} → label='{label}' | {len(images)} images")
            for img_name in tqdm(images, desc=folder):
                row = extract_row(os.path.join(folder_path, img_name))
                if row:
                    data.append(row)
                    labels.append(label)
    else:
        # Flat layout: derive label from filename
        print("No subfolders found — using filename-based labelling.")
        images = [
            f for f in os.listdir(data_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        for img_name in tqdm(images):
            label = "unknown"
            name = img_name.lower()
            for cls in CLASSES:
                if cls in name:
                    label = cls
                    break
            row = extract_row(os.path.join(data_dir, img_name))
            if row:
                data.append(row)
                labels.append(label)

    df = pd.DataFrame(data)
    df["target"] = labels
    df = df[df["target"] != "unknown"]
    df.to_csv(out_csv, index=False)

    print(f"\nDone. {len(df)} samples saved to '{out_csv}'")
    print(df["target"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./Yoga",
                        help="Root folder of the yoga image dataset")
    parser.add_argument("--out", default="yoga_landmarks.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    main(args.data_dir, args.out)
