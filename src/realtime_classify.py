"""
Real-time yoga pose classification using the trained PoseClassifier.

Loads the model trained by train.py and runs inference on a live webcam feed.

Usage:
    python src/realtime_classify.py
    python src/realtime_classify.py --model models/yoga_classifier.pth \
                                    --meta  models/yoga_classifier_meta.json

Controls: ESC to quit
"""

import argparse
import json

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn


class PoseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main(args):
    with open(args.meta) as f:
        meta = json.load(f)
    classes   = meta["classes"]
    input_dim = meta["input_dim"]

    model = PoseClassifier(input_dim, len(classes))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    mp_pose   = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.6,
                      min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            frame   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            label, conf = "–", 0.0
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                          mp_pose.POSE_CONNECTIONS)
                row = []
                for lm in results.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])

                x_t = torch.tensor([row], dtype=torch.float32)
                with torch.no_grad():
                    probs = torch.softmax(model(x_t), dim=1)[0]
                conf  = probs.max().item()
                label = classes[probs.argmax().item()]

            color = (0, 220, 100) if conf > 0.75 else (0, 165, 255)
            cv2.rectangle(frame, (0, 0), (350, 60), (20, 20, 20), -1)
            cv2.putText(frame, f"{label.upper()}  {conf*100:.0f}%",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

            cv2.imshow("Yoga Pose Classifier — ESC to quit", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/yoga_classifier.pth")
    p.add_argument("--meta",  default="models/yoga_classifier_meta.json")
    main(p.parse_args())
