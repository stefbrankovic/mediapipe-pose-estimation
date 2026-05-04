"""
Real-time pose analysis with MediaPipe.

Features
--------
- Bicep curl rep counter (left arm) — elbow angle tracking
- Clap detector — wrist proximity

Controls: ESC to quit
"""

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calc_angle(a, b, c):
    """Angle at joint b given three MediaPipe landmark objects."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def main():
    counter = 0
    stage = None          # "down" / "up"
    clap_active = False
    CLAP_DIST = 0.18      # normalised wrist distance threshold

    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
                lm = results.pose_landmarks.landmark

                # ── Bicep curl (left arm: shoulder=11, elbow=13, wrist=15) ──
                angle = calc_angle(lm[11], lm[13], lm[15])

                # Draw angle near elbow
                ex = int(lm[13].x * image.shape[1])
                ey = int(lm[13].y * image.shape[0])
                cv2.putText(image, str(int(angle)), (ex - 30, ey - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if angle > 100:
                    stage = "down"
                if angle < 40 and stage == "down":
                    stage = "up"
                    counter += 1
                    print(f"Rep: {counter}")

                # ── Clap detector ─────────────────────────────────────────
                lw, rw = lm[15], lm[16]
                dist = np.hypot(lw.x - rw.x, lw.y - rw.y)
                if dist < CLAP_DIST and not clap_active:
                    clap_active = True
                elif dist >= CLAP_DIST:
                    clap_active = False

            # ── HUD ──────────────────────────────────────────────────────
            cv2.rectangle(image, (0, 0), (225, 90), (245, 117, 16), -1)
            cv2.putText(image, "REPS", (15, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(image, str(counter), (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 2)

            if clap_active:
                h, w = image.shape[:2]
                cv2.putText(image, "CLAP!", (w // 2 - 70, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 200, 255), 4)

            cv2.imshow("Pose Demo — ESC to quit", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
