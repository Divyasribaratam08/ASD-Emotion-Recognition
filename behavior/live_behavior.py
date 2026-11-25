
import cv2
import mediapipe as mp
import numpy as np
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "behavior" / "pose_landmarker.task"
OUTPUT_FILE = ROOT / "behavior" / "live_behavior_features.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Pose model not found: {MODEL_PATH}"
    )

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE
)

# Body landmark connections for drawing
CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 31), (28, 32),
    (15, 17), (16, 18)
]


def get_point(landmarks, index):
    landmark = landmarks[index]
    return np.array([
        landmark.x,
        landmark.y,
        landmark.z
    ], dtype=float)


def calculate_distance(a, b):
    return float(np.linalg.norm(a - b))


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    if denominator == 0:
        return 0.0

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(
        np.degrees(np.arccos(cosine))
    )


def extract_live_features(landmarks, previous_points):
    nose = get_point(landmarks, 0)

    left_shoulder = get_point(landmarks, 11)
    right_shoulder = get_point(landmarks, 12)

    left_elbow = get_point(landmarks, 13)
    right_elbow = get_point(landmarks, 14)

    left_wrist = get_point(landmarks, 15)
    right_wrist = get_point(landmarks, 16)

    left_hip = get_point(landmarks, 23)
    right_hip = get_point(landmarks, 24)

    left_knee = get_point(landmarks, 25)
    right_knee = get_point(landmarks, 26)

    left_ankle = get_point(landmarks, 27)
    right_ankle = get_point(landmarks, 28)

    shoulder_center = (
        left_shoulder + right_shoulder
    ) / 2

    hip_center = (
        left_hip + right_hip
    ) / 2

    features = {
        "shoulder_width": calculate_distance(
            left_shoulder, right_shoulder
        ),
        "hip_width": calculate_distance(
            left_hip, right_hip
        ),
        "torso_length": calculate_distance(
            shoulder_center, hip_center
        ),
        "left_elbow_angle": calculate_angle(
            left_shoulder, left_elbow, left_wrist
        ),
        "right_elbow_angle": calculate_angle(
            right_shoulder, right_elbow, right_wrist
        ),
        "left_knee_angle": calculate_angle(
            left_hip, left_knee, left_ankle
        ),
        "right_knee_angle": calculate_angle(
            right_hip, right_knee, right_ankle
        ),
        "shoulder_angle": float(np.degrees(
            np.arctan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0]
            )
        )),
        "hip_angle": float(np.degrees(
            np.arctan2(
                right_hip[1] - left_hip[1],
                right_hip[0] - left_hip[0]
            )
        )),
        "head_to_torso": calculate_distance(
            nose, shoulder_center
        ),
        "left_wrist_to_hip": calculate_distance(
            left_wrist, left_hip
        ),
        "right_wrist_to_hip": calculate_distance(
            right_wrist, right_hip
        )
    }

    # Calculate movement relative to the previous frame.
    selected_points = {
        "nose": nose,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "left_knee": left_knee,
        "right_knee": right_knee
    }

    if previous_points is None:
        movement = 0.0
    else:
        movements = [
            calculate_distance(
                selected_points[name],
                previous_points[name]
            )
            for name in selected_points
        ]

        movement = float(np.mean(movements))

    features["movement_index"] = movement

    return features, selected_points


def draw_pose(frame, landmarks):
    height, width = frame.shape[:2]

    # Draw skeleton connections
    for start, end in CONNECTIONS:
        p1 = landmarks[start]
        p2 = landmarks[end]

        x1, y1 = int(p1.x * width), int(p1.y * height)
        x2, y2 = int(p2.x * width), int(p2.y * height)

        cv2.line(
            frame, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

    # Draw landmark points
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame, (x, y), 3,
            (0, 0, 255), -1
        )


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Camera could not be opened. "
            "Check camera permissions."
        )

    previous_points = None
    all_records = []

    print("Loading MediaPipe Pose Landmarker...")
    print("Live behaviour feature extraction started.")
    print("Press Q to stop.")

    try:
        with PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    print("Unable to read camera frame.")
                    break

                frame_rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb
                )

                result = landmarker.detect(mp_image)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]

                    draw_pose(frame, landmarks)

                    features, current_points = (
                        extract_live_features(
                            landmarks,
                            previous_points
                        )
                    )

                    previous_points = current_points

                    record = {
                        "timestamp": time.time(),
                        "features": features
                    }

                    all_records.append(record)

                    cv2.putText(
                        frame,
                        "BODY POSE DETECTED",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    display_items = [
                        ("Movement", features["movement_index"]),
                        ("Shoulder angle", features["shoulder_angle"]),
                        ("Left elbow", features["left_elbow_angle"]),
                        ("Right elbow", features["right_elbow_angle"]),
                        ("Left knee", features["left_knee_angle"]),
                        ("Right knee", features["right_knee_angle"])
                    ]

                    for i, (label, value) in enumerate(
                        display_items
                    ):
                        text = f"{label}: {value:.2f}"

                        cv2.putText(
                            frame,
                            text,
                            (20, 75 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

                else:
                    previous_points = None

                    cv2.putText(
                        frame,
                        "NO PERSON DETECTED",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow(
                    "Live Body Behaviour Feature Extraction",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

        OUTPUT_FILE.parent.mkdir(
            parents=True, exist_ok=True
        )

        with open(
            OUTPUT_FILE, "w", encoding="utf-8"
        ) as file:
            json.dump(all_records, file, indent=2)

        print("\nLive behaviour extraction stopped.")
        print("Frames with pose:", len(all_records))
        print("Saved features:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
