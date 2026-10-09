import json
import math
import numpy as np

INPUT_FILE = "behavior/pose_landmarks.json"
OUTPUT_FILE = "behavior/temporal_features.json"


def point(landmarks, index):
    landmark = landmarks[index]

    return np.array([
        landmark["x"],
        landmark["y"],
        landmark["z"]
    ], dtype=float)


def distance(a, b):
    return float(np.linalg.norm(a - b))


with open(INPUT_FILE, "r") as file:
    frames = json.load(file)


temporal_features = []

previous_points = None

for frame_data in frames:

    landmarks = frame_data["landmarks"]

    # Important body points
    nose = point(landmarks, 0)

    left_shoulder = point(landmarks, 11)
    right_shoulder = point(landmarks, 12)

    left_elbow = point(landmarks, 13)
    right_elbow = point(landmarks, 14)

    left_wrist = point(landmarks, 15)
    right_wrist = point(landmarks, 16)

    left_knee = point(landmarks, 25)
    right_knee = point(landmarks, 26)

    current_points = {
        "nose": nose,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "left_knee": left_knee,
        "right_knee": right_knee
    }

    if previous_points is None:

        movement = {
            "head_movement": 0.0,
            "left_shoulder_movement": 0.0,
            "right_shoulder_movement": 0.0,
            "left_elbow_movement": 0.0,
            "right_elbow_movement": 0.0,
            "left_wrist_movement": 0.0,
            "right_wrist_movement": 0.0,
            "left_knee_movement": 0.0,
            "right_knee_movement": 0.0
        }

    else:

        movement = {
            "head_movement": distance(
                current_points["nose"],
                previous_points["nose"]
            ),

            "left_shoulder_movement": distance(
                current_points["left_shoulder"],
                previous_points["left_shoulder"]
            ),

            "right_shoulder_movement": distance(
                current_points["right_shoulder"],
                previous_points["right_shoulder"]
            ),

            "left_elbow_movement": distance(
                current_points["left_elbow"],
                previous_points["left_elbow"]
            ),

            "right_elbow_movement": distance(
                current_points["right_elbow"],
                previous_points["right_elbow"]
            ),

            "left_wrist_movement": distance(
                current_points["left_wrist"],
                previous_points["left_wrist"]
            ),

            "right_wrist_movement": distance(
                current_points["right_wrist"],
                previous_points["right_wrist"]
            ),

            "left_knee_movement": distance(
                current_points["left_knee"],
                previous_points["left_knee"]
            ),

            "right_knee_movement": distance(
                current_points["right_knee"],
                previous_points["right_knee"]
            )
        }

    temporal_features.append({
        "frame": frame_data["frame"],
        "timestamp_ms": frame_data["timestamp_ms"],
        "movement": movement
    })

    previous_points = current_points


with open(OUTPUT_FILE, "w") as file:
    json.dump(temporal_features, file, indent=2)


print("Temporal feature extraction completed.")
print("Frames processed:", len(temporal_features))
print(
    "Movement features per frame:",
    len(temporal_features[0]["movement"])
)
print("Saved to:", OUTPUT_FILE)