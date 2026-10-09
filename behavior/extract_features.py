import json
import math
import numpy as np

INPUT_FILE = "behavior/pose_landmarks.json"
OUTPUT_FILE = "behavior/behavior_features.json"


def point(landmarks, index):
    landmark = landmarks[index]
    return np.array([
        landmark["x"],
        landmark["y"],
        landmark["z"]
    ], dtype=float)


def distance(a, b):
    return float(np.linalg.norm(a - b))


def angle(a, b, c):
    """
    Calculates angle ABC in degrees.
    """
    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return 0.0

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def extract_frame_features(landmarks):
    # Important MediaPipe landmark indices
    nose = point(landmarks, 0)

    left_shoulder = point(landmarks, 11)
    right_shoulder = point(landmarks, 12)

    left_elbow = point(landmarks, 13)
    right_elbow = point(landmarks, 14)

    left_wrist = point(landmarks, 15)
    right_wrist = point(landmarks, 16)

    left_hip = point(landmarks, 23)
    right_hip = point(landmarks, 24)

    left_knee = point(landmarks, 25)
    right_knee = point(landmarks, 26)

    left_ankle = point(landmarks, 27)
    right_ankle = point(landmarks, 28)

    # Centers
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2

    # Posture features
    shoulder_width = distance(left_shoulder, right_shoulder)
    hip_width = distance(left_hip, right_hip)

    torso_length = distance(
        shoulder_center,
        hip_center
    )

    # Joint angles
    left_elbow_angle = angle(
        left_shoulder,
        left_elbow,
        left_wrist
    )

    right_elbow_angle = angle(
        right_shoulder,
        right_elbow,
        right_wrist
    )

    left_knee_angle = angle(
        left_hip,
        left_knee,
        left_ankle
    )

    right_knee_angle = angle(
        right_hip,
        right_knee,
        right_ankle
    )

    # Shoulder and hip orientation
    shoulder_angle = math.degrees(
        math.atan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0]
        )
    )

    hip_angle = math.degrees(
        math.atan2(
            right_hip[1] - left_hip[1],
            right_hip[0] - left_hip[0]
        )
    )

    # Head position relative to torso
    head_to_torso = distance(
        nose,
        shoulder_center
    )

    # Wrist positions relative to body
    left_wrist_to_hip = distance(
        left_wrist,
        left_hip
    )

    right_wrist_to_hip = distance(
        right_wrist,
        right_hip
    )

    return {
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "torso_length": torso_length,
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "shoulder_angle": shoulder_angle,
        "hip_angle": hip_angle,
        "head_to_torso": head_to_torso,
        "left_wrist_to_hip": left_wrist_to_hip,
        "right_wrist_to_hip": right_wrist_to_hip
    }


# Load landmark data
with open(INPUT_FILE, "r") as file:
    frames = json.load(file)

all_features = []

for frame_data in frames:

    features = extract_frame_features(
        frame_data["landmarks"]
    )

    all_features.append({
        "frame": frame_data["frame"],
        "timestamp_ms": frame_data["timestamp_ms"],
        "features": features
    })


# Save extracted behavior features
with open(OUTPUT_FILE, "w") as file:
    json.dump(all_features, file, indent=2)


print("Behavior feature extraction completed.")
print("Frames processed:", len(all_features))
print("Features per frame:", len(all_features[0]["features"]))
print("Saved to:", OUTPUT_FILE)