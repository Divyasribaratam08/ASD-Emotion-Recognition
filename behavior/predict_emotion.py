import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "behavior_model_v3.pkl"
OUTPUT_FILE = ROOT / "behavior" / "behavior_prediction.json"

JOINTS = 18
COORDINATES = 3


def predict_emotion(csv_path):
    # Load the trained model and its preprocessing information.
    saved = joblib.load(MODEL_FILE)

    model = saved["model"]
    label_encoder = saved["label_encoder"]
    feature_columns = saved["feature_columns"]

    # Read the BERD skeleton CSV.
    df = pd.read_csv(csv_path)

    if df.shape[1] != 55:
        raise ValueError(
            f"Expected 55 columns, but found {df.shape[1]}"
        )

    # Extract 54 joint coordinates, excluding frame index.
    coordinates = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    if coordinates.shape[1] != JOINTS * COORDINATES:
        raise ValueError("Unexpected coordinate count.")

    if not np.isfinite(coordinates).all():
        raise ValueError("Input contains invalid coordinates.")

    # Reshape into frames, joints and XYZ coordinates.
    frames = coordinates.reshape(-1, JOINTS, COORDINATES)

    # Apply the same initial-joint-0 reference transformation
    # used by the V3 training preprocessing.
    initial_reference = frames[0, 0, :]
    relative_frames = frames - initial_reference

    # Preserve joint-by-joint XYZ feature ordering.
    X = relative_frames.reshape(len(relative_frames), -1)

    if X.shape[1] != len(feature_columns):
        raise ValueError("Feature count does not match the model.")

    # Predict probabilities for every frame.
    frame_probabilities = model.predict_proba(X)

    # Average probabilities to obtain a recording-level result.
    average_probabilities = frame_probabilities.mean(axis=0)

    emotion_names = label_encoder.inverse_transform(model.classes_)

    emotion_scores = {
        emotion: float(score)
        for emotion, score in zip(
            emotion_names, average_probabilities
        )
    }

    predicted_emotion = max(
        emotion_scores, key=emotion_scores.get
    )

    result = {
        "source": str(csv_path),
        "model": "RandomForestClassifier",
        "input_frames": int(len(frames)),
        "joints": JOINTS,
        "features_per_frame": len(feature_columns),
        "predicted_emotion": predicted_emotion,
        "emotions": emotion_scores,
        "note": (
            "Recording-level result obtained by averaging "
            "frame probabilities. This is an experimental "
            "emotion prediction, not an ASD diagnosis."
        )
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("\nBEHAVIOUR EMOTION PREDICTION")
    print("----------------------------")
    print("Input frames:", len(frames))
    print("Predicted emotion:", predicted_emotion)

    print("\nEmotion probabilities:")
    for emotion, score in emotion_scores.items():
        print(f"{emotion:10s}: {score:.4f}")

    print("\nSaved result:", OUTPUT_FILE)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python behavior/predict_emotion.py "
            "<BERD_CSV_PATH>"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    predict_emotion(input_path)