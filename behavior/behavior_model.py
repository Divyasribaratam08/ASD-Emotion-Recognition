
import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder


DATA_FILE = "behavior/berd_features_v3.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "behavior_model_v3.pkl")
RESULT_FILE = "results/behavior_evaluation_v3.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading BERD v3 features...")

df = pd.read_csv(DATA_FILE)

# Metadata must never be used as model input.
metadata_columns = [
    "participant",
    "emotion_id",
    "emotion",
    "trial",
    "frame_index",
]

required_columns = metadata_columns.copy()
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df.empty:
    raise ValueError("The feature dataset is empty.")

if df["emotion"].isna().any():
    raise ValueError("Missing emotion labels were found.")

if df["participant"].isna().any():
    raise ValueError("Missing participant IDs were found.")

feature_columns = [
    column for column in df.columns
    if column not in metadata_columns
]

if not feature_columns:
    raise ValueError("No feature columns were found.")

X = df[feature_columns].to_numpy(dtype=np.float32)
y = df["emotion"].astype(str).to_numpy()
groups = df["participant"].astype(str).to_numpy()

if not np.isfinite(X).all():
    raise ValueError(
        "Features contain NaN or infinite values."
    )

print("Total frames:", len(df))
print("Unique participants:", len(np.unique(groups)))
print("Number of features:", len(feature_columns))
print("Emotion distribution:")
print(df["emotion"].value_counts().sort_index())

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split by participant to prevent the same person's frames
# from appearing in both training and testing.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_indices, test_indices = next(
    splitter.split(X, y_encoded, groups=groups)
)

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y_encoded[train_indices]
y_test = y_encoded[test_indices]

train_participants = sorted(
    np.unique(groups[train_indices]).tolist()
)
test_participants = sorted(
    np.unique(groups[test_indices]).tolist()
)

# Explicitly verify participant separation.
overlap = set(train_participants) & set(test_participants)

if overlap:
    raise RuntimeError(
        f"Participant leakage detected: {sorted(overlap)}"
    )

print("\nTraining participants:", train_participants)
print("Testing participants:", test_participants)
print("Training frames:", len(X_train))
print("Testing frames:", len(X_test))

print("\nTraining Random Forest classifier...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(
    y_test, predictions
)

class_ids = np.arange(len(label_encoder.classes_))

report = classification_report(
    y_test,
    predictions,
    labels=class_ids,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0,
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=class_ids,
)

# Save the model and the exact feature-column order.
joblib.dump(
    {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "metadata_columns": metadata_columns,
    },
    MODEL_FILE,
)

evaluation = {
    "model": "RandomForestClassifier",
    "dataset": "BERD word-stimulus skeleton features v3",
    "split_method": "GroupShuffleSplit by participant",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "train_participants": train_participants,
    "test_participants": test_participants,
    "participant_overlap": sorted(overlap),
    "train_frames": int(len(X_train)),
    "test_frames": int(len(X_test)),
    "feature_count": int(len(feature_columns)),
    "class_names": label_encoder.classes_.tolist(),
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "note": (
        "Preliminary frame-level evaluation. Frames within a "
        "recording are correlated; results are not clinical "
        "ASD detection accuracy."
    ),
}

with open(RESULT_FILE, "w", encoding="utf-8") as file:
    json.dump(evaluation, file, indent=2)

print(f"\nParticipant-independent test accuracy: {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=class_ids,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)

print("\nConfusion matrix:")
print(matrix)

print("\nSaved model:", MODEL_FILE)
print("Saved evaluation:", RESULT_FILE)