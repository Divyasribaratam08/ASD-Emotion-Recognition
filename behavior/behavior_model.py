import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

DATA_FILE = "behavior/berd_features.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "behavior_model.pkl")
RESULT_FILE = "results/behavior_evaluation.json"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading BERD features...")
df = pd.read_csv(DATA_FILE)

metadata_columns = [
    "participant",
    "emotion_id",
    "emotion",
    "trial"
]

feature_columns = [
    column for column in df.columns
    if column not in metadata_columns
]

X = df[feature_columns].to_numpy(dtype=np.float32)
y = df["emotion"].to_numpy()
groups = df["participant"].to_numpy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
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

print("Training participants:", train_participants)
print("Testing participants:", test_participants)
print("Training frames:", len(X_train))
print("Testing frames:", len(X_test))

print("\nTraining Random Forest classifier...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

report = classification_report(
    y_test,
    predictions,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(label_encoder.classes_))
)

joblib.dump(
    {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns
    },
    MODEL_FILE
)

evaluation = {
    "model": "RandomForestClassifier",
    "dataset": "BERD word-stimulus skeleton features",
    "split_method": "GroupShuffleSplit by participant",
    "random_state": 42,
    "train_participants": train_participants,
    "test_participants": test_participants,
    "train_frames": len(X_train),
    "test_frames": len(X_test),
    "accuracy": float(accuracy),
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "class_names": label_encoder.classes_.tolist()
}

with open(RESULT_FILE, "w") as file:
    json.dump(evaluation, file, indent=2)

print(f"\nParticipant-independent test accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion matrix:")
print(matrix)

print("\nSaved model:", MODEL_FILE)
print("Saved evaluation:", RESULT_FILE)