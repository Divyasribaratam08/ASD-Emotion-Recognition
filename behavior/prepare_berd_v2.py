
import os
import csv
import json
import numpy as np

DATASET_DIR = "datasets/stimulus/word"
MAPPING_FILE = "behavior/emotion_mapping.json"
OUTPUT_FILE = "behavior/berd_features_v2.csv"

EMOTION_ORDER = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

EXPECTED_COLUMNS = 55
EXPECTED_JOINTS = 18

# Load the emotion ID-to-name mapping
with open(MAPPING_FILE, "r", encoding="utf-8") as file:
    emotion_mapping = json.load(file)

records = []
skipped_files = 0
processed_files = 0

# Process each dataset file in a consistent order
for filename in sorted(os.listdir(DATASET_DIR)):
    if not filename.lower().endswith(".csv"):
        continue

    parts = filename[:-4].split("_")

    # Expected filename: participant_emotionID_trial
    if len(parts) != 3:
        skipped_files += 1
        continue

    participant, emotion_id, trial = parts
    emotion = emotion_mapping.get(emotion_id)

    if emotion not in EMOTION_ORDER:
        skipped_files += 1
        continue

    file_path = os.path.join(DATASET_DIR, filename)
    frames = []

    try:
        with open(file_path, "r", newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip the header

            for row in reader:
                if len(row) != EXPECTED_COLUMNS:
                    continue

                try:
                    values = np.asarray(
                        [float(value) for value in row[1:]],
                        dtype=np.float32
                    )
                except ValueError:
                    continue

                if values.size != EXPECTED_JOINTS * 3:
                    continue

                if not np.isfinite(values).all():
                    continue

                frames.append(values.reshape(EXPECTED_JOINTS, 3))

    except (OSError, csv.Error) as error:
        print(f"Skipping {filename}: {error}")
        skipped_files += 1
        continue

    if not frames:
        skipped_files += 1
        continue

    frames = np.asarray(frames, dtype=np.float32)

    # Preserve the original coordinates for now.
    # Do not assume joint 1 is the hip until the dataset's
    # joint ordering has been confirmed.
    for frame_index, joints in enumerate(frames):
        x = joints[:, 0]
        y = joints[:, 1]
        z = joints[:, 2]

        # Feature order: all X values, then all Y, then all Z
        features = np.concatenate([x, y, z])

        records.append([
            participant,
            emotion_id,
            emotion,
            trial,
            frame_index,
            *features
        ])

    processed_files += 1

# Create feature column names
feature_names = [
    f"joint_{joint_index}_{axis}"
    for axis in ["x", "y", "z"]
    for joint_index in range(1, EXPECTED_JOINTS + 1)
]

header = [
    "participant",
    "emotion_id",
    "emotion",
    "trial",
    "frame_index",
] + feature_names

# Save to a new file, never overwrite the original CSV
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(records)

print("BERD v2 preprocessing completed.")
print("Processed files:", processed_files)
print("Skipped files:", skipped_files)
print("Total feature rows:", len(records))
print("Features per frame:", len(feature_names))
print("Saved to:", OUTPUT_FILE)