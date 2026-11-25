import os
import csv
import json
import numpy as np

DATASET_DIR = "datasets/stimulus/word"
MAPPING_FILE = "behavior/emotion_mapping.json"
OUTPUT_FILE = "behavior/berd_features.csv"

with open(MAPPING_FILE, "r") as file:
    emotion_mapping = json.load(file)

records = []

for filename in os.listdir(DATASET_DIR):

    if not filename.endswith(".csv"):
        continue

    parts = filename.replace(".csv", "").split("_")

    if len(parts) != 3:
        continue

    participant = parts[0]
    emotion_id = parts[1]
    trial = parts[2]

    emotion = emotion_mapping.get(emotion_id)

    if emotion is None:
        continue

    file_path = os.path.join(DATASET_DIR, filename)

    with open(file_path, "r") as file:
        reader = csv.reader(file)

        header = next(reader)

        for row in reader:

            if len(row) != 55:
                continue

            values = np.array(
                [float(value) for value in row[1:]],
                dtype=np.float32
            )

            joints = values.reshape(18, 3)

            x = joints[:, 0]
            y = joints[:, 1]
            z = joints[:, 2]

            center_x = np.mean(x)
            center_y = np.mean(y)
            center_z = np.mean(z)

            x = x - center_x
            y = y - center_y
            z = z - center_z

            scale = np.max(
                np.sqrt(x**2 + y**2 + z**2)
            )

            if scale > 0:
                x = x / scale
                y = y / scale
                z = z / scale

            features = np.concatenate([x, y, z])

            records.append(
                [
                    participant,
                    emotion_id,
                    emotion,
                    trial,
                    *features
                ]
            )

feature_names = [
    f"joint_{i+1}_{axis}"
    for axis in ["x", "y", "z"]
    for i in range(18)
]

header = [
    "participant",
    "emotion_id",
    "emotion",
    "trial"
] + feature_names

with open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(header)
    writer.writerows(records)

print("BERD preprocessing completed.")
print("Total feature rows:", len(records))
print("Saved to:", OUTPUT_FILE)