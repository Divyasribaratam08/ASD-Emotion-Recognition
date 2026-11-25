
import csv
import json
from pathlib import Path

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "datasets" / "stimulus" / "word"
MAPPING_FILE = ROOT / "behavior" / "emotion_mapping.json"
OUTPUT_FILE = ROOT / "behavior" / "berd_features_v3.csv"

EXPECTED_JOINTS = 18
COORDINATES_PER_JOINT = 3
EXPECTED_COLUMNS = 1 + EXPECTED_JOINTS * COORDINATES_PER_JOINT

# Filename format: participant_emotion_id_trial.csv
# Example: 10_1_1.csv

# --------------------------------------------------
# 2. Load emotion mapping
# --------------------------------------------------

with open(MAPPING_FILE, "r", encoding="utf-8") as file:
    emotion_mapping = json.load(file)

records = []
processed_files = 0
skipped_files = 0

feature_columns = [
    f"joint_{joint}_{axis}"
    for joint in range(EXPECTED_JOINTS)
    for axis in ("x", "y", "z")
]

# --------------------------------------------------
# 3. Process each CSV file
# --------------------------------------------------

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}"
    )

for file_path in sorted(DATA_DIR.glob("*.csv")):
    try:
        # Read CSV rows
        with open(
            file_path,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            rows = list(csv.reader(file))

        if not rows:
            print(f"Skipped {file_path.name}: empty file")
            skipped_files += 1
            continue

        # Convert valid numeric rows.
        # Each row must contain:
        # frame index + 18 joints * 3 coordinates = 55 values.
        numeric_rows = []

        for row in rows:
            if len(row) != EXPECTED_COLUMNS:
                continue

            try:
                numeric_row = [float(value) for value in row]
            except ValueError:
                # Allows a non-numeric header row to be ignored.
                continue

            numeric_rows.append(numeric_row)

        if not numeric_rows:
            print(
                f"Skipped {file_path.name}: "
                "no valid 55-column numeric rows"
            )
            skipped_files += 1
            continue

        # Extract the 54 joint coordinates.
        # The first CSV column is the frame index.
        coordinates = [
            row[1:] for row in numeric_rows
        ]

        # Reshape each frame into 18 joints, each with x, y, z.
        frames = [
            [
                coordinates[frame_index][
                    joint_index * COORDINATES_PER_JOINT:
                    (joint_index + 1) * COORDINATES_PER_JOINT
                ]
                for joint_index in range(EXPECTED_JOINTS)
            ]
            for frame_index in range(len(coordinates))
        ]

        # --------------------------------------------------
        # 4. Hip-relative coordinate transformation
        # --------------------------------------------------
        # Assumption: joint index 0 represents the hip,
        # based on the joint order described in the paper.
        # The joint-index mapping should be verified.
        #
        # Use the hip position in the FIRST frame as a fixed
        # reference for every joint in every frame.
        # Do NOT recenter each frame separately.

        initial_hip = frames[0][0]

        relative_frames = []

        for joints in frames:
            relative_joints = []

            for joint in joints:
                relative_joint = [
                    joint[axis] - initial_hip[axis]
                    for axis in range(COORDINATES_PER_JOINT)
                ]
                relative_joints.append(relative_joint)

            relative_frames.append(relative_joints)

        # --------------------------------------------------
        # 5. Parse filename metadata
        # --------------------------------------------------

        filename_parts = file_path.stem.split("_")

        if len(filename_parts) != 3:
            print(
                f"Skipped {file_path.name}: "
                "expected participant_emotion_trial"
            )
            skipped_files += 1
            continue

        participant_text, emotion_id_text, trial_text = filename_parts

        participant = int(participant_text)
        emotion_id = int(emotion_id_text)
        trial = int(trial_text)

        if emotion_id_text not in emotion_mapping:
            print(
                f"Skipped {file_path.name}: "
                f"emotion ID {emotion_id_text} not in mapping"
            )
            skipped_files += 1
            continue

        emotion = emotion_mapping[emotion_id_text]

        # --------------------------------------------------
        # 6. Save one record per frame
        # --------------------------------------------------

        for frame_index, joints in enumerate(relative_frames):
            flattened_coordinates = [
                coordinate
                for joint in joints
                for coordinate in joint
            ]

            records.append([
                participant,
                emotion_id,
                emotion,
                trial,
                frame_index,
                *flattened_coordinates
            ])

        processed_files += 1

    except (ValueError, IndexError, KeyError) as error:
        print(f"Skipped {file_path.name}: {error}")
        skipped_files += 1

# --------------------------------------------------
# 7. Write the output CSV
# --------------------------------------------------

columns = [
    "participant",
    "emotion_id",
    "emotion",
    "trial",
    "frame_index",
    *feature_columns
]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(columns)
    writer.writerows(records)

# --------------------------------------------------
# 8. Print summary
# --------------------------------------------------

print("\nBERD v3 preprocessing completed.")
print(f"Processed files: {processed_files}")
print(f"Skipped files: {skipped_files}")
print(f"Total feature rows: {len(records)}")
print(f"Features per frame: {len(feature_columns)}")
print(f"Total columns including metadata: {len(columns)}")
print(f"Saved to: {OUTPUT_FILE}")