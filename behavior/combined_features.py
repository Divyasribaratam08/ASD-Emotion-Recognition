import json

POSTURE_FILE = "behavior/behavior_features.json"
MOVEMENT_FILE = "behavior/temporal_features.json"
OUTPUT_FILE = "behavior/combined_features.json"


with open(POSTURE_FILE, "r") as file:
    posture_data = json.load(file)

with open(MOVEMENT_FILE, "r") as file:
    movement_data = json.load(file)


if len(posture_data) != len(movement_data):
    raise ValueError("Posture and movement frame counts do not match.")


combined_data = []

for posture_frame, movement_frame in zip(
    posture_data,
    movement_data
):

    if posture_frame["frame"] != movement_frame["frame"]:
        raise ValueError(
            f"Frame mismatch: "
            f"{posture_frame['frame']} vs "
            f"{movement_frame['frame']}"
        )

    combined_features = {}

    # Add posture features
    for name, value in posture_frame["features"].items():
        combined_features[name] = value

    # Add movement features
    for name, value in movement_frame["movement"].items():
        combined_features[name] = value

    combined_data.append({
        "frame": posture_frame["frame"],
        "timestamp_ms": posture_frame["timestamp_ms"],
        "features": combined_features
    })


with open(OUTPUT_FILE, "w") as file:
    json.dump(combined_data, file, indent=2)


print("Behavior feature combination completed.")
print("Frames combined:", len(combined_data))
print(
    "Total behavior features per frame:",
    len(combined_data[0]["features"])
)
print("Saved to:", OUTPUT_FILE)