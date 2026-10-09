import cv2
import mediapipe as mp
import json

print("Loading MediaPipe Pose Landmarker...")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

model_path = "behavior/pose_landmarker.task"

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO
)

video_path = "behavior/test_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

print("Video FPS:", fps)

all_frames = []
frame_number = 0

with PoseLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int((frame_number / fps) * 1000)

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.pose_landmarks:

            landmarks = []

            for landmark in result.pose_landmarks[0]:

                landmarks.append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                    "presence": landmark.presence
                })

            all_frames.append({
                "frame": frame_number,
                "timestamp_ms": timestamp_ms,
                "landmarks": landmarks
            })

            print(
                f"Frame {frame_number}: "
                f"{len(landmarks)} landmarks saved"
            )

        else:

            print(
                f"Frame {frame_number}: "
                "No person detected"
            )

        frame_number += 1

cap.release()

output_file = "behavior/pose_landmarks.json"

with open(output_file, "w") as file:
    json.dump(all_frames, file, indent=2)

print("\nVideo processing completed.")
print("Total frames processed:", frame_number)
print("Frames with landmarks:", len(all_frames))
print("Saved to:", output_file)