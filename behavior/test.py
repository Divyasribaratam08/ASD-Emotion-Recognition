import cv2
import mediapipe as mp

print("Loading MediaPipe Pose Landmarker...")

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
vision = mp.tasks.vision

model_path = "behavior/pose_landmarker.task"

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

image = cv2.imread("behavior/test_image.png")

if image is None:
    print("ERROR: Could not load test image.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=image_rgb
)

with PoseLandmarker.create_from_options(options) as landmarker:

    results = landmarker.detect(mp_image)

    if not results.pose_landmarks:
        print("No person detected.")
    else:
        print("\nPerson detected!")
        print(
            "Number of landmarks:",
            len(results.pose_landmarks[0])
        )

        print("\nSample landmarks:")

        for i, landmark in enumerate(
            results.pose_landmarks[0][:5]
        ):
            print(
                i,
                "x:", round(landmark.x, 3),
                "y:", round(landmark.y, 3),
                "z:", round(landmark.z, 3)
            )