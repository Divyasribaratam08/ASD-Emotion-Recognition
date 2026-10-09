from fer.fer import FER
import cv2
import json

INPUT_FILE = "face/myImage.png"
OUTPUT_FILE = "face/face_emotion.json"

# Common emotion order used by our fusion system
EMOTION_ORDER = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

detector = FER()

image = cv2.imread(INPUT_FILE)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {INPUT_FILE}"
    )

detections = detector.detect_emotions(image)

if not detections:
    raise ValueError("No face detected in the image.")

# Use the first detected face for this initial test
emotions = detections[0]["emotions"]

standardized_emotions = {
    emotion: float(emotions.get(emotion, 0.0))
    for emotion in EMOTION_ORDER
}

result = {
    "source": INPUT_FILE,
    "emotion_order": EMOTION_ORDER,
    "emotions": standardized_emotions
}

with open(OUTPUT_FILE, "w") as file:
    json.dump(result, file, indent=2)

print("Face emotion extraction completed.")
print("Emotion order:", EMOTION_ORDER)
print("Emotion probabilities:")

for emotion in EMOTION_ORDER:
    print(
        f"{emotion}: "
        f"{standardized_emotions[emotion] * 100:.2f}%"
    )

print("Saved to:", OUTPUT_FILE)