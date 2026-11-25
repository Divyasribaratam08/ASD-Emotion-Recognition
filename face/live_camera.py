from fer.fer import FER
import cv2

# Load the pretrained facial emotion recognition model
detector = FER()

# Open the default webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Camera could not be opened. Check webcam permissions."
    )

print("Live Facial Emotion Recognition started.")
print("Press Q to stop the camera.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Unable to read camera frame.")
        break

    # Detect faces and predict emotions
    detections = detector.detect_emotions(frame)

    for detection in detections:
        x, y, w, h = detection["box"]
        emotions = detection["emotions"]

        # Find the emotion with the highest score
        predicted_emotion = max(
            emotions, key=emotions.get
        )
        confidence = emotions[predicted_emotion]

        # Draw face bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display emotion and confidence
        label = (
            f"{predicted_emotion}: "
            f"{confidence * 100:.1f}%"
        )

        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    if not detections:
        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Live Facial Emotion Recognition",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Live camera stopped.")