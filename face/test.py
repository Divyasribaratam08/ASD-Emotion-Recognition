from fer.fer import FER
import cv2

detector = FER()

image = cv2.imread("face/myImage.png")

emotions = detector.detect_emotions(image)

print("\nDetected emotions:")
print(emotions)