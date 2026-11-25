import json
import numpy as np
import sounddevice as sd
import torch

from pathlib import Path
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification
)

# Project paths
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT / "speech" / "live_speech_emotion.json"

# Existing pretrained model
MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"

SAMPLE_RATE = 16000
CHUNK_DURATION = 3

EMOTION_ORDER = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Convert model labels into common emotion labels
EMOTION_MAPPING = {
    "angry": "angry",
    "calm": "neutral",
    "disgust": "disgust",
    "fearful": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprised": "surprise"
}


print("Loading speech emotion recognition model...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Model loaded successfully!")


def predict_emotion(audio):

    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    model_labels = model.config.id2label

    standardized_emotions = {
        emotion: 0.0
        for emotion in EMOTION_ORDER
    }

    for index, probability in enumerate(probabilities):

        model_emotion = model_labels[index].lower()

        common_emotion = EMOTION_MAPPING.get(
            model_emotion
        )

        if common_emotion:
            standardized_emotions[common_emotion] = float(
                probability.item()
            )

    return standardized_emotions


def main():

    print("\nLIVE SPEECH EMOTION RECOGNITION")
    print("--------------------------------")
    print("Speak into your microphone.")
    print("Emotion is predicted for every 3-second audio chunk.")
    print("Press Ctrl+C to stop.\n")

    try:

        while True:

            print("Listening for 3 seconds...")

            recording = sd.rec(
                int(CHUNK_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )

            sd.wait()

            audio = recording.flatten()

            if not np.isfinite(audio).all():
                print("Invalid audio detected. Trying again.")
                continue

            if np.max(np.abs(audio)) < 0.001:
                print("No speech detected. Please speak louder.")
                continue

            emotions = predict_emotion(audio)

            predicted_emotion = max(
                emotions,
                key=emotions.get
            )

            confidence = emotions[predicted_emotion]

            print("\nPredicted emotion:", predicted_emotion)
            print(
                "Confidence:",
                f"{confidence * 100:.2f}%"
            )

            print("\nAll emotion probabilities:")

            for emotion in EMOTION_ORDER:
                print(
                    f"{emotion:10s}: "
                    f"{emotions[emotion] * 100:.2f}%"
                )

            result = {
                "source": "Live microphone",
                "model": MODEL_NAME,
                "sample_rate": SAMPLE_RATE,
                "chunk_duration_seconds": CHUNK_DURATION,
                "predicted_emotion": predicted_emotion,
                "emotions": emotions
            }

            with open(
                OUTPUT_FILE,
                "w",
                encoding="utf-8"
            ) as file:
                json.dump(result, file, indent=4)

            print("\nResult saved successfully.")
            print("--------------------------------\n")

    except KeyboardInterrupt:
        print("\nLive speech recognition stopped.")

    finally:
        sd.stop()


if __name__ == "__main__":
    main()