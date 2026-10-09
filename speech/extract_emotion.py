import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import json

INPUT_FILE = "speech/Recording_fixed.wav"
OUTPUT_FILE = "speech/speech_emotion.json"

MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"

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

print("Loading speech emotion model...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Loading audio...")

waveform, sample_rate = torchaudio.load(INPUT_FILE)

# Convert stereo to mono if necessary
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Resample to 16 kHz if necessary
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=16000
    )
    waveform = resampler(waveform)

audio = waveform.squeeze().numpy()

inputs = feature_extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.softmax(
    outputs.logits,
    dim=-1
)[0]

model_labels = model.config.id2label

# Convert model labels into our common emotion names
emotion_mapping = {
    "angry": "angry",
    "calm": "neutral",
    "disgust": "disgust",
    "fearful": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprised": "surprise"
}

standardized_emotions = {
    emotion: 0.0
    for emotion in EMOTION_ORDER
}

for index, probability in enumerate(probabilities):

    model_emotion = model_labels[index].lower()

    common_emotion = emotion_mapping.get(
        model_emotion
    )

    if common_emotion:
        standardized_emotions[common_emotion] = float(
            probability.item()
        )

predicted_emotion = max(
    standardized_emotions,
    key=standardized_emotions.get
)

result = {
    "source": INPUT_FILE,
    "model": MODEL_NAME,
    "emotion_order": EMOTION_ORDER,
    "predicted_emotion": predicted_emotion,
    "emotions": standardized_emotions
}

with open(OUTPUT_FILE, "w") as file:
    json.dump(result, file, indent=2)

print("\nSpeech emotion extraction completed.")
print("Emotion order:", EMOTION_ORDER)

print("\nEmotion probabilities:")

for emotion in EMOTION_ORDER:

    print(
        f"{emotion}: "
        f"{standardized_emotions[emotion] * 100:.2f}%"
    )

print(
    "\nPredicted emotion:",
    predicted_emotion
)

print("Saved to:", OUTPUT_FILE)