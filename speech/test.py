import torch
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

model_name = "Dpngtm/wav2vec2-emotion-recognition"
audio_file = "speech/Recording_fixed.wav"

print("Loading speech emotion model...")

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

print("Loading audio...")

# Load WAV file using SoundFile
waveform, sample_rate = sf.read(audio_file)

# Convert stereo audio to mono
if len(waveform.shape) > 1:
    waveform = waveform.mean(axis=1)

# Convert to PyTorch tensor
waveform = torch.tensor(waveform, dtype=torch.float32)

# Resample to 16 kHz if necessary
target_sample_rate = 16000

if sample_rate != target_sample_rate:
    import torchaudio

    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=target_sample_rate
    )

    waveform = resampler(waveform)

# Prepare audio for the model
inputs = feature_extractor(
    waveform.numpy(),
    sampling_rate=target_sample_rate,
    return_tensors="pt"
)

# Predict emotion
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(
        outputs.logits, dim=-1
    )[0]

predicted_id = torch.argmax(probabilities).item()
predicted_emotion = model.config.id2label[predicted_id]

print("\nSpeech Emotion Prediction:")
print("Emotion:", predicted_emotion)
print(
    "Confidence:",
    round(probabilities[predicted_id].item() * 100, 2),
    "%"
)

print("\nAll Emotion Probabilities:")

for i, probability in enumerate(probabilities):
    print(
        model.config.id2label[i],
        ":",
        round(probability.item() * 100, 2),
        "%"
    )