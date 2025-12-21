from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Load the model and processor
processor = WhisperProcessor.from_pretrained("cahya/whisper-large-audio-captioning-v1.0")
model = WhisperForConditionalGeneration.from_pretrained("cahya/whisper-large-audio-captioning-v1.0")

# Load your audio file
import librosa
audio, sr = librosa.load("your_audio.wav", sr=16000)

# Process and generate caption
input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription[0])
