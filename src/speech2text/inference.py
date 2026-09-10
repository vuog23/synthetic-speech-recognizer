from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-base",
    cache_dir=r"D:\Project\Synthetic Speech Recognizer\.cache")

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-base",
    cache_dir=r"D:\Project\Synthetic Speech Recognizer\.cache")

model.config.forced_decoder_ids = None

audio_path = r"D:\Project\Synthetic Speech Recognizer\datasets\raw\for-2sec\for-2seconds\testing\fake\file2.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav"

audio, sampling_rate = librosa.load(
    audio_path,
    sr=16000,
    mono=True
)

input_features = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
).input_features

with torch.no_grad():
    predicted_ids = model.generate(input_features)

transcription = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True
)

print(transcription[0])