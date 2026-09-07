import os
from features.mel_spectrogram import AudioPreprocessor
from data.for_data import wav2melspec

root_path = r"D:\Project\Synthetic Speech Recognizer\datasets\raw\for-norm\for-norm"
output_dir = r"D:\Project\Synthetic Speech Recognizer\datasets\processed"

for split in os.listdir(root_path):
    split_path = os.path.join(root_path, split)

    if "train" in split:
        output_path = os.path.join(output_dir, "train.pt")

    elif "test" in split:
        output_path = os.path.join(output_dir, "test.pt")

    elif "val" in split:
        output_path = os.path.join(output_dir, "val.pt")
        
    wav2melspec(split_path, output_path, AudioPreprocessor())