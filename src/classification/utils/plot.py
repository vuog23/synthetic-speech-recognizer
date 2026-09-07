import matplotlib.pyplot as plt
import librosa
import numpy as np
from IPython.display import Audio

class AudioVisualizer:
    def __init__(self, sr=16000):
        self.sr = sr

    def plot_waveform(self, wav):
        time = np.arange(len(wav)) / self.sr
        plt.figure(figsize=(20, 3))
        plt.plot(time, wav)
        plt.title("Waveform")
        plt.tight_layout()
        plt.show()

    def plot_spectrogram(self, spec, hop_length=256):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            spec,
            sr=self.sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar()
        plt.title("Mel Spectrogram")
        plt.tight_layout()
        plt.show()