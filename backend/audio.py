"""The browser decodes recordings/uploads and sends PCM16 WAV to the API."""
import io
import wave

import numpy as np


def decode_wav(content: bytes, max_seconds: float = 180):
    try:
        with wave.open(io.BytesIO(content), "rb") as source:
            rate = source.getframerate()
            channels = source.getnchannels()
            frames = source.getnframes()
            if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
                raise ValueError("Send uncompressed PCM16 WAV audio.")
            if not 8000 <= rate <= 192000 or channels not in (1, 2):
                raise ValueError("Audio must be mono or stereo at 8–192 kHz.")
            if not 0.2 <= frames / rate <= max_seconds + 0.5:
                raise ValueError(f"Audio must be between 0.2 and {max_seconds} seconds.")
            raw = source.readframes(frames)
            if len(raw) != frames * channels * 2:
                raise ValueError("The WAV audio is incomplete.")
    except (wave.Error, EOFError) as error:
        raise ValueError("Invalid WAV audio. Record again or upload another file.") from error
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels)
    return samples.mean(axis=1) / 32768.0, rate


def encode_wav(samples, sample_rate: int = 24000) -> bytes:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("The speech model returned no valid audio.")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()
