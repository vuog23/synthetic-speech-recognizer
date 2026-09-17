import io
import wave
from dataclasses import replace
from threading import Lock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.audio import decode_wav, encode_wav
from backend.config import settings
from backend.inference import InferenceService, ModelUnavailable
from backend.main import app, get_service


class FakeService:
    def __init__(self):
        self.lock = Lock()
        self.received = None

    def transcribe(self, samples, rate):
        self.received = (samples, rate)
        return {"text": "A test recording."}

    def synthesize(self, text, voice, speed):
        self.received = (text, voice, speed)
        return encode_wav(np.zeros(24000), 24000)

    def classify(self, samples, rate):
        self.received = (samples, rate)
        return {"label": "Human", "probability": 0.8, "scores": [0.8, 0.2],
                "labels": ["Human", "Synthetic"], "model": "test", "windowSeconds": 3}


@pytest.fixture
def api():
    fake = FakeService()
    app.dependency_overrides[get_service] = lambda: fake
    with TestClient(app) as client:
        yield client, fake
    app.dependency_overrides.clear()


@pytest.fixture
def wav():
    return encode_wav(np.sin(np.arange(16000) * 0.1) * 0.25, 16000)


def test_page_and_static_boundaries(api):
    client, _ = api
    page = client.get("/")
    assert page.status_code == 200
    assert 'id="result-title"' in page.text
    for removed in ['class="sidebar"', 'class="results-column"', 'id="new-session"', 'id="settings-dialog"', 'TWO WAYS TO EXPLORE', 'class="breadcrumb"']:
        assert removed not in page.text
    assert client.get("/app/app.js").status_code == 200
    for private in ["/.git/config", "/models/convnext_base/best_model_int8.onnx", "/backend/config.py", "/app/../backend/config.py"]:
        assert client.get(private).status_code == 404
    assert client.get("/api/health").json()["status"] == "ok"


@pytest.mark.parametrize("endpoint", ["transcribe", "classify"])
def test_audio_endpoints_accept_browser_wav(api, wav, endpoint):
    client, fake = api
    response = client.post(f"/api/{endpoint}", content=wav, headers={"Content-Type": "audio/wav"})
    assert response.status_code == 200
    samples, rate = fake.received
    assert samples.shape == (16000,)
    assert rate == 16000
    assert np.isfinite(samples).all()


def test_synthesis_returns_wav_and_validates_text_voice_speed(api):
    client, fake = api
    response = client.post("/api/synthesize", json={"text": " Hello ", "voice": "af_heart", "speed": 1.2})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["cache-control"] == "no-store"
    assert decode_wav(response.content)[1] == 24000
    assert fake.received == ("Hello", "af_heart", 1.2)
    for body in [{"text": " "}, {"text": "x" * 1001}, {"text": "Hello", "voice": "../../secret"}, {"text": "Hello", "speed": 0}, {"text": "Hello", "model_path": "anything"}]:
        assert client.post("/api/synthesize", json=body).status_code == 422


def test_bad_audio_and_content_type(api):
    client, _ = api
    assert client.post("/api/transcribe", content=b"bad", headers={"Content-Type": "audio/wav"}).status_code == 422
    assert client.post("/api/classify", json={"samples": []}).status_code == 415
    too_short = encode_wav(np.zeros(10), 16000)
    assert client.post("/api/classify", content=too_short, headers={"Content-Type": "audio/wav"}).status_code == 422


def test_upload_limit_checked_before_inference(api, monkeypatch):
    import backend.main as main
    monkeypatch.setattr(main, "settings", replace(settings, max_upload_bytes=20))
    client, fake = api
    response = client.post("/api/transcribe", content=b"x" * 21, headers={"Content-Type": "audio/wav"})
    assert response.status_code == 413
    assert fake.received is None


def test_busy_server_and_failures_release_lock(api, wav):
    client, fake = api
    fake.lock.acquire()
    assert client.post("/api/synthesize", json={"text": "Hello"}).status_code == 409
    fake.lock.release()
    def fail(*args):
        raise ModelUnavailable("Missing classifier weights")
    fake.classify = fail
    response = client.post("/api/classify", content=wav, headers={"Content-Type": "audio/wav"})
    assert response.status_code == 503
    assert "Missing classifier" in response.json()["detail"]
    assert not fake.lock.locked()


def test_wav_downmix_and_truncated_input():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.tile(np.array([16384, -16384], dtype="<i2"), 16000).tobytes())
    samples, rate = decode_wav(buffer.getvalue())
    assert rate == 16000
    assert np.all(samples == 0)
    with pytest.raises(ValueError, match="incomplete"):
        decode_wav(buffer.getvalue()[:-10])


def test_classifier_is_lazy_and_reports_missing_weights(tmp_path):
    service = InferenceService(replace(settings, classifier_path=tmp_path / "missing.onnx"))
    assert service.whisper is service.kokoro is service.classifier is None
    with pytest.raises(ModelUnavailable, match="classifier_path"):
        service.classify(np.zeros(16000), 16000)
