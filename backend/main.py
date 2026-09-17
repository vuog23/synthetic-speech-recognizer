import logging
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .audio import decode_wav
from .config import ROOT, settings
from .inference import InferenceService, ModelUnavailable

logger = logging.getLogger(__name__)
app = FastAPI(title="Voxlab Speech API", version="2.0.0")
service = InferenceService(settings)


def get_service():
    return service


Service = Annotated[InferenceService, Depends(get_service)]


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=settings.max_text_characters)
    voice: Literal["af_heart", "af_bella", "am_michael", "bf_emma", "bm_george"] = "af_heart"
    speed: float = Field(default=1.0, ge=0.5, le=2.0, allow_inf_nan=False)

    @field_validator("text")
    @classmethod
    def require_words(cls, value):
        if not value.strip():
            raise ValueError("Enter text to generate speech.")
        return value


async def read_audio(request: Request):
    if request.headers.get("content-type", "").split(";")[0] not in ("audio/wav", "audio/x-wav", "application/octet-stream"):
        raise HTTPException(415, "Send the audio as PCM16 WAV.")
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > settings.max_upload_bytes:
            raise HTTPException(413, "Audio must be smaller than 25 MB.")
        content.extend(chunk)
    return bytes(content)


async def execute(models: InferenceService, operation, *args):
    # Fail fast while a model is running instead of accumulating large requests.
    if not models.lock.acquire(blocking=False):
        raise HTTPException(409, "The server is finishing another inference. Please try again shortly.")

    def work():
        try:
            return operation(*args)
        finally:
            models.lock.release()

    try:
        return await run_in_threadpool(work)
    except ModelUnavailable as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        logger.exception("Speech inference failed")
        raise HTTPException(503, "Inference failed. Check the server terminal for model, dependency, or download errors, then retry.") from error


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "app/index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health():
    return {"status": "ok", "classifier": settings.classifier_name}


@app.post("/api/transcribe")
async def transcribe(request: Request, models: Service):
    content = await read_audio(request)

    def process():
        samples, rate = decode_wav(content, settings.max_audio_seconds)
        return models.transcribe(samples, rate)

    return await execute(models, process)


@app.post("/api/synthesize", response_class=Response, responses={200: {"content": {"audio/wav": {}}}})
async def synthesize(body: SpeechRequest, models: Service):
    audio = await execute(models, models.synthesize, body.text, body.voice, body.speed)
    return Response(audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.post("/api/classify")
async def classify(request: Request, models: Service):
    content = await read_audio(request)

    def process():
        samples, rate = decode_wav(content, settings.max_audio_seconds)
        return models.classify(samples, rate)

    return await execute(models, process)


# Model weights, datasets and Python source are not exposed as static assets.
app.mount("/app", StaticFiles(directory=ROOT / "app"), name="app")
