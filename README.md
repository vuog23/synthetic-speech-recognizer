<div align="center">

# Synthetic Speech Recognizer

### Identify human and synthetic speech from audio

Record, upload, or generate speech, then classify it with a ConvNeXt V2 audio model.

<a href="#1-project-overview">Overview</a> ·
<a href="#2-tech-stack">Tech stack</a> ·
<a href="#3-ai-workflow">Workflow</a> ·
<a href="#4-dataset">Dataset</a> ·
<a href="#5-install-and-run">Installation</a>

</div>

<br>

<p align="center">
  <img src="./docs/images/application-preview.png" alt="Synthetic Speech Recognizer application preview" width="820">
</p>

<br>

## 1. Project Overview

Synthetic Speech Recognizer is a web application for recording, uploading, generating, and analyzing speech. It predicts whether an audio clip is **Human** or **Synthetic**.

The application supports two ways to start:

- Record or upload speech, transcribe it with Whisper, then analyze the audio.
- Enter text, generate speech with Kokoro, then analyze the generated audio.

The classifier uses a ConvNeXt Base INT8 ONNX model. Audio is processed locally by the FastAPI application; recordings and transcripts are not stored by the server.

> [!TIP]
> Model source: [ConvNeXt V2 Synthetic Speech Recognizer on Hugging Face](https://huggingface.co/vuog23/ConvNeXt_V2_Synthetic_Speech_Recognizer)

## 2. Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.2 or later">
  <img src="https://img.shields.io/badge/ONNX_Runtime-1.20%2B-005CED?logo=onnx&logoColor=white" alt="ONNX Runtime 1.20 or later">
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI 0.115 or later">
  <img src="https://img.shields.io/badge/Uvicorn-0.30%2B-4051B5?logo=uvicorn&logoColor=white" alt="Uvicorn 0.30 or later">
  <img src="https://img.shields.io/badge/Transformers-4.51.3-FFD21E?logo=huggingface&logoColor=black" alt="Transformers 4.51.3">
  <img src="https://img.shields.io/badge/Librosa-0.10%2B-4B8BBE" alt="Librosa 0.10 or later">
  <img src="https://img.shields.io/badge/Kokoro-0.9%2B-8B5CF6" alt="Kokoro 0.9 or later">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20JavaScript-F7DF1E?logo=javascript&logoColor=black" alt="HTML, CSS, and JavaScript">
  <img src="https://img.shields.io/badge/Speech--to--Text-Whisper%20Base-412991?logo=openai&logoColor=white" alt="Whisper Base">
  <img src="https://img.shields.io/badge/Classifier-ConvNeXt%20Base%20INT8-16A34A" alt="ConvNeXt Base INT8">
</p>

## 3. AI Workflow

```mermaid
flowchart LR
    A[Speech audio] --> B[Whisper transcription]
    C[Text input] --> D[Kokoro speech generation]
    D --> E[Generated audio]
    B --> F[Audio preprocessing]
    E --> F
    F --> G[ConvNeXt classifier]
    G --> H{Prediction}
    H --> I[Human]
    H --> J[Synthetic]
```

> [!NOTE]
> Whisper provides a transcript for recorded or uploaded speech. The final Human/Synthetic prediction is based on the audio signal, not the transcript.

## 4. Dataset

The training dataset combines the following sources:

| Source | Purpose |
| --- | --- |
| **ASVspoof 2019** | Genuine and spoofed speech samples |
| **LibriSpeech train-clean-100** | Genuine human speech samples |

The combined dataset is organized into `train`, `validation`, and `test` splits with two classes:

- `real` / Human
- `fake` / Synthetic

The classifier is trained to distinguish real human recordings from synthesized or spoofed speech.

> [!TIP]
> Dataset source: [ASVspoof + LibriSpeech on Kaggle](https://www.kaggle.com/datasets/trieuvuongnguyen/asvspoof-librispeech/settings)

## 5. Install and Run

### Requirements

- Python 3.11
- Internet access for the first model download
- A local ONNX classifier at `models/convnext_base/best_model_int8.onnx`

### Windows PowerShell

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

Open [http://localhost:3000](http://localhost:3000) in Chrome or Edge. Allow microphone access if you want to record audio.

> [!IMPORTANT]
> The first transcription or text-to-speech request downloads the required model files. Keep the terminal open while using the application.

To run the application again after installation:

```powershell
.\.venv\Scripts\python.exe run.py
```

Set a different port when needed:

```powershell
$env:PORT = 8000
.\.venv\Scripts\python.exe run.py
```

API documentation is available at [http://localhost:3000/docs](http://localhost:3000/docs).
