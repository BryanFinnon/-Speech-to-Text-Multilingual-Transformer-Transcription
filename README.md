# Multilingual Speech-to-Text with Whisper

[![Quality checks](https://github.com/BryanFinnon/multilingual-speech-to-text/actions/workflows/quality.yml/badge.svg)](https://github.com/BryanFinnon/multilingual-speech-to-text/actions/workflows/quality.yml)

A research prototype combining Whisper fine-tuning, Word Error Rate evaluation, subtitle generation and browser-based transcription.

## Components

| Component | Purpose |
|---|---|
| `main.py` | Fine-tuning experiment using a medical speech dataset |
| `evaluate_on_custom_dataset.py` | Evaluation with Word Error Rate |
| `Jax Transcription.py` | WAV transcription and SRT subtitle generation |
| `AUDIO_ANALYSIS.ipynb` | Exploratory audio analysis |
| `whisper-web-main/` | React and Vite transcription interface |

## Technology

Python · PyTorch · Hugging Face Transformers · Whisper · JAX · React · TypeScript · Vite

## Python setup

```bash
git clone https://github.com/BryanFinnon/multilingual-speech-to-text.git
cd multilingual-speech-to-text/Whisper_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If a private or gated Hugging Face resource is required, provide the token through the environment:

```bash
export HF_TOKEN=your_token
```

Never commit access tokens.

## Generate subtitles

```bash
python "Jax Transcription.py" \
  --path_to_audio_folder ./audio \
  --hf_model openai/whisper-tiny \
  --language EN
```

## Run the web interface

```bash
cd whisper-web-main
npm install
npm run dev
```

## Evaluation

The evaluation utilities use Word Error Rate. A reproducible comparison should record the dataset split, checkpoint, decoding parameters, hardware and dependency versions.

## Limitations

- This is an experimental repository, not a hosted transcription service.
- No reproducible latency or accuracy benchmark is currently published.
- SRT timings are estimated across groups of words rather than generated from word-level timestamps.
- Fine-tuning may require a CUDA-capable GPU.
