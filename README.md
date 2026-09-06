# Multilingual Speech-to-Text with Whisper

A research and engineering prototype for multilingual transcription, subtitle generation, and evaluation with OpenAI Whisper.

## What is included

- A Python fine-tuning experiment using Hugging Face Transformers and a medical ASR dataset.
- Evaluation utilities based on Word Error Rate (WER).
- A JAX transcription script that converts WAV files to SRT subtitles.
- A React/Vite browser interface for local transcription.
- Research notebooks and example evaluation data.

## Project status

This repository is a prototype, not a production service. Performance figures should only be treated as valid when they can be reproduced from a documented dataset split, model checkpoint, hardware configuration, and benchmark script. The repository currently does not publish a reproducible benchmark supporting latency or accuracy claims.

## Repository structure

```text
Whisper_project/
├── main.py                         # Whisper fine-tuning experiment
├── evaluate_on_custom_dataset.py   # WER evaluation
├── Jax Transcription.py            # WAV-to-SRT transcription
├── AUDIO_ANALYSIS.ipynb            # Exploratory analysis
├── requirements.txt                # Python dependencies
├── assets/                         # Screenshots
└── whisper-web-main/               # React/Vite interface
```

## Security and configuration

Never commit access tokens. Set the Hugging Face token in your shell when private or gated resources require it:

```bash
export HF_TOKEN="your_token_here"
```

The Python code reads `HF_TOKEN` from the environment. Public models and datasets may not require a token.

## Python setup

```bash
git clone https://github.com/BryanFinnon/-Speech-to-Text-Multilingual-Transformer-Transcription.git
cd -- -Speech-to-Text-Multilingual-Transformer-Transcription/Whisper_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the training experiment:

```bash
python main.py
```

Run WAV-to-SRT transcription:

```bash
python "Jax Transcription.py" \
  --path_to_audio_folder ./audio \
  --hf_model openai/whisper-tiny \
  --language EN
```

## Web interface

```bash
cd Whisper_project/whisper-web-main
npm install
npm run dev
```

## Evaluation

WER is the principal metric used by the evaluation script. For a trustworthy comparison, record:

1. Dataset and exact train/test split
2. Base model and fine-tuned checkpoint
3. Decoding parameters
4. Hardware and dependency versions
5. Mean and variance across repeated runs where relevant

## Limitations

- The training script is designed for experimentation and may require a CUDA GPU.
- Dataset access and model downloads depend on Hugging Face availability.
- Generated subtitles use approximate time allocation rather than word-level timestamps.
- No hosted API or production deployment is included.

## Author

Bryan Finnon — MSc Computer Science (Distinction), focused on applied AI and software engineering.
