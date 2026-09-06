import argparse
import datetime
import os
import wave

import jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline


def format_time(seconds):
    td = datetime.timedelta(seconds=seconds)
    return str(td)[:-3]


def get_audio_duration(file_path):
    with wave.open(file_path, "r") as audio:
        return audio.getnframes() / float(audio.getframerate())


def create_srt(transcription, duration, step=5):
    words = transcription.split()
    if not words:
        return ""

    srt = []
    for i in range(0, len(words), step):
        start_time = format_time(i * duration / len(words))
        end_time = format_time(min(i + step, len(words)) * duration / len(words))
        subtitle_text = " ".join(words[i : i + step])
        srt.append(
            f"{i // step + 1}\n{start_time} --> {end_time}\n{subtitle_text}\n"
        )
    return "\n".join(srt)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe WAV files with Whisper and generate SRT subtitles."
    )
    parser.add_argument(
        "--hf_model",
        default="openai/whisper-tiny",
        help="Hugging Face model name, for example openai/whisper-tiny.",
    )
    parser.add_argument(
        "--path_to_audio_folder",
        default="Music",
        help="Directory containing WAV audio files.",
    )
    parser.add_argument(
        "--language",
        default="EN",
        help="Two-letter transcription language code.",
    )
    parser.add_argument(
        "--half_precision",
        type=lambda value: value.lower() == "true",
        default=False,
        help="Run inference with float16 precision.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = jnp.float16 if args.half_precision else jnp.float32
    transcribe = FlaxWhisperPipline(
        model_id=args.hf_model,
        dtype=dtype,
        batch_size=args.batch_size,
    )
    transcribe.model.config.forced_decoder_ids = (
        transcribe.tokenizer.get_decoder_prompt_ids(
            language=args.language,
            task="transcribe",
        )
    )

    for filename in os.listdir(args.path_to_audio_folder):
        if not filename.lower().endswith(".wav"):
            continue

        filepath = os.path.join(args.path_to_audio_folder, filename)
        transcription = transcribe(filepath)["text"]
        srt_content = create_srt(transcription, get_audio_duration(filepath))
        srt_filename = f"{os.path.splitext(filename)[0]}.srt"
        srt_filepath = os.path.join(args.path_to_audio_folder, srt_filename)

        with open(srt_filepath, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)

        print(f"Saved subtitles for {filename} to {srt_filename}.")


if __name__ == "__main__":
    main()
