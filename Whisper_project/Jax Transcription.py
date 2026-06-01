import argparse
import os
import pandas as pd
from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline
import jax.numpy as jnp
import datetime

def format_time(seconds):
    td = datetime.timedelta(seconds=seconds)
    return str(td)[:-3]  # Remove microseconds

def create_srt(transcription, duration, step=5):
    words = transcription.split()
    srt = []
    for i in range(0, len(words), step):
        start_time = format_time(i * duration / len(words))
        end_time = format_time((i + step) * duration / len(words))
        subtitle_text = " ".join(words[i:i + step])
        srt.append(f"{i//step + 1}\n{start_time} --> {end_time}\n{subtitle_text}\n")
    return "\n".join(srt)

# Set up argument parser
parser = argparse.ArgumentParser(description='Script to transcribe audio files in a directory using Whisper Models and generate subtitle files.')
parser.add_argument(
    "--hf_model",
    type=str,
    default="openai/whisper-tiny",
    help="Huggingface model name. Example: openai/whisper-tiny",
) 
parser.add_argument(
    "--path_to_audio_folder",
    type=str,
    default="Music",
    help="Path to the folder containing audio files.",
)
parser.add_argument(
    "--language",
    type=str,
    default="EN",
    help="Two letter language code for the transcription language, e.g. 'EN' for English.",
)
parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="The device to run the pipeline on. -1 for CPU, 0 for the first GPU (default) and so on.",
)
parser.add_argument(
    "--half_precision",
    type=lambda x: (str(x).lower() == 'true'),
    default=False,
    help="Run with half precision.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for inference.",
)

args = parser.parse_args()

# Initialize the Whisper pipeline
model_id = args.hf_model
dtype = jnp.float16 if args.half_precision else jnp.float32

transcribe = FlaxWhisperPipline(
    model_id=model_id,
    dtype=dtype,
    batch_size=args.batch_size
)

transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language=args.language, task="transcribe")

# Process each audio file in the folder
for filename in os.listdir(args.path_to_audio_folder):
    if filename.endswith(".wav"):  # Check if the file is a WAV audio file
        filepath = os.path.join(args.path_to_audio_folder, filename)
        transcription = transcribe(filepath)["text"]
        
        # Assuming we have a function to get the duration of the audio file in seconds
        duration = get_audio_duration(filepath)
        
        # Create SRT content
        srt_content = create_srt(transcription, duration)
        
        # Save SRT file
        srt_filename = os.path.splitext(filename)[0] + ".srt"
        srt_filepath = os.path.join(args.path_to_audio_folder, srt_filename)
        
        with open(srt_filepath, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)
        
        print(f"Subtitles for {filename} have been successfully saved to {srt_filename}.")

# Function to get the duration of the audio file
def get_audio_duration(file_path):
    import wave
    with wave.open(file_path, 'r') as audio:
        frames = audio.getnframes()
        rate = audio.getframerate()
        duration = frames / float(rate)
        return duration
