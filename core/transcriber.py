#core/transcriber.py
import whisper
import subprocess
import tempfile
import os
from pydub import AudioSegment

_model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))

def extract_audio(video_path: str) -> str:
    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000", wav_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_file

def get_audio_duration_sec(wav_path: str) -> float:
    try:
        audio = AudioSegment.from_wav(wav_path)
        return round(audio.duration_seconds, 2)
    except Exception:
        return -1.0  # fallback

def transcribe_video(video_path: str) -> tuple[str, float]:
    wav = extract_audio(video_path)
    duration = get_audio_duration_sec(wav)
    result = _model.transcribe(wav, language="en")
    os.remove(wav)
    return result["text"].strip(), duration
