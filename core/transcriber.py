import whisper
import subprocess
import tempfile
import os
from pydub import AudioSegment

# load once
_model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))

def extract_audio(video_path: str) -> str:
    """Convert video → mono WAV for Whisper, return temp wav path."""
    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000", wav_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_file

def transcribe_video(video_path: str) -> str:
    """Extract audio and run Whisper transcription."""
    wav = extract_audio(video_path)
    result = _model.transcribe(wav, language="en")
    os.remove(wav)
    return result["text"].strip()
