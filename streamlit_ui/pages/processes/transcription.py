# pages/processes/transcription.py

import os
import io
import math
import tempfile
import contextlib

import streamlit as st

# MoviePy is used only for optional splitting of large uploads.
# If it's missing, we gracefully skip splitting.
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    _HAS_MOVIEPY = True
except Exception:
    _HAS_MOVIEPY = False

# Lazy-load local Whisper only when needed
_WHISPER_MODEL = None


def _get_provider() -> str:
    """Read provider from session state; default to OpenAI."""
    return st.session_state.get("service_provider") or "OpenAI"


def _save_streamlit_file_to_temp(video_file) -> str:
    """Persist an uploaded file to a temp path and return the path."""
    suffix = os.path.splitext(video_file.name or ".mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    # video_file can be read multiple times with .getvalue(); write bytes once here
    data = video_file.getvalue()
    tmp.write(data)
    tmp.close()
    return tmp.name


# ------------------------- Transcription Backends -------------------------

def _transcribe_openai(video_path: str, client) -> str:
    """Use OpenAI Whisper-1 via Chat Completions client.audio.transcriptions."""
    with open(video_path, "rb") as f:
        # whisper-1 only supports response_format="text" or JSON variants
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )
    # SDK returns a plain string when response_format="text"
    return resp if isinstance(resp, str) else getattr(resp, "text", str(resp))


def _ensure_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import whisper  # local whisper
        _WHISPER_MODEL = whisper.load_model("base")
    return _WHISPER_MODEL


def _transcribe_local_whisper(video_path: str) -> str:
    model = _ensure_whisper_model()
    result = model.transcribe(video_path)
    return result.get("text", "").strip()


# ------------------------- Public helpers (kept for compatibility) -------------------------

def transcribe(video_file, client, provider):
    """
    Backward-compatible function: save uploaded file, then call the right backend.
    """
    path = _save_streamlit_file_to_temp(video_file)
    try:
        if provider == "OpenAI":
            return _transcribe_openai(path, client)
        else:
            return _transcribe_local_whisper(path)
    finally:
        # keep temp file for debugging? remove to avoid clutter
        with contextlib.suppress(Exception):
            os.remove(path)


def transcribe2(video_file_path, client, provider):
    """
    Same as transcribe(), but accepts a filesystem path directly (used for split parts).
    """
    if provider == "OpenAI":
        return _transcribe_openai(video_file_path, client)
    else:
        return _transcribe_local_whisper(video_file_path)


# ------------------------- Optional splitting for large files -------------------------

def split_video(input_file: str, output_prefix: str, num_parts: int):
    """
    Split a video into num_parts pieces using MoviePy, returns list of file paths.
    If MoviePy is unavailable, returns [input_file] unchanged.
    """
    if not _HAS_MOVIEPY or num_parts <= 1:
        return [input_file]

    try:
        clip = VideoFileClip(input_file)
        total = clip.duration
        part_dur = total / num_parts
        out_paths = []
        for i in range(num_parts):
            start = i * part_dur
            end = min((i + 1) * part_dur, total)
            sub = clip.subclip(start, end)
            out_path = f"{output_prefix}_part{i+1}.mp4"
            # Quiet write
            sub.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            out_paths.append(out_path)
        clip.close()
        return out_paths
    except Exception as e:
        st.warning(f"Could not split video ({e}). Proceeding without splitting.")
        return [input_file]


def display_split_files(split_files):
    """Tiny helper to show preview of split files in Streamlit."""
    st.write("### Split Files:")
    for fp in split_files:
        st.markdown(f"**{os.path.basename(fp)}**")
        st.video(fp)


# ------------------------- Main entry used by 2_Analysis.py -------------------------

def transcriber(video_file, client):
    """
    Main transcription helper used by the UI.
    - Saves uploaded file to temp
    - Auto-splits if the file is large (threshold ~20 MB)
    - Uses OpenAI Whisper-1 when provider == 'OpenAI', otherwise local Whisper
    - Concats transcripts from parts
    """
    provider = _get_provider()
    temp_path = _save_streamlit_file_to_temp(video_file)

    try:
        # Check size on disk (more reliable than getvalue for very big files)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        threshold_mb = 20.0
        if size_mb >= threshold_mb:
            st.info("File is large. Splitting for transcription; this may take a few minutes.")
            parts = split_video(temp_path, os.path.splitext(temp_path)[0], math.ceil(size_mb / threshold_mb))
            # Optional preview
            # display_split_files(parts)
            transcript = ""
            for p in parts:
                transcript += transcribe2(p, client, provider)
        else:
            if provider == "OpenAI":
                transcript = _transcribe_openai(temp_path, client)
            else:
                transcript = _transcribe_local_whisper(temp_path)

        return transcript.strip()

    finally:
        # Clean up temp pieces (best-effort)
        with contextlib.suppress(Exception):
            os.remove(temp_path)
