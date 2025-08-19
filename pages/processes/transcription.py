# pages/processes/transcription.py

import os
import math
import tempfile
import contextlib
import streamlit as st

# MoviePy is used only for optional splitting of large uploads.
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip

    _HAS_MOVIEPY = True
except Exception:
    _HAS_MOVIEPY = False


def _get_provider() -> str:
    """Read provider from session state; default to OpenAI."""
    return st.session_state.get("service_provider") or "OpenAI"


def _save_streamlit_file_to_temp(video_file) -> str:
    """Persist an uploaded file to a temp path and return the path."""
    suffix = os.path.splitext(video_file.name or ".mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(video_file.getvalue())
    tmp.close()
    return tmp.name


# ------------------------- Transcription Backends -------------------------


def _transcribe_openai(video_path: str, client) -> str:
    """Use OpenAI Whisper-1 via client.audio.transcriptions."""
    with open(video_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )
    return resp if isinstance(resp, str) else getattr(resp, "text", str(resp))


def _transcribe_local_faster_whisper(video_path: str) -> str:
    """
    Use local faster-whisper (CTranslate2) in **CPU** mode to avoid CUDA/cuDNN issues on Windows.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception:
        st.error(
            "faster-whisper not installed. Run:\n"
            "pip install faster-whisper==1.0.1 ctranslate2==4.3.1"
        )
        return ""

    # Force CPU; int8 is fast and light. Increase beam_size for a bit more accuracy if desired.
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(video_path, beam_size=5)
    return " ".join(seg.text.strip() for seg in segments).strip()


# ------------------------- Public helpers -------------------------


def transcribe(video_file, client, provider):
    """Backward-compatible wrapper that chooses the right backend."""
    path = _save_streamlit_file_to_temp(video_file)
    try:
        if provider == "OpenAI":
            return _transcribe_openai(path, client)
        else:
            return _transcribe_local_faster_whisper(path)
    finally:
        with contextlib.suppress(Exception):
            os.remove(path)


def transcribe2(video_file_path, client, provider):
    """Same as transcribe(), but accepts a filesystem path directly."""
    if provider == "OpenAI":
        return _transcribe_openai(video_file_path, client)
    else:
        return _transcribe_local_faster_whisper(video_file_path)


# ------------------------- Optional splitting for large files -------------------------


def split_video(input_file: str, output_prefix: str, num_parts: int):
    """Split a video into num_parts pieces using MoviePy."""
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
            sub.write_videofile(
                out_path, codec="libx264", audio_codec="aac", verbose=False, logger=None
            )
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
    Main transcription helper:
    - Saves uploaded file to temp
    - Auto-splits if the file is large (threshold ~20 MB)
    - Uses OpenAI Whisper-1 when provider == 'OpenAI', otherwise local faster-whisper
    - Concats transcripts from parts
    """
    provider = _get_provider()
    temp_path = _save_streamlit_file_to_temp(video_file)

    try:
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        threshold_mb = 20.0
        if size_mb >= threshold_mb:
            st.info(
                "File is large. Splitting for transcription; this may take a few minutes."
            )
            parts = split_video(
                temp_path,
                os.path.splitext(temp_path)[0],
                math.ceil(size_mb / threshold_mb),
            )
            transcript = ""
            for p in parts:
                transcript += " " + transcribe2(p, client, provider)
        else:
            if provider == "OpenAI":
                transcript = _transcribe_openai(temp_path, client)
            else:
                transcript = _transcribe_local_faster_whisper(temp_path)

        return transcript.strip()

    finally:
        with contextlib.suppress(Exception):
            os.remove(temp_path)
