#app.py
import os
import time
import tempfile
import streamlit as st
import pandas as pd
import openai

from config.settings import MAX_UPLOAD_MB, GEMINI_MODEL
from core import transcriber, classifier, postprocess, visualize
from streamlit_option_menu import option_menu
import utils.st_session as ss

st.set_page_config(page_title="Automatic Misinformation Analysis", layout="wide")
ss.init_state()

# Caching toggle
USE_CACHE = st.sidebar.checkbox("Enable Caching", value=True)

@st.cache_data(show_spinner=False)
def cached_transcribe_and_classify(video_path, provider, api_key):
    transcript, audio_len = transcriber.transcribe_video(video_path)
    result = classifier.classify_transcript(transcript, provider=provider, api_key=api_key)
    return transcript, result, audio_len

# Sidebar navigation
with st.sidebar:
    page = option_menu(
        menu_title=None,
        options=["User Guide", "Analysis"],
        icons=["book", "bar-chart"],
        default_index=1,
        styles={"nav-link": {"font-size": "16px"}}
    )

if page == "User Guide":
    st.title("User Guide")
    st.markdown("""**Automatic Misinformation Analysis Tool**

1. Select your **LLM provider** (OpenAI, Gemini, or Ollama)  
2. If required, enter your **API Key**, then press Enter to validate  
3. Upload one or more **WAV/MP4/TikTok clips** (≤ 200 MB each)  
4. Click **Transcribe & Analyze**  
5. View **results**, **visualizations**, and **download CSV**
""")
    st.stop()

st.title("Automatic Misinformation Analysis")
st.session_state.setdefault("api_valid", False)
st.session_state.setdefault("api_error", "")

provider = st.selectbox("Select your LLM provider:", ["OpenAI", "Gemini", "Ollama"])

def validate_key():
    key = st.session_state._key_input.strip()

    if provider == "OpenAI":
        openai.api_key = key
        try:
            openai.Model.list()
            st.session_state.api_valid = True
            st.session_state.api_error = ""
        except Exception:
            st.session_state.api_valid = False
            st.session_state.api_error = "OpenAI key invalid. Please enter a correct key."

    elif provider == "Gemini":
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        body = {"contents": [{"role": "user", "parts": [{"text": "ping"}]}]}
        try:
            resp = requests.post(url, params={"key": key}, json=body, timeout=10)
            resp.raise_for_status()
            st.session_state.api_valid = True
            st.session_state.api_error = ""
        except Exception:
            st.session_state.api_valid = False
            st.session_state.api_error = "Gemini key invalid. Please enter a correct key."

    elif provider == "Ollama":
        st.session_state.api_valid = True
        st.session_state.api_error = ""

if provider in ["OpenAI", "Gemini"]:
    st.text_input(
        f"{provider} API Key",
        type="password",
        key="_key_input",
        on_change=validate_key,
    )
    if st.session_state.api_error:
        st.error(st.session_state.api_error)
else:
    st.info("Ollama will run a local model. No API key required.")
    st.session_state.api_valid = True

# File uploader
uploaded = st.file_uploader(
    "Upload video/audio files (≤ 200 MB each)",
    type=["mp4", "mpeg4", "mp3", "wav"],
    accept_multiple_files=True
)
valid_files = []
if uploaded:
    for file in uploaded:
        size_mb = file.size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            st.warning(f"Skipping `{file.name}` ({size_mb:.1f} MB > {MAX_UPLOAD_MB} MB).")
        else:
            valid_files.append(file)
    if not valid_files:
        st.stop()
    st.session_state.files = valid_files

# Analyze
if st.button("Transcribe & Analyze") and st.session_state.api_valid and st.session_state.get("files"):
    records = []
    progress = st.progress(0)
    files = st.session_state.files
    user_api_key = st.session_state.get("_key_input", "")

    for i, file in enumerate(files):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=file.name)
        tmp.write(file.read())
        tmp.close()

        start = time.time()
        if USE_CACHE:
            transcript, result, audio_len = cached_transcribe_and_classify(tmp.name, provider, user_api_key)
        else:
            transcript, audio_len = transcriber.transcribe_video(tmp.name)
            result = classifier.classify_transcript(
                transcript,
                provider=provider,
                api_key=user_api_key
            )
        result = postprocess.ensure_schema(result)
        end = time.time()

        records.append({
            "video_file": file.name,
            "transcript": transcript,
            "label": result["label"],
            "keywords": ";".join(result["keywords"]),
            "confidence_score": result["confidence_score"],
            "audio_length_sec": audio_len,
            "time_taken_sec": round(end - start, 2)
        })
        progress.progress((i + 1) / len(files))

    st.session_state.results = pd.DataFrame(records)

# Results
if st.session_state.get("results") is not None:
    df = st.session_state.results
    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    st.subheader("Visualizations")
    visualize.label_bar(df)
    visualize.keyword_network(df)

    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "results.csv")
