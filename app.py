import os
import tempfile

import streamlit as st
import pandas as pd
import openai

from config.settings import MAX_UPLOAD_MB, GEMINI_MODEL
from core import transcriber, classifier, postprocess, visualize
from streamlit_option_menu import option_menu
import utils.st_session as ss

# — Gemini SDK import —
try:
    from vertexai.language_models import TextGenerationModel
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False

st.set_page_config(page_title="Automatic Misinformation Analysis", layout="wide")
ss.init_state()

# — Sidebar navigation —
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
    st.markdown("""
    **Automatic Misinformation Analysis Tool**

    1. Select your **LLM provider** (OpenAI or Gemini)  
    2. Enter your **API Key**, then press Enter to validate  
    3. Upload one or more **WAV/MP4/TikTok clips** (≤ 200 MB each)  
    4. Click **Transcribe & Analyze**  
    5. View **results**, **visualizations**, and **download CSV**  
    """)
    st.stop()

# — Analysis page —
st.title("Automatic Misinformation Analysis")

# State
st.session_state.setdefault("api_valid", False)
st.session_state.setdefault("api_error", "")

# Provider
provider = st.selectbox("Select your LLM provider:", ["OpenAI", "Gemini"])

# Key validation
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

    else:
        if not _HAS_GEMINI:
            st.session_state.api_valid = False
            st.session_state.api_error = "Gemini SDK not installed. Run `pip install google-cloud-aiplatform` and `pip install google-cloud-vertex-ai`."
            return

        os.environ["GOOGLE_API_KEY"] = key
        try:
            model = TextGenerationModel.from_pretrained(GEMINI_MODEL)
            model.predict("ping", temperature=0, max_output_tokens=1)
            st.session_state.api_valid = True
            st.session_state.api_error = ""
        except Exception:
            st.session_state.api_valid = False
            st.session_state.api_error = "Gemini key invalid. Please enter a correct key."

# API Key Input
st.text_input(
    f"{provider} API Key",
    type="password",
    key="_key_input",
    on_change=validate_key,
)

if st.session_state.api_error:
    st.error(st.session_state.api_error)

# File Upload
uploaded = st.file_uploader(
    "Upload video/audio files (≤ 200 MB each)",
    type=["mp4", "mpeg4", "mp3", "wav"],
    accept_multiple_files=True,
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

    for i, file in enumerate(st.session_state.files):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=file.name)
        tmp.write(file.read())
        tmp.close()

        transcript = transcriber.transcribe_video(tmp.name)
        res = classifier.classify_transcript(
            transcript,
            provider=provider,
            api_key=st.session_state._key_input
        )
        res = postprocess.ensure_schema(res)

        records.append({
            "video_file": file.name,
            "transcript": transcript,
            "misinformation_status": res["label"],
            "keywords": ";".join(res["keywords"])
        })
        progress.progress((i + 1) / len(st.session_state.files))

    st.session_state.results = pd.DataFrame(records)

# Show results
if st.session_state.get("results") is not None:
    df = st.session_state.results
    st.subheader("📊 Results")
    st.dataframe(df, use_container_width=True)
    st.subheader("📈 Visualizations")
    visualize.label_bar(df)
    visualize.keyword_network(df)
    csv = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv, "results.csv")
