import os

# Whisper transcription model
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# OpenAI GPT model for classification
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo")

# Gemini model for classification
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")

# File upload limit (MB)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 200))

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Your provided Gemini key as a default; can be overridden via env/Streamlit secrets
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "AIzaSyB78Q39zNlHntE_kNOnSrAWMz_h32gFsEs"
)
