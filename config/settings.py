#config/ settings.py

import os

# Whisper transcription model
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# OpenAI GPT model
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo")

# Gemini model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")

# Ollama model for local inference
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# File upload limit (MB)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 200))

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

