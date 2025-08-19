# pages/processes/api_helpers.py
import streamlit as st

# Expose token limits so utils.py can import them
MODEL_TOKEN_LIMITS = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo-1106": 16385,
    "gpt-4": 8192,
    "gpt-4-1106-preview": 128000,
    "gpt-4-0125-preview": 128000,
}


def is_open_ai_api_key_valid(api_key, client) -> bool:
    """Probe OpenAI Chat Completions. Returns True/False silently."""
    if not api_key:
        return False
    try:
        _ = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello World!"}],
            temperature=0,
            timeout=10,
        )
        return True
    except Exception:
        return False


def is_azure_api_key_valid(api_key, client, model) -> bool:
    """Probe Azure OpenAI with the deployment name in `model`."""
    if not (api_key and model):
        return False
    try:
        _ = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello World!"}],
            temperature=0,
            timeout=10,
        )
        return True
    except Exception:
        return False


def is_ollama_ready() -> bool:
    """Check if Ollama is reachable locally."""
    try:
        import requests

        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        return True
    except Exception:
        return False


def get_model_selection() -> str:
    """Dropdown for OpenAI model selection."""
    models = (
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-4",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
    )
    return st.selectbox("Select the AI model to use:", models, index=0)
