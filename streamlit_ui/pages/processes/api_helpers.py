# pages/processes/api_helpers.py

import streamlit as st


def is_open_ai_api_key_valid(api_key, client) -> bool:
    """
    Quick, low-cost probe that the provided OpenAI key can successfully call Chat Completions.
    Returns True/False without throwing in the UI.
    """
    if not api_key:
        return False
    try:
        _ = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello World!"}],
            temperature=0,
            timeout=10,  # seconds (supported by recent SDKs)
        )
        return True
    except Exception:
        # Don't spam the user with stack traces here; just signal failure.
        return False


def is_azure_api_key_valid(api_key, client, model) -> bool:
    """
    Probe Azure OpenAI with the given deployment name (passed via `model`).
    Returns True/False without throwing in the UI.
    """
    if not (api_key and model):
        return False
    try:
        _ = client.chat.completions.create(
            model=model,  # NOTE: for Azure, this is the *deployment name*
            messages=[{"role": "user", "content": "Hello World!"}],
            temperature=0,
            timeout=10,
        )
        return True
    except Exception:
        return False


def get_model_selection() -> str:
    """
    Small wrapper for the model picker used in the OpenAI path.
    Keep the options aligned with what your account actually has.
    """
    models = (
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-4",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
    )
    return st.selectbox("Select the AI model to use:", models, index=0)
