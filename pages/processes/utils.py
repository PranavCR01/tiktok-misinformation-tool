import streamlit as st
import pandas as pd

# tiktoken is optional at runtime (we only need it for token counts)
try:
    import tiktoken

    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

# Reuse the shared model->token limits from api_helpers if available
try:
    from .api_helpers import MODEL_TOKEN_LIMITS
except Exception:
    MODEL_TOKEN_LIMITS = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-0125": 16385,
        "gpt-3.5-turbo-1106": 16385,
        "gpt-4": 8192,
        "gpt-4-1106-preview": 128000,
        "gpt-4-0125-preview": 128000,
    }

# ----------------------------- Session helpers -----------------------------


def remove_uploaded_files():
    """Clear any uploaded file references and previous results from the session."""
    if "uploaded_files" in st.session_state:
        for file_info in st.session_state.uploaded_files:
            st.session_state.pop(file_info.get("file_name"), None)
        st.session_state.pop("uploaded_files", None)

    for key in ("data", "page_state"):
        if key in st.session_state:
            st.session_state.pop(key, None)


# ----------------------------- Token utilities -----------------------------


def tokenizer(text: str) -> int:
    """Count tokens with tiktoken if available; otherwise use a rough fallback."""
    if _HAS_TIKTOKEN:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    return max(1, len((text or "").split()))


def tokens_check(tokens: int, model: str):
    """UI helper: show whether `tokens` fits within the selected model's limit."""
    limit = MODEL_TOKEN_LIMITS.get(model)
    if limit is None:
        st.error(f"Error: Model {model!r} not found in the token-limit table.")
        return

    if all(tokens > lim for lim in MODEL_TOKEN_LIMITS.values()):
        st.error(
            f"Warning: Token usage ({tokens}) is too high for any listed GPT models. "
            "Please choose a different/shorter video."
        )

    if tokens > limit:
        st.error(
            f"Warning: Token usage ({tokens}) exceeds the limit ({limit}) for model {model}."
        )
    else:
        st.success(
            f"Token usage ({tokens}) is within the limit ({limit}) for model {model}."
        )


# ----------------------------- Data utilities -----------------------------


def convert_df(df: pd.DataFrame) -> bytes:
    """Encode a DataFrame to CSV bytes for Streamlit's download button."""
    return df.to_csv(index=False).encode("utf-8")


def keyword(keywords: str):
    """Display keyword counts from a comma-separated string."""
    items = [k.strip() for k in (keywords or "").split(",") if k.strip()]
    counts = {k: items.count(k) for k in set(items)}
    df = pd.DataFrame(sorted(counts.items()), columns=["Keyword", "Occurrences"])
    st.dataframe(df, use_container_width=True)
