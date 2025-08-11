# pages/processes/analysis.py

import re
import json
import time
import tiktoken
import streamlit as st

# ---------- helpers ----------

_JSON_INSTRUCTIONS = """
You are a public‑health fact‑checking assistant.

Given the transcript below, do three things:
1) Choose exactly one label from this set:
   [NO_MISINFO, MISINFO, DEBUNKING, CANNOT_RECOGNIZE]
2) Extract up to 10 keywords that summarize the content (list, not strings with commas).
3) Provide a confidence score between 0 and 1.

Respond ONLY as a single JSON object, no prose, in the form:
{
  "label": "DEBUNKING|MISINFO|NO_MISINFO|CANNOT_RECOGNIZE",
  "keywords": ["kw1","kw2", "..."],
  "confidence": 0.87
}
"""

def _extract_json_block(text: str) -> dict:
    """
    Find the first JSON object in an LLM response and parse it.
    Falls back to a safe default if nothing valid is found.
    """
    if not isinstance(text, str):
        return {"label": "CANNOT_RECOGNIZE", "keywords": [], "confidence": 0.5}

    # Try to locate a JSON object
    for match in re.finditer(r"\{.*\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(match.group(0))
            # normalize expected keys
            label = str(obj.get("label", "CANNOT_RECOGNIZE")).strip().upper().rstrip(".")
            kws = obj.get("keywords", [])
            if isinstance(kws, str):
                kws = [k.strip() for k in kws.split(",") if k.strip()]
            conf = float(obj.get("confidence", 0.5))
            return {"label": label, "keywords": kws, "confidence": conf}
        except Exception:
            continue

    return {"label": "CANNOT_RECOGNIZE", "keywords": [], "confidence": 0.5}


def _token_limit_warning(transcript: str, model: str, container):
    """Your original token checks, kept intact."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = len(encoding.encode(transcript))
    model_dict = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-0125": 16385,
        "gpt-3.5-turbo-1106": 16385,
        "gpt-4": 8192,
        "gpt-4-1106-preview": 128000,
        "gpt-4-0125-preview": 128000,
    }
    model_limit = model_dict.get(model)
    exceeding_models = [m for m, limit in model_dict.items() if tokens > limit]
    if exceeding_models:
        container.update(
            label=(
                f"Warning: This video's token usage of {tokens} tokens is too big for any GPT models "
                f"to handle. Please choose a different video!"
            ),
            state="error",
            expanded=False,
        )
    if model_limit and tokens > model_limit:
        container.update(
            label=f"Warning: Token usage of {tokens} exceeds the limit of {model_limit} for model {model}.",
            state="error",
            expanded=False,
        )
    else:
        container.update(label="Token Usage within Model Limit", state="running", expanded=True)


# ---------- OpenAI (hosted) ----------

def analyze(transcript, model, client, container):
    """
    Uses OpenAI Chat Completions.
    Returns dict: {label, keywords[], confidence, time_taken_secs}
    """
    _token_limit_warning(transcript, model, container)

    chat_sequence = [
        {"role": "system", "content": _JSON_INSTRUCTIONS},
        {"role": "user", "content": transcript},
    ]

    start = time.time()
    try:
        resp = client.chat.completions.create(model=model, messages=chat_sequence, temperature=0)
        content = resp.choices[0].message.content
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return {
            "label": "CANNOT_RECOGNIZE",
            "keywords": [],
            "confidence": 0.0,
            "time_taken_secs": round(time.time() - start, 2),
        }

    result = _extract_json_block(content)
    result["time_taken_secs"] = round(time.time() - start, 2)
    return result


# ---------- Azure OpenAI (deployment name passed as `model`) ----------

def analyze2(transcript, client, container, model):
    """
    Uses Azure OpenAI Chat Completions.
    `model` here is your deployment name.
    Returns dict: {label, keywords[], confidence, time_taken_secs}
    """
    chat_sequence = [
        {"role": "system", "content": _JSON_INSTRUCTIONS},
        {"role": "user", "content": transcript},
    ]

    start = time.time()
    try:
        resp = client.chat.completions.create(model=model, messages=chat_sequence, temperature=0)
        content = resp.choices[0].message.content
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return {
            "label": "CANNOT_RECOGNIZE",
            "keywords": [],
            "confidence": 0.0,
            "time_taken_secs": round(time.time() - start, 2),
        }

    result = _extract_json_block(content)
    result["time_taken_secs"] = round(time.time() - start, 2)
    return result


