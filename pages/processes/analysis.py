# pages/processes/analysis.py

import re
import json
import time
import streamlit as st

# tiktoken only for token warnings; optional
try:
    import tiktoken

    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

# ---------- helpers ----------

_JSON_INSTRUCTIONS = """
You are a public-health fact-checking assistant.

Given the transcript below, do these tasks and then respond ONLY as JSON (no markdown, no prose):
1) Choose exactly one label from this set:
   [NO_MISINFO, MISINFO, DEBUNKING, CANNOT_RECOGNIZE]
2) Extract up to 10 keywords that summarize the content (list, not strings with commas).
3) Provide a confidence score between 0 and 1.
4) Provide a short explanation (2-4 sentences) describing why you chose that label.
5) Provide 1-3 exact quotes from the transcript (verbatim sentences or clauses) that most influenced your decision.

Return a single JSON object with this shape:
{
  "label": "DEBUNKING|MISINFO|NO_MISINFO|CANNOT_RECOGNIZE",
  "keywords": ["kw1","kw2", "..."],
  "confidence": 0.87,
  "explanation": "...",
  "evidence_sentences": ["...", "..."]
}
"""


def _extract_json_block(text: str) -> dict:
    """Find the first JSON object in an LLM response and parse it."""
    if not isinstance(text, str):
        return {
            "label": "CANNOT_RECOGNIZE",
            "keywords": [],
            "confidence": 0.5,
            "explanation": "",
            "evidence_sentences": [],
        }

    for match in re.finditer(r"\{.*\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(match.group(0))
            label = (
                str(obj.get("label", "CANNOT_RECOGNIZE")).strip().upper().rstrip(".")
            )
            kws = obj.get("keywords", [])
            if isinstance(kws, str):
                kws = [k.strip() for k in kws.split(",") if k.strip()]
            conf = float(obj.get("confidence", 0.5))
            explanation = str(obj.get("explanation", "")).strip()
            evidence_raw = obj.get("evidence_sentences", [])
            if isinstance(evidence_raw, str):
                evidence = [
                    item.strip()
                    for item in evidence_raw.split("\n")
                    if item.strip()
                ]
            elif isinstance(evidence_raw, list):
                evidence = [str(item).strip() for item in evidence_raw if str(item).strip()]
            else:
                evidence = []
            return {
                "label": label,
                "keywords": kws,
                "confidence": conf,
                "explanation": explanation,
                "evidence_sentences": evidence,
            }
        except Exception:
            continue

    return {
        "label": "CANNOT_RECOGNIZE",
        "keywords": [],
        "confidence": 0.5,
        "explanation": "",
        "evidence_sentences": [],
    }


def _token_limit_warning(transcript: str, model: str, container):
    """Optional token checks (shown in the UI)."""
    if not _HAS_TIKTOKEN:
        return
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
    if model_limit and tokens > model_limit:
        container.update(
            label=f"Warning: Token usage of {tokens} exceeds the limit of {model_limit} for model {model}.",
            state="error",
            expanded=False,
        )
    else:
        container.update(
            label="Token Usage within Model Limit", state="running", expanded=True
        )


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
        resp = client.chat.completions.create(
            model=model, messages=chat_sequence, temperature=0
        )
        content = resp.choices[0].message.content
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return {
            "label": "CANNOT_RECOGNIZE",
            "keywords": [],
            "confidence": 0.0,
            "explanation": "",
            "evidence_sentences": [],
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
        resp = client.chat.completions.create(
            model=model, messages=chat_sequence, temperature=0
        )
        content = resp.choices[0].message.content
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return {
            "label": "CANNOT_RECOGNIZE",
            "keywords": [],
            "confidence": 0.0,
            "explanation": "",
            "evidence_sentences": [],
            "time_taken_secs": round(time.time() - start, 2),
        }

    result = _extract_json_block(content)
    result["time_taken_secs"] = round(time.time() - start, 2)
    return result


# ---------- Ollama (local Mistral) ----------


def analyze_local_mistral(transcript: str, container, model_name: str = "mistral"):
    """
    Calls local Ollama (http://localhost:11434) chat API with Mistral and returns:
    {label, keywords[], confidence, explanation, evidence_sentences[], time_taken_secs}
    """
    start = time.time()
    try:
        import requests
    except Exception:
        st.error(
            "The 'requests' package is required for Ollama calls. pip install requests"
        )
        return {
            "label": "CANNOT_RECOGNIZE",
            "keywords": [],
            "confidence": 0.5,
            "explanation": "",
            "evidence_sentences": [],
            "time_taken_secs": round(time.time() - start, 2),
        }

    url = "http://localhost:11434/api/chat"
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _JSON_INSTRUCTIONS},
            {"role": "user", "content": transcript},
        ],
        "options": {"temperature": 0.0},
        "stream": False,  # ensure a single JSON response (non-streaming)
    }
    try:
        r = requests.post(url, json=body, timeout=120)
        r.raise_for_status()

        data = r.json()
        content = (
            data.get("message", {}).get("content", "")
            if isinstance(data, dict)
            else str(data)
        )

        result = _extract_json_block(content)
        result["time_taken_secs"] = round(time.time() - start, 2)
        return result
    except Exception as e:
        st.error(f"Ollama (Mistral) call failed: {e}")
        return {
            "label": "CANNOT_RECOGNIZE",
            "keywords": [],
            "confidence": 0.5,
            "explanation": "",
            "evidence_sentences": [],
            "time_taken_secs": round(time.time() - start, 2),
        }
