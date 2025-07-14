#core/classifier.pyimport os
import json
import re
import openai
import requests
import time
from config.settings import GPT_MODEL, GEMINI_MODEL, OLLAMA_MODEL

try:
    from ollama import Client
    client = Client(host="http://localhost:11434")
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False

_BASE_PROMPT = """
You are a public-health fact-checking assistant.
Given a transcript, choose exactly one label from:
[NO_MISINFO, MISINFO, DEBUNKING, CANNOT_RECOGNIZE]
Then output up to 10 comma-separated keywords that summarize the content.

Also provide a confidence score between 0 and 1 for your prediction.

Respond in JSON:
{"label":"...", "keywords":["kw1","kw2",...], "confidence": 0.87}
"""

def extract_json_block(text: str) -> dict:
    try:
        matches = re.findall(r"{.*?}", text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {e}")
    raise ValueError("No valid JSON object found in the response.")

def classify_transcript(text, provider="OpenAI", model=None, api_key=None) -> dict:
    prompt = _BASE_PROMPT + "\n\n" + text
    start_time = time.time()

    if provider == "Gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("No Gemini API key found.")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        resp = requests.post(url, params={"key": key}, json=body, timeout=10)
        resp.raise_for_status()
        content = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    elif provider == "Ollama":
        if not _HAS_OLLAMA:
            raise RuntimeError("Ollama not installed.")
        response = client.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        content = response["message"]["content"]

    else:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("No OpenAI API key found.")
        openai.api_key = key
        completion = openai.ChatCompletion.create(
            model=model or GPT_MODEL,
            messages=[
                {"role": "system", "content": _BASE_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0,
        )
        content = completion.choices[0].message.content

    result = extract_json_block(content)
    result["time_taken_secs"] = round(time.time() - start_time, 2)
    return result
