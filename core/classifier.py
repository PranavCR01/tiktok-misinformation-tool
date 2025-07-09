import os
import json
import re
import openai

# Optional: import Gemini client
try:
    from vertexai.language_models import TextGenerationModel
    import vertexai
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False

# Prompt for both providers
_BASE_PROMPT = """
You are a public-health fact-checking assistant.
Given a transcript, choose exactly one label from:
[NO_MISINFO, MISINFO, DEBUNKING, CANNOT_RECOGNIZE]
Then output up to 10 comma-separated keywords that summarize the content.

Respond in JSON:
{"label":"...", "keywords":["kw1","kw2",...]}
"""

def classify_transcript(text: str, provider: str = "OpenAI", model: str = "gpt-3.5-turbo") -> dict:
    prompt = _BASE_PROMPT + "\n\n" + text

    if provider == "Gemini":
        if not _HAS_GEMINI:
            raise RuntimeError("Gemini SDK not installed. Please install vertexai.")

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("No Gemini API key found in environment.")

        os.environ["GOOGLE_API_KEY"] = api_key  # Gemini requires it to be set like this

        # Initialize Gemini model
        model = TextGenerationModel.from_pretrained("text-bison")
        prediction = model.predict(prompt)
        content = getattr(prediction, "text", str(prediction))

    else:  # OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No OpenAI API key found in environment.")

        openai.api_key = api_key
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": _BASE_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0
        )
        content = completion.choices[0].message.content

    # Clean and parse
    cleaned = re.sub(r"```json|```", "", content).strip()
    return json.loads(cleaned)
