# scripts/run_mistral.py

import os
import csv
import sys
import glob
import time
import json
import argparse
from pathlib import Path

# ---- Import app modules first; if not importable, patch sys.path then retry ----
try:
    from pages.processes.analysis import _extract_json_block  # reuse robust JSON parser

    # from pages.processes.transcription import transcribe2 as transcribe_file_openapi
    from pages.processes.transcription import (
        _transcribe_local_faster_whisper,
    )  # internal but OK for CLI
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from pages.processes.analysis import _extract_json_block

    # from pages.processes.transcription import transcribe2 as transcribe_file_openapi
    from pages.processes.transcription import _transcribe_local_faster_whisper  # type: ignore

# ----- Optional: faster‑whisper for speed (CPU) -----
_FAST_WHISPER = None
try:
    from faster_whisper import WhisperModel as _FastModel  # type: ignore

    _FAST_WHISPER = _FastModel("base", device="cpu", compute_type="int8")
except Exception:
    _FAST_WHISPER = None


def fast_whisper_transcribe(path: str) -> str:
    if _FAST_WHISPER is None:
        return _transcribe_local_faster_whisper(path)
    segments, _ = _FAST_WHISPER.transcribe(path, vad_filter=True, beam_size=5)
    return " ".join(s.text.strip() for s in segments).strip()


# ----------------------------- Prompts -----------------------------

BASE_PROMPT = """You are a public-health fact-checking assistant.

Given the transcript below, do these tasks and respond ONLY as JSON (no prose, no markdown):
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

FEWSHOT_SUFFIX = """
Here are examples to calibrate your judgement:

Example A (DEBUNKING):
Transcript: "A doctor explains that vaccines do NOT cause autism and cites CDC."
JSON: {"label":"DEBUNKING","keywords":["vaccines","autism","CDC"],"confidence":0.9}

Example B (MISINFO):
Transcript: "Drinking bleach cures infections."
JSON: {"label":"MISINFO","keywords":["bleach","cure","infections"],"confidence":0.95}

Now classify the provided transcript with the same JSON schema only.
"""

REASONED_SUFFIX = """
Think step-by-step but only return the final JSON (do not include your reasoning).
"""


def build_prompt(kind: str) -> str:
    if kind == "baseline":
        return BASE_PROMPT
    if kind == "fewshot":
        return BASE_PROMPT + "\n" + FEWSHOT_SUFFIX
    if kind == "reasoned":
        return BASE_PROMPT + "\n" + REASONED_SUFFIX
    return BASE_PROMPT


# ------------------ Ollama (Mistral) caller ------------------


def analyze_with_mistral(
    transcript: str, prompt_kind: str, temperature: float, model_name: str
):
    """
    Call local Ollama chat API with Mistral and return:
    {label, keywords[], confidence, explanation, evidence_sentences[], time_taken_secs}

    Uses non‑streaming responses to avoid JSONDecodeError from concatenated JSON.
    Includes a defensive fallback that can parse line-delimited JSON if needed.
    """
    import requests

    start = time.time()

    url = "http://localhost:11434/api/chat"
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": build_prompt(prompt_kind)},
            {"role": "user", "content": transcript},
        ],
        "options": {"temperature": float(temperature)},
        "stream": False,  # IMPORTANT: single JSON object response
    }

    r = requests.post(url, json=body, timeout=120)
    r.raise_for_status()

    content = ""
    try:
        data = r.json()
        if isinstance(data, dict) and "message" in data:
            content = data["message"].get("content", "")
        else:
            # Some Ollama builds may still return multiple JSON lines — handle gracefully.
            try:
                lines = [
                    json.loads(line)
                    for line in r.text.splitlines()
                    if line.strip().startswith("{")
                ]
                content = "".join(
                    d.get("message", {}).get("content", "") for d in lines
                )
            except Exception:
                content = r.text
    except Exception:
        # If .json() fails, treat as plain text
        content = r.text

    result = _extract_json_block(content)
    result["time_taken_secs"] = round(time.time() - start, 2)
    return result


def transcribe_local(path: str) -> str:
    # Prefer faster‑whisper if available, else fallback to your local whisper helper
    try:
        return fast_whisper_transcribe(path)
    except Exception:
        return _transcribe_local_faster_whisper(path)


# ------------------ CLI entry ------------------


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    input_dir = cfg.get("input_dir", "data/videos")
    out_csv = cfg.get("out", "experiments/results.csv")
    prompt_kind = cfg.get("prompt", "baseline")  # baseline|fewshot|reasoned
    model_name = cfg.get("model_name", "mistral")
    temperature = float(cfg.get("temperature", 0.0))

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    video_paths = []
    for ext in ("*.mp4", "*.mp3", "*.wav", "*.m4a"):
        video_paths += glob.glob(os.path.join(input_dir, ext))
    video_paths = sorted(set(video_paths))

    fieldnames = [
        "prompt_id",
        "model_name",
        "video_file",
        "transcript",
        "label",
        "keywords",
        "confidence_score",
        "explanation",
        "evidence_sentences",
        "time_taken_sec",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for p in video_paths:
            bn = os.path.basename(p)
            print(f"[{bn}] transcribing…", flush=True)
            transcript = transcribe_local(p)

            print(f"[{bn}] analyzing (Mistral)…", flush=True)
            r = analyze_with_mistral(transcript, prompt_kind, temperature, model_name)

            row = {
                "prompt_id": prompt_kind,
                "model_name": model_name,
                "video_file": bn,
                "transcript": transcript,
                "label": r.get("label", "CANNOT_RECOGNIZE"),
                "keywords": ";".join(r.get("keywords", [])),
                "confidence_score": round(float(r.get("confidence", 0.0)), 2),
                "explanation": r.get("explanation", ""),
                "evidence_sentences": "|".join(r.get("evidence_sentences", [])),
                "time_taken_sec": r.get("time_taken_secs"),
            }
            w.writerow(row)
            print(
                f"[{bn}] done. label={row['label']} conf={row['confidence_score']}",
                flush=True,
            )

    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
