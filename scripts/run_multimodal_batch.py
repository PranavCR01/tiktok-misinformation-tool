"""
Multimodal Batch Experiment Script
Runs experiments with both audio transcription AND on-screen text extraction.
This is separate from run_mistral_batch.py to maintain backward compatibility.
"""

import os
import csv
import sys
import glob
import time
import json
import argparse
from pathlib import Path

# Import modules
try:
    from pages.processes.analysis import _extract_json_block
    from pages.processes.multimodal import extract_multimodal_content
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from pages.processes.analysis import _extract_json_block
    from pages.processes.multimodal import extract_multimodal_content

# Reuse prompt definitions from original script
BASE_PROMPT = """You are a public-health fact-checking assistant.

Given the content below (which may include both audio transcript and on-screen text), do three things:
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

FEWSHOT_SUFFIX = """
Here are examples to calibrate your judgement:

Example A (DEBUNKING):
Content: "A doctor explains that vaccines do NOT cause autism and cites CDC."
JSON: {"label":"DEBUNKING","keywords":["vaccines","autism","CDC"],"confidence":0.9}

Example B (MISINFO):
Content: "Drinking bleach cures infections."
JSON: {"label":"MISINFO","keywords":["bleach","cure","infections"],"confidence":0.95}

Now classify the provided content with the same JSON schema only.
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


def analyze_with_mistral(content: str, prompt_kind: str, temperature: float, model_name: str):
    """
    Call local Ollama to classify content (audio + visual combined).
    """
    import requests

    start = time.time()
    url = "http://localhost:11434/api/chat"
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": build_prompt(prompt_kind)},
            {"role": "user", "content": content},
        ],
        "options": {"temperature": float(temperature)},
        "stream": False,
    }

    r = requests.post(url, json=body, timeout=180)  # Longer timeout for multimodal
    r.raise_for_status()

    content_resp = ""
    try:
        data = r.json()
        if isinstance(data, dict) and "message" in data:
            content_resp = data["message"].get("content", "")
        else:
            content_resp = r.text
    except Exception:
        content_resp = r.text

    result = _extract_json_block(content_resp)
    result["time_taken_secs"] = round(time.time() - start, 2)
    return result


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    input_dir = cfg.get("input_dir", "data/videos")
    out_csv = cfg.get("out", "experiments/results.csv")
    prompt_kind = cfg.get("prompt", "baseline")
    model_name = cfg.get("model_name", "mistral")
    temperature = float(cfg.get("temperature", 0.0))
    
    # NEW: Multimodal options
    include_audio = cfg.get("include_audio", True)
    include_visual = cfg.get("include_visual", True)
    ocr_languages = cfg.get("ocr_languages", ["en"])
    ocr_sample_fps = float(cfg.get("ocr_sample_fps", 1.0))

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    video_paths = []
    for ext in ("*.mp4", "*.mp3", "*.wav", "*.m4a"):
        video_paths += glob.glob(os.path.join(input_dir, ext))
    video_paths = sorted(set(video_paths))

    fieldnames = [
        "prompt_id",
        "model_name",
        "video_file",
        "audio_transcript",
        "visual_text",
        "combined_content",
        "modalities_used",
        "label",
        "keywords",
        "confidence_score",
        "time_taken_sec",
    ]
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for p in video_paths:
            bn = os.path.basename(p)
            print(f"[{bn}] extracting multimodal content…", flush=True)
            
            # Extract both audio and visual
            multimodal_result = extract_multimodal_content(
                p,
                include_audio=include_audio,
                include_visual=include_visual,
                ocr_languages=ocr_languages
            )

            print(f"[{bn}] analyzing with {model_name}…", flush=True)
            
            # Use combined content for classification
            content_for_analysis = multimodal_result['combined_content']
            analysis_result = analyze_with_mistral(
                content_for_analysis, 
                prompt_kind, 
                temperature, 
                model_name
            )

            row = {
                "prompt_id": prompt_kind,
                "model_name": model_name,
                "video_file": bn,
                "audio_transcript": multimodal_result['audio_transcript'],
                "visual_text": multimodal_result['visual_text'],
                "combined_content": content_for_analysis,
                "modalities_used": ";".join(multimodal_result['modalities_used']),
                "label": analysis_result.get("label", "CANNOT_RECOGNIZE"),
                "keywords": ";".join(analysis_result.get("keywords", [])),
                "confidence_score": round(float(analysis_result.get("confidence", 0.0)), 2),
                "time_taken_sec": analysis_result.get("time_taken_secs"),
            }
            w.writerow(row)
            print(
                f"[{bn}] done. label={row['label']} conf={row['confidence_score']} "
                f"modalities={row['modalities_used']}",
                flush=True,
            )

    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()