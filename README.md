# TikTok Misinformation Detection – Experimentation Framework



## Project Overview

This project builds a pipeline to **detect, classify, and analyze misinformation in TikTok videos**.

It combines:

- **Transcription (Whisper / Faster-Whisper)** for extracting speech → text.

- **LLM-based classification (Mistral, others supported)** to detect:

&nbsp; - `MISINFO`

&nbsp; - `DEBUNKING`

&nbsp; - `NO_MISINFO`

&nbsp; - `CANNOT_RECOGNIZE`

- **Experimentation framework** for systematic evaluation of prompts, models, and settings.



>  Important: We have **not changed the original structure** of the app.

Instead, we **added an experimentation layer** (`scripts/`, `experiments/`) that allows reproducible testing across models and prompts.



---



## Repository Structure



```

TIKTOK-MISINFO-APP/

│
├── data/ # Input data (TikTok videos, audio files)
│
├── experiments/ # All experiment runs are logged here
│ ├── exp-001-mistral-baseline/ # Example experiment folder
│ │ ├── config.yaml # Config file (defines prompt, model, paths)
│ │ ├── results.csv # Raw outputs (transcript, label, keywords, confidence, latency)
│ │ ├── command.txt # Command used for reproducibility
│ │ ├── README.md # Auto-generated experiment summary
│ │ ├── NOTES.md # (Optional) manual notes per experiment
│ │ ├── confidence_hist.png # Confidence score histogram
│ │ ├── label_distribution.png# Label counts across dataset
│ │ ├── latency_hist.png # Latency distribution
│ │ ├── keywords_top.csv # Aggregated top keywords
│ │ └── keywords_top.png # Keyword frequency chart
│
├── lib/ # (Reserved for shared utils if needed)
│
├── pages/ # Streamlit app pages
│ ├── processes/ # Core logic modules
│ │ ├── analysis.py # LLM classification logic
│ │ └── transcription.py # ASR logic (OpenAI Whisper / Faster-Whisper)
│ ├── 2_Analysis.py # Streamlit UI for interactive analysis
│ └── init.py
│
├── scripts/ # Experimentation framework
│ ├── run_mistral_batch.py # Runs batch experiment → results.csv
│ └── analyze_experiment.py # Post-analysis → charts + README.md
│
├── .gitignore
├── packages.txt / requirements.txt # Dependencies
├── CHANGELOG.md
└── User_Guide.py # End-user documentation

```
---

## Workflow

### 1. Prepare Config

Each experiment is defined by a `config.yaml`, e.g.:

```yaml

input_dir: data/videos

out: experiments/exp-001-mistral-baseline/results.csv

prompt: baseline        # baseline | fewshot | reasoned

model_name: mistral     # Local model (via Ollama or API)

temperature: 0.0

provider: Local


### 2. Run Experiment (Batch Inference)

python scripts/run_mistral_batch.py --config experiments/exp-001-mistral-baseline/config.yaml

* Produces results.csv with columns:
- prompt_id
- model_name
- video_file
- transcript
- label
- keywords
- confidence_score
- time_taken_sec


3. Analyze Results

python scripts/analyze_experiment.py --exp-dir experiments/exp-001-mistral-baseline

* Produces in the same folder:
* Plots:
1. confidence_hist.png
2. label_distribution.png
3. latency_hist.png

* Aggregated keyword stats:

1. keywords_top.csv
2. keywords_top.png

* Auto-generated README.md with summary

### Why This Framework?

Instead of running ad-hoc tests, this framework allows:

* Reproducibility: Every run is logged in its own folder with configs + outputs.
* Comparability: Run baseline, few-shot, and reasoned prompts across models.
* Scalability: Teammates can plug in different LLMs (e.g., LLaMA, Gemma) without breaking structure.
* Documentation: Auto-generated experiment READMEs ensure no results are lost.

### Team Notes – How to Experiment

* My setup (Pranav): Running with Mistral locally via Ollama.
* Teammates (Tanmay, Atharva):
1. Copy an existing experiment folder → rename (e.g., exp-002-llama-fewshot)
2. Edit config.yaml → change model_name to your local model (llama, gemma, etc.)
3. Re-run steps 2 & 3 above.

* Important: Do not change core files (pages/, scripts/) → keep framework consistent.


### Summary
1. Run your batch experiment with run_mistral_batch.py.
2. Analyze + auto-document with analyze_experiment.py.
3. Each experiment folder is a self-contained record: inputs, outputs, configs, plots, and docs.

4. This ensures our research is organized, collaborative, and reproducible.
