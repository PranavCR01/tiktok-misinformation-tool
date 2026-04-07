# TikTok Misinformation Detection – Experimentation Framework

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-Whisper-412991?style=flat&logo=openai&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Mistral-black?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![UIUC](https://img.shields.io/badge/UIUC-IS492-13294B?style=flat)

---

## Quick Start

```bash
pip install -r requirements-compatible.txt  # Windows (Python 3.11+)
streamlit run User_Guide.py
# Opens at http://localhost:8501
```

---

## Project Overview

This project builds a pipeline to **detect, classify, and analyze misinformation in TikTok videos**.

It combines:

- **Transcription (Whisper / Faster-Whisper)** for extracting speech → text.
- **LLM-based classification (Mistral, others supported)** to detect:
  - `MISINFO`
  - `DEBUNKING`
  - `NO_MISINFO`
  - `CANNOT_RECOGNIZE`
- **Experimentation framework** for systematic evaluation of prompts, models, and settings.
- **Multi-language support** for both audio transcription and on-screen text extraction.

> **Note:** We have **not changed the original structure** of the app. Instead, we **added an experimentation layer** (`scripts/`, `experiments/`) that allows reproducible testing across models and prompts.

---

## Features

- **Interactive Streamlit UI** — upload videos, run transcription and analysis, and visualize results in the browser
- **Speech-to-text (ASR)** via OpenAI Whisper API or local Faster-Whisper (no API key required for local mode)
- **LLM classification** via Ollama (local), OpenAI API, or Azure OpenAI
- **Multi-language audio** — Whisper automatically detects the spoken language (99+ languages supported)
- **On-screen text extraction (OCR)** via EasyOCR, defaulting to English and Spanish
- **Structured batch experiments** — each run is self-contained with config, outputs, plots, and auto-generated docs
- **Data visualization** — label distribution, confidence histograms, latency charts, and top-keyword graphs
- **CSV export** — download analysis results from the UI or batch scripts

---

## Language Support

The tool supports **multiple languages** for comprehensive video analysis.

### Audio Transcription (Whisper)
- **Automatic language detection** – Whisper automatically detects the spoken language
- Supports **99+ languages** including English, Spanish, French, German, Chinese, and more
- No configuration needed – works out of the box for any language

### On-Screen Text (OCR)
- **Default languages**: English and Spanish (`['en', 'es']`)
- Powered by EasyOCR for accurate text detection
- Can detect multiple languages in the same video
- Easily extensible – add more language codes as needed (e.g., `['en', 'es', 'zh']` for Chinese)

### Usage
For most users, **no changes are required** – the system will automatically:
1. Detect and transcribe audio in any language (via Whisper)
2. Extract on-screen text in English and Spanish (via EasyOCR)

To add more OCR languages, update `ocr_languages` in your experiment config:
```yaml
ocr_languages: ['en', 'es', 'fr']  # Add French
```

---

## Repository Structure

```
TIKTOK-MISINFO-APP/
│
├── data/                            # Input data (TikTok videos, audio files)
│
├── experiments/                     # All experiment runs are logged here
│   └── exp-001-mistral-baseline/    # Example experiment folder
│       ├── config.yaml              # Config file (defines prompt, model, paths)
│       ├── results.csv              # Raw outputs (transcript, label, keywords, confidence, latency)
│       ├── command.txt              # Command used for reproducibility
│       ├── README.md                # Auto-generated experiment summary
│       ├── NOTES.md                 # (Optional) manual notes per experiment
│       ├── confidence_hist.png      # Confidence score histogram
│       ├── label_distribution.png   # Label counts across dataset
│       ├── latency_hist.png         # Latency distribution
│       ├── keywords_top.csv         # Aggregated top keywords
│       └── keywords_top.png         # Keyword frequency chart
│
├── lib/                             # (Reserved for shared utils if needed)
│
├── pages/                           # Streamlit app pages
│   ├── processes/                   # Core logic modules
│   │   ├── analysis.py              # LLM classification logic
│   │   └── transcription.py         # ASR logic (OpenAI Whisper / Faster-Whisper)
│   ├── 2_Analysis.py                # Streamlit UI for interactive analysis
│   └── __init__.py
│
├── scripts/                         # Experimentation framework
│   ├── run_mistral_batch.py         # Runs batch experiment → results.csv
│   └── analyze_experiment.py        # Post-analysis → charts + README.md
│
├── .gitignore
├── requirements.txt                 # Full pinned dependencies (original)
├── requirements-compatible.txt      # Loose deps for Python 3.11+ / Windows
├── packages.txt                     # System packages (ffmpeg)
├── CHANGELOG.md
└── User_Guide.py                    # App entry point + end-user documentation
```

### Code File Responsibilities

#### Core Streamlit App (`pages/`)

- **`pages/2_Analysis.py`** — Main Streamlit UI for uploading videos, running transcription + analysis, and visualizing results.
- **`pages/processes/analysis.py`** — LLM classification logic (Ollama, OpenAI, Azure). Parses JSON outputs and returns labels, keywords, and confidence.
- **`pages/processes/transcription.py`** — Handles speech-to-text (ASR) with OpenAI Whisper API and local Faster-Whisper via CTranslate2.
- **`pages/processes/api_helpers.py`** — API utilities: key validation, model dropdowns, Ollama detection, token limits.
- **`pages/processes/utils.py`** — Helper functions: session reset, token counting, file management, and CSV/keyword exports.

#### Experimentation Framework (`scripts/`)

- **`scripts/run_mistral_batch.py`** — Runs transcription + classification in batch mode using settings from `config.yaml`. Writes `results.csv`.
- **`scripts/analyze_experiment.py`** — Post-analysis script that generates plots (label distribution, confidence, latency, top keywords) and produces an auto-summary `README.md` inside each experiment folder.

#### Supporting Files

- `experiments/` — Self-contained experiment folders with configs, outputs, plots, and auto-generated docs.
- `data/` — Input TikTok videos/audio.
- `lib/` — Reserved for external JS/CSS libs.
- `User_Guide.py` — Documentation helper for end-users.
- `CHANGELOG.md` — Tracks repo changes.
- `requirements.txt` / `packages.txt` — Dependencies for reproducibility.

---

## Setup Instructions

### Windows (Python 3.11+)

> Use `requirements-compatible.txt` instead of `requirements.txt` on Windows with Python 3.11 or later.
> The original file pins Cython-compiled spaCy packages (`blis`, `thinc`, etc.) to versions that lack
> pre-built wheels for newer Python releases.

**Step 1 — Install Python 3.11**

Download and install Python 3.11 from [python.org/downloads](https://www.python.org/downloads/).
During installation, check **"Add python.exe to PATH"**.

Verify in a new terminal:
```
python --version   # should print Python 3.11.x
```

**Step 2 — Install ffmpeg**

ffmpeg is required for audio/video processing.

```
winget install ffmpeg
```

Open a new terminal after this so ffmpeg is on your PATH.

**Step 3 — Create a virtual environment**

```
python -m venv .venv
.venv\Scripts\activate
```

**Step 4 — Install dependencies**

```
pip install --upgrade pip
pip install -r requirements-compatible.txt
```

**Step 5 — Run the app**

```
streamlit run User_Guide.py
```

The app will open at `http://localhost:8501`. Use the sidebar to navigate to the **Analysis** page, where you will be prompted to enter your OpenAI API key.

---

### Mac / Linux

**Step 1 — Install Python 3.11+ and ffmpeg**

```bash
# macOS (Homebrew)
brew install python@3.11 ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install python3.11 python3.11-venv ffmpeg
```

**Step 2 — Create a virtual environment**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Step 3 — Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements-compatible.txt
```

**Step 4 — Run the app**

```bash
streamlit run User_Guide.py
# Opens at http://localhost:8501
```

---

### Running Batch Experiments

**Step 1 — Prepare a config file**

Each experiment is defined by a `config.yaml`:

```yaml
input_dir: data/videos
out: experiments/exp-001-mistral-baseline/results.csv
prompt: baseline        # baseline | fewshot | reasoned
model_name: mistral     # local model via Ollama, or an OpenAI model name
temperature: 0.0
provider: Local         # Local | OpenAI | Azure
```

**Step 2 — Run batch inference**

```bash
python scripts/run_mistral_batch.py --config experiments/exp-001-mistral-baseline/config.yaml
```

Produces `results.csv` with columns: `prompt_id`, `model_name`, `video_file`, `transcript`, `label`, `keywords`, `confidence_score`, `time_taken_sec`.

**Step 3 — Analyze results**

```bash
python scripts/analyze_experiment.py --exp-dir experiments/exp-001-mistral-baseline
```

Produces in the same folder:
- Plots: `confidence_hist.png`, `label_distribution.png`, `latency_hist.png`
- Keyword stats: `keywords_top.csv`, `keywords_top.png`
- Auto-generated `README.md` with experiment summary

---

## Workflow

### 1. Prepare Config

Each experiment is defined by a `config.yaml`. See the [Running Batch Experiments](#running-batch-experiments) section above for the full schema.

### 2. Run Experiment (Batch Inference)

```bash
python scripts/run_mistral_batch.py --config experiments/exp-001-mistral-baseline/config.yaml
```

### 3. Analyze Results

```bash
python scripts/analyze_experiment.py --exp-dir experiments/exp-001-mistral-baseline
```

---

## Experiment Framework

### Why This Framework?

Instead of running ad-hoc tests, this framework provides:

- **Reproducibility** — Every run is logged in its own folder with configs + outputs.
- **Comparability** — Run baseline, few-shot, and reasoned prompts across models side by side.
- **Scalability** — Teammates can plug in different LLMs (e.g., LLaMA, Gemma) without breaking structure.
- **Documentation** — Auto-generated experiment READMEs ensure no results are lost.

### Summary

1. Run your batch experiment with `run_mistral_batch.py`.
2. Analyze + auto-document with `analyze_experiment.py`.
3. Each experiment folder is a self-contained record: inputs, outputs, configs, plots, and docs.
4. This ensures our research is organized, collaborative, and reproducible.

---

## Team Notes – How to Experiment

- **Pranav's setup**: Running Mistral locally via Ollama.
- **Teammates (Tanmay, Atharva, others)**:
  1. Copy an existing experiment folder → rename (e.g., `exp-002-llama-fewshot`)
  2. Edit `config.yaml` → change `model_name` to your local model (`llama`, `gemma`, etc.)
  3. Re-run Steps 2 & 3 from the Workflow section above.

> **Important:** Do not modify core files (`pages/`, `scripts/`) — keep the framework consistent across all experiments.
